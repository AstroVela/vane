// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/native_resource_graph.hpp"
#include "vane_python/physical_plan_traversal.hpp"
#include "vane_python/python_objects.hpp"
#include "vane_python/python_udf_actor_resources.hpp"
#include "vane_python/pyrelation.hpp"

#include "duckdb/common/types/uuid.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/join/physical_delim_join.hpp"
#include "duckdb/execution/operator/projection/physical_tableinout_function.hpp"
#include "duckdb/execution/operator/projection/physical_udf_inout.hpp"
#include "duckdb/execution/operator/set/physical_cte.hpp"
#include "duckdb/execution/operator/set/physical_recursive_cte.hpp"
#include "duckdb/function/scalar/udf_functions.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/statement/relation_statement.hpp"
#include "duckdb/planner/planner.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace duckdb {
namespace {

struct NativeResourceNode {
	string name;
	vector<idx_t> inputs;
	vector<idx_t> materialized_inputs;
	bool is_sink;
	const UDFFunctionData *udf;
};

const UDFFunctionData *NativeUDFBindData(const PhysicalOperator &op) {
	const FunctionData *data = nullptr;
	if (op.type == PhysicalOperatorType::INOUT_FUNCTION) {
		data = op.Cast<PhysicalTableInOutFunction>().GetBindData();
	} else if (op.type == PhysicalOperatorType::STREAMING_UDF) {
		data = op.Cast<PhysicalStreamingUDF>().GetBindData();
	}
	return dynamic_cast<const UDFFunctionData *>(data);
}

class NativeResourceGraph {
public:
	NativeResourceGraph(PhysicalOperator &root, ClientContext &context) {
		// Assign all physical IDs first. Owned plans and shared scans are visited
		// exactly once; neither graph construction nor UDF binding copies a plan.
		VisitPhysicalExecutionGraph(root, [&](PhysicalOperator &op) {
			ids.emplace(&op, nodes.size());
			operators.push_back(&op);
			nodes.push_back({op.GetName(), {}, {}, op.IsSink(), NativeUDFBindData(op)});
		});
		for (auto *op : operators) {
			auto &node = nodes[Id(*op)];
			for (auto &child : op->GetChildren()) {
				const auto id = Id(child.get());
				if (std::find(node.inputs.begin(), node.inputs.end(), id) == node.inputs.end()) {
					node.inputs.push_back(id);
				}
			}
			for (auto &child : op->GetMaterializedInputChildren(context)) {
				const auto id = Id(child.get());
				if (std::find(node.inputs.begin(), node.inputs.end(), id) == node.inputs.end()) {
					throw InternalException("Native materialized input is not an execution child");
				}
				if (std::find(node.materialized_inputs.begin(), node.materialized_inputs.end(), id) ==
				    node.materialized_inputs.end()) {
					node.materialized_inputs.push_back(id);
				}
			}
		}
		// These operators own dependencies not expressed by ordinary input edges.
		// All other operators expose their own materialization contract above.
		for (auto *op : operators) {
			if (op->type == PhysicalOperatorType::CTE) {
				MaterializedCTE(op->Cast<PhysicalCTE>());
			} else if (op->type == PhysicalOperatorType::LEFT_DELIM_JOIN ||
			           op->type == PhysicalOperatorType::RIGHT_DELIM_JOIN) {
				DelimJoin(op->Cast<PhysicalDelimJoin>());
			}
		}
		// A materialized input does not prevent the other input's upstream
		// pipelines from running (e.g. a streaming UDF on a join's probe side).
		// Put partial barriers on their input edges, not on the whole operator.
		for (idx_t id = 0; id < operators.size(); id++) {
			const auto materialized = nodes[id].materialized_inputs;
			if (materialized.empty() || materialized.size() == nodes[id].inputs.size()) {
				continue;
			}
			const auto barrier = nodes.size();
			nodes.push_back({"MaterializedInput", materialized, materialized, true, nullptr});
			auto &node = nodes[id];
			node.inputs.erase(std::remove_if(node.inputs.begin(), node.inputs.end(),
			                                 [&](idx_t input) {
				                                 return std::find(materialized.begin(), materialized.end(), input) !=
				                                        materialized.end();
			                                 }),
			                  node.inputs.end());
			node.inputs.push_back(barrier);
			node.materialized_inputs.clear();
		}
		// A loop body's sinks run again on every iteration. The resource DAG
		// has one-shot frontiers: keep the body's resources eligible together,
		// including synthetic CTE/delim boundaries inside the repeated subtree.
		std::unordered_set<idx_t> repeated;
		for (auto *op : operators) {
			if (auto *recursive = dynamic_cast<PhysicalRecursiveCTE *>(op)) {
				VisitPhysicalExecutionGraph(recursive->children[1],
				                            [&](PhysicalOperator &child) { repeated.insert(Id(child)); });
			}
		}
		for (idx_t id = 0; id < nodes.size(); id++) {
			auto &node = nodes[id];
			if (repeated.count(id) || std::any_of(node.materialized_inputs.begin(), node.materialized_inputs.end(),
			                                      [&](idx_t child) { return repeated.count(child); })) {
				node.materialized_inputs.clear();
			}
		}
	}

	py::dict Metadata(const string &query_id, const ClientProperties &properties) const {
		py::list result_nodes;
		py::dict udf_node_ids;
		idx_t binding_id = 0;
		for (idx_t id = 0; id < nodes.size(); id++) {
			auto &node = nodes[id];
			py::dict metadata;
			metadata["node_id"] = std::to_string(id);
			metadata["node_name"] = node.name;
			metadata["input_node_ids"] = IdList(node.inputs);
			metadata["is_sink"] = node.is_sink;
			metadata["is_materialization_barrier"] = !node.materialized_inputs.empty();
			metadata["materialized_input_node_ids"] = IdList(node.materialized_inputs);
			metadata["num_partitions"] = 1;
			if (node.udf) {
				auto &payload = node.udf->payload;
				metadata["udf_payload"] = PythonObject::FromValue(payload, payload.type(), properties);
				udf_node_ids[py::str(std::to_string(id))] = std::to_string(binding_id++);
			} else {
				metadata["udf_payload"] = py::none();
			}
			result_nodes.append(std::move(metadata));
		}
		py::dict result;
		result["query_id"] = query_id;
		result["nodes"] = std::move(result_nodes);
		result["terminal_node_ids"] = IdList({0});
		result["udf_node_ids"] = std::move(udf_node_ids);
		return result;
	}

private:
	std::unordered_map<const PhysicalOperator *, idx_t> ids;
	vector<PhysicalOperator *> operators;
	vector<NativeResourceNode> nodes;

	idx_t Id(const PhysicalOperator &op) const {
		auto entry = ids.find(&op);
		if (entry == ids.end()) {
			throw InternalException("Native resource dependency is absent from the physical execution graph");
		}
		return entry->second;
	}

	static py::list IdList(const vector<idx_t> &values) {
		py::list result;
		for (auto id : values) {
			result.append(std::to_string(id));
		}
		return result;
	}

	idx_t InputBarrier(const string &name, const PhysicalOperator &producer) {
		auto id = nodes.size();
		nodes.push_back({name, {Id(producer)}, {Id(producer)}, true, nullptr});
		return id;
	}

	void MaterializedCTE(const PhysicalCTE &op) {
		if (op.children.size() != 2) {
			throw InternalException("Native CTE resource metadata requires a producer and a consumer");
		}
		auto input = InputBarrier("CTEInput", op.children[0]);
		for (auto &scan : op.cte_scans) {
			nodes[Id(scan.get())].inputs = {input};
		}
		auto &node = nodes[Id(op)];
		node.inputs = {Id(op.children[1]), input};
		node.materialized_inputs.clear();
	}

	void DelimJoin(const PhysicalDelimJoin &op) {
		if (op.children.size() != 1 || op.join.children.size() != 2) {
			throw InternalException("Native delim resource metadata requires an input and an owned binary join");
		}
		auto input = InputBarrier("DelimInput", op.children[0]);
		auto &distinct = nodes[Id(op.distinct)];
		distinct.inputs = {input};
		distinct.materialized_inputs.clear();
		for (auto &scan : op.delim_scans) {
			nodes[Id(scan.get())].inputs = {Id(op.distinct)};
		}
		const idx_t cached_side = op.type == PhysicalOperatorType::LEFT_DELIM_JOIN ? 0 : 1;
		nodes[Id(op.join.children[cached_side])].inputs = {input};
		if (op.type == PhysicalOperatorType::RIGHT_DELIM_JOIN) {
			// The delim producer already built the RHS (BuildJoinPipelines(false)).
			nodes[Id(op.join)].materialized_inputs.clear();
		}
		auto &node = nodes[Id(op)];
		node.inputs = {Id(op.join), Id(op.distinct)};
		node.materialized_inputs.clear();
	}
};

} // namespace

pybind11::dict CollectNativeResourceGraph(PhysicalOperator &root, const string &query_id, ClientContext &context) {
	return NativeResourceGraph(root, context).Metadata(query_id, context.GetClientProperties());
}

pybind11::dict CollectNativeLocalResourceGraph(ClientContext &context, PreparedStatementData &prepared) {
	if (!prepared.physical_plan || !prepared.physical_plan->HasRoot()) {
		throw InternalException("local runtime graph requires a prepared physical plan");
	}
	return CollectNativeResourceGraph(prepared.physical_plan->Root(), UUID::ToString(UUID::GenerateRandomUUID()),
	                                  context);
}

py::dict DuckDBPyRelation::CollectLocalResourceGraph() {
	auto query_lock = LockForQuery();
	if (GetRunnerType() != "local-fast") {
		throw InvalidInputException("Native resource metadata requires a local-fast relation");
	}
	auto relation = GetRelation();
	auto context = relation->context->GetContext();
	py::dict metadata;
	context->RunFunctionInTransaction([&]() {
		auto binder = Binder::CreateBinder(*context);
		Planner planner(*context);
		planner.CreatePlan(make_uniq<RelationStatement>(relation, *binder));
		if (!planner.plan || !planner.properties.bound_all_parameters) {
			throw InvalidInputException("Native resource metadata requires a fully bound relation");
		}
		auto plan = std::move(planner.plan);
		if (context->config.enable_optimizer && plan->RequireOptimizer()) {
			Optimizer optimizer(*planner.binder, *context);
			plan = optimizer.Optimize(std::move(plan));
		}
		PhysicalPlanGenerator generator(*context);
		auto physical = generator.Plan(std::move(plan));
		metadata = CollectNativeResourceGraph(physical->Root(), UUID::ToString(UUID::GenerateRandomUUID()), *context);
	});
	return metadata;
}

} // namespace duckdb
