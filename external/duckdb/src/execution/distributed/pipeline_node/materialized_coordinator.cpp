// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/materialized_coordinator.hpp"
#include "duckdb/execution/distributed/exchange/exchange.hpp"
#include "duckdb/execution/distributed/exchange/exchange_manager.hpp"
#include "duckdb/execution/distributed/pipeline_node/shuffles/repartition.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <unordered_set>

namespace duckdb {
namespace distributed {
namespace {

static std::string MakeCoordinatorExchangeId(const PipelineNodeContext &context) {
	std::string prefix = context.query_id().empty() ? std::string("query") : context.query_id();
	return prefix + "_coordinator_exchange_" + std::to_string(context.node_id());
}

static DuckDBResult<vector<LogicalType>> ResolveSchemaTypes(const SchemaRef &schema) {
	vector<LogicalType> types;
	if (!schema) {
		return DuckDBResult<vector<LogicalType>>::err(
		    DuckDBError::invalid_state_error("materialized coordinator requires an output schema"));
	}
	if (schema->id() == duckdb::LogicalTypeId::STRUCT) {
		const auto &children = duckdb::StructType::GetChildTypes(*schema);
		types.reserve(children.size());
		for (const auto &child : children) {
			types.push_back(child.second);
		}
	} else {
		types.push_back(*schema);
	}
	return DuckDBResult<vector<LogicalType>>::ok(std::move(types));
}

static vector<std::string> CollectCoordinatorSourceNodes(const std::vector<ExchangeSourceHandle> &handles) {
	std::unordered_set<std::string> node_set;
	for (const auto &handle : handles) {
		if (!handle.node_id.empty()) {
			node_set.insert(handle.node_id);
		}
	}
	vector<std::string> source_nodes(node_set.begin(), node_set.end());
	std::sort(source_nodes.begin(), source_nodes.end());
	return source_nodes;
}

static idx_t EstimateRowsFromHandles(const std::vector<ExchangeSourceHandle> &handles) {
	idx_t total = 0;
	for (const auto &handle : handles) {
		for (const auto &file : handle.files) {
			if (std::numeric_limits<idx_t>::max() - total < file.rows) {
				return std::numeric_limits<idx_t>::max();
			}
			total += file.rows;
		}
	}
	return total;
}

DuckDBResult<void> RunMaterializedCoordinator(const std::shared_ptr<PipelineNodeImpl> &node,
                                              const std::shared_ptr<SubmittableTaskStream<WorkerTask>> &input_stream,
                                              const std::shared_ptr<Sender<SubmittableTask<WorkerTask>>> &result_tx,
                                              const std::shared_ptr<TaskIDCounter> &task_id_counter,
                                              const MaterializedPlanBuilder &final_plan_builder,
                                              const PerTaskMaterializedPlanBuilderFactory &per_task_builder_factory,
                                              std::shared_ptr<FteTaskSubmitter> fte_task_submitter,
                                              ::duckdb::ClientContext *client_context,
                                              std::shared_ptr<ExchangeManager> exchange_mgr) {
	if (!exchange_mgr) {
		result_tx->close();
		return DuckDBResult<void>::err(
		    DuckDBError::invalid_state_error("materialized coordinator requires an ExchangeManager"));
	}

	auto types_res = ResolveSchemaTypes(node->config().schema());
	if (types_res.is_err()) {
		result_tx->close();
		return DuckDBResult<void>::err(types_res.error());
	}
	auto output_types = std::move(types_res.value());
	const auto exchange_id = MakeCoordinatorExchangeId(node->context());
	constexpr idx_t num_partitions = 1;

	ExchangeContext exchange_ctx;
	exchange_ctx.query_id = node->context().query_id();
	exchange_ctx.exchange_id = exchange_id;
	exchange_mgr->SetContext(client_context);
	auto exchange_unique = exchange_mgr->CreateExchange(exchange_ctx, num_partitions);
	if (!exchange_unique) {
		result_tx->close();
		return DuckDBResult<void>::err(
		    DuckDBError::internal_error("materialized coordinator failed to create exchange"));
	}
	auto exchange = std::shared_ptr<Exchange>(exchange_unique.release());
	auto sink_task_counter = std::make_shared<std::atomic<idx_t>>(0);

	MaterializedPlanBuilder local_plan_builder;
	if (per_task_builder_factory) {
		local_plan_builder = per_task_builder_factory(0);
	}

	auto sink_plan_builder = [local_plan_builder, exchange, exchange_mgr, exchange_id, sink_task_counter,
	                          num_partitions](DuckPhysicalPlanRef plan) -> DuckPhysicalPlanRef {
		if (local_plan_builder) {
			plan = local_plan_builder(std::move(plan));
		}
		auto task_partition_id = sink_task_counter->fetch_add(1);
		auto sink_handle = exchange->AddSink(task_partition_id);
		auto sink_instance = exchange->InstantiateSink(sink_handle, /*attempt_id=*/0);
		return AddRemoteExchangeSinkPlan(std::move(plan), nullptr, num_partitions, exchange_id, sink_instance,
		                                 exchange_mgr);
	};

	auto sink_stream = input_stream->pipeline_instruction(node, sink_plan_builder, client_context);
	auto mat_res = sink_stream.materialize(fte_task_submitter.get());
	if (!mat_res.success) {
		result_tx->close();
		exchange->Close();
		return DuckDBResult<void>::err(DuckDBError::internal_error(mat_res.error));
	}

	try {
		RecordRemoteExchangeFinishedSinks(*exchange, mat_res.outputs, "coordinator");
		exchange->AllRequiredSinksFinished();
	} catch (const std::exception &ex) {
		result_tx->close();
		exchange->Close();
		return DuckDBResult<void>::err(DuckDBError::internal_error(ex.what()));
	}

	auto source_handles = exchange->GetSourceHandles();
	auto source_nodes = CollectCoordinatorSourceNodes(source_handles);
	auto estimated_cardinality = EstimateRowsFromHandles(source_handles);
	auto source_plan =
	    MakeRemoteExchangeSourcePlan(output_types, estimated_cardinality, exchange_id, vector<idx_t> {0},
	                                 std::move(source_handles), exchange_mgr, source_nodes, optional_idx());
	if (!source_plan || !source_plan->HasRoot()) {
		result_tx->close();
		exchange->Close();
		return DuckDBResult<void>::err(
		    DuckDBError::internal_error("materialized coordinator failed to create exchange source plan"));
	}

	DuckPhysicalPlanRef final_plan;
	try {
		final_plan = final_plan_builder(std::move(source_plan));
	} catch (const std::exception &ex) {
		result_tx->close();
		exchange->Close();
		return DuckDBResult<void>::err(DuckDBError::internal_error(ex.what()));
	}

	TaskContext task_context =
	    TaskContext::from_node_context(node->context().query_idx(), node->node_id(), task_id_counter->next());
	WorkerTask final_task(task_context, final_plan, node->config().execution_config(), node->context().to_hashmap());

	auto send_result = result_tx->send(SubmittableTask<WorkerTask>(std::move(final_task)));
	result_tx->close();
	exchange->Close();
	if (send_result.is_err()) {
		return DuckDBResult<void>::ok();
	}
	return DuckDBResult<void>::ok();
}

} // namespace

bool ChildHasMultiplePartitions(const PipelineNodeRef &child) {
	if (!child) {
		return false;
	}
	auto clustering_spec = child->config().clustering_spec();
	if (!clustering_spec) {
		return false;
	}
	return clustering_spec->num_partitions() > 1;
}

SubmittableTaskStream<WorkerTask> ProduceWithMaterializedCoordinator(
    PlanExecutionContext &plan_context, const PipelineNodeRef &child, const std::shared_ptr<PipelineNodeImpl> &node,
    MaterializedPlanBuilder final_plan_builder, PerTaskMaterializedPlanBuilderFactory per_task_builder_factory,
    std::shared_ptr<ExchangeManager> exchange_mgr) {
	auto input_stream = child->produce_tasks(plan_context);

	auto channel_pair = create_channel<SubmittableTask<WorkerTask>>(1);
	auto result_tx = std::move(channel_pair.first);
	auto result_rx = std::move(channel_pair.second);
	auto input_stream_ptr = std::make_shared<SubmittableTaskStream<WorkerTask>>(std::move(input_stream));
	auto result_tx_ptr = std::make_shared<Sender<SubmittableTask<WorkerTask>>>(std::move(result_tx));
	auto task_id_counter = std::make_shared<TaskIDCounter>(plan_context.task_id_counter());
	auto client_context = plan_context.client_context();
	auto final_plan_builder_ptr = std::make_shared<MaterializedPlanBuilder>(std::move(final_plan_builder));
	auto per_task_builder_factory_ptr =
	    std::make_shared<PerTaskMaterializedPlanBuilderFactory>(std::move(per_task_builder_factory));
	auto fte_task_submitter = plan_context.fte_task_submitter_ref();

	plan_context.spawn([node, input_stream_ptr, result_tx_ptr, task_id_counter, final_plan_builder_ptr,
	                    per_task_builder_factory_ptr, fte_task_submitter, client_context,
	                    exchange_mgr]() mutable -> DuckDBResult<void> {
		return RunMaterializedCoordinator(node, input_stream_ptr, result_tx_ptr, task_id_counter,
		                                  *final_plan_builder_ptr, *per_task_builder_factory_ptr, fte_task_submitter,
		                                  client_context, exchange_mgr);
	});

	return SubmittableTaskStream<WorkerTask>::from_receiver(std::move(result_rx));
}

} // namespace distributed
} // namespace duckdb
