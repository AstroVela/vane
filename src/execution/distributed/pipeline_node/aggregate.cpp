// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

// Minimal implementation of AggregateNode and helpers translated from the Rust file.
#include "duckdb/execution/distributed/pipeline_node/aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_ungrouped_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_perfecthash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_partitioned_aggregate.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/function/aggregate_state.hpp"
#include "duckdb/function/scalar/generic_common.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

#include "duckdb/execution/distributed/plan/distributed_physical_plan.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"

#include <cstring>

namespace duckdb {
namespace distributed {

struct MergeAggregateBindData : public FunctionData {
	AggregateFunction child_function;
	idx_t state_size;
	unique_ptr<FunctionData> child_bind_data;
	unique_ptr<BoundAggregateExpression> child_expr;

	MergeAggregateBindData(AggregateFunction function_p, idx_t state_size_p, unique_ptr<FunctionData> bind_data_p,
	                       unique_ptr<BoundAggregateExpression> child_expr_p = nullptr)
	    : child_function(std::move(function_p)), state_size(state_size_p), child_bind_data(std::move(bind_data_p)),
	      child_expr(std::move(child_expr_p)) {
	}

	unique_ptr<FunctionData> Copy() const override {
		unique_ptr<BoundAggregateExpression> expr_copy;
		if (child_expr) {
			expr_copy = unique_ptr_cast<Expression, BoundAggregateExpression>(child_expr->Copy());
		}
		return make_uniq<MergeAggregateBindData>(
		    child_function, state_size, child_bind_data ? child_bind_data->Copy() : nullptr, std::move(expr_copy));
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<MergeAggregateBindData>();
		if (state_size != other.state_size || child_function != other.child_function) {
			return false;
		}
		if (child_bind_data && other.child_bind_data) {
			return child_bind_data->Equals(*other.child_bind_data);
		}
		if (child_bind_data || other.child_bind_data) {
			return false;
		}
		if (child_expr && other.child_expr) {
			return child_expr->Equals(*other.child_expr);
		}
		return child_expr == nullptr && other.child_expr == nullptr;
	}
};

struct MergeAggregateFunctionInfo : public AggregateFunctionInfo {
	AggregateFunction child_function;
	idx_t state_size;

	MergeAggregateFunctionInfo(AggregateFunction function_p, idx_t state_size_p)
	    : child_function(std::move(function_p)), state_size(state_size_p) {
	}
};

MergeAggregateBindData &GetMergeBindData(AggregateInputData &input) {
	if (!input.bind_data) {
		throw InternalException("merge aggregate missing bind data");
	}
	return input.bind_data->Cast<MergeAggregateBindData>();
}

idx_t MergeAggregateStateSize(const AggregateFunction &function) {
	if (!function.function_info) {
		throw InternalException("merge aggregate missing function info");
	}
	auto &info = function.function_info->Cast<MergeAggregateFunctionInfo>();
	return info.state_size;
}

void MergeAggregateInitialize(const AggregateFunction &function, data_ptr_t state) {
	if (!function.function_info) {
		throw InternalException("merge aggregate missing function info");
	}
	auto &info = function.function_info->Cast<MergeAggregateFunctionInfo>();
	info.child_function.initialize(info.child_function, state);
}

void MergeAggregateUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &state,
                          idx_t count) {
	if (count == 0) {
		return;
	}
	D_ASSERT(input_count == 1);
	auto &bind_data = GetMergeBindData(aggr_input_data);
	auto &child = bind_data.child_function;
	if (!child.combine) {
		throw InternalException("merge aggregate missing combine method");
	}

	UnifiedVectorFormat input_data;
	inputs[0].ToUnifiedFormat(count, input_data);
	auto input_states = UnifiedVectorFormat::GetData<string_t>(input_data);

	UnifiedVectorFormat state_data;
	state.ToUnifiedFormat(count, state_data);
	auto state_ptrs = UnifiedVectorFormat::GetData<data_ptr_t>(state_data);

	auto buffer = make_unsafe_uniq_array<data_t>(bind_data.state_size);
	Vector source_vec(Value::POINTER(CastPointerToValue(buffer.get())));
	AggregateInputData child_input(bind_data.child_bind_data.get(), aggr_input_data.allocator,
	                               AggregateCombineType::ALLOW_DESTRUCTIVE);

	for (idx_t row = 0; row < count; row++) {
		auto input_idx = input_data.sel->get_index(row);
		if (!input_data.validity.RowIsValid(input_idx)) {
			continue;
		}
		auto state_idx = state_data.sel->get_index(row);
		auto &entry = input_states[input_idx];
		if (entry.GetSize() != bind_data.state_size) {
			throw IOException("Aggregate state size mismatch, expect %llu, got %llu", bind_data.state_size,
			                  entry.GetSize());
		}
		memcpy(buffer.get(), entry.GetData(), bind_data.state_size);
		Vector target_vec(Value::POINTER(CastPointerToValue(state_ptrs[state_idx])));
		child.combine(source_vec, target_vec, child_input, 1);
	}
}

void MergeAggregateCombine(Vector &state, Vector &combined, AggregateInputData &aggr_input_data, idx_t count) {
	auto &bind_data = GetMergeBindData(aggr_input_data);
	auto &child = bind_data.child_function;
	if (!child.combine) {
		throw InternalException("merge aggregate missing combine method");
	}
	AggregateInputData child_input(bind_data.child_bind_data.get(), aggr_input_data.allocator,
	                               aggr_input_data.combine_type);
	child.combine(state, combined, child_input, count);
}

void MergeAggregateFinalize(Vector &state, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                            idx_t offset) {
	auto &bind_data = GetMergeBindData(aggr_input_data);
	auto &child = bind_data.child_function;
	if (!child.finalize) {
		throw InternalException("merge aggregate missing finalize method");
	}
	AggregateInputData child_input(bind_data.child_bind_data.get(), aggr_input_data.allocator,
	                               aggr_input_data.combine_type);
	child.finalize(state, child_input, result, count, offset);
}

void MergeAggregateSimpleUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                data_ptr_t state, idx_t count) {
	Vector state_vec(Value::POINTER(CastPointerToValue(state)));
	MergeAggregateUpdate(inputs, aggr_input_data, input_count, state_vec, count);
}

AggregateFunction MakeMergeAggregateFunction(const AggregateFunction &child_function, const LogicalType &state_type,
                                             idx_t state_size) {
	AggregateFunction merge_function("merge_" + child_function.name, {state_type}, child_function.GetReturnType(),
	                                 MergeAggregateStateSize, MergeAggregateInitialize, MergeAggregateUpdate,
	                                 MergeAggregateCombine, MergeAggregateFinalize,
	                                 FunctionNullHandling::SPECIAL_HANDLING, MergeAggregateSimpleUpdate);
	merge_function.order_dependent = child_function.order_dependent;
	merge_function.distinct_dependent = child_function.distinct_dependent;
	merge_function.function_info = make_shared_ptr<MergeAggregateFunctionInfo>(child_function, state_size);
	merge_function.serialize = MergeAggregateSerialize;
	merge_function.deserialize = MergeAggregateDeserialize;
	return merge_function;
}

void MergeAggregateSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data_p,
                             const AggregateFunction &function) {
	if (!bind_data_p) {
		throw SerializationException("Merge aggregate serialize requires bind data for %s", function.name);
	}
	auto &bind_data = bind_data_p->Cast<MergeAggregateBindData>();
	if (!bind_data.child_expr) {
		throw SerializationException("Merge aggregate serialize missing child expression for %s", function.name);
	}
	auto aggregate_expr = bind_data.child_expr->Copy();
	serializer.WriteProperty(100, "aggregate", aggregate_expr);
	serializer.WriteProperty(101, "state_size", bind_data.state_size);
}

unique_ptr<FunctionData> MergeAggregateDeserialize(Deserializer &deserializer, AggregateFunction &function) {
	auto aggregate_expr = deserializer.ReadProperty<unique_ptr<Expression>>(100, "aggregate");
	if (!aggregate_expr || aggregate_expr->GetExpressionType() != ExpressionType::BOUND_AGGREGATE) {
		throw SerializationException("Invalid merge aggregate payload for %s", function.name);
	}
	auto &bound_aggregate = aggregate_expr->Cast<BoundAggregateExpression>();
	idx_t state_size = 0;
	deserializer.ReadPropertyWithDefault<idx_t>(101, "state_size", state_size);
	if (state_size == 0) {
		state_size = bound_aggregate.function.state_size(bound_aggregate.function);
	}
	if (function.arguments.empty()) {
		throw SerializationException("Merge aggregate missing arguments for %s", function.name);
	}
	auto state_type = function.arguments[0];
	auto merge_function = MakeMergeAggregateFunction(bound_aggregate.function, state_type, state_size);
	function = std::move(merge_function);

	auto child_expr = unique_ptr_cast<Expression, BoundAggregateExpression>(std::move(aggregate_expr));
	unique_ptr<FunctionData> child_bind;
	if (child_expr->bind_info) {
		child_bind = child_expr->bind_info->Copy();
	}
	return make_uniq<MergeAggregateBindData>(child_expr->function, state_size, std::move(child_bind),
	                                         std::move(child_expr));
}

std::vector<BoundExpr> MakeGroupByReferences(const std::vector<BoundExpr> &group_by) {
	std::vector<BoundExpr> refs;
	refs.reserve(group_by.size());
	for (idx_t idx = 0; idx < group_by.size(); idx++) {
		if (!group_by[idx]) {
			refs.emplace_back(nullptr);
			continue;
		}
		auto ref = make_uniq<BoundReferenceExpression>(group_by[idx]->return_type, idx);
		refs.emplace_back(ExpressionRef(ref.release()));
	}
	return refs;
}

SchemaRef MakeSchemaFromTypes(const std::vector<LogicalType> &types, const SchemaRef &fallback) {
	auto schema = MakeSchemaRef(types);
	if (schema) {
		return schema;
	}
	return fallback;
}

AggregateNode::AggregateNode(NodeID node_id, const PlanConfig &plan_config, std::vector<BoundExprRef> group_by,
                             std::vector<BoundAggExprRef> aggs, SchemaRef output_schema,
                             DistributedPipelineNodeRef child)
    : config_(output_schema, plan_config.config, child ? child->config().clustering_spec() : ClusteringSpecRef()),
      context_(plan_config.query_idx, plan_config.query_id, node_id, AggregateNode::node_name(group_by)),
      node_id_(node_id), group_by_(std::move(group_by)), aggs_(std::move(aggs)), child_(std::move(child)) {
}

NodeName AggregateNode::node_name(const std::vector<BoundExprRef> &group_by) {
	if (group_by.empty()) {
		return "Aggregate";
	}
	return "GroupBy Aggregate";
}

DistributedPipelineNodeRef AggregateNode::into_node() {
	// Wrap this AggregateNode instance into a DistributedPipelineNode
	return std::make_shared<DistributedPipelineNode>(shared_from_this());
}

std::vector<PipelineNodeRef> AggregateNode::children() const {
	if (child_)
		return std::vector<PipelineNodeRef> {child_->inner()};
	return {};
}

std::vector<std::string> AggregateNode::multiline_display(bool verbose) const {
	std::vector<std::string> res;
	res.push_back("Aggregate Node");
	return res;
}

SubmittableTaskStream<WorkerTask> AggregateNode::produce_tasks(PlanExecutionContext &plan_context) {
	// Produce tasks by delegating to the child and inserting aggregation operators
	auto input_node = child_->produce_tasks(plan_context);
	auto self = shared_from_this();

	return input_node.pipeline_instruction(
	    self,
	    [self, &plan_context](DuckPhysicalPlanRef input) {
		    auto plan = input; // base scan plan
		    Allocator &alloc = Allocator::DefaultAllocator();

		    if (self->group_by_.empty()) {
			    // Un-grouped aggregate
			    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> agg_exprs;
			    duckdb::vector<duckdb::LogicalType> out_types;
			    for (const auto &a : self->aggs_) {
				    if (a) {
					    auto copy = a->Copy();
					    out_types.push_back(copy->return_type);
					    agg_exprs.push_back(std::move(copy));
				    }
			    }
			    auto &old_root = plan->Root();
			    auto &agg_op = plan->template Make<duckdb::PhysicalUngroupedAggregate>(
			        out_types, std::move(agg_exprs), 0, TupleDataValidityType::CAN_HAVE_NULL_VALUES);
			    agg_op.children.push_back(old_root);
			    plan->SetRoot(agg_op);
			    return plan;
		    } else {
			    auto *client = plan_context.client_context();
			    std::unique_ptr<duckdb::DuckDB> local_db;
			    std::unique_ptr<duckdb::Connection> local_conn;
			    if (!client) {
				    // Fall back to a local DuckDB context for plan construction when none is provided.
				    local_db = std::unique_ptr<duckdb::DuckDB>(new duckdb::DuckDB(nullptr));
				    local_conn = std::unique_ptr<duckdb::Connection>(new duckdb::Connection(*local_db));
				    client = local_conn->context.get();
			    }

			    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> group_exprs;
			    for (const auto &g : self->group_by_)
				    if (g)
					    group_exprs.push_back(g->Copy());

			    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> agg_exprs;
			    for (const auto &a : self->aggs_)
				    if (a)
					    agg_exprs.push_back(a->Copy());

			    duckdb::vector<duckdb::LogicalType> types;
			    for (auto &g : group_exprs)
				    types.push_back(g->return_type);
			    for (auto &a : agg_exprs)
				    types.push_back(a->return_type);

			    idx_t estimated_cardinality = 0;
			    // `plan` is a dependent type, so use `template` to disambiguate the
			    // template member call. Ensure the header above is included so the
			    // type `duckdb::PhysicalHashAggregate` is visible.
			    auto &old_root = plan->Root();
			    auto &hash_agg = plan->template Make<duckdb::PhysicalHashAggregate>(
			        *client, types, std::move(agg_exprs), std::move(group_exprs), estimated_cardinality);
			    hash_agg.children.push_back(old_root);
			    plan->SetRoot(hash_agg);
			    return plan;
		    }
	    },
	    plan_context.client_context());
}

static duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>
CopyExpressionsForSpecialAggregateNode(const std::vector<BoundExprRef> &exprs) {
	duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> copies;
	copies.reserve(exprs.size());
	for (const auto &expr : exprs) {
		if (expr) {
			copies.push_back(expr->Copy());
		} else {
			copies.push_back(nullptr);
		}
	}
	return copies;
}

PerfectHashAggregateNode::PerfectHashAggregateNode(NodeID node_id, std::vector<BoundExprRef> group_by,
                                                   std::vector<BoundAggExprRef> aggs, std::vector<Value> group_minima,
                                                   std::vector<idx_t> required_bits,
                                                   std::vector<LogicalType> output_types,
                                                   DistributedPipelineNodeRef child)
    : config_(MakeSchemaRef(output_types), child ? child->config().execution_config() : DuckDBExecutionConfigRef(),
              child ? child->config().clustering_spec() : ClusteringSpec::unknown_with_num_partitions(1)),
      context_(InheritPipelineNodeContext(child, node_id, "PerfectHashAggregate")), node_id_(node_id),
      group_by_(std::move(group_by)), aggs_(std::move(aggs)), group_minima_(std::move(group_minima)),
      required_bits_(std::move(required_bits)), child_(std::move(child)), output_types_(std::move(output_types)) {
}

std::vector<PipelineNodeRef> PerfectHashAggregateNode::children() const {
	if (child_) {
		return {child_->inner()};
	}
	return {};
}

std::vector<std::string> PerfectHashAggregateNode::multiline_display(bool /*verbose*/) const {
	return {std::string("PerfectHashAggregate")};
}

SubmittableTaskStream<WorkerTask> PerfectHashAggregateNode::produce_tasks(PlanExecutionContext &plan_context) {
	auto input_node = child_->produce_tasks(plan_context);
	auto self = shared_from_this();

	return input_node.pipeline_instruction(
	    self,
	    [self, &plan_context](DuckPhysicalPlanRef input) {
		    auto plan = input;
		    auto groups = CopyExpressionsForSpecialAggregateNode(self->group_by_);
		    auto aggs = CopyExpressionsForSpecialAggregateNode(self->aggs_);
		    duckdb::vector<Value> group_minima;
		    group_minima.reserve(self->group_minima_.size());
		    for (const auto &val : self->group_minima_) {
			    group_minima.push_back(val);
		    }
		    duckdb::vector<idx_t> required_bits;
		    required_bits.reserve(self->required_bits_.size());
		    for (const auto &bits : self->required_bits_) {
			    required_bits.push_back(bits);
		    }
		    duckdb::vector<LogicalType> types;
		    types.reserve(self->output_types_.size());
		    for (const auto &type : self->output_types_) {
			    types.push_back(type);
		    }
		    idx_t estimated_cardinality = plan->Root().estimated_cardinality;

		    auto *client = plan_context.client_context();
		    std::unique_ptr<duckdb::DuckDB> local_db;
		    std::unique_ptr<duckdb::Connection> local_conn;
		    if (!client) {
			    local_db = std::unique_ptr<duckdb::DuckDB>(new duckdb::DuckDB(nullptr));
			    local_conn = std::unique_ptr<duckdb::Connection>(new duckdb::Connection(*local_db));
			    client = local_conn->context.get();
		    }

		    auto &old_root = plan->Root();
		    auto &agg_op = plan->template Make<duckdb::PhysicalPerfectHashAggregate>(
		        *client, std::move(types), std::move(aggs), std::move(groups), std::move(group_minima),
		        std::move(required_bits), estimated_cardinality);
		    agg_op.children.push_back(old_root);
		    plan->SetRoot(agg_op);
		    return plan;
	    },
	    plan_context.client_context());
}

PartitionedAggregateNode::PartitionedAggregateNode(NodeID node_id, std::vector<BoundExprRef> group_by,
                                                   std::vector<BoundAggExprRef> aggs, std::vector<column_t> partitions,
                                                   std::vector<LogicalType> output_types,
                                                   DistributedPipelineNodeRef child)
    : config_(MakeSchemaRef(output_types), child ? child->config().execution_config() : DuckDBExecutionConfigRef(),
              child ? child->config().clustering_spec() : ClusteringSpec::unknown_with_num_partitions(1)),
      context_(InheritPipelineNodeContext(child, node_id, "PartitionedAggregate")), node_id_(node_id),
      group_by_(std::move(group_by)), aggs_(std::move(aggs)), partitions_(std::move(partitions)),
      child_(std::move(child)), output_types_(std::move(output_types)) {
}

std::vector<PipelineNodeRef> PartitionedAggregateNode::children() const {
	if (child_) {
		return {child_->inner()};
	}
	return {};
}

std::vector<std::string> PartitionedAggregateNode::multiline_display(bool /*verbose*/) const {
	return {std::string("PartitionedAggregate")};
}

SubmittableTaskStream<WorkerTask> PartitionedAggregateNode::produce_tasks(PlanExecutionContext &plan_context) {
	auto input_node = child_->produce_tasks(plan_context);
	auto self = shared_from_this();

	return input_node.pipeline_instruction(
	    self,
	    [self, &plan_context](DuckPhysicalPlanRef input) {
		    auto plan = input;
		    auto groups = CopyExpressionsForSpecialAggregateNode(self->group_by_);
		    auto aggs = CopyExpressionsForSpecialAggregateNode(self->aggs_);
		    duckdb::vector<column_t> partitions;
		    partitions.reserve(self->partitions_.size());
		    for (const auto &partition : self->partitions_) {
			    partitions.push_back(partition);
		    }
		    duckdb::vector<LogicalType> types;
		    types.reserve(self->output_types_.size());
		    for (const auto &type : self->output_types_) {
			    types.push_back(type);
		    }
		    idx_t estimated_cardinality = plan->Root().estimated_cardinality;

		    auto *client = plan_context.client_context();
		    std::unique_ptr<duckdb::DuckDB> local_db;
		    std::unique_ptr<duckdb::Connection> local_conn;
		    if (!client) {
			    local_db = std::unique_ptr<duckdb::DuckDB>(new duckdb::DuckDB(nullptr));
			    local_conn = std::unique_ptr<duckdb::Connection>(new duckdb::Connection(*local_db));
			    client = local_conn->context.get();
		    }

		    auto &old_root = plan->Root();
		    auto &agg_op = plan->template Make<duckdb::PhysicalPartitionedAggregate>(
		        *client, std::move(types), std::move(aggs), std::move(groups), std::move(partitions),
		        estimated_cardinality);
		    agg_op.children.push_back(old_root);
		    plan->SetRoot(agg_op);
		    return plan;
	    },
	    plan_context.client_context());
}

DuckDBResult<GroupByAggSplit> split_groupby_aggs(const std::vector<BoundExpr> &group_by,
                                                 const std::vector<BoundAggExpr> &aggs,
                                                 const std::vector<BoundExpr> &partition_by,
                                                 const SchemaRef &input_schema) {
	GroupByAggSplit res;
	res.first_stage_group_by = group_by;

	std::vector<BoundExpr> group_by_refs = MakeGroupByReferences(group_by);
	if (partition_by.empty()) {
		res.partition_by = {};
	} else {
		res.partition_by = group_by_refs;
	}
	res.second_stage_group_by = group_by_refs;

	std::vector<LogicalType> first_stage_types;
	std::vector<LogicalType> second_stage_types;
	first_stage_types.reserve(group_by.size() + aggs.size());
	second_stage_types.reserve(group_by.size() + aggs.size());

	for (auto &expr : group_by) {
		if (expr) {
			first_stage_types.push_back(expr->return_type);
			second_stage_types.push_back(expr->return_type);
		}
	}

	res.first_stage_aggs.reserve(aggs.size());
	res.second_stage_aggs.reserve(aggs.size());

	for (idx_t agg_idx = 0; agg_idx < aggs.size(); agg_idx++) {
		auto &agg_ref = aggs[agg_idx];
		if (!agg_ref) {
			return DuckDBResult<GroupByAggSplit>::err(DuckDBError::invalid_state_error("aggregate expression is null"));
		}
		if (agg_ref->GetExpressionType() != ExpressionType::BOUND_AGGREGATE) {
			return DuckDBResult<GroupByAggSplit>::err(
			    DuckDBError::invalid_state_error("aggregate expression is not a bound aggregate"));
		}
		auto &agg_expr = agg_ref->Cast<BoundAggregateExpression>();
		if (agg_expr.IsDistinct()) {
			return DuckDBResult<GroupByAggSplit>::err(
			    DuckDBError::value_error("distinct aggregates are not supported for distributed merge"));
		}
		if (agg_expr.filter) {
			return DuckDBResult<GroupByAggSplit>::err(
			    DuckDBError::value_error("filtered aggregates are not supported for distributed merge"));
		}
		if (agg_expr.order_bys) {
			return DuckDBResult<GroupByAggSplit>::err(
			    DuckDBError::value_error("ordered aggregates are not supported for distributed merge"));
		}

		unique_ptr<BoundAggregateExpression> agg_copy;
		try {
			agg_copy = unique_ptr_cast<Expression, BoundAggregateExpression>(agg_expr.Copy());
		} catch (const std::exception &ex) {
			return DuckDBResult<GroupByAggSplit>::err(DuckDBError::invalid_state_error(ex.what()));
		}

		unique_ptr<BoundAggregateExpression> export_agg;
		try {
			export_agg = ExportAggregateFunction::Bind(std::move(agg_copy));
		} catch (const Exception &ex) {
			return DuckDBResult<GroupByAggSplit>::err(DuckDBError::value_error(ex.what()));
		} catch (const std::exception &ex) {
			return DuckDBResult<GroupByAggSplit>::err(DuckDBError::value_error(ex.what()));
		}

		auto state_type = export_agg->return_type;
		first_stage_types.push_back(state_type);
		res.first_stage_aggs.emplace_back(ExpressionRef(export_agg.release()));

		auto state_size = agg_expr.function.state_size(agg_expr.function);
		auto merge_function = MakeMergeAggregateFunction(agg_expr.function, state_type, state_size);
		std::vector<unique_ptr<Expression>> merge_children;
		merge_children.push_back(make_uniq<BoundReferenceExpression>(state_type, group_by.size() + agg_idx));
		auto child_expr = unique_ptr_cast<Expression, BoundAggregateExpression>(agg_expr.Copy());
		auto merge_bind = make_uniq<MergeAggregateBindData>(agg_expr.function, state_size,
		                                                    agg_expr.bind_info ? agg_expr.bind_info->Copy() : nullptr,
		                                                    std::move(child_expr));
		auto merge_expr =
		    make_uniq<BoundAggregateExpression>(std::move(merge_function), std::move(merge_children), nullptr,
		                                        std::move(merge_bind), AggregateType::NON_DISTINCT);
		second_stage_types.push_back(merge_expr->return_type);
		res.second_stage_aggs.emplace_back(ExpressionRef(merge_expr.release()));
	}

	res.first_stage_schema = MakeSchemaFromTypes(first_stage_types, input_schema);
	res.second_stage_schema = MakeSchemaFromTypes(second_stage_types, input_schema);

	res.final_exprs.reserve(group_by.size() + aggs.size());
	for (auto &expr : group_by_refs) {
		res.final_exprs.emplace_back(expr);
	}
	for (idx_t agg_idx = 0; agg_idx < res.second_stage_aggs.size(); agg_idx++) {
		auto &agg_expr = res.second_stage_aggs[agg_idx];
		if (!agg_expr) {
			res.final_exprs.emplace_back(nullptr);
			continue;
		}
		auto ref = make_uniq<BoundReferenceExpression>(agg_expr->return_type, group_by.size() + agg_idx);
		res.final_exprs.emplace_back(ExpressionRef(ref.release()));
	}

	return DuckDBResult<GroupByAggSplit>::ok(std::move(res));
}

} // namespace distributed
} // namespace duckdb
