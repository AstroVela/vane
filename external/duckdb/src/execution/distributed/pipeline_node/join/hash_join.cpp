// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/join/hash_join.hpp"

#include <algorithm>
#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/execution/distributed/pipeline_node/binary_task_fan_in.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/hash_join_metadata.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/join_output_types.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/distributed/plan/exchange_source_task.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/execution/operator/join/physical_nested_loop_join.hpp"

namespace duckdb {
namespace distributed {

namespace {

MarkJoinBuildSummary ExtractMarkJoinBuildSummary(const TaskInputs &inputs, optional_idx source_node_id) {
	if (!source_node_id.IsValid()) {
		return MarkJoinBuildSummary();
	}
	auto entry = inputs.find(static_cast<SourceNodeId>(source_node_id.GetIndex()));
	if (entry == inputs.end() || entry->second.kind != TaskInput::Kind::ExchangeSourceTask) {
		throw InvalidInputException("HashJoinNode is missing its MARK build exchange source input");
	}
	auto descriptor = ExchangeSourceTaskDescriptor::DeserializeFromBytes(entry->second.exchange_source_task_bytes);
	if (!descriptor.mark_join_build_summary.valid || !descriptor.mark_join_build_summary.IsValid()) {
		throw InvalidInputException("HashJoinNode received an invalid global MARK build summary");
	}
	return descriptor.mark_join_build_summary;
}
} // namespace

HashJoinNode::HashJoinNode(NodeID node_id, const PlanConfig &plan_config, duckdb::vector<JoinCondition> conditions,
                           JoinType join_type, duckdb::vector<LogicalType> output_types,
                           duckdb::vector<LogicalType> delim_types, duckdb::vector<LogicalType> condition_types,
                           PhysicalHashJoin::JoinProjectionColumns payload_columns,
                           PhysicalHashJoin::JoinProjectionColumns lhs_output_columns,
                           PhysicalHashJoin::JoinProjectionColumns rhs_output_columns,
                           duckdb::vector<unique_ptr<BaseStatistics>> join_stats,
                           unique_ptr<JoinFilterPushdownInfo> filter_pushdown, idx_t estimated_cardinality,
                           std::shared_ptr<DistributedPipelineNode> left,
                           std::shared_ptr<DistributedPipelineNode> right, SchemaRef schema,
                           optional_idx mark_build_summary_source_node_id)
    : context_(plan_config.query_idx, plan_config.query_id, node_id, "HashJoin"), left_(std::move(left)),
      right_(std::move(right)), conditions_(std::move(conditions)), join_type_(join_type),
      output_types_(std::move(output_types)), delim_types_(std::move(delim_types)),
      condition_types_(std::move(condition_types)), payload_columns_(std::move(payload_columns)),
      lhs_output_columns_(std::move(lhs_output_columns)), rhs_output_columns_(std::move(rhs_output_columns)),
      join_stats_(std::move(join_stats)), filter_pushdown_(std::move(filter_pushdown)),
      estimated_cardinality_(estimated_cardinality),
      mark_build_summary_source_node_id_(mark_build_summary_source_node_id) {
	size_t num_partitions = 1;
	if (left_ && right_) {
		num_partitions = std::max(left_->config().clustering_spec()->num_partitions(),
		                          right_->config().clustering_spec()->num_partitions());
	} else if (left_) {
		num_partitions = left_->config().clustering_spec()->num_partitions();
	} else if (right_) {
		num_partitions = right_->config().clustering_spec()->num_partitions();
	}

	auto clustering = ClusteringSpec::unknown_with_num_partitions(num_partitions);
	config_ = PipelineNodeConfig(std::move(schema), plan_config.config, std::move(clustering));
}

std::shared_ptr<DistributedPipelineNode> HashJoinNode::into_node() {
	return std::make_shared<DistributedPipelineNode>(shared_from_this());
}

std::vector<PipelineNodeRef> HashJoinNode::children() const {
	std::vector<PipelineNodeRef> result;
	if (left_)
		result.push_back(left_->inner());
	if (right_)
		result.push_back(right_->inner());
	return result;
}

std::vector<std::string> HashJoinNode::multiline_display(bool /*verbose*/) const {
	std::vector<std::string> res;
	res.push_back("Hash Join");
	res.push_back("Join type: " + std::to_string(static_cast<int>(join_type_)));
	res.push_back("Conditions: " + std::to_string(conditions_.size()));
	return res;
}

duckdb::vector<JoinCondition> HashJoinNode::CopyConditions(const duckdb::vector<JoinCondition> &conditions) {
	duckdb::vector<JoinCondition> copy;
	copy.reserve(conditions.size());
	for (const auto &cond : conditions) {
		JoinCondition new_cond;
		new_cond.comparison = cond.comparison;
		if (cond.left) {
			new_cond.left = cond.left->Copy();
		}
		if (cond.right) {
			new_cond.right = cond.right->Copy();
		}
		copy.push_back(std::move(new_cond));
	}
	return copy;
}

unique_ptr<JoinFilterPushdownInfo> HashJoinNode::CopyFilterPushdownInfo(const JoinFilterPushdownInfo &info) {
	auto copy = make_uniq<JoinFilterPushdownInfo>();
	copy->join_condition = info.join_condition;
	copy->probe_info.reserve(info.probe_info.size());
	for (const auto &probe : info.probe_info) {
		JoinFilterPushdownFilter new_probe;
		new_probe.dynamic_filters = probe.dynamic_filters;
		new_probe.columns = probe.columns;
		copy->probe_info.push_back(std::move(new_probe));
	}
	copy->min_max_aggregates.reserve(info.min_max_aggregates.size());
	for (const auto &expr : info.min_max_aggregates) {
		if (expr) {
			copy->min_max_aggregates.push_back(expr->Copy());
		} else {
			copy->min_max_aggregates.push_back(nullptr);
		}
	}
	return copy;
}

duckdb::vector<unique_ptr<BaseStatistics>>
HashJoinNode::CopyJoinStats(const duckdb::vector<unique_ptr<BaseStatistics>> &stats) {
	duckdb::vector<unique_ptr<BaseStatistics>> copy;
	copy.reserve(stats.size());
	for (const auto &entry : stats) {
		if (entry) {
			copy.push_back(entry->ToUnique());
		} else {
			copy.push_back(nullptr);
		}
	}
	return copy;
}

SubmittableTask<WorkerTask> HashJoinNode::BuildHashJoinTask(SubmittableTask<WorkerTask> left_task,
                                                            SubmittableTask<WorkerTask> right_task,
                                                            TaskIDCounter &task_id_counter,
                                                            ClientContext *client_context) {
	auto left_plan_src = left_task.task()->plan();
	auto right_plan_src = right_task.task()->plan();
	if (!left_plan_src || !left_plan_src->HasRoot() || !right_plan_src || !right_plan_src->HasRoot()) {
		throw InvalidInputException("HashJoinNode cannot build join task from task without a physical plan root");
	}

	auto left_plan = ClonePhysicalPlanOrThrow(left_plan_src, "build_hash_join_task:left", client_context);
	auto &left_root = left_plan->Root();
	// A PhysicalOperator only stores references to its children. Clone the right
	// tree directly into left_plan so the composed join plan owns both sides.
	// Keeping the right tree in a temporary PhysicalPlan leaves a dangling child
	// as soon as this function returns and crashes later during serialization.
	auto &right_root =
	    ClonePhysicalPlanRootIntoPlanOrThrow(right_plan_src, *left_plan, "build_hash_join_task:right", client_context);

	LogicalComparisonJoin dummy_join(join_type_);
	dummy_join.types = output_types_;

	auto conditions = CopyConditions(conditions_);

	// Check if we have any equality conditions. PhysicalHashJoin requires at least
	// one equality condition (D_ASSERT(!equality_types.empty()) in JoinHashTable).
	// For pure non-equi joins (e.g. c_acctbal > avg(c_acctbal)), we must use
	// PhysicalNestedLoopJoin instead.
	bool has_equality = false;
	for (auto &cond : conditions) {
		if (cond.comparison == ExpressionType::COMPARE_EQUAL ||
		    cond.comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
			has_equality = true;
			break;
		}
	}

	if (!has_equality) {
		// Pure non-equi join: use PhysicalNestedLoopJoin (θ-join)
		auto &nlj = left_plan
		                ->Make<PhysicalNestedLoopJoin>(dummy_join, std::move(conditions), join_type_,
		                                               estimated_cardinality_, true)
		                .Cast<PhysicalNestedLoopJoin>();

		nlj.children.push_back(left_root);
		nlj.children.push_back(right_root);

		// Fix condition expression types to match actual child types
		const auto &left_types = left_root.GetTypes();
		const auto &right_types = right_root.GetTypes();
		FixHashJoinConditionTypes(nlj.conditions, left_types, right_types);

		// The translated output types are the schema contract for this node. Validate
		// that contract against the attached children, but do not replace it: doing so
		// would make the physical plan disagree with HashJoinNode::config().schema().
		auto child_derived_types = BuildJoinOutputTypes(join_type_, left_types, right_types);
		if (nlj.GetTypes() != child_derived_types) {
			throw InternalException(
			    "HashJoinNode translated a %s nested loop join with an output schema that does not match its children "
			    "(%llu translated columns, %llu child-derived columns)",
			    EnumUtil::ToString(join_type_), nlj.GetTypes().size(), child_derived_types.size());
		}

		left_plan->SetRoot(nlj);
	} else {
		auto mark_build_summary =
		    ExtractMarkJoinBuildSummary(right_task.task()->inputs(), mark_build_summary_source_node_id_);
		// Has equality conditions: use PhysicalHashJoin (normal path)
		auto &hash_join = left_plan
		                      ->Make<PhysicalHashJoin>(dummy_join, std::move(conditions), join_type_, delim_types_,
		                                               estimated_cardinality_, true)
		                      .Cast<PhysicalHashJoin>();

		hash_join.condition_types = condition_types_;
		hash_join.payload_columns = payload_columns_;
		hash_join.lhs_output_columns = lhs_output_columns_;
		hash_join.rhs_output_columns = rhs_output_columns_;
		hash_join.join_stats = CopyJoinStats(join_stats_);
		hash_join.mark_join_build_summary = mark_build_summary;
		if (filter_pushdown_) {
			hash_join.filter_pushdown = CopyFilterPushdownInfo(*filter_pushdown_);
		}

		hash_join.children.push_back(left_root);
		hash_join.children.push_back(right_root);
		left_plan->SetRoot(hash_join);

		RepairHashJoinMetadataAfterChildAttach(hash_join, left_root.GetTypes(), right_root.GetTypes());
	}

	TaskContext task_context = TaskContext::from_node_context(context_.query_idx(), node_id(), task_id_counter.next());
	auto merged_ctx = MergeTaskContext(left_task.task()->context(), right_task.task()->context());
	merged_ctx = MergeTaskContext(merged_ctx, context_.to_hashmap());
	WorkerTask new_task(task_context, left_plan, left_task.task()->config(), std::move(merged_ctx), "WorkerTask");
	auto &inputs = new_task.mutable_inputs();
	MoveTaskInputsOrThrow(inputs, left_task.task()->mutable_inputs(), context_.node_name());
	MoveTaskInputsOrThrow(inputs, right_task.task()->mutable_inputs(), context_.node_name());
	return std::move(left_task).with_new_task(std::move(new_task));
}

SubmittableTaskStream<WorkerTask> HashJoinNode::produce_tasks(PlanExecutionContext &plan_context) {
	if (!left_ || !right_) {
		return SubmittableTaskStream<WorkerTask>::from_receiver(Receiver<SubmittableTask<WorkerTask>>());
	}
	auto left_input = left_->produce_tasks(plan_context);
	auto right_input = right_->produce_tasks(plan_context);

	auto task_id_counter = std::make_shared<TaskIDCounter>(plan_context.task_id_counter());
	auto *client_context = plan_context.client_context();
	auto self_shared = shared_from_this();
	BinaryInitialTaskBuilder build_initial_task = [self_shared, task_id_counter,
	                                               client_context](BinarySubmittableTask left_task,
	                                                               BinarySubmittableTask right_task) mutable {
		return self_shared->BuildHashJoinTask(std::move(left_task), std::move(right_task), *task_id_counter,
		                                      client_context);
	};

	// The first event registers a physical join fragment containing both
	// exchange sources. Later events are dynamic-input updates for that same
	// fragment: their TaskInputs are keyed by source_node_id, so either side can
	// advance independently without a positional left/right zip.
	BinaryFragmentInputFanInStream stream(std::move(left_input), std::move(right_input), std::move(build_initial_task),
	                                      task_id_counter, context_.query_idx(), node_id(), context_.node_name());
	return SubmittableTaskStream<WorkerTask>(boxed<BinarySubmittableTask>(std::move(stream)));
}

} // namespace distributed
} // namespace duckdb
