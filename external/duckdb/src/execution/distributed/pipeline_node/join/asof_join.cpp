// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/join/asof_join.hpp"

#include <algorithm>

#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/execution/distributed/pipeline_node/binary_task_fan_in.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/hash_join_metadata.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/join_output_types.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/execution/operator/join/physical_asof_join.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"

namespace duckdb {
namespace distributed {

namespace {

PipelineNodeConfig MakeAsOfJoinConfig(const PlanConfig &plan_config, const SchemaRef &schema,
                                      const std::shared_ptr<DistributedPipelineNode> &left,
                                      const std::shared_ptr<DistributedPipelineNode> &right) {
	size_t num_partitions = 1;
	if (left && right) {
		num_partitions = std::max(left->config().clustering_spec()->num_partitions(),
		                          right->config().clustering_spec()->num_partitions());
	} else if (left) {
		num_partitions = left->config().clustering_spec()->num_partitions();
	} else if (right) {
		num_partitions = right->config().clustering_spec()->num_partitions();
	}
	auto clustering = ClusteringSpec::unknown_with_num_partitions(num_partitions);
	return PipelineNodeConfig(schema, plan_config.config, std::move(clustering));
}

duckdb::vector<LogicalType> BuildAsOfOutputTypes(JoinType join_type, const duckdb::vector<LogicalType> &left_types,
                                                 const duckdb::vector<LogicalType> &right_types,
                                                 const duckdb::vector<column_t> &right_projection_map) {
	duckdb::vector<LogicalType> result;
	if (JoinOutputsLeft(join_type)) {
		result.insert(result.end(), left_types.begin(), left_types.end());
	}
	if (JoinOutputsRight(join_type)) {
		if (right_projection_map.empty()) {
			result.insert(result.end(), right_types.begin(), right_types.end());
		} else {
			for (auto column_index : right_projection_map) {
				if (column_index >= right_types.size()) {
					throw InvalidInputException(
					    "AsOfJoinNode right projection column %llu is outside a %llu-column input", column_index,
					    right_types.size());
				}
				result.push_back(right_types[column_index]);
			}
		}
	}
	if (join_type == JoinType::MARK) {
		result.push_back(LogicalType::BOOLEAN);
	}
	return result;
}

} // namespace

AsOfJoinNode::AsOfJoinNode(NodeID node_id, const PlanConfig &plan_config, duckdb::vector<JoinCondition> conditions,
                           JoinType join_type, duckdb::vector<LogicalType> output_types,
                           duckdb::vector<column_t> right_projection_map, idx_t estimated_cardinality,
                           std::shared_ptr<DistributedPipelineNode> left,
                           std::shared_ptr<DistributedPipelineNode> right, SchemaRef schema)
    : config_(MakeAsOfJoinConfig(plan_config, schema, left, right)),
      context_(plan_config.query_idx, plan_config.query_id, node_id, "AsOfJoin"), conditions_(std::move(conditions)),
      join_type_(join_type), output_types_(std::move(output_types)),
      right_projection_map_(std::move(right_projection_map)), estimated_cardinality_(estimated_cardinality),
      left_(std::move(left)), right_(std::move(right)) {
	if (conditions_.empty()) {
		throw InvalidInputException("AsOfJoinNode requires at least one join condition");
	}
}

std::vector<PipelineNodeRef> AsOfJoinNode::children() const {
	std::vector<PipelineNodeRef> result;
	if (left_) {
		result.push_back(left_->inner());
	}
	if (right_) {
		result.push_back(right_->inner());
	}
	return result;
}

std::vector<std::string> AsOfJoinNode::multiline_display(bool /*verbose*/) const {
	return {"ASOF Join", "Join type: " + EnumUtil::ToString(join_type_)};
}

duckdb::vector<JoinCondition> AsOfJoinNode::CopyConditions(const duckdb::vector<JoinCondition> &conditions) {
	duckdb::vector<JoinCondition> result;
	result.reserve(conditions.size());
	for (const auto &condition : conditions) {
		JoinCondition copy;
		copy.comparison = condition.comparison;
		if (condition.left) {
			copy.left = condition.left->Copy();
		}
		if (condition.right) {
			copy.right = condition.right->Copy();
		}
		result.push_back(std::move(copy));
	}
	return result;
}

SubmittableTask<WorkerTask> AsOfJoinNode::BuildAsOfJoinTask(SubmittableTask<WorkerTask> left_task,
                                                            SubmittableTask<WorkerTask> right_task,
                                                            TaskIDCounter &task_id_counter,
                                                            ClientContext *client_context) {
	auto left_plan_src = left_task.task()->plan();
	auto right_plan_src = right_task.task()->plan();
	if (!left_plan_src || !left_plan_src->HasRoot() || !right_plan_src || !right_plan_src->HasRoot()) {
		throw InvalidInputException("AsOfJoinNode cannot build task from input without a physical plan root");
	}

	auto plan = ClonePhysicalPlanOrThrow(left_plan_src, "build_asof_join_task:left", client_context);
	auto &left_root = plan->Root();
	auto &right_root =
	    ClonePhysicalPlanRootIntoPlanOrThrow(right_plan_src, *plan, "build_asof_join_task:right", client_context);

	auto expected_types =
	    BuildAsOfOutputTypes(join_type_, left_root.GetTypes(), right_root.GetTypes(), right_projection_map_);
	if (expected_types != output_types_) {
		throw InternalException(
		    "AsOfJoinNode output schema does not match its children and right projection (%llu translated columns, "
		    "%llu child-derived columns)",
		    output_types_.size(), expected_types.size());
	}

	auto conditions = CopyConditions(conditions_);
	FixHashJoinConditionTypes(conditions, left_root.GetTypes(), right_root.GetTypes());
	LogicalComparisonJoin dummy_join(join_type_);
	dummy_join.types = output_types_;
	dummy_join.conditions = std::move(conditions);
	dummy_join.right_projection_map = right_projection_map_;
	dummy_join.estimated_cardinality = estimated_cardinality_;
	auto &join = plan->Make<PhysicalAsOfJoin>(dummy_join, left_root, right_root);
	plan->SetRoot(join);
	if (plan->Root().GetTypes() != output_types_) {
		throw InternalException("AsOfJoinNode worker plan output does not match the translated schema");
	}

	TaskContext task_context = TaskContext::from_node_context(context_.query_idx(), node_id(), task_id_counter.next());
	auto merged_context = MergeTaskContext(left_task.task()->context(), right_task.task()->context());
	merged_context = MergeTaskContext(merged_context, context_.to_hashmap());
	WorkerTask new_task(task_context, plan, left_task.task()->config(), std::move(merged_context), "WorkerTask");
	auto &inputs = new_task.mutable_inputs();
	MoveTaskInputsOrThrow(inputs, left_task.task()->mutable_inputs(), context_.node_name());
	MoveTaskInputsOrThrow(inputs, right_task.task()->mutable_inputs(), context_.node_name());
	return std::move(left_task).with_new_task(std::move(new_task));
}

SubmittableTaskStream<WorkerTask> AsOfJoinNode::produce_tasks(PlanExecutionContext &plan_context) {
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
		return self_shared->BuildAsOfJoinTask(std::move(left_task), std::move(right_task), *task_id_counter,
		                                      client_context);
	};

	BinaryFragmentInputFanInStream stream(std::move(left_input), std::move(right_input), std::move(build_initial_task),
	                                      task_id_counter, context_.query_idx(), node_id(), context_.node_name());
	return SubmittableTaskStream<WorkerTask>(boxed<BinarySubmittableTask>(std::move(stream)));
}

} // namespace distributed
} // namespace duckdb
