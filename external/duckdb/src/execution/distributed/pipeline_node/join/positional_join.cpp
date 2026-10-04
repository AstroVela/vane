// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/join/positional_join.hpp"

#include <algorithm>

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/distributed/pipeline_node/binary_task_fan_in.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/execution/operator/join/physical_positional_join.hpp"

namespace duckdb {
namespace distributed {

PositionalJoinNode::PositionalJoinNode(NodeID node_id, const PlanConfig &plan_config,
                                       duckdb::vector<LogicalType> output_types, idx_t estimated_cardinality,
                                       std::shared_ptr<DistributedPipelineNode> left,
                                       std::shared_ptr<DistributedPipelineNode> right, SchemaRef schema)
    : context_(plan_config.query_idx, plan_config.query_id, node_id, "PositionalJoin"),
      output_types_(std::move(output_types)), estimated_cardinality_(estimated_cardinality), left_(std::move(left)),
      right_(std::move(right)) {
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

std::vector<PipelineNodeRef> PositionalJoinNode::children() const {
	std::vector<PipelineNodeRef> result;
	if (left_) {
		result.push_back(left_->inner());
	}
	if (right_) {
		result.push_back(right_->inner());
	}
	return result;
}

std::vector<std::string> PositionalJoinNode::multiline_display(bool /*verbose*/) const {
	return {"Positional Join"};
}

SubmittableTask<WorkerTask> PositionalJoinNode::BuildPositionalJoinTask(SubmittableTask<WorkerTask> left_task,
                                                                        SubmittableTask<WorkerTask> right_task,
                                                                        TaskIDCounter &task_id_counter,
                                                                        ClientContext *client_context) {
	auto left_plan_src = left_task.task()->plan();
	auto right_plan_src = right_task.task()->plan();
	if (!left_plan_src || !left_plan_src->HasRoot() || !right_plan_src || !right_plan_src->HasRoot()) {
		throw InvalidInputException("PositionalJoinNode cannot build task from input without a physical plan root");
	}

	auto plan = ClonePhysicalPlanOrThrow(left_plan_src, "build_positional_join_task:left", client_context);
	auto &left_root = plan->Root();
	auto &right_root =
	    ClonePhysicalPlanRootIntoPlanOrThrow(right_plan_src, *plan, "build_positional_join_task:right", client_context);

	duckdb::vector<LogicalType> child_derived_types = left_root.GetTypes();
	const auto &right_types = right_root.GetTypes();
	child_derived_types.insert(child_derived_types.end(), right_types.begin(), right_types.end());
	if (child_derived_types != output_types_) {
		throw InternalException(
		    "PositionalJoinNode output schema does not match its children (%llu translated columns, %llu "
		    "child-derived columns)",
		    output_types_.size(), child_derived_types.size());
	}

	auto &positional_join =
	    plan->Make<PhysicalPositionalJoin>(output_types_, left_root, right_root, estimated_cardinality_);
	plan->SetRoot(positional_join);

	TaskContext task_context = TaskContext::from_node_context(context_.query_idx(), node_id(), task_id_counter.next());
	auto merged_context = MergeTaskContext(left_task.task()->context(), right_task.task()->context());
	merged_context = MergeTaskContext(merged_context, context_.to_hashmap());
	WorkerTask new_task(task_context, plan, left_task.task()->config(), std::move(merged_context), "WorkerTask");
	auto &inputs = new_task.mutable_inputs();
	MoveTaskInputsOrThrow(inputs, left_task.task()->mutable_inputs(), context_.node_name());
	MoveTaskInputsOrThrow(inputs, right_task.task()->mutable_inputs(), context_.node_name());
	return std::move(left_task).with_new_task(std::move(new_task));
}

SubmittableTaskStream<WorkerTask> PositionalJoinNode::produce_tasks(PlanExecutionContext &plan_context) {
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
		return self_shared->BuildPositionalJoinTask(std::move(left_task), std::move(right_task), *task_id_counter,
		                                            client_context);
	};

	BinaryFragmentInputFanInStream stream(std::move(left_input), std::move(right_input), std::move(build_initial_task),
	                                      task_id_counter, context_.query_idx(), node_id(), context_.node_name());
	return SubmittableTaskStream<WorkerTask>(boxed<BinarySubmittableTask>(std::move(stream)));
}

} // namespace distributed
} // namespace duckdb
