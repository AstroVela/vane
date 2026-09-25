// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/join/nested_loop_join.hpp"

#include <algorithm>

#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/execution/distributed/pipeline_node/binary_task_fan_in.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/hash_join_metadata.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/join_output_types.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/execution/operator/join/physical_blockwise_nl_join.hpp"
#include "duckdb/execution/operator/join/physical_nested_loop_join.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/joinside.hpp"
#include "duckdb/planner/operator/logical_any_join.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"

namespace duckdb {
namespace distributed {

namespace {

PipelineNodeConfig MakeNestedLoopJoinConfig(const PlanConfig &plan_config, const SchemaRef &schema,
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

void OffsetBoundReferenceIndices(Expression &expression, idx_t offset) {
	if (expression.GetExpressionClass() == ExpressionClass::BOUND_REF) {
		expression.Cast<BoundReferenceExpression>().index += offset;
	}
	ExpressionIterator::EnumerateChildren(expression,
	                                      [&](Expression &child) { OffsetBoundReferenceIndices(child, offset); });
}

struct WorkerJoinProjection {
	duckdb::vector<idx_t> column_indices;
	duckdb::vector<LogicalType> types;
};

void AppendProjectedSide(WorkerJoinProjection &projection, const duckdb::vector<LogicalType> &child_types,
                         const duckdb::vector<idx_t> &projection_map, idx_t output_offset) {
	const auto count = projection_map.empty() ? child_types.size() : projection_map.size();
	projection.column_indices.reserve(projection.column_indices.size() + count);
	projection.types.reserve(projection.types.size() + count);
	for (idx_t index = 0; index < count; index++) {
		auto child_index = projection_map.empty() ? index : projection_map[index];
		if (child_index >= child_types.size()) {
			throw InvalidInputException("NestedLoopJoinNode projection column %llu is outside a %llu-column input",
			                            child_index, child_types.size());
		}
		projection.column_indices.push_back(output_offset + child_index);
		projection.types.push_back(child_types[child_index]);
	}
}

WorkerJoinProjection BuildWorkerJoinProjection(JoinType join_type, const duckdb::vector<LogicalType> &left_types,
                                               const duckdb::vector<LogicalType> &right_types,
                                               const duckdb::vector<idx_t> &left_projection_map,
                                               const duckdb::vector<idx_t> &right_projection_map) {
	WorkerJoinProjection projection;
	idx_t output_offset = 0;
	if (JoinOutputsLeft(join_type)) {
		AppendProjectedSide(projection, left_types, left_projection_map, output_offset);
		output_offset += left_types.size();
	}
	if (JoinOutputsRight(join_type)) {
		AppendProjectedSide(projection, right_types, right_projection_map, output_offset);
		output_offset += right_types.size();
	}
	if (join_type == JoinType::MARK) {
		projection.column_indices.push_back(output_offset);
		projection.types.push_back(LogicalType::BOOLEAN);
	}
	return projection;
}

bool NeedsProjection(const WorkerJoinProjection &projection, idx_t full_output_count) {
	if (projection.column_indices.size() != full_output_count) {
		return true;
	}
	for (idx_t index = 0; index < projection.column_indices.size(); index++) {
		if (projection.column_indices[index] != index) {
			return true;
		}
	}
	return false;
}

} // namespace

NestedLoopJoinNode::NestedLoopJoinNode(NodeID node_id, const PlanConfig &plan_config, PhysicalOperatorType source_type,
                                       duckdb::vector<JoinCondition> conditions, unique_ptr<Expression> predicate,
                                       JoinType join_type, duckdb::vector<LogicalType> output_types,
                                       idx_t estimated_cardinality, std::shared_ptr<DistributedPipelineNode> left,
                                       std::shared_ptr<DistributedPipelineNode> right, SchemaRef schema,
                                       duckdb::vector<idx_t> left_projection_map,
                                       duckdb::vector<idx_t> right_projection_map)
    : config_(MakeNestedLoopJoinConfig(plan_config, schema, left, right)),
      context_(plan_config.query_idx, plan_config.query_id, node_id, "NestedLoopJoin"), source_type_(source_type),
      conditions_(std::move(conditions)), predicate_(std::move(predicate)), join_type_(join_type),
      output_types_(std::move(output_types)), left_projection_map_(std::move(left_projection_map)),
      right_projection_map_(std::move(right_projection_map)), estimated_cardinality_(estimated_cardinality),
      left_(std::move(left)), right_(std::move(right)) {
	if (source_type_ != PhysicalOperatorType::NESTED_LOOP_JOIN &&
	    source_type_ != PhysicalOperatorType::PIECEWISE_MERGE_JOIN && source_type_ != PhysicalOperatorType::IE_JOIN) {
		throw InvalidInputException("NestedLoopJoinNode cannot normalize operator type %s",
		                            EnumUtil::ToString(source_type_));
	}
}

NestedLoopJoinNode::NestedLoopJoinNode(NodeID node_id, const PlanConfig &plan_config, unique_ptr<Expression> condition,
                                       JoinType join_type, duckdb::vector<LogicalType> output_types,
                                       idx_t estimated_cardinality, std::shared_ptr<DistributedPipelineNode> left,
                                       std::shared_ptr<DistributedPipelineNode> right, SchemaRef schema)
    : config_(MakeNestedLoopJoinConfig(plan_config, schema, left, right)),
      context_(plan_config.query_idx, plan_config.query_id, node_id, "BlockwiseNLJoin"),
      source_type_(PhysicalOperatorType::BLOCKWISE_NL_JOIN), arbitrary_condition_(std::move(condition)),
      join_type_(join_type), output_types_(std::move(output_types)), estimated_cardinality_(estimated_cardinality),
      left_(std::move(left)), right_(std::move(right)) {
	if (!arbitrary_condition_) {
		throw InvalidInputException("NestedLoopJoinNode requires a blockwise join condition");
	}
}

std::vector<PipelineNodeRef> NestedLoopJoinNode::children() const {
	std::vector<PipelineNodeRef> result;
	if (left_) {
		result.push_back(left_->inner());
	}
	if (right_) {
		result.push_back(right_->inner());
	}
	return result;
}

std::vector<std::string> NestedLoopJoinNode::multiline_display(bool /*verbose*/) const {
	return {"Nested Loop Join", "Source operator: " + EnumUtil::ToString(source_type_),
	        "Join type: " + EnumUtil::ToString(join_type_)};
}

duckdb::vector<JoinCondition> NestedLoopJoinNode::CopyConditions(const duckdb::vector<JoinCondition> &conditions) {
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

SubmittableTask<WorkerTask> NestedLoopJoinNode::BuildNestedLoopJoinTask(SubmittableTask<WorkerTask> left_task,
                                                                        SubmittableTask<WorkerTask> right_task,
                                                                        TaskIDCounter &task_id_counter,
                                                                        ClientContext *client_context) {
	auto left_plan_src = left_task.task()->plan();
	auto right_plan_src = right_task.task()->plan();
	if (!left_plan_src || !left_plan_src->HasRoot() || !right_plan_src || !right_plan_src->HasRoot()) {
		throw InvalidInputException("NestedLoopJoinNode cannot build task from input without a physical plan root");
	}

	auto plan = ClonePhysicalPlanOrThrow(left_plan_src, "build_nested_loop_join_task:left", client_context);
	auto &left_root = plan->Root();
	auto &right_root = ClonePhysicalPlanRootIntoPlanOrThrow(right_plan_src, *plan, "build_nested_loop_join_task:right",
	                                                        client_context);
	auto full_output_types = BuildJoinOutputTypes(join_type_, left_root.GetTypes(), right_root.GetTypes());

	if (source_type_ == PhysicalOperatorType::BLOCKWISE_NL_JOIN) {
		LogicalAnyJoin dummy_join(join_type_);
		dummy_join.types = full_output_types;
		auto &join = plan->Make<PhysicalBlockwiseNLJoin>(
		    dummy_join, left_root, right_root, arbitrary_condition_->Copy(), join_type_, estimated_cardinality_);
		plan->SetRoot(join);
	} else {
		auto conditions = CopyConditions(conditions_);
		FixHashJoinConditionTypes(conditions, left_root.GetTypes(), right_root.GetTypes());
		if (PhysicalNestedLoopJoin::IsSupported(conditions, join_type_)) {
			LogicalComparisonJoin dummy_join(join_type_);
			dummy_join.types = full_output_types;
			dummy_join.predicate = predicate_ ? predicate_->Copy() : nullptr;
			auto &join = plan->Make<PhysicalNestedLoopJoin>(dummy_join, left_root, right_root, std::move(conditions),
			                                                join_type_, estimated_cardinality_);
			plan->SetRoot(join);
		} else {
			if (predicate_) {
				throw InternalException("NestedLoopJoinNode cannot combine an unsupported comparison type with a "
				                        "separate join predicate");
			}
			for (auto &condition : conditions) {
				if (condition.right) {
					OffsetBoundReferenceIndices(*condition.right, left_root.GetTypes().size());
				}
			}
			auto blockwise_condition = JoinCondition::CreateExpression(std::move(conditions));
			LogicalAnyJoin dummy_join(join_type_);
			dummy_join.types = full_output_types;
			auto &join = plan->Make<PhysicalBlockwiseNLJoin>(
			    dummy_join, left_root, right_root, std::move(blockwise_condition), join_type_, estimated_cardinality_);
			plan->SetRoot(join);
		}
	}

	auto projection = BuildWorkerJoinProjection(join_type_, left_root.GetTypes(), right_root.GetTypes(),
	                                            left_projection_map_, right_projection_map_);
	if (projection.types != output_types_) {
		throw InternalException(
		    "NestedLoopJoinNode translated a %s join with an output schema that does not match its projection "
		    "(%llu translated columns, %llu projected columns)",
		    EnumUtil::ToString(join_type_), output_types_.size(), projection.types.size());
	}
	if (NeedsProjection(projection, full_output_types.size())) {
		duckdb::vector<unique_ptr<Expression>> select_list;
		select_list.reserve(projection.column_indices.size());
		for (idx_t index = 0; index < projection.column_indices.size(); index++) {
			select_list.push_back(
			    make_uniq<BoundReferenceExpression>(projection.types[index], projection.column_indices[index]));
		}
		auto &join_root = plan->Root();
		auto &project = plan->Make<PhysicalProjection>(output_types_, std::move(select_list), estimated_cardinality_);
		project.children.push_back(join_root);
		plan->SetRoot(project);
	}
	if (plan->Root().GetTypes() != output_types_) {
		throw InternalException("NestedLoopJoinNode worker plan output does not match the translated schema");
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

SubmittableTaskStream<WorkerTask> NestedLoopJoinNode::produce_tasks(PlanExecutionContext &plan_context) {
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
		return self_shared->BuildNestedLoopJoinTask(std::move(left_task), std::move(right_task), *task_id_counter,
		                                            client_context);
	};

	BinaryFragmentInputFanInStream stream(std::move(left_input), std::move(right_input), std::move(build_initial_task),
	                                      task_id_counter, context_.query_idx(), node_id(), context_.node_name());
	return SubmittableTaskStream<WorkerTask>(boxed<BinarySubmittableTask>(std::move(stream)));
}

} // namespace distributed
} // namespace duckdb
