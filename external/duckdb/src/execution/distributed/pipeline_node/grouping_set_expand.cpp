// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/grouping_set_expand.hpp"

#include "duckdb/common/atomic.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/execution/operator/projection/physical_grouping_set_expand.hpp"

namespace duckdb {
namespace distributed {

GroupingSetExpandNode::GroupingSetExpandNode(NodeID node_id, PipelineNodeRef child, std::vector<BoundExpr> groups,
                                             std::vector<GroupingSet> grouping_sets,
                                             std::vector<std::vector<idx_t>> grouping_functions,
                                             std::vector<idx_t> aggregate_filter_indexes,
                                             std::vector<LogicalType> output_types)
    : ctx_(InheritPipelineNodeContext(child, node_id, "GroupingSetExpand")),
      config_(
          MakeSchemaRef(output_types), child ? child->config().execution_config() : DuckDBExecutionConfigRef(),
          ClusteringSpec::unknown_with_num_partitions(child ? child->config().clustering_spec()->num_partitions() : 1)),
      child_(std::move(child)), groups_(std::move(groups)), grouping_sets_(std::move(grouping_sets)),
      grouping_functions_(std::move(grouping_functions)),
      aggregate_filter_indexes_(std::move(aggregate_filter_indexes)), output_types_(std::move(output_types)),
      has_empty_grouping_set_(false) {
	for (const auto &grouping_set : grouping_sets_) {
		if (grouping_set.empty()) {
			has_empty_grouping_set_ = true;
			break;
		}
	}
}

std::vector<PipelineNodeRef> GroupingSetExpandNode::children() const {
	return {child_};
}

SubmittableTaskStream<WorkerTask> GroupingSetExpandNode::produce_tasks(PlanExecutionContext &plan_context) {
	auto input_stream = child_->produce_tasks(plan_context);
	auto self = shared_from_this();
	auto seed_plan_assigned = std::make_shared<atomic<bool>>(false);
	return input_stream.pipeline_instruction(
	    self,
	    [self, seed_plan_assigned](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
		    vector<unique_ptr<Expression>> groups;
		    groups.reserve(self->groups_.size());
		    for (const auto &group : self->groups_) {
			    groups.push_back(group->Copy());
		    }

		    vector<GroupingSet> grouping_sets(self->grouping_sets_.begin(), self->grouping_sets_.end());
		    vector<vector<idx_t>> grouping_functions;
		    grouping_functions.reserve(self->grouping_functions_.size());
		    for (const auto &grouping_function : self->grouping_functions_) {
			    grouping_functions.emplace_back(grouping_function.begin(), grouping_function.end());
		    }
		    vector<idx_t> filter_indexes(self->aggregate_filter_indexes_.begin(),
		                                 self->aggregate_filter_indexes_.end());
		    vector<LogicalType> output_types(self->output_types_.begin(), self->output_types_.end());
		    const auto input_column_count = input_plan->Root().GetTypes().size();
		    // One task is sufficient: seed rows hash to the same keys as real
		    // empty-set rows, and retries preserve this flag in the task plan.
		    const bool emit_empty_grouping_sets = self->has_empty_grouping_set_ && !seed_plan_assigned->exchange(true);
		    const auto estimated_cardinality = input_plan->Root().estimated_cardinality * self->grouping_sets_.size();

		    auto &old_root = input_plan->Root();
		    auto &expand = input_plan->Make<PhysicalGroupingSetExpand>(
		        std::move(output_types), std::move(groups), std::move(grouping_sets), std::move(grouping_functions),
		        std::move(filter_indexes), input_column_count, emit_empty_grouping_sets, estimated_cardinality);
		    expand.children.push_back(old_root);
		    input_plan->SetRoot(expand);
		    return input_plan;
	    },
	    plan_context.client_context());
}

std::vector<std::string> GroupingSetExpandNode::multiline_display(bool /*verbose*/) const {
	return {"GroupingSetExpand: " + std::to_string(grouping_sets_.size()) + " sets"};
}

} // namespace distributed
} // namespace duckdb
