// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/pivot.hpp"

#include "duckdb/execution/operator/projection/physical_pivot.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"

namespace duckdb {
namespace distributed {

static BoundPivotInfo CopyPivotInfoForPivotNode(const BoundPivotInfo &info) {
	BoundPivotInfo copy;
	copy.group_count = info.group_count;
	copy.types = info.types;
	copy.pivot_values = info.pivot_values;
	copy.aggregates.reserve(info.aggregates.size());
	for (const auto &expr : info.aggregates) {
		if (expr) {
			copy.aggregates.push_back(expr->Copy());
		} else {
			copy.aggregates.push_back(nullptr);
		}
	}
	return copy;
}

PivotNode::PivotNode(NodeID node_id, PipelineNodeRef child, BoundPivotInfo bound_pivot,
                     std::vector<LogicalType> output_types)
    : ctx_(InheritPipelineNodeContext(child, node_id, "Pivot")),
      config_(child ? child->config() : PipelineNodeConfig()), child_(std::move(child)),
      bound_pivot_(std::move(bound_pivot)), output_types_(std::move(output_types)) {
}

std::vector<PipelineNodeRef> PivotNode::children() const {
	return {child_};
}

SubmittableTaskStream<WorkerTask> PivotNode::produce_tasks(PlanExecutionContext &plan_context) {
	auto input_stream = child_->produce_tasks(plan_context);
	const auto *bound_pivot_ptr = &bound_pivot_;
	const auto *output_types_ptr = &output_types_;
	return input_stream.pipeline_instruction(
	    shared_from_this(),
	    [bound_pivot_ptr, output_types_ptr](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
		    auto bound_pivot = CopyPivotInfoForPivotNode(*bound_pivot_ptr);
		    auto types = *output_types_ptr;
		    idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;

		    auto &old_root = input_plan->Root();
		    auto &pivot_op = input_plan->Make<duckdb::PhysicalPivot>(std::move(types), std::move(bound_pivot),
		                                                             estimated_cardinality);
		    pivot_op.children.push_back(old_root);
		    input_plan->SetRoot(pivot_op);
		    return input_plan;
	    },
	    plan_context.client_context());
}

std::vector<std::string> PivotNode::multiline_display(bool /*verbose*/) const {
	return {std::string("Pivot")};
}

} // namespace distributed
} // namespace duckdb
