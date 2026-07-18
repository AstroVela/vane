// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"

namespace duckdb {
namespace distributed {

class FilterNode : public PipelineNodeImpl, public std::enable_shared_from_this<FilterNode> {
public:
	FilterNode(NodeID node_id, PipelineNodeRef child, ExpressionRef predicate)
	    : ctx_(InheritPipelineNodeContext(child, node_id, "Filter")),
	      config_(child ? child->config() : PipelineNodeConfig()), child_(std::move(child)),
	      predicate_(std::move(predicate)) {
	}

	std::string name() const override {
		return "Filter";
	}
	NodeID node_id() const override {
		return ctx_.node_id();
	}
	const PipelineNodeContext &context() const override {
		return ctx_;
	}
	const PipelineNodeConfig &config() const override {
		return config_;
	}

	std::vector<PipelineNodeRef> children() const override {
		return {child_};
	}

	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override {
		// Get the input task stream from the child node
		auto input_stream = child_->produce_tasks(plan_context);

		// Capture predicate for the plan-builder closure
		auto predicate = predicate_;

		// Apply a pipeline instruction that would append a filter to the local physical plan.
		return input_stream.pipeline_instruction(
		    shared_from_this(),
		    [predicate](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
			    if (!predicate) {
				    return input_plan;
			    }
			    // Build select_list for PhysicalFilter by cloning the predicate
			    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> select_list;
			    select_list.push_back(duckdb::unique_ptr<duckdb::Expression>(predicate->Copy().release()));

			    // Derive output types and estimated cardinality from current root
			    auto types = input_plan->Root().GetTypes();
			    idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;

			    // Attach the existing root as the child of the new filter so the
			    // plan remains a valid tree. Capture the old root first.
			    auto &old_root = input_plan->Root();
			    // Append a PhysicalFilter to the plan and set it as the new root
			    auto &filter =
			        input_plan->Make<::duckdb::PhysicalFilter>(types, std::move(select_list), estimated_cardinality);
			    filter.children.push_back(old_root);
			    input_plan->SetRoot(filter);
			    return input_plan;
		    },
		    plan_context.client_context());
	}

	std::vector<std::string> multiline_display(bool verbose) const override {
		std::string pred_name = predicate_ ? predicate_->GetName() : std::string("<none>");
		return {std::string("Filter: ") + pred_name};
	}

private:
	PipelineNodeContext ctx_;
	PipelineNodeConfig config_;
	PipelineNodeRef child_;
	ExpressionRef predicate_;
};

} // namespace distributed
} // namespace duckdb
