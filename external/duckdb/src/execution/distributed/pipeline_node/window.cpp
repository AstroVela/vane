// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/window.hpp"

#include "duckdb/execution/operator/aggregate/physical_streaming_window.hpp"
#include "duckdb/execution/operator/aggregate/physical_window.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"

namespace duckdb {
namespace distributed {

static duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>
CopyWindowSelectList(const std::vector<ExpressionRef> &select_list) {
	duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> copies;
	copies.reserve(select_list.size());
	for (const auto &expr_ref : select_list) {
		if (!expr_ref) {
			continue;
		}
		auto copy = expr_ref->Copy();
		copies.push_back(duckdb::unique_ptr<duckdb::Expression>(copy.release()));
	}
	return copies;
}

WindowNode::WindowNode(NodeID node_id, PipelineNodeRef child, std::vector<ExpressionRef> select_list,
                       std::vector<LogicalType> output_types)
    : ctx_(InheritPipelineNodeContext(child, node_id, "Window")),
      config_(child ? child->config() : PipelineNodeConfig()), child_(std::move(child)),
      select_list_(std::move(select_list)), output_types_(std::move(output_types)) {
}

std::vector<PipelineNodeRef> WindowNode::children() const {
	return {child_};
}

SubmittableTaskStream<WorkerTask> WindowNode::produce_tasks(PlanExecutionContext &plan_context) {
	auto input_stream = child_->produce_tasks(plan_context);
	const auto *select_list_ptr = &select_list_;
	const auto *output_types_ptr = &output_types_;
	return input_stream.pipeline_instruction(
	    shared_from_this(),
	    [select_list_ptr, output_types_ptr](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
		    if (select_list_ptr->empty()) {
			    return input_plan;
		    }
		    auto select_list = CopyWindowSelectList(*select_list_ptr);
		    auto types = *output_types_ptr;
		    idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;

		    auto &old_root = input_plan->Root();
		    auto &window_op = input_plan->Make<duckdb::PhysicalWindow>(std::move(types), std::move(select_list),
		                                                               estimated_cardinality);
		    window_op.children.push_back(old_root);
		    input_plan->SetRoot(window_op);
		    return input_plan;
	    },
	    plan_context.client_context());
}

std::vector<std::string> WindowNode::multiline_display(bool /*verbose*/) const {
	std::string exprs;
	for (size_t i = 0; i < select_list_.size(); ++i) {
		if (i > 0) {
			exprs += ", ";
		}
		exprs += select_list_[i] ? select_list_[i]->GetName() : std::string("<none>");
	}
	return {std::string("Window: ") + exprs};
}

StreamingWindowNode::StreamingWindowNode(NodeID node_id, PipelineNodeRef child, std::vector<ExpressionRef> select_list,
                                         std::vector<LogicalType> output_types)
    : ctx_(InheritPipelineNodeContext(child, node_id, "StreamingWindow")),
      config_(child ? child->config() : PipelineNodeConfig()), child_(std::move(child)),
      select_list_(std::move(select_list)), output_types_(std::move(output_types)) {
}

std::vector<PipelineNodeRef> StreamingWindowNode::children() const {
	return {child_};
}

SubmittableTaskStream<WorkerTask> StreamingWindowNode::produce_tasks(PlanExecutionContext &plan_context) {
	auto input_stream = child_->produce_tasks(plan_context);
	const auto *select_list_ptr = &select_list_;
	const auto *output_types_ptr = &output_types_;
	return input_stream.pipeline_instruction(
	    shared_from_this(),
	    [select_list_ptr, output_types_ptr](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
		    if (select_list_ptr->empty()) {
			    return input_plan;
		    }
		    auto select_list = CopyWindowSelectList(*select_list_ptr);
		    auto types = *output_types_ptr;
		    idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;

		    auto &old_root = input_plan->Root();
		    auto &window_op = input_plan->Make<duckdb::PhysicalStreamingWindow>(
		        std::move(types), std::move(select_list), estimated_cardinality);
		    window_op.children.push_back(old_root);
		    input_plan->SetRoot(window_op);
		    return input_plan;
	    },
	    plan_context.client_context());
}

std::vector<std::string> StreamingWindowNode::multiline_display(bool /*verbose*/) const {
	std::string exprs;
	for (size_t i = 0; i < select_list_.size(); ++i) {
		if (i > 0) {
			exprs += ", ";
		}
		exprs += select_list_[i] ? select_list_[i]->GetName() : std::string("<none>");
	}
	return {std::string("StreamingWindow: ") + exprs};
}

} // namespace distributed
} // namespace duckdb
