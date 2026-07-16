// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/expression_scan.hpp"

#include "duckdb/execution/operator/scan/physical_expression_scan.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"

namespace duckdb {
namespace distributed {

static duckdb::vector<duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>>
CopyExpressionScanExpressions(const std::vector<std::vector<ExpressionRef>> &expressions) {
	duckdb::vector<duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>> copies;
	copies.reserve(expressions.size());
	for (const auto &expr_list : expressions) {
		duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> list;
		list.reserve(expr_list.size());
		for (const auto &expr_ref : expr_list) {
			if (expr_ref) {
				auto copy = expr_ref->Copy();
				list.push_back(duckdb::unique_ptr<duckdb::Expression>(copy.release()));
			} else {
				list.push_back(nullptr);
			}
		}
		copies.push_back(std::move(list));
	}
	return copies;
}

ExpressionScanNode::ExpressionScanNode(NodeID node_id, PipelineNodeRef child,
                                       std::vector<std::vector<ExpressionRef>> expressions,
                                       std::vector<LogicalType> output_types)
    : ctx_(InheritPipelineNodeContext(child, node_id, "ExpressionScan")),
      config_(child ? child->config() : PipelineNodeConfig()), child_(std::move(child)),
      expressions_(std::move(expressions)), output_types_(std::move(output_types)) {
}

std::vector<PipelineNodeRef> ExpressionScanNode::children() const {
	return {child_};
}

SubmittableTaskStream<WorkerTask> ExpressionScanNode::produce_tasks(PlanExecutionContext &plan_context) {
	auto input_stream = child_->produce_tasks(plan_context);
	const auto *expressions_ptr = &expressions_;
	const auto *output_types_ptr = &output_types_;
	return input_stream.pipeline_instruction(
	    shared_from_this(),
	    [expressions_ptr, output_types_ptr](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
		    if (expressions_ptr->empty()) {
			    return input_plan;
		    }
		    auto expressions = CopyExpressionScanExpressions(*expressions_ptr);
		    auto types = *output_types_ptr;
		    idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;

		    auto &old_root = input_plan->Root();
		    auto &expr_scan = input_plan->Make<duckdb::PhysicalExpressionScan>(std::move(types), std::move(expressions),
		                                                                       estimated_cardinality);
		    expr_scan.children.push_back(old_root);
		    input_plan->SetRoot(expr_scan);
		    return input_plan;
	    },
	    plan_context.client_context());
}

std::vector<std::string> ExpressionScanNode::multiline_display(bool /*verbose*/) const {
	return {std::string("ExpressionScan: ") + std::to_string(expressions_.size()) + " rows"};
}

} // namespace distributed
} // namespace duckdb
