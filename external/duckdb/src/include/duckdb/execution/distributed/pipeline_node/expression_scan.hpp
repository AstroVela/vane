// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"

namespace duckdb {
namespace distributed {

class ExpressionScanNode : public PipelineNodeImpl, public std::enable_shared_from_this<ExpressionScanNode> {
public:
	ExpressionScanNode(NodeID node_id, PipelineNodeRef child, std::vector<std::vector<ExpressionRef>> expressions,
	                   std::vector<LogicalType> output_types);

	std::string name() const override {
		return "ExpressionScan";
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

	std::vector<PipelineNodeRef> children() const override;
	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override;
	std::vector<std::string> multiline_display(bool verbose) const override;

private:
	PipelineNodeContext ctx_;
	PipelineNodeConfig config_;
	PipelineNodeRef child_;
	std::vector<std::vector<ExpressionRef>> expressions_;
	std::vector<LogicalType> output_types_;
};

} // namespace distributed
} // namespace duckdb
