// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"

namespace duckdb {
namespace distributed {

class EmptyResultSourceNode : public PipelineNodeImpl {
public:
	EmptyResultSourceNode(PipelineNodeContext context, DuckPhysicalPlanRef empty_plan, SchemaRef schema,
	                      DuckDBExecutionConfigRef execution_config);

	std::string name() const override {
		return "EmptyResultSource";
	}
	NodeID node_id() const override {
		return context_.node_id();
	}
	const PipelineNodeContext &context() const override {
		return context_;
	}
	const PipelineNodeConfig &config() const override {
		return config_;
	}

	std::vector<PipelineNodeRef> children() const override {
		return {};
	}
	bool is_statically_empty_result() const override {
		return true;
	}

	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override;
	std::vector<std::string> multiline_display(bool verbose) const override;

private:
	PipelineNodeContext context_;
	PipelineNodeConfig config_;
	DuckPhysicalPlanRef empty_plan_;
};

} // namespace distributed
} // namespace duckdb
