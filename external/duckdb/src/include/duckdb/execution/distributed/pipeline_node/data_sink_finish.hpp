// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"

namespace duckdb {
namespace distributed {

static constexpr const char *DATA_SINK_NO_INTERNAL_RETRY_CONTEXT_KEY = "_vane_datasink_no_internal_retry";

class DataSinkFinishNode : public PipelineNodeImpl, public std::enable_shared_from_this<DataSinkFinishNode> {
public:
	DataSinkFinishNode(NodeID node_id, PipelineNodeRef child, string operation_id)
	    : context_(InheritPipelineNodeContext(child, node_id, "DataSinkFinish")), child_(std::move(child)),
	      operation_id_(std::move(operation_id)) {
		if (!child_) {
			throw InternalException("DataSinkFinish requires a child node");
		}
	}

	bool is_sink() const override {
		return true;
	}
	NodeID result_node_id() const {
		return node_id();
	}
	const PipelineNodeContext &context() const override {
		return context_;
	}
	const PipelineNodeConfig &config() const override {
		return child_->config();
	}
	std::vector<PipelineNodeRef> children() const override {
		return {child_};
	}
	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override;
	const string &operation_id() const {
		return operation_id_;
	}
	std::vector<std::string> multiline_display(bool) const override {
		return {"DataSinkFinish: " + operation_id_};
	}

private:
	PipelineNodeContext context_;
	PipelineNodeRef child_;
	string operation_id_;
};

} // namespace distributed
} // namespace duckdb
