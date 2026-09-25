// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/function/distributed_write.hpp"

namespace duckdb {
namespace distributed {

//! Wraps each distributed input fragment with the generic callback-backed
//! physical write sink. The original extension root stays on the coordinator.
class ExtensionWriteSinkNode : public PipelineNodeImpl, public std::enable_shared_from_this<ExtensionWriteSinkNode> {
public:
	ExtensionWriteSinkNode(NodeID node_id, PipelineNodeRef child, DistributedExtensionWriteInfo info);

	string name() const override {
		return "ExtensionWriteSink";
	}
	bool is_sink() const override {
		return true;
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
	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override;
	std::vector<std::string> multiline_display(bool) const override;

	const DistributedExtensionWriteInfo &write_info() const {
		return info_;
	}

private:
	PipelineNodeContext ctx_;
	PipelineNodeConfig config_;
	PipelineNodeRef child_;
	DistributedExtensionWriteInfo info_;
};

} // namespace distributed
} // namespace duckdb
