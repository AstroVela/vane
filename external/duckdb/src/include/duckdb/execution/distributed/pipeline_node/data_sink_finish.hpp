// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/pipeline_node/finalizable_sink.hpp"

namespace duckdb {
namespace distributed {

class DataSinkFinishNode : public FinalizableSinkNode {
public:
	DataSinkFinishNode(NodeID node_id, PipelineNodeRef child, std::string operation_id)
	    : context_(InheritPipelineNodeContext(child, node_id, "DataSinkFinish")), child_(std::move(child)),
	      operation_id_(std::move(operation_id)) {
		if (!child_) {
			throw InternalException("DataSinkFinish requires a child node");
		}
	}

	std::string name() const override {
		return "DataSinkFinish";
	}
	NodeID node_id() const override {
		return context_.node_id();
	}
	NodeID result_node_id() const override {
		return child_->node_id();
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
	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override {
		return child_->produce_tasks(plan_context);
	}
	DuckDBResult<FinalizedSinkResult> finalize(const std::vector<ResultPartitionRef> &partitions,
	                                           ClientContext &context) override {
		(void)context;
		auto parsed = ParseDataSinkPartitions(operation_id_, partitions);
		if (parsed.is_err()) {
			return DuckDBResult<FinalizedSinkResult>::err(parsed.error());
		}
		return DuckDBResult<FinalizedSinkResult>::ok(FinalizedSinkResult::MakeDataSink(std::move(parsed).value()));
	}
	std::vector<std::string> multiline_display(bool /*verbose*/) const override {
		return {"DataSinkFinish: " + operation_id_};
	}

private:
	PipelineNodeContext context_;
	PipelineNodeRef child_;
	std::string operation_id_;
};

} // namespace distributed
} // namespace duckdb
