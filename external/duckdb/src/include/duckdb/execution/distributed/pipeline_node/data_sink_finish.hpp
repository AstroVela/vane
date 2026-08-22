// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/data_sink.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"

namespace duckdb {
namespace distributed {

class DataSinkFinishNode : public PipelineNodeImpl {
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
	DuckDBResult<DistributedDataSinkResult> finalize(const vector<ResultPartitionRef> &partitions) const {
		return ParseDataSinkPartitions(operation_id_, partitions);
	}
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
