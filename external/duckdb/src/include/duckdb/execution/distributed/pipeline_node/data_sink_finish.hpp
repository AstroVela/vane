// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/operator/helper/physical_data_sink.hpp"

namespace duckdb {
namespace distributed {

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
	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override {
		auto input_stream = child_->produce_tasks(plan_context);
		auto self = shared_from_this();
		auto operation_id = operation_id_;
		return input_stream.pipeline_instruction(
		    self,
		    [operation_id = std::move(operation_id)](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
			    auto &old_root = input_plan->Root();
			    auto output_types = old_root.GetTypes();
			    auto estimated_cardinality = old_root.estimated_cardinality;
			    auto &sink = input_plan->Make<::duckdb::PhysicalDataSink>(std::move(output_types), operation_id,
			                                                              estimated_cardinality);
			    sink.children.push_back(old_root);
			    input_plan->SetRoot(sink);
			    return input_plan;
		    },
		    plan_context.client_context());
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
