// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/sample.hpp"

#include "duckdb/execution/operator/helper/physical_reservoir_sample.hpp"
#include "duckdb/execution/operator/helper/physical_streaming_sample.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"

namespace duckdb {
namespace distributed {

ReservoirSampleNode::ReservoirSampleNode(NodeID node_id, PipelineNodeRef child, unique_ptr<SampleOptions> options,
                                         std::vector<LogicalType> output_types)
    : ctx_(InheritPipelineNodeContext(child, node_id, "ReservoirSample")),
      config_(child ? child->config() : PipelineNodeConfig()), child_(std::move(child)), options_(std::move(options)),
      output_types_(std::move(output_types)) {
}

std::vector<PipelineNodeRef> ReservoirSampleNode::children() const {
	return {child_};
}

SubmittableTaskStream<WorkerTask> ReservoirSampleNode::produce_tasks(PlanExecutionContext &plan_context) {
	auto input_stream = child_->produce_tasks(plan_context);
	auto *options_ptr = options_.get();
	const auto *output_types_ptr = &output_types_;
	return input_stream.pipeline_instruction(
	    shared_from_this(),
	    [options_ptr, output_types_ptr](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
		    if (!options_ptr) {
			    return input_plan;
		    }
		    auto options_copy = options_ptr->Copy();
		    auto types = *output_types_ptr;
		    idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;

		    auto &old_root = input_plan->Root();
		    auto &sample_op = input_plan->Make<duckdb::PhysicalReservoirSample>(
		        std::move(types), std::move(options_copy), estimated_cardinality);
		    sample_op.children.push_back(old_root);
		    input_plan->SetRoot(sample_op);
		    return input_plan;
	    },
	    plan_context.client_context());
}

std::vector<std::string> ReservoirSampleNode::multiline_display(bool /*verbose*/) const {
	return {std::string("ReservoirSample")};
}

StreamingSampleNode::StreamingSampleNode(NodeID node_id, PipelineNodeRef child, unique_ptr<SampleOptions> options,
                                         std::vector<LogicalType> output_types)
    : ctx_(InheritPipelineNodeContext(child, node_id, "StreamingSample")),
      config_(child ? child->config() : PipelineNodeConfig()), child_(std::move(child)), options_(std::move(options)),
      output_types_(std::move(output_types)) {
}

std::vector<PipelineNodeRef> StreamingSampleNode::children() const {
	return {child_};
}

SubmittableTaskStream<WorkerTask> StreamingSampleNode::produce_tasks(PlanExecutionContext &plan_context) {
	auto input_stream = child_->produce_tasks(plan_context);
	auto *options_ptr = options_.get();
	const auto *output_types_ptr = &output_types_;
	return input_stream.pipeline_instruction(
	    shared_from_this(),
	    [options_ptr, output_types_ptr](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
		    if (!options_ptr) {
			    return input_plan;
		    }
		    auto options_copy = options_ptr->Copy();
		    auto types = *output_types_ptr;
		    idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;

		    auto &old_root = input_plan->Root();
		    auto &sample_op = input_plan->Make<duckdb::PhysicalStreamingSample>(
		        std::move(types), std::move(options_copy), estimated_cardinality);
		    sample_op.children.push_back(old_root);
		    input_plan->SetRoot(sample_op);
		    return input_plan;
	    },
	    plan_context.client_context());
}

std::vector<std::string> StreamingSampleNode::multiline_display(bool /*verbose*/) const {
	return {std::string("StreamingSample")};
}

} // namespace distributed
} // namespace duckdb
