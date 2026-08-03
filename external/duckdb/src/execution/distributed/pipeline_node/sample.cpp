// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/sample.hpp"

#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/execution/distributed/pipeline_node/materialized_coordinator.hpp"
#include "duckdb/execution/operator/helper/physical_reservoir_sample.hpp"
#include "duckdb/execution/operator/helper/physical_distributed_reservoir_sample.hpp"
#include "duckdb/execution/operator/helper/physical_streaming_sample.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"

namespace duckdb {
namespace distributed {

namespace {

PipelineNodeConfig MakeReservoirSampleConfig(const PipelineNodeRef &child, const SampleOptions *options) {
	if (!child) {
		return PipelineNodeConfig();
	}
	if (!options || options->is_percentage) {
		return child->config();
	}
	return PipelineNodeConfig(child->config().schema(), child->config().execution_config(),
	                          ClusteringSpec::unknown_with_num_partitions(1));
}

} // namespace

ReservoirSampleNode::ReservoirSampleNode(NodeID node_id, PipelineNodeRef child, unique_ptr<SampleOptions> options,
                                         std::vector<LogicalType> output_types,
                                         std::shared_ptr<ExchangeManager> exchange_mgr)
    : ctx_(InheritPipelineNodeContext(child, node_id, "ReservoirSample")),
      config_(MakeReservoirSampleConfig(child, options.get())), child_(std::move(child)), options_(std::move(options)),
      output_types_(std::move(output_types)), exchange_mgr_(std::move(exchange_mgr)) {
}

std::vector<PipelineNodeRef> ReservoirSampleNode::children() const {
	return {child_};
}

SubmittableTaskStream<WorkerTask> ReservoirSampleNode::produce_tasks(PlanExecutionContext &plan_context) {
	auto options_template = std::shared_ptr<SampleOptions>(options_ ? options_->Copy().release() : nullptr);
	auto output_types = std::make_shared<vector<LogicalType>>(output_types_.begin(), output_types_.end());
	auto single_stage_builder = [options_template,
	                             output_types](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
		if (!options_template) {
			return input_plan;
		}
		auto options_copy = options_template->Copy();
		auto types = *output_types;
		idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;

		auto &old_root = input_plan->Root();
		auto &sample_op = input_plan->Make<duckdb::PhysicalReservoirSample>(std::move(types), std::move(options_copy),
		                                                                    estimated_cardinality);
		sample_op.children.push_back(old_root);
		input_plan->SetRoot(sample_op);
		return input_plan;
	};

	if (!options_template || options_template->is_percentage) {
		auto input_stream = child_->produce_tasks(plan_context);
		return input_stream.pipeline_instruction(shared_from_this(), std::move(single_stage_builder),
		                                         plan_context.client_context());
	}

	// A logical partition can still produce multiple task-stream events (for
	// example, successive repartition sink waves or FTE task expansion).
	// Fixed-row sampling therefore always needs a global merge stage.
	auto state_types =
	    std::make_shared<vector<LogicalType>>(vector<LogicalType> {LogicalType::UBIGINT, LogicalType::BLOB});
	auto final_plan_builder = [options_template, output_types](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
		auto options_copy = options_template->Copy();
		auto types = *output_types;
		const auto estimated_cardinality = NumericCast<idx_t>(options_copy->sample_size.GetValue<int64_t>());

		auto &old_root = input_plan->Root();
		auto &sample_op = input_plan->Make<duckdb::PhysicalDistributedReservoirSample>(
		    std::move(types), std::move(options_copy), DistributedReservoirSampleStage::FINAL, 0,
		    estimated_cardinality);
		sample_op.children.push_back(old_root);
		input_plan->SetRoot(sample_op);
		return input_plan;
	};
	// The factory argument is only a fragment-template position and can depend
	// on asynchronous child arrival. The LOCAL operator stays unbound until FTE
	// applies its stable runtime task partition before execution.
	auto per_task_builder_factory = [options_template,
	                                 state_types](idx_t /*provisional_task_index*/) -> MaterializedPlanBuilder {
		return [options_template, state_types](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
			auto options_copy = options_template->Copy();
			auto types = *state_types;

			auto &old_root = input_plan->Root();
			auto &sample_op = input_plan->Make<duckdb::PhysicalDistributedReservoirSample>(
			    std::move(types), std::move(options_copy), DistributedReservoirSampleStage::LOCAL,
			    DConstants::INVALID_INDEX, 1);
			sample_op.children.push_back(old_root);
			input_plan->SetRoot(sample_op);
			return input_plan;
		};
	};

	return ProduceWithMaterializedCoordinator(
	    plan_context, child_, std::static_pointer_cast<PipelineNodeImpl>(shared_from_this()),
	    std::move(final_plan_builder), std::move(per_task_builder_factory), exchange_mgr_, MakeSchemaRef(*state_types));
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
