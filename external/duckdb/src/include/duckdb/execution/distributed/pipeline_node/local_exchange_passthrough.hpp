// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

// LocalExchangePassthroughNode: keeps LOCAL_EXCHANGE in the physical plan
// instead of converting to a distributed RepartitionNode shuffle.
// This lets the worker's native DuckDB executor create two concurrent pipelines
// connected by a bounded buffer — enabling CPU/GPU UDF overlap within a single worker.
#pragma once

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/operator/exchange/physical_local_exchange.hpp"
#include "duckdb/execution/operator/exchange/repartition.hpp"

namespace duckdb {
namespace distributed {

class LocalExchangePassthroughNode : public PipelineNodeImpl,
                                     public std::enable_shared_from_this<LocalExchangePassthroughNode> {
public:
	LocalExchangePassthroughNode(NodeID node_id, PipelineNodeRef child,
	                             std::shared_ptr<RepartitionSpec> repartition_spec, SchemaRef schema,
	                             idx_t estimated_cardinality)
	    : ctx_(InheritPipelineNodeContext(child, node_id, "LocalExchange")),
	      config_(std::move(schema), child ? child->config().execution_config() : DuckDBExecutionConfigRef(),
	              child ? child->config().clustering_spec() : ClusteringSpec::unknown_with_num_partitions(1)),
	      child_(std::move(child)), repartition_spec_(std::move(repartition_spec)),
	      estimated_cardinality_(estimated_cardinality) {
	}

	std::string name() const override {
		return "LocalExchange";
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

	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override {
		auto input_stream = child_->produce_tasks(plan_context);
		auto spec = repartition_spec_;
		auto est_card = estimated_cardinality_;

		return input_stream.pipeline_instruction(
		    shared_from_this(),
		    [spec, est_card](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
			    auto types = input_plan->Root().GetTypes();
			    auto &old_root = input_plan->Root();
			    // Make<T>(args...) calls T(*this, args...) where *this is the PhysicalPlan
			    auto &exchange = input_plan->Make<PhysicalLocalExchange>(std::move(types), spec, est_card);
			    exchange.children.push_back(old_root);
			    input_plan->SetRoot(exchange);
			    return input_plan;
		    },
		    plan_context.client_context());
	}

	std::vector<std::string> multiline_display(bool verbose) const override {
		return {"LocalExchange (passthrough)"};
	}

private:
	PipelineNodeContext ctx_;
	PipelineNodeConfig config_;
	PipelineNodeRef child_;
	std::shared_ptr<RepartitionSpec> repartition_spec_;
	idx_t estimated_cardinality_;
};

} // namespace distributed
} // namespace duckdb
