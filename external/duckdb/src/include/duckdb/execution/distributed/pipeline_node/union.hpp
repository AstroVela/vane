// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/distributed/plan/plan_config.hpp"

namespace duckdb {
namespace distributed {

//! Driver-side fan-in for the independent task streams below a physical UNION.
//! UNION is intentionally absent from worker plans: every emitted task retains
//! its branch plan and source inputs, while the task lineage records this node.
class UnionNode : public PipelineNodeImpl, public std::enable_shared_from_this<UnionNode> {
public:
	UnionNode(NodeID node_id, const PlanConfig &plan_config, std::vector<DistributedPipelineNodeRef> children,
	          SchemaRef schema, bool allow_out_of_order);

	const PipelineNodeContext &context() const override {
		return context_;
	}
	const PipelineNodeConfig &config() const override {
		return config_;
	}
	std::vector<PipelineNodeRef> children() const override;
	bool is_statically_empty_result() const override {
		return statically_empty_;
	}
	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override;
	std::vector<std::string> multiline_display(bool verbose) const override;

	bool allow_out_of_order() const {
		return allow_out_of_order_;
	}

private:
	PipelineNodeContext context_;
	PipelineNodeConfig config_;
	std::vector<DistributedPipelineNodeRef> children_;
	bool allow_out_of_order_;
	bool statically_empty_;
};

} // namespace distributed
} // namespace duckdb
