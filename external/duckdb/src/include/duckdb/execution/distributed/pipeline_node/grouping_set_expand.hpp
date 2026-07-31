// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/parser/group_by_node.hpp"

namespace duckdb {
namespace distributed {

class GroupingSetExpandNode : public PipelineNodeImpl, public std::enable_shared_from_this<GroupingSetExpandNode> {
public:
	GroupingSetExpandNode(NodeID node_id, PipelineNodeRef child, std::vector<BoundExpr> groups,
	                      std::vector<GroupingSet> grouping_sets, std::vector<std::vector<idx_t>> grouping_functions,
	                      std::vector<idx_t> aggregate_filter_indexes, std::vector<LogicalType> output_types);

	std::string name() const override {
		return "GroupingSetExpand";
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

	std::vector<PipelineNodeRef> children() const override;
	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override;
	std::vector<std::string> multiline_display(bool verbose) const override;

private:
	PipelineNodeContext ctx_;
	PipelineNodeConfig config_;
	PipelineNodeRef child_;
	std::vector<BoundExpr> groups_;
	std::vector<GroupingSet> grouping_sets_;
	std::vector<std::vector<idx_t>> grouping_functions_;
	std::vector<idx_t> aggregate_filter_indexes_;
	std::vector<LogicalType> output_types_;
	bool has_empty_grouping_set_;
};

} // namespace distributed
} // namespace duckdb
