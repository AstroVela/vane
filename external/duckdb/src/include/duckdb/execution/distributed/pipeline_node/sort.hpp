// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/planner/bound_result_modifier.hpp"

namespace duckdb {
namespace distributed {

class ExchangeManager;

class OrderByNode : public PipelineNodeImpl, public std::enable_shared_from_this<OrderByNode> {
public:
	OrderByNode(NodeID node_id, PipelineNodeRef child, vector<BoundOrderByNode> orders, vector<idx_t> projections,
	            vector<LogicalType> output_types, bool is_index_sort,
	            std::shared_ptr<ExchangeManager> exchange_mgr = nullptr);

	std::string name() const override {
		return "OrderBy";
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
	vector<BoundOrderByNode> orders_;
	vector<idx_t> projections_;
	vector<LogicalType> output_types_;
	bool is_index_sort_;
	std::shared_ptr<ExchangeManager> exchange_mgr_;
};

class TopNNode : public PipelineNodeImpl, public std::enable_shared_from_this<TopNNode> {
public:
	TopNNode(NodeID node_id, PipelineNodeRef child, vector<BoundOrderByNode> orders, idx_t limit, idx_t offset,
	         std::shared_ptr<ExchangeManager> exchange_mgr = nullptr);

	std::string name() const override {
		return "TopN";
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
	std::vector<std::string> multiline_display(bool verbose) const override;

private:
	PipelineNodeContext ctx_;
	PipelineNodeConfig config_;
	PipelineNodeRef child_;
	vector<BoundOrderByNode> orders_;
	idx_t limit_;
	idx_t offset_;
	std::shared_ptr<ExchangeManager> exchange_mgr_;
};

} // namespace distributed
} // namespace duckdb
