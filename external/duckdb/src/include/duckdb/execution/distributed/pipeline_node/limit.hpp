// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/planner/bound_result_modifier.hpp"

namespace duckdb {
namespace distributed {

class ExchangeManager;

BoundLimitNode CopyBoundLimitNode(const BoundLimitNode &node);

class LimitNode : public PipelineNodeImpl, public std::enable_shared_from_this<LimitNode> {
public:
	LimitNode(NodeID node_id, PipelineNodeRef child, BoundLimitNode limit_val, BoundLimitNode offset_val,
	          std::shared_ptr<ExchangeManager> exchange_mgr = nullptr);

	std::string name() const override {
		return "Limit";
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
	BoundLimitNode limit_val_;
	BoundLimitNode offset_val_;
	std::shared_ptr<ExchangeManager> exchange_mgr_;
};

class StreamingLimitNode : public PipelineNodeImpl, public std::enable_shared_from_this<StreamingLimitNode> {
public:
	StreamingLimitNode(NodeID node_id, PipelineNodeRef child, BoundLimitNode limit_val, BoundLimitNode offset_val,
	                   bool parallel, std::shared_ptr<ExchangeManager> exchange_mgr = nullptr);

	std::string name() const override {
		return "StreamingLimit";
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
	BoundLimitNode limit_val_;
	BoundLimitNode offset_val_;
	bool parallel_;
	std::shared_ptr<ExchangeManager> exchange_mgr_;
};

class LimitPercentNode : public PipelineNodeImpl, public std::enable_shared_from_this<LimitPercentNode> {
public:
	LimitPercentNode(NodeID node_id, PipelineNodeRef child, BoundLimitNode limit_val, BoundLimitNode offset_val,
	                 std::shared_ptr<ExchangeManager> exchange_mgr = nullptr);

	std::string name() const override {
		return "LimitPercent";
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
	BoundLimitNode limit_val_;
	BoundLimitNode offset_val_;
	std::shared_ptr<ExchangeManager> exchange_mgr_;
};

} // namespace distributed
} // namespace duckdb
