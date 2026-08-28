// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/operator/exchange/repartition.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/execution/distributed/exchange/exchange.hpp"
#include "duckdb/execution/distributed/exchange/exchange_handles.hpp"
#include "duckdb/planner/bound_result_modifier.hpp"
// Forward-declare PlanExecutionContext to avoid circular includes
namespace duckdb {
namespace distributed {
class PlanExecutionContext;
class PlanConfig;
class TaskIDCounter;
class ExchangeManager;
} // namespace distributed
} // namespace duckdb

namespace duckdb {
namespace distributed {

class RepartitionNode : public PipelineNodeImpl, public std::enable_shared_from_this<RepartitionNode> {
private:
	PipelineNodeConfig config_;
	PipelineNodeContext context_;
	std::shared_ptr<::duckdb::RepartitionSpec> repartition_spec_;
	size_t num_partitions_;
	std::shared_ptr<DistributedPipelineNode> child_;
	std::shared_ptr<ExchangeManager> exchange_mgr_;
	bool preserve_order_ = false;
	bool collect_mark_join_build_summary_ = false;
	vector<unique_ptr<Expression>> mark_join_build_expressions_;

	static constexpr const char *NODE_NAME = "Repartition";

public:
	static std::shared_ptr<RepartitionNode> create(NodeID node_id, const std::shared_ptr<PlanConfig> &plan_config,
	                                               std::shared_ptr<::duckdb::RepartitionSpec> repartition_spec,
	                                               size_t num_partitions, SchemaRef schema,
	                                               std::shared_ptr<DistributedPipelineNode> child,
	                                               std::shared_ptr<ExchangeManager> exchange_mgr = nullptr,
	                                               bool preserve_order = false);

	std::shared_ptr<DistributedPipelineNode> into_node();

	const PipelineNodeContext &context() const override;

	const PipelineNodeConfig &config() const override;

	std::vector<PipelineNodeRef> children() const override;

	std::vector<std::string> multiline_display(bool verbose) const override;

	void EnableMarkJoinBuildSummary(vector<unique_ptr<Expression>> expressions) {
		collect_mark_join_build_summary_ = true;
		mark_join_build_expressions_ = std::move(expressions);
	}

	bool CollectsMarkJoinBuildSummary() const {
		return collect_mark_join_build_summary_;
	}

	bool PreservesOrder() const {
		return preserve_order_;
	}

	bool is_materialization_barrier() const override {
		return collect_mark_join_build_summary_ || preserve_order_;
	}

	std::vector<NodeID> materialized_input_node_ids() const override {
		return is_materialization_barrier() && child_ ? std::vector<NodeID> {child_->node_id()}
		                                              : std::vector<NodeID> {};
	}

	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override;

	NodeID node_id() const override {
		return context_.node_id();
	}

private:
	RepartitionNode(PipelineNodeConfig config, PipelineNodeContext context,
	                std::shared_ptr<::duckdb::RepartitionSpec> repartition_spec, size_t num_partitions,
	                std::shared_ptr<DistributedPipelineNode> child, std::shared_ptr<ExchangeManager> exchange_mgr,
	                bool preserve_order);

	// No separate execution_loop; production logic implemented in .cpp
};

// ─── Shared exchange plan builders (used by both RepartitionNode and BroadcastJoinNode) ────

DuckPhysicalPlanRef AddRemoteExchangeSinkPlan(DuckPhysicalPlanRef plan,
                                              const std::shared_ptr<::duckdb::RepartitionSpec> &spec,
                                              const Exchange &exchange, std::shared_ptr<ExchangeManager> exchange_mgr,
                                              bool collect_mark_join_build_summary = false,
                                              vector<unique_ptr<Expression>> mark_join_build_expressions = {},
                                              bool preserve_order = false);

DuckPhysicalPlanRef AddRemoteRangeExchangeSinkPlan(DuckPhysicalPlanRef plan,
                                                   const vector<::duckdb::BoundOrderByNode> &orders,
                                                   const Exchange &exchange,
                                                   std::shared_ptr<ExchangeManager> exchange_mgr,
                                                   vector<string> boundary_keys);

DuckPhysicalPlanRef MakeRemoteExchangeSourcePlan(const vector<LogicalType> &types, idx_t estimated_cardinality,
                                                 const std::string &exchange_id, vector<idx_t> partition_indices,
                                                 std::vector<ExchangeSourceHandle> source_handles,
                                                 std::shared_ptr<ExchangeManager> exchange_mgr,
                                                 const vector<std::string> &source_nodes,
                                                 optional_idx runtime_source_node_id = optional_idx(),
                                                 bool preserve_order = false);

} // namespace distributed
} // namespace duckdb
