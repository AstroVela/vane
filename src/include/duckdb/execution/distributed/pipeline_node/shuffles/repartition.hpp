// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/operator/exchange/repartition.hpp"
#include "duckdb/common/optional_idx.hpp"
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

// 重分区节点类
class RepartitionNode : public PipelineNodeImpl, public std::enable_shared_from_this<RepartitionNode> {
private:
	PipelineNodeConfig config_;
	PipelineNodeContext context_;
	std::shared_ptr<::duckdb::RepartitionSpec> repartition_spec_;
	size_t num_partitions_;
	std::shared_ptr<DistributedPipelineNode> child_;
	std::shared_ptr<ExchangeManager> exchange_mgr_;

	static constexpr const char *NODE_NAME = "Repartition";

public:
	// 构造函数
	static std::shared_ptr<RepartitionNode> create(NodeID node_id, const std::shared_ptr<PlanConfig> &plan_config,
	                                               std::shared_ptr<::duckdb::RepartitionSpec> repartition_spec,
	                                               size_t num_partitions, SchemaRef schema,
	                                               std::shared_ptr<DistributedPipelineNode> child,
	                                               std::shared_ptr<ExchangeManager> exchange_mgr = nullptr);

	// 转换为分布式管道节点
	std::shared_ptr<DistributedPipelineNode> into_node();

	// PipelineNodeImpl 接口实现
	const PipelineNodeContext &context() const override;

	const PipelineNodeConfig &config() const override;

	std::vector<PipelineNodeRef> children() const override;

	std::vector<std::string> multiline_display(bool verbose) const override;

	// 生成任务流（核心方法）
	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override;

	// 节点ID获取方法
	NodeID node_id() const override {
		return context_.node_id();
	}

private:
	// 私有构造函数
	RepartitionNode(PipelineNodeConfig config, PipelineNodeContext context,
	                std::shared_ptr<::duckdb::RepartitionSpec> repartition_spec, size_t num_partitions,
	                std::shared_ptr<DistributedPipelineNode> child, std::shared_ptr<ExchangeManager> exchange_mgr);

	// No separate execution_loop; production logic implemented in .cpp
};

// ─── Shared exchange plan builders (used by both RepartitionNode and BroadcastJoinNode) ────

DuckPhysicalPlanRef AddRemoteExchangeSinkPlan(DuckPhysicalPlanRef plan,
                                              const std::shared_ptr<::duckdb::RepartitionSpec> &spec,
                                              idx_t num_partitions, const std::string &exchange_id,
                                              const ExchangeSinkInstanceHandle &sink_handle,
                                              std::shared_ptr<ExchangeManager> exchange_mgr);

DuckPhysicalPlanRef AddRemoteRangeExchangeSinkPlan(DuckPhysicalPlanRef plan,
                                                   const vector<::duckdb::BoundOrderByNode> &orders,
                                                   idx_t num_partitions, const std::string &exchange_id,
                                                   const ExchangeSinkInstanceHandle &sink_handle,
                                                   std::shared_ptr<ExchangeManager> exchange_mgr,
                                                   vector<string> boundary_keys);

DuckPhysicalPlanRef MakeRemoteExchangeSourcePlan(const vector<LogicalType> &types, idx_t estimated_cardinality,
                                                 const std::string &exchange_id, vector<idx_t> partition_indices,
                                                 std::vector<ExchangeSourceHandle> source_handles,
                                                 std::shared_ptr<ExchangeManager> exchange_mgr,
                                                 const vector<std::string> &source_nodes,
                                                 optional_idx runtime_source_node_id = optional_idx());

} // namespace distributed
} // namespace duckdb
