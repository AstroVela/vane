// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include "duckdb/execution/distributed/utils/optional.hpp"
#include <string>
#include <vector>

#include "duckdb/common/enums/join_type.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/distributed/plan/plan_config.hpp"
#include "duckdb/execution/distributed/exchange/exchange_manager.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/join/join_filter_pushdown.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"

namespace duckdb {
namespace distributed {

class BroadcastJoinNode : public PipelineNodeImpl, public std::enable_shared_from_this<BroadcastJoinNode> {
public:
	BroadcastJoinNode(NodeID node_id, const PlanConfig &plan_config, duckdb::vector<JoinCondition> conditions,
	                  JoinType join_type, duckdb::vector<LogicalType> output_types,
	                  duckdb::vector<LogicalType> delim_types, duckdb::vector<LogicalType> condition_types,
	                  PhysicalHashJoin::JoinProjectionColumns payload_columns,
	                  PhysicalHashJoin::JoinProjectionColumns lhs_output_columns,
	                  PhysicalHashJoin::JoinProjectionColumns rhs_output_columns,
	                  duckdb::vector<unique_ptr<BaseStatistics>> join_stats,
	                  unique_ptr<JoinFilterPushdownInfo> filter_pushdown, idx_t estimated_cardinality, bool is_swapped,
	                  std::shared_ptr<DistributedPipelineNode> broadcaster,
	                  std::shared_ptr<DistributedPipelineNode> receiver, SchemaRef schema,
	                  std::shared_ptr<ExchangeManager> exchange_mgr = nullptr);

	std::shared_ptr<DistributedPipelineNode> into_node();

	const PipelineNodeContext &context() const override {
		return context_;
	}
	const PipelineNodeConfig &config() const override {
		return config_;
	}
	std::vector<PipelineNodeRef> children() const override;
	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override;
	std::vector<std::string> multiline_display(bool verbose) const override;

private:
	static duckdb::vector<JoinCondition> CopyConditions(const duckdb::vector<JoinCondition> &conditions);
	static unique_ptr<JoinFilterPushdownInfo> CopyFilterPushdownInfo(const JoinFilterPushdownInfo &info);
	static duckdb::vector<unique_ptr<BaseStatistics>>
	CopyJoinStats(const duckdb::vector<unique_ptr<BaseStatistics>> &stats);
	static std::pair<bool, SubmittableTask<WorkerTask>> PollNextWithWait(SubmittableTaskStream<WorkerTask> &stream);

	SubmittableTask<WorkerTask> BuildBroadcastHashJoinTask(SubmittableTask<WorkerTask> receiver_task,
	                                                       const DuckPhysicalPlanRef &broadcast_plan,
	                                                       ::duckdb::ClientContext *client_context);

private:
	PipelineNodeConfig config_;
	PipelineNodeContext context_;
	std::shared_ptr<DistributedPipelineNode> broadcaster_;
	std::shared_ptr<DistributedPipelineNode> receiver_;
	bool is_swapped_ = false;

	duckdb::vector<JoinCondition> conditions_;
	JoinType join_type_;
	duckdb::vector<LogicalType> output_types_;
	duckdb::vector<LogicalType> delim_types_;
	duckdb::vector<LogicalType> condition_types_;
	PhysicalHashJoin::JoinProjectionColumns payload_columns_;
	PhysicalHashJoin::JoinProjectionColumns lhs_output_columns_;
	PhysicalHashJoin::JoinProjectionColumns rhs_output_columns_;
	duckdb::vector<unique_ptr<BaseStatistics>> join_stats_;
	unique_ptr<JoinFilterPushdownInfo> filter_pushdown_;
	idx_t estimated_cardinality_;
	std::shared_ptr<ExchangeManager> exchange_mgr_;
};

} // namespace distributed
} // namespace duckdb
