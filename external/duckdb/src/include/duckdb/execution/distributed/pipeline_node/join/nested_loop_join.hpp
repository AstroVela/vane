// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "duckdb/common/vector.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/distributed/plan/plan_config.hpp"
#include "duckdb/execution/operator/join/physical_comparison_join.hpp"

namespace duckdb {
namespace distributed {

class TaskIDCounter;

//! Executes comparison and arbitrary-expression joins with a correctness-first
//! nested-loop strategy after their inputs have been co-located.
class NestedLoopJoinNode : public PipelineNodeImpl, public std::enable_shared_from_this<NestedLoopJoinNode> {
public:
	NestedLoopJoinNode(NodeID node_id, const PlanConfig &plan_config, PhysicalOperatorType source_type,
	                   duckdb::vector<JoinCondition> conditions, unique_ptr<Expression> predicate, JoinType join_type,
	                   duckdb::vector<LogicalType> output_types, idx_t estimated_cardinality,
	                   std::shared_ptr<DistributedPipelineNode> left, std::shared_ptr<DistributedPipelineNode> right,
	                   SchemaRef schema, duckdb::vector<idx_t> left_projection_map = {},
	                   duckdb::vector<idx_t> right_projection_map = {});

	NestedLoopJoinNode(NodeID node_id, const PlanConfig &plan_config, unique_ptr<Expression> condition,
	                   JoinType join_type, duckdb::vector<LogicalType> output_types, idx_t estimated_cardinality,
	                   std::shared_ptr<DistributedPipelineNode> left, std::shared_ptr<DistributedPipelineNode> right,
	                   SchemaRef schema);

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

	SubmittableTask<WorkerTask> BuildNestedLoopJoinTask(SubmittableTask<WorkerTask> left_task,
	                                                    SubmittableTask<WorkerTask> right_task,
	                                                    TaskIDCounter &task_id_counter,
	                                                    ::duckdb::ClientContext *client_context);

private:
	PipelineNodeConfig config_;
	PipelineNodeContext context_;
	PhysicalOperatorType source_type_;
	duckdb::vector<JoinCondition> conditions_;
	unique_ptr<Expression> predicate_;
	unique_ptr<Expression> arbitrary_condition_;
	JoinType join_type_;
	duckdb::vector<LogicalType> output_types_;
	duckdb::vector<idx_t> left_projection_map_;
	duckdb::vector<idx_t> right_projection_map_;
	idx_t estimated_cardinality_;
	std::shared_ptr<DistributedPipelineNode> left_;
	std::shared_ptr<DistributedPipelineNode> right_;
};

} // namespace distributed
} // namespace duckdb
