// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
// Distributed aggregate pipeline nodes and aggregate split helpers.
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/common/types/value.hpp"

#include "duckdb/execution/distributed/plan/distributed_physical_plan.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"

namespace duckdb {
namespace distributed {

// Reuse aliases from common_types.hpp
using BoundExprRef = BoundExpr;       // alias from common_types.hpp
using BoundAggExprRef = BoundAggExpr; // alias from common_types.hpp

/// Aggregate node: performs grouped or ungrouped aggregation on upstream tasks
class AggregateNode : public PipelineNodeImpl, public std::enable_shared_from_this<AggregateNode> {
public:
	AggregateNode(NodeID node_id, const PlanConfig &plan_config, std::vector<BoundExprRef> group_by,
	              std::vector<BoundAggExprRef> aggs, SchemaRef output_schema, DistributedPipelineNodeRef child);

	static NodeName node_name(const std::vector<BoundExprRef> &group_by);

	DistributedPipelineNodeRef into_node();

	// PipelineNodeImpl interface
	PipelineNodeContext &context() {
		return context_;
	}
	const PipelineNodeContext &context() const {
		return context_;
	}
	PipelineNodeConfig &config() {
		return config_;
	}
	const PipelineNodeConfig &config() const {
		return config_;
	}
	std::vector<PipelineNodeRef> children() const override;
	std::string name() const override {
		return "Aggregate";
	}
	NodeID node_id() const override {
		return node_id_;
	}
	std::vector<std::string> multiline_display(bool verbose) const;
	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override;

private:
	PipelineNodeConfig config_;
	PipelineNodeContext context_;
	NodeID node_id_;
	std::vector<BoundExprRef> group_by_;
	std::vector<BoundAggExprRef> aggs_;
	DistributedPipelineNodeRef child_;
};

class PerfectHashAggregateNode : public PipelineNodeImpl,
                                 public std::enable_shared_from_this<PerfectHashAggregateNode> {
public:
	PerfectHashAggregateNode(NodeID node_id, std::vector<BoundExprRef> group_by, std::vector<BoundAggExprRef> aggs,
	                         std::vector<Value> group_minima, std::vector<idx_t> required_bits,
	                         std::vector<LogicalType> output_types, DistributedPipelineNodeRef child);

	std::string name() const override {
		return "PerfectHashAggregate";
	}
	NodeID node_id() const override {
		return node_id_;
	}
	const PipelineNodeContext &context() const override {
		return context_;
	}
	const PipelineNodeConfig &config() const override {
		return config_;
	}
	std::vector<PipelineNodeRef> children() const override;
	std::vector<std::string> multiline_display(bool verbose) const override;
	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override;

private:
	PipelineNodeConfig config_;
	PipelineNodeContext context_;
	NodeID node_id_;
	std::vector<BoundExprRef> group_by_;
	std::vector<BoundAggExprRef> aggs_;
	std::vector<Value> group_minima_;
	std::vector<idx_t> required_bits_;
	DistributedPipelineNodeRef child_;
	std::vector<LogicalType> output_types_;
};

class PartitionedAggregateNode : public PipelineNodeImpl,
                                 public std::enable_shared_from_this<PartitionedAggregateNode> {
public:
	PartitionedAggregateNode(NodeID node_id, std::vector<BoundExprRef> group_by, std::vector<BoundAggExprRef> aggs,
	                         std::vector<column_t> partitions, std::vector<LogicalType> output_types,
	                         DistributedPipelineNodeRef child);

	std::string name() const override {
		return "PartitionedAggregate";
	}
	NodeID node_id() const override {
		return node_id_;
	}
	const PipelineNodeContext &context() const override {
		return context_;
	}
	const PipelineNodeConfig &config() const override {
		return config_;
	}
	std::vector<PipelineNodeRef> children() const override;
	std::vector<std::string> multiline_display(bool verbose) const override;
	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override;

private:
	PipelineNodeConfig config_;
	PipelineNodeContext context_;
	NodeID node_id_;
	std::vector<BoundExprRef> group_by_;
	std::vector<BoundAggExprRef> aggs_;
	std::vector<column_t> partitions_;
	DistributedPipelineNodeRef child_;
	std::vector<LogicalType> output_types_;
};

/// Split details used to implement two-stage/grouped aggregation
struct GroupByAggSplit {
	std::vector<BoundAggExpr> first_stage_aggs;
	SchemaRef first_stage_schema;
	std::vector<BoundExpr> first_stage_group_by;

	std::vector<BoundExpr> partition_by;

	std::vector<BoundAggExpr> second_stage_aggs;
	SchemaRef second_stage_schema;
	std::vector<BoundExpr> second_stage_group_by;

	std::vector<BoundExpr> final_exprs;
};

/// Split aggregations into two-stage plan and final projection. This function
/// mirrors Rust `split_groupby_aggs` and returns a GroupByAggSplit or an error.
DuckDBResult<GroupByAggSplit> split_groupby_aggs(const std::vector<BoundExpr> &group_by,
                                                 const std::vector<BoundAggExpr> &aggs,
                                                 const std::vector<BoundExpr> &partition_by,
                                                 const SchemaRef &input_schema);

} // namespace distributed
} // namespace duckdb
