// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "duckdb/common/vector.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/distributed/plan/plan_config.hpp"

namespace duckdb {
namespace distributed {

class TaskIDCounter;

class PositionalJoinNode : public PipelineNodeImpl, public std::enable_shared_from_this<PositionalJoinNode> {
public:
	PositionalJoinNode(NodeID node_id, const PlanConfig &plan_config, duckdb::vector<LogicalType> output_types,
	                   idx_t estimated_cardinality, std::shared_ptr<DistributedPipelineNode> left,
	                   std::shared_ptr<DistributedPipelineNode> right, SchemaRef schema);

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
	SubmittableTask<WorkerTask> BuildPositionalJoinTask(SubmittableTask<WorkerTask> left_task,
	                                                    SubmittableTask<WorkerTask> right_task,
	                                                    TaskIDCounter &task_id_counter,
	                                                    ::duckdb::ClientContext *client_context);

private:
	PipelineNodeConfig config_;
	PipelineNodeContext context_;
	duckdb::vector<LogicalType> output_types_;
	idx_t estimated_cardinality_;
	std::shared_ptr<DistributedPipelineNode> left_;
	std::shared_ptr<DistributedPipelineNode> right_;
};

} // namespace distributed
} // namespace duckdb
