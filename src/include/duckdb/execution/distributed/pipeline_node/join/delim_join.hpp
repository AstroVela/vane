// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/shared_ptr.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/operator/join/physical_delim_join.hpp"

namespace duckdb {
class DatabaseInstance;
namespace distributed {

class DelimJoinNode : public PipelineNodeImpl, public std::enable_shared_from_this<DelimJoinNode> {
public:
	DelimJoinNode(NodeID node_id, const PhysicalDelimJoin &delim_join, std::shared_ptr<DistributedPipelineNode> child,
	              SchemaRef schema, shared_ptr<DatabaseInstance> db);

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
	static std::string SerializeOperator(const PhysicalOperator &op);
	static unique_ptr<PhysicalOperator> DeserializeOperator(const std::string &data, PhysicalPlan &plan,
	                                                        shared_ptr<DatabaseInstance> db);
	static void GatherDelimScans(PhysicalOperator &op, vector<const_reference<PhysicalOperator>> &delim_scans,
	                             optional_idx delim_idx);

private:
	PipelineNodeContext context_;
	PipelineNodeConfig config_;
	std::shared_ptr<DistributedPipelineNode> child_;
	PhysicalOperatorType delim_type_;
	vector<LogicalType> output_types_;
	optional_idx delim_idx_;
	idx_t estimated_cardinality_;
	std::string join_bytes_;
	std::string distinct_bytes_;
	shared_ptr<DatabaseInstance> db_;
};

} // namespace distributed
} // namespace duckdb
