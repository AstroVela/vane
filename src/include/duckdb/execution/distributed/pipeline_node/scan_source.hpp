// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/distributed/plan/scan_task.hpp"
#include "duckdb/execution/operator/exchange/repartition.hpp"

namespace duckdb {
namespace distributed {

class ScanSourceNode : public PipelineNodeImpl, public std::enable_shared_from_this<ScanSourceNode> {
public:
	ScanSourceNode(PipelineNodeContext context, DuckPhysicalPlanRef scan_plan,
	               std::vector<ScanTaskDescriptor> scan_tasks, SchemaRef schema, DuckDBExecutionConfigRef exec_cfg,
	               bool require_scan_tasks)
	    : ctx_(std::move(context)),
	      config_(std::move(schema), std::move(exec_cfg),
	              ClusteringSpec::unknown_with_num_partitions(scan_tasks.empty() ? 1 : scan_tasks.size())),
	      scan_plan_(std::move(scan_plan)), scan_tasks_(std::move(scan_tasks)), require_scan_tasks_(require_scan_tasks),
	      scan_pset_key_(std::to_string(ctx_.node_id())) {
	}

	std::string name() const override {
		return "ScanSource";
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
	const std::vector<ScanTaskDescriptor> &scan_tasks() const {
		return scan_tasks_;
	}
	const std::string &scan_pset_key() const {
		return scan_pset_key_;
	}
	bool require_scan_tasks() const {
		return require_scan_tasks_;
	}

	std::vector<PipelineNodeRef> children() const override {
		return {};
	}

	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override;

	std::vector<std::string> multiline_display(bool verbose) const override {
		std::vector<std::string> s;
		s.push_back("ScanTaskSource:");
		s.push_back("Num Scan Tasks = " + std::to_string(scan_tasks_.size()));
		s.push_back("Schema: {" + config_.schema()->ToString() + "}");
		return s;
	}

private:
	PipelineNodeContext ctx_;
	PipelineNodeConfig config_;
	DuckPhysicalPlanRef scan_plan_;
	std::vector<ScanTaskDescriptor> scan_tasks_;
	bool require_scan_tasks_;
	std::string scan_pset_key_;
};

} // namespace distributed
} // namespace duckdb
