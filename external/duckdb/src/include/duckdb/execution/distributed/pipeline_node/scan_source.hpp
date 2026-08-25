// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/distributed/plan/scan_split.hpp"
#include "duckdb/execution/operator/exchange/repartition.hpp"

namespace duckdb {
namespace distributed {

class ScanSourceNode : public PipelineNodeImpl, public std::enable_shared_from_this<ScanSourceNode> {
public:
	ScanSourceNode(PipelineNodeContext context, DuckPhysicalPlanRef scan_plan, std::vector<ScanSplit> scan_splits,
	               SchemaRef schema, DuckDBExecutionConfigRef exec_cfg, bool require_scan_splits)
	    : ctx_(std::move(context)),
	      config_(std::move(schema), std::move(exec_cfg),
	              ClusteringSpec::unknown_with_num_partitions(scan_splits.empty() ? 1 : scan_splits.size())),
	      scan_plan_(std::move(scan_plan)), scan_splits_(std::move(scan_splits)),
	      require_scan_splits_(require_scan_splits), scan_pset_key_(std::to_string(ctx_.node_id())) {
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
	const std::vector<ScanSplit> &scan_splits() const {
		return scan_splits_;
	}
	const std::string &scan_pset_key() const {
		return scan_pset_key_;
	}
	bool require_scan_splits() const {
		return require_scan_splits_;
	}

	std::vector<PipelineNodeRef> children() const override {
		return {};
	}

	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override;

	std::vector<std::string> multiline_display(bool verbose) const override {
		std::vector<std::string> s;
		s.push_back("ScanSplitSource:");
		s.push_back("Num Scan Splits = " + std::to_string(scan_splits_.size()));
		s.push_back("Schema: {" + config_.schema()->ToString() + "}");
		return s;
	}

private:
	PipelineNodeContext ctx_;
	PipelineNodeConfig config_;
	DuckPhysicalPlanRef scan_plan_;
	std::vector<ScanSplit> scan_splits_;
	bool require_scan_splits_;
	std::string scan_pset_key_;
};

} // namespace distributed
} // namespace duckdb
