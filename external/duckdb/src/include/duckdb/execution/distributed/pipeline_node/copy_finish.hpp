// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

// CopyFinishNode — coordinator-side copy finalize operator.
// Wraps a CopySinkNode. After all worker tasks complete, `finalize()` collects
// file metadata from worker outputs and renames staging files to final paths.
#pragma once

#include "duckdb/execution/distributed/pipeline_node/sink.hpp"
#include "duckdb/execution/distributed/copy_finalize.hpp"

namespace duckdb {
namespace distributed {

class CopyFinishNode : public PipelineNodeImpl {
public:
	CopyFinishNode(NodeID node_id, std::shared_ptr<CopySinkNode> copy_sink)
	    : ctx_(InheritPipelineNodeContext(copy_sink, node_id, "CopyFinish")), copy_sink_(std::move(copy_sink)) {
	}

	std::string name() const override {
		return "CopyFinish";
	}
	bool is_sink() const override {
		return true;
	}
	NodeID node_id() const override {
		return ctx_.node_id();
	}
	const PipelineNodeContext &context() const override {
		return ctx_;
	}
	const PipelineNodeConfig &config() const override {
		return copy_sink_->config();
	}

	std::vector<PipelineNodeRef> children() const override {
		return {copy_sink_};
	}

	/// Delegates task production to the wrapped CopySinkNode.
	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override {
		return copy_sink_->produce_tasks(plan_context);
	}

	/// Access the CopySinkNode for spec/staging info.
	const std::shared_ptr<CopySinkNode> &copy_sink() const {
		return copy_sink_;
	}
	const DistributedCopySpec &spec() const {
		return copy_sink_->spec();
	}
	const std::string &staging_root_base() const {
		return copy_sink_->staging_root_base();
	}
	const std::string &staging_run_id() const {
		return copy_sink_->staging_run_id();
	}

	/// ── Coordinator-side finalize ───────────────────────────────────────
	/// Called by the runner after all worker tasks have completed.
	/// 1. Parses ColumnDataResultPartitions → DistributedCopyFileInfo
	/// 2. Renames staging files to final paths
	/// 3. Cleans up staging directory
	/// Returns the aggregated copy result.
	DuckDBResult<DistributedCopyResult> finalize(const std::vector<ResultPartitionRef> &partitions,
	                                             ClientContext &context) {
		// Step 1: parse worker output fragments → file infos
		auto file_infos_res = ParseCopyPartitions(partitions);
		if (file_infos_res.is_err()) {
			return DuckDBResult<DistributedCopyResult>::err(file_infos_res.error());
		}
		auto files = std::move(file_infos_res).value();

		// Step 2: finalize (staging→final rename + cleanup)
		// When staging_root_base is empty, pass empty staging_root so
		// FinalizeCopyFiles skips MoveFile entirely (remote/object storage, or
		// default local direct-write mode).
		std::string staging_root;
		if (!staging_root_base().empty()) {
			auto &fs = FileSystem::GetFileSystem(context);
			staging_root = fs.JoinPath(staging_root_base(), staging_run_id());
		}
		return FinalizeCopyFiles(spec(), staging_root, std::move(files), context, staging_run_id());
	}

	std::vector<std::string> multiline_display(bool /*verbose*/) const override {
		return {std::string("CopyFinish: ") + copy_sink_->spec().file_path};
	}

private:
	PipelineNodeContext ctx_;
	std::shared_ptr<CopySinkNode> copy_sink_;
};

} // namespace distributed
} // namespace duckdb
