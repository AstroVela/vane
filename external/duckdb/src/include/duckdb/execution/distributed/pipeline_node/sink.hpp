// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

// Distributed COPY sink node
#pragma once

#include "duckdb/execution/distributed/copy_to_file.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/operator/persistent/physical_batch_copy_to_file.hpp"
#include "duckdb/execution/operator/persistent/physical_copy_to_file.hpp"

#include "duckdb/common/types/uuid.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/exception.hpp"

#include <cstdlib>

namespace duckdb {
namespace distributed {

inline bool DistributedCopyReadBoolEnv(const char *name, bool &result) {
	auto raw = std::getenv(name);
	if (!raw || !*raw) {
		return false;
	}
	auto value = StringUtil::Lower(std::string(raw));
	result = value != "0" && value != "false" && value != "no" && value != "off";
	return true;
}

inline bool DistributedCopyLocalStagingEnabled() {
	bool result = false;
	if (DistributedCopyReadBoolEnv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", result)) {
		return result;
	}
	return false;
}

class CopySinkNode : public PipelineNodeImpl, public std::enable_shared_from_this<CopySinkNode> {
public:
	CopySinkNode(NodeID node_id, PipelineNodeRef child, DistributedCopySpec spec)
	    : ctx_(InheritPipelineNodeContext(child, node_id, "CopySink")),
	      config_(child ? child->config() : PipelineNodeConfig()), child_(std::move(child)), spec_(std::move(spec)) {
		const auto local_staging_enabled = DistributedCopyLocalStagingEnabled();
		if (FileSystem::IsRemoteFile(spec_.file_path) && local_staging_enabled) {
			throw InvalidInputException(StringUtil::Format(
			    "VANE_DISTRIBUTED_COPY_LOCAL_STAGING cannot be enabled for remote distributed COPY output: %s",
			    spec_.file_path));
		}

		// Direct-write is the default for every filesystem. Workers write
		// visible run-prefixed output files under the requested output path.
		// Shared local filesystems can opt back into staging + MoveFile/rename
		// with VANE_DISTRIBUTED_COPY_LOCAL_STAGING=1.
		if (!local_staging_enabled) {
			staging_root_base_ = "";
		} else {
			staging_root_base_ = spec_.file_path + ".duckdb_staging";
		}
		staging_run_id_ = UUID::ToString(UUID::GenerateRandomUUID());
	}

	std::string name() const override {
		return "CopySink";
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

	const DistributedCopySpec &spec() const {
		return spec_;
	}
	const std::string &staging_root_base() const {
		return staging_root_base_;
	}
	const std::string &staging_run_id() const {
		return staging_run_id_;
	}

	std::vector<PipelineNodeRef> children() const override {
		return {child_};
	}
	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &plan_context) override;

	std::vector<std::string> multiline_display(bool /*verbose*/) const override {
		return {std::string("CopySink: ") + spec_.file_path};
	}

private:
	PipelineNodeContext ctx_;
	PipelineNodeConfig config_;
	PipelineNodeRef child_;
	DistributedCopySpec spec_;
	std::string staging_root_base_;
	std::string staging_run_id_;
};

} // namespace distributed
} // namespace duckdb
