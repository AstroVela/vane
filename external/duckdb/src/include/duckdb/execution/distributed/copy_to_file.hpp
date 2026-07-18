// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include "duckdb/common/enums/copy_overwrite_mode.hpp"
#include "duckdb/common/filename_pattern.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/function/copy_function.hpp"

namespace duckdb {
namespace distributed {

static constexpr const char *DISTRIBUTED_COPY_OUTPUT_PLACEHOLDER_PREFIX = "__duckdb_distributed_copy_placeholder__";
static constexpr const char *DISTRIBUTED_COPY_DIRECT_WRITE_RUN_PREFIX = "_vane_direct_write_";
static constexpr const char *DISTRIBUTED_COPY_DIRECT_WRITE_LIFECYCLE_FILE = "lifecycle.txt";

enum class DistributedCopyType : uint8_t { COPY_TO_FILE = 0, BATCH_COPY_TO_FILE = 1 };

struct DistributedCopySpec {
	DistributedCopyType type = DistributedCopyType::COPY_TO_FILE;
	CopyFunction function {""};
	unique_ptr<FunctionData> bind_data;
	std::string file_path;
	bool use_tmp_file = false;
	FilenamePattern filename_pattern;
	std::string file_extension;
	CopyOverwriteMode overwrite_mode = CopyOverwriteMode::COPY_ERROR_ON_CONFLICT;
	bool parallel = false;
	bool per_thread_output = false;
	optional_idx file_size_bytes;
	bool rotate = false;
	CopyFunctionReturnType return_type = CopyFunctionReturnType::CHANGED_ROWS;
	bool partition_output = false;
	bool write_partition_columns = false;
	bool write_empty_file = true;
	bool hive_file_pattern = true;
	vector<idx_t> partition_columns;
	vector<std::string> names;
	vector<LogicalType> expected_types;

	DistributedCopySpec Clone() const {
		DistributedCopySpec copy;
		copy.type = type;
		copy.function = function;
		if (bind_data) {
			copy.bind_data = bind_data->Copy();
		}
		copy.file_path = file_path;
		copy.use_tmp_file = use_tmp_file;
		copy.filename_pattern = filename_pattern;
		copy.file_extension = file_extension;
		copy.overwrite_mode = overwrite_mode;
		copy.parallel = parallel;
		copy.per_thread_output = per_thread_output;
		copy.file_size_bytes = file_size_bytes;
		copy.rotate = rotate;
		copy.return_type = return_type;
		copy.partition_output = partition_output;
		copy.write_partition_columns = write_partition_columns;
		copy.write_empty_file = write_empty_file;
		copy.hive_file_pattern = hive_file_pattern;
		copy.partition_columns = partition_columns;
		copy.names = names;
		copy.expected_types = expected_types;
		return copy;
	}
};

struct DistributedCopyFileInfo {
	// Historical name kept for compatibility:
	// - local FS distributed COPY: worker-local staging path
	// - direct-write distributed COPY: worker output path already at final file/object location
	std::string staging_path;
	std::string final_path;
	idx_t row_count = 0;
	idx_t file_size_bytes = 0;
	Value footer_size_bytes;
	Value column_statistics;
	Value partition_keys;
};

struct DistributedCopyResult {
	std::vector<DistributedCopyFileInfo> files;
	idx_t rows_copied = 0;
	idx_t staging_write_ms = 0;
	idx_t finalize_ms = 0;
	idx_t cleanup_ms = 0;
	std::string output_base_path;
	std::string output_run_id;
	std::string output_commit_dir;
	std::string output_manifest_path;
	std::string output_committed_marker_path;
	std::string output_lifecycle_path;
	bool output_direct_write = false;
	bool output_committed = false;
};

// ── Shared path utilities (used by scheduler + worker) ──────────────────────

inline bool IsDistributedCopyOutputPlaceholder(const std::string &path) {
	return StringUtil::StartsWith(path, DISTRIBUTED_COPY_OUTPUT_PLACEHOLDER_PREFIX);
}

inline std::string GetCopyBaseName(const DistributedCopySpec &spec) {
	auto base_name = StringUtil::GetFileName(spec.file_path);
	if (base_name.empty()) {
		base_name = "data";
	}
	if (!spec.file_extension.empty() && base_name.find('.') == std::string::npos) {
		if (spec.file_extension[0] != '.') {
			base_name += ".";
		}
		base_name += spec.file_extension;
	}
	return base_name;
}

inline bool CopySpecNeedsDirectory(const DistributedCopySpec &spec) {
	return spec.partition_output || spec.per_thread_output || spec.rotate;
}

inline std::string BuildCopyDirectWriteRunDirectory(const std::string &base_path, const std::string &run_id,
                                                    const std::string &separator = "/") {
	auto root = base_path;
	StringUtil::RTrim(root, separator);
	if (root.empty()) {
		root = ".";
	}
	if (run_id.empty()) {
		return root;
	}
	return root + separator + DISTRIBUTED_COPY_DIRECT_WRITE_RUN_PREFIX + run_id;
}

inline std::string BuildCopyDirectWriteTaskDirectory(const std::string &base_path, const std::string &run_id,
                                                     const std::string &worker_dir_name,
                                                     const std::string &separator = "/") {
	if (run_id.empty()) {
		auto root = base_path;
		StringUtil::RTrim(root, separator);
		if (root.empty()) {
			root = ".";
		}
		return root + separator + worker_dir_name;
	}
	return BuildCopyDirectWriteRunDirectory(base_path, run_id, separator) + separator + worker_dir_name;
}

inline std::string BuildCopyDirectTargetFileName(const std::string &run_id, const std::string &worker_dir_name,
                                                 const std::string &file_name) {
	return run_id + "_" + worker_dir_name + "_" + file_name;
}

inline std::string BuildCopyDirectTargetFilePath(const std::string &base_path, const std::string &run_id,
                                                 const std::string &worker_dir_name, const std::string &file_name,
                                                 const std::string &separator = "/") {
	auto root = base_path;
	StringUtil::RTrim(root, separator);
	if (root.empty()) {
		root = ".";
	}
	return root + separator + BuildCopyDirectTargetFileName(run_id, worker_dir_name, file_name);
}

inline std::string BuildCopyDirectTargetFilenamePattern(const std::string &run_id, const std::string &worker_dir_name) {
	return run_id + "_" + worker_dir_name + "_{i}";
}

inline bool CopyDirectTargetFileNameMatchesRun(const std::string &file_name, const std::string &run_id) {
	return !run_id.empty() && StringUtil::StartsWith(file_name, run_id + "_");
}

/// Build the placeholder path embedded in the cached fragment plan template.
inline std::string BuildCopyPlanTemplatePath(const DistributedCopySpec &spec, idx_t node_id) {
	auto placeholder_root =
	    std::string(DISTRIBUTED_COPY_OUTPUT_PLACEHOLDER_PREFIX) + "/node_" + std::to_string(node_id);
	if (spec.type == DistributedCopyType::BATCH_COPY_TO_FILE) {
		return placeholder_root + "/" + GetCopyBaseName(spec);
	}
	if (CopySpecNeedsDirectory(spec)) {
		return placeholder_root;
	}
	return placeholder_root + "/" + GetCopyBaseName(spec);
}

} // namespace distributed
} // namespace duckdb
