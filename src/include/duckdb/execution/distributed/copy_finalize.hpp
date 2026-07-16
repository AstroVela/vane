// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

// Standalone COPY finalize utilities — extracted from PlanRunner.
// Used by CopyFinishNode::finalize() and PlanRunner::run_copy_plan.
#pragma once

#include "duckdb/execution/distributed/copy_to_file.hpp"
#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/execution/distributed/utils/channel.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/hive_partitioning.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/main/client_context.hpp"

#include <chrono>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace duckdb {
namespace distributed {

inline idx_t DistributedCopyElapsedMillis(std::chrono::steady_clock::time_point started) {
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started);
	if (elapsed.count() <= 0) {
		return 0;
	}
	auto max_value = static_cast<unsigned long long>(std::numeric_limits<idx_t>::max());
	auto value = static_cast<unsigned long long>(elapsed.count());
	return static_cast<idx_t>(value > max_value ? max_value : value);
}

inline void ListDistributedCopyFilesRecursive(FileSystem &fs, const std::string &dir, std::vector<std::string> &out) {
	fs.ListFiles(dir, [&](const std::string &path, bool is_dir) {
		auto full_path = fs.JoinPath(dir, path);
		if (is_dir) {
			ListDistributedCopyFilesRecursive(fs, full_path, out);
		} else {
			out.push_back(full_path);
		}
	});
}

inline void RemoveDistributedCopyDirectoryTree(FileSystem &fs, const std::string &dir) {
	if (dir.empty()) {
		return;
	}
	try {
		if (!fs.DirectoryExists(dir)) {
			return;
		}
	} catch (...) {
		return;
	}

	std::vector<std::string> child_dirs;
	try {
		fs.ListFiles(dir, [&](const std::string &path, bool is_dir) {
			auto full_path = fs.JoinPath(dir, path);
			if (is_dir) {
				child_dirs.push_back(full_path);
				return;
			}
			try {
				fs.RemoveFile(full_path);
			} catch (...) {
			}
		});
	} catch (...) {
	}

	for (auto &child_dir : child_dirs) {
		RemoveDistributedCopyDirectoryTree(fs, child_dir);
	}

	try {
		fs.RemoveDirectory(dir);
	} catch (...) {
	}
}

inline void RemoveDistributedCopyDirectoryIfEmpty(FileSystem &fs, const std::string &dir) {
	if (dir.empty()) {
		return;
	}
	try {
		if (!fs.DirectoryExists(dir)) {
			return;
		}
		bool empty = true;
		fs.ListFiles(dir, [&](const std::string &, bool) { empty = false; });
		if (empty) {
			fs.RemoveDirectory(dir);
		}
	} catch (...) {
	}
}

inline void RemoveDistributedCopyDirectoryTree(const std::string &dir, ClientContext &context) {
	if (dir.empty()) {
		return;
	}
	auto &fs = FileSystem::GetFileSystem(context);
	RemoveDistributedCopyDirectoryTree(fs, dir);
}

struct DistributedCopyFinalizeCommitPaths {
	std::string commit_dir;
	std::string manifest_path;
	std::string committed_marker_path;
	std::string lifecycle_path;
};

inline bool DistributedCopyFileExistsNoThrow(FileSystem &fs, const std::string &path) {
	if (path.empty()) {
		return false;
	}
	try {
		return fs.FileExists(path);
	} catch (...) {
		return false;
	}
}

inline bool DistributedCopyDirectoryExistsNoThrow(FileSystem &fs, const std::string &path) {
	if (path.empty()) {
		return false;
	}
	try {
		return fs.DirectoryExists(path);
	} catch (...) {
		return false;
	}
}

inline DuckDBResult<idx_t> ParseDistributedCopyFinalizeIdx(const std::string &value, const std::string &field) {
	if (value.empty()) {
		return DuckDBResult<idx_t>::err(
		    DuckDBError::value_error("distributed COPY finalize manifest empty numeric field: " + field));
	}
	for (auto ch : value) {
		if (ch < '0' || ch > '9') {
			return DuckDBResult<idx_t>::err(
			    DuckDBError::value_error("distributed COPY finalize manifest invalid numeric field: " + field));
		}
	}
	try {
		size_t parsed_chars = 0;
		auto parsed = std::stoull(value, &parsed_chars);
		if (parsed_chars != value.size() ||
		    parsed > static_cast<unsigned long long>(std::numeric_limits<idx_t>::max())) {
			return DuckDBResult<idx_t>::err(
			    DuckDBError::value_error("distributed COPY finalize manifest invalid numeric field: " + field));
		}
		return DuckDBResult<idx_t>::ok(static_cast<idx_t>(parsed));
	} catch (const std::exception &) {
		return DuckDBResult<idx_t>::err(
		    DuckDBError::value_error("distributed COPY finalize manifest invalid numeric field: " + field));
	}
}

inline std::vector<std::string> SplitDistributedCopyFinalizeFields(const std::string &value) {
	std::vector<std::string> fields;
	size_t start = 0;
	while (start <= value.size()) {
		auto pos = value.find('\t', start);
		if (pos == std::string::npos) {
			fields.push_back(value.substr(start));
			break;
		}
		fields.push_back(value.substr(start, pos - start));
		start = pos + 1;
	}
	return fields;
}

inline DistributedCopyFinalizeCommitPaths
BuildDistributedCopyFinalizeCommitPaths(FileSystem &fs, const std::string &base_path, const std::string &run_id) {
	auto normalized_run_id = run_id;
	if (normalized_run_id.empty()) {
		normalized_run_id = "run";
	}

	DistributedCopyFinalizeCommitPaths paths;
	paths.commit_dir = fs.JoinPath(base_path + ".duckdb_commit", normalized_run_id);
	paths.manifest_path = fs.JoinPath(paths.commit_dir, "manifest.txt");
	paths.committed_marker_path = fs.JoinPath(paths.commit_dir, "committed");
	paths.lifecycle_path = fs.JoinPath(paths.commit_dir, DISTRIBUTED_COPY_DIRECT_WRITE_LIFECYCLE_FILE);
	return paths;
}

inline std::string DistributedCopyFinalizeRunIdFromStagingRoot(FileSystem &fs, const std::string &staging_root) {
	auto trimmed_staging_root = staging_root;
	StringUtil::RTrim(trimmed_staging_root, fs.PathSeparator(trimmed_staging_root));
	auto run_id = StringUtil::GetFileName(trimmed_staging_root);
	if (run_id.empty()) {
		run_id = "run";
	}
	return run_id;
}

inline DistributedCopyFinalizeCommitPaths
BuildDistributedCopyFinalizeCommitPathsFromStagingRoot(FileSystem &fs, const std::string &base_path,
                                                       const std::string &staging_root) {
	return BuildDistributedCopyFinalizeCommitPaths(fs, base_path,
	                                               DistributedCopyFinalizeRunIdFromStagingRoot(fs, staging_root));
}

inline void AttachDistributedCopyCommitInfo(DistributedCopyResult &result,
                                            const DistributedCopyFinalizeCommitPaths &paths,
                                            const std::string &base_path, const std::string &run_id, bool direct_write,
                                            bool committed) {
	result.output_base_path = base_path;
	result.output_run_id = run_id;
	result.output_commit_dir = paths.commit_dir;
	result.output_manifest_path = paths.manifest_path;
	result.output_committed_marker_path = paths.committed_marker_path;
	result.output_lifecycle_path = paths.lifecycle_path;
	result.output_direct_write = direct_write;
	result.output_committed = committed;
}

inline DuckDBResult<std::string> ReadDistributedCopyTextFile(FileSystem &fs, const std::string &path) {
	try {
		auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ);
		auto file_size = handle->GetFileSize();
		std::string contents;
		contents.resize(file_size);
		if (file_size > 0) {
			auto read_bytes = handle->Read(&contents[0], file_size);
			if (read_bytes < 0 || static_cast<idx_t>(read_bytes) != file_size) {
				return DuckDBResult<std::string>::err(
				    DuckDBError::io_error("failed to read distributed COPY finalize manifest: " + path));
			}
		}
		return DuckDBResult<std::string>::ok(std::move(contents));
	} catch (const std::exception &ex) {
		return DuckDBResult<std::string>::err(DuckDBError::io_error(
		    StringUtil::Format("failed to read distributed COPY finalize manifest \"%s\": %s", path, ex.what())));
	}
}

inline DuckDBResult<void> WriteDistributedCopyTextFileAtomically(FileSystem &fs, const std::string &path,
                                                                 const std::string &contents) {
	if (FileSystem::IsRemoteFile(path)) {
		try {
			fs.TryRemoveFile(path);
			auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE_NEW);
			if (!contents.empty()) {
				auto data = const_cast<char *>(contents.data());
				auto written = handle->Write(data, contents.size());
				if (written < 0 || static_cast<idx_t>(written) != contents.size()) {
					return DuckDBResult<void>::err(
					    DuckDBError::io_error("failed to write distributed COPY finalize file: " + path));
				}
			}
			handle->Sync();
			handle->Close();
			return DuckDBResult<void>::ok();
		} catch (const std::exception &ex) {
			return DuckDBResult<void>::err(DuckDBError::io_error(
			    StringUtil::Format("failed to commit distributed COPY finalize file \"%s\": %s", path, ex.what())));
		}
	}

	auto tmp_path = path + ".tmp";
	try {
		fs.TryRemoveFile(tmp_path);
		auto handle = fs.OpenFile(tmp_path, FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE_NEW);
		if (!contents.empty()) {
			auto data = const_cast<char *>(contents.data());
			auto written = handle->Write(data, contents.size());
			if (written < 0 || static_cast<idx_t>(written) != contents.size()) {
				return DuckDBResult<void>::err(
				    DuckDBError::io_error("failed to write distributed COPY finalize file: " + tmp_path));
			}
		}
		handle->Sync();
		handle->Close();
		fs.TryRemoveFile(path);
		fs.MoveFile(tmp_path, path);
		return DuckDBResult<void>::ok();
	} catch (const std::exception &ex) {
		try {
			fs.TryRemoveFile(tmp_path);
		} catch (...) {
		}
		return DuckDBResult<void>::err(DuckDBError::io_error(
		    StringUtil::Format("failed to commit distributed COPY finalize file \"%s\": %s", path, ex.what())));
	}
}

inline DuckDBResult<void> WriteDistributedCopyFinalizeManifest(FileSystem &fs,
                                                               const DistributedCopyFinalizeCommitPaths &paths,
                                                               const std::string &base_path,
                                                               const std::string &staging_root,
                                                               const std::vector<DistributedCopyFileInfo> &files) {
	try {
		fs.CreateDirectoriesRecursive(paths.commit_dir);
	} catch (const std::exception &ex) {
		return DuckDBResult<void>::err(DuckDBError::io_error(StringUtil::Format(
		    "failed to create distributed COPY finalize commit dir \"%s\": %s", paths.commit_dir, ex.what())));
	}

	idx_t rows_copied = 0;
	std::ostringstream manifest;
	manifest << "version=1\n";
	manifest << "base_path=" << base_path << "\n";
	manifest << "staging_root=" << staging_root << "\n";
	manifest << "file_count=" << files.size() << "\n";
	for (idx_t file_idx = 0; file_idx < files.size(); file_idx++) {
		const auto &file = files[file_idx];
		if (file.staging_path.empty() || file.final_path.empty()) {
			return DuckDBResult<void>::err(
			    DuckDBError::value_error("distributed COPY finalize manifest requires staging and final paths"));
		}
		rows_copied += file.row_count;
		manifest << "file=" << file_idx << "\t" << file.row_count << "\t" << file.file_size_bytes << "\t"
		         << file.staging_path << "\t" << file.final_path << "\n";
	}
	manifest << "rows_copied=" << rows_copied << "\n";
	return WriteDistributedCopyTextFileAtomically(fs, paths.manifest_path, manifest.str());
}

inline idx_t DistributedCopyCurrentEpochMillis() {
	auto now = std::chrono::system_clock::now().time_since_epoch();
	return static_cast<idx_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

inline DuckDBResult<void> WriteDistributedCopyDirectWriteLifecycle(FileSystem &fs, const std::string &base_path,
                                                                   const std::string &run_id,
                                                                   idx_t created_epoch_ms = 0) {
	if (base_path.empty()) {
		return DuckDBResult<void>::err(DuckDBError::value_error("direct-write lifecycle requires non-empty base_path"));
	}
	if (run_id.empty()) {
		return DuckDBResult<void>::err(DuckDBError::value_error("direct-write lifecycle requires non-empty run_id"));
	}

	auto paths = BuildDistributedCopyFinalizeCommitPaths(fs, base_path, run_id);
	if (DistributedCopyFileExistsNoThrow(fs, paths.committed_marker_path)) {
		return DuckDBResult<void>::ok();
	}
	if (created_epoch_ms == 0) {
		created_epoch_ms = DistributedCopyCurrentEpochMillis();
	}

	try {
		fs.CreateDirectoriesRecursive(paths.commit_dir);
	} catch (const std::exception &ex) {
		return DuckDBResult<void>::err(DuckDBError::io_error(
		    StringUtil::Format("failed to create direct-write lifecycle dir \"%s\": %s", paths.commit_dir, ex.what())));
	}

	auto direct_write_run_dir = BuildCopyDirectWriteRunDirectory(base_path, run_id, fs.PathSeparator(base_path));
	std::ostringstream lifecycle;
	lifecycle << "version=1\n";
	lifecycle << "mode=direct_write\n";
	lifecycle << "base_path=" << base_path << "\n";
	lifecycle << "run_id=" << run_id << "\n";
	lifecycle << "created_epoch_ms=" << created_epoch_ms << "\n";
	lifecycle << "direct_write_run_dir=" << direct_write_run_dir << "\n";
	return WriteDistributedCopyTextFileAtomically(fs, paths.lifecycle_path, lifecycle.str());
}

struct DistributedCopyDirectWriteLifecycleInfo {
	std::string base_path;
	std::string run_id;
	idx_t created_epoch_ms = 0;
};

inline DuckDBResult<DistributedCopyDirectWriteLifecycleInfo>
ReadDistributedCopyDirectWriteLifecycle(FileSystem &fs, const DistributedCopyFinalizeCommitPaths &paths,
                                        const std::string &expected_base_path, const std::string &expected_run_id) {
	if (!DistributedCopyFileExistsNoThrow(fs, paths.lifecycle_path)) {
		return DuckDBResult<DistributedCopyDirectWriteLifecycleInfo>::err(
		    DuckDBError::io_error("direct-write lifecycle missing: " + paths.lifecycle_path));
	}
	auto text_res = ReadDistributedCopyTextFile(fs, paths.lifecycle_path);
	if (text_res.is_err()) {
		return DuckDBResult<DistributedCopyDirectWriteLifecycleInfo>::err(text_res.error());
	}

	bool seen_version = false;
	bool seen_mode = false;
	bool seen_base_path = false;
	bool seen_run_id = false;
	bool seen_created_epoch_ms = false;
	DistributedCopyDirectWriteLifecycleInfo info;

	std::istringstream input(text_res.value());
	std::string line;
	idx_t line_no = 0;
	while (std::getline(input, line)) {
		line_no++;
		if (!line.empty() && line.back() == '\r') {
			line.pop_back();
		}
		if (line.empty()) {
			continue;
		}
		auto sep = line.find('=');
		if (sep == std::string::npos) {
			return DuckDBResult<DistributedCopyDirectWriteLifecycleInfo>::err(
			    DuckDBError::value_error("direct-write lifecycle invalid line at " + std::to_string(line_no)));
		}
		auto key = line.substr(0, sep);
		auto value = line.substr(sep + 1);
		if (key == "version") {
			if (seen_version) {
				return DuckDBResult<DistributedCopyDirectWriteLifecycleInfo>::err(
				    DuckDBError::value_error("direct-write lifecycle duplicate version"));
			}
			seen_version = true;
			if (value != "1") {
				return DuckDBResult<DistributedCopyDirectWriteLifecycleInfo>::err(
				    DuckDBError::value_error("unsupported direct-write lifecycle version: " + value));
			}
		} else if (key == "mode") {
			if (seen_mode) {
				return DuckDBResult<DistributedCopyDirectWriteLifecycleInfo>::err(
				    DuckDBError::value_error("direct-write lifecycle duplicate mode"));
			}
			seen_mode = true;
			if (value != "direct_write") {
				return DuckDBResult<DistributedCopyDirectWriteLifecycleInfo>::err(
				    DuckDBError::value_error("direct-write lifecycle mode mismatch"));
			}
		} else if (key == "base_path") {
			if (seen_base_path) {
				return DuckDBResult<DistributedCopyDirectWriteLifecycleInfo>::err(
				    DuckDBError::value_error("direct-write lifecycle duplicate base_path"));
			}
			seen_base_path = true;
			info.base_path = std::move(value);
			if (info.base_path != expected_base_path) {
				return DuckDBResult<DistributedCopyDirectWriteLifecycleInfo>::err(
				    DuckDBError::value_error("direct-write lifecycle base_path mismatch"));
			}
		} else if (key == "run_id") {
			if (seen_run_id) {
				return DuckDBResult<DistributedCopyDirectWriteLifecycleInfo>::err(
				    DuckDBError::value_error("direct-write lifecycle duplicate run_id"));
			}
			seen_run_id = true;
			info.run_id = std::move(value);
			if (info.run_id != expected_run_id) {
				return DuckDBResult<DistributedCopyDirectWriteLifecycleInfo>::err(
				    DuckDBError::value_error("direct-write lifecycle run_id mismatch"));
			}
		} else if (key == "created_epoch_ms") {
			if (seen_created_epoch_ms) {
				return DuckDBResult<DistributedCopyDirectWriteLifecycleInfo>::err(
				    DuckDBError::value_error("direct-write lifecycle duplicate created_epoch_ms"));
			}
			seen_created_epoch_ms = true;
			auto created_res = ParseDistributedCopyFinalizeIdx(value, "created_epoch_ms");
			if (created_res.is_err()) {
				return DuckDBResult<DistributedCopyDirectWriteLifecycleInfo>::err(created_res.error());
			}
			info.created_epoch_ms = created_res.value();
		} else if (key == "direct_write_run_dir") {
			continue;
		} else {
			return DuckDBResult<DistributedCopyDirectWriteLifecycleInfo>::err(
			    DuckDBError::value_error("direct-write lifecycle unknown field: " + key));
		}
	}

	if (!seen_version || !seen_mode || !seen_base_path || !seen_run_id || !seen_created_epoch_ms) {
		return DuckDBResult<DistributedCopyDirectWriteLifecycleInfo>::err(
		    DuckDBError::value_error("direct-write lifecycle missing required fields"));
	}
	return DuckDBResult<DistributedCopyDirectWriteLifecycleInfo>::ok(std::move(info));
}

inline DuckDBResult<DistributedCopyResult>
ReadDistributedCopyFinalizeManifest(FileSystem &fs, const DistributedCopyFinalizeCommitPaths &paths,
                                    const std::string &expected_base_path, const std::string &expected_staging_root,
                                    bool require_committed_marker) {
	if (require_committed_marker && !DistributedCopyFileExistsNoThrow(fs, paths.committed_marker_path)) {
		return DuckDBResult<DistributedCopyResult>::err(DuckDBError::invalid_state_error(
		    "distributed COPY finalize manifest is not committed: " + paths.manifest_path));
	}
	if (!DistributedCopyFileExistsNoThrow(fs, paths.manifest_path)) {
		return DuckDBResult<DistributedCopyResult>::err(
		    DuckDBError::io_error("distributed COPY finalize manifest missing: " + paths.manifest_path));
	}

	auto text_res = ReadDistributedCopyTextFile(fs, paths.manifest_path);
	if (text_res.is_err()) {
		return DuckDBResult<DistributedCopyResult>::err(text_res.error());
	}

	bool seen_version = false;
	bool seen_base_path = false;
	bool seen_staging_root = false;
	bool seen_file_count = false;
	bool seen_rows_copied = false;
	idx_t expected_file_count = 0;
	idx_t manifest_rows_copied = 0;

	DistributedCopyResult result;
	std::istringstream input(text_res.value());
	std::string line;
	idx_t line_no = 0;
	while (std::getline(input, line)) {
		line_no++;
		if (!line.empty() && line.back() == '\r') {
			line.pop_back();
		}
		if (line.empty()) {
			continue;
		}
		if (line.rfind("file=", 0) == 0) {
			auto fields = SplitDistributedCopyFinalizeFields(line.substr(5));
			if (fields.size() != 5) {
				return DuckDBResult<DistributedCopyResult>::err(DuckDBError::value_error(
				    "distributed COPY finalize manifest invalid file line at " + std::to_string(line_no)));
			}
			auto index_res = ParseDistributedCopyFinalizeIdx(fields[0], "file.index");
			if (index_res.is_err()) {
				return DuckDBResult<DistributedCopyResult>::err(index_res.error());
			}
			if (index_res.value() != result.files.size()) {
				return DuckDBResult<DistributedCopyResult>::err(
				    DuckDBError::value_error("distributed COPY finalize manifest file index order mismatch"));
			}
			auto rows_res = ParseDistributedCopyFinalizeIdx(fields[1], "file.rows");
			if (rows_res.is_err()) {
				return DuckDBResult<DistributedCopyResult>::err(rows_res.error());
			}
			auto size_res = ParseDistributedCopyFinalizeIdx(fields[2], "file.bytes");
			if (size_res.is_err()) {
				return DuckDBResult<DistributedCopyResult>::err(size_res.error());
			}
			if (fields[3].empty() || fields[4].empty()) {
				return DuckDBResult<DistributedCopyResult>::err(
				    DuckDBError::value_error("distributed COPY finalize manifest empty file path"));
			}

			DistributedCopyFileInfo info;
			info.row_count = rows_res.value();
			info.file_size_bytes = size_res.value();
			info.staging_path = std::move(fields[3]);
			info.final_path = std::move(fields[4]);
			result.rows_copied += info.row_count;
			result.files.push_back(std::move(info));
			continue;
		}

		auto sep = line.find('=');
		if (sep == std::string::npos) {
			return DuckDBResult<DistributedCopyResult>::err(DuckDBError::value_error(
			    "distributed COPY finalize manifest invalid line at " + std::to_string(line_no)));
		}
		auto key = line.substr(0, sep);
		auto value = line.substr(sep + 1);
		if (key == "version") {
			if (seen_version) {
				return DuckDBResult<DistributedCopyResult>::err(
				    DuckDBError::value_error("distributed COPY finalize manifest duplicate version"));
			}
			seen_version = true;
			if (value != "1") {
				return DuckDBResult<DistributedCopyResult>::err(
				    DuckDBError::value_error("unsupported distributed COPY finalize manifest version: " + value));
			}
		} else if (key == "base_path") {
			if (seen_base_path) {
				return DuckDBResult<DistributedCopyResult>::err(
				    DuckDBError::value_error("distributed COPY finalize manifest duplicate base_path"));
			}
			seen_base_path = true;
			if (value != expected_base_path) {
				return DuckDBResult<DistributedCopyResult>::err(
				    DuckDBError::value_error("distributed COPY finalize manifest base_path mismatch"));
			}
		} else if (key == "staging_root") {
			if (seen_staging_root) {
				return DuckDBResult<DistributedCopyResult>::err(
				    DuckDBError::value_error("distributed COPY finalize manifest duplicate staging_root"));
			}
			seen_staging_root = true;
			if (value != expected_staging_root) {
				return DuckDBResult<DistributedCopyResult>::err(
				    DuckDBError::value_error("distributed COPY finalize manifest staging_root mismatch"));
			}
		} else if (key == "file_count") {
			if (seen_file_count) {
				return DuckDBResult<DistributedCopyResult>::err(
				    DuckDBError::value_error("distributed COPY finalize manifest duplicate file_count"));
			}
			seen_file_count = true;
			auto count_res = ParseDistributedCopyFinalizeIdx(value, "file_count");
			if (count_res.is_err()) {
				return DuckDBResult<DistributedCopyResult>::err(count_res.error());
			}
			expected_file_count = count_res.value();
		} else if (key == "rows_copied") {
			if (seen_rows_copied) {
				return DuckDBResult<DistributedCopyResult>::err(
				    DuckDBError::value_error("distributed COPY finalize manifest duplicate rows_copied"));
			}
			seen_rows_copied = true;
			auto rows_res = ParseDistributedCopyFinalizeIdx(value, "rows_copied");
			if (rows_res.is_err()) {
				return DuckDBResult<DistributedCopyResult>::err(rows_res.error());
			}
			manifest_rows_copied = rows_res.value();
		} else {
			return DuckDBResult<DistributedCopyResult>::err(
			    DuckDBError::value_error("distributed COPY finalize manifest unknown field: " + key));
		}
	}

	if (!seen_version || !seen_base_path || !seen_staging_root || !seen_file_count || !seen_rows_copied) {
		return DuckDBResult<DistributedCopyResult>::err(
		    DuckDBError::value_error("distributed COPY finalize manifest missing required fields"));
	}
	if (expected_file_count != result.files.size()) {
		return DuckDBResult<DistributedCopyResult>::err(
		    DuckDBError::value_error("distributed COPY finalize manifest file_count mismatch"));
	}
	if (manifest_rows_copied != result.rows_copied) {
		return DuckDBResult<DistributedCopyResult>::err(
		    DuckDBError::value_error("distributed COPY finalize manifest rows_copied mismatch"));
	}
	const std::string direct_prefix = "direct:";
	const bool direct_write = StringUtil::StartsWith(expected_staging_root, direct_prefix);
	const std::string run_id = direct_write ? expected_staging_root.substr(direct_prefix.size())
	                                        : DistributedCopyFinalizeRunIdFromStagingRoot(fs, expected_staging_root);
	AttachDistributedCopyCommitInfo(result, paths, expected_base_path, run_id, direct_write, require_committed_marker);
	return DuckDBResult<DistributedCopyResult>::ok(std::move(result));
}

inline DuckDBResult<void> ValidateDistributedCopyFinalFile(FileSystem &fs, const DistributedCopyFileInfo &info) {
	try {
		if (info.final_path.empty() || !fs.FileExists(info.final_path)) {
			return DuckDBResult<void>::err(
			    DuckDBError::io_error("distributed COPY finalized file missing: " + info.final_path));
		}
		if (info.file_size_bytes > 0) {
			auto handle = fs.OpenFile(info.final_path, FileFlags::FILE_FLAGS_READ);
			auto actual_size = handle->GetFileSize();
			if (actual_size != info.file_size_bytes) {
				return DuckDBResult<void>::err(DuckDBError::io_error(
				    StringUtil::Format("distributed COPY finalized file size mismatch: \"%s\"", info.final_path)));
			}
		}
		return DuckDBResult<void>::ok();
	} catch (const std::exception &ex) {
		return DuckDBResult<void>::err(DuckDBError::io_error(StringUtil::Format(
		    "failed to validate distributed COPY finalized file \"%s\": %s", info.final_path, ex.what())));
	}
}

inline DuckDBResult<void> ValidateDistributedCopyDirectWriteFinalFile(FileSystem &fs,
                                                                      const DistributedCopyFileInfo &info) {
	if (FileSystem::IsRemoteFile(info.final_path) || DistributedCopyFileExistsNoThrow(fs, info.final_path)) {
		return ValidateDistributedCopyFinalFile(fs, info);
	}
	// Local direct-write output may live on another worker's node-local disk.
	// In that mode the coordinator cannot distinguish "remote node-local file"
	// from "missing local file", so commit trusts the worker COPY metadata.
	return DuckDBResult<void>::ok();
}

inline bool DistributedCopyPathIsInDirectory(const std::string &path, const std::string &directory,
                                             const std::string &separator) {
	if (path.empty() || directory.empty()) {
		return false;
	}
	auto normalized_directory = directory;
	StringUtil::RTrim(normalized_directory, separator);
	if (normalized_directory.empty()) {
		return false;
	}
	return path == normalized_directory || StringUtil::StartsWith(path, normalized_directory + separator);
}

inline bool DistributedCopyDirectWriteFinalPathBelongsToRun(FileSystem &fs, const std::string &base_path,
                                                            const std::string &run_id, const std::string &final_path) {
	if (base_path.empty() || run_id.empty() || final_path.empty()) {
		return false;
	}

	auto separator = fs.PathSeparator(base_path);
	auto direct_write_run_dir = BuildCopyDirectWriteRunDirectory(base_path, run_id, separator);
	if (DistributedCopyPathIsInDirectory(final_path, direct_write_run_dir, separator)) {
		return true;
	}

	if (!CopyDirectTargetFileNameMatchesRun(StringUtil::GetFileName(final_path), run_id)) {
		return false;
	}
	return DistributedCopyPathIsInDirectory(final_path, base_path, separator);
}

inline DuckDBResult<DistributedCopyResult>
ReadCommittedDistributedCopyDirectWriteResult(FileSystem &fs, const std::string &base_path, const std::string &run_id) {
	if (base_path.empty()) {
		return DuckDBResult<DistributedCopyResult>::err(
		    DuckDBError::value_error("direct-write committed reader requires non-empty base_path"));
	}
	if (run_id.empty()) {
		return DuckDBResult<DistributedCopyResult>::err(
		    DuckDBError::value_error("direct-write committed reader requires non-empty run_id"));
	}

	auto normalized_base_path = base_path;
	StringUtil::RTrim(normalized_base_path, fs.PathSeparator(normalized_base_path));
	auto commit_paths = BuildDistributedCopyFinalizeCommitPaths(fs, normalized_base_path, run_id);
	auto manifest_root = "direct:" + run_id;
	auto read_res = ReadDistributedCopyFinalizeManifest(fs, commit_paths, normalized_base_path, manifest_root, true);
	if (read_res.is_err()) {
		return DuckDBResult<DistributedCopyResult>::err(read_res.error());
	}

	auto result = std::move(read_res).value();
	for (const auto &info : result.files) {
		if (!DistributedCopyDirectWriteFinalPathBelongsToRun(fs, normalized_base_path, run_id, info.final_path)) {
			return DuckDBResult<DistributedCopyResult>::err(DuckDBError::invalid_state_error(StringUtil::Format(
			    "distributed COPY direct-write committed manifest file is outside run output: \"%s\"",
			    info.final_path)));
		}
		auto validate_res = ValidateDistributedCopyFinalFile(fs, info);
		if (validate_res.is_err()) {
			return DuckDBResult<DistributedCopyResult>::err(validate_res.error());
		}
	}
	return DuckDBResult<DistributedCopyResult>::ok(std::move(result));
}

inline DuckDBResult<void> MoveDistributedCopyFileOrReplay(FileSystem &fs, const DistributedCopyFileInfo &info) {
	auto parent_dir = StringUtil::GetFilePath(info.final_path);
	try {
		if (!parent_dir.empty() && !fs.DirectoryExists(parent_dir)) {
			fs.CreateDirectoriesRecursive(parent_dir);
		}
		if (fs.FileExists(info.final_path)) {
			return ValidateDistributedCopyFinalFile(fs, info);
		}
		if (!fs.FileExists(info.staging_path)) {
			return DuckDBResult<void>::err(DuckDBError::io_error(
			    StringUtil::Format("Distributed COPY worker output is not visible on the coordinator: "
			                       "worker_output_path \"%s\" for final path \"%s\" does not exist on this node, "
			                       "and no replayed final file is present.",
			                       info.staging_path, info.final_path)));
		}
		fs.MoveFile(info.staging_path, info.final_path);
		return ValidateDistributedCopyFinalFile(fs, info);
	} catch (const std::exception &ex) {
		return DuckDBResult<void>::err(DuckDBError::io_error(StringUtil::Format(
		    "Distributed COPY failed to finalize local output from worker_output_path \"%s\" to \"%s\": %s",
		    info.staging_path, info.final_path, ex.what())));
	}
}

inline DuckDBResult<void> WriteDistributedCopyFinalizeCommittedMarker(FileSystem &fs,
                                                                      const DistributedCopyFinalizeCommitPaths &paths) {
	try {
		fs.CreateDirectoriesRecursive(paths.commit_dir);
	} catch (const std::exception &ex) {
		return DuckDBResult<void>::err(DuckDBError::io_error(StringUtil::Format(
		    "failed to create distributed COPY finalize commit dir \"%s\": %s", paths.commit_dir, ex.what())));
	}
	return WriteDistributedCopyTextFileAtomically(fs, paths.committed_marker_path, "committed\n");
}

inline void
CleanupDistributedCopyDirectWriteUnselectedFiles(FileSystem &fs, const std::string &direct_write_run_dir,
                                                 const std::vector<DistributedCopyFileInfo> &selected_files) {
	if (direct_write_run_dir.empty()) {
		return;
	}

	try {
		if (!fs.DirectoryExists(direct_write_run_dir)) {
			return;
		}
	} catch (const std::exception &ex) {
		return;
	} catch (...) {
		return;
	}

	std::unordered_set<std::string> selected_paths;
	for (const auto &info : selected_files) {
		if (!info.final_path.empty()) {
			selected_paths.insert(info.final_path);
		}
	}

	std::vector<std::string> all_files;
	try {
		ListDistributedCopyFilesRecursive(fs, direct_write_run_dir, all_files);
	} catch (const std::exception &ex) {
		return;
	} catch (...) {
		return;
	}

	for (const auto &path : all_files) {
		if (selected_paths.find(path) != selected_paths.end()) {
			continue;
		}
		try {
			fs.RemoveFile(path);
		} catch (...) {
		}
	}
}

inline void
CleanupDistributedCopyDirectTargetUnselectedFiles(FileSystem &fs, const std::string &base_path,
                                                  const std::string &run_id,
                                                  const std::vector<DistributedCopyFileInfo> &selected_files) {
	if (base_path.empty() || run_id.empty()) {
		return;
	}

	std::unordered_set<std::string> selected_paths;
	std::unordered_set<std::string> scan_dirs;
	scan_dirs.insert(base_path);
	for (const auto &info : selected_files) {
		if (!info.final_path.empty()) {
			selected_paths.insert(info.final_path);
			auto parent = StringUtil::GetFilePath(info.final_path);
			if (!parent.empty()) {
				scan_dirs.insert(parent);
			}
		}
	}

	for (const auto &dir : scan_dirs) {
		try {
			if (!fs.DirectoryExists(dir)) {
				continue;
			}
		} catch (...) {
			continue;
		}

		std::vector<std::string> all_files;
		try {
			ListDistributedCopyFilesRecursive(fs, dir, all_files);
		} catch (...) {
			continue;
		}

		for (const auto &path : all_files) {
			if (selected_paths.find(path) != selected_paths.end()) {
				continue;
			}
			if (!CopyDirectTargetFileNameMatchesRun(StringUtil::GetFileName(path), run_id)) {
				continue;
			}
			try {
				fs.RemoveFile(path);
			} catch (...) {
			}
		}
	}
}

inline bool DistributedCopyUsesDirectTargetLayout(FileSystem &fs, const std::string &base_path,
                                                  const std::vector<DistributedCopyFileInfo> &files,
                                                  const std::string &run_id) {
	if (base_path.empty() || run_id.empty()) {
		return false;
	}
	auto separator = fs.PathSeparator(base_path);
	auto direct_write_run_dir = BuildCopyDirectWriteRunDirectory(base_path, run_id, separator);
	auto is_direct_target_path = [&](const std::string &path) {
		if (!CopyDirectTargetFileNameMatchesRun(StringUtil::GetFileName(path), run_id)) {
			return false;
		}
		if (!DistributedCopyPathIsInDirectory(path, base_path, separator)) {
			return false;
		}
		return !DistributedCopyPathIsInDirectory(path, direct_write_run_dir, separator);
	};

	for (const auto &info : files) {
		if (!info.final_path.empty() && is_direct_target_path(info.final_path)) {
			return true;
		}
		if (!info.staging_path.empty() && is_direct_target_path(info.staging_path)) {
			return true;
		}
	}
	return false;
}

struct DistributedCopyDirectWriteRunCleanupResult {
	bool skipped_committed = false;
	bool data_run_dir_existed = false;
	bool data_run_dir_removed = false;
	bool commit_dir_existed = false;
	bool commit_dir_removed = false;
};

inline DuckDBResult<DistributedCopyDirectWriteRunCleanupResult>
CleanupDistributedCopyUncommittedDirectWriteRun(FileSystem &fs, const std::string &base_path,
                                                const std::string &run_id) {
	if (base_path.empty()) {
		return DuckDBResult<DistributedCopyDirectWriteRunCleanupResult>::err(
		    DuckDBError::value_error("direct-write cleanup requires non-empty base_path"));
	}
	if (run_id.empty()) {
		return DuckDBResult<DistributedCopyDirectWriteRunCleanupResult>::err(
		    DuckDBError::value_error("direct-write cleanup requires non-empty run_id"));
	}

	DistributedCopyDirectWriteRunCleanupResult result;
	auto commit_paths = BuildDistributedCopyFinalizeCommitPaths(fs, base_path, run_id);
	if (DistributedCopyFileExistsNoThrow(fs, commit_paths.committed_marker_path)) {
		result.skipped_committed = true;
		return DuckDBResult<DistributedCopyDirectWriteRunCleanupResult>::ok(std::move(result));
	}

	auto direct_write_run_dir = BuildCopyDirectWriteRunDirectory(base_path, run_id, fs.PathSeparator(base_path));
	result.data_run_dir_existed = DistributedCopyDirectoryExistsNoThrow(fs, direct_write_run_dir);
	if (result.data_run_dir_existed) {
		RemoveDistributedCopyDirectoryTree(fs, direct_write_run_dir);
		if (DistributedCopyDirectoryExistsNoThrow(fs, direct_write_run_dir)) {
			return DuckDBResult<DistributedCopyDirectWriteRunCleanupResult>::err(
			    DuckDBError::io_error("failed to cleanup direct-write run dir: " + direct_write_run_dir));
		}
		result.data_run_dir_removed = true;
	}

	CleanupDistributedCopyDirectTargetUnselectedFiles(fs, base_path, run_id, {});

	result.commit_dir_existed = DistributedCopyDirectoryExistsNoThrow(fs, commit_paths.commit_dir);
	if (result.commit_dir_existed) {
		RemoveDistributedCopyDirectoryTree(fs, commit_paths.commit_dir);
		if (DistributedCopyDirectoryExistsNoThrow(fs, commit_paths.commit_dir)) {
			return DuckDBResult<DistributedCopyDirectWriteRunCleanupResult>::err(
			    DuckDBError::io_error("failed to cleanup direct-write commit dir: " + commit_paths.commit_dir));
		}
		result.commit_dir_removed = true;
		RemoveDistributedCopyDirectoryIfEmpty(fs, StringUtil::GetFilePath(commit_paths.commit_dir));
	}

	return DuckDBResult<DistributedCopyDirectWriteRunCleanupResult>::ok(std::move(result));
}

struct DistributedCopyDirectWriteCleanupScanResult {
	idx_t scanned_runs = 0;
	idx_t cleaned_runs = 0;
	idx_t committed_runs = 0;
	idx_t active_runs = 0;
	idx_t skipped_unregistered_runs = 0;
	idx_t errors = 0;
	std::vector<std::string> cleaned_run_ids;
	std::vector<std::string> error_messages;
};

inline DuckDBResult<DistributedCopyDirectWriteCleanupScanResult>
CleanupExpiredDistributedCopyDirectWriteRuns(FileSystem &fs, const std::string &base_path, idx_t min_age_ms,
                                             idx_t now_epoch_ms = 0) {
	if (base_path.empty()) {
		return DuckDBResult<DistributedCopyDirectWriteCleanupScanResult>::err(
		    DuckDBError::value_error("direct-write cleanup scan requires non-empty base_path"));
	}
	if (now_epoch_ms == 0) {
		now_epoch_ms = DistributedCopyCurrentEpochMillis();
	}

	DistributedCopyDirectWriteCleanupScanResult result;
	auto commit_root = base_path + ".duckdb_commit";
	if (!DistributedCopyDirectoryExistsNoThrow(fs, commit_root)) {
		return DuckDBResult<DistributedCopyDirectWriteCleanupScanResult>::ok(std::move(result));
	}

	std::vector<std::string> run_ids;
	try {
		fs.ListFiles(commit_root, [&](const std::string &path, bool is_dir) {
			if (!is_dir) {
				return;
			}
			auto run_id = StringUtil::GetFileName(path);
			if (run_id.empty()) {
				run_id = path;
				StringUtil::RTrim(run_id, fs.PathSeparator(run_id));
				run_id = StringUtil::GetFileName(run_id);
			}
			if (!run_id.empty()) {
				run_ids.push_back(run_id);
			}
		});
	} catch (const std::exception &ex) {
		return DuckDBResult<DistributedCopyDirectWriteCleanupScanResult>::err(DuckDBError::io_error(
		    StringUtil::Format("failed to list direct-write commit root \"%s\": %s", commit_root, ex.what())));
	}

	for (const auto &run_id : run_ids) {
		result.scanned_runs++;
		auto paths = BuildDistributedCopyFinalizeCommitPaths(fs, base_path, run_id);
		if (DistributedCopyFileExistsNoThrow(fs, paths.committed_marker_path)) {
			result.committed_runs++;
			continue;
		}
		if (!DistributedCopyFileExistsNoThrow(fs, paths.lifecycle_path)) {
			result.skipped_unregistered_runs++;
			continue;
		}
		auto lifecycle_res = ReadDistributedCopyDirectWriteLifecycle(fs, paths, base_path, run_id);
		if (lifecycle_res.is_err()) {
			result.errors++;
			result.error_messages.push_back(lifecycle_res.error().what());
			continue;
		}
		const auto &lifecycle = lifecycle_res.value();
		if (now_epoch_ms < lifecycle.created_epoch_ms || now_epoch_ms - lifecycle.created_epoch_ms < min_age_ms) {
			result.active_runs++;
			continue;
		}

		auto cleanup_res = CleanupDistributedCopyUncommittedDirectWriteRun(fs, base_path, run_id);
		if (cleanup_res.is_err()) {
			result.errors++;
			result.error_messages.push_back(cleanup_res.error().what());
			continue;
		}
		if (cleanup_res.value().skipped_committed) {
			result.committed_runs++;
			continue;
		}
		result.cleaned_runs++;
		result.cleaned_run_ids.push_back(run_id);
	}

	return DuckDBResult<DistributedCopyDirectWriteCleanupScanResult>::ok(std::move(result));
}

/// Parse ColumnDataResultPartitions (worker COPY output) into DistributedCopyFileInfo structs.
inline DuckDBResult<std::vector<DistributedCopyFileInfo>>
ParseCopyPartitions(const std::vector<ResultPartitionRef> &parts) {
	std::vector<DistributedCopyFileInfo> files;
	idx_t part_idx = 0;
	for (auto &part : parts) {
		auto collection_ref = part ? part->to_column_data() : nullptr;
		if (!collection_ref) {
			return DuckDBResult<std::vector<DistributedCopyFileInfo>>::err(
			    DuckDBError("Distributed COPY expects tabular ResultPartition results"));
		}
		auto &collection = *collection_ref;
		ColumnDataScanState scan_state;
		collection.InitializeScan(scan_state);
		DataChunk chunk;
		collection.InitializeScanChunk(chunk);

		while (collection.Scan(scan_state, chunk)) {
			if (chunk.ColumnCount() < 6) {
				return DuckDBResult<std::vector<DistributedCopyFileInfo>>::err(
				    DuckDBError("Distributed COPY result schema mismatch"));
			}
			for (idx_t row = 0; row < chunk.size(); row++) {
				DistributedCopyFileInfo info;
				auto file_val = chunk.GetValue(0, row);
				if (file_val.IsNull()) {
					return DuckDBResult<std::vector<DistributedCopyFileInfo>>::err(
					    DuckDBError("Distributed COPY result missing file path"));
				}
				info.staging_path = file_val.GetValue<std::string>();
				auto row_val = chunk.GetValue(1, row);
				auto size_val = chunk.GetValue(2, row);
				info.row_count = row_val.IsNull() ? 0 : static_cast<idx_t>(row_val.GetValue<uint64_t>());
				info.file_size_bytes = size_val.IsNull() ? 0 : static_cast<idx_t>(size_val.GetValue<uint64_t>());
				info.footer_size_bytes = chunk.GetValue(3, row);
				info.column_statistics = chunk.GetValue(4, row);
				info.partition_keys = chunk.GetValue(5, row);
				files.push_back(std::move(info));
			}
		}
		part_idx++;
	}
	return DuckDBResult<std::vector<DistributedCopyFileInfo>>::ok(std::move(files));
}

/// Finalize: assign final paths and rename worker-local staging files to their
/// destination. When staging_root is empty, workers already wrote directly to
/// the requested output base (remote/object storage and local paths by default)
/// and MoveFile is skipped. Shared local filesystems can still use staging +
/// MoveFile/rename by passing a non-empty staging_root.
inline DuckDBResult<DistributedCopyResult> FinalizeCopyFiles(const DistributedCopySpec &spec,
                                                             const std::string &staging_root,
                                                             std::vector<DistributedCopyFileInfo> files,
                                                             ClientContext &context,
                                                             std::string direct_write_run_id = std::string()) {
	auto finalize_started = std::chrono::steady_clock::now();
	const bool skip_move = staging_root.empty();

	DistributedCopyResult result;
	result.files = std::move(files);
	auto &fs = FileSystem::GetFileSystem(context);

	std::string base_path = spec.file_path;
	StringUtil::RTrim(base_path, fs.PathSeparator(base_path));
	if (spec.use_tmp_file) {
		auto base = StringUtil::GetFileName(base_path);
		auto dir = StringUtil::GetFilePath(base_path);
		if (base.rfind("tmp_", 0) == 0) {
			base = base.substr(4);
		}
		base_path = dir.empty() ? base : fs.JoinPath(dir, base);
	}

	const bool output_is_dir = spec.partition_output || spec.per_thread_output || spec.rotate;

	DistributedCopyFinalizeCommitPaths commit_paths;
	bool replaying_finalize_manifest = false;
	bool replaying_direct_write_manifest = false;
	const bool direct_write_commit_enabled = skip_move && !direct_write_run_id.empty();
	std::string direct_write_manifest_root;
	std::string direct_write_cleanup_run_dir;
	if (!skip_move) {
		commit_paths = BuildDistributedCopyFinalizeCommitPathsFromStagingRoot(fs, base_path, staging_root);

		if (DistributedCopyFileExistsNoThrow(fs, commit_paths.committed_marker_path)) {
			auto committed_res = ReadDistributedCopyFinalizeManifest(fs, commit_paths, base_path, staging_root, true);
			if (committed_res.is_err()) {
				return DuckDBResult<DistributedCopyResult>::err(committed_res.error());
			}
			auto committed_result = std::move(committed_res).value();
			for (const auto &info : committed_result.files) {
				auto validate_res = ValidateDistributedCopyFinalFile(fs, info);
				if (validate_res.is_err()) {
					return DuckDBResult<DistributedCopyResult>::err(validate_res.error());
				}
			}
			committed_result.finalize_ms = DistributedCopyElapsedMillis(finalize_started);
			auto cleanup_started = std::chrono::steady_clock::now();
			RemoveDistributedCopyDirectoryTree(fs, staging_root);
			RemoveDistributedCopyDirectoryIfEmpty(fs, StringUtil::GetFilePath(staging_root));
			committed_result.cleanup_ms = DistributedCopyElapsedMillis(cleanup_started);
			return DuckDBResult<DistributedCopyResult>::ok(std::move(committed_result));
		}

		if (DistributedCopyFileExistsNoThrow(fs, commit_paths.manifest_path)) {
			auto manifest_res = ReadDistributedCopyFinalizeManifest(fs, commit_paths, base_path, staging_root, false);
			if (manifest_res.is_err()) {
				return DuckDBResult<DistributedCopyResult>::err(manifest_res.error());
			}
			result = std::move(manifest_res).value();
			replaying_finalize_manifest = true;
		} else {
			for (const auto &info : result.files) {
				try {
					if (!fs.FileExists(info.staging_path)) {
						return DuckDBResult<DistributedCopyResult>::err(DuckDBError(
						    StringUtil::Format("Distributed COPY worker output is not visible on the coordinator: "
						                       "worker_output_path \"%s\" does not exist. "
						                       "Finalize aborted before moving any final output.",
						                       info.staging_path)));
					}
				} catch (const std::exception &ex) {
					return DuckDBResult<DistributedCopyResult>::err(DuckDBError(
					    StringUtil::Format("Distributed COPY failed to preflight worker_output_path \"%s\": %s",
					                       info.staging_path, ex.what())));
				}
			}
		}
	} else if (direct_write_commit_enabled) {
		commit_paths = BuildDistributedCopyFinalizeCommitPaths(fs, base_path, direct_write_run_id);
		direct_write_manifest_root = "direct:" + direct_write_run_id;

		if (DistributedCopyFileExistsNoThrow(fs, commit_paths.committed_marker_path)) {
			auto committed_res =
			    ReadDistributedCopyFinalizeManifest(fs, commit_paths, base_path, direct_write_manifest_root, true);
			if (committed_res.is_err()) {
				return DuckDBResult<DistributedCopyResult>::err(committed_res.error());
			}
			auto committed_result = std::move(committed_res).value();
			for (const auto &info : committed_result.files) {
				auto validate_res = ValidateDistributedCopyDirectWriteFinalFile(fs, info);
				if (validate_res.is_err()) {
					return DuckDBResult<DistributedCopyResult>::err(validate_res.error());
				}
			}
			committed_result.finalize_ms = DistributedCopyElapsedMillis(finalize_started);
			auto direct_write_run_dir =
			    BuildCopyDirectWriteRunDirectory(base_path, direct_write_run_id, fs.PathSeparator(base_path));
			auto cleanup_started = std::chrono::steady_clock::now();
			CleanupDistributedCopyDirectWriteUnselectedFiles(fs, direct_write_run_dir, committed_result.files);
			if (DistributedCopyUsesDirectTargetLayout(fs, base_path, committed_result.files, direct_write_run_id)) {
				CleanupDistributedCopyDirectTargetUnselectedFiles(fs, base_path, direct_write_run_id,
				                                                  committed_result.files);
			}
			committed_result.cleanup_ms = DistributedCopyElapsedMillis(cleanup_started);
			return DuckDBResult<DistributedCopyResult>::ok(std::move(committed_result));
		}

		if (DistributedCopyFileExistsNoThrow(fs, commit_paths.manifest_path)) {
			auto manifest_res =
			    ReadDistributedCopyFinalizeManifest(fs, commit_paths, base_path, direct_write_manifest_root, false);
			if (manifest_res.is_err()) {
				return DuckDBResult<DistributedCopyResult>::err(manifest_res.error());
			}
			result = std::move(manifest_res).value();
			replaying_direct_write_manifest = true;
		}
	}

	auto check_directory = [&](const std::string &dir) {
		if (!fs.DirectoryExists(dir)) {
			fs.CreateDirectoriesRecursive(dir);
			return;
		}
		if (spec.overwrite_mode == CopyOverwriteMode::COPY_OVERWRITE_OR_IGNORE ||
		    spec.overwrite_mode == CopyOverwriteMode::COPY_APPEND) {
			return;
		}
		if (fs.IsRemoteFile(dir) && spec.overwrite_mode == CopyOverwriteMode::COPY_OVERWRITE) {
			throw NotImplementedException("OVERWRITE is not supported for remote file systems");
		}
		std::vector<std::string> existing_files;
		ListDistributedCopyFilesRecursive(fs, dir, existing_files);
		if (existing_files.empty()) {
			return;
		}
		if (spec.overwrite_mode == CopyOverwriteMode::COPY_OVERWRITE) {
			for (auto &file : existing_files) {
				fs.RemoveFile(file);
			}
			return;
		}
		throw IOException("Directory \"%s\" is not empty! Enable OVERWRITE option to overwrite files", dir);
	};

	// Direct-write workers have already created files under base_path by the
	// time finalize runs, so the local staging path is the only mode where the
	// coordinator can still enforce an empty destination directory here.
	if (output_is_dir && !skip_move && !replaying_finalize_manifest) {
		try {
			check_directory(base_path);
		} catch (const std::exception &ex) {
			return DuckDBResult<DistributedCopyResult>::err(DuckDBError(ex.what()));
		}
	}

	auto parse_partition_keys = [](const Value &map_val) {
		std::unordered_map<std::string, Value> kv;
		if (map_val.IsNull()) {
			return kv;
		}
		auto entries = MapValue::GetChildren(map_val);
		for (auto &entry : entries) {
			auto &children = StructValue::GetChildren(entry);
			if (children.size() < 2) {
				continue;
			}
			auto key = children[0].ToString();
			kv.emplace(key, children[1]);
		}
		return kv;
	};

	std::unordered_map<std::string, idx_t> partition_offsets;
	idx_t global_offset = 0;

	auto make_partition_key = [&](const std::unordered_map<std::string, Value> &kv) {
		std::string key;
		if (!spec.partition_columns.empty()) {
			for (auto col_idx : spec.partition_columns) {
				if (col_idx >= spec.names.size()) {
					continue;
				}
				const auto &name = spec.names[col_idx];
				auto it = kv.find(name);
				if (it == kv.end()) {
					continue;
				}
				key += name;
				key += "=";
				key += it->second.ToString();
				key += "|";
			}
		}
		return key;
	};

	if (skip_move) {
		if (!replaying_direct_write_manifest) {
			for (auto &info : result.files) {
				info.final_path = info.staging_path;
			}
		}

		std::unordered_set<std::string> planned_final_paths;
		result.rows_copied = 0;
		for (const auto &info : result.files) {
			if (!planned_final_paths.insert(info.final_path).second) {
				return DuckDBResult<DistributedCopyResult>::err(DuckDBError(StringUtil::Format(
				    "Distributed COPY direct-write finalize contains duplicate final path \"%s\"", info.final_path)));
			}

			auto validate_res = ValidateDistributedCopyDirectWriteFinalFile(fs, info);
			if (validate_res.is_err()) {
				return DuckDBResult<DistributedCopyResult>::err(validate_res.error());
			}
			result.rows_copied += info.row_count;
		}

		if (direct_write_commit_enabled) {
			if (!replaying_direct_write_manifest) {
				auto manifest_res = WriteDistributedCopyFinalizeManifest(fs, commit_paths, base_path,
				                                                         direct_write_manifest_root, result.files);
				if (manifest_res.is_err()) {
					return DuckDBResult<DistributedCopyResult>::err(manifest_res.error());
				}
			}
			auto marker_res = WriteDistributedCopyFinalizeCommittedMarker(fs, commit_paths);
			if (marker_res.is_err()) {
				return DuckDBResult<DistributedCopyResult>::err(marker_res.error());
			}
			AttachDistributedCopyCommitInfo(result, commit_paths, base_path, direct_write_run_id, true, true);
			direct_write_cleanup_run_dir =
			    BuildCopyDirectWriteRunDirectory(base_path, direct_write_run_id, fs.PathSeparator(base_path));
		}
	} else {
		std::unordered_set<std::string> planned_final_paths;
		if (!replaying_finalize_manifest) {
			std::unordered_set<std::string> created_dirs;
			for (auto &info : result.files) {
				auto partition_kv = parse_partition_keys(info.partition_keys);
				std::string target_dir = base_path;
				if (spec.partition_output && spec.hive_file_pattern) {
					for (auto col_idx : spec.partition_columns) {
						if (col_idx >= spec.names.size()) {
							continue;
						}
						const auto &name = spec.names[col_idx];
						auto it = partition_kv.find(name);
						if (it == partition_kv.end()) {
							return DuckDBResult<DistributedCopyResult>::err(
							    DuckDBError(StringUtil::Format("Missing partition key \"%s\" in COPY result", name)));
						}
						const auto value_str = it->second.ToString();
						std::string part_dir =
						    HivePartitioning::Escape(name) + "=" + HivePartitioning::Escape(value_str);
						target_dir = fs.JoinPath(target_dir, part_dir);
					}
				}

				idx_t offset = 0;
				if (spec.partition_output && spec.hive_file_pattern) {
					auto key = make_partition_key(partition_kv);
					offset = partition_offsets[key]++;
				} else {
					offset = global_offset++;
				}
				info.final_path = spec.filename_pattern.CreateFilename(fs, target_dir, spec.file_extension, offset);

				auto parent_dir = StringUtil::GetFilePath(info.final_path);
				if (!parent_dir.empty() && created_dirs.insert(parent_dir).second) {
					if (!fs.DirectoryExists(parent_dir)) {
						fs.CreateDirectoriesRecursive(parent_dir);
					}
				}

				if (spec.overwrite_mode == CopyOverwriteMode::COPY_APPEND) {
					while (fs.FileExists(info.final_path)) {
						if (!spec.filename_pattern.HasUUID()) {
							return DuckDBResult<DistributedCopyResult>::err(
							    DuckDBError("COPY_APPEND requires {uuid} in filename_pattern when file exists"));
						}
						info.final_path =
						    spec.filename_pattern.CreateFilename(fs, target_dir, spec.file_extension, offset);
					}
				} else if (spec.overwrite_mode == CopyOverwriteMode::COPY_OVERWRITE) {
					if (fs.FileExists(info.final_path)) {
						fs.RemoveFile(info.final_path);
					}
				} else if (spec.overwrite_mode == CopyOverwriteMode::COPY_ERROR_ON_CONFLICT && output_is_dir) {
					if (fs.FileExists(info.final_path)) {
						return DuckDBResult<DistributedCopyResult>::err(DuckDBError(
						    StringUtil::Format("Cannot write to \"%s\" - file already exists", info.final_path)));
					}
				}

				if (!planned_final_paths.insert(info.final_path).second) {
					return DuckDBResult<DistributedCopyResult>::err(DuckDBError(StringUtil::Format(
					    "Distributed COPY finalize generated duplicate final path \"%s\"", info.final_path)));
				}
			}

			auto manifest_res =
			    WriteDistributedCopyFinalizeManifest(fs, commit_paths, base_path, staging_root, result.files);
			if (manifest_res.is_err()) {
				return DuckDBResult<DistributedCopyResult>::err(manifest_res.error());
			}
		} else {
			for (const auto &info : result.files) {
				if (!planned_final_paths.insert(info.final_path).second) {
					return DuckDBResult<DistributedCopyResult>::err(DuckDBError(StringUtil::Format(
					    "Distributed COPY finalize manifest contains duplicate final path \"%s\"", info.final_path)));
				}
			}
		}

		result.rows_copied = 0;
		for (const auto &info : result.files) {
			auto move_res = MoveDistributedCopyFileOrReplay(fs, info);
			if (move_res.is_err()) {
				return DuckDBResult<DistributedCopyResult>::err(move_res.error());
			}
			result.rows_copied += info.row_count;
		}

		auto marker_res = WriteDistributedCopyFinalizeCommittedMarker(fs, commit_paths);
		if (marker_res.is_err()) {
			return DuckDBResult<DistributedCopyResult>::err(marker_res.error());
		}
		AttachDistributedCopyCommitInfo(result, commit_paths, base_path,
		                                DistributedCopyFinalizeRunIdFromStagingRoot(fs, staging_root), false, true);
	}

	result.finalize_ms = DistributedCopyElapsedMillis(finalize_started);
	if (!direct_write_cleanup_run_dir.empty()) {
		auto cleanup_started = std::chrono::steady_clock::now();
		CleanupDistributedCopyDirectWriteUnselectedFiles(fs, direct_write_cleanup_run_dir, result.files);
		if (DistributedCopyUsesDirectTargetLayout(fs, base_path, result.files, direct_write_run_id)) {
			CleanupDistributedCopyDirectTargetUnselectedFiles(fs, base_path, direct_write_run_id, result.files);
		}
		result.cleanup_ms += DistributedCopyElapsedMillis(cleanup_started);
	}
	if (!staging_root.empty()) {
		auto cleanup_started = std::chrono::steady_clock::now();
		RemoveDistributedCopyDirectoryTree(fs, staging_root);
		RemoveDistributedCopyDirectoryIfEmpty(fs, StringUtil::GetFilePath(staging_root));
		result.cleanup_ms += DistributedCopyElapsedMillis(cleanup_started);
	}

	return DuckDBResult<DistributedCopyResult>::ok(std::move(result));
}

} // namespace distributed
} // namespace duckdb
