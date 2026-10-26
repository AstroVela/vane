// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "ray_binding_internal.hpp"

namespace duckdb {
namespace ray_bindings {
namespace py = ::duckdb::py;

void RegisterRayMiscTestsBindings(py::module_ &m) {
	m.def(
	    "distributed_copy_direct_write_committed_reader_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;
		    auto &fs = FileSystem::GetFileSystem(context);

		    const std::string run_id = "run-reader";
		    auto final_root = local_dir + "/copy_direct_reader";
		    auto selected_dir = BuildCopyDirectWriteTaskDirectory(final_root, run_id, "w_selected");
		    auto loser_dir = BuildCopyDirectWriteTaskDirectory(final_root, run_id, "w_loser");
		    auto selected_file = selected_dir + "/part.parquet";
		    auto loser_file = loser_dir + "/part.parquet";

		    auto write_file = [&](const std::string &path, const std::string &body) {
			    auto parent = StringUtil::GetFilePath(path);
			    if (!parent.empty() && !fs.DirectoryExists(parent)) {
				    fs.CreateDirectoriesRecursive(parent);
			    }
			    std::ofstream output(path, std::ios::out | std::ios::binary);
			    output << body;
			    output.close();
			    if (!output) {
				    throw std::runtime_error("failed to write test file: " + path);
			    }
		    };

		    const std::string selected_body = "selected";
		    const std::string loser_body = "loser";
		    write_file(selected_file, selected_body);
		    write_file(loser_file, loser_body);

		    DistributedCopyFileInfo selected_info;
		    selected_info.staging_path = selected_file;
		    selected_info.final_path = selected_file;
		    selected_info.row_count = 7;
		    selected_info.file_size_bytes = selected_body.size();
		    vector<DistributedCopyFileInfo> selected_files;
		    selected_files.push_back(std::move(selected_info));

		    auto commit_paths = BuildDistributedCopyFinalizeCommitPaths(fs, final_root, run_id);
		    auto lifecycle_res = WriteDistributedCopyDirectWriteLifecycle(fs, final_root, run_id);
		    if (lifecycle_res.is_err()) {
			    throw std::runtime_error(lifecycle_res.error().what());
		    }
		    auto manifest_res =
		        WriteDistributedCopyFinalizeManifest(fs, commit_paths, final_root, "direct:" + run_id, selected_files);
		    if (manifest_res.is_err()) {
			    throw std::runtime_error(manifest_res.error().what());
		    }

		    auto uncommitted_read = ReadCommittedDistributedCopyDirectWriteResult(fs, final_root, run_id);
		    auto marker_res = WriteDistributedCopyFinalizeCommittedMarker(fs, commit_paths);
		    if (marker_res.is_err()) {
			    throw std::runtime_error(marker_res.error().what());
		    }
		    auto committed_read = ReadCommittedDistributedCopyDirectWriteResult(fs, final_root, run_id);

		    bool committed_contains_loser = false;
		    std::string committed_file_path;
		    idx_t committed_rows = 0;
		    idx_t committed_file_count = 0;
		    if (committed_read.is_ok()) {
			    const auto &result = committed_read.value();
			    committed_rows = result.rows_copied;
			    committed_file_count = result.files.size();
			    for (const auto &info : result.files) {
				    if (info.final_path == loser_file) {
					    committed_contains_loser = true;
				    }
			    }
			    if (!result.files.empty()) {
				    committed_file_path = result.files[0].final_path;
			    }
		    }

		    py::dict out;
		    out["base_path"] = final_root;
		    out["run_id"] = run_id;
		    out["manifest_exists"] = fs.FileExists(commit_paths.manifest_path);
		    out["marker_exists"] = fs.FileExists(commit_paths.committed_marker_path);
		    out["selected_file_exists"] = fs.FileExists(selected_file);
		    out["loser_file_exists"] = fs.FileExists(loser_file);
		    out["uncommitted_error"] = uncommitted_read.is_err();
		    out["uncommitted_error_message"] = uncommitted_read.is_err() ? uncommitted_read.error().what() : "";
		    out["committed_error"] = committed_read.is_err();
		    out["committed_error_message"] = committed_read.is_err() ? committed_read.error().what() : "";
		    out["committed_rows"] = committed_rows;
		    out["committed_file_count"] = committed_file_count;
		    out["committed_file_path"] = committed_file_path;
		    out["committed_contains_loser"] = committed_contains_loser;
		    return out;
	    },
	    py::arg("local_dir"),
	    "Exercise COPY direct-write committed-reader visibility through committed manifest only.");

	m.def(
	    "distributed_copy_direct_write_uncommitted_stale_cleanup_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;
		    auto &fs = FileSystem::GetFileSystem(context);

		    auto final_root = local_dir + "/copy_direct_stale_cleanup";
		    const std::string stale_run_id = "run-stale";
		    const std::string committed_run_id = "run-committed";

		    auto write_file = [&](const std::string &path, const std::string &body) {
			    auto parent = StringUtil::GetFilePath(path);
			    if (!parent.empty() && !fs.DirectoryExists(parent)) {
				    fs.CreateDirectoriesRecursive(parent);
			    }
			    std::ofstream output(path, std::ios::out | std::ios::binary);
			    output << body;
			    output.close();
			    if (!output) {
				    throw std::runtime_error("failed to write test file: " + path);
			    }
		    };

		    auto make_file_info = [](const std::string &path, idx_t rows, idx_t bytes) {
			    DistributedCopyFileInfo info;
			    info.staging_path = path;
			    info.final_path = path;
			    info.row_count = rows;
			    info.file_size_bytes = bytes;
			    return info;
		    };

		    auto stale_dir = BuildCopyDirectWriteTaskDirectory(final_root, stale_run_id, "w_stale");
		    auto stale_file = stale_dir + "/part.parquet";
		    const std::string stale_body = "stale";
		    write_file(stale_file, stale_body);

		    auto stale_commit_paths = BuildDistributedCopyFinalizeCommitPaths(fs, final_root, stale_run_id);
		    auto stale_lifecycle_res = WriteDistributedCopyDirectWriteLifecycle(fs, final_root, stale_run_id);
		    if (stale_lifecycle_res.is_err()) {
			    throw std::runtime_error(stale_lifecycle_res.error().what());
		    }
		    vector<DistributedCopyFileInfo> stale_files;
		    stale_files.push_back(make_file_info(stale_file, 1, stale_body.size()));
		    auto stale_manifest_res = WriteDistributedCopyFinalizeManifest(fs, stale_commit_paths, final_root,
		                                                                   "direct:" + stale_run_id, stale_files);
		    if (stale_manifest_res.is_err()) {
			    throw std::runtime_error(stale_manifest_res.error().what());
		    }

		    auto committed_dir = BuildCopyDirectWriteTaskDirectory(final_root, committed_run_id, "w_committed");
		    auto committed_file = committed_dir + "/part.parquet";
		    const std::string committed_body = "committed";
		    write_file(committed_file, committed_body);

		    auto committed_paths = BuildDistributedCopyFinalizeCommitPaths(fs, final_root, committed_run_id);
		    auto committed_lifecycle_res = WriteDistributedCopyDirectWriteLifecycle(fs, final_root, committed_run_id);
		    if (committed_lifecycle_res.is_err()) {
			    throw std::runtime_error(committed_lifecycle_res.error().what());
		    }
		    vector<DistributedCopyFileInfo> committed_files;
		    committed_files.push_back(make_file_info(committed_file, 1, committed_body.size()));
		    auto committed_manifest_res = WriteDistributedCopyFinalizeManifest(
		        fs, committed_paths, final_root, "direct:" + committed_run_id, committed_files);
		    if (committed_manifest_res.is_err()) {
			    throw std::runtime_error(committed_manifest_res.error().what());
		    }
		    auto committed_marker_res = WriteDistributedCopyFinalizeCommittedMarker(fs, committed_paths);
		    if (committed_marker_res.is_err()) {
			    throw std::runtime_error(committed_marker_res.error().what());
		    }

		    auto stale_cleanup_res = CleanupDistributedCopyUncommittedDirectWriteRun(fs, final_root, stale_run_id);
		    if (stale_cleanup_res.is_err()) {
			    throw std::runtime_error(stale_cleanup_res.error().what());
		    }
		    auto committed_cleanup_res =
		        CleanupDistributedCopyUncommittedDirectWriteRun(fs, final_root, committed_run_id);
		    if (committed_cleanup_res.is_err()) {
			    throw std::runtime_error(committed_cleanup_res.error().what());
		    }
		    auto stale_cleanup = std::move(stale_cleanup_res.value());
		    auto committed_cleanup = std::move(committed_cleanup_res.value());

		    py::dict out;
		    out["stale_skipped_committed"] = stale_cleanup.skipped_committed;
		    out["stale_data_run_dir_existed"] = stale_cleanup.data_run_dir_existed;
		    out["stale_data_run_dir_removed"] = stale_cleanup.data_run_dir_removed;
		    out["stale_commit_dir_existed"] = stale_cleanup.commit_dir_existed;
		    out["stale_commit_dir_removed"] = stale_cleanup.commit_dir_removed;
		    out["stale_run_dir_exists"] = DistributedCopyDirectoryExistsNoThrow(fs, stale_dir);
		    out["stale_file_exists"] = DistributedCopyFileExistsNoThrow(fs, stale_file);
		    out["stale_manifest_exists"] = DistributedCopyFileExistsNoThrow(fs, stale_commit_paths.manifest_path);
		    out["stale_commit_dir_exists"] = DistributedCopyDirectoryExistsNoThrow(fs, stale_commit_paths.commit_dir);
		    out["committed_skipped_committed"] = committed_cleanup.skipped_committed;
		    out["committed_data_run_dir_removed"] = committed_cleanup.data_run_dir_removed;
		    out["committed_commit_dir_removed"] = committed_cleanup.commit_dir_removed;
		    out["committed_run_dir_exists"] = DistributedCopyDirectoryExistsNoThrow(fs, committed_dir);
		    out["committed_file_exists"] = DistributedCopyFileExistsNoThrow(fs, committed_file);
		    out["committed_manifest_exists"] = DistributedCopyFileExistsNoThrow(fs, committed_paths.manifest_path);
		    out["committed_marker_exists"] =
		        DistributedCopyFileExistsNoThrow(fs, committed_paths.committed_marker_path);
		    out["committed_commit_dir_exists"] = DistributedCopyDirectoryExistsNoThrow(fs, committed_paths.commit_dir);
		    return out;
	    },
	    py::arg("local_dir"), "Exercise explicit cleanup of uncommitted stale COPY direct-write runs.");

	m.def(
	    "split_exchange_source_task_by_partition",
	    [](py::bytes bytes_obj) {
		    using namespace duckdb::distributed;
		    string raw(bytes_obj);
		    auto desc = ExchangeSourceTaskDescriptor::DeserializeFromBytes(raw);
		    idx_t partition_count = desc.source_partition_count;
		    if (partition_count == 0 && !desc.partition_indices.empty()) {
			    for (auto partition_id : desc.partition_indices) {
				    partition_count = std::max(partition_count, partition_id + 1);
			    }
		    }

		    py::list out;
		    for (auto partition_id : desc.partition_indices) {
			    ExchangeSourceTaskDescriptor single;
			    single.partition_indices = vector<idx_t> {partition_id};
			    single.source_partition_count = partition_count;
			    single.source_task_count = desc.source_task_count;
			    single.replicated = desc.replicated;
			    single.mark_join_build_summary = desc.mark_join_build_summary;
			    for (const auto &handle : desc.source_handles) {
				    if (handle.partition_id == partition_id) {
					    single.source_handles.push_back(handle);
				    }
			    }
			    auto bytes = single.SerializeToBytes();
			    out.append(py::make_tuple(partition_id, py::bytes(bytes), partition_count, desc.source_task_count,
			                              desc.replicated));
		    }
		    return out;
	    },
	    py::arg("bytes"), "Split a raw ExchangeSourceTaskDescriptor into one descriptor per source partition.");

	// Stop the poller before CPython enters late finalization. The callback is

	// registered after the other Ray cleanup hook, so it runs first at exit.

	py::module_::import("atexit").attr("register")(
	    py::cpp_function([]() { duckdb::distributed::python::ray::ShutdownRayTaskResultPoller(); }));
}
} // namespace ray_bindings
} // namespace duckdb
