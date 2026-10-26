// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "ray_binding_internal.hpp"

namespace duckdb {
namespace ray_bindings {
namespace py = ::duckdb::py;

void RegisterRayDescriptorTestsBindings(py::module_ &m) {
	m.def(
	    "distributed_copy_finalize_missing_staging_preflight_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    auto staging_root = local_dir + "/copy_staging/run-1";
		    auto first_staged = staging_root + "/w_0/part.parquet";
		    auto missing_staged = staging_root + "/w_1/part.parquet";
		    auto final_root = local_dir + "/copy_final";
		    auto &fs = FileSystem::GetFileSystem(context);
		    fs.CreateDirectoriesRecursive(staging_root + "/w_0");
		    {
			    std::ofstream output(first_staged, std::ios::out | std::ios::binary);
			    output << "first";
		    }

		    DistributedCopySpec spec;
		    spec.file_path = final_root;
		    spec.file_extension = "parquet";
		    spec.per_thread_output = true;
		    spec.overwrite_mode = CopyOverwriteMode::COPY_ERROR_ON_CONFLICT;

		    vector<DistributedCopyFileInfo> files;
		    DistributedCopyFileInfo first;
		    first.staging_path = first_staged;
		    first.row_count = 1;
		    files.push_back(std::move(first));
		    DistributedCopyFileInfo second;
		    second.staging_path = missing_staged;
		    second.row_count = 1;
		    files.push_back(std::move(second));

		    auto finalize_res = FinalizeCopyFiles(spec, staging_root, std::move(files), context);

		    std::vector<std::string> final_files;
		    bool final_root_exists = false;
		    try {
			    final_root_exists = fs.DirectoryExists(final_root);
			    if (final_root_exists) {
				    ListDistributedCopyFilesRecursive(fs, final_root, final_files);
			    }
		    } catch (...) {
			    final_root_exists = false;
		    }

		    py::dict out;
		    out["finalize_error"] = finalize_res.is_err();
		    out["error"] = finalize_res.is_err() ? finalize_res.error().what() : "";
		    out["first_staging_exists"] = fs.FileExists(first_staged);
		    out["missing_staging_exists"] = fs.FileExists(missing_staged);
		    out["final_root_exists"] = final_root_exists;
		    out["final_file_count"] = final_files.size();
		    return out;
	    },
	    py::arg("local_dir"),
	    "Exercise COPY finalize preflight: missing staging files fail before any final output is moved.");

	m.def(
	    "distributed_copy_finalize_commit_manifest_idempotent_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;
		    auto &fs = FileSystem::GetFileSystem(context);

		    auto staging_root = local_dir + "/copy_staging/run-commit";
		    auto first_staged = staging_root + "/w_0/part.parquet";
		    auto second_staged = staging_root + "/w_1/part.parquet";
		    auto final_root = local_dir + "/copy_final_commit";
		    fs.CreateDirectoriesRecursive(staging_root + "/w_0");
		    fs.CreateDirectoriesRecursive(staging_root + "/w_1");

		    const std::string first_body = "first";
		    const std::string second_body = "second";
		    {
			    std::ofstream output(first_staged, std::ios::out | std::ios::binary);
			    output << first_body;
		    }
		    {
			    std::ofstream output(second_staged, std::ios::out | std::ios::binary);
			    output << second_body;
		    }

		    DistributedCopySpec spec;
		    spec.file_path = final_root;
		    spec.file_extension = "parquet";
		    spec.per_thread_output = true;
		    spec.overwrite_mode = CopyOverwriteMode::COPY_ERROR_ON_CONFLICT;

		    auto make_files = [&]() {
			    vector<DistributedCopyFileInfo> files;
			    DistributedCopyFileInfo first;
			    first.staging_path = first_staged;
			    first.row_count = 1;
			    first.file_size_bytes = first_body.size();
			    files.push_back(std::move(first));
			    DistributedCopyFileInfo second;
			    second.staging_path = second_staged;
			    second.row_count = 2;
			    second.file_size_bytes = second_body.size();
			    files.push_back(std::move(second));
			    return files;
		    };

		    auto commit_paths = BuildDistributedCopyFinalizeCommitPathsFromStagingRoot(fs, final_root, staging_root);
		    auto first_res = FinalizeCopyFiles(spec, staging_root, make_files(), context);
		    auto second_res = FinalizeCopyFiles(spec, staging_root, make_files(), context);

		    std::vector<std::string> final_files;
		    bool final_root_exists = false;
		    try {
			    final_root_exists = fs.DirectoryExists(final_root);
			    if (final_root_exists) {
				    ListDistributedCopyFilesRecursive(fs, final_root, final_files);
			    }
		    } catch (...) {
			    final_root_exists = false;
		    }

		    py::dict out;
		    out["first_finalize_error"] = first_res.is_err();
		    out["first_error"] = first_res.is_err() ? first_res.error().what() : "";
		    out["second_finalize_error"] = second_res.is_err();
		    out["second_error"] = second_res.is_err() ? second_res.error().what() : "";
		    out["first_rows_copied"] = first_res.is_ok() ? first_res.value().rows_copied : 0;
		    out["second_rows_copied"] = second_res.is_ok() ? second_res.value().rows_copied : 0;
		    out["manifest_exists"] = fs.FileExists(commit_paths.manifest_path);
		    out["committed_exists"] = fs.FileExists(commit_paths.committed_marker_path);
		    out["staging_root_exists"] = fs.DirectoryExists(staging_root);
		    out["final_root_exists"] = final_root_exists;
		    out["final_file_count"] = final_files.size();
		    return out;
	    },
	    py::arg("local_dir"), "Exercise COPY finalize durable commit manifest and committed-marker idempotence.");

	m.def(
	    "distributed_copy_finalize_replays_inprogress_manifest_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;
		    auto &fs = FileSystem::GetFileSystem(context);

		    auto staging_root = local_dir + "/copy_staging/run-replay";
		    auto first_staged = staging_root + "/w_0/part.parquet";
		    auto second_staged = staging_root + "/w_1/part.parquet";
		    auto final_root = local_dir + "/copy_final_replay";
		    auto first_final = final_root + "/part-0.parquet";
		    auto second_final = final_root + "/part-1.parquet";
		    fs.CreateDirectoriesRecursive(staging_root + "/w_0");
		    fs.CreateDirectoriesRecursive(staging_root + "/w_1");
		    fs.CreateDirectoriesRecursive(final_root);

		    const std::string first_body = "first";
		    const std::string second_body = "second";
		    {
			    std::ofstream output(first_staged, std::ios::out | std::ios::binary);
			    output << first_body;
		    }
		    {
			    std::ofstream output(second_staged, std::ios::out | std::ios::binary);
			    output << second_body;
		    }

		    DistributedCopySpec spec;
		    spec.file_path = final_root;
		    spec.file_extension = "parquet";
		    spec.per_thread_output = true;
		    spec.overwrite_mode = CopyOverwriteMode::COPY_ERROR_ON_CONFLICT;

		    vector<DistributedCopyFileInfo> manifest_files;
		    DistributedCopyFileInfo first_manifest;
		    first_manifest.staging_path = first_staged;
		    first_manifest.final_path = first_final;
		    first_manifest.row_count = 1;
		    first_manifest.file_size_bytes = first_body.size();
		    manifest_files.push_back(std::move(first_manifest));
		    DistributedCopyFileInfo second_manifest;
		    second_manifest.staging_path = second_staged;
		    second_manifest.final_path = second_final;
		    second_manifest.row_count = 2;
		    second_manifest.file_size_bytes = second_body.size();
		    manifest_files.push_back(std::move(second_manifest));

		    auto commit_paths = BuildDistributedCopyFinalizeCommitPathsFromStagingRoot(fs, final_root, staging_root);
		    auto manifest_res =
		        WriteDistributedCopyFinalizeManifest(fs, commit_paths, final_root, staging_root, manifest_files);
		    if (manifest_res.is_err()) {
			    throw std::runtime_error(manifest_res.error().what());
		    }
		    fs.MoveFile(first_staged, first_final);

		    auto make_input_files = [&]() {
			    vector<DistributedCopyFileInfo> files;
			    DistributedCopyFileInfo first;
			    first.staging_path = first_staged;
			    first.row_count = 1;
			    first.file_size_bytes = first_body.size();
			    files.push_back(std::move(first));
			    DistributedCopyFileInfo second;
			    second.staging_path = second_staged;
			    second.row_count = 2;
			    second.file_size_bytes = second_body.size();
			    files.push_back(std::move(second));
			    return files;
		    };

		    const bool committed_before = fs.FileExists(commit_paths.committed_marker_path);
		    const bool first_final_before = fs.FileExists(first_final);
		    const bool second_final_before = fs.FileExists(second_final);
		    const bool first_staging_before = fs.FileExists(first_staged);
		    const bool second_staging_before = fs.FileExists(second_staged);
		    auto finalize_res = FinalizeCopyFiles(spec, staging_root, make_input_files(), context);
		    auto idempotent_res = FinalizeCopyFiles(spec, staging_root, make_input_files(), context);

		    std::vector<std::string> final_files;
		    bool final_root_exists = false;
		    try {
			    final_root_exists = fs.DirectoryExists(final_root);
			    if (final_root_exists) {
				    ListDistributedCopyFilesRecursive(fs, final_root, final_files);
			    }
		    } catch (...) {
			    final_root_exists = false;
		    }

		    py::dict out;
		    out["committed_before"] = committed_before;
		    out["first_final_before"] = first_final_before;
		    out["second_final_before"] = second_final_before;
		    out["first_staging_before"] = first_staging_before;
		    out["second_staging_before"] = second_staging_before;
		    out["finalize_error"] = finalize_res.is_err();
		    out["error"] = finalize_res.is_err() ? finalize_res.error().what() : "";
		    out["idempotent_error"] = idempotent_res.is_err();
		    out["idempotent_error_message"] = idempotent_res.is_err() ? idempotent_res.error().what() : "";
		    out["rows_copied"] = finalize_res.is_ok() ? finalize_res.value().rows_copied : 0;
		    out["idempotent_rows_copied"] = idempotent_res.is_ok() ? idempotent_res.value().rows_copied : 0;
		    out["manifest_exists"] = fs.FileExists(commit_paths.manifest_path);
		    out["committed_after"] = fs.FileExists(commit_paths.committed_marker_path);
		    out["staging_root_exists"] = fs.DirectoryExists(staging_root);
		    out["final_root_exists"] = final_root_exists;
		    out["final_file_count"] = final_files.size();
		    return out;
	    },
	    py::arg("local_dir"),
	    "Exercise COPY finalize replay after manifest was written and only a prefix of files was moved.");

	m.def(
	    "distributed_copy_direct_write_commit_manifest_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;
		    auto &fs = FileSystem::GetFileSystem(context);

		    const std::string run_id = "run-direct";
		    auto final_root = local_dir + "/copy_direct_final";
		    auto first_dir = BuildCopyDirectWriteTaskDirectory(final_root, run_id, "w_0");
		    auto second_dir = BuildCopyDirectWriteTaskDirectory(final_root, run_id, "w_1");
		    auto loser_dir = BuildCopyDirectWriteTaskDirectory(final_root, run_id, "w_loser");
		    auto first_file = first_dir + "/part.parquet";
		    auto second_file = second_dir + "/part.parquet";
		    auto loser_file = loser_dir + "/part.parquet";
		    fs.CreateDirectoriesRecursive(first_dir);
		    fs.CreateDirectoriesRecursive(second_dir);
		    fs.CreateDirectoriesRecursive(loser_dir);

		    const std::string first_body = "first";
		    const std::string second_body = "second";
		    const std::string loser_body = "loser";
		    {
			    std::ofstream output(first_file, std::ios::out | std::ios::binary);
			    output << first_body;
		    }
		    {
			    std::ofstream output(second_file, std::ios::out | std::ios::binary);
			    output << second_body;
		    }
		    {
			    std::ofstream output(loser_file, std::ios::out | std::ios::binary);
			    output << loser_body;
		    }

		    DistributedCopySpec spec;
		    spec.file_path = final_root;
		    spec.file_extension = "parquet";
		    spec.per_thread_output = true;
		    spec.overwrite_mode = CopyOverwriteMode::COPY_ERROR_ON_CONFLICT;

		    auto make_files = [&]() {
			    vector<DistributedCopyFileInfo> files;
			    DistributedCopyFileInfo first;
			    first.staging_path = first_file;
			    first.row_count = 1;
			    first.file_size_bytes = first_body.size();
			    files.push_back(std::move(first));
			    DistributedCopyFileInfo second;
			    second.staging_path = second_file;
			    second.row_count = 2;
			    second.file_size_bytes = second_body.size();
			    files.push_back(std::move(second));
			    return files;
		    };

		    auto commit_paths = BuildDistributedCopyFinalizeCommitPaths(fs, final_root, run_id);
		    auto lifecycle_res = WriteDistributedCopyDirectWriteLifecycle(fs, final_root, run_id);
		    if (lifecycle_res.is_err()) {
			    throw std::runtime_error(lifecycle_res.error().what());
		    }
		    auto first_res = FinalizeCopyFiles(spec, "", make_files(), context, run_id);
		    auto replay_loser_dir = BuildCopyDirectWriteTaskDirectory(final_root, run_id, "w_replay_loser");
		    auto replay_loser_file = replay_loser_dir + "/part.parquet";
		    fs.CreateDirectoriesRecursive(replay_loser_dir);
		    {
			    std::ofstream output(replay_loser_file, std::ios::out | std::ios::binary);
			    output << "replay_loser";
		    }
		    auto second_res = FinalizeCopyFiles(spec, "", make_files(), context, run_id);

		    py::dict out;
		    out["first_finalize_error"] = first_res.is_err();
		    out["first_error"] = first_res.is_err() ? first_res.error().what() : "";
		    out["second_finalize_error"] = second_res.is_err();
		    out["second_error"] = second_res.is_err() ? second_res.error().what() : "";
		    out["first_rows_copied"] = first_res.is_ok() ? first_res.value().rows_copied : 0;
		    out["second_rows_copied"] = second_res.is_ok() ? second_res.value().rows_copied : 0;
		    out["first_final_path"] = first_res.is_ok() ? first_res.value().files[0].final_path : "";
		    out["second_final_path"] = first_res.is_ok() ? first_res.value().files[1].final_path : "";
		    out["first_output_run_id"] = first_res.is_ok() ? first_res.value().output_run_id : "";
		    out["first_output_manifest_path"] = first_res.is_ok() ? first_res.value().output_manifest_path : "";
		    out["first_output_committed_marker_path"] =
		        first_res.is_ok() ? first_res.value().output_committed_marker_path : "";
		    out["first_output_direct_write"] = first_res.is_ok() ? first_res.value().output_direct_write : false;
		    out["first_output_committed"] = first_res.is_ok() ? first_res.value().output_committed : false;
		    out["manifest_exists"] = fs.FileExists(commit_paths.manifest_path);
		    out["committed_exists"] = fs.FileExists(commit_paths.committed_marker_path);
		    out["direct_prefix_exists"] = fs.DirectoryExists(final_root + "/_vane_direct_write_" + run_id);
		    out["first_file_exists"] = fs.FileExists(first_file);
		    out["second_file_exists"] = fs.FileExists(second_file);
		    out["loser_file_exists"] = fs.FileExists(loser_file);
		    out["replay_loser_file_exists"] = fs.FileExists(replay_loser_file);
		    return out;
	    },
	    py::arg("local_dir"), "Exercise COPY direct-write run isolation plus selected-output commit manifest.");

	m.def(
	    "distributed_copy_direct_target_visible_commit_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;
		    auto &fs = FileSystem::GetFileSystem(context);

		    const std::string run_id = "run-visible";
		    const std::string other_run_id = "other-run";
		    auto final_root = local_dir + "/copy_direct_target_visible";
		    auto first_file = BuildCopyDirectTargetFilePath(fs, final_root, run_id, "w_0", "part0.parquet");
		    auto second_file = BuildCopyDirectTargetFilePath(fs, final_root, run_id, "w_1", "part1.parquet");
		    auto loser_file = BuildCopyDirectTargetFilePath(fs, final_root, run_id, "w_loser", "part.parquet");
		    auto replay_loser_file =
		        BuildCopyDirectTargetFilePath(fs, final_root, run_id, "w_replay_loser", "part.parquet");
		    auto other_run_file =
		        BuildCopyDirectTargetFilePath(fs, final_root, other_run_id, "w_other", "part.parquet");

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

		    const std::string first_body = "first";
		    const std::string second_body = "second";
		    write_file(first_file, first_body);
		    write_file(second_file, second_body);
		    write_file(loser_file, "loser");
		    write_file(other_run_file, "other");

		    DistributedCopySpec spec;
		    spec.file_path = final_root;
		    spec.file_extension = "parquet";
		    spec.per_thread_output = true;
		    spec.overwrite_mode = CopyOverwriteMode::COPY_ERROR_ON_CONFLICT;

		    auto make_files = [&]() {
			    vector<DistributedCopyFileInfo> files;
			    DistributedCopyFileInfo first;
			    first.staging_path = first_file;
			    first.row_count = 1;
			    first.file_size_bytes = first_body.size();
			    files.push_back(std::move(first));
			    DistributedCopyFileInfo second;
			    second.staging_path = second_file;
			    second.row_count = 2;
			    second.file_size_bytes = second_body.size();
			    files.push_back(std::move(second));
			    return files;
		    };

		    auto commit_paths = BuildDistributedCopyFinalizeCommitPaths(fs, final_root, run_id);
		    auto lifecycle_res = WriteDistributedCopyDirectWriteLifecycle(fs, final_root, run_id);
		    if (lifecycle_res.is_err()) {
			    throw std::runtime_error(lifecycle_res.error().what());
		    }
		    auto first_res = FinalizeCopyFiles(spec, "", make_files(), context, run_id);
		    write_file(replay_loser_file, "replay_loser");
		    auto second_res = FinalizeCopyFiles(spec, "", make_files(), context, run_id);
		    auto read_res = ReadCommittedDistributedCopyDirectWriteResult(fs, final_root, run_id);

		    py::dict out;
		    out["first_finalize_error"] = first_res.is_err();
		    out["first_error"] = first_res.is_err() ? first_res.error().what() : "";
		    out["second_finalize_error"] = second_res.is_err();
		    out["second_error"] = second_res.is_err() ? second_res.error().what() : "";
		    out["read_committed_error"] = read_res.is_err();
		    out["read_committed_error_message"] = read_res.is_err() ? read_res.error().what() : "";
		    out["read_committed_rows_copied"] = read_res.is_ok() ? read_res.value().rows_copied : 0;
		    out["first_rows_copied"] = first_res.is_ok() ? first_res.value().rows_copied : 0;
		    out["second_rows_copied"] = second_res.is_ok() ? second_res.value().rows_copied : 0;
		    out["first_final_path"] = first_res.is_ok() ? first_res.value().files[0].final_path : "";
		    out["first_output_run_id"] = first_res.is_ok() ? first_res.value().output_run_id : "";
		    out["first_output_direct_write"] = first_res.is_ok() ? first_res.value().output_direct_write : false;
		    out["first_output_committed"] = first_res.is_ok() ? first_res.value().output_committed : false;
		    out["manifest_exists"] = fs.FileExists(commit_paths.manifest_path);
		    out["committed_exists"] = fs.FileExists(commit_paths.committed_marker_path);
		    out["direct_prefix_exists"] = fs.DirectoryExists(final_root + "/_vane_direct_write_" + run_id);
		    out["first_file_exists"] = fs.FileExists(first_file);
		    out["second_file_exists"] = fs.FileExists(second_file);
		    out["loser_file_exists"] = fs.FileExists(loser_file);
		    out["replay_loser_file_exists"] = fs.FileExists(replay_loser_file);
		    out["other_run_file_exists"] = fs.FileExists(other_run_file);
		    return out;
	    },
	    py::arg("local_dir"), "Exercise visible direct-target COPY commit cleanup.");

	m.def(
	    "distributed_copy_direct_target_remote_path_for_test",
	    [](const std::string &base_path, const std::string &run_id, const std::string &worker_dir_name,
	       const std::string &file_name) {
		    using namespace duckdb::distributed;

		    py::dict out;
		    out["direct_target_file"] = BuildCopyDirectTargetFilePath(base_path, run_id, worker_dir_name, file_name);
		    out["legacy_task_directory"] = BuildCopyDirectWriteTaskDirectory(base_path, run_id, worker_dir_name);
		    out["filename_pattern"] = BuildCopyDirectTargetFilenamePattern(run_id, worker_dir_name);
		    return out;
	    },
	    py::arg("base_path"), py::arg("run_id"), py::arg("worker_dir_name"), py::arg("file_name"),
	    "Exercise visible direct-target COPY path helpers.");

	m.def(
	    "distributed_copy_sink_mode_for_test",
	    [](const std::string &output_path, bool use_tmp_file) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    py::dict out;
		    DistributedCopySpec spec;
		    spec.file_path = output_path;
		    spec.file_extension = "parquet";
		    spec.per_thread_output = true;
		    spec.use_tmp_file = use_tmp_file;
		    try {
			    auto sink = std::make_shared<CopySinkNode>(1, PipelineNodeRef(), std::move(spec));
			    out["construct_error"] = false;
			    out["error"] = "";
			    out["staging_root_base"] = sink->staging_root_base();
			    out["staging_run_id"] = sink->staging_run_id();
			    out["uses_direct_write"] = sink->staging_root_base().empty();
			    out["uses_visible_direct_target"] = sink->staging_root_base().empty();
		    } catch (const std::exception &ex) {
			    out["construct_error"] = true;
			    out["error"] = ex.what();
			    out["staging_root_base"] = "";
			    out["staging_run_id"] = "";
			    out["uses_direct_write"] = false;
			    out["uses_visible_direct_target"] = false;
		    }
		    return out;
	    },
	    py::arg("output_path"), py::arg("use_tmp_file") = false,
	    "Return distributed COPY sink mode for an output path.");

	m.def(
	    "distributed_copy_direct_write_local_invisible_file_commit_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;
		    auto &fs = FileSystem::GetFileSystem(context);

		    const std::string run_id = "run-local-invisible";
		    auto final_root = local_dir + "/copy_direct_invisible";
		    auto invisible_file = BuildCopyDirectWriteTaskDirectory(final_root, run_id, "w_remote") + "/part.parquet";

		    DistributedCopySpec spec;
		    spec.file_path = final_root;
		    spec.file_extension = "parquet";
		    spec.per_thread_output = true;
		    spec.overwrite_mode = CopyOverwriteMode::COPY_ERROR_ON_CONFLICT;

		    vector<DistributedCopyFileInfo> files;
		    DistributedCopyFileInfo info;
		    info.staging_path = invisible_file;
		    info.row_count = 4;
		    info.file_size_bytes = 123;
		    files.push_back(std::move(info));

		    auto commit_paths = BuildDistributedCopyFinalizeCommitPaths(fs, final_root, run_id);
		    auto lifecycle_res = WriteDistributedCopyDirectWriteLifecycle(fs, final_root, run_id);
		    if (lifecycle_res.is_err()) {
			    throw std::runtime_error(lifecycle_res.error().what());
		    }
		    auto finalize_res = FinalizeCopyFiles(spec, "", std::move(files), context, run_id);

		    py::dict out;
		    out["finalize_error"] = finalize_res.is_err();
		    out["error"] = finalize_res.is_err() ? finalize_res.error().what() : "";
		    out["rows_copied"] = finalize_res.is_ok() ? finalize_res.value().rows_copied : 0;
		    out["output_direct_write"] = finalize_res.is_ok() ? finalize_res.value().output_direct_write : false;
		    out["output_committed"] = finalize_res.is_ok() ? finalize_res.value().output_committed : false;
		    out["manifest_exists"] = fs.FileExists(commit_paths.manifest_path);
		    out["committed_exists"] = fs.FileExists(commit_paths.committed_marker_path);
		    out["invisible_file_exists"] = DistributedCopyFileExistsNoThrow(fs, invisible_file);
		    return out;
	    },
	    py::arg("local_dir"),
	    "Exercise local direct-write commit when coordinator cannot see a worker node-local output file.");
}
} // namespace ray_bindings
} // namespace duckdb
