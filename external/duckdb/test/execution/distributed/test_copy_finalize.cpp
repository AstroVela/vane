// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"
#include "test_helpers.hpp"

#include "duckdb/common/local_file_system.hpp"
#include "duckdb/execution/distributed/copy_finalize.hpp"

using namespace duckdb;
using namespace duckdb::distributed;

namespace {

class FileOnlyRecursiveListFileSystem : public LocalFileSystem {
public:
	explicit FileOnlyRecursiveListFileSystem(bool qualified_paths_p = false) : qualified_paths(qualified_paths_p) {
	}

	bool DirectoryExists(const string &, optional_ptr<FileOpener> = nullptr) override {
		return false;
	}

	bool ListFiles(const string &directory, const std::function<void(const string &, bool)> &callback,
	               FileOpener * = nullptr) override {
		return ListObjectKeys(directory, directory, callback);
	}

	string GetName() const override {
		return "FileOnlyRecursiveListFileSystem";
	}

private:
	bool ListObjectKeys(const string &root, const string &directory,
	                    const std::function<void(const string &, bool)> &callback) {
		bool found = false;
		backing_fs.ListFiles(directory, [&](const string &path, bool is_dir) {
			auto full_path = backing_fs.JoinPath(directory, path);
			if (is_dir) {
				found = ListObjectKeys(root, full_path, callback) || found;
				return;
			}
			auto callback_path = full_path;
			if (!qualified_paths) {
				callback_path = full_path.substr(root.size());
				auto separator = backing_fs.PathSeparator(callback_path);
				if (StringUtil::StartsWith(callback_path, separator)) {
					callback_path = callback_path.substr(separator.size());
				}
			}
			callback(callback_path, false);
			callback(callback_path, false);
			found = true;
		});
		return found;
	}

	LocalFileSystem backing_fs;
	bool qualified_paths;
};

class CountingFileOnlyRecursiveListFileSystem : public FileOnlyRecursiveListFileSystem {
public:
	bool ListFiles(const string &directory, const std::function<void(const string &, bool)> &callback,
	               FileOpener *opener = nullptr) override {
		list_calls[directory]++;
		return FileOnlyRecursiveListFileSystem::ListFiles(directory, callback, opener);
	}

	idx_t ListCallCount(const string &directory) const {
		auto entry = list_calls.find(directory);
		return entry == list_calls.end() ? 0 : entry->second;
	}

private:
	std::unordered_map<string, idx_t> list_calls;
};

class MarkerCheckFailureFileSystem : public FileOnlyRecursiveListFileSystem {
public:
	explicit MarkerCheckFailureFileSystem(string marker_path) : marker_path(std::move(marker_path)) {
	}

	unique_ptr<FileHandle> OpenFile(const string &path, FileOpenFlags flags,
	                                optional_ptr<FileOpener> opener = nullptr) override {
		if (path == marker_path) {
			throw IOException("injected marker check failure");
		}
		return LocalFileSystem::OpenFile(path, flags, opener);
	}

private:
	string marker_path;
};

class MissingLocalMarkerFileSystem : public LocalFileSystem {
public:
	explicit MissingLocalMarkerFileSystem(string marker_path) : marker_path(std::move(marker_path)) {
	}

	unique_ptr<FileHandle> OpenFile(const string &path, FileOpenFlags flags,
	                                optional_ptr<FileOpener> opener = nullptr) override {
		if (path == marker_path) {
			used_null_if_missing = flags.ReturnNullIfNotExists();
			if (used_null_if_missing) {
				return nullptr;
			}
			throw IOException("injected platform-specific missing-file error");
		}
		return LocalFileSystem::OpenFile(path, flags, opener);
	}

	bool used_null_if_missing = false;

private:
	string marker_path;
};

class RemoteMarkerStatusFileSystem : public LocalFileSystem {
public:
	explicit RemoteMarkerStatusFileSystem(string status_code) : status_code(std::move(status_code)) {
	}

	unique_ptr<FileHandle> OpenFile(const string &, FileOpenFlags flags, optional_ptr<FileOpener> = nullptr) override {
		used_null_if_missing = flags.ReturnNullIfNotExists();
		throw Exception({{"status_code", status_code}}, ExceptionType::HTTP, "injected remote marker response");
	}

	bool used_null_if_missing = false;

private:
	string status_code;
};

class FileRemovalFailureFileSystem : public FileOnlyRecursiveListFileSystem {
public:
	explicit FileRemovalFailureFileSystem(string failed_path) : failed_path(std::move(failed_path)) {
	}

	void RemoveFile(const string &path, optional_ptr<FileOpener> opener = nullptr) override {
		if (path == failed_path) {
			throw IOException("injected object removal failure");
		}
		LocalFileSystem::RemoveFile(path, opener);
	}

private:
	string failed_path;
};

class CopyFinalizeTestDirectory {
public:
	explicit CopyFinalizeTestDirectory(const string &name) : path(TestCreatePath(name)) {
		if (fs.DirectoryExists(path)) {
			fs.RemoveDirectory(path);
		}
		fs.CreateDirectoriesRecursive(path);
	}

	~CopyFinalizeTestDirectory() {
		try {
			if (fs.DirectoryExists(path)) {
				fs.RemoveDirectory(path);
			}
		} catch (...) {
		}
	}

	LocalFileSystem fs;
	string path;
};

void WriteTestFile(FileSystem &fs, const string &path, const string &contents) {
	auto parent = StringUtil::GetFilePath(path);
	if (!parent.empty() && !fs.DirectoryExists(parent)) {
		fs.CreateDirectoriesRecursive(parent);
	}
	auto write_res = WriteDistributedCopyTextFileAtomically(fs, path, contents);
	REQUIRE(write_res.is_ok());
}

} // namespace

TEST_CASE("Distributed COPY canonical base path handles temporary and trailing paths",
          "[distributed][copy][lifecycle][path]") {
	LocalFileSystem fs;
	auto parent = TestCreatePath("copy_finalize_canonical_path");
	DistributedCopySpec spec;
	auto output_path = fs.JoinPath(parent, "copy-output");
	spec.file_path = output_path + fs.PathSeparator(output_path);

	auto trailing_res = CanonicalDistributedCopyBasePath(fs, spec);
	REQUIRE(trailing_res.is_ok());
	REQUIRE(trailing_res.value() == output_path);

	spec.file_path = fs.JoinPath(parent, "tmp_copy-output");
	spec.use_tmp_file = true;
	auto temporary_res = CanonicalDistributedCopyBasePath(fs, spec);
	REQUIRE(temporary_res.is_ok());
	REQUIRE(temporary_res.value() == output_path);

	spec.use_tmp_file = false;
	auto literal_res = CanonicalDistributedCopyBasePath(fs, spec);
	REQUIRE(literal_res.is_ok());
	REQUIRE(literal_res.value() == fs.JoinPath(parent, "tmp_copy-output"));

	auto root = fs.PathSeparator(std::string());
	auto root_output_path = root + "copy-output";
	auto root_temporary_output_path = root + "tmp_copy-output";
	auto root_res = CanonicalDistributedCopyBasePath(fs, root + root + root);
	REQUIRE(root_res.is_ok());
	REQUIRE(root_res.value() == root);
	auto root_paths = BuildDistributedCopyFinalizeCommitPaths(fs, root_res.value(), "run-root");
	REQUIRE(root_paths.commit_dir == root + ".duckdb_commit" + root + "run-root");

	auto authority_root_res = CanonicalDistributedCopyBasePath(fs, "s3://bucket///");
	REQUIRE(authority_root_res.is_ok());
	REQUIRE(authority_root_res.value() == "s3://bucket/");
	auto authority_root_without_separator_res = CanonicalDistributedCopyBasePath(fs, "s3://bucket");
	REQUIRE(authority_root_without_separator_res.is_ok());
	REQUIRE(authority_root_without_separator_res.value() == "s3://bucket/");
	auto authority_paths =
	    BuildDistributedCopyFinalizeCommitPaths(fs, authority_root_res.value(), "run-authority-root");
	REQUIRE(authority_paths.commit_dir == "s3://bucket/.duckdb_commit/run-authority-root");
	auto authority_prefix_res = CanonicalDistributedCopyBasePath(fs, "s3://bucket/prefix///");
	REQUIRE(authority_prefix_res.is_ok());
	REQUIRE(authority_prefix_res.value() == "s3://bucket/prefix");
	auto empty_authority_root_res = CanonicalDistributedCopyBasePath(fs, "file:////");
	REQUIRE(empty_authority_root_res.is_ok());
	REQUIRE(empty_authority_root_res.value() == "file:///");

	spec.file_path = root_temporary_output_path;
	spec.use_tmp_file = true;
	auto root_temporary_res = CanonicalDistributedCopyBasePath(fs, spec);
	REQUIRE(root_temporary_res.is_ok());
	REQUIRE(root_temporary_res.value() == root_output_path);
	REQUIRE(DistributedCopyTemporaryBasePath(fs, root_output_path) == root_temporary_output_path);
	REQUIRE(DistributedCopyWorkerBaseMatchesCanonical(fs, root_output_path, root_temporary_output_path));
}

TEST_CASE("Distributed COPY temporary direct output preserves the canonical target",
          "[distributed][copy][lifecycle][path]") {
	CopyFinalizeTestDirectory test_dir("copy_finalize_temporary_replacement");
	auto &fs = test_dir.fs;
	auto output_path = fs.JoinPath(test_dir.path, "copy-output");
	auto temporary_output_path = fs.JoinPath(test_dir.path, "tmp_copy-output");
	const string run_id = "run-tmp";
	auto worker_file = BuildCopyDirectTargetFilePath(temporary_output_path, run_id, "w_0", "part.parquet");
	const string replacement_contents = "replacement";

	WriteTestFile(fs, output_path, "old");
	WriteTestFile(fs, worker_file, replacement_contents);
	REQUIRE(WriteDistributedCopyDirectWriteLifecycle(fs, output_path, run_id, 1, temporary_output_path).is_ok());

	DuckDB db(nullptr);
	Connection connection(db);
	DistributedCopySpec spec;
	spec.file_path = temporary_output_path;
	spec.use_tmp_file = true;
	spec.file_extension = "parquet";

	auto make_files = [&]() {
		vector<DistributedCopyFileInfo> files;
		DistributedCopyFileInfo file;
		file.staging_path = worker_file;
		file.row_count = 2;
		file.file_size_bytes = replacement_contents.size();
		files.push_back(std::move(file));
		return files;
	};

	auto first_res = FinalizeCopyFiles(spec, "", make_files(), *connection.context, run_id);
	REQUIRE(first_res.is_ok());
	auto first = std::move(first_res).value();
	REQUIRE(first.output_base_path == output_path);
	REQUIRE(first.output_direct_write);
	REQUIRE(first.output_committed);
	REQUIRE(first.rows_copied == 2);
	REQUIRE(first.files.size() == 1);
	REQUIRE(first.files[0].final_path == worker_file);
	REQUIRE(fs.FileExists(output_path));
	REQUIRE(ReadDistributedCopyTextFile(fs, output_path).value() == "old");
	REQUIRE(ReadDistributedCopyTextFile(fs, first.files[0].final_path).value() == replacement_contents);
	REQUIRE_FALSE(fs.DirectoryExists(temporary_output_path + ".duckdb_commit"));

	auto committed_res = ReadCommittedDistributedCopyDirectWriteResult(fs, output_path, run_id);
	REQUIRE(committed_res.is_ok());
	REQUIRE(committed_res.value().rows_copied == 2);
	REQUIRE(committed_res.value().files[0].final_path == worker_file);

	const string stale_run_id = "run-tmp-stale";
	auto stale_worker_file =
	    BuildCopyDirectTargetFilePath(temporary_output_path, stale_run_id, "w_failed", "part.parquet");
	WriteTestFile(fs, stale_worker_file, "stale");
	REQUIRE(WriteDistributedCopyDirectWriteLifecycle(fs, output_path, stale_run_id, 1, temporary_output_path).is_ok());
	auto cleanup_res = CleanupDistributedCopyUncommittedDirectWriteRun(fs, output_path, stale_run_id);
	REQUIRE(cleanup_res.is_ok());
	REQUIRE_FALSE(cleanup_res.value().skipped_committed);
	REQUIRE_FALSE(fs.FileExists(stale_worker_file));
	REQUIRE(fs.FileExists(worker_file));
	REQUIRE(fs.FileExists(output_path));
	REQUIRE(ReadDistributedCopyTextFile(fs, output_path).value() == "old");
}

TEST_CASE("Distributed COPY resolves relative and qualified list paths",
          "[distributed][copy][lifecycle][object-storage][path]") {
	LocalFileSystem fs;
	const string directory = "memory://bucket/out.duckdb_commit";
	const string qualified_path = directory + "/run/lifecycle.txt";

	REQUIRE(ResolveDistributedCopyListedPath(fs, directory, "run/lifecycle.txt") == qualified_path);
	REQUIRE(ResolveDistributedCopyListedPath(fs, directory, qualified_path) == qualified_path);
	REQUIRE(ResolveDistributedCopyListedPath(fs, directory, "/bucket/out.duckdb_commit/run/lifecycle.txt") ==
	        qualified_path);
	REQUIRE(ResolveDistributedCopyListedPath(fs, directory, "bucket/out.duckdb_commit/run/lifecycle.txt") ==
	        qualified_path);

	auto local_directory = TestCreatePath("copy_finalize_qualified_list_path");
	auto local_path = fs.JoinPath(local_directory, "lifecycle.txt");
	REQUIRE(ResolveDistributedCopyListedPath(fs, local_directory, local_path) == local_path);
	auto root = fs.PathSeparator(std::string());
	REQUIRE(ResolveDistributedCopyListedPath(fs, root, "lifecycle.txt") == root + "lifecycle.txt");
}

TEST_CASE("Distributed COPY removes directory trees from qualified file-only listings",
          "[distributed][copy][lifecycle][object-storage][path]") {
	CopyFinalizeTestDirectory test_dir("copy_finalize_qualified_directory_cleanup");
	auto &local_fs = test_dir.fs;
	auto cleanup_root = local_fs.JoinPath(test_dir.path, "cleanup");
	auto first_file = local_fs.JoinPath(cleanup_root, "first.txt");
	auto nested_file = local_fs.JoinPath(cleanup_root, "nested", "second.txt");
	WriteTestFile(local_fs, first_file, "first");
	WriteTestFile(local_fs, nested_file, "second");

	FileOnlyRecursiveListFileSystem qualified_fs(true);
	RemoveDistributedCopyDirectoryTree(qualified_fs, cleanup_root);

	REQUIRE_FALSE(local_fs.FileExists(first_file));
	REQUIRE_FALSE(local_fs.FileExists(nested_file));
}

TEST_CASE("Distributed COPY strict marker checks use the portable local missing-file contract",
          "[distributed][copy][lifecycle][path]") {
	auto marker_path = TestCreatePath("copy_finalize_missing_local_marker");
	MissingLocalMarkerFileSystem fs(marker_path);

	auto exists_res = CheckDistributedCopyFileExists(fs, marker_path);

	REQUIRE(exists_res.is_ok());
	REQUIRE_FALSE(exists_res.value());
	REQUIRE(fs.used_null_if_missing);
}

TEST_CASE("Distributed COPY strict marker checks distinguish remote missing and access failures",
          "[distributed][copy][lifecycle][object-storage]") {
	const string marker_path = "s3://bucket/out.duckdb_commit/run/committed";

	RemoteMarkerStatusFileSystem missing_fs("404");
	auto missing_res = CheckDistributedCopyFileExists(missing_fs, marker_path);
	REQUIRE(missing_res.is_ok());
	REQUIRE_FALSE(missing_res.value());
	REQUIRE_FALSE(missing_fs.used_null_if_missing);

	RemoteMarkerStatusFileSystem forbidden_fs("403");
	auto forbidden_res = CheckDistributedCopyFileExists(forbidden_fs, marker_path);
	REQUIRE(forbidden_res.is_err());
	REQUIRE(StringUtil::Contains(forbidden_res.error().what(), "injected remote marker response"));
	REQUIRE_FALSE(forbidden_fs.used_null_if_missing);
}

TEST_CASE("Expired direct-write cleanup discovers file-only object listings",
          "[distributed][copy][lifecycle][object-storage]") {
	CopyFinalizeTestDirectory test_dir("copy_finalize_file_only_listing");
	auto &local_fs = test_dir.fs;
	auto base_path = local_fs.JoinPath(test_dir.path, "out");

	const string stale_run_id = "run-stale";
	const string second_stale_run_id = "run-stale-two";
	const string active_run_id = "run-active";
	const string committed_run_id = "run-committed";

	REQUIRE(WriteDistributedCopyDirectWriteLifecycle(local_fs, base_path, stale_run_id, 1).is_ok());
	auto stale_run_dir = BuildCopyDirectWriteRunDirectory(base_path, stale_run_id, local_fs.PathSeparator(base_path));
	auto stale_file = local_fs.JoinPath(stale_run_dir, "w_failed", "part.parquet");
	WriteTestFile(local_fs, stale_file, "stale");
	auto stale_direct_target_file = local_fs.JoinPath(base_path, stale_run_id + "_w_failed_part.parquet");
	WriteTestFile(local_fs, stale_direct_target_file, "stale direct target");
	auto stale_paths = BuildDistributedCopyFinalizeCommitPaths(local_fs, base_path, stale_run_id);
	WriteTestFile(local_fs, stale_paths.manifest_path, "partial");

	REQUIRE(WriteDistributedCopyDirectWriteLifecycle(local_fs, base_path, second_stale_run_id, 2).is_ok());
	auto second_stale_direct_target_file = local_fs.JoinPath(base_path, second_stale_run_id + "_w_failed_part.parquet");
	WriteTestFile(local_fs, second_stale_direct_target_file, "second stale direct target");

	REQUIRE(WriteDistributedCopyDirectWriteLifecycle(local_fs, base_path, active_run_id, 95).is_ok());
	auto active_file =
	    local_fs.JoinPath(BuildCopyDirectWriteRunDirectory(base_path, active_run_id, local_fs.PathSeparator(base_path)),
	                      "w_running", "part.parquet");
	WriteTestFile(local_fs, active_file, "active");

	REQUIRE(WriteDistributedCopyDirectWriteLifecycle(local_fs, base_path, committed_run_id, 1).is_ok());
	auto committed_file = local_fs.JoinPath(
	    BuildCopyDirectWriteRunDirectory(base_path, committed_run_id, local_fs.PathSeparator(base_path)), "w_selected",
	    "part.parquet");
	WriteTestFile(local_fs, committed_file, "committed");
	auto committed_paths = BuildDistributedCopyFinalizeCommitPaths(local_fs, base_path, committed_run_id);
	REQUIRE(WriteDistributedCopyFinalizeCommittedMarker(local_fs, committed_paths).is_ok());

	auto unregistered_path = local_fs.JoinPath(base_path + ".duckdb_commit", "run-without-lifecycle", "manifest.txt");
	WriteTestFile(local_fs, unregistered_path, "not registered");

	CountingFileOnlyRecursiveListFileSystem object_fs;
	auto cleanup_res = CleanupExpiredDistributedCopyDirectWriteRuns(object_fs, base_path, 10, 100);
	REQUIRE(cleanup_res.is_ok());
	auto cleanup = std::move(cleanup_res).value();

	REQUIRE(cleanup.scanned_runs == 4);
	REQUIRE(cleanup.cleaned_runs == 2);
	REQUIRE(cleanup.committed_runs == 1);
	REQUIRE(cleanup.active_runs == 1);
	REQUIRE(cleanup.skipped_unregistered_runs == 0);
	REQUIRE(cleanup.errors == 0);
	REQUIRE(cleanup.cleaned_run_ids == vector<string> {stale_run_id, second_stale_run_id});
	REQUIRE(object_fs.ListCallCount(base_path) == 1);
	REQUIRE_FALSE(local_fs.FileExists(stale_file));
	REQUIRE_FALSE(local_fs.FileExists(stale_direct_target_file));
	REQUIRE_FALSE(local_fs.FileExists(second_stale_direct_target_file));
	REQUIRE_FALSE(local_fs.FileExists(stale_paths.lifecycle_path));
	REQUIRE_FALSE(local_fs.FileExists(stale_paths.manifest_path));
	REQUIRE(local_fs.FileExists(active_file));
	REQUIRE(local_fs.FileExists(committed_file));
	REQUIRE(local_fs.FileExists(committed_paths.committed_marker_path));
	REQUIRE(local_fs.FileExists(unregistered_path));
}

TEST_CASE("Direct-write cleanup fails closed when committed marker status is unknown",
          "[distributed][copy][lifecycle][object-storage]") {
	CopyFinalizeTestDirectory test_dir("copy_finalize_marker_check_failure");
	auto &local_fs = test_dir.fs;
	auto base_path = local_fs.JoinPath(test_dir.path, "out");
	const string run_id = "run-unknown-commit";

	REQUIRE(WriteDistributedCopyDirectWriteLifecycle(local_fs, base_path, run_id, 1).is_ok());
	auto data_file = local_fs.JoinPath(base_path, run_id + "_w_failed_part.parquet");
	WriteTestFile(local_fs, data_file, "must survive");
	auto paths = BuildDistributedCopyFinalizeCommitPaths(local_fs, base_path, run_id);

	MarkerCheckFailureFileSystem object_fs(paths.committed_marker_path);
	auto cleanup_res = CleanupExpiredDistributedCopyDirectWriteRuns(object_fs, base_path, 1, 10);
	REQUIRE(cleanup_res.is_ok());
	auto cleanup = std::move(cleanup_res).value();

	REQUIRE(cleanup.scanned_runs == 1);
	REQUIRE(cleanup.cleaned_runs == 0);
	REQUIRE(cleanup.errors == 1);
	REQUIRE(cleanup.error_messages.size() == 1);
	REQUIRE(StringUtil::Contains(cleanup.error_messages[0], "injected marker check failure"));
	REQUIRE(local_fs.FileExists(data_file));
	REQUIRE(local_fs.FileExists(paths.lifecycle_path));
}

TEST_CASE("Expired direct-write cleanup accepts qualified file-only listings",
          "[distributed][copy][lifecycle][object-storage]") {
	CopyFinalizeTestDirectory test_dir("copy_finalize_qualified_file_listing");
	auto &local_fs = test_dir.fs;
	auto base_path = local_fs.JoinPath(test_dir.path, "out");
	const string run_id = "run-qualified";

	REQUIRE(WriteDistributedCopyDirectWriteLifecycle(local_fs, base_path, run_id, 1).is_ok());
	auto data_file = local_fs.JoinPath(base_path, run_id + "_w_failed_part.parquet");
	WriteTestFile(local_fs, data_file, "stale");

	FileOnlyRecursiveListFileSystem qualified_fs(true);
	auto cleanup_res = CleanupExpiredDistributedCopyDirectWriteRuns(qualified_fs, base_path, 1, 10);
	REQUIRE(cleanup_res.is_ok());
	REQUIRE(cleanup_res.value().scanned_runs == 1);
	REQUIRE(cleanup_res.value().cleaned_runs == 1);
	REQUIRE(cleanup_res.value().errors == 0);
	REQUIRE_FALSE(local_fs.FileExists(data_file));
	auto paths = BuildDistributedCopyFinalizeCommitPaths(local_fs, base_path, run_id);
	REQUIRE_FALSE(local_fs.FileExists(paths.lifecycle_path));
}

TEST_CASE("Direct-write cleanup keeps lifecycle registration until metadata cleanup finishes",
          "[distributed][copy][lifecycle][object-storage]") {
	CopyFinalizeTestDirectory test_dir("copy_finalize_retryable_metadata_cleanup");
	auto &local_fs = test_dir.fs;
	auto base_path = local_fs.JoinPath(test_dir.path, "out");
	const string run_id = "run-retry-metadata";

	REQUIRE(WriteDistributedCopyDirectWriteLifecycle(local_fs, base_path, run_id, 1).is_ok());
	auto data_file = local_fs.JoinPath(base_path, run_id + "_w_failed_part.parquet");
	WriteTestFile(local_fs, data_file, "stale");
	auto paths = BuildDistributedCopyFinalizeCommitPaths(local_fs, base_path, run_id);
	WriteTestFile(local_fs, paths.manifest_path, "partial");

	FileRemovalFailureFileSystem failing_fs(paths.manifest_path);
	auto first_cleanup = CleanupDistributedCopyUncommittedDirectWriteRun(failing_fs, base_path, run_id);
	REQUIRE(first_cleanup.is_err());
	REQUIRE(StringUtil::Contains(first_cleanup.error().what(), "injected object removal failure"));
	REQUIRE_FALSE(local_fs.FileExists(data_file));
	REQUIRE(local_fs.FileExists(paths.manifest_path));
	REQUIRE(local_fs.FileExists(paths.lifecycle_path));

	FileOnlyRecursiveListFileSystem retry_fs;
	auto retry_cleanup = CleanupExpiredDistributedCopyDirectWriteRuns(retry_fs, base_path, 1, 10);
	REQUIRE(retry_cleanup.is_ok());
	REQUIRE(retry_cleanup.value().scanned_runs == 1);
	REQUIRE(retry_cleanup.value().cleaned_runs == 1);
	REQUIRE(retry_cleanup.value().errors == 0);
	REQUIRE_FALSE(local_fs.FileExists(paths.manifest_path));
	REQUIRE_FALSE(local_fs.FileExists(paths.lifecycle_path));
}
