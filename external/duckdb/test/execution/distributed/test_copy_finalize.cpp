// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"
#include "test_helpers.hpp"

#include "duckdb/common/local_file_system.hpp"
#include "duckdb/common/virtual_file_system.hpp"
#include "duckdb/execution/distributed/copy_finalize.hpp"

#include <chrono>
#include <thread>

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

class MissingPrefixFileOnlyRecursiveListFileSystem : public FileOnlyRecursiveListFileSystem {
public:
	explicit MissingPrefixFileOnlyRecursiveListFileSystem(string missing_prefix)
	    : missing_prefix(std::move(missing_prefix)) {
	}

	bool ListFiles(const string &directory, const std::function<void(const string &, bool)> &callback,
	               FileOpener *opener = nullptr) override {
		if (directory == missing_prefix) {
			throw IOException({{"errno", std::to_string(ENOENT)}}, "injected missing prefix");
		}
		return FileOnlyRecursiveListFileSystem::ListFiles(directory, callback, opener);
	}

private:
	string missing_prefix;
};

class ErrnoListFileSystem : public LocalFileSystem {
public:
	explicit ErrnoListFileSystem(int error_number, bool emit_entry = false)
	    : error_number(error_number), emit_entry(emit_entry) {
	}

	bool ListFiles(const string &, const std::function<void(const string &, bool)> &callback,
	               FileOpener * = nullptr) override {
		if (emit_entry) {
			callback("partial", false);
		}
		throw IOException({{"errno", std::to_string(error_number)}}, "injected list failure");
	}

private:
	int error_number;
	bool emit_entry;
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

class CoordinatorHiddenWorkerFileSystem : public LocalFileSystem {
public:
	explicit CoordinatorHiddenWorkerFileSystem(string hidden_path) : hidden_path(std::move(hidden_path)) {
	}

	bool FileExists(const string &path, optional_ptr<FileOpener> opener = nullptr) override {
		if (path == hidden_path) {
			return false;
		}
		return LocalFileSystem::FileExists(path, opener);
	}

private:
	string hidden_path;
};

class MappedRemoteFileSystem : public LocalFileSystem {
public:
	explicit MappedRemoteFileSystem(string local_root, string remote_prefix = "s3://bucket")
	    : local_root(std::move(local_root)), remote_prefix(std::move(remote_prefix)) {
	}

	unique_ptr<FileHandle> OpenFile(const string &path, FileOpenFlags flags,
	                                optional_ptr<FileOpener> opener = nullptr) override {
		return backing_fs.OpenFile(MapPath(path), flags, opener);
	}

	bool DirectoryExists(const string &path, optional_ptr<FileOpener> opener = nullptr) override {
		return backing_fs.DirectoryExists(MapPath(path), opener);
	}

	void CreateDirectory(const string &path, optional_ptr<FileOpener> opener = nullptr) override {
		backing_fs.CreateDirectory(MapPath(path), opener);
	}

	void CreateDirectoriesRecursive(const string &path, optional_ptr<FileOpener> opener = nullptr) override {
		backing_fs.CreateDirectoriesRecursive(MapPath(path), opener);
	}

	void RemoveDirectory(const string &path, optional_ptr<FileOpener> opener = nullptr) override {
		backing_fs.RemoveDirectory(MapPath(path), opener);
	}

	bool ListFiles(const string &path, const std::function<void(const string &, bool)> &callback,
	               FileOpener *opener = nullptr) override {
		return backing_fs.ListFiles(MapPath(path), callback, opener);
	}

	void MoveFile(const string &source, const string &target, optional_ptr<FileOpener> opener = nullptr) override {
		backing_fs.MoveFile(MapPath(source), MapPath(target), opener);
	}

	bool FileExists(const string &path, optional_ptr<FileOpener> opener = nullptr) override {
		return backing_fs.FileExists(MapPath(path), opener);
	}

	void RemoveFile(const string &path, optional_ptr<FileOpener> opener = nullptr) override {
		if (fail_removal && path == failed_removal_path) {
			throw IOException("injected remote object removal failure");
		}
		backing_fs.RemoveFile(MapPath(path), opener);
	}

	bool TryRemoveFile(const string &path, optional_ptr<FileOpener> opener = nullptr) override {
		return backing_fs.TryRemoveFile(MapPath(path), opener);
	}

	bool CanHandleFile(const string &path) override {
		return StringUtil::StartsWith(path, remote_prefix);
	}

	bool IsLocalFileSystem() const override {
		return false;
	}

	void FailRemovalOf(string path) {
		failed_removal_path = std::move(path);
		fail_removal = true;
	}

	void AllowRemoval() {
		fail_removal = false;
	}

	string GetName() const override {
		return "MappedRemoteFileSystem";
	}

protected:
	bool ListFilesExtended(const string &path, const std::function<void(OpenFileInfo &info)> &callback,
	                       optional_ptr<FileOpener> opener) override {
		return backing_fs.ListFiles(MapPath(path), callback, opener);
	}

private:
	string MapPath(const string &path) const {
		if (!StringUtil::StartsWith(path, remote_prefix)) {
			throw InternalException("unexpected mapped remote path: " + path);
		}
		return local_root + path.substr(remote_prefix.size());
	}

	LocalFileSystem backing_fs;
	string local_root;
	string remote_prefix;
	string failed_removal_path;
	bool fail_removal = false;
};

class PhantomRemoteDirectoryFileSystem : public MappedRemoteFileSystem {
public:
	PhantomRemoteDirectoryFileSystem(string local_root, string remote_prefix, string phantom_directory)
	    : MappedRemoteFileSystem(std::move(local_root), std::move(remote_prefix)),
	      phantom_directory(std::move(phantom_directory)) {
	}

	bool DirectoryExists(const string &path, optional_ptr<FileOpener> opener = nullptr) override {
		if (path == phantom_directory) {
			return true;
		}
		return MappedRemoteFileSystem::DirectoryExists(path, opener);
	}

	void RemoveDirectory(const string &path, optional_ptr<FileOpener> opener = nullptr) override {
		if (path == phantom_directory) {
			return;
		}
		MappedRemoteFileSystem::RemoveDirectory(path, opener);
	}

private:
	string phantom_directory;
};

class WindowsPathFileSystem : public LocalFileSystem {
public:
	string PathSeparator(const string &) override {
		return "\\";
	}

	bool IsPathAbsolute(const string &path) override {
		return StringUtil::StartsWith(path, "\\\\") ||
		       (path.size() >= 3 && path[1] == ':' && (path[2] == '\\' || path[2] == '/'));
	}
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
		if (fail_removal && path == failed_path) {
			throw IOException("injected object removal failure");
		}
		LocalFileSystem::RemoveFile(path, opener);
	}

	void AllowRemoval() {
		fail_removal = false;
	}

private:
	string failed_path;
	bool fail_removal = true;
};

class DelayedFileRemovalFileSystem : public FileOnlyRecursiveListFileSystem {
public:
	DelayedFileRemovalFileSystem(string delayed_path, std::chrono::milliseconds delay)
	    : delayed_path(std::move(delayed_path)), delay(delay) {
	}

	void RemoveFile(const string &path, optional_ptr<FileOpener> opener = nullptr) override {
		if (path == delayed_path) {
			std::this_thread::sleep_for(delay);
		}
		LocalFileSystem::RemoveFile(path, opener);
	}

private:
	string delayed_path;
	std::chrono::milliseconds delay;
};

class DirectoryRemovalFailureFileSystem : public LocalFileSystem {
public:
	explicit DirectoryRemovalFailureFileSystem(string failed_path) : failed_path(std::move(failed_path)) {
	}

	void RemoveDirectory(const string &directory, optional_ptr<FileOpener> opener = nullptr) override {
		if (fail_removal && directory == failed_path) {
			throw IOException("injected directory removal failure");
		}
		LocalFileSystem::RemoveDirectory(directory, opener);
	}

	void AllowRemoval() {
		fail_removal = false;
	}

private:
	string failed_path;
	bool fail_removal = true;
};

enum class MarkerReadbackMode : uint8_t { VISIBLE, MISSING, ERROR };

class PersistThenFailMarkerFileSystem : public LocalFileSystem {
public:
	PersistThenFailMarkerFileSystem(string marker_path, MarkerReadbackMode readback_mode)
	    : marker_path(std::move(marker_path)), readback_mode(readback_mode) {
	}

	void MoveFile(const string &source, const string &target, optional_ptr<FileOpener> opener = nullptr) override {
		LocalFileSystem::MoveFile(source, target, opener);
		if (target == marker_path) {
			marker_write_failed = true;
			throw IOException("injected lost committed-marker response");
		}
	}

	bool FileExists(const string &path, optional_ptr<FileOpener> opener = nullptr) override {
		if (marker_write_failed && path == marker_path) {
			switch (readback_mode) {
			case MarkerReadbackMode::VISIBLE:
				break;
			case MarkerReadbackMode::MISSING:
				return false;
			case MarkerReadbackMode::ERROR:
				throw IOException("injected committed-marker readback failure");
			}
		}
		return LocalFileSystem::FileExists(path, opener);
	}

private:
	string marker_path;
	MarkerReadbackMode readback_mode;
	bool marker_write_failed = false;
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
	auto root_run_dir = BuildCopyDirectWriteRunDirectory(root, "run-root", root);
	REQUIRE(root_run_dir == root + "_vane_direct_write_run-root");
	REQUIRE(BuildCopyDirectWriteTaskDirectory(root, "run-root", "w_0", root) == root_run_dir + root + "w_0");
	auto root_direct_target = BuildCopyDirectTargetFilePath(root, "run-root", "w_0", "part.parquet");
	REQUIRE(root_direct_target == root + "run-root_w_0_part.parquet");
	REQUIRE(DistributedCopyPathIsInDirectory(root_direct_target, root, root));
	REQUIRE(DistributedCopyDirectWriteFinalPathBelongsToRun(fs, root, "run-root", root_direct_target));

	auto authority_root_res = CanonicalDistributedCopyBasePath(fs, "s3://bucket///");
	REQUIRE(authority_root_res.is_ok());
	REQUIRE(authority_root_res.value() == "s3://bucket/");
	auto authority_root_without_separator_res = CanonicalDistributedCopyBasePath(fs, "s3://bucket");
	REQUIRE(authority_root_without_separator_res.is_ok());
	REQUIRE(authority_root_without_separator_res.value() == "s3://bucket/");
	auto authority_paths =
	    BuildDistributedCopyFinalizeCommitPaths(fs, authority_root_res.value(), "run-authority-root");
	REQUIRE(authority_paths.commit_dir == "s3://bucket/.duckdb_commit/run-authority-root");
	REQUIRE(BuildCopyDirectWriteRunDirectory(authority_root_res.value(), "run-authority-root") ==
	        "s3://bucket/_vane_direct_write_run-authority-root");
	auto authority_direct_target =
	    BuildCopyDirectTargetFilePath(authority_root_res.value(), "run-authority-root", "w_0", "part.parquet");
	REQUIRE(DistributedCopyDirectWriteFinalPathBelongsToRun(fs, authority_root_res.value(), "run-authority-root",
	                                                        authority_direct_target));
	auto authority_prefix_res = CanonicalDistributedCopyBasePath(fs, "s3://bucket/prefix///");
	REQUIRE(authority_prefix_res.is_ok());
	REQUIRE(authority_prefix_res.value() == "s3://bucket/prefix");
	auto empty_authority_root_res = CanonicalDistributedCopyBasePath(fs, "file:////");
	REQUIRE(empty_authority_root_res.is_ok());
	REQUIRE(empty_authority_root_res.value() == "file:///");
	REQUIRE(BuildCopyDirectWriteRunDirectory(empty_authority_root_res.value(), "run-file-root") ==
	        "file:///_vane_direct_write_run-file-root");
	auto file_root_direct_target =
	    BuildCopyDirectTargetFilePath(empty_authority_root_res.value(), "run-file-root", "w_0", "part.parquet");
	REQUIRE(DistributedCopyDirectWriteFinalPathBelongsToRun(fs, empty_authority_root_res.value(), "run-file-root",
	                                                        file_root_direct_target));

	WindowsPathFileSystem windows_fs;
	auto unc_root_res = CanonicalDistributedCopyBasePath(windows_fs, R"(\\server\share)");
	REQUIRE(unc_root_res.is_ok());
	REQUIRE(unc_root_res.value() == R"(\\server\share\)");
	auto unc_paths = BuildDistributedCopyFinalizeCommitPaths(windows_fs, unc_root_res.value(), "run-unc-root");
	REQUIRE(unc_paths.commit_dir == R"(\\server\share\.duckdb_commit\run-unc-root)");
	auto unc_prefix_res = CanonicalDistributedCopyBasePath(windows_fs, R"(\\server\share\prefix\\)");
	REQUIRE(unc_prefix_res.is_ok());
	REQUIRE(unc_prefix_res.value() == R"(\\server\share\prefix)");
	auto drive_root_res = CanonicalDistributedCopyBasePath(windows_fs, R"(C:\\)");
	REQUIRE(drive_root_res.is_ok());
	REQUIRE(drive_root_res.value() == R"(C:\)");
	REQUIRE(NormalizeCopyDirectWriteRoot(drive_root_res.value(), R"(\)") == R"(C:\)");
	REQUIRE(BuildCopyDirectWriteRunDirectory(drive_root_res.value(), "", R"(\)") == R"(C:\)");
	REQUIRE(BuildCopyDirectWriteRunDirectory(drive_root_res.value(), "run-drive-root", R"(\)") ==
	        R"(C:\_vane_direct_write_run-drive-root)");
	REQUIRE_FALSE(DistributedCopyPathIsInDirectory(R"(C:)", drive_root_res.value(), R"(\)"));
	REQUIRE(DistributedCopyPathIsInDirectory(R"(C:\run-drive-root_w_0_part.parquet)", drive_root_res.value(), R"(\)"));
	auto forward_slash_drive_root_res = CanonicalDistributedCopyBasePath(windows_fs, "C:////");
	REQUIRE(forward_slash_drive_root_res.is_ok());
	REQUIRE(forward_slash_drive_root_res.value() == R"(C:\)");

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

	CoordinatorHiddenWorkerFileSystem coordinator_fs(worker_file);
	auto node_local_res = ReadCommittedDistributedCopyDirectWriteResult(coordinator_fs, output_path, run_id);
	REQUIRE(node_local_res.is_ok());
	REQUIRE(node_local_res.value().rows_copied == 2);
	REQUIRE(node_local_res.value().files[0].final_path == worker_file);

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

TEST_CASE("Distributed COPY direct-target empty result removes loser files before commit",
          "[distributed][copy][lifecycle]") {
	CopyFinalizeTestDirectory test_dir("copy_finalize_empty_direct_target");
	auto &fs = test_dir.fs;
	auto base_path = fs.JoinPath(test_dir.path, "out");
	const string run_id = "run-empty";
	auto loser_file = BuildCopyDirectTargetFilePath(base_path, run_id, "w_loser", "part.parquet");
	WriteTestFile(fs, loser_file, "loser");
	REQUIRE(WriteDistributedCopyDirectWriteLifecycle(fs, base_path, run_id).is_ok());

	DuckDB db(nullptr);
	Connection connection(db);
	DistributedCopySpec spec;
	spec.file_path = base_path;
	spec.file_extension = "parquet";
	spec.per_thread_output = true;

	auto finalize_res = FinalizeCopyFiles(spec, "", {}, *connection.context, run_id);

	REQUIRE(finalize_res.is_ok());
	REQUIRE(finalize_res.value().output_committed);
	REQUIRE(finalize_res.value().files.empty());
	auto commit_paths = BuildDistributedCopyFinalizeCommitPaths(fs, base_path, run_id);
	REQUIRE(fs.FileExists(commit_paths.committed_marker_path));
	REQUIRE_FALSE(fs.FileExists(loser_file));
}

TEST_CASE("Distributed COPY direct-target cleanup failure prevents commit",
          "[distributed][copy][lifecycle][object-storage]") {
	CopyFinalizeTestDirectory test_dir("copy_finalize_cleanup_before_commit");
	auto &local_fs = test_dir.fs;
	auto base_path = local_fs.JoinPath(test_dir.path, "out");
	const string run_id = "run-cleanup-failure";
	auto selected_file = BuildCopyDirectTargetFilePath(base_path, run_id, "w_selected", "part.parquet");
	auto loser_file = BuildCopyDirectTargetFilePath(base_path, run_id, "w_loser", "part.parquet");
	const string selected_contents = "selected";
	WriteTestFile(local_fs, selected_file, selected_contents);
	WriteTestFile(local_fs, loser_file, "loser");

	DBConfig config;
	auto failure_fs = make_uniq<FileRemovalFailureFileSystem>(loser_file);
	auto failure_fs_ptr = failure_fs.get();
	config.file_system = make_uniq<VirtualFileSystem>(std::move(failure_fs));
	DuckDB db(nullptr, &config);
	Connection connection(db);
	auto &fs = FileSystem::GetFileSystem(*connection.context);
	REQUIRE(WriteDistributedCopyDirectWriteLifecycle(fs, base_path, run_id).is_ok());

	DistributedCopySpec spec;
	spec.file_path = base_path;
	spec.file_extension = "parquet";
	spec.per_thread_output = true;
	DistributedCopyFileInfo selected_info;
	selected_info.staging_path = selected_file;
	selected_info.row_count = 1;
	selected_info.file_size_bytes = selected_contents.size();
	vector<DistributedCopyFileInfo> selected_files;
	selected_files.push_back(std::move(selected_info));

	auto finalize_res = FinalizeCopyFiles(spec, "", std::move(selected_files), *connection.context, run_id);

	REQUIRE(finalize_res.is_err());
	REQUIRE(StringUtil::Contains(finalize_res.error().what(), "injected object removal failure"));
	auto commit_paths = BuildDistributedCopyFinalizeCommitPaths(fs, base_path, run_id);
	REQUIRE(fs.FileExists(commit_paths.manifest_path));
	REQUIRE_FALSE(fs.FileExists(commit_paths.committed_marker_path));
	REQUIRE(fs.FileExists(selected_file));
	REQUIRE(fs.FileExists(loser_file));

	failure_fs_ptr->AllowRemoval();
	auto retry_res = FinalizeCopyFiles(spec, "", {}, *connection.context, run_id);
	REQUIRE(retry_res.is_ok());
	REQUIRE(retry_res.value().output_committed);
	REQUIRE(retry_res.value().files.size() == 1);
	REQUIRE(fs.FileExists(commit_paths.committed_marker_path));
	REQUIRE(fs.FileExists(selected_file));
	REQUIRE_FALSE(fs.FileExists(loser_file));
}

TEST_CASE("Distributed COPY direct-target cleanup time is excluded from finalize time",
          "[distributed][copy][lifecycle][object-storage]") {
	CopyFinalizeTestDirectory test_dir("copy_finalize_exclusive_timing");
	auto &local_fs = test_dir.fs;
	auto base_path = local_fs.JoinPath(test_dir.path, "out");
	const string run_id = "run-exclusive-timing";
	auto selected_file = BuildCopyDirectTargetFilePath(base_path, run_id, "w_selected", "part.parquet");
	auto loser_file = BuildCopyDirectTargetFilePath(base_path, run_id, "w_loser", "part.parquet");
	const string selected_contents = "selected";
	WriteTestFile(local_fs, selected_file, selected_contents);
	WriteTestFile(local_fs, loser_file, "loser");

	DBConfig config;
	auto delayed_fs = make_uniq<DelayedFileRemovalFileSystem>(loser_file, std::chrono::milliseconds(150));
	config.file_system = make_uniq<VirtualFileSystem>(std::move(delayed_fs));
	DuckDB db(nullptr, &config);
	Connection connection(db);
	auto &fs = FileSystem::GetFileSystem(*connection.context);
	REQUIRE(WriteDistributedCopyDirectWriteLifecycle(fs, base_path, run_id).is_ok());

	DistributedCopySpec spec;
	spec.file_path = base_path;
	spec.file_extension = "parquet";
	spec.per_thread_output = true;
	DistributedCopyFileInfo selected_info;
	selected_info.staging_path = selected_file;
	selected_info.row_count = 1;
	selected_info.file_size_bytes = selected_contents.size();
	vector<DistributedCopyFileInfo> selected_files;
	selected_files.push_back(std::move(selected_info));

	auto wall_started = std::chrono::steady_clock::now();
	auto finalize_res = FinalizeCopyFiles(spec, "", std::move(selected_files), *connection.context, run_id);
	auto wall_ms = DistributedCopyElapsedMillis(wall_started);

	REQUIRE(finalize_res.is_ok());
	const auto &result = finalize_res.value();
	REQUIRE(result.cleanup_ms >= 100);
	REQUIRE(result.finalize_ms + result.cleanup_ms <= wall_ms + 10);
	REQUIRE_FALSE(fs.FileExists(loser_file));
}

TEST_CASE("Distributed COPY accepts a committed marker after its write response is lost",
          "[distributed][copy][lifecycle][object-storage]") {
	CopyFinalizeTestDirectory test_dir("copy_finalize_marker_response_lost");
	auto &local_fs = test_dir.fs;
	auto base_path = local_fs.JoinPath(test_dir.path, "out");
	const string run_id = "run-marker-response-lost";
	auto selected_file = BuildCopyDirectTargetFilePath(base_path, run_id, "w_selected", "part.parquet");
	const string selected_contents = "selected";
	WriteTestFile(local_fs, selected_file, selected_contents);
	REQUIRE(WriteDistributedCopyDirectWriteLifecycle(local_fs, base_path, run_id).is_ok());
	auto commit_paths = BuildDistributedCopyFinalizeCommitPaths(local_fs, base_path, run_id);

	DBConfig config;
	config.file_system = make_uniq<VirtualFileSystem>(
	    make_uniq<PersistThenFailMarkerFileSystem>(commit_paths.committed_marker_path, MarkerReadbackMode::VISIBLE));
	DuckDB db(nullptr, &config);
	Connection connection(db);

	DistributedCopySpec spec;
	spec.file_path = base_path;
	spec.file_extension = "parquet";
	spec.per_thread_output = true;
	DistributedCopyFileInfo selected_info;
	selected_info.staging_path = selected_file;
	selected_info.row_count = 3;
	selected_info.file_size_bytes = selected_contents.size();
	auto expected_footer_size = Value::UBIGINT(17);
	auto expected_column_statistics = Value::LIST({Value("column-stats")});
	auto expected_partition_keys =
	    Value::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR, {Value("part")}, {Value("value")});
	selected_info.footer_size_bytes = expected_footer_size;
	selected_info.column_statistics = expected_column_statistics;
	selected_info.partition_keys = expected_partition_keys;
	vector<DistributedCopyFileInfo> selected_files;
	selected_files.push_back(std::move(selected_info));

	auto finalize_res = FinalizeCopyFiles(spec, "", std::move(selected_files), *connection.context, run_id);

	REQUIRE(finalize_res.is_ok());
	REQUIRE(finalize_res.value().output_committed);
	REQUIRE_FALSE(finalize_res.value().output_outcome_unknown);
	REQUIRE(finalize_res.value().rows_copied == 3);
	REQUIRE(finalize_res.value().files.size() == 1);
	REQUIRE(finalize_res.value().files[0].footer_size_bytes == expected_footer_size);
	REQUIRE(finalize_res.value().files[0].column_statistics == expected_column_statistics);
	REQUIRE(finalize_res.value().files[0].partition_keys == expected_partition_keys);
	REQUIRE(local_fs.FileExists(commit_paths.committed_marker_path));
}

TEST_CASE("Distributed COPY reports an unknown outcome when committed marker readback is inconclusive",
          "[distributed][copy][lifecycle][object-storage]") {
	for (auto readback_mode : {MarkerReadbackMode::MISSING, MarkerReadbackMode::ERROR}) {
		CopyFinalizeTestDirectory test_dir(readback_mode == MarkerReadbackMode::MISSING
		                                       ? "copy_finalize_marker_readback_missing"
		                                       : "copy_finalize_marker_readback_error");
		auto &local_fs = test_dir.fs;
		auto base_path = local_fs.JoinPath(test_dir.path, "out");
		const string run_id = "run-marker-readback-unknown";
		auto selected_file = BuildCopyDirectTargetFilePath(base_path, run_id, "w_selected", "part.parquet");
		const string selected_contents = "selected";
		WriteTestFile(local_fs, selected_file, selected_contents);
		REQUIRE(WriteDistributedCopyDirectWriteLifecycle(local_fs, base_path, run_id).is_ok());
		auto commit_paths = BuildDistributedCopyFinalizeCommitPaths(local_fs, base_path, run_id);

		DBConfig config;
		config.file_system = make_uniq<VirtualFileSystem>(
		    make_uniq<PersistThenFailMarkerFileSystem>(commit_paths.committed_marker_path, readback_mode));
		DuckDB db(nullptr, &config);
		Connection connection(db);

		DistributedCopySpec spec;
		spec.file_path = base_path;
		spec.file_extension = "parquet";
		spec.per_thread_output = true;
		DistributedCopyFileInfo selected_info;
		selected_info.staging_path = selected_file;
		selected_info.row_count = 3;
		selected_info.file_size_bytes = selected_contents.size();
		vector<DistributedCopyFileInfo> selected_files;
		selected_files.push_back(std::move(selected_info));

		auto finalize_res = FinalizeCopyFiles(spec, "", std::move(selected_files), *connection.context, run_id);

		REQUIRE(finalize_res.is_ok());
		const auto &result = finalize_res.value();
		REQUIRE_FALSE(result.output_committed);
		REQUIRE(result.output_outcome_unknown);
		REQUIRE(result.output_base_path == base_path);
		REQUIRE(result.output_run_id == run_id);
		REQUIRE(result.output_manifest_path == commit_paths.manifest_path);
		REQUIRE(result.output_committed_marker_path == commit_paths.committed_marker_path);
		REQUIRE(StringUtil::Contains(result.output_outcome_error, "injected lost committed-marker response"));
		REQUIRE_FALSE(result.output_outcome_error.empty());
		REQUIRE(local_fs.FileExists(selected_file));
		REQUIRE(local_fs.FileExists(commit_paths.committed_marker_path));
	}
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
	const string authority_root = "memory://bucket/";
	REQUIRE(ResolveDistributedCopyListedPath(fs, authority_root, "bucket/key") == "memory://bucket/key");
	REQUIRE(ResolveDistributedCopyListedPath(fs, authority_root, "/bucket/key") == "memory://bucket/key");
	REQUIRE(ResolveDistributedCopyListedPath(fs, "memory:///", "key") == "memory:///key");

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

TEST_CASE("Distributed COPY treats only not-found list failures as empty",
          "[distributed][copy][lifecycle][object-storage]") {
	ErrnoListFileSystem missing_fs(ENOENT);
	auto missing_res = ListDistributedCopyFilesUnderPrefix(missing_fs, "/missing");
	REQUIRE(missing_res.is_ok());
	REQUIRE(missing_res.value().empty());

	std::runtime_error python_missing("FileNotFoundError: /missing");
	REQUIRE(DistributedCopyExceptionIsNotFound(python_missing));

	ErrnoListFileSystem partial_missing_fs(ENOENT, true);
	auto partial_missing_res = ListDistributedCopyFilesUnderPrefix(partial_missing_fs, "/partial");
	REQUIRE(partial_missing_res.is_err());

	ErrnoListFileSystem denied_fs(EACCES);
	auto denied_res = ListDistributedCopyFilesUnderPrefix(denied_fs, "/denied");
	REQUIRE(denied_res.is_err());
	REQUIRE(StringUtil::Contains(denied_res.error().what(), "injected list failure"));
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

TEST_CASE("Expired direct-target cleanup accepts a missing legacy run prefix",
          "[distributed][copy][lifecycle][object-storage]") {
	CopyFinalizeTestDirectory test_dir("copy_finalize_missing_legacy_run_prefix");
	auto &local_fs = test_dir.fs;
	auto base_path = local_fs.JoinPath(test_dir.path, "out");
	const string run_id = "run-direct-target";

	REQUIRE(WriteDistributedCopyDirectWriteLifecycle(local_fs, base_path, run_id, 1).is_ok());
	auto data_file = BuildCopyDirectTargetFilePath(base_path, run_id, "w_failed", "part.parquet");
	WriteTestFile(local_fs, data_file, "stale");
	auto missing_run_prefix = BuildCopyDirectWriteRunDirectory(base_path, run_id, local_fs.PathSeparator(base_path));

	MissingPrefixFileOnlyRecursiveListFileSystem object_fs(missing_run_prefix);
	auto cleanup_res = CleanupExpiredDistributedCopyDirectWriteRuns(object_fs, base_path, 1, 10);
	REQUIRE(cleanup_res.is_ok());
	REQUIRE(cleanup_res.value().scanned_runs == 1);
	REQUIRE(cleanup_res.value().cleaned_runs == 1);
	REQUIRE(cleanup_res.value().errors == 0);
	REQUIRE_FALSE(local_fs.FileExists(data_file));
	auto paths = BuildDistributedCopyFinalizeCommitPaths(local_fs, base_path, run_id);
	REQUIRE_FALSE(local_fs.FileExists(paths.lifecycle_path));
}

TEST_CASE("Direct-target cleanup ignores phantom remote directories",
          "[distributed][copy][lifecycle][object-storage]") {
	CopyFinalizeTestDirectory test_dir("copy_finalize_phantom_remote_directory");
	const string remote_prefix = "custom-object://bucket";
	const string base_path = remote_prefix + "/out";
	const string run_id = "run-direct-target";
	auto run_prefix = BuildCopyDirectWriteRunDirectory(base_path, run_id, "/");
	VirtualFileSystem fs;
	fs.RegisterSubSystem(make_uniq<PhantomRemoteDirectoryFileSystem>(test_dir.fs.JoinPath(test_dir.path, "remote"),
	                                                                 remote_prefix, run_prefix));
	REQUIRE_FALSE(FileSystem::IsRemoteFile(base_path));
	REQUIRE_FALSE(fs.HasDirectorySemantics(run_prefix));

	REQUIRE(WriteDistributedCopyDirectWriteLifecycle(fs, base_path, run_id, 1).is_ok());
	auto data_file = BuildCopyDirectTargetFilePath(base_path, run_id, "w_failed", "part.parquet");
	WriteTestFile(fs, data_file, "stale");
	REQUIRE(fs.DirectoryExists(run_prefix, nullptr));

	auto cleanup_res = CleanupDistributedCopyUncommittedDirectWriteRun(fs, base_path, run_id);

	auto cleanup_error = cleanup_res.is_err() ? cleanup_res.error().what() : string();
	INFO(cleanup_error);
	REQUIRE(cleanup_res.is_ok());
	REQUIRE_FALSE(fs.FileExists(data_file, nullptr));
	auto paths = BuildDistributedCopyFinalizeCommitPaths(fs, base_path, run_id);
	REQUIRE_FALSE(fs.FileExists(paths.lifecycle_path, nullptr));
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

TEST_CASE("Direct-write cleanup requires lifecycle registration", "[distributed][copy][lifecycle]") {
	CopyFinalizeTestDirectory test_dir("copy_finalize_requires_lifecycle");
	auto &fs = test_dir.fs;
	auto base_path = fs.JoinPath(test_dir.path, "out");
	const string run_id = "run-unregistered";
	auto data_file = BuildCopyDirectTargetFilePath(base_path, run_id, "w_failed", "part.parquet");
	WriteTestFile(fs, data_file, "must survive");

	auto cleanup_res = CleanupDistributedCopyUncommittedDirectWriteRun(fs, base_path, run_id);

	REQUIRE(cleanup_res.is_err());
	REQUIRE(StringUtil::Contains(cleanup_res.error().what(), "requires lifecycle registration"));
	REQUIRE(fs.FileExists(data_file));
}

TEST_CASE("Direct-write force abort refuses node-local output without mutating the run",
          "[distributed][copy][lifecycle]") {
	CopyFinalizeTestDirectory test_dir("copy_finalize_force_abort_node_local");
	auto &local_fs = test_dir.fs;
	auto base_path = local_fs.JoinPath(test_dir.path, "out");
	const string run_id = "run-force-abort-node-local";
	auto data_file = BuildCopyDirectTargetFilePath(base_path, run_id, "w_selected", "part.parquet");
	WriteTestFile(local_fs, data_file, "discard me");
	REQUIRE(WriteDistributedCopyDirectWriteLifecycle(local_fs, base_path, run_id, 1).is_ok());
	auto paths = BuildDistributedCopyFinalizeCommitPaths(local_fs, base_path, run_id);
	WriteTestFile(local_fs, paths.manifest_path, "manifest");
	REQUIRE(WriteDistributedCopyFinalizeCommittedMarker(local_fs, paths).is_ok());

	auto abort_res = ForceAbortDistributedCopyDirectWriteRun(local_fs, base_path, run_id);

	REQUIRE(abort_res.is_err());
	REQUIRE(StringUtil::Contains(abort_res.error().what(), "cannot prove node-local worker output was removed"));
	REQUIRE(local_fs.FileExists(paths.committed_marker_path));
	REQUIRE(local_fs.FileExists(data_file));
	REQUIRE(local_fs.FileExists(paths.lifecycle_path));
	REQUIRE(local_fs.FileExists(paths.manifest_path));
}

TEST_CASE("Direct-write force abort cleans shared remote output before reporting safe retry",
          "[distributed][copy][lifecycle][object-storage]") {
	CopyFinalizeTestDirectory test_dir("copy_finalize_force_abort_remote");
	MappedRemoteFileSystem fs(test_dir.fs.JoinPath(test_dir.path, "remote"));
	const string base_path = "s3://bucket/out";
	const string run_id = "run-force-abort-remote";
	auto data_file = BuildCopyDirectTargetFilePath(base_path, run_id, "w_selected", "part.parquet");
	WriteTestFile(fs, data_file, "discard me");
	REQUIRE(WriteDistributedCopyDirectWriteLifecycle(fs, base_path, run_id, 1).is_ok());
	auto paths = BuildDistributedCopyFinalizeCommitPaths(fs, base_path, run_id);
	WriteTestFile(fs, paths.manifest_path, "manifest");
	REQUIRE(WriteDistributedCopyFinalizeCommittedMarker(fs, paths).is_ok());

	fs.FailRemovalOf(paths.committed_marker_path);
	auto failed_abort = ForceAbortDistributedCopyDirectWriteRun(fs, base_path, run_id);

	REQUIRE(failed_abort.is_err());
	REQUIRE(StringUtil::Contains(failed_abort.error().what(), "injected remote object removal failure"));
	REQUIRE(fs.FileExists(paths.committed_marker_path));
	REQUIRE(fs.FileExists(data_file));
	REQUIRE(fs.FileExists(paths.lifecycle_path));

	fs.AllowRemoval();
	auto abort_res = ForceAbortDistributedCopyDirectWriteRun(fs, base_path, run_id);

	REQUIRE(abort_res.is_ok());
	REQUIRE_FALSE(abort_res.value().skipped_committed);
	REQUIRE_FALSE(fs.FileExists(paths.committed_marker_path));
	REQUIRE_FALSE(fs.FileExists(data_file));
	REQUIRE_FALSE(fs.FileExists(paths.lifecycle_path));
	REQUIRE_FALSE(fs.DirectoryExists(paths.commit_dir));
}

TEST_CASE("Direct-write force abort preserves node-local lifecycle when the marker is absent",
          "[distributed][copy][lifecycle]") {
	CopyFinalizeTestDirectory test_dir("copy_finalize_force_abort_missing_marker");
	auto &local_fs = test_dir.fs;
	auto base_path = local_fs.JoinPath(test_dir.path, "out");
	const string run_id = "run-force-abort-missing-marker";
	auto data_file = BuildCopyDirectTargetFilePath(base_path, run_id, "w_selected", "part.parquet");
	WriteTestFile(local_fs, data_file, "discard me");
	REQUIRE(WriteDistributedCopyDirectWriteLifecycle(local_fs, base_path, run_id, 1).is_ok());
	auto paths = BuildDistributedCopyFinalizeCommitPaths(local_fs, base_path, run_id);
	WriteTestFile(local_fs, paths.manifest_path, "manifest");
	REQUIRE_FALSE(local_fs.FileExists(paths.committed_marker_path));

	auto abort_res = ForceAbortDistributedCopyDirectWriteRun(local_fs, base_path, run_id);

	REQUIRE(abort_res.is_err());
	REQUIRE(StringUtil::Contains(abort_res.error().what(), "cannot prove node-local worker output was removed"));
	REQUIRE_FALSE(local_fs.FileExists(paths.committed_marker_path));
	REQUIRE(local_fs.FileExists(data_file));
	REQUIRE(local_fs.FileExists(paths.lifecycle_path));
	REQUIRE(local_fs.FileExists(paths.manifest_path));
}

TEST_CASE("Direct-write cleanup restores lifecycle when directory removal fails", "[distributed][copy][lifecycle]") {
	CopyFinalizeTestDirectory test_dir("copy_finalize_restore_lifecycle");
	auto &local_fs = test_dir.fs;
	auto base_path = local_fs.JoinPath(test_dir.path, "out");
	const string run_id = "run-retry-directory";
	REQUIRE(WriteDistributedCopyDirectWriteLifecycle(local_fs, base_path, run_id, 1).is_ok());
	auto data_file = BuildCopyDirectTargetFilePath(base_path, run_id, "w_failed", "part.parquet");
	WriteTestFile(local_fs, data_file, "stale");
	auto paths = BuildDistributedCopyFinalizeCommitPaths(local_fs, base_path, run_id);
	WriteTestFile(local_fs, paths.manifest_path, "partial");

	DirectoryRemovalFailureFileSystem failing_fs(paths.commit_dir);
	auto first_cleanup = CleanupDistributedCopyUncommittedDirectWriteRun(failing_fs, base_path, run_id);

	REQUIRE(first_cleanup.is_err());
	REQUIRE(StringUtil::Contains(first_cleanup.error().what(), "injected directory removal failure"));
	REQUIRE(StringUtil::Contains(first_cleanup.error().what(), "lifecycle registration restored for retry"));
	REQUIRE_FALSE(local_fs.FileExists(data_file));
	REQUIRE_FALSE(local_fs.FileExists(paths.manifest_path));
	REQUIRE(local_fs.FileExists(paths.lifecycle_path));
	REQUIRE(local_fs.DirectoryExists(paths.commit_dir));

	failing_fs.AllowRemoval();
	auto retry_cleanup = CleanupDistributedCopyUncommittedDirectWriteRun(failing_fs, base_path, run_id);

	REQUIRE(retry_cleanup.is_ok());
	REQUIRE(retry_cleanup.value().commit_dir_removed);
	REQUIRE_FALSE(local_fs.FileExists(paths.lifecycle_path));
	REQUIRE_FALSE(local_fs.DirectoryExists(paths.commit_dir));
}
