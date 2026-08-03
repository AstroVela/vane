// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/exchange/shuffle_cache.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/file_opener.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/local_file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/main/client_data.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/storage/temporary_memory_manager.hpp"

#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <arrow/ipc/api.h>
#include <arrow/io/api.h>
#include <algorithm>
#include <charconv>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace duckdb {
namespace distributed {

namespace {

std::string ShuffleCacheSanitizePathComponent(const std::string &value) {
	std::string out = value;
	for (auto &ch : out) {
		if (ch == '/' || ch == '\\') {
			ch = '_';
		}
	}
	return out;
}

bool ShuffleCacheLooksLikeUnsupportedObjectPath(const std::string &path) {
	auto scheme_end = path.find("://");
	if (scheme_end == std::string::npos) {
		return false;
	}
	auto scheme = path.substr(0, scheme_end);
	std::transform(scheme.begin(), scheme.end(), scheme.begin(),
	               [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
	return scheme != "file";
}

DuckDBError ShuffleCacheArrowToError(const arrow::Status &status, const std::string &context) {
	return DuckDBError::external_error(context + ": " + status.ToString());
}

DuckDBResult<std::shared_ptr<arrow::io::OutputStream>> OpenPosixArrowOutput(const std::string &path) {
	auto out_res = arrow::io::FileOutputStream::Open(path);
	if (!out_res.ok()) {
		return DuckDBResult<std::shared_ptr<arrow::io::OutputStream>>::err(
		    ShuffleCacheArrowToError(out_res.status(), "open shuffle output"));
	}
	return DuckDBResult<std::shared_ptr<arrow::io::OutputStream>>::ok(std::move(out_res).ValueOrDie());
}

DuckDBResult<std::shared_ptr<arrow::io::InputStream>> OpenPosixArrowInput(const std::string &path) {
	auto in_res = arrow::io::ReadableFile::Open(path);
	if (!in_res.ok()) {
		return DuckDBResult<std::shared_ptr<arrow::io::InputStream>>::err(
		    ShuffleCacheArrowToError(in_res.status(), "open shuffle input"));
	}
	return DuckDBResult<std::shared_ptr<arrow::io::InputStream>>::ok(std::move(in_res).ValueOrDie());
}

DuckDBResult<vector<string>> ResolveColumnNames(idx_t count, const vector<string> &names) {
	if (!names.empty()) {
		if (names.size() != count) {
			return DuckDBResult<vector<string>>::err(
			    DuckDBError::value_error("shuffle cache column name count mismatch"));
		}
		return DuckDBResult<vector<string>>::ok(names);
	}
	vector<string> resolved;
	resolved.reserve(count);
	for (idx_t idx = 0; idx < count; idx++) {
		resolved.push_back("c" + std::to_string(idx));
	}
	return DuckDBResult<vector<string>>::ok(std::move(resolved));
}

bool IsAggregateStateType(const LogicalType &type) {
	return type.id() == LogicalTypeId::AGGREGATE_STATE;
}

vector<LogicalType> ToArrowTypes(const vector<LogicalType> &types) {
	vector<LogicalType> result;
	result.reserve(types.size());
	for (auto &type : types) {
		if (IsAggregateStateType(type)) {
			result.push_back(LogicalType::BLOB);
		} else {
			result.push_back(type);
		}
	}
	return result;
}

unsigned long long ParsePositiveByteCount(const char *name, const char *raw) {
	unsigned long long value = 0;
	auto end = raw + std::strlen(raw);
	auto result = std::from_chars(raw, end, value, 10);
	if (result.ec != std::errc() || result.ptr != end || value == 0) {
		throw InvalidInputException(std::string(name) + " must be a positive integer byte count");
	}
	return value;
}

idx_t ResolveShuffleCacheFlushThresholdBytes() {
	const char *raw = std::getenv("VANE_SHUFFLE_CACHE_FLUSH_THRESHOLD_BYTES");
	if (!raw || !*raw) {
		return ShuffleCache::DEFAULT_FLUSH_THRESHOLD_BYTES;
	}

	auto value = ParsePositiveByteCount("VANE_SHUFFLE_CACHE_FLUSH_THRESHOLD_BYTES", raw);
	if (value > static_cast<unsigned long long>(std::numeric_limits<idx_t>::max())) {
		throw InvalidInputException("VANE_SHUFFLE_CACHE_FLUSH_THRESHOLD_BYTES exceeds idx_t max");
	}
	return static_cast<idx_t>(value);
}

optional_idx ResolveShuffleCacheMaxBufferedBytes() {
	const char *raw = std::getenv("VANE_SHUFFLE_CACHE_MAX_BUFFERED_BYTES");
	if (!raw || !*raw) {
		return optional_idx();
	}

	auto value = ParsePositiveByteCount("VANE_SHUFFLE_CACHE_MAX_BUFFERED_BYTES", raw);
	if (value >= static_cast<unsigned long long>(std::numeric_limits<idx_t>::max())) {
		throw InvalidInputException("VANE_SHUFFLE_CACHE_MAX_BUFFERED_BYTES must be less than idx_t max");
	}
	return optional_idx(static_cast<idx_t>(value));
}

void CastChunk(ClientContext &context, DataChunk &input, DataChunk &output, const vector<LogicalType> &target_types) {
	output.SetCardinality(input.size());
	for (idx_t col = 0; col < target_types.size(); col++) {
		if (input.data[col].GetType() == target_types[col]) {
			output.data[col].Reference(input.data[col]);
		} else {
			VectorOperations::Cast(context, input.data[col], output.data[col], input.size());
		}
	}
}

DuckDBResult<idx_t> ParseManifestIdx(const std::string &value, const std::string &field) {
	if (value.empty()) {
		return DuckDBResult<idx_t>::err(
		    DuckDBError::value_error("shuffle attempt manifest empty numeric field: " + field));
	}
	for (auto ch : value) {
		if (ch < '0' || ch > '9') {
			return DuckDBResult<idx_t>::err(
			    DuckDBError::value_error("shuffle attempt manifest invalid numeric field: " + field));
		}
	}
	try {
		size_t parsed_chars = 0;
		auto parsed = std::stoull(value, &parsed_chars);
		if (parsed_chars != value.size() ||
		    parsed > static_cast<unsigned long long>(std::numeric_limits<idx_t>::max())) {
			return DuckDBResult<idx_t>::err(
			    DuckDBError::value_error("shuffle attempt manifest invalid numeric field: " + field));
		}
		return DuckDBResult<idx_t>::ok(static_cast<idx_t>(parsed));
	} catch (const std::exception &) {
		return DuckDBResult<idx_t>::err(
		    DuckDBError::value_error("shuffle attempt manifest invalid numeric field: " + field));
	}
}

std::vector<std::string> SplitManifestFileFields(const std::string &value) {
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

DuckDBResult<void> CheckDuplicateManifestField(bool &seen, const std::string &field) {
	if (seen) {
		return DuckDBResult<void>::err(DuckDBError::value_error("shuffle attempt manifest duplicate field: " + field));
	}
	seen = true;
	return DuckDBResult<void>::ok();
}

class PosixShuffleStorage final : public ShuffleStorage {
public:
	DuckDBResult<void> CreateDirectories(const std::string &path) const override {
		try {
			fs_.CreateDirectoriesRecursive(path);
		} catch (const std::exception &ex) {
			return DuckDBResult<void>::err(
			    DuckDBError::io_error("failed to create shuffle directory: " + std::string(ex.what())));
		}
		return DuckDBResult<void>::ok();
	}

	bool IsRegularFile(const std::string &path) const override {
		return fs_.FileExists(path);
	}

	DuckDBResult<idx_t> FileSize(const std::string &path) const override {
		try {
			auto handle = fs_.OpenFile(path, FileOpenFlags(FileOpenFlags::FILE_FLAGS_READ));
			return DuckDBResult<idx_t>::ok(handle->GetFileSize());
		} catch (const std::exception &ex) {
			return DuckDBResult<idx_t>::err(
			    DuckDBError::io_error("failed to stat shuffle file: " + std::string(ex.what())));
		}
	}

	DuckDBResult<void> WriteTextFileAtomically(const std::string &path, const std::string &contents) const override {
		auto tmp_path = path + ".tmp";
		{
			std::ofstream output(tmp_path, std::ios::out | std::ios::trunc);
			if (!output) {
				return DuckDBResult<void>::err(DuckDBError::io_error("failed to open shuffle text file: " + tmp_path));
			}
			output << contents;
			output.close();
			if (!output) {
				return DuckDBResult<void>::err(DuckDBError::io_error("failed to write shuffle text file: " + tmp_path));
			}
		}

		try {
			fs_.TryRemoveFile(path);
			fs_.MoveFile(tmp_path, path);
		} catch (const std::exception &ex) {
			fs_.TryRemoveFile(tmp_path);
			return DuckDBResult<void>::err(
			    DuckDBError::io_error("failed to commit shuffle text file: " + std::string(ex.what())));
		}
		return DuckDBResult<void>::ok();
	}

	DuckDBResult<std::string> ReadTextFile(const std::string &path) const override {
		std::ifstream input(path, std::ios::in | std::ios::binary);
		if (!input.good()) {
			return DuckDBResult<std::string>::err(DuckDBError::io_error("failed to open shuffle text file: " + path));
		}
		std::ostringstream contents;
		contents << input.rdbuf();
		if (input.bad()) {
			return DuckDBResult<std::string>::err(DuckDBError::io_error("failed to read shuffle text file: " + path));
		}
		return DuckDBResult<std::string>::ok(contents.str());
	}

	DuckDBResult<idx_t> RemoveAll(const std::string &path) const override {
		return RemoveAllRecursive(path);
	}

	DuckDBResult<std::shared_ptr<arrow::io::OutputStream>> OpenArrowOutput(const std::string &path) const override {
		return OpenPosixArrowOutput(path);
	}

	DuckDBResult<std::shared_ptr<arrow::io::InputStream>> OpenArrowInput(const std::string &path) const override {
		return OpenPosixArrowInput(path);
	}

private:
	DuckDBResult<idx_t> RemoveAllRecursive(const std::string &path) const {
		if (path.empty()) {
			return DuckDBResult<idx_t>::ok(0);
		}

		idx_t removed = 0;
		try {
			if (fs_.FileExists(path)) {
				fs_.RemoveFile(path);
				return DuckDBResult<idx_t>::ok(1);
			}
		} catch (...) {
		}

		try {
			if (!fs_.DirectoryExists(path)) {
				return DuckDBResult<idx_t>::ok(0);
			}
		} catch (...) {
			return DuckDBResult<idx_t>::ok(0);
		}

		std::vector<std::string> child_dirs;
		try {
			fs_.ListFiles(path, [&](const std::string &child, bool is_dir) {
				auto full_path = fs_.JoinPath(path, child);
				if (is_dir) {
					child_dirs.push_back(full_path);
					return;
				}
				fs_.RemoveFile(full_path);
				removed++;
			});
			for (const auto &child_dir : child_dirs) {
				auto child_res = RemoveAllRecursive(child_dir);
				if (child_res.is_err()) {
					return child_res;
				}
				removed += child_res.value();
			}
			fs_.RemoveDirectory(path);
			removed++;
			return DuckDBResult<idx_t>::ok(removed);
		} catch (const std::exception &ex) {
			return DuckDBResult<idx_t>::err(
			    DuckDBError::io_error("failed to remove shuffle attempt storage: " + std::string(ex.what())));
		}
	}

	mutable LocalFileSystem fs_;
};

class DuckDBFileSystemArrowOutputStream final : public arrow::io::OutputStream {
public:
	explicit DuckDBFileSystemArrowOutputStream(unique_ptr<FileHandle> handle) : handle_(std::move(handle)) {
		set_mode(arrow::io::FileMode::WRITE);
	}

	arrow::Status Close() override {
		if (closed_) {
			return arrow::Status::OK();
		}
		try {
			if (handle_) {
				handle_->Sync();
				handle_->Close();
				handle_.reset();
			}
			closed_ = true;
			return arrow::Status::OK();
		} catch (const std::exception &ex) {
			return arrow::Status::IOError("close DuckDB shuffle output: ", ex.what());
		}
	}

	bool closed() const override {
		return closed_;
	}

	arrow::Result<int64_t> Tell() const override {
		return position_;
	}

	arrow::Status Write(const void *data, int64_t nbytes) override {
		if (closed_) {
			return arrow::Status::Invalid("DuckDB shuffle output is closed");
		}
		if (nbytes < 0) {
			return arrow::Status::Invalid("DuckDB shuffle output write size is negative");
		}
		if (nbytes == 0) {
			return arrow::Status::OK();
		}
		try {
			auto written = handle_->Write(const_cast<void *>(data), static_cast<idx_t>(nbytes));
			if (written != nbytes) {
				return arrow::Status::IOError("short write to DuckDB shuffle output");
			}
			position_ += written;
			return arrow::Status::OK();
		} catch (const std::exception &ex) {
			return arrow::Status::IOError("write DuckDB shuffle output: ", ex.what());
		}
	}

	arrow::Status Flush() override {
		if (closed_) {
			return arrow::Status::OK();
		}
		try {
			if (handle_) {
				handle_->Sync();
			}
			return arrow::Status::OK();
		} catch (const std::exception &ex) {
			return arrow::Status::IOError("flush DuckDB shuffle output: ", ex.what());
		}
	}

private:
	unique_ptr<FileHandle> handle_;
	int64_t position_ = 0;
	bool closed_ = false;
};

class DuckDBFileSystemArrowInputStream final : public arrow::io::InputStream {
public:
	explicit DuckDBFileSystemArrowInputStream(unique_ptr<FileHandle> handle) : handle_(std::move(handle)) {
		set_mode(arrow::io::FileMode::READ);
	}

	arrow::Status Close() override {
		if (closed_) {
			return arrow::Status::OK();
		}
		try {
			if (handle_) {
				handle_->Close();
				handle_.reset();
			}
			closed_ = true;
			return arrow::Status::OK();
		} catch (const std::exception &ex) {
			return arrow::Status::IOError("close DuckDB shuffle input: ", ex.what());
		}
	}

	bool closed() const override {
		return closed_;
	}

	arrow::Result<int64_t> Tell() const override {
		return position_;
	}

	arrow::Result<int64_t> Read(int64_t nbytes, void *out) override {
		if (closed_) {
			return arrow::Status::Invalid("DuckDB shuffle input is closed");
		}
		if (nbytes < 0) {
			return arrow::Status::Invalid("DuckDB shuffle input read size is negative");
		}
		if (nbytes == 0) {
			return int64_t(0);
		}
		try {
			auto read = handle_->Read(out, static_cast<idx_t>(nbytes));
			if (read < 0) {
				return arrow::Status::IOError("failed to read DuckDB shuffle input");
			}
			position_ += read;
			return read;
		} catch (const std::exception &ex) {
			return arrow::Status::IOError("read DuckDB shuffle input: ", ex.what());
		}
	}

	arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
		if (nbytes < 0) {
			return arrow::Status::Invalid("DuckDB shuffle input read size is negative");
		}
		auto buffer_res = arrow::AllocateResizableBuffer(nbytes);
		if (!buffer_res.ok()) {
			return buffer_res.status();
		}
		auto buffer = std::move(buffer_res).ValueOrDie();
		auto read_res = Read(nbytes, buffer->mutable_data());
		if (!read_res.ok()) {
			return read_res.status();
		}
		auto resize_status = buffer->Resize(read_res.ValueOrDie());
		if (!resize_status.ok()) {
			return resize_status;
		}
		return std::shared_ptr<arrow::Buffer>(buffer.release());
	}

private:
	unique_ptr<FileHandle> handle_;
	int64_t position_ = 0;
	bool closed_ = false;
};

class DuckDBFileSystemShuffleStorage final : public ShuffleStorage {
public:
	explicit DuckDBFileSystemShuffleStorage(FileSystem &fs, FileOpener *opener = nullptr) : fs_(fs), opener_(opener) {
	}

	bool SupportsObjectPaths() const override {
		return true;
	}

	DuckDBResult<void> CreateDirectories(const std::string &path) const override {
		if (path.empty() || FileSystem::IsRemoteFile(path)) {
			return DuckDBResult<void>::ok();
		}
		try {
			fs_.CreateDirectoriesRecursive(path, opener_);
			return DuckDBResult<void>::ok();
		} catch (const std::exception &ex) {
			return DuckDBResult<void>::err(
			    DuckDBError::io_error("failed to create shuffle directory via DuckDB FS: " + std::string(ex.what())));
		}
	}

	bool IsRegularFile(const std::string &path) const override {
		try {
			return fs_.FileExists(path, opener_);
		} catch (...) {
			return false;
		}
	}

	DuckDBResult<idx_t> FileSize(const std::string &path) const override {
		try {
			auto handle = fs_.OpenFile(path, FileOpenFlags(FileOpenFlags::FILE_FLAGS_READ), opener_);
			auto size = handle->GetFileSize();
			if (size > std::numeric_limits<idx_t>::max()) {
				return DuckDBResult<idx_t>::err(DuckDBError::io_error("shuffle file size exceeds idx_t max: " + path));
			}
			return DuckDBResult<idx_t>::ok(static_cast<idx_t>(size));
		} catch (const std::exception &ex) {
			return DuckDBResult<idx_t>::err(
			    DuckDBError::io_error("failed to stat shuffle file via DuckDB FS: " + std::string(ex.what())));
		}
	}

	DuckDBResult<void> WriteTextFileAtomically(const std::string &path, const std::string &contents) const override {
		auto parent_res = EnsureParentDirectory(path);
		if (parent_res.is_err()) {
			return parent_res;
		}
		try {
			fs_.TryRemoveFile(path, opener_);
			auto handle = fs_.OpenFile(
			    path, FileOpenFlags(FileOpenFlags::FILE_FLAGS_WRITE | FileOpenFlags::FILE_FLAGS_FILE_CREATE), opener_);
			if (!contents.empty()) {
				auto written = handle->Write(const_cast<char *>(contents.data()), contents.size());
				if (written < 0 || static_cast<idx_t>(written) != contents.size()) {
					return DuckDBResult<void>::err(
					    DuckDBError::io_error("failed to write shuffle text file via DuckDB FS: " + path));
				}
			}
			handle->Sync();
			handle->Close();
			return DuckDBResult<void>::ok();
		} catch (const std::exception &ex) {
			return DuckDBResult<void>::err(
			    DuckDBError::io_error("failed to commit shuffle text file via DuckDB FS: " + std::string(ex.what())));
		}
	}

	DuckDBResult<std::string> ReadTextFile(const std::string &path) const override {
		try {
			auto handle = fs_.OpenFile(path, FileOpenFlags(FileOpenFlags::FILE_FLAGS_READ), opener_);
			auto size = handle->GetFileSize();
			std::string contents;
			contents.resize(size);
			if (size > 0) {
				auto read = handle->Read(&contents[0], static_cast<idx_t>(size));
				if (read < 0 || read != size) {
					return DuckDBResult<std::string>::err(
					    DuckDBError::io_error("failed to read shuffle text file via DuckDB FS: " + path));
				}
			}
			return DuckDBResult<std::string>::ok(std::move(contents));
		} catch (const std::exception &ex) {
			return DuckDBResult<std::string>::err(
			    DuckDBError::io_error("failed to open shuffle text file via DuckDB FS: " + std::string(ex.what())));
		}
	}

	DuckDBResult<idx_t> RemoveAll(const std::string &path) const override {
		return RemoveAllRecursive(path);
	}

	DuckDBResult<std::shared_ptr<arrow::io::OutputStream>> OpenArrowOutput(const std::string &path) const override {
		auto parent_res = EnsureParentDirectory(path);
		if (parent_res.is_err()) {
			return DuckDBResult<std::shared_ptr<arrow::io::OutputStream>>::err(parent_res.error());
		}
		try {
			fs_.TryRemoveFile(path, opener_);
			auto handle = fs_.OpenFile(
			    path, FileOpenFlags(FileOpenFlags::FILE_FLAGS_WRITE | FileOpenFlags::FILE_FLAGS_FILE_CREATE), opener_);
			return DuckDBResult<std::shared_ptr<arrow::io::OutputStream>>::ok(
			    std::make_shared<DuckDBFileSystemArrowOutputStream>(std::move(handle)));
		} catch (const std::exception &ex) {
			return DuckDBResult<std::shared_ptr<arrow::io::OutputStream>>::err(
			    DuckDBError::io_error("failed to open shuffle output via DuckDB FS: " + std::string(ex.what())));
		}
	}

	DuckDBResult<std::shared_ptr<arrow::io::InputStream>> OpenArrowInput(const std::string &path) const override {
		try {
			auto handle = fs_.OpenFile(path, FileOpenFlags(FileOpenFlags::FILE_FLAGS_READ), opener_);
			return DuckDBResult<std::shared_ptr<arrow::io::InputStream>>::ok(
			    std::make_shared<DuckDBFileSystemArrowInputStream>(std::move(handle)));
		} catch (const std::exception &ex) {
			return DuckDBResult<std::shared_ptr<arrow::io::InputStream>>::err(
			    DuckDBError::io_error("failed to open shuffle input via DuckDB FS: " + std::string(ex.what())));
		}
	}

private:
	DuckDBResult<void> EnsureParentDirectory(const std::string &path) const {
		auto parent = StringUtil::GetFilePath(path);
		if (parent.empty() || FileSystem::IsRemoteFile(parent)) {
			return DuckDBResult<void>::ok();
		}
		return CreateDirectories(parent);
	}

	DuckDBResult<idx_t> RemoveAllRecursive(const std::string &path) const {
		if (path.empty()) {
			return DuckDBResult<idx_t>::ok(0);
		}

		idx_t removed = 0;
		try {
			if (fs_.FileExists(path, opener_)) {
				fs_.RemoveFile(path, opener_);
				return DuckDBResult<idx_t>::ok(1);
			}
		} catch (...) {
		}

		try {
			if (!fs_.DirectoryExists(path, opener_)) {
				return DuckDBResult<idx_t>::ok(0);
			}
		} catch (...) {
			return DuckDBResult<idx_t>::ok(0);
		}

		std::vector<std::string> child_dirs;
		try {
			fs_.ListFiles(
			    path,
			    [&](const std::string &child, bool is_dir) {
				    auto full_path = fs_.JoinPath(path, child);
				    if (is_dir) {
					    child_dirs.push_back(full_path);
					    return;
				    }
				    fs_.RemoveFile(full_path, opener_);
				    removed++;
			    },
			    opener_.get_mutable());
			for (const auto &child_dir : child_dirs) {
				auto child_res = RemoveAllRecursive(child_dir);
				if (child_res.is_err()) {
					return child_res;
				}
				removed += child_res.value();
			}
			fs_.RemoveDirectory(path, opener_);
			removed++;
			return DuckDBResult<idx_t>::ok(removed);
		} catch (const std::exception &ex) {
			return DuckDBResult<idx_t>::err(
			    DuckDBError::io_error("failed to remove shuffle storage via DuckDB FS: " + std::string(ex.what())));
		}
	}

	FileSystem &fs_;
	optional_ptr<FileOpener> opener_;
};

std::shared_ptr<ShuffleStorage> MakePosixShuffleStorage() {
	return std::make_shared<PosixShuffleStorage>();
}

} // namespace

std::shared_ptr<ShuffleStorage> MakeDuckDBFileSystemShuffleStorage(FileSystem &fs) {
	return std::make_shared<DuckDBFileSystemShuffleStorage>(fs);
}

std::shared_ptr<ShuffleStorage> MakeDuckDBFileSystemShuffleStorage(FileSystem &fs, FileOpener *opener) {
	return std::make_shared<DuckDBFileSystemShuffleStorage>(fs, opener);
}

static std::atomic<uint64_t> g_shuffle_cache_counter {0};

ShuffleCache::ShuffleCache(ShuffleCacheConfig config) : ShuffleCache(std::move(config), MakePosixShuffleStorage()) {
}

ShuffleCache::ShuffleCache(ShuffleCacheConfig config, std::shared_ptr<ShuffleStorage> storage)
    : config_(std::move(config)), partitions_(config_.num_partitions), next_file_ids_(config_.num_partitions),
      storage_(std::move(storage)), write_buffers_(config_.num_partitions), buffer_bytes_(config_.num_partitions),
      flush_threshold_bytes_(ResolveShuffleCacheFlushThresholdBytes()),
      configured_max_buffered_bytes_(ResolveShuffleCacheMaxBufferedBytes()) {
	if (!storage_) {
		throw InvalidInputException("ShuffleCache storage backend must not be null");
	}
	for (const auto &dir : config_.local_dirs) {
		if (ShuffleCacheLooksLikeUnsupportedObjectPath(dir) && !storage_->SupportsObjectPaths()) {
			throw InvalidInputException("ShuffleCache local_dirs currently require POSIX/shared-filesystem paths. "
			                            "Object storage durable exchange backend is not implemented yet: %s",
			                            dir);
		}
	}
	// Generate a unique instance ID: pid + global counter to prevent
	// batch file overwrites from concurrent sink tasks.
	auto counter = g_shuffle_cache_counter.fetch_add(1);
	std::ostringstream ss;
	ss << getpid() << "_" << counter;
	instance_id_ = ss.str();
	for (auto &entry : next_file_ids_) {
		entry.store(0);
	}
	for (auto &entry : buffer_bytes_) {
		entry.store(0);
	}
	buffer_mutexes_.reserve(config_.num_partitions);
	for (idx_t i = 0; i < config_.num_partitions; i++) {
		buffer_mutexes_.push_back(make_uniq<std::mutex>());
	}
}

ShuffleCache::~ShuffleCache() {
	// FlushAll is called explicitly via SinkFinalize; never write from a destructor.
	DiscardBufferedData();
}

const ShuffleCacheConfig &ShuffleCache::config() const {
	return config_;
}

vector<string> ShuffleCache::BufferedNames() const {
	std::lock_guard<std::mutex> lock(buffered_names_mutex_);
	return buffered_names_;
}

DuckDBResult<void> ShuffleCache::EnsurePartitionDirectory(idx_t partition_idx) const {
	if (partition_idx >= partitions_.size()) {
		return DuckDBResult<void>::err(DuckDBError::value_error("shuffle partition index out of range"));
	}
	if (config_.local_dirs.empty()) {
		return DuckDBResult<void>::err(DuckDBError::value_error("shuffle cache local_dirs is empty"));
	}
	return storage_->CreateDirectories(PartitionDirectory(partition_idx));
}

std::string ShuffleCache::NodeDirectory() const {
	auto base_dir = config_.local_dirs.empty() ? std::string() : config_.local_dirs[0];
	auto stage = ShuffleCacheSanitizePathComponent(config_.shuffle_stage_id);
	auto node = ShuffleCacheSanitizePathComponent(config_.node_id);
	std::ostringstream ss;
	ss << base_dir << "/shuffle_" << stage << "/node_" << node;
	return ss.str();
}

std::string ShuffleCache::PartitionDirectory(idx_t partition_idx) const {
	auto base_dir = config_.local_dirs[partition_idx % config_.local_dirs.size()];
	auto stage = ShuffleCacheSanitizePathComponent(config_.shuffle_stage_id);
	auto node = ShuffleCacheSanitizePathComponent(config_.node_id);
	std::ostringstream ss;
	ss << base_dir << "/shuffle_" << stage << "/node_" << node << "/partition_" << partition_idx;
	return ss.str();
}

std::string ShuffleCache::SchemaFilePath() const {
	return NodeDirectory() + "/schema.arrow";
}

std::string ShuffleCache::ManifestFilePath() const {
	return NodeDirectory() + "/manifest.txt";
}

std::string ShuffleCache::CommittedMarkerPath() const {
	return NodeDirectory() + "/committed";
}

std::string ShuffleCache::MakePartitionFilePath(idx_t partition_idx) {
	auto file_id = next_file_ids_[partition_idx].fetch_add(1);
	std::ostringstream ss;
	ss << PartitionDirectory(partition_idx) << "/batch_" << instance_id_ << "_" << file_id << ".arrow";
	return ss.str();
}

DuckDBResult<void> ShuffleCache::WriteAttemptManifest(idx_t sink_partition_id, idx_t attempt_id) {
	if (config_.local_dirs.empty()) {
		return DuckDBResult<void>::err(DuckDBError::value_error("shuffle cache local_dirs is empty"));
	}

	std::vector<ShufflePartitionFiles> partition_snapshot;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		partition_snapshot = partitions_;
	}

	auto dir_res = storage_->CreateDirectories(NodeDirectory());
	if (dir_res.is_err()) {
		return DuckDBResult<void>::err(dir_res.error());
	}

	auto manifest_path = ManifestFilePath();
	std::ostringstream manifest;
	manifest << "version=1\n";
	manifest << "shuffle_stage_id=" << config_.shuffle_stage_id << "\n";
	manifest << "node_id=" << config_.node_id << "\n";
	manifest << "sink_partition_id=" << sink_partition_id << "\n";
	manifest << "attempt_id=" << attempt_id << "\n";
	manifest << "output_partition_count=" << config_.num_partitions << "\n";
	for (idx_t partition_id = 0; partition_id < partition_snapshot.size(); partition_id++) {
		for (const auto &file : partition_snapshot[partition_id].files) {
			manifest << "file=" << partition_id << "\t" << file.bytes << "\t" << file.rows << "\t" << file.path << "\n";
		}
	}
	auto write_manifest_res = storage_->WriteTextFileAtomically(manifest_path, manifest.str());
	if (write_manifest_res.is_err()) {
		return DuckDBResult<void>::err(write_manifest_res.error());
	}

	auto marker_path = CommittedMarkerPath();
	auto write_marker_res = storage_->WriteTextFileAtomically(marker_path, "committed\n");
	if (write_marker_res.is_err()) {
		return DuckDBResult<void>::err(write_marker_res.error());
	}

	return DuckDBResult<void>::ok();
}

bool ShuffleCache::HasCommittedManifest() const {
	if (config_.local_dirs.empty()) {
		return false;
	}
	return storage_->IsRegularFile(CommittedMarkerPath()) && storage_->IsRegularFile(ManifestFilePath());
}

DuckDBResult<ShuffleAttemptManifest> ShuffleCache::ReadAttemptManifest(const std::string &manifest_path,
                                                                       const std::string &committed_marker_path) {
	auto storage = MakePosixShuffleStorage();
	return ReadAttemptManifest(*storage, manifest_path, committed_marker_path);
}

DuckDBResult<ShuffleAttemptManifest> ShuffleCache::ReadAttemptManifest(const ShuffleStorage &storage,
                                                                       const std::string &manifest_path,
                                                                       const std::string &committed_marker_path) {
	if (!storage.IsRegularFile(committed_marker_path)) {
		return DuckDBResult<ShuffleAttemptManifest>::err(
		    DuckDBError::invalid_state_error("shuffle attempt manifest is not committed: " + manifest_path));
	}
	if (!storage.IsRegularFile(manifest_path)) {
		return DuckDBResult<ShuffleAttemptManifest>::err(
		    DuckDBError::io_error("shuffle attempt manifest missing: " + manifest_path));
	}

	auto text_res = storage.ReadTextFile(manifest_path);
	if (text_res.is_err()) {
		return DuckDBResult<ShuffleAttemptManifest>::err(text_res.error());
	}
	std::istringstream input(text_res.value());

	ShuffleAttemptManifest manifest;
	bool seen_version = false;
	bool seen_shuffle_stage_id = false;
	bool seen_node_id = false;
	bool seen_sink_partition_id = false;
	bool seen_attempt_id = false;
	bool seen_output_partition_count = false;

	std::string line;
	idx_t line_no = 0;
	while (std::getline(input, line)) {
		line_no++;
		if (line.empty()) {
			continue;
		}
		if (line.rfind("file=", 0) == 0) {
			auto fields = SplitManifestFileFields(line.substr(5));
			if (fields.size() != 4) {
				return DuckDBResult<ShuffleAttemptManifest>::err(DuckDBError::value_error(
				    "shuffle attempt manifest invalid file line at " + std::to_string(line_no)));
			}
			auto partition_res = ParseManifestIdx(fields[0], "file.partition_id");
			if (partition_res.is_err()) {
				return DuckDBResult<ShuffleAttemptManifest>::err(partition_res.error());
			}
			auto bytes_res = ParseManifestIdx(fields[1], "file.bytes");
			if (bytes_res.is_err()) {
				return DuckDBResult<ShuffleAttemptManifest>::err(bytes_res.error());
			}
			auto rows_res = ParseManifestIdx(fields[2], "file.rows");
			if (rows_res.is_err()) {
				return DuckDBResult<ShuffleAttemptManifest>::err(rows_res.error());
			}
			if (fields[3].empty()) {
				return DuckDBResult<ShuffleAttemptManifest>::err(
				    DuckDBError::value_error("shuffle attempt manifest empty file path"));
			}

			if (!storage.IsRegularFile(fields[3])) {
				return DuckDBResult<ShuffleAttemptManifest>::err(
				    DuckDBError::io_error("shuffle attempt manifest file missing: " + fields[3]));
			}
			auto size_res = storage.FileSize(fields[3]);
			if (size_res.is_err()) {
				return DuckDBResult<ShuffleAttemptManifest>::err(size_res.error());
			}
			if (bytes_res.value() != size_res.value()) {
				return DuckDBResult<ShuffleAttemptManifest>::err(
				    DuckDBError::io_error("shuffle attempt manifest file size mismatch: " + fields[3]));
			}

			ShuffleManifestPartitionFile entry;
			entry.partition_id = partition_res.value();
			entry.file.bytes = bytes_res.value();
			entry.file.rows = rows_res.value();
			entry.file.path = std::move(fields[3]);
			manifest.files.push_back(std::move(entry));
			continue;
		}

		auto sep = line.find('=');
		if (sep == std::string::npos) {
			return DuckDBResult<ShuffleAttemptManifest>::err(
			    DuckDBError::value_error("shuffle attempt manifest invalid line at " + std::to_string(line_no)));
		}
		auto key = line.substr(0, sep);
		auto value = line.substr(sep + 1);
		if (key == "version") {
			auto dup_res = CheckDuplicateManifestField(seen_version, key);
			if (dup_res.is_err()) {
				return DuckDBResult<ShuffleAttemptManifest>::err(dup_res.error());
			}
			if (value != "1") {
				return DuckDBResult<ShuffleAttemptManifest>::err(
				    DuckDBError::value_error("unsupported shuffle attempt manifest version: " + value));
			}
		} else if (key == "shuffle_stage_id") {
			auto dup_res = CheckDuplicateManifestField(seen_shuffle_stage_id, key);
			if (dup_res.is_err()) {
				return DuckDBResult<ShuffleAttemptManifest>::err(dup_res.error());
			}
			if (value.empty()) {
				return DuckDBResult<ShuffleAttemptManifest>::err(
				    DuckDBError::value_error("shuffle attempt manifest empty shuffle_stage_id"));
			}
			manifest.shuffle_stage_id = std::move(value);
		} else if (key == "node_id") {
			auto dup_res = CheckDuplicateManifestField(seen_node_id, key);
			if (dup_res.is_err()) {
				return DuckDBResult<ShuffleAttemptManifest>::err(dup_res.error());
			}
			manifest.node_id = std::move(value);
		} else if (key == "sink_partition_id") {
			auto dup_res = CheckDuplicateManifestField(seen_sink_partition_id, key);
			if (dup_res.is_err()) {
				return DuckDBResult<ShuffleAttemptManifest>::err(dup_res.error());
			}
			auto parsed = ParseManifestIdx(value, key);
			if (parsed.is_err()) {
				return DuckDBResult<ShuffleAttemptManifest>::err(parsed.error());
			}
			manifest.sink_partition_id = parsed.value();
		} else if (key == "attempt_id") {
			auto dup_res = CheckDuplicateManifestField(seen_attempt_id, key);
			if (dup_res.is_err()) {
				return DuckDBResult<ShuffleAttemptManifest>::err(dup_res.error());
			}
			auto parsed = ParseManifestIdx(value, key);
			if (parsed.is_err()) {
				return DuckDBResult<ShuffleAttemptManifest>::err(parsed.error());
			}
			manifest.attempt_id = parsed.value();
		} else if (key == "output_partition_count") {
			auto dup_res = CheckDuplicateManifestField(seen_output_partition_count, key);
			if (dup_res.is_err()) {
				return DuckDBResult<ShuffleAttemptManifest>::err(dup_res.error());
			}
			auto parsed = ParseManifestIdx(value, key);
			if (parsed.is_err()) {
				return DuckDBResult<ShuffleAttemptManifest>::err(parsed.error());
			}
			manifest.output_partition_count = parsed.value();
		} else {
			return DuckDBResult<ShuffleAttemptManifest>::err(
			    DuckDBError::value_error("shuffle attempt manifest unknown field: " + key));
		}
	}
	if (input.bad()) {
		return DuckDBResult<ShuffleAttemptManifest>::err(
		    DuckDBError::io_error("failed to read shuffle attempt manifest: " + manifest_path));
	}
	if (!seen_version || !seen_shuffle_stage_id || !seen_node_id || !seen_sink_partition_id || !seen_attempt_id ||
	    !seen_output_partition_count) {
		return DuckDBResult<ShuffleAttemptManifest>::err(
		    DuckDBError::value_error("shuffle attempt manifest missing required field: " + manifest_path));
	}
	if (manifest.output_partition_count == 0) {
		return DuckDBResult<ShuffleAttemptManifest>::err(
		    DuckDBError::value_error("shuffle attempt manifest output_partition_count is zero"));
	}
	for (const auto &entry : manifest.files) {
		if (entry.partition_id >= manifest.output_partition_count) {
			return DuckDBResult<ShuffleAttemptManifest>::err(
			    DuckDBError::value_error("shuffle attempt manifest partition id out of range"));
		}
	}
	std::sort(manifest.files.begin(), manifest.files.end(),
	          [](const ShuffleManifestPartitionFile &lhs, const ShuffleManifestPartitionFile &rhs) {
		          if (lhs.partition_id != rhs.partition_id) {
			          return lhs.partition_id < rhs.partition_id;
		          }
		          return lhs.file.path < rhs.file.path;
	          });
	return DuckDBResult<ShuffleAttemptManifest>::ok(std::move(manifest));
}

DuckDBResult<ShufflePartitionFiles> ShuffleCache::GetPartitionFilesFromManifest(const ShuffleAttemptManifest &manifest,
                                                                                idx_t partition_idx) {
	if (manifest.output_partition_count == 0 || partition_idx >= manifest.output_partition_count) {
		return DuckDBResult<ShufflePartitionFiles>::err(
		    DuckDBError::value_error("shuffle manifest partition index out of range"));
	}
	ShufflePartitionFiles files;
	for (const auto &entry : manifest.files) {
		if (entry.partition_id != partition_idx) {
			continue;
		}
		files.total_rows += entry.file.rows;
		files.total_bytes += entry.file.bytes;
		files.files.push_back(entry.file);
	}
	std::sort(files.files.begin(), files.files.end(),
	          [](const ShufflePartitionFile &lhs, const ShufflePartitionFile &rhs) { return lhs.path < rhs.path; });
	return DuckDBResult<ShufflePartitionFiles>::ok(std::move(files));
}

DuckDBResult<ShufflePartitionFiles> ShuffleCache::GetPartitionFilesFromManifest(idx_t partition_idx) const {
	if (partition_idx >= partitions_.size()) {
		return DuckDBResult<ShufflePartitionFiles>::err(
		    DuckDBError::value_error("shuffle partition index out of range"));
	}
	if (!HasCommittedManifest()) {
		ShufflePartitionFiles empty;
		return DuckDBResult<ShufflePartitionFiles>::ok(std::move(empty));
	}
	auto manifest_res = ReadAttemptManifest(*storage_, ManifestFilePath(), CommittedMarkerPath());
	if (manifest_res.is_err()) {
		return DuckDBResult<ShufflePartitionFiles>::err(manifest_res.error());
	}
	auto manifest = std::move(manifest_res.value());
	if (manifest.shuffle_stage_id != config_.shuffle_stage_id) {
		return DuckDBResult<ShufflePartitionFiles>::err(
		    DuckDBError::value_error("shuffle attempt manifest stage id mismatch"));
	}
	if (!config_.node_id.empty() && manifest.node_id != config_.node_id) {
		return DuckDBResult<ShufflePartitionFiles>::err(
		    DuckDBError::value_error("shuffle attempt manifest node id mismatch"));
	}
	return GetPartitionFilesFromManifest(manifest, partition_idx);
}

DuckDBResult<idx_t> ShuffleCache::RemoveAttemptStorage() const {
	return RemoveShuffleAttemptStorage(config_, *storage_);
}

DuckDBResult<idx_t> RemoveShuffleAttemptStorage(const ShuffleCacheConfig &config, const ShuffleStorage &storage) {
	if (config.shuffle_stage_id.empty()) {
		return DuckDBResult<idx_t>::err(DuckDBError::value_error("shuffle cache stage id is empty"));
	}
	if (config.node_id.empty()) {
		return DuckDBResult<idx_t>::err(DuckDBError::value_error("shuffle cache node id is empty"));
	}
	if (config.local_dirs.empty()) {
		return DuckDBResult<idx_t>::ok(0);
	}

	auto stage = ShuffleCacheSanitizePathComponent(config.shuffle_stage_id);
	auto node = ShuffleCacheSanitizePathComponent(config.node_id);
	std::unordered_set<std::string> seen_attempt_dirs;
	idx_t removed_total = 0;
	for (const auto &base_dir : config.local_dirs) {
		if (base_dir.empty()) {
			continue;
		}
		std::ostringstream ss;
		ss << base_dir << "/shuffle_" << stage << "/node_" << node;
		auto attempt_dir = ss.str();
		if (!seen_attempt_dirs.insert(attempt_dir).second) {
			continue;
		}
		auto removed_res = storage.RemoveAll(attempt_dir);
		if (removed_res.is_err()) {
			return DuckDBResult<idx_t>::err(removed_res.error());
		}
		if (removed_total > std::numeric_limits<idx_t>::max() - removed_res.value()) {
			removed_total = std::numeric_limits<idx_t>::max();
		} else {
			removed_total += removed_res.value();
		}
	}
	return DuckDBResult<idx_t>::ok(static_cast<idx_t>(removed_total));
}

DuckDBResult<void> ShuffleCache::EnsureSchemaFile(ClientContext &context, const vector<LogicalType> &types,
                                                  const vector<string> &names) {
	if (schema_written_.load()) {
		return DuckDBResult<void>::ok();
	}
	std::lock_guard<std::mutex> lock(mutex_);
	if (schema_written_.load()) {
		return DuckDBResult<void>::ok();
	}
	if (config_.local_dirs.empty()) {
		return DuckDBResult<void>::err(DuckDBError::value_error("shuffle cache local_dirs is empty"));
	}

	auto schema_path = SchemaFilePath();
	if (storage_->IsRegularFile(schema_path)) {
		schema_written_.store(true);
		return DuckDBResult<void>::ok();
	}

	auto dir_res = storage_->CreateDirectories(NodeDirectory());
	if (dir_res.is_err()) {
		return DuckDBResult<void>::err(dir_res.error());
	}

	ArrowSchema schema;
	schema.Init();
	auto client_properties = context.GetClientProperties();
	ArrowConverter::ToArrowSchema(&schema, types, names, client_properties);

	auto schema_res = arrow::ImportSchema(&schema);
	if (!schema_res.ok()) {
		if (schema.release) {
			schema.release(&schema);
		}
		return DuckDBResult<void>::err(ShuffleCacheArrowToError(schema_res.status(), "import schema"));
	}
	auto record_schema = std::move(schema_res).ValueOrDie();
	if (schema.release) {
		schema.release(&schema);
	}

	auto output_res = storage_->OpenArrowOutput(schema_path);
	if (output_res.is_err()) {
		return DuckDBResult<void>::err(output_res.error());
	}
	auto output = std::move(output_res.value());
	auto writer_res = arrow::ipc::MakeStreamWriter(output.get(), record_schema);
	if (!writer_res.ok()) {
		return DuckDBResult<void>::err(ShuffleCacheArrowToError(writer_res.status(), "create schema writer"));
	}
	auto writer = std::move(writer_res).ValueOrDie();
	auto status = writer->Close();
	if (!status.ok()) {
		return DuckDBResult<void>::err(ShuffleCacheArrowToError(status, "close schema writer"));
	}
	status = output->Close();
	if (!status.ok()) {
		return DuckDBResult<void>::err(ShuffleCacheArrowToError(status, "close schema output"));
	}

	schema_written_.store(true);
	return DuckDBResult<void>::ok();
}

void ShuffleCache::EnsureMemoryBudget(ClientContext &context) {
	if (memory_budget_initialized_.load()) {
		return;
	}

	std::lock_guard<std::mutex> guard(memory_state_mutex_);
	if (memory_budget_initialized_.load()) {
		return;
	}
	temporary_memory_state_ = TemporaryMemoryManager::Get(context).Register(context);
	auto max_bytes = std::numeric_limits<idx_t>::max();
	auto aggregate_target = config_.num_partitions > max_bytes / flush_threshold_bytes_
	                            ? max_bytes
	                            : config_.num_partitions * flush_threshold_bytes_;
	auto desired_bytes = MinValue(temporary_memory_state_->GetRemainingSize(), aggregate_target);
	if (configured_max_buffered_bytes_.IsValid()) {
		desired_bytes = MinValue(desired_bytes, configured_max_buffered_bytes_.GetIndex());
	}
	if (desired_bytes > 0) {
		// One target-size partition is the guaranteed floor. The manager may grant
		// more of the default share while memory is available; that grant remains
		// accounted for until this state refreshes or releases it.
		auto minimum_reservation = MinValue(temporary_memory_state_->GetMinimumReservation(), flush_threshold_bytes_);
		minimum_reservation = MinValue(minimum_reservation, desired_bytes);
		temporary_memory_state_->SetMinimumReservation(minimum_reservation);
		temporary_memory_state_->SetRemainingSizeAndUpdateReservation(context, desired_bytes);
	}
	auto budget = temporary_memory_state_->GetReservation();
	if (configured_max_buffered_bytes_.IsValid()) {
		budget = MinValue(budget, configured_max_buffered_bytes_.GetIndex());
	}
	buffer_budget_bytes_.store(budget);
	memory_budget_initialized_.store(true);
}

void ShuffleCache::RefreshMemoryBudget(ClientContext &context) {
	std::lock_guard<std::mutex> guard(memory_state_mutex_);
	if (!temporary_memory_state_) {
		return;
	}
	temporary_memory_state_->UpdateReservation(context);
	auto budget = temporary_memory_state_->GetReservation();
	if (configured_max_buffered_bytes_.IsValid()) {
		budget = MinValue(budget, configured_max_buffered_bytes_.GetIndex());
	}
	buffer_budget_bytes_.store(budget);
}

void ShuffleCache::ReleaseMemoryReservation() {
	std::lock_guard<std::mutex> guard(memory_state_mutex_);
	if (temporary_memory_state_) {
		temporary_memory_state_->SetZero();
		temporary_memory_state_.reset();
	}
	buffer_budget_bytes_.store(0);
	memory_budget_initialized_.store(false);
}

void ShuffleCache::UpdatePeakBufferedBytes(idx_t buffered_bytes) {
	auto peak = peak_buffered_bytes_.load();
	while (peak < buffered_bytes && !peak_buffered_bytes_.compare_exchange_weak(peak, buffered_bytes)) {
	}
}

DuckDBResult<void> ShuffleCache::WriteChunk(ClientContext &context, DataChunk &chunk, idx_t partition_idx,
                                            const vector<string> &names) {
	if (partition_idx >= partitions_.size()) {
		return DuckDBResult<void>::err(DuckDBError::value_error("shuffle partition index out of range"));
	}
	if (chunk.ColumnCount() == 0) {
		return DuckDBResult<void>::err(DuckDBError::value_error("shuffle cache write requires at least one column"));
	}
	if (chunk.size() == 0) {
		return DuckDBResult<void>::ok();
	}
	EnsureMemoryBudget(context);

	// Capture column names for later FlushAll calls.
	if (!names.empty()) {
		std::lock_guard<std::mutex> names_guard(buffered_names_mutex_);
		if (buffered_names_.empty()) {
			buffered_names_ = names;
		}
	}

	{
		// Lock this partition's buffer for thread-safe access.
		std::lock_guard<std::mutex> lock(*buffer_mutexes_[partition_idx]);

		// Keep disk-first buffers in memory, but charge them to DuckDB's memory limit.
		if (!write_buffers_[partition_idx]) {
			write_buffers_[partition_idx] =
			    make_uniq<ColumnDataCollection>(BufferAllocator::Get(context), chunk.GetTypes());
		}

		// Account allocated capacity rather than row payload size, including varlen blocks.
		auto previous_bytes = buffer_bytes_[partition_idx].load();
		write_buffers_[partition_idx]->Append(chunk);
		auto current_bytes = write_buffers_[partition_idx]->AllocationSize();
		D_ASSERT(current_bytes >= previous_bytes);
		buffer_bytes_[partition_idx].store(current_bytes);
		auto allocated_bytes = current_bytes - previous_bytes;
		auto total_bytes = buffered_bytes_.fetch_add(allocated_bytes) + allocated_bytes;
		UpdatePeakBufferedBytes(total_bytes);

		// The per-partition threshold remains a target output segment size.
		if (current_bytes >= flush_threshold_bytes_) {
			auto flush_res = FlushBufferLocked(context, partition_idx, names);
			if (flush_res.is_err()) {
				return DuckDBResult<void>::err(flush_res.error());
			}
		}
	}

	if (buffered_bytes_.load() > buffer_budget_bytes_.load()) {
		return FlushUnderMemoryPressure(context, names);
	}
	return DuckDBResult<void>::ok();
}

DuckDBResult<ShufflePartitionFile> ShuffleCache::FlushBuffer(ClientContext &context, idx_t partition_idx,
                                                             const vector<string> &names) {
	std::lock_guard<std::mutex> lock(*buffer_mutexes_[partition_idx]);
	return FlushBufferLocked(context, partition_idx, names);
}

DuckDBResult<ShufflePartitionFile> ShuffleCache::FlushBufferLocked(ClientContext &context, idx_t partition_idx,
                                                                   const vector<string> &names) {
	auto &buffer = write_buffers_[partition_idx];
	if (!buffer) {
		ShufflePartitionFile empty;
		return DuckDBResult<ShufflePartitionFile>::ok(std::move(empty));
	}
	if (buffer->Count() == 0) {
		// InitializeAppend can allocate storage before seeing an empty chunk. Do not
		// leave an unflushable allocation in the pressure-victim set.
		auto released_bytes = buffer_bytes_[partition_idx].load();
		buffer.reset();
		if (released_bytes > 0) {
			auto previous_total = buffered_bytes_.fetch_sub(released_bytes);
			D_ASSERT(previous_total >= released_bytes);
		}
		buffer_bytes_[partition_idx].store(0);
		ShufflePartitionFile empty;
		return DuckDBResult<ShufflePartitionFile>::ok(std::move(empty));
	}

	auto result = WriteCollectionToFile(context, *buffer, partition_idx, names);

	// Release accounting only after the synchronous write no longer needs the buffer.
	auto released_bytes = buffer_bytes_[partition_idx].load();
	buffer.reset();
	if (released_bytes > 0) {
		auto previous_total = buffered_bytes_.fetch_sub(released_bytes);
		D_ASSERT(previous_total >= released_bytes);
	}
	buffer_bytes_[partition_idx].store(0);

	return result;
}

DuckDBResult<void> ShuffleCache::FlushUnderMemoryPressure(ClientContext &context, const vector<string> &names) {
	std::lock_guard<std::mutex> pressure_guard(pressure_mutex_);
	RefreshMemoryBudget(context);

	while (buffered_bytes_.load() > buffer_budget_bytes_.load()) {
		optional_idx victim_partition;
		idx_t victim_bytes = 0;
		for (idx_t partition_idx = 0; partition_idx < buffer_bytes_.size(); partition_idx++) {
			auto partition_bytes = buffer_bytes_[partition_idx].load();
			if (partition_bytes > victim_bytes) {
				victim_partition = partition_idx;
				victim_bytes = partition_bytes;
			}
		}
		if (!victim_partition.IsValid()) {
			// A concurrent threshold flush can release the last candidate between the
			// aggregate check and this unlocked scan.
			if (buffered_bytes_.load() <= buffer_budget_bytes_.load()) {
				continue;
			}
			return DuckDBResult<void>::err(
			    DuckDBError::internal_error("shuffle cache memory accounting exceeded its budget without a buffer"));
		}

		auto partition_idx = victim_partition.GetIndex();
		std::lock_guard<std::mutex> partition_guard(*buffer_mutexes_[partition_idx]);
		if (buffer_bytes_[partition_idx].load() == 0) {
			continue;
		}
		auto flush_res = FlushBufferLocked(context, partition_idx, names);
		if (flush_res.is_err()) {
			return DuckDBResult<void>::err(flush_res.error());
		}
	}
	return DuckDBResult<void>::ok();
}

DuckDBResult<void> ShuffleCache::FlushAll(ClientContext &context, const vector<string> &names) {
	std::lock_guard<std::mutex> pressure_guard(pressure_mutex_);
	if (flushed_) {
		return DuckDBResult<void>::ok();
	}
	flushed_ = true;

	for (idx_t i = 0; i < write_buffers_.size(); i++) {
		auto res = FlushBuffer(context, i, names);
		if (res.is_err()) {
			return DuckDBResult<void>::err(res.error());
		}
	}
	ReleaseMemoryReservation();
	return DuckDBResult<void>::ok();
}

void ShuffleCache::DiscardBufferedData() {
	std::lock_guard<std::mutex> pressure_guard(pressure_mutex_);
	for (idx_t partition_idx = 0; partition_idx < write_buffers_.size(); partition_idx++) {
		std::lock_guard<std::mutex> partition_guard(*buffer_mutexes_[partition_idx]);
		auto released_bytes = buffer_bytes_[partition_idx].load();
		write_buffers_[partition_idx].reset();
		if (released_bytes > 0) {
			auto previous_total = buffered_bytes_.fetch_sub(released_bytes);
			D_ASSERT(previous_total >= released_bytes);
		}
		buffer_bytes_[partition_idx].store(0);
	}
	ReleaseMemoryReservation();
}

DuckDBResult<ShufflePartitionFile> ShuffleCache::WriteCollectionToFile(ClientContext &context,
                                                                       ColumnDataCollection &collection,
                                                                       idx_t partition_idx,
                                                                       const vector<string> &names) {
	if (collection.Count() == 0) {
		ShufflePartitionFile empty;
		return DuckDBResult<ShufflePartitionFile>::ok(std::move(empty));
	}

	auto dir_res = EnsurePartitionDirectory(partition_idx);
	if (dir_res.is_err()) {
		return DuckDBResult<ShufflePartitionFile>::err(dir_res.error());
	}

	auto name_res = ResolveColumnNames(collection.Types().size(), names);
	if (name_res.is_err()) {
		return DuckDBResult<ShufflePartitionFile>::err(name_res.error());
	}
	auto column_names = std::move(name_res.value());

	auto arrow_types = ToArrowTypes(collection.Types());

	auto schema_result = EnsureSchemaFile(context, arrow_types, column_names);
	if (schema_result.is_err()) {
		return DuckDBResult<ShufflePartitionFile>::err(schema_result.error());
	}

	auto file_path = MakePartitionFilePath(partition_idx);
	auto output_res = storage_->OpenArrowOutput(file_path);
	if (output_res.is_err()) {
		return DuckDBResult<ShufflePartitionFile>::err(output_res.error());
	}
	auto output = std::move(output_res.value());

	// Create writer — we'll write multiple record batches from the collection's chunks.
	auto client_properties = context.GetClientProperties();
	std::shared_ptr<arrow::Schema> record_schema;
	std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
	bool writer_initialized = false;

	idx_t total_rows = 0;
	for (auto &chunk : collection.Chunks()) {
		auto arrow_typed = ToArrowTypes(chunk.GetTypes());
		DataChunk *write_chunk = &chunk;
		unique_ptr<DataChunk> converted_chunk;
		if (arrow_typed != chunk.GetTypes()) {
			converted_chunk = make_uniq<DataChunk>();
			converted_chunk->Initialize(Allocator::DefaultAllocator(), arrow_typed);
			CastChunk(context, chunk, *converted_chunk, arrow_typed);
			write_chunk = converted_chunk.get();
		}

		ArrowSchema arrow_schema;
		arrow_schema.Init();
		ArrowArray arrow_array;
		arrow_array.Init();

		ArrowConverter::ToArrowSchema(&arrow_schema, write_chunk->GetTypes(), column_names, client_properties);
		unordered_map<idx_t, const shared_ptr<ArrowTypeExtensionData>> extension_type_cast;
		ArrowConverter::ToArrowArray(*write_chunk, &arrow_array, client_properties, extension_type_cast);

		if (!writer_initialized) {
			auto import_schema_res = arrow::ImportSchema(&arrow_schema);
			if (!import_schema_res.ok()) {
				return DuckDBResult<ShufflePartitionFile>::err(
				    ShuffleCacheArrowToError(import_schema_res.status(), "import schema"));
			}
			record_schema = std::move(import_schema_res).ValueOrDie();
			if (arrow_schema.release) {
				arrow_schema.release(&arrow_schema);
			}

			auto writer_res = arrow::ipc::MakeStreamWriter(output.get(), record_schema);
			if (!writer_res.ok()) {
				return DuckDBResult<ShufflePartitionFile>::err(
				    ShuffleCacheArrowToError(writer_res.status(), "create ipc writer"));
			}
			writer = std::move(writer_res).ValueOrDie();
			writer_initialized = true;
		} else {
			if (arrow_schema.release) {
				arrow_schema.release(&arrow_schema);
			}
		}

		auto batch_res = arrow::ImportRecordBatch(&arrow_array, record_schema);
		if (!batch_res.ok()) {
			return DuckDBResult<ShufflePartitionFile>::err(
			    ShuffleCacheArrowToError(batch_res.status(), "import record batch"));
		}
		auto record_batch = std::move(batch_res).ValueOrDie();
		auto status = writer->WriteRecordBatch(*record_batch);
		if (!status.ok()) {
			return DuckDBResult<ShufflePartitionFile>::err(ShuffleCacheArrowToError(status, "write record batch"));
		}
		total_rows += write_chunk->size();
	}

	if (writer) {
		auto status = writer->Close();
		if (!status.ok()) {
			return DuckDBResult<ShufflePartitionFile>::err(ShuffleCacheArrowToError(status, "close ipc writer"));
		}
	}
	auto close_output_status = output->Close();
	if (!close_output_status.ok()) {
		return DuckDBResult<ShufflePartitionFile>::err(
		    ShuffleCacheArrowToError(close_output_status, "close shuffle output"));
	}

	auto file_size_res = storage_->FileSize(file_path);
	if (file_size_res.is_err()) {
		return DuckDBResult<ShufflePartitionFile>::err(file_size_res.error());
	}

	ShufflePartitionFile file;
	file.path = std::move(file_path);
	file.rows = total_rows;
	file.bytes = file_size_res.value();

	auto reg_res = RegisterPartitionFile(partition_idx, file);
	if (reg_res.is_err()) {
		return DuckDBResult<ShufflePartitionFile>::err(reg_res.error());
	}
	return DuckDBResult<ShufflePartitionFile>::ok(std::move(file));
}

DuckDBResult<std::vector<ShufflePartitionFile>> ShuffleCache::WriteCollection(ClientContext &context,
                                                                              ColumnDataCollection &collection,
                                                                              idx_t partition_idx,
                                                                              const vector<string> &names) {
	// Buffer all chunks from the collection.
	for (auto &chunk : collection.Chunks()) {
		auto res = WriteChunk(context, chunk, partition_idx, names);
		if (res.is_err()) {
			return DuckDBResult<std::vector<ShufflePartitionFile>>::err(res.error());
		}
	}
	// Flush remaining buffered data.
	auto flush_res = FlushBuffer(context, partition_idx, names);
	if (flush_res.is_err()) {
		return DuckDBResult<std::vector<ShufflePartitionFile>>::err(flush_res.error());
	}
	// Return the files written for this partition.
	auto files_res = GetPartitionFiles(partition_idx);
	if (files_res.is_err()) {
		return DuckDBResult<std::vector<ShufflePartitionFile>>::err(files_res.error());
	}
	return DuckDBResult<std::vector<ShufflePartitionFile>>::ok(std::move(files_res.value().files));
}

DuckDBResult<std::shared_ptr<arrow::io::InputStream>> ShuffleCache::OpenPartitionFile(const std::string &path) const {
	return storage_->OpenArrowInput(path);
}

DuckDBResult<void> ShuffleCache::RegisterPartitionFile(idx_t partition_idx, ShufflePartitionFile file) {
	if (partition_idx >= partitions_.size()) {
		return DuckDBResult<void>::err(DuckDBError::value_error("shuffle partition index out of range"));
	}
	std::lock_guard<std::mutex> lock(mutex_);
	auto &partition = partitions_[partition_idx];
	partition.total_rows += file.rows;
	partition.total_bytes += file.bytes;
	partition.files.push_back(std::move(file));
	return DuckDBResult<void>::ok();
}

DuckDBResult<ShufflePartitionFiles> ShuffleCache::GetPartitionFiles(idx_t partition_idx) const {
	if (partition_idx >= partitions_.size()) {
		return DuckDBResult<ShufflePartitionFiles>::err(
		    DuckDBError::value_error("shuffle partition index out of range"));
	}
	std::lock_guard<std::mutex> lock(mutex_);
	return DuckDBResult<ShufflePartitionFiles>::ok(partitions_[partition_idx]);
}

} // namespace distributed
} // namespace duckdb
