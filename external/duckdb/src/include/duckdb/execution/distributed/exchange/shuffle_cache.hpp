// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/common_types.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace arrow {
namespace io {
class InputStream;
class OutputStream;
} // namespace io
} // namespace arrow

namespace duckdb {
class Allocator;
class ClientContext;
class DataChunk;
class ColumnDataCollection;
class FileOpener;
class FileSystem;
class LogicalType;

namespace distributed {

struct ShufflePartitionFile {
	std::string path;
	idx_t rows = 0;
	idx_t bytes = 0;
};

struct ShufflePartitionFiles {
	std::vector<ShufflePartitionFile> files;
	idx_t total_rows = 0;
	idx_t total_bytes = 0;
};

struct ShuffleManifestPartitionFile {
	idx_t partition_id = 0;
	ShufflePartitionFile file;
};

struct ShuffleAttemptManifest {
	std::string shuffle_stage_id;
	std::string node_id;
	idx_t sink_partition_id = 0;
	idx_t attempt_id = 0;
	idx_t output_partition_count = 0;
	std::vector<ShuffleManifestPartitionFile> files;
};

struct ShuffleCacheConfig {
	std::string shuffle_stage_id;
	std::string node_id;
	idx_t num_partitions = 0;
	std::vector<std::string> local_dirs;
};

class ShuffleStorage {
public:
	virtual ~ShuffleStorage() = default;

	virtual bool SupportsObjectPaths() const {
		return false;
	}
	virtual DuckDBResult<void> CreateDirectories(const std::string &path) const = 0;
	virtual bool IsRegularFile(const std::string &path) const = 0;
	virtual DuckDBResult<idx_t> FileSize(const std::string &path) const = 0;
	virtual DuckDBResult<void> WriteTextFileAtomically(const std::string &path, const std::string &contents) const = 0;
	virtual DuckDBResult<std::string> ReadTextFile(const std::string &path) const = 0;
	virtual DuckDBResult<idx_t> RemoveAll(const std::string &path) const = 0;
	virtual DuckDBResult<std::shared_ptr<arrow::io::OutputStream>> OpenArrowOutput(const std::string &path) const = 0;
	virtual DuckDBResult<std::shared_ptr<arrow::io::InputStream>> OpenArrowInput(const std::string &path) const = 0;
};

std::shared_ptr<ShuffleStorage> MakeDuckDBFileSystemShuffleStorage(FileSystem &fs);
std::shared_ptr<ShuffleStorage> MakeDuckDBFileSystemShuffleStorage(FileSystem &fs, FileOpener *opener);

class ShuffleCache {
public:
	//! Default flush threshold in bytes. Accumulated data beyond this triggers a file write.
	static constexpr idx_t DEFAULT_FLUSH_THRESHOLD_BYTES = 64ULL * 1024 * 1024; // 64 MB

	explicit ShuffleCache(ShuffleCacheConfig config);
	ShuffleCache(ShuffleCacheConfig config, std::shared_ptr<ShuffleStorage> storage);
	~ShuffleCache();

	const ShuffleCacheConfig &config() const;

	//! Buffer a DataChunk for the given partition. When the accumulated size
	//! exceeds the flush threshold the buffer is flushed to a single IPC file.
	DuckDBResult<void> WriteChunk(ClientContext &context, DataChunk &chunk, idx_t partition_idx,
	                              const vector<string> &names);
	DuckDBResult<std::vector<ShufflePartitionFile>> WriteCollection(ClientContext &context,
	                                                                ColumnDataCollection &collection,
	                                                                idx_t partition_idx, const vector<string> &names);
	DuckDBResult<std::unique_ptr<ColumnDataCollection>> ReadPartition(ClientContext &context, idx_t partition_idx,
	                                                                  const vector<LogicalType> &expected_types);
	DuckDBResult<std::shared_ptr<arrow::io::InputStream>> OpenPartitionFile(const std::string &path) const;

	//! Flush all remaining buffered data to disk. Called by the destructor and
	//! can also be called explicitly before the cache is destroyed.
	DuckDBResult<void> FlushAll(ClientContext &context, const vector<string> &names);

	//! Get the buffered column names.
	const vector<string> &BufferedNames() const {
		return buffered_names_;
	}

	DuckDBResult<void> RegisterPartitionFile(idx_t partition_idx, ShufflePartitionFile file);
	DuckDBResult<ShufflePartitionFiles> GetPartitionFiles(idx_t partition_idx) const;

	//! Write the schema file even if no data was written (needed for 0-row sinks).
	DuckDBResult<void> EnsureSchemaFile(ClientContext &context, const vector<LogicalType> &types,
	                                    const vector<string> &names);

	//! Persist an attempt-level commit manifest and marker after all buffered
	//! data has been flushed. This is the durable visibility boundary for a
	//! completed sink attempt.
	DuckDBResult<void> WriteAttemptManifest(idx_t sink_partition_id, idx_t attempt_id);
	bool HasCommittedManifest() const;
	DuckDBResult<ShufflePartitionFiles> GetPartitionFilesFromManifest(idx_t partition_idx) const;
	static DuckDBResult<ShuffleAttemptManifest> ReadAttemptManifest(const std::string &manifest_path,
	                                                                const std::string &committed_marker_path);
	static DuckDBResult<ShuffleAttemptManifest> ReadAttemptManifest(const ShuffleStorage &storage,
	                                                                const std::string &manifest_path,
	                                                                const std::string &committed_marker_path);
	static DuckDBResult<ShufflePartitionFiles> GetPartitionFilesFromManifest(const ShuffleAttemptManifest &manifest,
	                                                                         idx_t partition_idx);
	DuckDBResult<idx_t> RemoveAttemptStorage() const;
	std::string ManifestFilePath() const;
	std::string CommittedMarkerPath() const;

private:
	//! Flush a single partition buffer to an IPC file.
	DuckDBResult<ShufflePartitionFile> FlushBuffer(ClientContext &context, idx_t partition_idx,
	                                               const vector<string> &names);
	//! Write a ColumnDataCollection to a single IPC file (shared implementation).
	DuckDBResult<ShufflePartitionFile> WriteCollectionToFile(ClientContext &context, ColumnDataCollection &collection,
	                                                         idx_t partition_idx, const vector<string> &names);
	DuckDBResult<void> EnsurePartitionDirectory(idx_t partition_idx) const;
	std::string PartitionDirectory(idx_t partition_idx) const;
	std::string MakePartitionFilePath(idx_t partition_idx);
	std::string NodeDirectory() const;
	std::string SchemaFilePath() const;

	ShuffleCacheConfig config_;
	std::shared_ptr<ShuffleStorage> storage_;
	//! Unique identifier for this cache instance, used to avoid batch file
	//! overwrites when multiple sink tasks create separate ShuffleCache objects.
	std::string instance_id_;
	std::vector<ShufflePartitionFiles> partitions_;
	std::vector<std::atomic<uint64_t>> next_file_ids_;
	std::atomic<bool> schema_written_ {false};
	mutable std::mutex mutex_;

	//! Per-partition write buffers.
	std::vector<std::unique_ptr<ColumnDataCollection>> write_buffers_;
	//! Accumulated allocated bytes per partition buffer.
	std::vector<idx_t> buffer_bytes_;
	//! Flush threshold resolved when the cache is constructed.
	idx_t flush_threshold_bytes_;
	//! Per-partition mutex for thread-safe buffer access.
	std::vector<std::unique_ptr<std::mutex>> buffer_mutexes_;
	//! Column names captured on first WriteChunk call.
	vector<string> buffered_names_;
	//! Whether FlushAll has already been called.
	bool flushed_ = false;
};

} // namespace distributed
} // namespace duckdb
