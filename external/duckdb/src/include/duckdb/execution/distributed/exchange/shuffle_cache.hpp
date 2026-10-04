// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/optional_idx.hpp"
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
class TemporaryMemoryState;

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
	std::string exchange_id;
	std::string node_id;
	idx_t sink_partition_id = 0;
	idx_t attempt_id = 0;
	idx_t output_partition_count = 0;
	std::vector<ShuffleManifestPartitionFile> files;
};

struct ShuffleCacheConfig {
	std::string exchange_id;
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
DuckDBResult<idx_t> RemoveShuffleAttemptStorage(const ShuffleCacheConfig &config, const ShuffleStorage &storage);

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
	DuckDBResult<std::shared_ptr<arrow::io::InputStream>> OpenPartitionFile(const std::string &path) const;

	//! Flush all remaining buffered data to disk. This must be called explicitly
	//! before publishing the cache; the destructor only discards buffered data.
	DuckDBResult<void> FlushAll(ClientContext &context, const vector<string> &names);

	//! Get the buffered column names.
	vector<string> BufferedNames() const;
	//! Retained partition-buffer allocations charged to DuckDB's buffer pool.
	idx_t GetBufferedBytes() const {
		return buffered_bytes_.load();
	}
	//! Current manager-provided upper bound for retained partition buffers.
	idx_t GetBufferBudgetBytes() const {
		return buffer_budget_bytes_.load();
	}
	//! Peak retained bytes, including transient overshoot by concurrent appends.
	idx_t GetPeakBufferedBytes() const {
		return peak_buffered_bytes_.load();
	}
	//! Release uncommitted in-memory buffers without writing them.
	void DiscardBufferedData();

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
	std::string SchemaFilePath() const;

private:
	//! Flush a single partition buffer to an IPC file.
	DuckDBResult<ShufflePartitionFile> FlushBuffer(ClientContext &context, idx_t partition_idx,
	                                               const vector<string> &names);
	//! Flush a partition whose buffer mutex is already held.
	DuckDBResult<ShufflePartitionFile> FlushBufferLocked(ClientContext &context, idx_t partition_idx,
	                                                     const vector<string> &names);
	DuckDBResult<void> FlushUnderMemoryPressure(ClientContext &context, const vector<string> &names);
	void EnsureMemoryBudget(ClientContext &context);
	void RefreshMemoryBudget(ClientContext &context);
	void ReleaseMemoryReservation();
	void UpdatePeakBufferedBytes(idx_t buffered_bytes);
	//! Write a ColumnDataCollection to a single IPC file (shared implementation).
	DuckDBResult<ShufflePartitionFile> WriteCollectionToFile(ClientContext &context, ColumnDataCollection &collection,
	                                                         idx_t partition_idx, const vector<string> &names);
	DuckDBResult<void> EnsurePartitionDirectory(idx_t partition_idx) const;
	std::string PartitionDirectory(idx_t partition_idx) const;
	std::string MakePartitionFilePath(idx_t partition_idx);
	std::string NodeDirectory() const;

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
	std::vector<std::atomic<idx_t>> buffer_bytes_;
	//! Flush threshold resolved when the cache is constructed.
	idx_t flush_threshold_bytes_;
	//! Optional configured upper bound on the manager-provided memory budget.
	optional_idx configured_max_buffered_bytes_;
	std::atomic<idx_t> buffered_bytes_ {0};
	std::atomic<idx_t> peak_buffered_bytes_ {0};
	std::atomic<idx_t> buffer_budget_bytes_ {0};
	std::atomic<bool> memory_budget_initialized_ {false};
	//! Per-partition mutex for thread-safe buffer access.
	std::vector<std::unique_ptr<std::mutex>> buffer_mutexes_;
	//! Serializes cross-partition pressure flushes without serializing normal appends.
	std::mutex pressure_mutex_;
	std::mutex memory_state_mutex_;
	std::unique_ptr<TemporaryMemoryState> temporary_memory_state_;
	//! Column names captured on first WriteChunk call.
	mutable std::mutex buffered_names_mutex_;
	vector<string> buffered_names_;
	//! Whether FlushAll has already been called.
	bool flushed_ = false;
};

} // namespace distributed
} // namespace duckdb
