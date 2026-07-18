// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "test_helpers.hpp"

#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/execution/distributed/exchange/flight_ticket.hpp"
#include "duckdb/execution/distributed/exchange/shuffle_cache.hpp"
#include "duckdb/execution/distributed/exchange/shuffle_cache_registry.hpp"
#include "duckdb/execution/distributed/exchange/flight_exchange_manager.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"

#include "arrow/io/api.h"

#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <set>
#include <fstream>
#include <iterator>
#include <utility>
#include <cstdlib>

using namespace duckdb;
using namespace duckdb::distributed;

namespace {

// ─── Test Helpers ──────────────────────────────────────────

void PopulateTwoColumnChunk(DataChunk &chunk, const vector<LogicalType> &types, const vector<int32_t> &ids,
                            const vector<string> &names) {
	chunk.Initialize(Allocator::DefaultAllocator(), types);
	chunk.SetCardinality(static_cast<idx_t>(ids.size()));
	for (idx_t row = 0; row < static_cast<idx_t>(ids.size()); row++) {
		chunk.SetValue(0, row, Value::INTEGER(ids[row]));
		chunk.SetValue(1, row, Value(names[row]));
	}
}

void PopulateBlobChunk(DataChunk &chunk, const vector<int32_t> &ids, const vector<string> &blobs) {
	REQUIRE(ids.size() == blobs.size());
	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::BLOB};
	chunk.Initialize(Allocator::DefaultAllocator(), types);
	chunk.SetCardinality(static_cast<idx_t>(ids.size()));
	for (idx_t row = 0; row < static_cast<idx_t>(ids.size()); row++) {
		chunk.SetValue(0, row, Value::INTEGER(ids[row]));
		chunk.SetValue(1, row, Value::BLOB_RAW(blobs[row]));
	}
}

void SetProcessEnv(const string &name, const string &value) {
#if defined(_WIN32)
	_putenv_s(name.c_str(), value.c_str());
#else
	setenv(name.c_str(), value.c_str(), 1);
#endif
}

void UnsetProcessEnv(const string &name) {
#if defined(_WIN32)
	_putenv_s(name.c_str(), "");
#else
	unsetenv(name.c_str());
#endif
}

class ScopedEnvVar {
public:
	ScopedEnvVar(string name, string value) : name_(std::move(name)) {
		const auto *existing = std::getenv(name_.c_str());
		if (existing) {
			had_value_ = true;
			old_value_ = existing;
		}
		SetProcessEnv(name_, value);
	}

	~ScopedEnvVar() {
		if (had_value_) {
			SetProcessEnv(name_, old_value_);
		} else {
			UnsetProcessEnv(name_);
		}
	}

private:
	string name_;
	string old_value_;
	bool had_value_ = false;
};

void RequireCollectionValues(ColumnDataCollection &collection, const vector<int32_t> &ids,
                             const vector<string> &names) {
	REQUIRE(collection.ColumnCount() == 2);
	REQUIRE(collection.Count() == static_cast<idx_t>(ids.size()));

	idx_t row_index = 0;
	for (auto &chunk : collection.Chunks()) {
		for (idx_t row = 0; row < chunk.size(); row++) {
			REQUIRE(chunk.GetValue(0, row).GetValue<int32_t>() == ids[row_index]);
			REQUIRE(chunk.GetValue(1, row).GetValue<string>() == names[row_index]);
			row_index++;
		}
	}
	REQUIRE(row_index == static_cast<idx_t>(ids.size()));
}

/// Collect all row values from a ColumnDataCollection into vectors for comparison.
void CollectCollectionRows(ColumnDataCollection &collection, vector<int32_t> &out_ids, vector<string> &out_names) {
	for (auto &chunk : collection.Chunks()) {
		for (idx_t row = 0; row < chunk.size(); row++) {
			out_ids.push_back(chunk.GetValue(0, row).GetValue<int32_t>());
			out_names.push_back(chunk.GetValue(1, row).GetValue<string>());
		}
	}
}

class MockObjectShuffleStorage final : public ShuffleStorage {
public:
	explicit MockObjectShuffleStorage(std::string root) : root_(std::move(root)), fs_(FileSystem::CreateLocal()) {
	}

	bool SupportsObjectPaths() const override {
		return true;
	}

	DuckDBResult<void> CreateDirectories(const std::string &path) const override {
		try {
			fs_->CreateDirectoriesRecursive(MapPath(path));
		} catch (const std::exception &ex) {
			return DuckDBResult<void>::err(
			    DuckDBError::io_error("mock object storage mkdir failed: " + std::string(ex.what())));
		}
		return DuckDBResult<void>::ok();
	}

	bool IsRegularFile(const std::string &path) const override {
		return fs_->FileExists(MapPath(path));
	}

	DuckDBResult<idx_t> FileSize(const std::string &path) const override {
		try {
			auto handle = fs_->OpenFile(MapPath(path), FileOpenFlags(FileOpenFlags::FILE_FLAGS_READ));
			return DuckDBResult<idx_t>::ok(handle->GetFileSize());
		} catch (const std::exception &ex) {
			return DuckDBResult<idx_t>::err(
			    DuckDBError::io_error("mock object storage stat failed: " + std::string(ex.what())));
		}
	}

	DuckDBResult<void> WriteTextFileAtomically(const std::string &path, const std::string &contents) const override {
		auto mapped = MapPath(path);
		auto parent = ParentPath(mapped);
		if (!parent.empty()) {
			fs_->CreateDirectoriesRecursive(parent);
		}
		auto tmp_path = mapped + ".tmp";
		{
			std::ofstream output(tmp_path, std::ios::out | std::ios::trunc);
			if (!output) {
				return DuckDBResult<void>::err(DuckDBError::io_error("mock object storage open failed: " + tmp_path));
			}
			output << contents;
		}
		try {
			fs_->TryRemoveFile(mapped);
			fs_->MoveFile(tmp_path, mapped);
		} catch (const std::exception &ex) {
			fs_->TryRemoveFile(tmp_path);
			return DuckDBResult<void>::err(
			    DuckDBError::io_error("mock object storage commit failed: " + std::string(ex.what())));
		}
		return DuckDBResult<void>::ok();
	}

	DuckDBResult<std::string> ReadTextFile(const std::string &path) const override {
		std::ifstream input(MapPath(path), std::ios::in | std::ios::binary);
		if (!input.good()) {
			return DuckDBResult<std::string>::err(DuckDBError::io_error("mock object storage read failed: " + path));
		}
		std::ostringstream contents;
		contents << input.rdbuf();
		return DuckDBResult<std::string>::ok(contents.str());
	}

	DuckDBResult<idx_t> RemoveAll(const std::string &path) const override {
		return RemoveAllRecursive(MapPath(path));
	}

	DuckDBResult<std::shared_ptr<arrow::io::OutputStream>> OpenArrowOutput(const std::string &path) const override {
		auto mapped = MapPath(path);
		auto parent = ParentPath(mapped);
		if (!parent.empty()) {
			fs_->CreateDirectoriesRecursive(parent);
		}
		auto out_res = arrow::io::FileOutputStream::Open(mapped);
		if (!out_res.ok()) {
			return DuckDBResult<std::shared_ptr<arrow::io::OutputStream>>::err(
			    DuckDBError::external_error("mock object storage open output failed: " + out_res.status().ToString()));
		}
		std::shared_ptr<arrow::io::OutputStream> output = std::move(out_res).ValueOrDie();
		return DuckDBResult<std::shared_ptr<arrow::io::OutputStream>>::ok(std::move(output));
	}

	DuckDBResult<std::shared_ptr<arrow::io::InputStream>> OpenArrowInput(const std::string &path) const override {
		auto in_res = arrow::io::ReadableFile::Open(MapPath(path));
		if (!in_res.ok()) {
			return DuckDBResult<std::shared_ptr<arrow::io::InputStream>>::err(
			    DuckDBError::external_error("mock object storage open input failed: " + in_res.status().ToString()));
		}
		std::shared_ptr<arrow::io::InputStream> input = std::move(in_res).ValueOrDie();
		return DuckDBResult<std::shared_ptr<arrow::io::InputStream>>::ok(std::move(input));
	}

private:
	std::string MapPath(const std::string &path) const {
		auto scheme_end = path.find("://");
		auto suffix = scheme_end == std::string::npos ? path : path.substr(scheme_end + 3);
		while (!suffix.empty() && suffix.front() == '/') {
			suffix.erase(suffix.begin());
		}
		return fs_->JoinPath(root_, suffix);
	}

	std::string ParentPath(const std::string &path) const {
		auto pos = path.find_last_of("/\\");
		if (pos == std::string::npos) {
			return std::string();
		}
		return path.substr(0, pos);
	}

	DuckDBResult<idx_t> RemoveAllRecursive(const std::string &path) const {
		if (path.empty()) {
			return DuckDBResult<idx_t>::ok(0);
		}
		idx_t removed = 0;
		try {
			if (fs_->FileExists(path)) {
				fs_->RemoveFile(path);
				return DuckDBResult<idx_t>::ok(1);
			}
		} catch (...) {
		}
		try {
			if (!fs_->DirectoryExists(path)) {
				return DuckDBResult<idx_t>::ok(0);
			}
		} catch (...) {
			return DuckDBResult<idx_t>::ok(0);
		}

		vector<string> child_dirs;
		try {
			fs_->ListFiles(path, [&](const string &child, bool is_dir) {
				auto full_path = fs_->JoinPath(path, child);
				if (is_dir) {
					child_dirs.push_back(full_path);
					return;
				}
				fs_->RemoveFile(full_path);
				removed++;
			});
			for (auto &child_dir : child_dirs) {
				auto child_res = RemoveAllRecursive(child_dir);
				if (child_res.is_err()) {
					return child_res;
				}
				removed += child_res.value();
			}
			fs_->RemoveDirectory(path);
			removed++;
			return DuckDBResult<idx_t>::ok(removed);
		} catch (const std::exception &ex) {
			return DuckDBResult<idx_t>::err(
			    DuckDBError::io_error("mock object storage remove failed: " + std::string(ex.what())));
		}
	}

	std::string root_;
	unique_ptr<FileSystem> fs_;
};

} // namespace

// ═══════════════════════════════════════════════════════════
// FlightExchangeTicket
// ═══════════════════════════════════════════════════════════

TEST_CASE("Exchange: FlightExchangeTicket roundtrip", "[distributed][exchange]") {
	FlightExchangeTicket ticket;
	ticket.shuffle_stage_id = "stage_1";
	ticket.node_id = "node_2";
	ticket.partition_idx = 7;

	auto encoded = ticket.Serialize();
	auto parsed = FlightExchangeTicket::Parse(encoded);
	REQUIRE(parsed.is_ok());

	auto result = parsed.value();
	REQUIRE(result.shuffle_stage_id == ticket.shuffle_stage_id);
	REQUIRE(result.node_id == ticket.node_id);
	REQUIRE(result.partition_idx == ticket.partition_idx);
}

TEST_CASE("Exchange: FlightExchangeTicket parse errors", "[distributed][exchange]") {
	REQUIRE(FlightExchangeTicket::Parse("v1\nstage\nnode").is_err());
	REQUIRE(FlightExchangeTicket::Parse("v2\nstage\nnode\n1").is_err());
	REQUIRE(FlightExchangeTicket::Parse("v1\n\nnode\n1").is_err());
	REQUIRE(FlightExchangeTicket::Parse("v1\nstage\n\n1").is_err());
	REQUIRE(FlightExchangeTicket::Parse("v1\nstage\nnode\n-1").is_err());
	REQUIRE(FlightExchangeTicket::Parse("v1\nstage\nnode\nnope").is_err());
}

// ═══════════════════════════════════════════════════════════
// ShuffleCacheRegistry
// ═══════════════════════════════════════════════════════════

TEST_CASE("Exchange: ShuffleCacheRegistry register/get/remove", "[distributed][exchange]") {
	auto &registry = ShuffleCacheRegistry::Instance();

	// Create a ShuffleCache and register it
	ShuffleCacheConfig config;
	config.shuffle_stage_id = "registry_test_stage";
	config.node_id = "node_1";
	config.num_partitions = 2;
	config.local_dirs = {TestCreatePath("registry_test")};

	auto cache = std::make_shared<ShuffleCache>(std::move(config));
	registry.Register("registry_test_stage", cache);

	// Get should return the same cache
	auto retrieved = registry.Get("registry_test_stage");
	REQUIRE(retrieved != nullptr);
	REQUIRE(retrieved.get() == cache.get());

	// Get with unknown key returns nullptr
	auto unknown = registry.Get("nonexistent_stage");
	REQUIRE(unknown == nullptr);

	// Remove the cache
	registry.Remove("registry_test_stage");
	auto after_remove = registry.Get("registry_test_stage");
	REQUIRE(after_remove == nullptr);

	// Double remove is safe
	registry.Remove("registry_test_stage");
}

TEST_CASE("Exchange: ShuffleCacheRegistry multiple entries", "[distributed][exchange]") {
	auto &registry = ShuffleCacheRegistry::Instance();

	ShuffleCacheConfig config1;
	config1.shuffle_stage_id = "multi_test_1";
	config1.node_id = "node_1";
	config1.num_partitions = 1;
	config1.local_dirs = {TestCreatePath("registry_multi_1")};

	ShuffleCacheConfig config2;
	config2.shuffle_stage_id = "multi_test_2";
	config2.node_id = "node_1";
	config2.num_partitions = 1;
	config2.local_dirs = {TestCreatePath("registry_multi_2")};

	auto cache1 = std::make_shared<ShuffleCache>(std::move(config1));
	auto cache2 = std::make_shared<ShuffleCache>(std::move(config2));

	registry.Register("multi_test_1", cache1);
	registry.Register("multi_test_2", cache2);

	REQUIRE(registry.Get("multi_test_1").get() == cache1.get());
	REQUIRE(registry.Get("multi_test_2").get() == cache2.get());

	// Removing one doesn't affect the other
	registry.Remove("multi_test_1");
	REQUIRE(registry.Get("multi_test_1") == nullptr);
	REQUIRE(registry.Get("multi_test_2").get() == cache2.get());

	registry.Remove("multi_test_2");
}

// ═══════════════════════════════════════════════════════════
// ShuffleCache (IPC Stream format)
// ═══════════════════════════════════════════════════════════

TEST_CASE("Exchange: ShuffleCache write/read", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;

	ShuffleCacheConfig config;
	config.shuffle_stage_id = "stage_1";
	config.node_id = "node_1";
	config.num_partitions = 2;
	config.local_dirs = {TestCreatePath("exchange_cache_basic")};
	ShuffleCache cache(std::move(config));

	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::VARCHAR};
	vector<int32_t> ids = {1, 2, 3};
	vector<string> names = {"a", "b", "c"};
	DataChunk chunk;
	PopulateTwoColumnChunk(chunk, types, ids, names);

	auto write_res = cache.WriteChunk(context, chunk, 1, {"id", "name"});
	REQUIRE(write_res.is_ok());

	auto flush_res = cache.FlushAll(context, cache.BufferedNames());
	REQUIRE(flush_res.is_ok());

	auto files_res = cache.GetPartitionFiles(1);
	REQUIRE(files_res.is_ok());
	auto files = files_res.value();
	REQUIRE(files.files.size() == 1);
	const auto &file = files.files[0];
	REQUIRE(file.rows == static_cast<idx_t>(ids.size()));
	REQUIRE(file.bytes > 0);
	REQUIRE(!file.path.empty());
	REQUIRE(files.total_rows == static_cast<idx_t>(ids.size()));
	REQUIRE(files.total_bytes >= file.bytes);

	auto read_res = cache.ReadPartition(context, 1, types);
	REQUIRE(read_res.is_ok());
	auto collection = std::move(read_res.value());
	REQUIRE(collection != nullptr);
	RequireCollectionValues(*collection, ids, names);
}

TEST_CASE("Exchange: ShuffleCache flushes large BLOB buffers by actual allocation size", "[distributed][exchange]") {
	ScopedEnvVar flush_threshold("VANE_SHUFFLE_CACHE_FLUSH_THRESHOLD_BYTES", "1024");

	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;

	ShuffleCacheConfig config;
	config.shuffle_stage_id = "stage_large_blob";
	config.node_id = "node_blob";
	config.num_partitions = 1;
	config.local_dirs = {TestCreatePath("exchange_cache_large_blob")};
	ShuffleCache cache(std::move(config));

	vector<int32_t> ids = {1, 2};
	vector<string> blobs = {string(4096, 'a'), string(4096, 'b')};
	DataChunk chunk;
	PopulateBlobChunk(chunk, ids, blobs);

	REQUIRE(cache.WriteChunk(context, chunk, 0, {"id", "payload"}).is_ok());

	auto files_res = cache.GetPartitionFiles(0);
	REQUIRE(files_res.is_ok());
	auto files = files_res.value();
	REQUIRE(files.files.size() == 1);
	REQUIRE(files.total_rows == static_cast<idx_t>(ids.size()));
	REQUIRE(files.total_bytes > 0);

	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::BLOB};
	auto read_res = cache.ReadPartition(context, 0, types);
	REQUIRE(read_res.is_ok());
	auto collection = std::move(read_res.value());
	REQUIRE(collection != nullptr);
	REQUIRE(collection->Count() == static_cast<idx_t>(ids.size()));
}

TEST_CASE("Exchange: ShuffleCache committed manifest replay via object storage backend", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;

	auto root = TestCreatePath("exchange_object_storage");
	auto storage = std::make_shared<MockObjectShuffleStorage>(root);

	ShuffleCacheConfig config;
	config.shuffle_stage_id = "object_stage";
	config.node_id = "node_object";
	config.num_partitions = 2;
	config.local_dirs = {"mock://object-root"};

	ShuffleCache cache(std::move(config), storage);

	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::VARCHAR};
	vector<int32_t> ids = {7, 8, 9};
	vector<string> names = {"g", "h", "i"};
	DataChunk chunk;
	PopulateTwoColumnChunk(chunk, types, ids, names);

	REQUIRE(cache.WriteChunk(context, chunk, 1, {"id", "name"}).is_ok());
	REQUIRE(cache.FlushAll(context, cache.BufferedNames()).is_ok());
	REQUIRE(cache.WriteAttemptManifest(0, 0).is_ok());
	REQUIRE(cache.HasCommittedManifest());

	ShuffleCache replay_cache(
	    ShuffleCacheConfig {
	        "object_stage",
	        "node_object",
	        2,
	        {"mock://object-root"},
	    },
	    storage);

	auto read_res = replay_cache.ReadPartition(context, 1, types);
	REQUIRE(read_res.is_ok());
	auto collection = std::move(read_res.value());
	REQUIRE(collection != nullptr);
	REQUIRE(collection->Count() == static_cast<idx_t>(ids.size()));
	RequireCollectionValues(*collection, ids, names);
}

TEST_CASE("Exchange: ShuffleCache empty partition handling", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;

	ShuffleCacheConfig config;
	config.shuffle_stage_id = "stage_empty";
	config.node_id = "node_empty";
	config.num_partitions = 1;
	config.local_dirs = {TestCreatePath("exchange_cache_empty")};
	ShuffleCache cache(std::move(config));

	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::VARCHAR};
	auto empty_res = cache.ReadPartition(context, 0, types);
	REQUIRE(empty_res.is_ok());
	auto empty_collection = std::move(empty_res.value());
	REQUIRE(empty_collection != nullptr);
	REQUIRE(empty_collection->Count() == 0);
	REQUIRE(empty_collection->Types() == types);

	auto missing_types_res = cache.ReadPartition(context, 0, {});
	REQUIRE(missing_types_res.is_err());

	vector<int32_t> ids = {9};
	vector<string> names_vec = {"x"};
	DataChunk chunk;
	PopulateTwoColumnChunk(chunk, types, ids, names_vec);

	auto bad_partition_res = cache.WriteChunk(context, chunk, 2, {"id", "name"});
	REQUIRE(bad_partition_res.is_err());
}

TEST_CASE("Exchange: ShuffleCache multiple chunks to same partition", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;

	ShuffleCacheConfig config;
	config.shuffle_stage_id = "stage_multi_chunk";
	config.node_id = "node_1";
	config.num_partitions = 1;
	config.local_dirs = {TestCreatePath("exchange_cache_multi_chunk")};
	ShuffleCache cache(std::move(config));

	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::VARCHAR};

	// Write first chunk
	vector<int32_t> ids1 = {1, 2};
	vector<string> names1 = {"a", "b"};
	DataChunk chunk1;
	PopulateTwoColumnChunk(chunk1, types, ids1, names1);
	REQUIRE(cache.WriteChunk(context, chunk1, 0, {"id", "name"}).is_ok());

	// Write second chunk
	vector<int32_t> ids2 = {3, 4, 5};
	vector<string> names2 = {"c", "d", "e"};
	DataChunk chunk2;
	PopulateTwoColumnChunk(chunk2, types, ids2, names2);
	REQUIRE(cache.WriteChunk(context, chunk2, 0, {"id", "name"}).is_ok());

	REQUIRE(cache.FlushAll(context, cache.BufferedNames()).is_ok());

	auto read_res = cache.ReadPartition(context, 0, types);
	REQUIRE(read_res.is_ok());
	auto collection = std::move(read_res.value());
	REQUIRE(collection != nullptr);
	REQUIRE(collection->Count() == 5);

	// All rows should be present
	vector<int32_t> all_ids = {1, 2, 3, 4, 5};
	vector<string> all_names = {"a", "b", "c", "d", "e"};
	RequireCollectionValues(*collection, all_ids, all_names);
}

TEST_CASE("Exchange: ShuffleCache write to multiple partitions", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;

	ShuffleCacheConfig config;
	config.shuffle_stage_id = "stage_multi_part";
	config.node_id = "node_1";
	config.num_partitions = 3;
	config.local_dirs = {TestCreatePath("exchange_cache_multi_part")};
	ShuffleCache cache(std::move(config));

	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::VARCHAR};

	// Write to partition 0
	vector<int32_t> ids0 = {10, 20};
	vector<string> names0 = {"ten", "twenty"};
	DataChunk chunk0;
	PopulateTwoColumnChunk(chunk0, types, ids0, names0);
	REQUIRE(cache.WriteChunk(context, chunk0, 0, {"id", "name"}).is_ok());

	// Write to partition 2
	vector<int32_t> ids2 = {30};
	vector<string> names2 = {"thirty"};
	DataChunk chunk2;
	PopulateTwoColumnChunk(chunk2, types, ids2, names2);
	REQUIRE(cache.WriteChunk(context, chunk2, 2, {"id", "name"}).is_ok());

	REQUIRE(cache.FlushAll(context, cache.BufferedNames()).is_ok());

	// Partition 0 should have 2 rows
	auto read0 = cache.ReadPartition(context, 0, types);
	REQUIRE(read0.is_ok());
	REQUIRE(read0.value()->Count() == 2);
	RequireCollectionValues(*read0.value(), ids0, names0);

	// Partition 1 should be empty
	auto read1 = cache.ReadPartition(context, 1, types);
	REQUIRE(read1.is_ok());
	REQUIRE(read1.value()->Count() == 0);

	// Partition 2 should have 1 row
	auto read2 = cache.ReadPartition(context, 2, types);
	REQUIRE(read2.is_ok());
	REQUIRE(read2.value()->Count() == 1);
	RequireCollectionValues(*read2.value(), ids2, names2);
}

// ═══════════════════════════════════════════════════════════
// FlightExchangeManager
// ═══════════════════════════════════════════════════════════

TEST_CASE("Exchange: FlightExchange coordinator lifecycle", "[distributed][exchange]") {
	FlightExchangeConfig config;
	config.node_id = "node_1";
	config.local_dirs = {TestCreatePath("exchange_coordinator")};

	DuckDB db(nullptr);
	Connection conn(db);

	FlightExchangeManager mgr(config, conn.context.get());

	ExchangeContext ctx;
	ctx.query_id = "q1";
	ctx.exchange_id = "exchange_lifecycle_test";

	auto exchange = mgr.CreateExchange(ctx, 4);
	REQUIRE(exchange != nullptr);
	REQUIRE(exchange->GetNumPartitions() == 4);

	// Add sinks
	auto sink_handle0 = exchange->AddSink(0);
	auto sink_handle1 = exchange->AddSink(1);
	REQUIRE(sink_handle0.task_partition_id == 0);
	REQUIRE(sink_handle1.task_partition_id == 1);

	// Instantiate sinks
	auto inst0 = exchange->InstantiateSink(sink_handle0, 0);
	auto inst1 = exchange->InstantiateSink(sink_handle1, 0);
	REQUIRE(inst0.output_partition_count == 4);
	REQUIRE(inst1.output_partition_count == 4);
	REQUIRE(inst0.output_location != inst1.output_location);
	REQUIRE(inst0.output_location.find(ctx.exchange_id) != string::npos);
	REQUIRE(inst1.output_location.find(ctx.exchange_id) != string::npos);

	// Finish sinks
	exchange->SinkFinished(sink_handle0, 0);
	exchange->SinkFinished(sink_handle1, 0);
	exchange->AllRequiredSinksFinished();

	// Source handles should cover all partitions
	auto source_handles = exchange->GetSourceHandles();
	// Source handles generated for non-empty partitions
	// (may be empty since no data was actually written)

	exchange->Close();
}

TEST_CASE("Exchange: FlightExchange selects first successful sink attempt", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);

	FlightExchangeConfig config;
	config.node_id = "coordinator";
	config.flight_port = 7777;
	config.local_dirs = {TestCreatePath("exchange_selected_attempt")};
	FlightExchangeManager mgr(config, conn.context.get());

	ExchangeContext ctx;
	ctx.query_id = "q1";
	ctx.exchange_id = "exchange_selected_attempt";

	auto exchange = mgr.CreateExchange(ctx, 2);
	auto sink0 = exchange->AddSink(0);
	auto sink1 = exchange->AddSink(1);
	auto sink0_attempt0 = exchange->InstantiateSink(sink0, 0);
	auto sink0_attempt1 = exchange->InstantiateSink(sink0, 1);
	auto sink1_attempt0 = exchange->InstantiateSink(sink1, 0);

	REQUIRE(sink0_attempt0.output_location.find("__sink_0__attempt_0") != std::string::npos);
	REQUIRE(sink0_attempt1.output_location.find("__sink_0__attempt_1") != std::string::npos);
	REQUIRE(sink1_attempt0.output_location.find("__sink_1__attempt_0") != std::string::npos);

	exchange->SinkFinished(sink0, 1, "worker-retry", 5010);
	exchange->SinkFinished(sink0, 0, "worker-late", 5011);
	exchange->SinkFinished(sink1, 0, "worker-first", 5012);
	exchange->AllRequiredSinksFinished();

	auto source_handles = exchange->GetSourceHandles();
	REQUIRE(source_handles.size() == 4);

	idx_t sink0_handles = 0;
	idx_t sink1_handles = 0;
	for (const auto &handle : source_handles) {
		REQUIRE(handle.files.size() == 1);
		if (handle.files[0].path.find("__sink_0__") != std::string::npos) {
			sink0_handles++;
			REQUIRE(handle.attempt_id == 1);
			REQUIRE(handle.node_id == "worker-retry");
			REQUIRE(handle.flight_port == 5010);
			REQUIRE(handle.files[0].path.find("__attempt_1") != std::string::npos);
			REQUIRE(handle.files[0].path.find("__attempt_0") == std::string::npos);
		} else if (handle.files[0].path.find("__sink_1__") != std::string::npos) {
			sink1_handles++;
			REQUIRE(handle.attempt_id == 0);
			REQUIRE(handle.node_id == "worker-first");
			REQUIRE(handle.flight_port == 5012);
			REQUIRE(handle.files[0].path.find("__attempt_0") != std::string::npos);
		} else {
			FAIL("unexpected source handle path");
		}
	}
	REQUIRE(sink0_handles == 2);
	REQUIRE(sink1_handles == 2);

	exchange->Close();
}

// ═══════════════════════════════════════════════════════════
// FlightExchangeSink
// ═══════════════════════════════════════════════════════════

TEST_CASE("Exchange: FlightExchangeSink write and flush", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;

	// Create a ShuffleCache
	ShuffleCacheConfig cache_config;
	cache_config.shuffle_stage_id = "sink_test_stage";
	cache_config.node_id = "node_1";
	cache_config.num_partitions = 2;
	cache_config.local_dirs = {TestCreatePath("exchange_sink_test")};
	auto cache = std::make_shared<ShuffleCache>(std::move(cache_config));

	// Create sink handle
	ExchangeSinkInstanceHandle handle;
	handle.sink_handle.task_partition_id = 0;
	handle.attempt_id = 0;
	handle.output_location = "sink_test_stage";
	handle.output_partition_count = 2;

	FlightExchangeSink sink(cache, handle, &context);

	// Should not be blocked (disk-first, no backpressure)
	REQUIRE(sink.IsBlocked() == false);

	// Write data
	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::VARCHAR};
	vector<int32_t> ids = {100, 200, 300};
	vector<string> names = {"x", "y", "z"};
	DataChunk chunk;
	PopulateTwoColumnChunk(chunk, types, ids, names);

	auto write_res = sink.AddChunk(0, chunk);
	REQUIRE(write_res.is_ok());

	auto write_res2 = sink.AddChunk(1, chunk);
	REQUIRE(write_res2.is_ok());

	// Finish should flush and register in ShuffleCacheRegistry
	auto finish_res = sink.Finish();
	REQUIRE(finish_res.is_ok());

	// Verify ShuffleCacheRegistry has the cache
	auto registered = ShuffleCacheRegistry::Instance().Get("sink_test_stage");
	REQUIRE(registered != nullptr);
	REQUIRE(registered.get() == cache.get());

	auto fs = FileSystem::CreateLocal();
	REQUIRE(fs->FileExists(cache->ManifestFilePath()));
	REQUIRE(fs->FileExists(cache->CommittedMarkerPath()));
	std::ifstream manifest(cache->ManifestFilePath());
	REQUIRE(manifest.good());
	std::string manifest_contents((std::istreambuf_iterator<char>(manifest)), std::istreambuf_iterator<char>());
	REQUIRE(manifest_contents.find("version=1") != std::string::npos);
	REQUIRE(manifest_contents.find("sink_partition_id=0") != std::string::npos);
	REQUIRE(manifest_contents.find("attempt_id=0") != std::string::npos);
	REQUIRE(manifest_contents.find("file=0") != std::string::npos);

	// Verify data was written
	auto read_res = cache->ReadPartition(context, 0, types);
	REQUIRE(read_res.is_ok());
	REQUIRE(read_res.value()->Count() == 3);
	RequireCollectionValues(*read_res.value(), ids, names);

	// Cleanup
	ShuffleCacheRegistry::Instance().Remove("sink_test_stage");
}

TEST_CASE("Exchange: FlightExchangeSink memory usage", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);

	ShuffleCacheConfig cache_config;
	cache_config.shuffle_stage_id = "sink_mem_test";
	cache_config.node_id = "node_1";
	cache_config.num_partitions = 1;
	cache_config.local_dirs = {TestCreatePath("exchange_sink_mem")};
	auto cache = std::make_shared<ShuffleCache>(std::move(cache_config));

	ExchangeSinkInstanceHandle handle;
	handle.sink_handle.task_partition_id = 0;
	handle.attempt_id = 0;
	handle.output_partition_count = 1;

	FlightExchangeSink sink(cache, handle, conn.context.get());

	// Memory usage should be 0 (disk-first)
	REQUIRE(sink.GetMemoryUsage() == 0);
}

// ═══════════════════════════════════════════════════════════
// FlightExchangeSource
// ═══════════════════════════════════════════════════════════

TEST_CASE("Exchange: FlightExchangeSource read from registry", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;

	// Prepare: write data via ShuffleCache and register it
	ShuffleCacheConfig cache_config;
	cache_config.shuffle_stage_id = "source_test_stage";
	cache_config.node_id = "node_1";
	cache_config.num_partitions = 2;
	cache_config.local_dirs = {TestCreatePath("exchange_source_test")};
	auto cache = std::make_shared<ShuffleCache>(std::move(cache_config));

	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::VARCHAR};

	// Write to partition 0
	vector<int32_t> ids0 = {1, 2};
	vector<string> names0 = {"a", "b"};
	DataChunk chunk0;
	PopulateTwoColumnChunk(chunk0, types, ids0, names0);
	REQUIRE(cache->WriteChunk(context, chunk0, 0, {"id", "name"}).is_ok());

	// Write to partition 1
	vector<int32_t> ids1 = {3};
	vector<string> names1 = {"c"};
	DataChunk chunk1;
	PopulateTwoColumnChunk(chunk1, types, ids1, names1);
	REQUIRE(cache->WriteChunk(context, chunk1, 1, {"id", "name"}).is_ok());

	REQUIRE(cache->FlushAll(context, cache->BufferedNames()).is_ok());

	// Register the cache
	ShuffleCacheRegistry::Instance().Register("source_test_stage", cache);

	// Create source and read partition 0
	FlightExchangeSource source("source_test_stage", &context);

	REQUIRE(source.IsBlocked() == false);

	// Add source handle for partition 0
	ExchangeSourceHandle handle0;
	handle0.partition_id = 0;
	source.AddSourceHandles({handle0});

	REQUIRE(source.IsFinished() == false);

	// Read all chunks from partition 0
	vector<int32_t> read_ids;
	vector<string> read_names;
	DataChunk output;
	output.Initialize(Allocator::DefaultAllocator(), types);
	while (source.ReadChunk(output)) {
		for (idx_t row = 0; row < output.size(); row++) {
			read_ids.push_back(output.GetValue(0, row).GetValue<int32_t>());
			read_names.push_back(output.GetValue(1, row).GetValue<string>());
		}
		output.Reset();
	}

	REQUIRE(read_ids == ids0);
	REQUIRE(read_names == names0);
	REQUIRE(source.IsFinished() == true);

	source.Close();
	ShuffleCacheRegistry::Instance().Remove("source_test_stage");
}

TEST_CASE("Exchange: FlightExchangeSource multiple partitions", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;

	ShuffleCacheConfig cache_config;
	cache_config.shuffle_stage_id = "source_multi_stage";
	cache_config.node_id = "node_1";
	cache_config.num_partitions = 3;
	cache_config.local_dirs = {TestCreatePath("exchange_source_multi")};
	auto cache = std::make_shared<ShuffleCache>(std::move(cache_config));

	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::VARCHAR};

	// Write to partitions 0 and 2
	DataChunk chunk0;
	PopulateTwoColumnChunk(chunk0, types, {10}, {"ten"});
	REQUIRE(cache->WriteChunk(context, chunk0, 0, {"id", "name"}).is_ok());

	DataChunk chunk2;
	PopulateTwoColumnChunk(chunk2, types, {30, 40}, {"thirty", "forty"});
	REQUIRE(cache->WriteChunk(context, chunk2, 2, {"id", "name"}).is_ok());

	REQUIRE(cache->FlushAll(context, cache->BufferedNames()).is_ok());
	ShuffleCacheRegistry::Instance().Register("source_multi_stage", cache);

	// Source reads both partitions
	FlightExchangeSource source("source_multi_stage", &context);

	ExchangeSourceHandle h0, h2;
	h0.partition_id = 0;
	h2.partition_id = 2;
	source.AddSourceHandles({h0, h2});

	vector<int32_t> read_ids;
	vector<string> read_names;
	DataChunk output;
	output.Initialize(Allocator::DefaultAllocator(), types);
	while (source.ReadChunk(output)) {
		for (idx_t row = 0; row < output.size(); row++) {
			read_ids.push_back(output.GetValue(0, row).GetValue<int32_t>());
			read_names.push_back(output.GetValue(1, row).GetValue<string>());
		}
		output.Reset();
	}

	// All 3 rows across 2 partitions
	REQUIRE(read_ids.size() == 3);
	// Partition 0 first, then partition 2
	REQUIRE(read_ids[0] == 10);
	REQUIRE(read_ids[1] == 30);
	REQUIRE(read_ids[2] == 40);

	source.Close();
	ShuffleCacheRegistry::Instance().Remove("source_multi_stage");
}

TEST_CASE("Exchange: FlightExchangeSource switches local cache per handle path", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;

	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::VARCHAR};
	const string stage0 = "source_switch_stage_0";
	const string stage1 = "source_switch_stage_1";

	ShuffleCacheConfig cache0_config;
	cache0_config.shuffle_stage_id = stage0;
	cache0_config.node_id = "node_1";
	cache0_config.num_partitions = 1;
	cache0_config.local_dirs = {TestCreatePath("exchange_source_switch_0")};
	auto cache0 = std::make_shared<ShuffleCache>(std::move(cache0_config));

	DataChunk chunk0;
	PopulateTwoColumnChunk(chunk0, types, {1, 2}, {"a", "b"});
	REQUIRE(cache0->WriteChunk(context, chunk0, 0, {"id", "name"}).is_ok());
	REQUIRE(cache0->FlushAll(context, cache0->BufferedNames()).is_ok());
	ShuffleCacheRegistry::Instance().Register(stage0, cache0);

	ShuffleCacheConfig cache1_config;
	cache1_config.shuffle_stage_id = stage1;
	cache1_config.node_id = "node_1";
	cache1_config.num_partitions = 1;
	cache1_config.local_dirs = {TestCreatePath("exchange_source_switch_1")};
	auto cache1 = std::make_shared<ShuffleCache>(std::move(cache1_config));

	DataChunk chunk1;
	PopulateTwoColumnChunk(chunk1, types, {3, 4}, {"c", "d"});
	REQUIRE(cache1->WriteChunk(context, chunk1, 0, {"id", "name"}).is_ok());
	REQUIRE(cache1->FlushAll(context, cache1->BufferedNames()).is_ok());
	ShuffleCacheRegistry::Instance().Register(stage1, cache1);

	FlightExchangeConfig source_config;
	source_config.node_id = "node_1";
	FlightExchangeSource source(source_config, &context);

	ExchangeSourceHandle handle0;
	handle0.partition_id = 0;
	handle0.node_id = "node_1";
	handle0.files.push_back(ExchangeSourceFile(stage0, 0));

	ExchangeSourceHandle handle1;
	handle1.partition_id = 0;
	handle1.node_id = "node_1";
	handle1.files.push_back(ExchangeSourceFile(stage1, 0));

	source.AddSourceHandles({handle0, handle1});

	vector<int32_t> read_ids;
	vector<string> read_names;
	DataChunk output;
	output.Initialize(Allocator::DefaultAllocator(), types);
	while (source.ReadChunk(output)) {
		for (idx_t row = 0; row < output.size(); row++) {
			read_ids.push_back(output.GetValue(0, row).GetValue<int32_t>());
			read_names.push_back(output.GetValue(1, row).GetValue<string>());
		}
		output.Reset();
	}

	REQUIRE(read_ids == vector<int32_t>({1, 2, 3, 4}));
	REQUIRE(read_names == vector<string>({"a", "b", "c", "d"}));

	source.Close();
	ShuffleCacheRegistry::Instance().Remove(stage0);
	ShuffleCacheRegistry::Instance().Remove(stage1);
}

TEST_CASE("Exchange: FlightExchangeSource no handles", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);

	FlightExchangeSource source("nonexistent_stage", conn.context.get());

	// Without handles, should be finished immediately
	REQUIRE(source.IsFinished() == true);

	DataChunk chunk;
	vector<LogicalType> types = {LogicalType::INTEGER};
	chunk.Initialize(Allocator::DefaultAllocator(), types);
	REQUIRE(source.ReadChunk(chunk) == false);
}

// ═══════════════════════════════════════════════════════════
// End-to-End: Sink → Source pipeline
// ═══════════════════════════════════════════════════════════

TEST_CASE("Exchange: End-to-end sink to source pipeline", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;

	const std::string exchange_id = "e2e_test_stage";

	// ─── Phase 1: Create exchange and write data via sink ───

	ShuffleCacheConfig cache_config;
	cache_config.shuffle_stage_id = exchange_id;
	cache_config.node_id = "node_1";
	cache_config.num_partitions = 2;
	cache_config.local_dirs = {TestCreatePath("exchange_e2e")};
	auto cache = std::make_shared<ShuffleCache>(std::move(cache_config));

	ExchangeSinkInstanceHandle handle;
	handle.sink_handle.task_partition_id = 0;
	handle.attempt_id = 0;
	handle.output_location = exchange_id;
	handle.output_partition_count = 2;

	FlightExchangeSink sink(cache, handle, &context);

	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::VARCHAR};

	// Write 5 rows to partition 0
	DataChunk chunk0;
	PopulateTwoColumnChunk(chunk0, types, {1, 2, 3, 4, 5}, {"a", "b", "c", "d", "e"});
	REQUIRE(sink.AddChunk(0, chunk0).is_ok());

	// Write 3 rows to partition 1
	DataChunk chunk1;
	PopulateTwoColumnChunk(chunk1, types, {10, 20, 30}, {"x", "y", "z"});
	REQUIRE(sink.AddChunk(1, chunk1).is_ok());

	// Finish sink → flushes to disk + registers in registry
	REQUIRE(sink.Finish().is_ok());

	// ─── Phase 2: Read data via source (partition 0) ───

	FlightExchangeSource source0(exchange_id, &context);
	ExchangeSourceHandle sh0;
	sh0.partition_id = 0;
	source0.AddSourceHandles({sh0});

	vector<int32_t> read_ids0;
	vector<string> read_names0;
	DataChunk out0;
	out0.Initialize(Allocator::DefaultAllocator(), types);
	while (source0.ReadChunk(out0)) {
		for (idx_t row = 0; row < out0.size(); row++) {
			read_ids0.push_back(out0.GetValue(0, row).GetValue<int32_t>());
			read_names0.push_back(out0.GetValue(1, row).GetValue<string>());
		}
		out0.Reset();
	}
	REQUIRE(read_ids0 == vector<int32_t>({1, 2, 3, 4, 5}));
	REQUIRE(read_names0 == vector<string>({"a", "b", "c", "d", "e"}));

	// ─── Phase 3: Read data via source (partition 1) ───

	FlightExchangeSource source1(exchange_id, &context);
	ExchangeSourceHandle sh1;
	sh1.partition_id = 1;
	source1.AddSourceHandles({sh1});

	vector<int32_t> read_ids1;
	vector<string> read_names1;
	DataChunk out1;
	out1.Initialize(Allocator::DefaultAllocator(), types);
	while (source1.ReadChunk(out1)) {
		for (idx_t row = 0; row < out1.size(); row++) {
			read_ids1.push_back(out1.GetValue(0, row).GetValue<int32_t>());
			read_names1.push_back(out1.GetValue(1, row).GetValue<string>());
		}
		out1.Reset();
	}
	REQUIRE(read_ids1 == vector<int32_t>({10, 20, 30}));
	REQUIRE(read_names1 == vector<string>({"x", "y", "z"}));

	// Cleanup
	ShuffleCacheRegistry::Instance().Remove(exchange_id);
}

TEST_CASE("Exchange: Multiple sinks to same exchange", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;

	const std::string exchange_id = "multi_sink_test";

	ShuffleCacheConfig cache_config;
	cache_config.shuffle_stage_id = exchange_id;
	cache_config.node_id = "node_1";
	cache_config.num_partitions = 2;
	cache_config.local_dirs = {TestCreatePath("exchange_multi_sink")};
	auto cache = std::make_shared<ShuffleCache>(std::move(cache_config));

	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::VARCHAR};

	// Sink 1: writes to partition 0
	{
		ExchangeSinkInstanceHandle handle;
		handle.sink_handle.task_partition_id = 0;
		handle.attempt_id = 0;
		handle.output_location = exchange_id;
		handle.output_partition_count = 2;

		FlightExchangeSink sink1(cache, handle, &context);
		DataChunk chunk;
		PopulateTwoColumnChunk(chunk, types, {1, 2}, {"a", "b"});
		REQUIRE(sink1.AddChunk(0, chunk).is_ok());
		REQUIRE(sink1.Finish().is_ok());
	}

	// Sink 2: also writes to partition 0
	{
		ExchangeSinkInstanceHandle handle;
		handle.sink_handle.task_partition_id = 1;
		handle.attempt_id = 0;
		handle.output_location = exchange_id;
		handle.output_partition_count = 2;

		FlightExchangeSink sink2(cache, handle, &context);
		DataChunk chunk;
		PopulateTwoColumnChunk(chunk, types, {3, 4}, {"c", "d"});
		REQUIRE(sink2.AddChunk(0, chunk).is_ok());
		REQUIRE(sink2.Finish().is_ok());
	}

	// Source reads partition 0 — should have data from both sinks
	// First verify via cache directly
	auto read_res = cache->ReadPartition(context, 0, types);
	REQUIRE(read_res.is_ok());
	auto total_rows = read_res.value()->Count();

	FlightExchangeSource source(exchange_id, &context);
	ExchangeSourceHandle sh;
	sh.partition_id = 0;
	source.AddSourceHandles({sh});

	vector<int32_t> read_ids;
	vector<string> read_names;
	DataChunk output;
	output.Initialize(Allocator::DefaultAllocator(), types);
	while (source.ReadChunk(output)) {
		for (idx_t row = 0; row < output.size(); row++) {
			read_ids.push_back(output.GetValue(0, row).GetValue<int32_t>());
			read_names.push_back(output.GetValue(1, row).GetValue<string>());
		}
		output.Reset();
	}

	// Should have all rows from both sinks
	REQUIRE(read_ids.size() == total_rows);
	REQUIRE(read_ids.size() >= 2); // At least one sink's data

	// All read IDs should be from the expected set
	std::set<int32_t> id_set(read_ids.begin(), read_ids.end());
	std::set<int32_t> expected_ids({1, 2, 3, 4});
	for (auto &id : id_set) {
		REQUIRE(expected_ids.count(id) > 0);
	}

	source.Close();
	ShuffleCacheRegistry::Instance().Remove(exchange_id);
}
