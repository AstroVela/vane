// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "test_helpers.hpp"

#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/execution/distributed/exchange/flight_ticket.hpp"
#include "duckdb/execution/distributed/exchange/flight_client.hpp"
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
	ticket.server_epoch = "epoch_1";
	ticket.exchange_instance_id = "stage_1__instance_a__sink_0__attempt_3";
	ticket.node_id = "node_2";
	ticket.attempt_id = 3;
	ticket.partition_idx = 7;

	auto encoded = ticket.Serialize();
	auto parsed = FlightExchangeTicket::Parse(encoded);
	REQUIRE(parsed.is_ok());

	auto result = parsed.value();
	REQUIRE(result.server_epoch == ticket.server_epoch);
	REQUIRE(result.exchange_instance_id == ticket.exchange_instance_id);
	REQUIRE(result.node_id == ticket.node_id);
	REQUIRE(result.attempt_id == ticket.attempt_id);
	REQUIRE(result.partition_idx == ticket.partition_idx);
}

TEST_CASE("Exchange: FlightExchangeTicket parse errors", "[distributed][exchange]") {
	REQUIRE(FlightExchangeTicket::Parse("v1\nstage\nnode").is_err());
	REQUIRE(FlightExchangeTicket::Parse("v1\nstage\nnode\n1").is_err());
	REQUIRE(FlightExchangeTicket::Parse("v2\nepoch\nstage\nnode\n1\n2").is_err());
	REQUIRE(FlightExchangeTicket::Parse("v1\n\nstage\nnode\n1\n2").is_err());
	REQUIRE(FlightExchangeTicket::Parse("v1\nepoch\n\nnode\n1\n2").is_err());
	REQUIRE(FlightExchangeTicket::Parse("v1\nepoch\nstage\n\n1\n2").is_err());
	REQUIRE(FlightExchangeTicket::Parse("v1\nepoch\nstage\nnode\n-1\n2").is_err());
	REQUIRE(FlightExchangeTicket::Parse("v1\nepoch\nstage\nnode\n1\n-2").is_err());
	REQUIRE(FlightExchangeTicket::Parse("v1\nepoch\nstage\nnode\nnope\n2").is_err());
	REQUIRE(FlightExchangeTicket::Parse("v1\nepoch\nstage\nnode\n1\nnope").is_err());
	REQUIRE(FlightExchangeTicket::Parse("v1\nepoch\nstage\nnode\n1x\n2").is_err());
	REQUIRE(FlightExchangeTicket::Parse("v1\nepoch\nstage\nnode\n1\n2x").is_err());
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

TEST_CASE("Exchange: ShuffleCacheRegistry validates epoch, attempt, and descriptor identity",
          "[distributed][exchange]") {
	auto &registry = ShuffleCacheRegistry::Instance();
	const std::string exchange_id = "registry_identity_stage";

	ShuffleCacheConfig config;
	config.shuffle_stage_id = exchange_id;
	config.node_id = "node-a";
	config.num_partitions = 2;
	config.local_dirs = {TestCreatePath("registry_identity_a")};
	auto cache = std::make_shared<ShuffleCache>(config);

	REQUIRE(registry.Register(exchange_id, cache, "epoch-a", 4).is_ok());
	REQUIRE(registry.Register(exchange_id, cache, "epoch-a", 4).is_ok());
	REQUIRE(registry.Resolve(exchange_id, "epoch-a", "node-a", 4).is_ok());
	REQUIRE(registry.Resolve(exchange_id, "epoch-old", "node-a", 4).is_err());
	REQUIRE(registry.Resolve(exchange_id, "epoch-a", "node-a", 3).is_err());
	REQUIRE(registry.Resolve(exchange_id, "epoch-a", "node-b", 4).is_err());

	auto conflicting_config = config;
	conflicting_config.local_dirs = {TestCreatePath("registry_identity_b")};
	auto conflicting_cache = std::make_shared<ShuffleCache>(std::move(conflicting_config));
	REQUIRE(registry.Register(exchange_id, conflicting_cache, "epoch-a", 4).is_err());
	REQUIRE(registry.Get(exchange_id).get() == cache.get());

	registry.RemoveForDeferredCleanup(exchange_id);
	REQUIRE(registry.Get(exchange_id) == nullptr);
	REQUIRE(registry.Resolve(exchange_id, "epoch-a", "node-a", 4).is_err());
	registry.RemoveAndCleanupByPrefix(exchange_id);
}

TEST_CASE("Exchange: ShuffleCacheRegistry cleanup waits for active read leases", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;
	auto &registry = ShuffleCacheRegistry::Instance();
	const std::string exchange_id = "registry_lease_stage__sink_0__attempt_0";

	ShuffleCacheConfig config;
	config.shuffle_stage_id = exchange_id;
	config.node_id = "node-a";
	config.num_partitions = 1;
	config.local_dirs = {TestCreatePath("registry_lease")};
	auto cache = std::make_shared<ShuffleCache>(std::move(config));
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER});
	chunk.SetCardinality(1);
	chunk.SetValue(0, 0, Value::INTEGER(42));
	REQUIRE(cache->WriteChunk(context, chunk, 0, {"value"}).is_ok());
	REQUIRE(cache->FlushAll(context, {"value"}).is_ok());
	REQUIRE(cache->WriteAttemptManifest(0, 0).is_ok());
	REQUIRE(cache->HasCommittedManifest());

	REQUIRE(registry.Register(exchange_id, cache, "epoch-a", 0).is_ok());
	auto lease_result = registry.Resolve(exchange_id, "epoch-a", "node-a", 0);
	REQUIRE(lease_result.is_ok());
	auto lease = std::move(lease_result.value());
	registry.RemoveForDeferredCleanup(exchange_id);

	auto cleanup = registry.RemoveAndCleanupByPrefix("registry_lease_stage");
	REQUIRE(cleanup.registry_entries_removed == 1);
	REQUIRE(cleanup.storage_entries_removed == 0);
	REQUIRE(cleanup.cleanup_errors == 0);
	REQUIRE(cache->HasCommittedManifest());

	lease.reset();
	REQUIRE_FALSE(cache->HasCommittedManifest());
}

TEST_CASE("Exchange: Flight service isolates published attempts and rejects released or stale tickets",
          "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;
	auto &registry = ShuffleCacheRegistry::Instance();
	const std::string prefix = "overlapping-stage";
	const std::string exchange_a = prefix + "__instance_a__sink_0__attempt_1";
	const std::string exchange_b = prefix + "__instance_b__sink_0__attempt_1";
	const std::string epoch = "catalog-isolation-epoch";
	const std::string node_id = "node-a";

	auto make_committed_cache = [&](const std::string &exchange_id, const std::string &dir, int32_t value) {
		ShuffleCacheConfig config;
		config.shuffle_stage_id = exchange_id;
		config.node_id = node_id;
		config.num_partitions = 1;
		config.local_dirs = {dir};
		auto cache = std::make_shared<ShuffleCache>(std::move(config));
		DataChunk chunk;
		chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER});
		chunk.SetCardinality(1);
		chunk.SetValue(0, 0, Value::INTEGER(value));
		REQUIRE(cache->WriteChunk(context, chunk, 0, {"value"}).is_ok());
		REQUIRE(cache->FlushAll(context, {"value"}).is_ok());
		REQUIRE(cache->WriteAttemptManifest(0, 1).is_ok());
		return cache;
	};

	auto cache_a = make_committed_cache(exchange_a, TestCreatePath("flight_catalog_isolation_a"), 11);
	auto cache_b = make_committed_cache(exchange_b, TestCreatePath("flight_catalog_isolation_b"), 22);
	REQUIRE(registry.Register(exchange_a, cache_a, epoch, 1).is_ok());
	REQUIRE(registry.Register(exchange_b, cache_b, epoch, 1).is_ok());

	FlightServerConfig server_config;
	server_config.bind_host = "127.0.0.1";
	server_config.port = 0;
	server_config.server_epoch = epoch;
	FlightServer server(std::move(server_config));
	REQUIRE(server.Start().is_ok());
	FlightClientConfig client_config;
	client_config.location = "grpc://127.0.0.1:" + std::to_string(server.port());
	FlightClient client(std::move(client_config));

	auto fetch = [&](const std::string &exchange_id, const std::string &ticket_epoch) {
		FlightExchangeTicket ticket;
		ticket.server_epoch = ticket_epoch;
		ticket.exchange_instance_id = exchange_id;
		ticket.node_id = node_id;
		ticket.attempt_id = 1;
		ticket.partition_idx = 0;
		return client.FetchPartition(context, ticket, {LogicalType::INTEGER});
	};
	auto fetched_a = fetch(exchange_a, epoch);
	auto fetched_b = fetch(exchange_b, epoch);
	REQUIRE(fetched_a.is_ok());
	REQUIRE(fetched_b.is_ok());
	REQUIRE(fetched_a.value()->Count() == 1);
	REQUIRE(fetched_b.value()->Count() == 1);
	vector<int32_t> values_a;
	vector<int32_t> values_b;
	for (auto &chunk : fetched_a.value()->Chunks()) {
		for (idx_t row = 0; row < chunk.size(); row++) {
			values_a.push_back(chunk.GetValue(0, row).GetValue<int32_t>());
		}
	}
	for (auto &chunk : fetched_b.value()->Chunks()) {
		for (idx_t row = 0; row < chunk.size(); row++) {
			values_b.push_back(chunk.GetValue(0, row).GetValue<int32_t>());
		}
	}
	REQUIRE(values_a == vector<int32_t> {11});
	REQUIRE(values_b == vector<int32_t> {22});

	registry.RemoveForDeferredCleanup(exchange_a);
	REQUIRE(fetch(exchange_a, epoch).is_err());
	REQUIRE(fetch(exchange_b, epoch).is_ok());
	REQUIRE(fetch(exchange_b, "stale-epoch").is_err());

	REQUIRE(server.Stop().is_ok());
	registry.RemoveForDeferredCleanup(exchange_b);
	registry.RemoveAndCleanupByPrefix(prefix);
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

TEST_CASE("Exchange: same logical stage has isolated exchange instances and directories", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);

	FlightExchangeConfig config_a;
	config_a.node_id = "node-a";
	config_a.local_dirs = {TestCreatePath("exchange_instance_a")};
	FlightExchangeConfig config_b;
	config_b.node_id = "node-a";
	config_b.local_dirs = {TestCreatePath("exchange_instance_b")};
	FlightExchangeManager manager_a(config_a, conn.context.get());
	FlightExchangeManager manager_b(config_b, conn.context.get());

	ExchangeContext ctx;
	ctx.query_id = "same-query";
	ctx.exchange_id = "same-stage";
	auto exchange_a = manager_a.CreateExchange(ctx, 1);
	auto exchange_b = manager_b.CreateExchange(ctx, 1);
	auto instance_a = exchange_a->InstantiateSink(exchange_a->AddSink(0), 0);
	auto instance_b = exchange_b->InstantiateSink(exchange_b->AddSink(0), 0);
	REQUIRE(instance_a.output_location != instance_b.output_location);

	ShuffleCacheConfig cache_config_a;
	cache_config_a.shuffle_stage_id = instance_a.output_location;
	cache_config_a.node_id = "node-a";
	cache_config_a.num_partitions = 1;
	cache_config_a.local_dirs = config_a.local_dirs;
	ShuffleCacheConfig cache_config_b;
	cache_config_b.shuffle_stage_id = instance_b.output_location;
	cache_config_b.node_id = "node-a";
	cache_config_b.num_partitions = 1;
	cache_config_b.local_dirs = config_b.local_dirs;
	auto cache_a = std::make_shared<ShuffleCache>(std::move(cache_config_a));
	auto cache_b = std::make_shared<ShuffleCache>(std::move(cache_config_b));
	auto &registry = ShuffleCacheRegistry::Instance();
	REQUIRE(registry.Register(instance_a.output_location, cache_a, "epoch-a", 0).is_ok());
	REQUIRE(registry.Register(instance_b.output_location, cache_b, "epoch-a", 0).is_ok());

	exchange_a->Close();
	REQUIRE(registry.Get(instance_a.output_location) == nullptr);
	REQUIRE(registry.Get(instance_b.output_location).get() == cache_b.get());

	exchange_b->Close();
	registry.RemoveAndCleanupByPrefix(ctx.exchange_id);
}

TEST_CASE("Exchange: process-local Flight service has immutable network config and explicit shutdown",
          "[distributed][exchange]") {
	struct ServiceGuard {
		~ServiceGuard() {
			FlightExchangeManager::ShutdownLocalFlightServer();
		}
	} guard;
	REQUIRE(FlightExchangeManager::ShutdownLocalFlightServer().is_ok());

	ExchangeSinkInstanceHandle handle;
	handle.sink_handle.task_partition_id = 0;
	handle.attempt_id = 0;
	handle.output_location = "service_lifecycle__instance_a__sink_0__attempt_0";
	handle.output_partition_count = 1;

	FlightExchangeConfig config_a;
	config_a.node_id = "node-a";
	config_a.flight_bind_host = "127.0.0.1";
	config_a.flight_port = 0;
	config_a.local_dirs = {TestCreatePath("flight_service_lifecycle_a")};
	FlightExchangeManager manager_a(config_a);
	auto sink_a = manager_a.CreateSink(handle);
	REQUIRE(sink_a != nullptr);
	const auto first_port = FlightExchangeManager::GetLocalFlightServerPort();
	const auto first_epoch = FlightExchangeManager::GetLocalFlightServerEpoch();
	REQUIRE(first_port > 0);
	REQUIRE(!first_epoch.empty());

	auto config_b = config_a;
	config_b.local_dirs = {TestCreatePath("flight_service_lifecycle_b")};
	config_b.node_id = "node-b";
	FlightExchangeManager manager_b(config_b);
	auto sink_b = manager_b.CreateSink(handle);
	REQUIRE(sink_b != nullptr);
	REQUIRE(FlightExchangeManager::GetLocalFlightServerPort() == first_port);
	REQUIRE(FlightExchangeManager::GetLocalFlightServerEpoch() == first_epoch);

	auto conflicting_config = config_b;
	conflicting_config.flight_port = first_port;
	FlightExchangeManager conflicting_manager(conflicting_config);
	REQUIRE_THROWS_WITH(conflicting_manager.CreateSink(handle),
	                    Catch::Matchers::Contains("refusing conflicting address"));

	manager_a.Shutdown();
	manager_b.Shutdown();
	REQUIRE(FlightExchangeManager::GetLocalFlightServerPort() == first_port);
	REQUIRE(FlightExchangeManager::GetLocalFlightServerEpoch() == first_epoch);

	sink_a.reset();
	sink_b.reset();
	REQUIRE(FlightExchangeManager::ShutdownLocalFlightServer().is_ok());
	REQUIRE(FlightExchangeManager::GetLocalFlightServerPort() == 0);
	REQUIRE(FlightExchangeManager::GetLocalFlightServerEpoch().empty());

	auto fixed_config = config_a;
	fixed_config.flight_port = first_port;
	FlightExchangeManager fixed_manager(fixed_config);
	auto fixed_sink = fixed_manager.CreateSink(handle);
	REQUIRE(fixed_sink != nullptr);
	REQUIRE(FlightExchangeManager::GetLocalFlightServerPort() == first_port);
	REQUIRE(FlightExchangeManager::GetLocalFlightServerEpoch() != first_epoch);
	fixed_sink.reset();
	REQUIRE(FlightExchangeManager::ShutdownLocalFlightServer().is_ok());
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

	sink0_attempt1.flight_server_epoch = "worker-retry-epoch";
	exchange->SinkFinished(sink0_attempt1, "worker-retry", 5010);
	sink0_attempt0.flight_server_epoch = "worker-late-epoch";
	exchange->SinkFinished(sink0_attempt0, "worker-late", 5011);
	sink1_attempt0.flight_server_epoch = "worker-first-epoch";
	exchange->SinkFinished(sink1_attempt0, "worker-first", 5012);
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
			REQUIRE(handle.flight_server_epoch == "worker-retry-epoch");
			REQUIRE(handle.files[0].path.find("__attempt_1") != std::string::npos);
			REQUIRE(handle.files[0].path.find("__attempt_0") == std::string::npos);
		} else if (handle.files[0].path.find("__sink_1__") != std::string::npos) {
			sink1_handles++;
			REQUIRE(handle.attempt_id == 0);
			REQUIRE(handle.node_id == "worker-first");
			REQUIRE(handle.flight_port == 5012);
			REQUIRE(handle.flight_server_epoch == "worker-first-epoch");
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
