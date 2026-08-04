// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "test_helpers.hpp"

#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/execution/distributed/exchange/flight_ticket.hpp"
#include "duckdb/execution/distributed/exchange/shuffle_cache.hpp"
#include "duckdb/execution/distributed/exchange/shuffle_cache_registry.hpp"
#include "duckdb/execution/distributed/exchange/flight_exchange_manager.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"

#include "arrow/api.h"
#include "arrow/flight/api.h"
#include "arrow/io/api.h"
#include "arrow/ipc/api.h"

#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <set>
#include <fstream>
#include <future>
#include <iterator>
#include <limits>
#include <utility>
#include <cstdlib>
#include <condition_variable>
#include <thread>

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

class ScopedShuffleCacheRegistration {
public:
	explicit ScopedShuffleCacheRegistration(string exchange_id) : exchange_id_(std::move(exchange_id)) {
	}

	~ScopedShuffleCacheRegistration() {
		ShuffleCacheRegistry::Instance().Remove(exchange_id_);
	}

	ScopedShuffleCacheRegistration(const ScopedShuffleCacheRegistration &) = delete;
	ScopedShuffleCacheRegistration &operator=(const ScopedShuffleCacheRegistration &) = delete;

private:
	string exchange_id_;
};

class BlockingFlightState {
public:
	void MarkStarted() {
		std::lock_guard<std::mutex> guard(mutex_);
		started_ = true;
		condition_.notify_all();
	}

	bool WaitUntilStarted(std::chrono::milliseconds timeout) {
		std::unique_lock<std::mutex> guard(mutex_);
		return condition_.wait_for(guard, timeout, [&]() { return started_; });
	}

	void WaitUntilReleased() {
		std::unique_lock<std::mutex> guard(mutex_);
		condition_.wait(guard, [&]() { return released_; });
	}

	bool WaitUntilReleased(std::chrono::milliseconds timeout) {
		std::unique_lock<std::mutex> guard(mutex_);
		return condition_.wait_for(guard, timeout, [&]() { return released_; });
	}

	void Release() {
		std::lock_guard<std::mutex> guard(mutex_);
		released_ = true;
		condition_.notify_all();
	}

private:
	std::mutex mutex_;
	std::condition_variable condition_;
	bool started_ = false;
	bool released_ = false;
};

class BlockingRecordBatchReader final : public arrow::RecordBatchReader {
public:
	explicit BlockingRecordBatchReader(std::shared_ptr<BlockingFlightState> state)
	    : state_(std::move(state)), schema_(arrow::schema({arrow::field("value", arrow::int32())})) {
	}

	std::shared_ptr<arrow::Schema> schema() const override {
		return schema_;
	}

	arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *batch) override {
		state_->MarkStarted();
		state_->WaitUntilReleased();
		*batch = nullptr;
		return arrow::Status::OK();
	}

	arrow::Status Close() override {
		state_->Release();
		return arrow::Status::OK();
	}

private:
	std::shared_ptr<BlockingFlightState> state_;
	std::shared_ptr<arrow::Schema> schema_;
};

class BlockingReadFlightServer final : public arrow::flight::FlightServerBase {
public:
	explicit BlockingReadFlightServer(std::shared_ptr<BlockingFlightState> state) : state_(std::move(state)) {
	}

	arrow::Status DoGet(const arrow::flight::ServerCallContext &, const arrow::flight::Ticket &,
	                    std::unique_ptr<arrow::flight::FlightDataStream> *stream) override {
		auto reader = std::make_shared<BlockingRecordBatchReader>(state_);
		*stream = std::unique_ptr<arrow::flight::RecordBatchStream>(new arrow::flight::RecordBatchStream(reader));
		return arrow::Status::OK();
	}

private:
	std::shared_ptr<BlockingFlightState> state_;
};

class BlockingDoGetFlightServer final : public arrow::flight::FlightServerBase {
public:
	explicit BlockingDoGetFlightServer(std::shared_ptr<BlockingFlightState> state) : state_(std::move(state)) {
	}

	arrow::Status DoGet(const arrow::flight::ServerCallContext &context, const arrow::flight::Ticket &,
	                    std::unique_ptr<arrow::flight::FlightDataStream> *) override {
		state_->MarkStarted();
		while (!context.is_cancelled()) {
			if (state_->WaitUntilReleased(std::chrono::milliseconds(5))) {
				return arrow::Status::Cancelled("blocking DoGet test released");
			}
		}
		return arrow::Status::Cancelled("blocking DoGet test canceled");
	}

private:
	std::shared_ptr<BlockingFlightState> state_;
};

void StartTestFlightServer(arrow::flight::FlightServerBase &server) {
	auto location = arrow::flight::Location::ForGrpcTcp("127.0.0.1", 0);
	REQUIRE(location.ok());
	arrow::flight::FlightServerOptions options(std::move(location).ValueOrDie());
	REQUIRE(server.Init(options).ok());
}

using MaterializedRows = vector<vector<Value>>;

ExchangeSourceHandle MakeSourceHandle(const string &output_location, const string &node_id, idx_t partition_id,
                                      idx_t attempt_id = 0) {
	ExchangeSourceHandle handle;
	handle.partition_id = partition_id;
	handle.attempt_id = attempt_id;
	handle.node_id = node_id;
	handle.files.push_back(ExchangeSourceFile(output_location, 0));
	return handle;
}

ExchangeSourceHandle MakeRemoteSourceHandle(int port) {
	auto handle = MakeSourceHandle("blocking-flight-test", "producer-node", 0);
	handle.flight_host = "127.0.0.1";
	handle.flight_port = port;
	handle.flight_server_epoch = "blocking-flight-epoch";
	return handle;
}

MaterializedRows ReadSourceRows(ClientContext &context, FlightExchangeConfig config,
                                vector<ExchangeSourceHandle> handles) {
	if (config.expected_types.empty()) {
		throw std::runtime_error("test source requires expected types");
	}
	FlightExchangeSource source(config, &context);
	source.AddSourceHandles(std::move(handles));

	MaterializedRows rows;
	vector<LogicalType> output_types(config.expected_types.begin(), config.expected_types.end());
	DataChunk output;
	output.Initialize(Allocator::DefaultAllocator(), output_types);
	while (source.ReadChunk(output)) {
		for (idx_t row_idx = 0; row_idx < output.size(); row_idx++) {
			vector<Value> row;
			row.reserve(output.ColumnCount());
			for (idx_t col_idx = 0; col_idx < output.ColumnCount(); col_idx++) {
				row.push_back(output.GetValue(col_idx, row_idx));
			}
			rows.push_back(std::move(row));
		}
	}
	source.Close();
	return rows;
}

void RequireTwoColumnRows(const MaterializedRows &rows, const vector<int32_t> &ids, const vector<string> &names) {
	REQUIRE(rows.size() == ids.size());
	REQUIRE(rows.size() == names.size());
	for (idx_t row_idx = 0; row_idx < rows.size(); row_idx++) {
		REQUIRE(rows[row_idx].size() == 2);
		REQUIRE(rows[row_idx][0].GetValue<int32_t>() == ids[row_idx]);
		REQUIRE(rows[row_idx][1].GetValue<string>() == names[row_idx]);
	}
}

class MockObjectShuffleStorage final : public ShuffleStorage {
public:
	explicit MockObjectShuffleStorage(std::string root, idx_t remove_failures = 0)
	    : root_(std::move(root)), fs_(FileSystem::CreateLocal()), remove_failures_remaining_(remove_failures) {
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
		if (remove_failures_remaining_ > 0) {
			remove_failures_remaining_--;
			return DuckDBResult<idx_t>::err(DuckDBError::io_error("injected mock object cleanup failure"));
		}
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
	mutable idx_t remove_failures_remaining_ = 0;
};

} // namespace

// ═══════════════════════════════════════════════════════════
// FlightExchangeTicket
// ═══════════════════════════════════════════════════════════

TEST_CASE("Exchange: FlightExchangeTicket roundtrip", "[distributed][exchange]") {
	FlightExchangeTicket ticket;
	ticket.server_epoch = "epoch_1";
	ticket.exchange_instance_id = "55f8c578-9c57-4b9d-bdc2-ef62d1dfc323__sink_0__attempt_3";
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

TEST_CASE("Exchange: Flight timeouts resolve from the worker environment", "[distributed][exchange]") {
	SECTION("configured values") {
		ScopedEnvVar call_timeout("VANE_FLIGHT_CALL_TIMEOUT_S", "12.5");
		ScopedEnvVar idle_timeout("VANE_FLIGHT_READ_IDLE_TIMEOUT_S", "3.25");
		auto config = ResolveFlightExchangeConfigFromEnv();
		REQUIRE(config.flight_timeout_seconds == Approx(12.5));
		REQUIRE(config.flight_read_idle_timeout_seconds == Approx(3.25));
	}

	SECTION("zero explicitly disables a timeout") {
		ScopedEnvVar call_timeout("VANE_FLIGHT_CALL_TIMEOUT_S", "0");
		ScopedEnvVar idle_timeout("VANE_FLIGHT_READ_IDLE_TIMEOUT_S", "0");
		auto config = ResolveFlightExchangeConfigFromEnv();
		REQUIRE(config.flight_timeout_seconds == 0.0);
		REQUIRE(config.flight_read_idle_timeout_seconds == 0.0);
	}

	SECTION("invalid values retain safe defaults") {
		ScopedEnvVar call_timeout("VANE_FLIGHT_CALL_TIMEOUT_S", "not-a-timeout");
		ScopedEnvVar idle_timeout("VANE_FLIGHT_READ_IDLE_TIMEOUT_S", "-1");
		auto config = ResolveFlightExchangeConfigFromEnv();
		REQUIRE(config.flight_timeout_seconds == FlightExchangeConfig::DEFAULT_FLIGHT_TIMEOUT_SECONDS);
		REQUIRE(config.flight_read_idle_timeout_seconds ==
		        FlightExchangeConfig::DEFAULT_FLIGHT_READ_IDLE_TIMEOUT_SECONDS);
	}
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
	REQUIRE(registry.Register("registry_test_stage", cache, "registry-test-query").is_ok());

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

	REQUIRE(registry.Register("multi_test_1", cache1, "registry-multi-query").is_ok());
	REQUIRE(registry.Register("multi_test_2", cache2, "registry-multi-query").is_ok());

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
	const std::string query_id = "registry-identity-query";

	ShuffleCacheConfig config;
	config.shuffle_stage_id = exchange_id;
	config.node_id = "node-a";
	config.num_partitions = 2;
	config.local_dirs = {TestCreatePath("registry_identity_a")};
	auto cache = std::make_shared<ShuffleCache>(config);

	REQUIRE(registry.Register(exchange_id, cache, query_id, "epoch-a", 4).is_ok());
	REQUIRE(registry.Register(exchange_id, cache, query_id, "epoch-a", 4).is_ok());
	REQUIRE(registry.Resolve(exchange_id, "epoch-a", "node-a", 4).is_ok());
	REQUIRE(registry.Resolve(exchange_id, "epoch-old", "node-a", 4).is_err());
	REQUIRE(registry.Resolve(exchange_id, "epoch-a", "node-a", 3).is_err());
	REQUIRE(registry.Resolve(exchange_id, "epoch-a", "node-b", 4).is_err());

	auto conflicting_config = config;
	conflicting_config.local_dirs = {TestCreatePath("registry_identity_b")};
	auto conflicting_cache = std::make_shared<ShuffleCache>(std::move(conflicting_config));
	REQUIRE(registry.Register(exchange_id, conflicting_cache, query_id, "epoch-a", 4).is_err());
	REQUIRE(registry.Get(exchange_id).get() == cache.get());

	auto mismatched_config = config;
	mismatched_config.shuffle_stage_id = "different-exchange";
	auto mismatched_cache = std::make_shared<ShuffleCache>(std::move(mismatched_config));
	REQUIRE(registry.TrackPending(exchange_id, mismatched_cache, query_id, "epoch-a", 4).is_err());
	REQUIRE(registry.Register(exchange_id, mismatched_cache, query_id, "epoch-a", 4).is_err());

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
	const std::string query_id = "registry-lease-query";

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

	REQUIRE(registry.Register(exchange_id, cache, query_id, "epoch-a", 0).is_ok());
	auto lease_result = registry.Resolve(exchange_id, "epoch-a", "node-a", 0);
	REQUIRE(lease_result.is_ok());
	auto lease = std::move(lease_result.value());
	registry.RemoveForDeferredCleanup(exchange_id);

	auto cleanup = registry.RemoveAndCleanupByPrefix("registry_lease_stage");
	REQUIRE(cleanup.registry_entries_removed == 0);
	REQUIRE(cleanup.storage_entries_removed == 0);
	REQUIRE(cleanup.cleanup_errors == 0);
	REQUIRE(cleanup.cleanup_pending == 1);
	REQUIRE(cache->HasCommittedManifest());

	lease.reset();
	REQUIRE(cache->HasCommittedManifest());
	cleanup = registry.RemoveAndCleanupByPrefix("registry_lease_stage");
	REQUIRE(cleanup.cleanup_pending == 0);
	REQUIRE(cleanup.cleanup_errors == 0);
	REQUIRE(cleanup.storage_entries_removed > 0);
	REQUIRE_FALSE(cache->HasCommittedManifest());
}

TEST_CASE("Exchange: ShuffleCacheRegistry retains and retries failed cleanup", "[distributed][exchange]") {
	auto &registry = ShuffleCacheRegistry::Instance();
	const std::string query_id = "registry-cleanup-retry-query";
	const std::string exchange_id = "registry_cleanup_retry__sink_0__attempt_0";

	ShuffleCacheConfig config;
	config.shuffle_stage_id = exchange_id;
	config.node_id = "node-a";
	config.num_partitions = 1;
	config.local_dirs = {"mock://shuffle"};
	auto storage =
	    std::make_shared<MockObjectShuffleStorage>(TestCreatePath("registry_cleanup_retry"), /*remove_failures=*/1);
	auto cache = std::make_shared<ShuffleCache>(std::move(config), std::move(storage));

	REQUIRE(registry.Register(exchange_id, cache, query_id, "epoch-a", 0).is_ok());
	registry.RemoveForDeferredCleanup(exchange_id);

	auto first_cleanup = registry.RemoveAndCleanupByQuery(query_id);
	REQUIRE(first_cleanup.cleanup_errors == 1);
	REQUIRE(first_cleanup.cleanup_pending == 1);
	REQUIRE(first_cleanup.last_error.find("injected mock object cleanup failure") != std::string::npos);

	auto retry_cleanup = registry.RemoveAndCleanupByQuery(query_id);
	REQUIRE(retry_cleanup.cleanup_errors == 0);
	REQUIRE(retry_cleanup.cleanup_pending == 0);
	REQUIRE(retry_cleanup.last_error.empty());
	REQUIRE(registry.RetireQuery(query_id).is_ok());
}

TEST_CASE("Exchange: object shuffle cleanup reconstructs storage without retaining task cache",
          "[distributed][exchange]") {
	auto &registry = ShuffleCacheRegistry::Instance();
	const std::string query_id = "registry-object-cleanup-context-query";
	const std::string exchange_id = "registry_object_cleanup_context__sink_0__attempt_0";
	auto storage_root = TestCreatePath("registry_object_cleanup_context");

	ShuffleCacheConfig config;
	config.shuffle_stage_id = exchange_id;
	config.node_id = "node-a";
	config.num_partitions = 1;
	config.local_dirs = {"mock://shuffle"};
	auto task_storage = std::make_shared<MockObjectShuffleStorage>(storage_root);
	auto cache = std::make_shared<ShuffleCache>(config, task_storage);
	REQUIRE(cache->WriteAttemptManifest(0, 0).is_ok());
	REQUIRE(cache->HasCommittedManifest());

	std::weak_ptr<ShuffleStorage> task_storage_lifetime = task_storage;
	std::weak_ptr<ShuffleCache> task_cache_lifetime = cache;
	auto lease_result = registry.TrackPending(exchange_id, cache, query_id, "epoch-a", 0,
	                                          ShuffleCacheRegistry::CacheRetention::DESCRIPTOR_ONLY);
	REQUIRE(lease_result.is_ok());
	auto writer_lease = std::move(lease_result.value());
	REQUIRE(registry.Publish(exchange_id, cache, query_id, "epoch-a", 0).is_ok());
	REQUIRE(registry.Get(exchange_id) == nullptr);
	REQUIRE(registry.Resolve(exchange_id, "epoch-a", "node-a", 0).is_err());

	writer_lease.reset();
	cache.reset();
	task_storage.reset();
	REQUIRE(task_cache_lifetime.expired());
	REQUIRE(task_storage_lifetime.expired());

	auto without_context = registry.RemoveAndCleanupByQuery(query_id);
	REQUIRE(without_context.cleanup_errors == 1);
	REQUIRE(without_context.cleanup_storage_required == 1);
	REQUIRE(without_context.cleanup_pending == 1);
	REQUIRE(without_context.last_error.find("requires a live filesystem context") != std::string::npos);

	auto cleanup_storage = std::make_shared<MockObjectShuffleStorage>(storage_root);
	auto with_context = registry.RemoveAndCleanupByQuery(query_id, cleanup_storage);
	REQUIRE(with_context.cleanup_errors == 0);
	REQUIRE(with_context.cleanup_storage_required == 0);
	REQUIRE(with_context.cleanup_pending == 0);
	REQUIRE(with_context.storage_entries_removed > 0);
	ShuffleCache cleanup_probe(config, std::move(cleanup_storage));
	REQUIRE_FALSE(cleanup_probe.HasCommittedManifest());
	REQUIRE(registry.RetireQuery(query_id).is_ok());
}

TEST_CASE("Exchange: deferred cleanup retains exclusive attempt identity", "[distributed][exchange]") {
	auto &registry = ShuffleCacheRegistry::Instance();
	const std::string exchange_id = "registry_deferred_identity__sink_0__attempt_0";

	ShuffleCacheConfig config;
	config.shuffle_stage_id = exchange_id;
	config.node_id = "node-a";
	config.num_partitions = 1;
	config.local_dirs = {TestCreatePath("registry_deferred_identity")};
	auto original_cache = std::make_shared<ShuffleCache>(config);
	REQUIRE(registry.Register(exchange_id, original_cache, "registry-deferred-owner", "epoch-a", 0).is_ok());
	registry.RemoveForDeferredCleanup(exchange_id);

	auto replacement_cache = std::make_shared<ShuffleCache>(config);
	REQUIRE(registry.TrackPending(exchange_id, replacement_cache, "registry-replacement-owner", "epoch-b", 0).is_err());
	REQUIRE(registry.Register(exchange_id, replacement_cache, "registry-replacement-owner", "epoch-b", 0).is_err());

	auto cleanup = registry.RemoveAndCleanupByPrefix(exchange_id);
	REQUIRE(cleanup.cleanup_errors == 0);
	REQUIRE(cleanup.cleanup_pending == 0);
}

TEST_CASE("Exchange: unpublished sink attempts stay hidden and retain cleanup ownership", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;
	auto &registry = ShuffleCacheRegistry::Instance();
	const std::string query_id = "registry-pending-writer-query";
	const std::string exchange_id = "registry_pending_writer__sink_0__attempt_0";

	ShuffleCacheConfig config;
	config.shuffle_stage_id = exchange_id;
	config.node_id = "node-a";
	config.num_partitions = 1;
	config.local_dirs = {TestCreatePath("registry_pending_writer")};
	auto cache = std::make_shared<ShuffleCache>(std::move(config));

	auto lease_result = registry.TrackPending(exchange_id, cache, query_id, "epoch-a", 0);
	REQUIRE(lease_result.is_ok());
	auto writer_lease = std::move(lease_result.value());
	REQUIRE(registry.Get(exchange_id) == nullptr);
	REQUIRE(registry.Resolve(exchange_id, "epoch-a", "node-a", 0).is_err());

	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER});
	chunk.SetCardinality(1);
	chunk.SetValue(0, 0, Value::INTEGER(42));
	REQUIRE(cache->WriteChunk(context, chunk, 0, {"value"}).is_ok());
	REQUIRE(cache->FlushAll(context, {"value"}).is_ok());
	auto files = cache->GetPartitionFiles(0);
	REQUIRE(files.is_ok());
	REQUIRE(files.value().files.size() == 1);
	auto fs = FileSystem::CreateLocal();
	REQUIRE(fs->FileExists(files.value().files[0].path));

	REQUIRE(registry.CloseQuery(query_id).is_ok());
	auto while_writing = registry.RemoveAndCleanupByQuery(query_id);
	REQUIRE(while_writing.cleanup_pending == 1);
	REQUIRE(while_writing.storage_entries_removed == 0);
	REQUIRE(fs->FileExists(files.value().files[0].path));

	writer_lease.reset();
	auto after_writer = registry.RemoveAndCleanupByQuery(query_id);
	REQUIRE(after_writer.cleanup_pending == 0);
	REQUIRE(after_writer.cleanup_errors == 0);
	REQUIRE(after_writer.storage_entries_removed > 0);
	REQUIRE_FALSE(fs->FileExists(files.value().files[0].path));
	REQUIRE(registry.RetireQuery(query_id).is_ok());

	// Retirement bounds tombstone growth and permits an explicitly new
	// generation to reuse the same identity after every old job was joined.
	REQUIRE(registry.BeginQueryExecution(query_id).is_ok());
	REQUIRE(registry.EndQueryExecution(query_id).is_ok());
}

TEST_CASE("Exchange: query close fences late cache publication until native execution drains",
          "[distributed][exchange]") {
	auto &registry = ShuffleCacheRegistry::Instance();
	const std::string query_id = "registry-close-race-query";
	const std::string exchange_id = "registry_close_race__sink_0__attempt_0";

	ShuffleCacheConfig config;
	config.shuffle_stage_id = exchange_id;
	config.node_id = "node-a";
	config.num_partitions = 1;
	config.local_dirs = {TestCreatePath("registry_close_race")};
	auto cache = std::make_shared<ShuffleCache>(std::move(config));

	REQUIRE(registry.BeginQueryExecution(query_id).is_ok());
	REQUIRE(registry.CloseQuery(query_id).is_ok());
	std::mutex publish_mutex;
	std::condition_variable publish_ready;
	bool may_publish = false;
	bool publication_rejected = false;
	bool execution_released = false;
	std::thread publisher([&]() {
		{
			std::unique_lock<std::mutex> lock(publish_mutex);
			publish_ready.wait(lock, [&]() { return may_publish; });
		}
		publication_rejected = registry.Register(exchange_id, cache, query_id, "epoch-a", 0).is_err();
		execution_released = registry.EndQueryExecution(query_id).is_ok();
	});

	auto while_active = registry.RemoveAndCleanupByQuery(query_id);
	{
		std::lock_guard<std::mutex> lock(publish_mutex);
		may_publish = true;
	}
	publish_ready.notify_one();
	publisher.join();

	REQUIRE(while_active.active_executions == 1);
	REQUIRE(publication_rejected);
	REQUIRE(execution_released);
	auto after_drain = registry.RemoveAndCleanupByQuery(query_id);
	REQUIRE(after_drain.active_executions == 0);
	REQUIRE(after_drain.cleanup_pending == 0);
	REQUIRE(after_drain.cleanup_errors == 0);
	REQUIRE(registry.Get(exchange_id) == nullptr);
	REQUIRE(registry.RetireQuery(query_id).is_ok());
}

TEST_CASE("Exchange: late pending sink cannot delete a borrowed closed-query attempt", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;
	auto &registry = ShuffleCacheRegistry::Instance();
	const std::string query_id = "registry-late-pending-query";
	const std::string exchange_id = "registry_late_pending__sink_0__attempt_0";

	ShuffleCacheConfig config;
	config.shuffle_stage_id = exchange_id;
	config.node_id = "node-a";
	config.num_partitions = 1;
	config.local_dirs = {TestCreatePath("registry_late_pending")};
	auto published_cache = std::make_shared<ShuffleCache>(config);
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER});
	chunk.SetCardinality(1);
	chunk.SetValue(0, 0, Value::INTEGER(42));
	REQUIRE(published_cache->WriteChunk(context, chunk, 0, {"value"}).is_ok());
	REQUIRE(published_cache->FlushAll(context, {"value"}).is_ok());
	REQUIRE(published_cache->WriteAttemptManifest(0, 0).is_ok());
	auto files = published_cache->GetPartitionFiles(0);
	REQUIRE(files.is_ok());
	REQUIRE(files.value().files.size() == 1);
	auto fs = FileSystem::CreateLocal();
	REQUIRE(fs->FileExists(files.value().files[0].path));

	REQUIRE(registry.Register(exchange_id, published_cache, query_id, "epoch-a", 0).is_ok());
	auto borrowed_result = registry.Resolve(exchange_id, "epoch-a", "node-a", 0);
	REQUIRE(borrowed_result.is_ok());
	auto borrowed = std::move(borrowed_result.value());
	REQUIRE(registry.CloseQuery(query_id).is_ok());

	auto late_cache = std::make_shared<ShuffleCache>(config);
	REQUIRE(registry.TrackPending(exchange_id, late_cache, query_id, "epoch-a", 0).is_err());
	REQUIRE(fs->FileExists(files.value().files[0].path));
	auto late_registered_cache = std::make_shared<ShuffleCache>(config);
	REQUIRE(registry.Register(exchange_id, late_registered_cache, query_id, "epoch-a", 0).is_err());
	REQUIRE(fs->FileExists(files.value().files[0].path));
	auto while_borrowed = registry.RemoveAndCleanupByQuery(query_id);
	REQUIRE(while_borrowed.cleanup_pending == 1);
	REQUIRE(while_borrowed.storage_entries_removed == 0);
	REQUIRE(fs->FileExists(files.value().files[0].path));

	borrowed.reset();
	auto after_release = registry.RemoveAndCleanupByQuery(query_id);
	REQUIRE(after_release.cleanup_pending == 0);
	REQUIRE(after_release.cleanup_errors == 0);
	REQUIRE(after_release.storage_entries_removed > 0);
	REQUIRE_FALSE(fs->FileExists(files.value().files[0].path));
	REQUIRE(registry.RetireQuery(query_id).is_ok());
}

TEST_CASE("Exchange: query cleanup waits for native execution before deleting committed storage",
          "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;
	auto &registry = ShuffleCacheRegistry::Instance();
	const std::string query_id = "registry-native-cleanup-fence-query";
	const std::string exchange_id = "registry_native_cleanup_fence__sink_0__attempt_0";

	ShuffleCacheConfig config;
	config.shuffle_stage_id = exchange_id;
	config.node_id = "node-a";
	config.num_partitions = 1;
	config.local_dirs = {TestCreatePath("registry_native_cleanup_fence")};
	auto cache = std::make_shared<ShuffleCache>(std::move(config));
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER});
	chunk.SetCardinality(1);
	chunk.SetValue(0, 0, Value::INTEGER(42));
	REQUIRE(cache->WriteChunk(context, chunk, 0, {"value"}).is_ok());
	REQUIRE(cache->FlushAll(context, {"value"}).is_ok());
	REQUIRE(cache->WriteAttemptManifest(0, 0).is_ok());
	REQUIRE(cache->HasCommittedManifest());

	REQUIRE(registry.BeginQueryExecution(query_id).is_ok());
	REQUIRE(registry.Register(exchange_id, cache, query_id, "epoch-a", 0).is_ok());
	auto while_active = registry.RemoveAndCleanupByQuery(query_id);
	REQUIRE(while_active.registry_entries_removed == 1);
	REQUIRE(while_active.active_executions == 1);
	REQUIRE(while_active.cleanup_pending == 1);
	REQUIRE(while_active.storage_entries_removed == 0);
	REQUIRE(cache->HasCommittedManifest());

	REQUIRE(registry.EndQueryExecution(query_id).is_ok());
	auto after_drain = registry.RemoveAndCleanupByQuery(query_id);
	REQUIRE(after_drain.active_executions == 0);
	REQUIRE(after_drain.cleanup_pending == 0);
	REQUIRE(after_drain.cleanup_errors == 0);
	REQUIRE(after_drain.storage_entries_removed > 0);
	REQUIRE_FALSE(cache->HasCommittedManifest());
	REQUIRE(registry.RetireQuery(query_id).is_ok());
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
	const std::string query_id = "catalog-isolation-query";

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
	REQUIRE(registry.Register(exchange_a, cache_a, query_id, epoch, 1).is_ok());
	REQUIRE(registry.Register(exchange_b, cache_b, query_id, epoch, 1).is_ok());

	FlightServerConfig server_config;
	server_config.bind_host = "127.0.0.1";
	server_config.port = 0;
	server_config.server_epoch = epoch;
	FlightServer server(std::move(server_config));
	REQUIRE(server.Start().is_ok());
	FlightExchangeConfig source_config;
	source_config.local_dirs = {TestCreatePath("flight_catalog_isolation_reader")};
	source_config.node_id = "reader-node";
	source_config.expected_types = {LogicalType::INTEGER};

	auto fetch = [&](const std::string &exchange_id, const std::string &ticket_epoch) {
		auto handle = MakeSourceHandle(exchange_id, node_id, 0, 1);
		handle.flight_host = "127.0.0.1";
		handle.flight_port = server.port();
		handle.flight_server_epoch = ticket_epoch;
		try {
			auto rows = ReadSourceRows(context, source_config, {std::move(handle)});
			vector<int32_t> values;
			for (const auto &row : rows) {
				values.push_back(row[0].GetValue<int32_t>());
			}
			return std::make_pair(std::move(values), string());
		} catch (const std::exception &ex) {
			return std::make_pair(vector<int32_t>(), string(ex.what()));
		}
	};
	auto fetched_a = fetch(exchange_a, epoch);
	auto fetched_b = fetch(exchange_b, epoch);
	REQUIRE(fetched_a.second.empty());
	REQUIRE(fetched_b.second.empty());
	REQUIRE(fetched_a.first == vector<int32_t> {11});
	REQUIRE(fetched_b.first == vector<int32_t> {22});

	registry.RemoveForDeferredCleanup(exchange_a);
	REQUIRE_FALSE(fetch(exchange_a, epoch).second.empty());
	REQUIRE(fetch(exchange_b, epoch).second.empty());
	REQUIRE_FALSE(fetch(exchange_b, "stale-epoch").second.empty());

	REQUIRE(server.Stop().is_ok());
	registry.RemoveForDeferredCleanup(exchange_b);
	registry.RemoveAndCleanupByPrefix(prefix);
}

TEST_CASE("Exchange: process-local Flight shutdown is bounded and releases its service lock",
          "[distributed][exchange]") {
	struct ServiceGuard {
		~ServiceGuard() {
			FlightExchangeManager::ShutdownLocalFlightServer();
		}
	} guard;
	REQUIRE(FlightExchangeManager::ShutdownLocalFlightServer().is_ok());

	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;
	auto &registry = ShuffleCacheRegistry::Instance();
	const std::string exchange_id = "stalled-flight-shutdown__sink_0__attempt_0";
	const std::string query_id = "stalled-flight-shutdown-query";
	const std::string node_id = "stalled-flight-shutdown-node";

	ShuffleCacheConfig cache_config;
	cache_config.shuffle_stage_id = exchange_id;
	cache_config.node_id = node_id;
	cache_config.num_partitions = 1;
	cache_config.local_dirs = {TestCreatePath("stalled_flight_shutdown")};
	auto cache = std::make_shared<ShuffleCache>(std::move(cache_config));
	const std::string payload(256 * 1024, 'x');
	for (idx_t batch = 0; batch < 64; batch++) {
		DataChunk chunk;
		chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::BLOB});
		chunk.SetCardinality(1);
		chunk.SetValue(0, 0, Value::BLOB_RAW(payload));
		REQUIRE(cache->WriteChunk(context, chunk, 0, {"payload"}).is_ok());
	}
	REQUIRE(cache->FlushAll(context, {"payload"}).is_ok());
	REQUIRE(cache->WriteAttemptManifest(0, 0).is_ok());

	FlightServiceConfig service_config;
	service_config.bind_host = "127.0.0.1";
	service_config.advertise_host = "127.0.0.1";
	service_config.port = 0;
	service_config.shutdown_grace_period = std::chrono::seconds(1);
	REQUIRE(FlightExchangeManager::EnsureLocalFlightServerStarted(service_config).is_ok());
	const auto port = FlightExchangeManager::GetLocalFlightServerPort();
	const auto epoch = FlightExchangeManager::GetLocalFlightServerEpoch();
	REQUIRE(port > 0);
	REQUIRE(!epoch.empty());
	REQUIRE(registry.Register(exchange_id, cache, query_id, epoch, 0).is_ok());

	auto location = arrow::flight::Location::Parse("grpc://127.0.0.1:" + std::to_string(port));
	REQUIRE(location.ok());
	auto client_result = arrow::flight::FlightClient::Connect(std::move(location).ValueOrDie());
	REQUIRE(client_result.ok());
	auto client = std::move(client_result).ValueOrDie();
	FlightExchangeTicket ticket;
	ticket.server_epoch = epoch;
	ticket.exchange_instance_id = exchange_id;
	ticket.node_id = node_id;
	ticket.attempt_id = 0;
	ticket.partition_idx = 0;
	arrow::flight::Ticket flight_ticket;
	flight_ticket.ticket = ticket.Serialize();
	auto reader_result = client->DoGet(flight_ticket);
	REQUIRE(reader_result.ok());
	auto reader = std::move(reader_result).ValueOrDie();
	REQUIRE(reader->GetSchema().ok());

	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	auto stop_future =
	    std::async(std::launch::async, []() { return FlightExchangeManager::ShutdownLocalFlightServer(); });
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	auto port_future =
	    std::async(std::launch::async, []() { return FlightExchangeManager::GetLocalFlightServerPort(); });
	const bool port_lookup_ready = port_future.wait_for(std::chrono::milliseconds(250)) == std::future_status::ready;
	auto start_future = std::async(
	    std::launch::async, [&]() { return FlightExchangeManager::EnsureLocalFlightServerStarted(service_config); });
	const bool concurrent_start_ready =
	    start_future.wait_for(std::chrono::milliseconds(250)) == std::future_status::ready;
	const bool stopped_while_reader_open = stop_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
	if (!stopped_while_reader_open) {
		reader->Cancel();
	}
	auto stop_result = stop_future.get();
	const auto observed_port = port_future.get();
	auto concurrent_start_result = start_future.get();
	reader->Cancel();
	if (concurrent_start_result.is_ok()) {
		REQUIRE(FlightExchangeManager::ShutdownLocalFlightServer().is_ok());
	}

	REQUIRE(stop_result.is_ok());
	REQUIRE(port_lookup_ready);
	REQUIRE(observed_port == 0);
	REQUIRE(concurrent_start_ready);
	REQUIRE(concurrent_start_result.is_err());
	REQUIRE_THAT(concurrent_start_result.error().what(), Catch::Matchers::Contains("shutting down"));
	REQUIRE(stopped_while_reader_open);
	registry.RemoveAndCleanupByQuery(query_id);
	REQUIRE(registry.RetireQuery(query_id).is_ok());
}

TEST_CASE("Exchange: Flight source idle watchdog cancels a stalled batch read", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto state = std::make_shared<BlockingFlightState>();
	BlockingReadFlightServer server(state);
	StartTestFlightServer(server);

	FlightExchangeConfig config;
	config.local_dirs = {TestCreatePath("flight_idle_watchdog")};
	config.node_id = "reader-node";
	config.expected_types = {LogicalType::INTEGER};
	config.flight_timeout_seconds = 5.0;
	config.flight_read_idle_timeout_seconds = 0.5;
	auto handle = MakeRemoteSourceHandle(server.port());
	auto read_future = std::async(std::launch::async, [&]() {
		try {
			ReadSourceRows(*conn.context, config, {std::move(handle)});
			return string();
		} catch (const std::exception &ex) {
			return string(ex.what());
		}
	});

	const bool read_started = state->WaitUntilStarted(std::chrono::seconds(2));
	const bool read_stopped =
	    read_started && read_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
	state->Release();
	read_future.wait();
	auto read_error = read_future.get();
	auto shutdown_status = server.Shutdown();

	REQUIRE(read_started);
	REQUIRE(read_stopped);
	REQUIRE_THAT(read_error, Catch::Matchers::Contains("flight read batch idle timeout"));
	REQUIRE(shutdown_status.ok());
}

TEST_CASE("Exchange: Flight source idle watchdog cancels a stalled initial schema", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto state = std::make_shared<BlockingFlightState>();
	BlockingDoGetFlightServer server(state);
	StartTestFlightServer(server);

	FlightExchangeConfig config;
	config.local_dirs = {TestCreatePath("flight_schema_idle_watchdog")};
	config.node_id = "reader-node";
	config.expected_types = {LogicalType::INTEGER};
	config.flight_timeout_seconds = 5.0;
	config.flight_read_idle_timeout_seconds = 0.5;
	auto handle = MakeRemoteSourceHandle(server.port());
	auto read_future = std::async(std::launch::async, [&]() {
		try {
			ReadSourceRows(*conn.context, config, {std::move(handle)});
			return string();
		} catch (const std::exception &ex) {
			return string(ex.what());
		}
	});

	const bool do_get_started = state->WaitUntilStarted(std::chrono::seconds(2));
	const bool read_stopped =
	    do_get_started && read_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
	state->Release();
	read_future.wait();
	auto read_error = read_future.get();
	auto shutdown_status = server.Shutdown();

	REQUIRE(do_get_started);
	REQUIRE(read_stopped);
	REQUIRE_THAT(read_error, Catch::Matchers::Contains("flight get schema idle timeout"));
	REQUIRE(shutdown_status.ok());
}

TEST_CASE("Exchange: Flight call deadline bounds a stalled initial schema", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto state = std::make_shared<BlockingFlightState>();
	BlockingDoGetFlightServer server(state);
	StartTestFlightServer(server);

	FlightExchangeConfig config;
	config.local_dirs = {TestCreatePath("flight_call_deadline")};
	config.node_id = "reader-node";
	config.expected_types = {LogicalType::INTEGER};
	config.flight_timeout_seconds = 0.5;
	config.flight_read_idle_timeout_seconds = 5.0;
	auto handle = MakeRemoteSourceHandle(server.port());
	auto read_future = std::async(std::launch::async, [&]() {
		try {
			ReadSourceRows(*conn.context, config, {std::move(handle)});
			return string();
		} catch (const std::exception &ex) {
			return string(ex.what());
		}
	});

	const bool do_get_started = state->WaitUntilStarted(std::chrono::seconds(2));
	const bool read_stopped =
	    do_get_started && read_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
	state->Release();
	read_future.wait();
	auto read_error = read_future.get();
	auto shutdown_status = server.Shutdown();

	REQUIRE(do_get_started);
	REQUIRE(read_stopped);
	REQUIRE_THAT(read_error, Catch::Matchers::Contains("Deadline Exceeded"));
	REQUIRE(shutdown_status.ok());
}

TEST_CASE("Exchange: query interrupt cancels a stalled Flight DoGet", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto state = std::make_shared<BlockingFlightState>();
	BlockingDoGetFlightServer server(state);
	StartTestFlightServer(server);

	FlightExchangeConfig config;
	config.local_dirs = {TestCreatePath("flight_interrupt_watchdog")};
	config.node_id = "reader-node";
	config.expected_types = {LogicalType::INTEGER};
	config.flight_timeout_seconds = 5.0;
	config.flight_read_idle_timeout_seconds = 5.0;
	auto handle = MakeRemoteSourceHandle(server.port());
	auto read_future = std::async(std::launch::async, [&]() {
		try {
			ReadSourceRows(*conn.context, config, {std::move(handle)});
			return string("completed");
		} catch (const InterruptException &) {
			return string("interrupted");
		} catch (const std::exception &ex) {
			return string(ex.what());
		}
	});

	const bool do_get_started = state->WaitUntilStarted(std::chrono::seconds(2));
	if (do_get_started) {
		conn.context->Interrupt();
	}
	const bool read_stopped =
	    do_get_started && read_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
	state->Release();
	read_future.wait();
	auto outcome = read_future.get();
	conn.context->ClearInterrupt();
	auto shutdown_status = server.Shutdown();

	REQUIRE(do_get_started);
	REQUIRE(read_stopped);
	REQUIRE(outcome == "interrupted");
	REQUIRE(shutdown_status.ok());
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
	auto cache = std::make_shared<ShuffleCache>(std::move(config));

	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::VARCHAR};
	vector<int32_t> ids = {1, 2, 3};
	vector<string> names = {"a", "b", "c"};
	DataChunk chunk;
	PopulateTwoColumnChunk(chunk, types, ids, names);

	auto write_res = cache->WriteChunk(context, chunk, 1, {"id", "name"});
	REQUIRE(write_res.is_ok());

	auto flush_res = cache->FlushAll(context, cache->BufferedNames());
	REQUIRE(flush_res.is_ok());

	auto files_res = cache->GetPartitionFiles(1);
	REQUIRE(files_res.is_ok());
	auto files = files_res.value();
	REQUIRE(files.files.size() == 1);
	const auto &file = files.files[0];
	REQUIRE(file.rows == static_cast<idx_t>(ids.size()));
	REQUIRE(file.bytes > 0);
	REQUIRE(!file.path.empty());
	REQUIRE(files.total_rows == static_cast<idx_t>(ids.size()));
	REQUIRE(files.total_bytes >= file.bytes);

	REQUIRE(cache->WriteAttemptManifest(0, 0).is_ok());
	REQUIRE(ShuffleCacheRegistry::Instance().Register("stage_1", cache, "cache-basic-query").is_ok());
	ScopedShuffleCacheRegistration registration("stage_1");
	FlightExchangeConfig source_config;
	source_config.node_id = "node_1";
	source_config.expected_types = types;
	auto rows = ReadSourceRows(context, source_config, {MakeSourceHandle("stage_1", "node_1", 1)});
	RequireTwoColumnRows(rows, ids, names);
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
	auto cache = std::make_shared<ShuffleCache>(std::move(config));

	vector<int32_t> ids = {1, 2};
	vector<string> blobs = {string(4096, 'a'), string(4096, 'b')};
	DataChunk chunk;
	PopulateBlobChunk(chunk, ids, blobs);

	REQUIRE(cache->WriteChunk(context, chunk, 0, {"id", "payload"}).is_ok());

	auto files_res = cache->GetPartitionFiles(0);
	REQUIRE(files_res.is_ok());
	auto files = files_res.value();
	REQUIRE(files.files.size() == 1);
	REQUIRE(files.total_rows == static_cast<idx_t>(ids.size()));
	REQUIRE(files.total_bytes > 0);

	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::BLOB};
	REQUIRE(cache->FlushAll(context, {"id", "payload"}).is_ok());
	REQUIRE(cache->WriteAttemptManifest(0, 0).is_ok());
	REQUIRE(ShuffleCacheRegistry::Instance().Register("stage_large_blob", cache, "cache-blob-query").is_ok());
	ScopedShuffleCacheRegistration registration("stage_large_blob");
	FlightExchangeConfig source_config;
	source_config.node_id = "node_blob";
	source_config.expected_types = types;
	auto rows = ReadSourceRows(context, source_config, {MakeSourceHandle("stage_large_blob", "node_blob", 0)});
	REQUIRE(rows.size() == ids.size());
}

TEST_CASE("Exchange: ShuffleCache bounds aggregate buffers across partitions", "[distributed][exchange]") {
	static constexpr idx_t PARTITION_COUNT = 512;
	static constexpr idx_t WRITER_COUNT = 8;
	static constexpr idx_t MAX_BUFFERED_BYTES = 1024 * 1024;
	ScopedEnvVar max_buffered_bytes("VANE_SHUFFLE_CACHE_MAX_BUFFERED_BYTES", std::to_string(MAX_BUFFERED_BYTES));

	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;

	ShuffleCacheConfig config;
	config.shuffle_stage_id = "stage_aggregate_buffer_budget";
	config.node_id = "node_budget";
	config.num_partitions = PARTITION_COUNT;
	config.local_dirs = {TestCreatePath("exchange_cache_aggregate_buffer_budget")};
	ShuffleCache cache(std::move(config));

	vector<LogicalType> types = {LogicalType::INTEGER};
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), types);
	chunk.SetCardinality(1);
	chunk.SetValue(0, 0, Value::INTEGER(42));

	REQUIRE(cache.WriteChunk(context, chunk, 0, {}).is_ok());
	const auto single_partition_bytes = cache.GetBufferedBytes();
	const auto buffer_budget = cache.GetBufferBudgetBytes();
	REQUIRE(single_partition_bytes > 0);
	REQUIRE(buffer_budget > 0);
	REQUIRE(buffer_budget <= MAX_BUFFERED_BYTES);

	std::mutex start_mutex;
	std::condition_variable start_cv;
	idx_t ready_writers = 0;
	bool start_writers = false;
	vector<std::future<bool>> writers;
	for (idx_t writer_idx = 0; writer_idx < WRITER_COUNT; writer_idx++) {
		writers.push_back(std::async(std::launch::async, [&, writer_idx]() {
			DataChunk writer_chunk;
			writer_chunk.Initialize(Allocator::DefaultAllocator(), types);
			writer_chunk.SetCardinality(1);
			writer_chunk.SetValue(0, 0, Value::INTEGER(static_cast<int32_t>(writer_idx)));
			{
				std::unique_lock<std::mutex> lock(start_mutex);
				ready_writers++;
				start_cv.notify_all();
				start_cv.wait(lock, [&]() { return start_writers; });
			}
			for (idx_t partition_idx = writer_idx + 1; partition_idx < PARTITION_COUNT; partition_idx += WRITER_COUNT) {
				if (cache.WriteChunk(context, writer_chunk, partition_idx, {"value"}).is_err()) {
					return false;
				}
			}
			return true;
		}));
	}
	{
		std::unique_lock<std::mutex> lock(start_mutex);
		start_cv.wait(lock, [&]() { return ready_writers == WRITER_COUNT; });
		start_writers = true;
	}
	start_cv.notify_all();
	for (auto &writer : writers) {
		REQUIRE(writer.get());
	}
	REQUIRE(cache.BufferedNames() == vector<string> {"value"});
	REQUIRE(cache.GetBufferedBytes() <= buffer_budget);
	REQUIRE(cache.GetPeakBufferedBytes() <= buffer_budget + WRITER_COUNT * single_partition_bytes);

	idx_t pressure_flushed_files = 0;
	for (idx_t partition_idx = 0; partition_idx < PARTITION_COUNT; partition_idx++) {
		auto files_res = cache.GetPartitionFiles(partition_idx);
		REQUIRE(files_res.is_ok());
		pressure_flushed_files += files_res.value().files.size();
	}
	REQUIRE(pressure_flushed_files > 0);

	REQUIRE(cache.FlushAll(context, {"value"}).is_ok());
	REQUIRE(cache.GetBufferedBytes() == 0);
	idx_t total_rows = 0;
	for (idx_t partition_idx = 0; partition_idx < PARTITION_COUNT; partition_idx++) {
		auto files_res = cache.GetPartitionFiles(partition_idx);
		REQUIRE(files_res.is_ok());
		total_rows += files_res.value().total_rows;
	}
	REQUIRE(total_rows == PARTITION_COUNT);
}

TEST_CASE("Exchange: ShuffleCache validates buffer byte settings", "[distributed][exchange]") {
	auto make_config = []() {
		ShuffleCacheConfig config;
		config.shuffle_stage_id = "stage_invalid_buffer_budget";
		config.node_id = "node_budget";
		config.num_partitions = 1;
		config.local_dirs = {TestCreatePath("exchange_cache_invalid_buffer_budget")};
		return config;
	};

	SECTION("aggregate budget cannot use the optional_idx sentinel") {
		ScopedEnvVar max_buffered_bytes("VANE_SHUFFLE_CACHE_MAX_BUFFERED_BYTES",
		                                std::to_string(std::numeric_limits<idx_t>::max()));
		REQUIRE_THROWS_WITH(ShuffleCache(make_config()), Catch::Matchers::Contains("must be less than idx_t max"));
	}
	SECTION("flush threshold rejects a whitespace-prefixed negative value") {
		ScopedEnvVar flush_threshold("VANE_SHUFFLE_CACHE_FLUSH_THRESHOLD_BYTES", " -2");
		REQUIRE_THROWS_WITH(ShuffleCache(make_config()), Catch::Matchers::Contains("positive integer byte count"));
	}
	SECTION("aggregate budget rejects a whitespace-prefixed negative value") {
		ScopedEnvVar max_buffered_bytes("VANE_SHUFFLE_CACHE_MAX_BUFFERED_BYTES", " -2");
		REQUIRE_THROWS_WITH(ShuffleCache(make_config()), Catch::Matchers::Contains("positive integer byte count"));
	}
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

	auto files_res = replay_cache.GetPartitionFilesFromManifest(1);
	REQUIRE(files_res.is_ok());
	REQUIRE(files_res.value().total_rows == static_cast<idx_t>(ids.size()));
	REQUIRE(files_res.value().files.size() == 1);
	auto input_res = replay_cache.OpenPartitionFile(files_res.value().files[0].path);
	REQUIRE(input_res.is_ok());
	auto reader_res = arrow::ipc::RecordBatchStreamReader::Open(input_res.value());
	REQUIRE(reader_res.ok());
	auto reader = std::move(reader_res).ValueOrDie();
	std::shared_ptr<arrow::RecordBatch> batch;
	REQUIRE(reader->ReadNext(&batch).ok());
	REQUIRE(batch != nullptr);
	REQUIRE(batch->num_rows() == static_cast<int64_t>(ids.size()));
	auto id_array = std::static_pointer_cast<arrow::Int32Array>(batch->column(0));
	auto name_array = std::static_pointer_cast<arrow::StringArray>(batch->column(1));
	for (idx_t row_idx = 0; row_idx < ids.size(); row_idx++) {
		REQUIRE(id_array->Value(row_idx) == ids[row_idx]);
		REQUIRE(name_array->GetString(row_idx) == names[row_idx]);
	}
	REQUIRE(reader->ReadNext(&batch).ok());
	REQUIRE(batch == nullptr);
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
	auto cache = std::make_shared<ShuffleCache>(std::move(config));

	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::VARCHAR};
	DataChunk zero_row_chunk;
	zero_row_chunk.Initialize(Allocator::DefaultAllocator(), types);
	REQUIRE(cache->WriteChunk(context, zero_row_chunk, 0, {"id", "name"}).is_ok());
	REQUIRE(cache->GetBufferedBytes() == 0);
	REQUIRE(cache->GetBufferBudgetBytes() == 0);
	REQUIRE(cache->EnsureSchemaFile(context, types, {"id", "name"}).is_ok());
	REQUIRE(cache->FlushAll(context, {"id", "name"}).is_ok());
	REQUIRE(cache->WriteAttemptManifest(0, 0).is_ok());
	REQUIRE(ShuffleCacheRegistry::Instance().Register("stage_empty", cache, "cache-empty-query").is_ok());
	ScopedShuffleCacheRegistration registration("stage_empty");
	FlightExchangeConfig source_config;
	source_config.node_id = "node_empty";
	source_config.expected_types = types;
	auto rows = ReadSourceRows(context, source_config, {MakeSourceHandle("stage_empty", "node_empty", 0)});
	REQUIRE(rows.empty());

	vector<int32_t> ids = {9};
	vector<string> names_vec = {"x"};
	DataChunk chunk;
	PopulateTwoColumnChunk(chunk, types, ids, names_vec);

	auto bad_partition_res = cache->WriteChunk(context, chunk, 2, {"id", "name"});
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
	auto cache = std::make_shared<ShuffleCache>(std::move(config));

	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::VARCHAR};

	// Write first chunk
	vector<int32_t> ids1 = {1, 2};
	vector<string> names1 = {"a", "b"};
	DataChunk chunk1;
	PopulateTwoColumnChunk(chunk1, types, ids1, names1);
	REQUIRE(cache->WriteChunk(context, chunk1, 0, {"id", "name"}).is_ok());

	// Write second chunk
	vector<int32_t> ids2 = {3, 4, 5};
	vector<string> names2 = {"c", "d", "e"};
	DataChunk chunk2;
	PopulateTwoColumnChunk(chunk2, types, ids2, names2);
	REQUIRE(cache->WriteChunk(context, chunk2, 0, {"id", "name"}).is_ok());

	REQUIRE(cache->FlushAll(context, cache->BufferedNames()).is_ok());
	REQUIRE(cache->WriteAttemptManifest(0, 0).is_ok());
	REQUIRE(ShuffleCacheRegistry::Instance().Register("stage_multi_chunk", cache, "cache-multi-chunk-query").is_ok());
	ScopedShuffleCacheRegistration registration("stage_multi_chunk");
	FlightExchangeConfig source_config;
	source_config.node_id = "node_1";
	source_config.expected_types = types;
	auto rows = ReadSourceRows(context, source_config, {MakeSourceHandle("stage_multi_chunk", "node_1", 0)});
	REQUIRE(rows.size() == 5);

	// All rows should be present
	vector<int32_t> all_ids = {1, 2, 3, 4, 5};
	vector<string> all_names = {"a", "b", "c", "d", "e"};
	RequireTwoColumnRows(rows, all_ids, all_names);
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
	auto cache = std::make_shared<ShuffleCache>(std::move(config));

	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::VARCHAR};

	// Write to partition 0
	vector<int32_t> ids0 = {10, 20};
	vector<string> names0 = {"ten", "twenty"};
	DataChunk chunk0;
	PopulateTwoColumnChunk(chunk0, types, ids0, names0);
	REQUIRE(cache->WriteChunk(context, chunk0, 0, {"id", "name"}).is_ok());

	// Write to partition 2
	vector<int32_t> ids2 = {30};
	vector<string> names2 = {"thirty"};
	DataChunk chunk2;
	PopulateTwoColumnChunk(chunk2, types, ids2, names2);
	REQUIRE(cache->WriteChunk(context, chunk2, 2, {"id", "name"}).is_ok());

	REQUIRE(cache->FlushAll(context, cache->BufferedNames()).is_ok());
	REQUIRE(cache->WriteAttemptManifest(0, 0).is_ok());
	REQUIRE(ShuffleCacheRegistry::Instance().Register("stage_multi_part", cache, "cache-multi-part-query").is_ok());
	ScopedShuffleCacheRegistration registration("stage_multi_part");
	FlightExchangeConfig source_config;
	source_config.node_id = "node_1";
	source_config.expected_types = types;

	// Partition 0 should have 2 rows
	auto rows0 = ReadSourceRows(context, source_config, {MakeSourceHandle("stage_multi_part", "node_1", 0)});
	RequireTwoColumnRows(rows0, ids0, names0);

	// Partition 1 should be empty
	auto rows1 = ReadSourceRows(context, source_config, {MakeSourceHandle("stage_multi_part", "node_1", 1)});
	REQUIRE(rows1.empty());

	// Partition 2 should have 1 row
	auto rows2 = ReadSourceRows(context, source_config, {MakeSourceHandle("stage_multi_part", "node_1", 2)});
	RequireTwoColumnRows(rows2, ids2, names2);
}

// ═══════════════════════════════════════════════════════════
// FlightExchangeManager
// ═══════════════════════════════════════════════════════════

TEST_CASE("Exchange: local-disk sink starts and publishes the cluster-internal Flight service",
          "[distributed][exchange]") {
	REQUIRE(FlightExchangeManager::ShutdownLocalFlightServer().is_ok());
	ScopedEnvVar bind_host("VANE_FLIGHT_BIND_HOST", "127.0.0.1");
	ScopedEnvVar advertise_host("VANE_FLIGHT_ADVERTISE_HOST", "127.0.0.1");
	ScopedEnvVar configured_port("DUCKDB_FLIGHT_PORT", "0");
	ScopedEnvVar worker_id("VANE_WORKER_ID", "process-local-node");
	auto config = ResolveFlightExchangeConfigFromEnv();
	config.local_dirs = {TestCreatePath("exchange_process_local_default")};
	auto service_config = ResolveFlightServiceConfigFromEnv();
	REQUIRE(service_config.bind_host == "127.0.0.1");
	REQUIRE(service_config.advertise_host == "127.0.0.1");
	REQUIRE(service_config.port == 0);

	DuckDB db(nullptr);
	Connection conn(db);
	FlightExchangeManager manager(config, conn.context.get());
	ExchangeContext exchange_context;
	exchange_context.query_id = "process-local-default-query";
	exchange_context.exchange_id = "process-local-default-stage";
	auto exchange = manager.CreateExchange(exchange_context, 1);
	auto instance = exchange->InstantiateSink(exchange->AddSink(0), 0);
	auto sink = manager.CreateSink(instance);
	REQUIRE(sink != nullptr);
	REQUIRE(manager.GetPublishedFlightServerHost() == "127.0.0.1");
	REQUIRE(manager.GetPublishedFlightServerPort() > 0);
	REQUIRE_FALSE(manager.GetPublishedFlightServerEpoch().empty());
	REQUIRE(FlightExchangeManager::GetLocalFlightServerHost() == "127.0.0.1");

	REQUIRE(sink->EnsureSchema(*conn.context, {LogicalType::INTEGER}, {"value"}).is_ok());
	DataChunk input;
	input.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER});
	input.SetCardinality(1);
	input.SetValue(0, 0, Value::INTEGER(57));
	REQUIRE(sink->AddChunk(0, input).is_ok());
	REQUIRE(sink->Finish().is_ok());
	instance.flight_host = manager.GetPublishedFlightServerHost();
	instance.flight_server_epoch = manager.GetPublishedFlightServerEpoch();
	exchange->SinkFinished(instance, config.node_id, manager.GetPublishedFlightServerPort());
	auto handles = exchange->GetSourceHandles();
	REQUIRE(handles.size() == 1);
	REQUIRE(handles[0].node_id == "process-local-node");
	REQUIRE(handles[0].flight_host == "127.0.0.1");
	REQUIRE(handles[0].flight_port == manager.GetPublishedFlightServerPort());
	REQUIRE(handles[0].flight_server_epoch == manager.GetPublishedFlightServerEpoch());

	auto source = manager.CreateSource();
	source->AddSourceHandles(handles);
	DataChunk output;
	output.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER});
	REQUIRE(source->ReadChunk(output));
	REQUIRE(output.size() == 1);
	REQUIRE(output.GetValue(0, 0) == Value::INTEGER(57));
	REQUIRE_FALSE(source->ReadChunk(output));
	source->Close();
	sink.reset();
	exchange->Close();
	auto cleanup = ShuffleCacheRegistry::Instance().RemoveAndCleanupByQuery(exchange_context.query_id);
	REQUIRE(cleanup.cleanup_errors == 0);
	REQUIRE(cleanup.cleanup_pending == 0);
	REQUIRE(ShuffleCacheRegistry::Instance().RetireQuery(exchange_context.query_id).is_ok());
	REQUIRE(FlightExchangeManager::ShutdownLocalFlightServer().is_ok());
}

TEST_CASE("Exchange: Flight service normalizes IPv6 hosts before building Arrow URIs",
          "[distributed][exchange][security]") {
	ScopedEnvVar bind_host("VANE_FLIGHT_BIND_HOST", "");
	for (const auto &configured_host : {"::1", "[::1]"}) {
		INFO("configured host: " << configured_host);
		ScopedEnvVar advertise_host("VANE_FLIGHT_ADVERTISE_HOST", configured_host);
		auto config = ResolveFlightServiceConfigFromEnv();
		REQUIRE(config.advertise_host == "::1");
		REQUIRE(config.bind_host == "::1");

		auto location = BuildFlightLocation(config.bind_host, 1234);
		REQUIRE(location == "grpc+tcp://[::1]:1234");
		REQUIRE(arrow::flight::Location::Parse(location).ok());
	}

	ScopedEnvVar advertise_host("VANE_FLIGHT_ADVERTISE_HOST", "");
	ScopedEnvVar ray_node_ip("RAY_NODE_IP_ADDRESS", "::1");
	auto config = ResolveFlightServiceConfigFromEnv();
	REQUIRE(config.advertise_host == "::1");
	REQUIRE(config.bind_host == "::1");
}

TEST_CASE("Exchange: Flight service rejects wildcard advertised IP addresses", "[distributed][exchange][security]") {
	for (const auto &configured_host : {"0.0.0.0", "::", "[::]", "[::0]", "0:0:0:0:0:0:0:0", "[0:0:0:0:0:0:0:0]",
	                                    "::ffff:0.0.0.0", "[::ffff:0:0]", "0:0:0:0:0:ffff:0:0"}) {
		INFO("configured host: " << configured_host);
		ScopedEnvVar advertise_host("VANE_FLIGHT_ADVERTISE_HOST", configured_host);
		REQUIRE_THROWS_WITH(ResolveFlightServiceConfigFromEnv(), Catch::Matchers::Contains("must be a routable host"));
	}
}

TEST_CASE("Exchange: Flight service rejects non-canonical numeric IPv4 hosts", "[distributed][exchange][security]") {
	for (const auto &configured_host : {"00.00.00.00", "000.000.000.000", "0.0.0.00"}) {
		INFO("configured host: " << configured_host);
		ScopedEnvVar advertise_host("VANE_FLIGHT_ADVERTISE_HOST", configured_host);
		REQUIRE_THROWS_WITH(ResolveFlightServiceConfigFromEnv(),
		                    Catch::Matchers::Contains("not a canonical IPv4 literal"));
	}
}

TEST_CASE("Exchange: remote local-disk source rejects a wildcard advertised host before connecting",
          "[distributed][exchange][security]") {
	DuckDB db(nullptr);
	Connection conn(db);
	FlightExchangeConfig config;
	config.node_id = "reader-node";
	config.local_dirs = {TestCreatePath("exchange_remote_disabled")};
	config.expected_types = {LogicalType::INTEGER};
	FlightExchangeSource source(config, conn.context.get());

	ExchangeSourceHandle handle;
	handle.partition_id = 0;
	handle.attempt_id = 1;
	handle.node_id = "writer-worker";
	handle.flight_host = "[::ffff:0.0.0.0]";
	handle.flight_port = 1;
	handle.flight_server_epoch = "remote-epoch";
	handle.files.push_back(ExchangeSourceFile("remote-disabled-attempt", 0));
	source.AddSourceHandles({handle});

	DataChunk output;
	output.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER});
	REQUIRE_THROWS_WITH(source.ReadChunk(output), Catch::Matchers::Contains("advertises a wildcard host"));
}

TEST_CASE("Exchange: remote local-disk source rejects non-canonical numeric IPv4 advertised hosts",
          "[distributed][exchange][security]") {
	DuckDB db(nullptr);
	Connection conn(db);
	FlightExchangeConfig config;
	config.node_id = "reader-node";
	config.local_dirs = {TestCreatePath("exchange_remote_noncanonical_ipv4")};
	config.expected_types = {LogicalType::INTEGER};

	for (const auto &configured_host : {"00.00.00.00", "000.000.000.000", "0.0.0.00"}) {
		INFO("configured host: " << configured_host);
		FlightExchangeSource source(config, conn.context.get());
		ExchangeSourceHandle handle;
		handle.partition_id = 0;
		handle.attempt_id = 1;
		handle.node_id = "writer-worker";
		handle.flight_host = configured_host;
		handle.flight_port = 1;
		handle.flight_server_epoch = "remote-epoch";
		handle.files.push_back(ExchangeSourceFile("remote-disabled-attempt", 0));
		source.AddSourceHandles({handle});

		DataChunk output;
		output.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER});
		REQUIRE_THROWS_WITH(source.ReadChunk(output), Catch::Matchers::Contains("not a canonical IPv4 literal"));
	}
}

TEST_CASE("Exchange: local-disk source requires explicit producer and endpoint metadata",
          "[distributed][exchange][security]") {
	DuckDB db(nullptr);
	Connection conn(db);
	FlightExchangeConfig config;
	config.node_id = "reader-node";
	config.local_dirs = {TestCreatePath("exchange_remote_strict_endpoint")};
	config.expected_types = {LogicalType::INTEGER};

	auto read_handle = [&](ExchangeSourceHandle handle) {
		FlightExchangeSource source(config, conn.context.get());
		source.AddSourceHandles({std::move(handle)});
		DataChunk output;
		output.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER});
		source.ReadChunk(output);
	};

	ExchangeSourceHandle complete;
	complete.partition_id = 0;
	complete.attempt_id = 1;
	complete.node_id = "writer-worker";
	complete.flight_host = "flight-writer.internal";
	complete.flight_port = 5010;
	complete.flight_server_epoch = "writer-epoch";
	complete.files.push_back(ExchangeSourceFile("strict-endpoint-attempt", 0));

	auto missing_identity = complete;
	missing_identity.node_id.clear();
	REQUIRE_THROWS_WITH(read_handle(std::move(missing_identity)), Catch::Matchers::Contains("producer identity"));

	auto missing_host = complete;
	missing_host.flight_host.clear();
	REQUIRE_THROWS_WITH(read_handle(std::move(missing_host)),
	                    Catch::Matchers::Contains("requires producer identity, host, port, and server epoch"));

	auto missing_port = complete;
	missing_port.flight_port = 0;
	REQUIRE_THROWS_WITH(read_handle(std::move(missing_port)),
	                    Catch::Matchers::Contains("requires producer identity, host, port, and server epoch"));

	auto missing_epoch = complete;
	missing_epoch.flight_server_epoch.clear();
	REQUIRE_THROWS_WITH(read_handle(std::move(missing_epoch)),
	                    Catch::Matchers::Contains("requires producer identity, host, port, and server epoch"));
}

TEST_CASE("Exchange: object-storage source rejects Flight endpoint metadata", "[distributed][exchange][security]") {
	DuckDB db(nullptr);
	Connection conn(db);
	FlightExchangeConfig config;
	config.node_id = "reader-node";
	config.local_dirs = {"s3://bucket/shuffle"};
	config.expected_types = {LogicalType::INTEGER};
	FlightExchangeSource source(config, conn.context.get());

	ExchangeSourceHandle handle;
	handle.partition_id = 0;
	handle.attempt_id = 1;
	handle.node_id = "writer-worker";
	handle.flight_host = "flight-writer.internal";
	handle.flight_port = 5010;
	handle.flight_server_epoch = "writer-epoch";
	handle.files.push_back(ExchangeSourceFile("s3://bucket/shuffle/strict-endpoint-attempt", 0));
	source.AddSourceHandles({std::move(handle)});

	DataChunk output;
	output.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER});
	REQUIRE_THROWS_WITH(source.ReadChunk(output),
	                    Catch::Matchers::Contains("non-local-disk Flight source handle cannot contain"));
}

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
	REQUIRE(exchange->AddSink(0).task_partition_id == 0);
	REQUIRE_THROWS_WITH(exchange->InstantiateSink(ExchangeSinkHandle {99}, 0),
	                    Catch::Matchers::Contains("partition was not registered"));

	// Instantiate sinks
	auto inst0 = exchange->InstantiateSink(sink_handle0, 0);
	auto inst1 = exchange->InstantiateSink(sink_handle1, 0);
	REQUIRE(inst0.output_partition_count == 4);
	REQUIRE(inst1.output_partition_count == 4);
	REQUIRE(inst0.output_location != inst1.output_location);
	REQUIRE(inst0.output_location.find(ctx.exchange_id) == string::npos);
	REQUIRE(inst1.output_location.find(ctx.exchange_id) == string::npos);
	REQUIRE(inst0.output_location.find("__sink_0__attempt_0") != string::npos);
	REQUIRE(inst1.output_location.find("__sink_1__attempt_0") != string::npos);
	inst0.flight_host = "flight-worker-0.internal";
	inst0.flight_server_epoch = "worker-0-epoch";
	inst1.flight_host = "flight-worker-1.internal";
	inst1.flight_server_epoch = "worker-1-epoch";

	auto wrong_query = inst0;
	wrong_query.query_id = "other-query";
	REQUIRE_THROWS_WITH(exchange->SinkFinished(wrong_query, "worker-0", 5000),
	                    Catch::Matchers::Contains("query does not match"));
	auto wrong_location = inst0;
	wrong_location.output_location += "__wrong";
	REQUIRE_THROWS_WITH(exchange->SinkFinished(wrong_location, "worker-0", 5000),
	                    Catch::Matchers::Contains("output location does not match"));
	auto wrong_partition_count = inst0;
	wrong_partition_count.output_partition_count++;
	REQUIRE_THROWS_WITH(exchange->SinkFinished(wrong_partition_count, "worker-0", 5000),
	                    Catch::Matchers::Contains("partition count does not match"));
	REQUIRE_THROWS_WITH(exchange->SinkFinished(inst0, "worker-0", -1),
	                    Catch::Matchers::Contains("port must be non-negative"));
	REQUIRE_THROWS_WITH(exchange->SinkFinished(sink_handle0, 99),
	                    Catch::Matchers::Contains("attempt was not instantiated"));
	REQUIRE_THROWS_WITH(exchange->SinkFinished(inst0, "", 5000),
	                    Catch::Matchers::Contains("missing its worker identity"));
	auto missing_host = inst0;
	missing_host.flight_host.clear();
	REQUIRE_THROWS_WITH(exchange->SinkFinished(missing_host, "worker-0", 5000),
	                    Catch::Matchers::Contains("endpoint requires host, port, and server epoch"));
	REQUIRE_THROWS_WITH(exchange->SinkFinished(inst0, "worker-0", 0),
	                    Catch::Matchers::Contains("endpoint requires host, port, and server epoch"));
	auto missing_epoch = inst0;
	missing_epoch.flight_server_epoch.clear();
	REQUIRE_THROWS_WITH(exchange->SinkFinished(missing_epoch, "worker-0", 5000),
	                    Catch::Matchers::Contains("endpoint requires host, port, and server epoch"));

	// Finish sinks
	exchange->SinkFinished(inst0, "worker-0", 5000);
	exchange->SinkFinished(inst0, "worker-0", 5000);
	auto wrong_node = inst0;
	REQUIRE_THROWS_WITH(exchange->SinkFinished(wrong_node, "worker-other", 5000),
	                    Catch::Matchers::Contains("node does not match"));
	auto wrong_host = inst0;
	wrong_host.flight_host = "flight-worker-other.internal";
	REQUIRE_THROWS_WITH(exchange->SinkFinished(wrong_host, "worker-0", 5000),
	                    Catch::Matchers::Contains("host does not match"));
	auto wrong_port = inst0;
	REQUIRE_THROWS_WITH(exchange->SinkFinished(wrong_port, "worker-0", 5001),
	                    Catch::Matchers::Contains("port does not match"));
	auto wrong_epoch = inst0;
	wrong_epoch.flight_server_epoch = "worker-other-epoch";
	REQUIRE_THROWS_WITH(exchange->SinkFinished(wrong_epoch, "worker-0", 5000),
	                    Catch::Matchers::Contains("epoch does not match"));
	exchange->SinkFinished(inst1, "worker-1", 5001);
	exchange->AllRequiredSinksFinished();

	// Source handles should cover all partitions
	auto source_handles = exchange->GetSourceHandles();
	// Source handles generated for non-empty partitions
	// (may be empty since no data was actually written)

	exchange->Close();
	REQUIRE_THROWS_WITH(exchange->AddSink(2), Catch::Matchers::Contains("exchange is closed"));
	REQUIRE_THROWS_WITH(exchange->GetSourceHandles(), Catch::Matchers::Contains("exchange is closed"));
}

TEST_CASE("Exchange: FlightExchange with no sinks has no unpublished source handles", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);

	FlightExchangeConfig config;
	config.node_id = "node-empty";
	config.local_dirs = {TestCreatePath("exchange_no_sinks")};
	FlightExchangeManager manager(config, conn.context.get());

	ExchangeContext ctx;
	ctx.query_id = "empty-query";
	ctx.exchange_id = "empty-stage";
	auto exchange = manager.CreateExchange(ctx, 4);
	exchange->AllRequiredSinksFinished();
	REQUIRE(exchange->GetSourceHandles().empty());
	exchange->Close();
}

TEST_CASE("Exchange: FlightExchange reduces MARK build summaries from selected attempts only",
          "[distributed][exchange][join]") {
	DuckDB db(nullptr);
	Connection conn(db);

	FlightExchangeConfig config;
	config.node_id = "mark-summary-coordinator";
	config.local_dirs = {TestCreatePath("exchange_mark_build_summary")};
	FlightExchangeManager manager(config, conn.context.get());

	ExchangeContext ctx;
	ctx.query_id = "mark-summary-query";
	ctx.exchange_id = "mark-summary-stage";
	auto exchange = manager.CreateExchange(ctx, 3);
	auto first = exchange->InstantiateSink(exchange->AddSink(0), 0);
	auto first_retry = exchange->InstantiateSink(exchange->AddSink(0), 1);
	auto second = exchange->InstantiateSink(exchange->AddSink(1), 0);
	first.flight_host = "mark-worker-0.internal";
	first.flight_server_epoch = "mark-epoch-0";
	first.mark_join_build_summary = MarkJoinBuildSummary::Create(false, false);
	first_retry.flight_host = "mark-worker-retry.internal";
	first_retry.flight_server_epoch = "mark-epoch-retry";
	first_retry.mark_join_build_summary = MarkJoinBuildSummary::Create(true, true);
	second.flight_host = "mark-worker-1.internal";
	second.flight_server_epoch = "mark-epoch-1";
	second.mark_join_build_summary = MarkJoinBuildSummary::Create(true, false);

	auto malformed = first;
	malformed.mark_join_build_summary = MarkJoinBuildSummary();
	malformed.mark_join_build_summary.has_rows = true;
	REQUIRE_THROWS_WITH(exchange->SinkFinished(malformed, "mark-worker-0", 5100),
	                    Catch::Matchers::Contains("invalid MARK join build summary"));

	exchange->SinkFinished(first, "mark-worker-0", 5100);
	exchange->SinkFinished(first_retry, "mark-worker-retry", 5102);
	exchange->SinkFinished(second, "mark-worker-1", 5101);
	exchange->AllRequiredSinksFinished();
	auto handles = exchange->GetSourceHandles();
	REQUIRE(handles.size() == 6);
	for (const auto &handle : handles) {
		REQUIRE(handle.mark_join_build_summary.valid);
		REQUIRE(handle.mark_join_build_summary.has_rows);
		REQUIRE_FALSE(handle.mark_join_build_summary.has_null);
	}
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
	REQUIRE(instance_a.output_location.find(ctx.exchange_id) == string::npos);
	REQUIRE(instance_b.output_location.find(ctx.exchange_id) == string::npos);

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
	REQUIRE(registry.Register(instance_a.output_location, cache_a, ctx.query_id, "epoch-a", 0).is_ok());
	REQUIRE(registry.Register(instance_b.output_location, cache_b, ctx.query_id, "epoch-a", 0).is_ok());

	exchange_a->Close();
	REQUIRE(registry.Get(instance_a.output_location).get() == cache_a.get());
	REQUIRE(registry.Get(instance_b.output_location).get() == cache_b.get());

	exchange_b->Close();
	REQUIRE(registry.Get(instance_a.output_location).get() == cache_a.get());
	REQUIRE(registry.Get(instance_b.output_location).get() == cache_b.get());
	auto cleanup = registry.RemoveAndCleanupByQuery(ctx.query_id);
	REQUIRE(cleanup.registry_entries_removed == 2);
	REQUIRE(cleanup.cleanup_pending == 0);
	REQUIRE(cleanup.cleanup_errors == 0);
	REQUIRE(registry.RetireQuery(ctx.query_id).is_ok());
}

TEST_CASE("Exchange: FlightExchange accepts validated dynamically derived retry attempts", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);

	FlightExchangeConfig config;
	config.node_id = "coordinator";
	config.local_dirs = {TestCreatePath("exchange_dynamic_retry")};
	FlightExchangeManager manager(config, conn.context.get());
	ExchangeContext ctx;
	ctx.query_id = "dynamic-retry-query";
	ctx.exchange_id = "diagnostic-stage";
	auto exchange = manager.CreateExchange(ctx, 1);
	auto sink = exchange->AddSink(0);
	auto initial = exchange->InstantiateSink(sink, 0);

	auto retry = initial;
	retry.attempt_id = 2;
	auto suffix = retry.output_location.rfind("__attempt_0");
	REQUIRE(suffix != std::string::npos);
	retry.output_location.replace(suffix, std::string("__attempt_0").size(), "__attempt_2");
	retry.flight_host = "[::1]";
	retry.flight_server_epoch = "retry-epoch";
	exchange->SinkFinished(retry, "worker-retry", 5010);

	auto handles = exchange->GetSourceHandles();
	REQUIRE(handles.size() == 1);
	REQUIRE(handles[0].attempt_id == 2);
	REQUIRE(handles[0].node_id == "worker-retry");
	REQUIRE(handles[0].flight_host == "::1");
	REQUIRE(handles[0].flight_port == 5010);
	REQUIRE(handles[0].flight_server_epoch == "retry-epoch");
	REQUIRE(handles[0].files.size() == 1);
	REQUIRE(handles[0].files[0].path == retry.output_location);

	auto forged_retry = initial;
	forged_retry.attempt_id = 3;
	forged_retry.output_location = "forged-output";
	forged_retry.flight_host = "retry.internal";
	forged_retry.flight_server_epoch = "retry-epoch";
	REQUIRE_THROWS_WITH(exchange->SinkFinished(forged_retry, "worker-retry", 5010),
	                    Catch::Matchers::Contains("retry output location does not match"));
	REQUIRE_THROWS_WITH(exchange->SinkFinished(sink, 3), Catch::Matchers::Contains("attempt was not instantiated"));
	exchange->Close();
}

TEST_CASE("Exchange: process-local Flight service has immutable network config and explicit shutdown",
          "[distributed][exchange]") {
	struct ServiceGuard {
		~ServiceGuard() {
			FlightExchangeManager::ShutdownLocalFlightServer();
		}
	} guard;
	REQUIRE(FlightExchangeManager::ShutdownLocalFlightServer().is_ok());

	FlightServiceConfig service_config;
	service_config.bind_host = "127.0.0.1";
	service_config.advertise_host = "127.0.0.1";
	service_config.port = 0;
	REQUIRE(FlightExchangeManager::EnsureLocalFlightServerStarted(service_config).is_ok());
	REQUIRE(FlightExchangeManager::GetLocalFlightServerHost() == "127.0.0.1");
	const auto first_port = FlightExchangeManager::GetLocalFlightServerPort();
	const auto first_epoch = FlightExchangeManager::GetLocalFlightServerEpoch();
	REQUIRE(first_port > 0);
	REQUIRE(!first_epoch.empty());

	FlightExchangeConfig config_a;
	config_a.node_id = "node-a";
	config_a.local_dirs = {TestCreatePath("flight_service_lifecycle_a")};
	FlightExchangeManager manager_a(config_a);
	REQUIRE(manager_a.GetPublishedFlightServerHost() == "127.0.0.1");
	REQUIRE(manager_a.GetPublishedFlightServerPort() == first_port);
	REQUIRE(manager_a.GetPublishedFlightServerEpoch() == first_epoch);

	auto object_storage_config = config_a;
	object_storage_config.local_dirs = {"s3://bucket/shuffle"};
	FlightExchangeManager object_storage_manager(object_storage_config);
	REQUIRE(object_storage_manager.GetPublishedFlightServerHost().empty());
	REQUIRE(object_storage_manager.GetPublishedFlightServerPort() == 0);
	REQUIRE(object_storage_manager.GetPublishedFlightServerEpoch().empty());
	REQUIRE(FlightExchangeManager::GetLocalFlightServerPort() == first_port);
	REQUIRE(FlightExchangeManager::GetLocalFlightServerEpoch() == first_epoch);

	REQUIRE(FlightExchangeManager::EnsureLocalFlightServerStarted(service_config).is_ok());
	REQUIRE(FlightExchangeManager::GetLocalFlightServerPort() == first_port);
	REQUIRE(FlightExchangeManager::GetLocalFlightServerEpoch() == first_epoch);

	auto conflicting_config = service_config;
	conflicting_config.advertise_host = "flight-worker.internal";
	auto conflicting_start = FlightExchangeManager::EnsureLocalFlightServerStarted(conflicting_config);
	REQUIRE(conflicting_start.is_err());
	REQUIRE_THAT(conflicting_start.error().what(), Catch::Matchers::Contains("refusing conflicting address"));

	auto conflicting_grace_config = service_config;
	conflicting_grace_config.shutdown_grace_period += std::chrono::milliseconds(1);
	auto conflicting_grace_start = FlightExchangeManager::EnsureLocalFlightServerStarted(conflicting_grace_config);
	REQUIRE(conflicting_grace_start.is_err());
	REQUIRE_THAT(conflicting_grace_start.error().what(), Catch::Matchers::Contains("conflicting grace period"));

	manager_a.Shutdown();
	object_storage_manager.Shutdown();
	REQUIRE(FlightExchangeManager::GetLocalFlightServerPort() == first_port);
	REQUIRE(FlightExchangeManager::GetLocalFlightServerEpoch() == first_epoch);

	REQUIRE(FlightExchangeManager::ShutdownLocalFlightServer().is_ok());
	REQUIRE(FlightExchangeManager::GetLocalFlightServerHost().empty());
	REQUIRE(FlightExchangeManager::GetLocalFlightServerPort() == 0);
	REQUIRE(FlightExchangeManager::GetLocalFlightServerEpoch().empty());

	auto fixed_config = service_config;
	fixed_config.port = first_port;
	REQUIRE(FlightExchangeManager::EnsureLocalFlightServerStarted(fixed_config).is_ok());
	REQUIRE(FlightExchangeManager::GetLocalFlightServerPort() == first_port);
	REQUIRE(FlightExchangeManager::GetLocalFlightServerEpoch() != first_epoch);
	REQUIRE(FlightExchangeManager::ShutdownLocalFlightServer().is_ok());
}

TEST_CASE("Exchange: FlightExchange selects first successful sink attempt", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);

	FlightExchangeConfig config;
	config.node_id = "coordinator";
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

	sink0_attempt1.flight_host = "flight-retry.internal";
	sink0_attempt1.flight_server_epoch = "worker-retry-epoch";
	exchange->SinkFinished(sink0_attempt1, "worker-retry", 5010);
	sink0_attempt0.flight_host = "flight-late.internal";
	sink0_attempt0.flight_server_epoch = "worker-late-epoch";
	exchange->SinkFinished(sink0_attempt0, "worker-late", 5011);
	sink1_attempt0.flight_host = "flight-first.internal";
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
			REQUIRE(handle.source_task_partition_id == 0);
			REQUIRE(handle.attempt_id == 1);
			REQUIRE(handle.node_id == "worker-retry");
			REQUIRE(handle.flight_host == "flight-retry.internal");
			REQUIRE(handle.flight_port == 5010);
			REQUIRE(handle.flight_server_epoch == "worker-retry-epoch");
			REQUIRE(handle.files[0].path.find("__attempt_1") != std::string::npos);
			REQUIRE(handle.files[0].path.find("__attempt_0") == std::string::npos);
		} else if (handle.files[0].path.find("__sink_1__") != std::string::npos) {
			sink1_handles++;
			REQUIRE(handle.source_task_partition_id == 1);
			REQUIRE(handle.attempt_id == 0);
			REQUIRE(handle.node_id == "worker-first");
			REQUIRE(handle.flight_host == "flight-first.internal");
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

TEST_CASE("Exchange: failed unselected-attempt cleanup releases its retry claim", "[distributed][exchange]") {
	FlightExchangeConfig config;
	config.node_id = "coordinator";
	config.local_dirs = {"s3://bucket/shuffle"};
	FlightExchangeManager manager(config);

	ExchangeContext ctx;
	ctx.query_id = "unselected-cleanup-retry-query";
	ctx.exchange_id = "unselected-cleanup-retry-stage";
	auto exchange = manager.CreateExchange(ctx, 1);
	auto sink = exchange->AddSink(0);
	auto selected = exchange->InstantiateSink(sink, 0);
	auto unselected = exchange->InstantiateSink(sink, 1);

	auto endpoint_bearing = selected;
	endpoint_bearing.flight_host = "flight-worker.internal";
	endpoint_bearing.flight_server_epoch = "worker-epoch";
	REQUIRE_THROWS_WITH(exchange->SinkFinished(endpoint_bearing, "worker-selected", 5010),
	                    Catch::Matchers::Contains("non-local-disk Flight sink cannot publish"));
	exchange->SinkFinished(selected, "worker-selected", 0);
	REQUIRE_THROWS_WITH(exchange->SinkFinished(unselected, "worker-unselected", 0),
	                    Catch::Matchers::Contains("requires ClientContext"));

	// A thrown cleanup must release the in-flight claim. The final barrier
	// therefore retries the same attempt instead of silently treating it as
	// already cleaned.
	REQUIRE_THROWS_WITH(exchange->AllRequiredSinksFinished(), Catch::Matchers::Contains("requires ClientContext"));
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
	handle.query_id = "sink-test-query";
	handle.output_location = "sink_test_stage";
	handle.output_partition_count = 2;

	FlightExchangeSink sink(cache, handle, &context);

	// Synchronous pressure flushing does not expose an asynchronous blocked state.
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
	REQUIRE(sink.GetMemoryUsage() == 0);

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
	FlightExchangeConfig source_config;
	source_config.node_id = "node_1";
	source_config.expected_types = types;
	auto rows = ReadSourceRows(context, source_config, {MakeSourceHandle("sink_test_stage", "node_1", 0)});
	RequireTwoColumnRows(rows, ids, names);

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
	handle.query_id = "sink-memory-query";
	handle.output_location = "sink_mem_test";
	handle.output_partition_count = 1;

	FlightExchangeSink sink(cache, handle, conn.context.get());

	vector<LogicalType> types = {LogicalType::INTEGER};
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), types);
	chunk.SetCardinality(1);
	chunk.SetValue(0, 0, Value::INTEGER(42));
	REQUIRE(sink.AddChunk(0, chunk).is_ok());
	REQUIRE(sink.GetMemoryUsage() > 0);
	REQUIRE(sink.GetMemoryUsage() == cache->GetBufferedBytes());
	REQUIRE(sink.Abort().is_ok());
	REQUIRE(sink.GetMemoryUsage() == 0);
	auto cleanup = ShuffleCacheRegistry::Instance().RemoveAndCleanupByQuery(handle.query_id);
	REQUIRE(cleanup.cleanup_errors == 0);
	REQUIRE(cleanup.cleanup_pending == 0);
	REQUIRE(ShuffleCacheRegistry::Instance().RetireQuery(handle.query_id).is_ok());
}

TEST_CASE("Exchange: FlightExchangeSink abort retains partially flushed attempt cleanup", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;

	const std::string query_id = "sink-abort-query";
	const std::string exchange_id = "sink_abort__sink_0__attempt_0";
	ShuffleCacheConfig cache_config;
	cache_config.shuffle_stage_id = exchange_id;
	cache_config.node_id = "node_1";
	cache_config.num_partitions = 1;
	cache_config.local_dirs = {TestCreatePath("exchange_sink_abort")};
	auto cache = std::make_shared<ShuffleCache>(std::move(cache_config));

	ExchangeSinkInstanceHandle handle;
	handle.sink_handle.task_partition_id = 0;
	handle.attempt_id = 0;
	handle.query_id = query_id;
	handle.output_location = exchange_id;
	handle.output_partition_count = 1;

	FlightExchangeSink sink(cache, handle, &context);
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER});
	chunk.SetCardinality(1);
	chunk.SetValue(0, 0, Value::INTEGER(42));
	REQUIRE(sink.AddChunk(0, chunk).is_ok());
	REQUIRE(cache->FlushAll(context, {"value"}).is_ok());
	auto files = cache->GetPartitionFiles(0);
	REQUIRE(files.is_ok());
	REQUIRE(files.value().files.size() == 1);
	auto file_path = files.value().files[0].path;
	auto fs = FileSystem::CreateLocal();
	REQUIRE(fs->FileExists(file_path));
	REQUIRE(ShuffleCacheRegistry::Instance().Get(exchange_id) == nullptr);

	REQUIRE(sink.Abort().is_ok());
	auto cleanup = ShuffleCacheRegistry::Instance().RemoveAndCleanupByQuery(query_id);
	REQUIRE(cleanup.cleanup_errors == 0);
	REQUIRE(cleanup.cleanup_pending == 0);
	REQUIRE(cleanup.storage_entries_removed > 0);
	REQUIRE_FALSE(fs->FileExists(file_path));
	REQUIRE(ShuffleCacheRegistry::Instance().RetireQuery(query_id).is_ok());
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
	REQUIRE(cache->WriteAttemptManifest(0, 0).is_ok());

	// Register the cache
	REQUIRE(ShuffleCacheRegistry::Instance().Register("source_test_stage", cache, "source-test-query").is_ok());

	// Create source and read partition 0
	FlightExchangeConfig source_config;
	source_config.node_id = "node_1";
	FlightExchangeSource source(source_config, &context);

	REQUIRE(source.IsBlocked() == false);

	// Add source handle for partition 0
	ExchangeSourceHandle handle0;
	handle0.partition_id = 0;
	handle0.node_id = "node_1";
	handle0.files.push_back(ExchangeSourceFile("source_test_stage", 0));
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

TEST_CASE("Exchange: FlightExchangeSource accepts RecordBatches larger than a standard vector",
          "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;
	const string output_location = "source_oversized_batch";
	const string node_id = "node_oversized";
	const idx_t row_count = STANDARD_VECTOR_SIZE + 1;
	const auto file_path = TestCreatePath("exchange_source_oversized_batch.arrow");

	arrow::Int32Builder builder;
	for (idx_t row_idx = 0; row_idx < row_count; row_idx++) {
		REQUIRE(builder.Append(static_cast<int32_t>(row_idx)).ok());
	}
	std::shared_ptr<arrow::Array> values;
	REQUIRE(builder.Finish(&values).ok());
	auto schema = arrow::schema({arrow::field("value", arrow::int32())});
	auto batch = arrow::RecordBatch::Make(schema, static_cast<int64_t>(row_count), {std::move(values)});
	auto output_res = arrow::io::FileOutputStream::Open(file_path);
	REQUIRE(output_res.ok());
	auto file_output = std::move(output_res).ValueOrDie();
	auto writer_res = arrow::ipc::MakeStreamWriter(file_output, schema);
	REQUIRE(writer_res.ok());
	auto writer = std::move(writer_res).ValueOrDie();
	REQUIRE(writer->WriteRecordBatch(*batch).ok());
	REQUIRE(writer->Close().ok());
	REQUIRE(file_output->Close().ok());

	ShuffleCacheConfig cache_config;
	cache_config.shuffle_stage_id = output_location;
	cache_config.node_id = node_id;
	cache_config.num_partitions = 1;
	cache_config.local_dirs = {TestCreatePath("exchange_source_oversized_cache")};
	auto cache = std::make_shared<ShuffleCache>(std::move(cache_config));
	auto fs = FileSystem::CreateLocal();
	auto file_handle = fs->OpenFile(file_path, FileOpenFlags(FileOpenFlags::FILE_FLAGS_READ));
	ShufflePartitionFile file;
	file.path = file_path;
	file.rows = row_count;
	file.bytes = file_handle->GetFileSize();
	REQUIRE(cache->RegisterPartitionFile(0, std::move(file)).is_ok());
	REQUIRE(cache->WriteAttemptManifest(0, 0).is_ok());
	REQUIRE(ShuffleCacheRegistry::Instance().Register(output_location, cache, "source-oversized-query").is_ok());
	ScopedShuffleCacheRegistration registration(output_location);

	FlightExchangeConfig source_config;
	source_config.node_id = node_id;
	source_config.expected_types = {LogicalType::INTEGER};
	FlightExchangeSource source(source_config, &context);
	source.AddSourceHandles({MakeSourceHandle(output_location, node_id, 0)});
	vector<LogicalType> output_types = {LogicalType::INTEGER};
	DataChunk output;
	output.Initialize(Allocator::DefaultAllocator(), output_types);
	REQUIRE(output.GetCapacity() == STANDARD_VECTOR_SIZE);
	REQUIRE(source.ReadChunk(output));
	REQUIRE(output.size() == row_count);
	REQUIRE(output.GetCapacity() >= row_count);
	REQUIRE(output.GetValue(0, 0).GetValue<int32_t>() == 0);
	REQUIRE(output.GetValue(0, row_count - 1).GetValue<int32_t>() == static_cast<int32_t>(row_count - 1));
	REQUIRE_FALSE(source.ReadChunk(output));
	source.Close();
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
	REQUIRE(cache->WriteAttemptManifest(0, 0).is_ok());
	REQUIRE(ShuffleCacheRegistry::Instance().Register("source_multi_stage", cache, "source-multi-test-query").is_ok());

	// Source reads both partitions
	FlightExchangeConfig source_config;
	source_config.node_id = "node_1";
	FlightExchangeSource source(source_config, &context);

	ExchangeSourceHandle h0, h2;
	h0.partition_id = 0;
	h0.node_id = "node_1";
	h0.files.push_back(ExchangeSourceFile("source_multi_stage", 0));
	h2.partition_id = 2;
	h2.node_id = "node_1";
	h2.files.push_back(ExchangeSourceFile("source_multi_stage", 0));
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
	REQUIRE(cache0->WriteAttemptManifest(0, 0).is_ok());
	REQUIRE(ShuffleCacheRegistry::Instance().Register(stage0, cache0, "source-switch-test-query").is_ok());

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
	REQUIRE(cache1->WriteAttemptManifest(1, 0).is_ok());
	REQUIRE(ShuffleCacheRegistry::Instance().Register(stage1, cache1, "source-switch-test-query").is_ok());

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

TEST_CASE("Exchange: FlightExchangeSource revalidates local handle attempt identity", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto &context = *conn.context;
	const std::string exchange_id = "source_local_identity__sink_0__attempt_0";
	const std::string query_id = "source-local-identity-query";

	ShuffleCacheConfig cache_config;
	cache_config.shuffle_stage_id = exchange_id;
	cache_config.node_id = "node-local";
	cache_config.num_partitions = 1;
	cache_config.local_dirs = {TestCreatePath("exchange_source_local_identity")};
	auto cache = std::make_shared<ShuffleCache>(std::move(cache_config));
	DataChunk input;
	input.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER});
	input.SetCardinality(1);
	input.SetValue(0, 0, Value::INTEGER(42));
	REQUIRE(cache->WriteChunk(context, input, 0, {"value"}).is_ok());
	REQUIRE(cache->FlushAll(context, {"value"}).is_ok());
	REQUIRE(cache->WriteAttemptManifest(0, 0).is_ok());
	REQUIRE(ShuffleCacheRegistry::Instance().Register(exchange_id, cache, query_id, "", 0).is_ok());

	ExchangeSourceHandle valid;
	valid.partition_id = 0;
	valid.attempt_id = 0;
	valid.node_id = "node-local";
	valid.files.push_back(ExchangeSourceFile(exchange_id, 0));
	auto invalid = valid;
	invalid.attempt_id = 1;

	FlightExchangeConfig source_config;
	source_config.node_id = "node-local";
	source_config.expected_types = {LogicalType::INTEGER};
	FlightExchangeSource source(source_config, &context);
	source.AddSourceHandles({valid, invalid});

	DataChunk output;
	output.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER});
	REQUIRE(source.ReadChunk(output));
	REQUIRE(output.size() == 1);
	REQUIRE(output.GetValue(0, 0).GetValue<int32_t>() == 42);
	REQUIRE_THROWS_WITH(source.ReadChunk(output),
	                    Catch::Matchers::Contains("attempt id does not match the published attempt"));

	source.Close();
	ShuffleCacheRegistry::Instance().Remove(exchange_id);
}

TEST_CASE("Exchange: FlightExchangeSource no handles", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);

	FlightExchangeConfig source_config;
	FlightExchangeSource source(source_config, conn.context.get());

	// Without handles, should be finished immediately
	REQUIRE(source.IsFinished() == true);

	DataChunk chunk;
	vector<LogicalType> types = {LogicalType::INTEGER};
	chunk.Initialize(Allocator::DefaultAllocator(), types);
	REQUIRE(source.ReadChunk(chunk) == false);
}

TEST_CASE("Exchange: FlightExchangeSource rejects handles without catalog identity", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);

	FlightExchangeConfig source_config;
	source_config.node_id = "node-1";
	FlightExchangeSource source(source_config, conn.context.get());
	ExchangeSourceHandle handle;
	handle.partition_id = 0;
	handle.node_id = "node-1";
	source.AddSourceHandles({handle});

	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER});
	REQUIRE_THROWS_WITH(source.ReadChunk(chunk), Catch::Matchers::Contains("missing its catalog identity"));
}

TEST_CASE("Exchange: FlightExchangeSource close releases catalog borrow for query cleanup", "[distributed][exchange]") {
	DuckDB db(nullptr);
	Connection conn(db);

	const std::string query_id = "source-close-cleanup-query";
	const std::string exchange_id = "source_close_cleanup_attempt";
	ShuffleCacheConfig cache_config;
	cache_config.shuffle_stage_id = exchange_id;
	cache_config.node_id = "node-1";
	cache_config.num_partitions = 1;
	cache_config.local_dirs = {TestCreatePath("exchange_source_close_cleanup")};
	auto cache = std::make_shared<ShuffleCache>(std::move(cache_config));
	DataChunk input;
	input.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER});
	input.SetCardinality(1);
	input.SetValue(0, 0, Value::INTEGER(42));
	REQUIRE(cache->WriteChunk(*conn.context, input, 0, {"value"}).is_ok());
	REQUIRE(cache->FlushAll(*conn.context, {"value"}).is_ok());
	REQUIRE(cache->WriteAttemptManifest(0, 0).is_ok());
	auto &registry = ShuffleCacheRegistry::Instance();
	REQUIRE(registry.Register(exchange_id, cache, query_id).is_ok());

	FlightExchangeConfig source_config;
	source_config.node_id = "node-1";
	FlightExchangeSource source(source_config, conn.context.get());
	ExchangeSourceHandle handle;
	handle.partition_id = 0;
	handle.node_id = "node-1";
	handle.files.push_back(ExchangeSourceFile(exchange_id, 0));
	source.AddSourceHandles({handle});

	DataChunk output;
	output.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER});
	REQUIRE(source.ReadChunk(output));
	REQUIRE(output.GetValue(0, 0).GetValue<int32_t>() == 42);
	auto while_borrowed = registry.RemoveAndCleanupByQuery(query_id);
	REQUIRE(while_borrowed.cleanup_pending == 1);

	source.Close();
	auto after_close = registry.RemoveAndCleanupByQuery(query_id);
	REQUIRE(after_close.cleanup_errors == 0);
	REQUIRE(after_close.cleanup_pending == 0);
	REQUIRE(registry.RetireQuery(query_id).is_ok());
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
	handle.query_id = "exchange-e2e-query";
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

	FlightExchangeConfig source_config;
	source_config.node_id = "node_1";
	FlightExchangeSource source0(source_config, &context);
	ExchangeSourceHandle sh0;
	sh0.partition_id = 0;
	sh0.node_id = "node_1";
	sh0.files.push_back(ExchangeSourceFile(exchange_id, 0));
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

	FlightExchangeSource source1(source_config, &context);
	ExchangeSourceHandle sh1;
	sh1.partition_id = 1;
	sh1.node_id = "node_1";
	sh1.files.push_back(ExchangeSourceFile(exchange_id, 0));
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
	const std::string output0 = exchange_id + "__sink_0__attempt_0";
	const std::string output1 = exchange_id + "__sink_1__attempt_0";
	auto make_cache = [&](const std::string &output_location, const std::string &dir) {
		ShuffleCacheConfig cache_config;
		cache_config.shuffle_stage_id = output_location;
		cache_config.node_id = "node_1";
		cache_config.num_partitions = 2;
		cache_config.local_dirs = {TestCreatePath(dir)};
		return std::make_shared<ShuffleCache>(std::move(cache_config));
	};
	auto cache0 = make_cache(output0, "exchange_multi_sink_0");
	auto cache1 = make_cache(output1, "exchange_multi_sink_1");

	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::VARCHAR};

	// Sink 1: writes to partition 0
	{
		ExchangeSinkInstanceHandle handle;
		handle.sink_handle.task_partition_id = 0;
		handle.attempt_id = 0;
		handle.query_id = "exchange-multi-sink-query";
		handle.output_location = output0;
		handle.output_partition_count = 2;

		FlightExchangeSink sink1(cache0, handle, &context);
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
		handle.query_id = "exchange-multi-sink-query";
		handle.output_location = output1;
		handle.output_partition_count = 2;

		FlightExchangeSink sink2(cache1, handle, &context);
		DataChunk chunk;
		PopulateTwoColumnChunk(chunk, types, {3, 4}, {"c", "d"});
		REQUIRE(sink2.AddChunk(0, chunk).is_ok());
		REQUIRE(sink2.Finish().is_ok());
	}

	FlightExchangeConfig source_config;
	source_config.node_id = "node_1";
	FlightExchangeSource source(source_config, &context);
	ExchangeSourceHandle sh0;
	sh0.partition_id = 0;
	sh0.node_id = "node_1";
	sh0.files.push_back(ExchangeSourceFile(output0, 0));
	ExchangeSourceHandle sh1;
	sh1.partition_id = 0;
	sh1.node_id = "node_1";
	sh1.files.push_back(ExchangeSourceFile(output1, 0));
	source.AddSourceHandles({sh0, sh1});

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
	REQUIRE(read_ids.size() == 4);

	// All read IDs should be from the expected set
	std::set<int32_t> id_set(read_ids.begin(), read_ids.end());
	std::set<int32_t> expected_ids({1, 2, 3, 4});
	for (auto &id : id_set) {
		REQUIRE(expected_ids.count(id) > 0);
	}

	source.Close();
	ShuffleCacheRegistry::Instance().Remove(output0);
	ShuffleCacheRegistry::Instance().Remove(output1);
}
