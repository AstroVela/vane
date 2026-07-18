// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file flight_exchange_manager.cpp
 * @brief Concrete FlightExchange implementation — disk-first + Arrow Flight.
 *
 * Aligned with Vane's shuffle service:
 *   - Sink writes DataChunks to ShuffleCache (buffered → IPC file flush)
 *   - On Finish(), registers ShuffleCache in ShuffleCacheRegistry
 *   - Source streams selected ShuffleCache partition files batch by batch
 */

#include "duckdb/execution/distributed/exchange/flight_exchange_manager.hpp"
#include "duckdb/execution/distributed/exchange/flight_client.hpp"
#include "duckdb/execution/distributed/exchange/flight_ticket.hpp"
#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/main/client_data.hpp"
#include "duckdb/main/config.hpp"

#include <arrow/c/bridge.h>
#include <arrow/flight/api.h>
#include <arrow/ipc/api.h>

#include <algorithm>
#include <mutex>
#include <sstream>
#include <unordered_set>

namespace duckdb {
namespace distributed {

namespace {

std::mutex &LocalFlightServerMutex() {
	static std::mutex mutex;
	return mutex;
}

std::unique_ptr<FlightServer> &LocalFlightServerInstance() {
	static std::unique_ptr<FlightServer> server;
	return server;
}

std::string &LocalFlightServerNodeId() {
	static std::string node_id;
	return node_id;
}

int &LocalFlightServerPort() {
	static int port = 0;
	return port;
}

std::string BuildFlightLocation(const FlightExchangeConfig &config, const std::string &node_id) {
	auto location = config.flight_location_template;
	if (location.empty()) {
		location = "grpc://{node}:" + std::to_string(config.flight_port);
	}
	auto pos = location.find("{node}");
	if (pos != std::string::npos) {
		location.replace(pos, std::string("{node}").size(), node_id);
	}
	return location;
}

std::string BuildFlightLocationForPort(const FlightExchangeConfig &config, const std::string &node_id, int port) {
	auto effective_config = config;
	if (port > 0) {
		effective_config.flight_port = port;
		effective_config.flight_location_template.clear();
	}
	return BuildFlightLocation(effective_config, node_id);
}

bool FlightExchangeLooksLikeObjectPath(const std::string &path) {
	auto scheme_end = path.find("://");
	if (scheme_end == std::string::npos) {
		return false;
	}
	auto scheme = path.substr(0, scheme_end);
	std::transform(scheme.begin(), scheme.end(), scheme.begin(),
	               [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
	return scheme != "file";
}

bool FlightExchangeUsesObjectStorage(const FlightExchangeConfig &config) {
	for (const auto &dir : config.local_dirs) {
		if (FlightExchangeLooksLikeObjectPath(dir)) {
			return true;
		}
	}
	return false;
}

std::shared_ptr<ShuffleCache> MakeFlightExchangeShuffleCache(const ShuffleCacheConfig &cache_config,
                                                             const FlightExchangeConfig &exchange_config,
                                                             ClientContext *context) {
	if (!FlightExchangeUsesObjectStorage(exchange_config)) {
		return std::make_shared<ShuffleCache>(cache_config);
	}
	if (!context) {
		throw InvalidInputException("object-storage FlightExchange shuffle cache requires ClientContext");
	}
	auto &fs = *DBConfig::GetConfig(*context).file_system;
	auto *opener = ClientData::Get(*context).file_opener.get();
	return std::make_shared<ShuffleCache>(cache_config, MakeDuckDBFileSystemShuffleStorage(fs, opener));
}

DuckDBError FlightExchangeArrowToError(const arrow::Status &status, const std::string &context) {
	return DuckDBError::external_error(context + ": " + status.ToString());
}

bool IsFlightExchangeArrowCompatibleType(const LogicalType &arrow_type, const LogicalType &expected_type) {
	if (arrow_type == expected_type) {
		return true;
	}
	if (expected_type.id() == LogicalTypeId::AGGREGATE_STATE && arrow_type.id() == LogicalTypeId::BLOB) {
		return true;
	}
	return false;
}

void CastFlightExchangeChunk(ClientContext &context, DataChunk &input, DataChunk &output,
                             const vector<LogicalType> &target_types) {
	output.SetCardinality(input.size());
	for (idx_t col = 0; col < target_types.size(); col++) {
		if (input.data[col].GetType() == target_types[col]) {
			output.data[col].Reference(input.data[col]);
		} else {
			VectorOperations::Cast(context, input.data[col], output.data[col], input.size());
		}
	}
}

struct ArrowBatchConverter {
	unique_ptr<ArrowTableSchema> arrow_table;
	vector<LogicalType> arrow_types;
	vector<LogicalType> output_types;
	bool needs_cast = false;
};

DuckDBResult<ArrowBatchConverter> BuildArrowBatchConverter(ClientContext &context,
                                                           const std::shared_ptr<arrow::Schema> &schema,
                                                           const vector<LogicalType> &expected_types) {
	ArrowSchema c_schema;
	c_schema.Init();
	auto export_status = arrow::ExportSchema(*schema, &c_schema);
	if (!export_status.ok()) {
		return DuckDBResult<ArrowBatchConverter>::err(FlightExchangeArrowToError(export_status, "export schema"));
	}

	ArrowBatchConverter converter;
	converter.arrow_table = make_uniq<ArrowTableSchema>();
	ArrowTableFunction::PopulateArrowTableSchema(context, *converter.arrow_table, c_schema);
	if (c_schema.release) {
		c_schema.release(&c_schema);
	}

	auto &types = converter.arrow_table->GetTypes();
	converter.arrow_types.insert(converter.arrow_types.end(), types.begin(), types.end());
	if (!expected_types.empty()) {
		converter.output_types.insert(converter.output_types.end(), expected_types.begin(), expected_types.end());
		if (converter.arrow_types.size() != expected_types.size()) {
			return DuckDBResult<ArrowBatchConverter>::err(
			    DuckDBError::value_error("flight exchange partition types mismatch"));
		}
		for (idx_t idx = 0; idx < converter.arrow_types.size(); idx++) {
			if (!IsFlightExchangeArrowCompatibleType(converter.arrow_types[idx], expected_types[idx])) {
				return DuckDBResult<ArrowBatchConverter>::err(
				    DuckDBError::value_error("flight exchange partition types mismatch"));
			}
			if (converter.arrow_types[idx] != expected_types[idx]) {
				converter.needs_cast = true;
			}
		}
	} else {
		converter.output_types.insert(converter.output_types.end(), converter.arrow_types.begin(),
		                              converter.arrow_types.end());
	}

	return DuckDBResult<ArrowBatchConverter>::ok(std::move(converter));
}

DuckDBResult<void> ConvertArrowRecordBatchToChunk(ClientContext &context, const ArrowTableSchema &arrow_table,
                                                  const vector<LogicalType> &arrow_types,
                                                  const vector<LogicalType> &output_types, bool needs_cast,
                                                  const std::shared_ptr<arrow::RecordBatch> &batch, DataChunk &chunk) {
	ArrowArray c_array;
	c_array.Init();
	auto export_array_status = arrow::ExportRecordBatch(*batch, &c_array);
	if (!export_array_status.ok()) {
		return DuckDBResult<void>::err(FlightExchangeArrowToError(export_array_status, "export record batch"));
	}

	auto array_wrapper = make_uniq<ArrowArrayWrapper>();
	array_wrapper->arrow_array = c_array;
	ArrowScanLocalState scan_state(std::move(array_wrapper), context);
	scan_state.chunk_offset = 0;

	const auto row_count = static_cast<idx_t>(batch->num_rows());
	DataChunk output;
	output.Initialize(Allocator::DefaultAllocator(), arrow_types, row_count);
	output.SetCardinality(row_count);
	ArrowTableFunction::ArrowToDuckDB(scan_state, arrow_table.GetColumns(), output, 0);

	if (needs_cast) {
		DataChunk casted;
		casted.Initialize(Allocator::DefaultAllocator(), output_types, row_count);
		CastFlightExchangeChunk(context, output, casted, output_types);
		chunk.Move(casted);
	} else {
		chunk.Move(output);
	}
	return DuckDBResult<void>::ok();
}

DuckDBResult<std::unique_ptr<arrow::flight::FlightClient>> ConnectFlightExchangeClient(const std::string &location) {
	auto location_res = arrow::flight::Location::Parse(location);
	if (!location_res.ok()) {
		return DuckDBResult<std::unique_ptr<arrow::flight::FlightClient>>::err(
		    FlightExchangeArrowToError(location_res.status(), "parse flight location"));
	}
	auto client_res = arrow::flight::FlightClient::Connect(std::move(location_res).ValueOrDie());
	if (!client_res.ok()) {
		return DuckDBResult<std::unique_ptr<arrow::flight::FlightClient>>::err(
		    FlightExchangeArrowToError(client_res.status(), "connect flight client"));
	}
	return DuckDBResult<std::unique_ptr<arrow::flight::FlightClient>>::ok(std::move(client_res).ValueOrDie());
}

DuckDBResult<void> EnsureLocalFlightServerStarted(const FlightExchangeConfig &config) {
	if (config.local_dirs.empty()) {
		return DuckDBResult<void>::ok();
	}
	if (FlightExchangeUsesObjectStorage(config)) {
		return DuckDBResult<void>::ok();
	}

	auto &server_mutex = LocalFlightServerMutex();
	auto &server = LocalFlightServerInstance();
	auto &started_node_id = LocalFlightServerNodeId();
	auto &started_port = LocalFlightServerPort();
	std::lock_guard<std::mutex> guard(server_mutex);
	if (server && started_node_id == config.node_id) {
		return DuckDBResult<void>::ok();
	}
	if (server && started_node_id != config.node_id) {
		auto stop_res = server->Stop();
		server.reset();
		started_port = 0;
		if (stop_res.is_err()) {
			return stop_res;
		}
	}

	FlightServerConfig server_config;
	server_config.bind_host = config.flight_bind_host;
	server_config.port = config.flight_port;
	server_config.local_dirs = config.local_dirs;
	server = std::unique_ptr<FlightServer>(new FlightServer(std::move(server_config)));
	auto start_res = server->Start();
	if (start_res.is_err()) {
		server.reset();
		return start_res;
	}
	started_node_id = config.node_id;
	started_port = server->port();
	return DuckDBResult<void>::ok();
}

int CurrentLocalFlightServerPort() {
	auto &server_mutex = LocalFlightServerMutex();
	std::lock_guard<std::mutex> guard(server_mutex);
	return LocalFlightServerPort();
}

std::string BuildSinkOutputLocation(const ExchangeContext &ctx, const ExchangeSinkHandle &handle, idx_t attempt_id) {
	std::ostringstream ss;
	ss << ctx.exchange_id << "__sink_" << handle.task_partition_id << "__attempt_" << attempt_id;
	return ss.str();
}

} // namespace

// ─── FlightExchange ──────────────────────────────────────

FlightExchange::FlightExchange(const ExchangeContext &ctx, idx_t output_partition_count,
                               const FlightExchangeConfig &config, ClientContext *context)
    : ctx_(ctx), output_partition_count_(output_partition_count), config_(config), context_(context) {
}

FlightExchange::~FlightExchange() {
	Close();
}

ExchangeSinkHandle FlightExchange::AddSink(idx_t task_partition_id) {
	std::lock_guard<std::mutex> lock(mutex_);
	all_sinks_.push_back(task_partition_id);
	return ExchangeSinkHandle {task_partition_id};
}

ExchangeSinkInstanceHandle FlightExchange::InstantiateSink(const ExchangeSinkHandle &handle, idx_t attempt_id) {
	std::lock_guard<std::mutex> lock(mutex_);
	ExchangeSinkInstanceHandle instance;
	instance.sink_handle = handle;
	instance.attempt_id = attempt_id;
	instance.output_location = BuildSinkOutputLocation(ctx_, handle, attempt_id);
	instance.output_partition_count = output_partition_count_;
	auto &attempt_metadata = sink_attempts_[handle.task_partition_id][attempt_id];
	attempt_metadata.task_partition_id = handle.task_partition_id;
	attempt_metadata.attempt_id = attempt_id;
	attempt_metadata.output_location = instance.output_location;
	return instance;
}

void FlightExchange::SinkFinished(const ExchangeSinkHandle &handle, idx_t attempt_id) {
	SinkFinished(handle, attempt_id, std::string(), 0);
}

void FlightExchange::SinkFinished(const ExchangeSinkHandle &handle, idx_t attempt_id, const std::string &node_id,
                                  int flight_port) {
	bool cleanup_unselected = false;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		auto &attempt_metadata = sink_attempts_[handle.task_partition_id][attempt_id];
		attempt_metadata.task_partition_id = handle.task_partition_id;
		attempt_metadata.attempt_id = attempt_id;
		if (attempt_metadata.output_location.empty()) {
			attempt_metadata.output_location = BuildSinkOutputLocation(ctx_, handle, attempt_id);
		}
		if (!node_id.empty()) {
			attempt_metadata.node_id = node_id;
		}
		if (flight_port > 0) {
			attempt_metadata.flight_port = flight_port;
		}

		auto selected_entry = selected_attempts_.find(handle.task_partition_id);
		if (selected_entry == selected_attempts_.end()) {
			selected_attempts_[handle.task_partition_id] = attempt_id;
			return;
		}
		if (selected_entry->second == attempt_id) {
			return;
		}
		cleanup_unselected = true;
	}
	if (cleanup_unselected) {
		CleanupUnselectedAttempts();
	}
}

void FlightExchange::AllRequiredSinksFinished() {
	CleanupUnselectedAttempts();
}

std::vector<FlightExchange::SinkAttemptMetadata> FlightExchange::CollectUnselectedAttemptsForCleanupLocked() {
	std::vector<SinkAttemptMetadata> attempts;
	for (const auto &sink_entry : sink_attempts_) {
		auto selected_entry = selected_attempts_.find(sink_entry.first);
		if (selected_entry == selected_attempts_.end()) {
			continue;
		}
		for (const auto &attempt_entry : sink_entry.second) {
			if (attempt_entry.first == selected_entry->second) {
				continue;
			}
			const auto &attempt_metadata = attempt_entry.second;
			if (attempt_metadata.output_location.empty()) {
				continue;
			}
			if (cleaned_output_locations_.find(attempt_metadata.output_location) != cleaned_output_locations_.end()) {
				continue;
			}
			attempts.push_back(attempt_metadata);
		}
	}
	return attempts;
}

bool FlightExchange::CleanupAttemptStorage(const SinkAttemptMetadata &attempt_metadata, const char *reason) {
	if (attempt_metadata.output_location.empty()) {
		return true;
	}
	ShuffleCacheRegistry::Instance().Remove(attempt_metadata.output_location);
	if (config_.local_dirs.empty() || attempt_metadata.node_id.empty()) {
		return false;
	}

	ShuffleCacheConfig cache_config;
	cache_config.shuffle_stage_id = attempt_metadata.output_location;
	cache_config.node_id = attempt_metadata.node_id;
	cache_config.num_partitions = output_partition_count_;
	cache_config.local_dirs = config_.local_dirs;
	auto cleanup_cache = MakeFlightExchangeShuffleCache(cache_config, config_, context_);
	auto cleanup_res = cleanup_cache->RemoveAttemptStorage();
	if (cleanup_res.is_err()) {
		return false;
	}
	return true;
}

void FlightExchange::CleanupUnselectedAttempts() {
	std::vector<SinkAttemptMetadata> attempts;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		attempts = CollectUnselectedAttemptsForCleanupLocked();
	}
	for (const auto &attempt : attempts) {
		if (!CleanupAttemptStorage(attempt, "unselected")) {
			continue;
		}
		std::lock_guard<std::mutex> lock(mutex_);
		cleaned_output_locations_.insert(attempt.output_location);
	}
}

std::vector<ExchangeSourceHandle> FlightExchange::GetSourceHandles() {
	std::vector<ExchangeSourceHandle> handles;
	std::vector<std::pair<idx_t, SinkAttemptMetadata>> selected_attempts;
	bool has_sinks = false;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		has_sinks = !all_sinks_.empty();
		std::vector<std::pair<idx_t, idx_t>> selected_attempt_ids(selected_attempts_.begin(), selected_attempts_.end());
		std::sort(selected_attempt_ids.begin(), selected_attempt_ids.end(),
		          [](const std::pair<idx_t, idx_t> &lhs, const std::pair<idx_t, idx_t> &rhs) {
			          return lhs.first < rhs.first;
		          });
		for (const auto &entry : selected_attempt_ids) {
			const auto sink_partition_id = entry.first;
			const auto attempt_id = entry.second;
			SinkAttemptMetadata attempt_metadata;
			auto sink_entry = sink_attempts_.find(sink_partition_id);
			if (sink_entry != sink_attempts_.end()) {
				auto attempt_entry = sink_entry->second.find(attempt_id);
				if (attempt_entry != sink_entry->second.end()) {
					attempt_metadata = attempt_entry->second;
				}
			}
			if (attempt_metadata.output_location.empty()) {
				ExchangeSinkHandle sink_handle;
				sink_handle.task_partition_id = sink_partition_id;
				attempt_metadata.output_location = BuildSinkOutputLocation(ctx_, sink_handle, attempt_id);
			}
			attempt_metadata.task_partition_id = sink_partition_id;
			attempt_metadata.attempt_id = attempt_id;
			selected_attempts.emplace_back(sink_partition_id, std::move(attempt_metadata));
		}
	}
	if (selected_attempts.empty() && has_sinks) {
		return handles;
	}
	if (selected_attempts.empty()) {
		for (idx_t i = 0; i < output_partition_count_; i++) {
			ExchangeSourceHandle handle;
			handle.partition_id = i;
			handle.attempt_id = 0;
			handle.flight_port = config_.flight_port;
			ExchangeSourceFile file;
			file.path = ctx_.exchange_id;
			file.file_size = 0;
			handle.files.push_back(std::move(file));
			handles.push_back(std::move(handle));
		}
		return handles;
	}

	auto build_source_file = [&](const SinkAttemptMetadata &attempt_metadata, idx_t partition_id) {
		ExchangeSourceFile file;
		file.path = attempt_metadata.output_location;
		auto source_node_id = attempt_metadata.node_id.empty() ? config_.node_id : attempt_metadata.node_id;
		if (file.path.empty() || source_node_id.empty() || config_.local_dirs.empty()) {
			return file;
		}
		try {
			ShuffleCacheConfig cache_config;
			cache_config.shuffle_stage_id = attempt_metadata.output_location;
			cache_config.node_id = std::move(source_node_id);
			cache_config.num_partitions = output_partition_count_;
			cache_config.local_dirs = config_.local_dirs;
			auto manifest_cache = MakeFlightExchangeShuffleCache(cache_config, config_, context_);
			auto files_res = manifest_cache->GetPartitionFilesFromManifest(partition_id);
			if (files_res.is_ok()) {
				file.rows = files_res.value().total_rows;
				file.file_size = static_cast<size_t>(files_res.value().total_bytes);
			}
		} catch (...) {
		}
		return file;
	};

	for (idx_t partition_id = 0; partition_id < output_partition_count_; partition_id++) {
		for (const auto &entry : selected_attempts) {
			const auto attempt_id = entry.second.attempt_id;
			const auto &attempt_metadata = entry.second;
			ExchangeSourceHandle handle;
			handle.partition_id = partition_id;
			handle.attempt_id = attempt_id;
			handle.node_id = attempt_metadata.node_id;
			handle.flight_port = attempt_metadata.flight_port > 0 ? attempt_metadata.flight_port : config_.flight_port;
			handle.files.push_back(build_source_file(attempt_metadata, partition_id));
			handles.push_back(std::move(handle));
		}
	}
	return handles;
}

idx_t FlightExchange::GetNumPartitions() const {
	return output_partition_count_;
}

void FlightExchange::Close() {
	std::lock_guard<std::mutex> lock(mutex_);
	if (closed_) {
		return;
	}
	closed_ = true;
	// Clean up the ShuffleCacheRegistry entry
	ShuffleCacheRegistry::Instance().Remove(ctx_.exchange_id);
	std::unordered_set<std::string> output_locations;
	for (const auto &sink_entry : sink_attempts_) {
		for (const auto &attempt_entry : sink_entry.second) {
			if (!attempt_entry.second.output_location.empty()) {
				output_locations.insert(attempt_entry.second.output_location);
			}
		}
	}
	for (const auto &output_location : output_locations) {
		ShuffleCacheRegistry::Instance().RemoveForDeferredCleanup(output_location);
	}
}

// ─── FlightExchangeSink ─────────────────────────────────

FlightExchangeSink::FlightExchangeSink(std::shared_ptr<ShuffleCache> shuffle_cache,
                                       const ExchangeSinkInstanceHandle &handle, ClientContext *context)
    : shuffle_cache_(std::move(shuffle_cache)), handle_(handle), context_(context) {
}

FlightExchangeSink::~FlightExchangeSink() {
	if (!finished_) {
		Abort();
	}
}

DuckDBResult<void> FlightExchangeSink::AddChunk(idx_t partition_id, DataChunk &chunk) {
	if (finished_) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error("sink already finished"));
	}
	if (partition_id >= handle_.output_partition_count) {
		return DuckDBResult<void>::err(DuckDBError::value_error("partition_id " + std::to_string(partition_id) +
		                                                        " >= output_partition_count " +
		                                                        std::to_string(handle_.output_partition_count)));
	}
	if (chunk.size() == 0) {
		return DuckDBResult<void>::ok();
	}
	if (!context_) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error("sink has no client context"));
	}

	// Write chunk to ShuffleCache (buffered → IPC file on disk)
	return shuffle_cache_->WriteChunk(*context_, chunk, partition_id, shuffle_cache_->BufferedNames());
}

bool FlightExchangeSink::IsBlocked() const {
	// Disk-first: no memory backpressure needed
	return false;
}

void FlightExchangeSink::WaitUnblocked() {
	// No-op: disk writes don't block
}

DuckDBResult<void> FlightExchangeSink::Finish() {
	finished_ = true;
	if (!context_) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error("sink has no client context"));
	}
	// Flush all remaining buffered data to disk IPC files
	auto flush_res = shuffle_cache_->FlushAll(*context_, shuffle_cache_->BufferedNames());
	if (flush_res.is_err()) {
		return flush_res;
	}
	auto manifest_res = shuffle_cache_->WriteAttemptManifest(handle_.sink_handle.task_partition_id, handle_.attempt_id);
	if (manifest_res.is_err()) {
		return manifest_res;
	}
	// Register the ShuffleCache in the global registry
	// so FlightServer / FlightExchangeSource can find it
	ShuffleCacheRegistry::Instance().Register(handle_.output_location, shuffle_cache_);
	return DuckDBResult<void>::ok();
}

DuckDBResult<void> FlightExchangeSink::Abort() {
	finished_ = true;
	return DuckDBResult<void>::ok();
}

size_t FlightExchangeSink::GetMemoryUsage() const {
	return 0; // Disk-first: memory usage is transient buffer only
}

DuckDBResult<void> FlightExchangeSink::EnsureSchema(ClientContext &context, const vector<LogicalType> &types,
                                                    const vector<string> &names) {
	return shuffle_cache_->EnsureSchemaFile(context, types, names);
}

// ─── FlightExchangeSource ───────────────────────────────

struct FlightExchangeSource::PartitionStreamState {
	enum class Kind : uint8_t { LOCAL_FILES = 1, FLIGHT = 2 };

	Kind kind = Kind::LOCAL_FILES;
	vector<LogicalType> expected_types;

	std::shared_ptr<ShuffleCache> cache;
	vector<ShufflePartitionFile> files;
	idx_t file_idx = 0;
	std::shared_ptr<arrow::io::InputStream> current_input;
	std::shared_ptr<arrow::ipc::RecordBatchStreamReader> current_ipc_reader;

	std::unique_ptr<arrow::flight::FlightClient> flight_client;
	std::unique_ptr<arrow::flight::FlightStreamReader> flight_reader;

	unique_ptr<ArrowTableSchema> arrow_table;
	vector<LogicalType> arrow_types;
	vector<LogicalType> output_types;
	bool needs_cast = false;
};

FlightExchangeSource::FlightExchangeSource(const std::string &exchange_id, ClientContext *context)
    : exchange_id_(exchange_id), context_(context) {
}

FlightExchangeSource::FlightExchangeSource(const FlightExchangeConfig &config, ClientContext *context)
    : config_(config), context_(context) {
}

FlightExchangeSource::~FlightExchangeSource() {
	Close();
}

void FlightExchangeSource::AddSourceHandles(std::vector<ExchangeSourceHandle> handles) {
	// Extract exchange_id from the first handle's file path if not yet set
	if (exchange_id_.empty() && !handles.empty() && !handles[0].files.empty()) {
		exchange_id_ = handles[0].files[0].path;
	}
	handles_.insert(handles_.end(), std::make_move_iterator(handles.begin()), std::make_move_iterator(handles.end()));
}

DuckDBResult<std::unique_ptr<FlightExchangeSource::PartitionStreamState>>
FlightExchangeSource::OpenPartitionStream(const ExchangeSourceHandle &handle) {
	if (!context_) {
		return DuckDBResult<std::unique_ptr<PartitionStreamState>>::err(
		    DuckDBError::invalid_state_error("FlightExchangeSource requires ClientContext"));
	}

	auto partition_id = handle.partition_id;
	auto source_node_id = handle.node_id.empty() ? config_.node_id : handle.node_id;
	auto output_location =
	    (!handle.files.empty() && !handle.files[0].path.empty()) ? handle.files[0].path : exchange_id_;
	auto source_flight_port = handle.flight_port > 0 ? handle.flight_port : config_.flight_port;

	auto stream = make_uniq<PartitionStreamState>();
	stream->expected_types.insert(stream->expected_types.end(), config_.expected_types.begin(),
	                              config_.expected_types.end());

	DuckDBResult<ShufflePartitionFiles> files_res = DuckDBResult<ShufflePartitionFiles>::err(
	    DuckDBError::invalid_state_error("uninitialized exchange source file result"));
	std::shared_ptr<ShuffleCache> file_cache;
	bool committed_manifest_recovery_failed = false;
	const bool uses_object_storage = FlightExchangeUsesObjectStorage(config_);
	auto try_manifest_recovery = [&]() -> bool {
		if (config_.local_dirs.empty()) {
			return false;
		}
		ShuffleCacheConfig cache_config;
		cache_config.shuffle_stage_id = output_location;
		cache_config.node_id = source_node_id;
		cache_config.num_partitions = std::max<idx_t>(partition_id + 1, 1);
		cache_config.local_dirs = config_.local_dirs;
		auto manifest_cache = MakeFlightExchangeShuffleCache(cache_config, config_, context_);
		if (!manifest_cache->HasCommittedManifest()) {
			if (uses_object_storage) {
				committed_manifest_recovery_failed = true;
				files_res = DuckDBResult<ShufflePartitionFiles>::err(DuckDBError::invalid_state_error(
				    "object-storage shuffle attempt manifest is not committed: " + manifest_cache->ManifestFilePath()));
				return true;
			}
			return false;
		}
		auto manifest_res = manifest_cache->GetPartitionFilesFromManifest(partition_id);
		if (manifest_res.is_ok()) {
			files_res = std::move(manifest_res);
			file_cache = std::move(manifest_cache);
		} else {
			committed_manifest_recovery_failed = true;
			files_res = DuckDBResult<ShufflePartitionFiles>::err(manifest_res.error());
		}
		return true;
	};

	// Object storage must be replayed from the durable manifest using this
	// task's ClientContext/FileOpener, not the sink task's registry cache.
	if (!uses_object_storage && source_node_id == config_.node_id) {
		if (!cache_ || cache_key_ != output_location) {
			cache_ = ShuffleCacheRegistry::Instance().Get(output_location);
			cache_key_ = output_location;
		}
		if (cache_) {
			files_res = cache_->GetPartitionFiles(partition_id);
			if (files_res.is_ok() && files_res.value().files.empty() && cache_->HasCommittedManifest()) {
				auto manifest_res = cache_->GetPartitionFilesFromManifest(partition_id);
				if (manifest_res.is_ok()) {
					files_res = std::move(manifest_res);
				}
			}
			if (files_res.is_ok()) {
				file_cache = cache_;
			}
		}
	}
	if (files_res.is_err()) {
		try_manifest_recovery();
	}
	if (files_res.is_ok()) {
		stream->kind = PartitionStreamState::Kind::LOCAL_FILES;
		stream->cache = std::move(file_cache);
		stream->files = std::move(files_res.value().files);
		return DuckDBResult<std::unique_ptr<PartitionStreamState>>::ok(std::move(stream));
	}
	if (committed_manifest_recovery_failed || uses_object_storage) {
		const auto prefix = uses_object_storage
		                        ? "object-storage FlightExchange source failed to replay committed manifest"
		                        : "FlightExchange source failed to read selected attempt";
		return DuckDBResult<std::unique_ptr<PartitionStreamState>>::err(DuckDBError::external_error(
		    std::string(prefix) + " for exchange_id=" + output_location + " source_node_id=" + source_node_id +
		    " partition=" + std::to_string(partition_id) + ": " + files_res.error().what()));
	}

	auto location = BuildFlightLocationForPort(config_, source_node_id, source_flight_port);
	auto client_res = ConnectFlightExchangeClient(location);
	if (client_res.is_err()) {
		return DuckDBResult<std::unique_ptr<PartitionStreamState>>::err(client_res.error());
	}
	stream->flight_client = std::move(client_res.value());

	arrow::flight::Ticket flight_ticket;
	FlightExchangeTicket ticket;
	ticket.shuffle_stage_id = output_location;
	ticket.node_id = source_node_id;
	ticket.partition_idx = partition_id;
	flight_ticket.ticket = ticket.Serialize();

	arrow::flight::FlightCallOptions call_options;
	if (config_.flight_timeout_seconds > 0.0) {
		call_options.timeout = arrow::flight::TimeoutDuration(config_.flight_timeout_seconds);
	}
	auto reader_res = stream->flight_client->DoGet(call_options, flight_ticket);
	if (!reader_res.ok()) {
		return DuckDBResult<std::unique_ptr<PartitionStreamState>>::err(
		    FlightExchangeArrowToError(reader_res.status(), "flight do_get"));
	}
	stream->flight_reader = std::move(reader_res).ValueOrDie();
	auto schema_res = stream->flight_reader->GetSchema();
	if (!schema_res.ok()) {
		return DuckDBResult<std::unique_ptr<PartitionStreamState>>::err(
		    FlightExchangeArrowToError(schema_res.status(), "flight get schema"));
	}
	auto converter_res =
	    BuildArrowBatchConverter(*context_, std::move(schema_res).ValueOrDie(), stream->expected_types);
	if (converter_res.is_err()) {
		return DuckDBResult<std::unique_ptr<PartitionStreamState>>::err(converter_res.error());
	}
	auto converter = std::move(converter_res.value());
	stream->arrow_table = std::move(converter.arrow_table);
	stream->arrow_types = std::move(converter.arrow_types);
	stream->output_types = std::move(converter.output_types);
	stream->needs_cast = converter.needs_cast;
	stream->kind = PartitionStreamState::Kind::FLIGHT;
	return DuckDBResult<std::unique_ptr<PartitionStreamState>>::ok(std::move(stream));
}

DuckDBResult<bool> FlightExchangeSource::ReadStreamChunk(DataChunk &chunk) {
	if (!stream_state_) {
		return DuckDBResult<bool>::ok(false);
	}
	if (stream_state_->kind == PartitionStreamState::Kind::FLIGHT) {
		while (true) {
			auto next_res = stream_state_->flight_reader->Next();
			if (!next_res.ok()) {
				return DuckDBResult<bool>::err(FlightExchangeArrowToError(next_res.status(), "flight read batch"));
			}
			auto flight_chunk = std::move(next_res).ValueOrDie();
			if (!flight_chunk.data) {
				return DuckDBResult<bool>::ok(false);
			}
			if (flight_chunk.data->num_rows() == 0) {
				continue;
			}
			auto convert_res = ConvertArrowRecordBatchToChunk(*context_, *stream_state_->arrow_table,
			                                                  stream_state_->arrow_types, stream_state_->output_types,
			                                                  stream_state_->needs_cast, flight_chunk.data, chunk);
			if (convert_res.is_err()) {
				return DuckDBResult<bool>::err(convert_res.error());
			}
			return DuckDBResult<bool>::ok(true);
		}
	}

	while (stream_state_->file_idx < stream_state_->files.size()) {
		if (!stream_state_->current_ipc_reader) {
			if (!stream_state_->cache) {
				return DuckDBResult<bool>::err(
				    DuckDBError::invalid_state_error("FlightExchangeSource local partition stream has no cache"));
			}
			auto input_res =
			    stream_state_->cache->OpenPartitionFile(stream_state_->files[stream_state_->file_idx].path);
			if (input_res.is_err()) {
				return DuckDBResult<bool>::err(input_res.error());
			}
			stream_state_->current_input = std::move(input_res.value());
			auto reader_res = arrow::ipc::RecordBatchStreamReader::Open(stream_state_->current_input);
			if (!reader_res.ok()) {
				return DuckDBResult<bool>::err(FlightExchangeArrowToError(reader_res.status(), "open ipc reader"));
			}
			stream_state_->current_ipc_reader = std::move(reader_res).ValueOrDie();
			auto converter_res = BuildArrowBatchConverter(*context_, stream_state_->current_ipc_reader->schema(),
			                                              stream_state_->expected_types);
			if (converter_res.is_err()) {
				return DuckDBResult<bool>::err(converter_res.error());
			}
			auto converter = std::move(converter_res.value());
			stream_state_->arrow_table = std::move(converter.arrow_table);
			stream_state_->arrow_types = std::move(converter.arrow_types);
			stream_state_->output_types = std::move(converter.output_types);
			stream_state_->needs_cast = converter.needs_cast;
		}

		while (true) {
			std::shared_ptr<arrow::RecordBatch> batch;
			auto next_status = stream_state_->current_ipc_reader->ReadNext(&batch);
			if (!next_status.ok()) {
				return DuckDBResult<bool>::err(FlightExchangeArrowToError(next_status, "read ipc record batch"));
			}
			if (!batch) {
				stream_state_->current_ipc_reader.reset();
				stream_state_->current_input.reset();
				stream_state_->arrow_table.reset();
				stream_state_->arrow_types.clear();
				stream_state_->output_types.clear();
				stream_state_->needs_cast = false;
				stream_state_->file_idx++;
				break;
			}
			if (batch->num_rows() == 0) {
				continue;
			}
			auto convert_res =
			    ConvertArrowRecordBatchToChunk(*context_, *stream_state_->arrow_table, stream_state_->arrow_types,
			                                   stream_state_->output_types, stream_state_->needs_cast, batch, chunk);
			if (convert_res.is_err()) {
				return DuckDBResult<bool>::err(convert_res.error());
			}
			return DuckDBResult<bool>::ok(true);
		}
	}

	return DuckDBResult<bool>::ok(false);
}

bool FlightExchangeSource::ReadChunk(DataChunk &chunk) {
	chunk.Reset();
	if (closed_ || current_handle_idx_ >= handles_.size()) {
		return false;
	}

	while (current_handle_idx_ < handles_.size()) {
		if (!stream_state_) {
			auto stream_res = OpenPartitionStream(handles_[current_handle_idx_]);
			if (stream_res.is_err()) {
				throw std::runtime_error(stream_res.error().what());
			}
			stream_state_ = std::move(stream_res.value());
		}

		auto read_res = ReadStreamChunk(chunk);
		if (read_res.is_err()) {
			throw std::runtime_error(read_res.error().what());
		}
		if (read_res.value() && chunk.size() > 0) {
			return true;
		}

		stream_state_.reset();
		current_handle_idx_++;
	}

	return false;
}

bool FlightExchangeSource::IsBlocked() const {
	// Batch mode: data is already fully written, never blocked
	return false;
}

void FlightExchangeSource::WaitUnblocked() {
	// No-op in batch mode
}

bool FlightExchangeSource::IsFinished() const {
	if (closed_) {
		return true;
	}
	return current_handle_idx_ >= handles_.size();
}

size_t FlightExchangeSource::GetMemoryUsage() const {
	return 0;
}

void FlightExchangeSource::Close() {
	closed_ = true;
	stream_state_.reset();
}

// ─── FlightExchangeManager ─────────────────────────────

FlightExchangeManager::FlightExchangeManager(FlightExchangeConfig config, ClientContext *context)
    : config_(std::move(config)), context_(context) {
}

FlightExchangeManager::~FlightExchangeManager() {
	Shutdown();
}

void FlightExchangeManager::RefreshRuntimeNodeId() {
	auto runtime_node_id = ResolveFlightExchangeNodeIdFromEnv();
	if (runtime_node_id.empty() || runtime_node_id == config_.node_id) {
		return;
	}
	config_.node_id = std::move(runtime_node_id);
}

std::unique_ptr<Exchange> FlightExchangeManager::CreateExchange(const ExchangeContext &ctx,
                                                                idx_t output_partition_count) {
	return std::unique_ptr<Exchange>(new FlightExchange(ctx, output_partition_count, config_, context_));
}

std::unique_ptr<ExchangeSink> FlightExchangeManager::CreateSink(const ExchangeSinkInstanceHandle &handle) {
	RefreshRuntimeNodeId();
	// Create a ShuffleCache for this sink instance
	ShuffleCacheConfig cache_config;
	cache_config.shuffle_stage_id = handle.output_location;
	cache_config.node_id = config_.node_id;
	cache_config.num_partitions = handle.output_partition_count;
	cache_config.local_dirs = config_.local_dirs;

	auto server_res = EnsureLocalFlightServerStarted(config_);
	if (server_res.is_err()) {
		throw std::runtime_error(server_res.error().what());
	}
	auto shuffle_cache = MakeFlightExchangeShuffleCache(cache_config, config_, context_);
	return std::unique_ptr<ExchangeSink>(new FlightExchangeSink(shuffle_cache, handle, context_));
}

std::unique_ptr<ExchangeSource> FlightExchangeManager::CreateSource() {
	RefreshRuntimeNodeId();
	return std::unique_ptr<ExchangeSource>(new FlightExchangeSource(config_, context_));
}

int FlightExchangeManager::GetLocalFlightServerPort() {
	return CurrentLocalFlightServerPort();
}

void FlightExchangeManager::Shutdown() {
}

} // namespace distributed
} // namespace duckdb
