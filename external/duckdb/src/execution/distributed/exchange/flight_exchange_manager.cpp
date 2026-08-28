// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#endif

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
#include "duckdb/execution/distributed/exchange/flight_ticket.hpp"
#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/uuid.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/main/client_data.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"

#include <arrow/c/bridge.h>
#include <arrow/device.h>
#include <arrow/flight/api.h>
#include <arrow/flight/transport.h>
#include <arrow/ipc/api.h>
#include <arrow/util/uri.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

// Arrow 24 exports this idempotent registration hook, but does not install the
// transport-specific header that declares it.
namespace arrow {
namespace flight {
namespace transport {
namespace grpc {
ARROW_FLIGHT_EXPORT void InitializeFlightGrpcClient();
} // namespace grpc
} // namespace transport
} // namespace flight
} // namespace arrow

namespace duckdb {
namespace distributed {

namespace {

struct ParsedFlightHost {
	std::string host;
	bool is_wildcard = false;
};

int ParseFlightIpLiteral(int family, const std::string &host, void *address) {
#if defined(_WIN32)
	return InetPtonA(family, host.c_str(), address);
#else
	return inet_pton(family, host.c_str(), address);
#endif
}

DuckDBResult<ParsedFlightHost> ParseFlightHost(std::string host) {
	StringUtil::Trim(host);
	if (host.empty()) {
		return DuckDBResult<ParsedFlightHost>::ok({std::move(host), false});
	}

	const bool starts_with_bracket = host.front() == '[';
	const bool ends_with_bracket = host.back() == ']';
	if (starts_with_bracket != ends_with_bracket) {
		return DuckDBResult<ParsedFlightHost>::err(
		    DuckDBError::value_error("Flight host has mismatched brackets: " + host));
	}
	if (starts_with_bracket) {
		host = host.substr(1, host.size() - 2);
		if (host.empty()) {
			return DuckDBResult<ParsedFlightHost>::err(DuckDBError::value_error("Flight host must not be empty"));
		}
	}

	std::array<unsigned char, 16> address {};
	const auto is_zero = [](unsigned char byte) {
		return byte == 0;
	};
	if (ParseFlightIpLiteral(AF_INET, host, address.data()) == 1) {
		const bool is_unspecified = std::all_of(address.begin(), address.begin() + 4, is_zero);
		return DuckDBResult<ParsedFlightHost>::ok({std::move(host), is_unspecified});
	}
	if (ParseFlightIpLiteral(AF_INET6, host, address.data()) == 1) {
		const bool is_unspecified = std::all_of(address.begin(), address.end(), is_zero);
		const bool is_ipv4_mapped_unspecified = std::all_of(address.begin(), address.begin() + 10, is_zero) &&
		                                        address[10] == 0xff && address[11] == 0xff &&
		                                        std::all_of(address.begin() + 12, address.end(), is_zero);
		return DuckDBResult<ParsedFlightHost>::ok({std::move(host), is_unspecified || is_ipv4_mapped_unspecified});
	}
	const bool is_numeric_ipv4_like = std::all_of(host.begin(), host.end(), [](char character) {
		return (character >= '0' && character <= '9') || character == '.';
	});
	if (is_numeric_ipv4_like) {
		return DuckDBResult<ParsedFlightHost>::err(
		    DuckDBError::value_error("Flight host is not a canonical IPv4 literal: " + host));
	}
	if (starts_with_bracket || host.find(':') != std::string::npos) {
		return DuckDBResult<ParsedFlightHost>::err(
		    DuckDBError::value_error("Flight host is not a valid IP literal: " + host));
	}
	return DuckDBResult<ParsedFlightHost>::ok({std::move(host), false});
}

struct LocalFlightServiceState {
	std::mutex mutex;
	std::unique_ptr<FlightServer> server;
	bool shutting_down = false;
	std::string bind_host;
	std::string advertise_host;
	int requested_port = 0;
	int actual_port = 0;
	std::string server_epoch;
	std::chrono::milliseconds shutdown_grace_period {FlightServerConfig::DEFAULT_SHUTDOWN_GRACE_PERIOD};
};

LocalFlightServiceState &LocalFlightService() {
	// The service can call into the catalog until its server thread joins.
	// Construct the catalog first so static teardown destroys the service first.
	(void)ShuffleCacheRegistry::Instance();
	static LocalFlightServiceState service;
	return service;
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

bool FlightExchangeUsesNetworkTransport(const FlightExchangeConfig &config) {
	return !config.local_dirs.empty() && !FlightExchangeUsesObjectStorage(config);
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

// FlightClient::DoGet reads the initial schema before returning its FlightStreamReader. Use Arrow's exported
// transport stream directly so query interruption can cancel the first blocking read.
DuckDBResult<std::unique_ptr<arrow::flight::internal::ClientTransport>>
ConnectFlightExchangeTransport(const std::string &location_string) {
	auto location_res = arrow::flight::Location::Parse(location_string);
	if (!location_res.ok()) {
		return DuckDBResult<std::unique_ptr<arrow::flight::internal::ClientTransport>>::err(
		    FlightExchangeArrowToError(location_res.status(), "parse flight location"));
	}
	auto location = std::move(location_res).ValueOrDie();
	arrow::flight::transport::grpc::InitializeFlightGrpcClient();

	auto uri_res = arrow::util::Uri::FromString(location.ToString());
	if (!uri_res.ok()) {
		return DuckDBResult<std::unique_ptr<arrow::flight::internal::ClientTransport>>::err(
		    FlightExchangeArrowToError(uri_res.status(), "parse flight transport URI"));
	}
	auto transport_res = arrow::flight::internal::GetDefaultTransportRegistry()->MakeClient(location.scheme());
	if (!transport_res.ok()) {
		return DuckDBResult<std::unique_ptr<arrow::flight::internal::ClientTransport>>::err(
		    FlightExchangeArrowToError(transport_res.status(), "create flight client transport"));
	}
	auto transport = std::move(transport_res).ValueOrDie();
	auto uri = std::move(uri_res).ValueOrDie();
	auto init_status = transport->Init(arrow::flight::FlightClientOptions::Defaults(), location, uri);
	if (!init_status.ok()) {
		return DuckDBResult<std::unique_ptr<arrow::flight::internal::ClientTransport>>::err(
		    FlightExchangeArrowToError(init_status, "connect flight client transport"));
	}
	return DuckDBResult<std::unique_ptr<arrow::flight::internal::ClientTransport>>::ok(std::move(transport));
}

class FlightExchangeIpcMessageReader final : public arrow::ipc::MessageReader {
public:
	FlightExchangeIpcMessageReader(std::shared_ptr<arrow::flight::internal::ClientDataStream> stream,
	                               std::shared_ptr<arrow::MemoryManager> memory_manager)
	    : stream_(std::move(stream)), memory_manager_(std::move(memory_manager)) {
	}

	arrow::Result<std::unique_ptr<arrow::ipc::Message>> ReadNextMessage() override {
		if (finished_) {
			return nullptr;
		}
		while (true) {
			arrow::flight::internal::FlightData data;
			if (!stream_->ReadData(&data)) {
				finished_ = true;
				auto status = stream_->Finish(arrow::Status::OK());
				if (!status.ok()) {
					return status;
				}
				return nullptr;
			}
			if (!data.metadata) {
				continue;
			}
			if (data.body) {
				auto body_res = arrow::Buffer::ViewOrCopy(data.body, memory_manager_);
				if (!body_res.ok()) {
					return body_res.status();
				}
				data.body = std::move(body_res).ValueOrDie();
			}
			return data.OpenMessage();
		}
	}

private:
	std::shared_ptr<arrow::flight::internal::ClientDataStream> stream_;
	std::shared_ptr<arrow::MemoryManager> memory_manager_;
	bool finished_ = false;
};

} // namespace

class FlightStreamInterrupter {
public:
	explicit FlightStreamInterrupter(ClientContext &context)
	    : context_(context), interrupt_thread_([this]() { Run(); }) {
	}

	~FlightStreamInterrupter() {
		Stop();
	}

	void SetStream(arrow::flight::internal::ClientDataStream *stream) {
		bool cancel_stream = false;
		{
			std::lock_guard<std::mutex> guard(mutex_);
			stream_ = stream;
			if (stream_ && (cancellation_requested_ || context_.IsInterrupted())) {
				cancellation_requested_ = true;
				cancel_stream = true;
			}
		}
		condition_.notify_all();
		if (cancel_stream) {
			stream->TryCancel();
		}
	}

	void Stop() {
		{
			std::lock_guard<std::mutex> guard(mutex_);
			stopping_ = true;
			stream_ = nullptr;
		}
		condition_.notify_all();
		if (interrupt_thread_.joinable()) {
			interrupt_thread_.join();
		}
	}

private:
	static constexpr std::chrono::milliseconds INTERRUPT_POLL_INTERVAL {25};

	void Run() {
		std::unique_lock<std::mutex> guard(mutex_);
		while (!stopping_) {
			if (cancellation_requested_) {
				condition_.wait(guard, [&]() { return stopping_; });
				continue;
			}
			condition_.wait_for(guard, INTERRUPT_POLL_INTERVAL);
			if (stopping_ || !context_.IsInterrupted()) {
				continue;
			}
			cancellation_requested_ = true;
			auto *stream = stream_;
			guard.unlock();
			if (stream) {
				stream->TryCancel();
			}
			guard.lock();
		}
	}

	ClientContext &context_;
	std::mutex mutex_;
	std::condition_variable condition_;
	arrow::flight::internal::ClientDataStream *stream_ = nullptr;
	bool cancellation_requested_ = false;
	bool stopping_ = false;
	std::thread interrupt_thread_;
};

namespace {

DuckDBResult<void> EnsureLocalFlightServerStartedInternal(const FlightServiceConfig &config) {
	auto advertise_host_res = ParseFlightHost(config.advertise_host);
	if (advertise_host_res.is_err()) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error(std::string("invalid Flight advertise host: ") +
		                                                                advertise_host_res.error().what()));
	}
	auto parsed_advertise_host = std::move(advertise_host_res.value());
	auto advertise_host = std::move(parsed_advertise_host.host);
	if (advertise_host.empty()) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error("Flight advertise host is empty"));
	}
	if (parsed_advertise_host.is_wildcard || advertise_host == "*") {
		return DuckDBResult<void>::err(
		    DuckDBError::invalid_state_error("Flight advertise host must not be a wildcard address"));
	}
	auto bind_host_res = ParseFlightHost(config.bind_host);
	if (bind_host_res.is_err()) {
		return DuckDBResult<void>::err(
		    DuckDBError::invalid_state_error(std::string("invalid Flight bind host: ") + bind_host_res.error().what()));
	}
	auto bind_host = std::move(bind_host_res.value().host);
	if (bind_host.empty()) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error("Flight bind host is empty"));
	}

	auto &service = LocalFlightService();
	std::lock_guard<std::mutex> guard(service.mutex);
	if (service.shutting_down) {
		return DuckDBResult<void>::err(
		    DuckDBError::invalid_state_error("process-local Flight service is shutting down"));
	}
	if (service.server) {
		if (service.bind_host != bind_host || service.advertise_host != advertise_host ||
		    service.requested_port != config.port) {
			std::ostringstream message;
			message << "process-local Flight service already listens on requested address " << service.bind_host << ":"
			        << service.requested_port << " (actual port " << service.actual_port << ", advertised as "
			        << service.advertise_host << "); refusing conflicting address " << bind_host << ":" << config.port
			        << " advertised as " << advertise_host;
			return DuckDBResult<void>::err(DuckDBError::invalid_state_error(message.str()));
		}
		if (service.shutdown_grace_period != config.shutdown_grace_period) {
			std::ostringstream message;
			message << "process-local Flight service already uses a shutdown grace period of "
			        << service.shutdown_grace_period.count() << " ms; refusing conflicting grace period "
			        << config.shutdown_grace_period.count() << " ms";
			return DuckDBResult<void>::err(DuckDBError::invalid_state_error(message.str()));
		}
		return DuckDBResult<void>::ok();
	}

	FlightServerConfig server_config;
	server_config.bind_host = bind_host;
	server_config.port = config.port;
	server_config.server_epoch = UUID::ToString(UUID::GenerateRandomUUID());
	server_config.shutdown_grace_period = config.shutdown_grace_period;
	auto server = std::unique_ptr<FlightServer>(new FlightServer(server_config));
	auto start_res = server->Start();
	if (start_res.is_err()) {
		return start_res;
	}
	service.bind_host = std::move(bind_host);
	service.advertise_host = std::move(advertise_host);
	service.requested_port = config.port;
	service.actual_port = server->port();
	service.server_epoch = std::move(server_config.server_epoch);
	service.shutdown_grace_period = config.shutdown_grace_period;
	service.server = std::move(server);
	std::cerr << "Started plaintext cluster-internal Flight shuffle service at "
	          << BuildFlightLocation(service.advertise_host, service.actual_port)
	          << ". This endpoint provides no TLS or client authentication and must remain inside the trusted Ray "
	             "cluster network."
	          << std::endl;
	return DuckDBResult<void>::ok();
}

std::string CurrentLocalFlightServerHost() {
	auto &service = LocalFlightService();
	std::lock_guard<std::mutex> guard(service.mutex);
	return service.advertise_host;
}

int CurrentLocalFlightServerPort() {
	auto &service = LocalFlightService();
	std::lock_guard<std::mutex> guard(service.mutex);
	return service.actual_port;
}

std::string CurrentLocalFlightServerEpoch() {
	auto &service = LocalFlightService();
	std::lock_guard<std::mutex> guard(service.mutex);
	return service.server_epoch;
}

DuckDBResult<void> ShutdownLocalFlightServerInternal() {
	auto &service = LocalFlightService();
	std::unique_ptr<FlightServer> server;
	{
		std::lock_guard<std::mutex> guard(service.mutex);
		if (service.shutting_down) {
			return DuckDBResult<void>::err(
			    DuckDBError::invalid_state_error("process-local Flight service is already shutting down"));
		}
		if (!service.server) {
			return DuckDBResult<void>::ok();
		}
		service.shutting_down = true;
		server = std::move(service.server);
		service.bind_host.clear();
		service.advertise_host.clear();
		service.requested_port = 0;
		service.actual_port = 0;
		service.server_epoch.clear();
		service.shutdown_grace_period = FlightServerConfig::DEFAULT_SHUTDOWN_GRACE_PERIOD;
	}
	auto stop_res = server->Stop();
	server.reset();
	{
		std::lock_guard<std::mutex> guard(service.mutex);
		service.shutting_down = false;
	}
	return stop_res;
}

std::string BuildSinkOutputLocation(const std::string &exchange_instance_id, const ExchangeSinkHandle &handle,
                                    idx_t attempt_id) {
	std::ostringstream ss;
	ss << exchange_instance_id << "__sink_" << handle.task_partition_id << "__attempt_" << attempt_id;
	return ss.str();
}

} // namespace

FlightServiceConfig ResolveFlightServiceConfigFromEnv() {
	FlightServiceConfig config;
	config.advertise_host = ResolveFlightExchangeEnvString("VANE_FLIGHT_ADVERTISE_HOST");
	if (config.advertise_host.empty()) {
		config.advertise_host = ResolveFlightExchangeEnvString("RAY_NODE_IP_ADDRESS");
	}
	if (config.advertise_host.empty()) {
		config.advertise_host = "127.0.0.1";
	}

	auto advertise_host_res = ParseFlightHost(config.advertise_host);
	if (advertise_host_res.is_err()) {
		throw InvalidInputException("invalid VANE_FLIGHT_ADVERTISE_HOST: %s", advertise_host_res.error().what());
	}
	auto parsed_advertise_host = std::move(advertise_host_res.value());
	config.advertise_host = std::move(parsed_advertise_host.host);
	if (config.advertise_host.empty()) {
		throw InvalidInputException("VANE_FLIGHT_ADVERTISE_HOST must not be empty");
	}
	if (parsed_advertise_host.is_wildcard || config.advertise_host == "*") {
		throw InvalidInputException("VANE_FLIGHT_ADVERTISE_HOST must be a routable host, not a wildcard address");
	}

	config.bind_host = ResolveFlightExchangeEnvString("VANE_FLIGHT_BIND_HOST");
	if (config.bind_host.empty()) {
		config.bind_host = config.advertise_host;
	}
	auto bind_host_res = ParseFlightHost(config.bind_host);
	if (bind_host_res.is_err()) {
		throw InvalidInputException("invalid VANE_FLIGHT_BIND_HOST: %s", bind_host_res.error().what());
	}
	config.bind_host = std::move(bind_host_res.value().host);
	if (config.bind_host.empty()) {
		config.bind_host = config.advertise_host;
	}
	config.port = ResolveFlightExchangeEnvInt("DUCKDB_FLIGHT_PORT", 0);
	return config;
}

// ─── FlightExchange ──────────────────────────────────────

FlightExchange::FlightExchange(const ExchangeContext &ctx, idx_t output_partition_count,
                               const FlightExchangeConfig &config, ClientContext *context)
    : ctx_(ctx), output_partition_count_(output_partition_count), config_(config), context_(context),
      exchange_instance_id_(UUID::ToString(UUID::GenerateRandomUUID())) {
}

FlightExchange::~FlightExchange() {
	Close();
}

ExchangeSinkHandle FlightExchange::AddSink(idx_t task_partition_id) {
	std::lock_guard<std::mutex> lock(mutex_);
	if (closed_) {
		throw InvalidInputException("Flight exchange is closed");
	}
	if (std::find(all_sinks_.begin(), all_sinks_.end(), task_partition_id) == all_sinks_.end()) {
		all_sinks_.push_back(task_partition_id);
	}
	return ExchangeSinkHandle {task_partition_id};
}

ExchangeSinkInstanceHandle FlightExchange::InstantiateSink(const ExchangeSinkHandle &handle, idx_t attempt_id) {
	std::lock_guard<std::mutex> lock(mutex_);
	if (closed_) {
		throw InvalidInputException("Flight exchange is closed");
	}
	if (std::find(all_sinks_.begin(), all_sinks_.end(), handle.task_partition_id) == all_sinks_.end()) {
		throw InvalidInputException("Flight sink partition was not registered");
	}
	ExchangeSinkInstanceHandle instance;
	instance.sink_handle = handle;
	instance.attempt_id = attempt_id;
	instance.query_id = ctx_.query_id;
	instance.output_location = BuildSinkOutputLocation(exchange_instance_id_, handle, attempt_id);
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
	ExchangeSinkInstanceHandle instance;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (closed_) {
			throw InvalidInputException("Flight exchange is closed");
		}
		auto sink_entry = sink_attempts_.find(handle.task_partition_id);
		if (sink_entry == sink_attempts_.end()) {
			throw InvalidInputException("finished Flight sink partition was not instantiated");
		}
		auto attempt_entry = sink_entry->second.find(attempt_id);
		if (attempt_entry == sink_entry->second.end()) {
			throw InvalidInputException("finished Flight sink attempt was not instantiated");
		}
		instance.sink_handle = handle;
		instance.attempt_id = attempt_id;
		instance.query_id = ctx_.query_id;
		instance.output_location = attempt_entry->second.output_location;
		instance.output_partition_count = output_partition_count_;
	}
	SinkFinished(instance, node_id, flight_port);
}

void FlightExchange::SinkFinished(const ExchangeSinkInstanceHandle &instance, const std::string &node_id,
                                  int flight_port) {
	if (!instance.mark_join_build_summary.IsConsistent()) {
		throw InvalidInputException("finished Flight sink has an invalid MARK join build summary");
	}
	if (instance.query_id != ctx_.query_id) {
		throw InvalidInputException("finished Flight sink query does not match its exchange");
	}
	if (instance.output_partition_count != output_partition_count_) {
		throw InvalidInputException("finished Flight sink partition count does not match its exchange");
	}
	if (flight_port < 0) {
		throw InvalidInputException("finished Flight sink port must be non-negative");
	}
	auto flight_host = instance.flight_host;
	if (node_id.empty()) {
		throw InvalidInputException("finished Flight sink is missing its worker identity");
	}
	const bool uses_network_transport = FlightExchangeUsesNetworkTransport(config_);
	const bool has_flight_endpoint = !flight_host.empty() || flight_port > 0 || !instance.flight_server_epoch.empty();
	if (!uses_network_transport && has_flight_endpoint) {
		throw InvalidInputException("non-local-disk Flight sink cannot publish a Flight endpoint");
	}
	if (uses_network_transport) {
		bool flight_host_is_wildcard = false;
		if (!flight_host.empty()) {
			auto flight_host_res = ParseFlightHost(flight_host);
			if (flight_host_res.is_err()) {
				throw InvalidInputException("finished Flight sink has an invalid advertised host: %s",
				                            flight_host_res.error().what());
			}
			flight_host_is_wildcard = flight_host_res.value().is_wildcard;
			flight_host = std::move(flight_host_res.value().host);
		}
		if (flight_host.empty() || flight_port == 0 || instance.flight_server_epoch.empty()) {
			throw InvalidInputException("finished Flight sink endpoint requires host, port, and server epoch");
		}
		if (flight_host_is_wildcard || flight_host == "*") {
			throw InvalidInputException("finished Flight sink cannot advertise a wildcard host");
		}
	}
	bool cleanup_unselected = false;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (closed_) {
			throw InvalidInputException("Flight exchange is closed");
		}
		const auto &handle = instance.sink_handle;
		const auto attempt_id = instance.attempt_id;
		if (std::find(all_sinks_.begin(), all_sinks_.end(), handle.task_partition_id) == all_sinks_.end()) {
			throw InvalidInputException("finished Flight sink partition was not registered");
		}
		auto sink_entry = sink_attempts_.find(handle.task_partition_id);
		if (sink_entry == sink_attempts_.end()) {
			sink_entry =
			    sink_attempts_.emplace(handle.task_partition_id, std::unordered_map<idx_t, SinkAttemptMetadata> {})
			        .first;
		}
		auto attempt_entry = sink_entry->second.find(attempt_id);
		if (attempt_entry == sink_entry->second.end()) {
			auto expected_output_location = BuildSinkOutputLocation(exchange_instance_id_, handle, attempt_id);
			if (instance.output_location.empty() || instance.output_location != expected_output_location) {
				throw InvalidInputException("finished Flight retry output location does not match its exchange");
			}
			SinkAttemptMetadata metadata;
			metadata.task_partition_id = handle.task_partition_id;
			metadata.attempt_id = attempt_id;
			metadata.source_task_order = instance.source_task_order;
			metadata.output_location = instance.output_location;
			attempt_entry = sink_entry->second.emplace(attempt_id, std::move(metadata)).first;
		}
		auto &attempt_metadata = attempt_entry->second;
		if (attempt_metadata.source_task_order != DConstants::INVALID_INDEX &&
		    instance.source_task_order != DConstants::INVALID_INDEX &&
		    attempt_metadata.source_task_order != instance.source_task_order) {
			throw InvalidInputException("finished Flight sink source task order changed for one attempt");
		}
		if (instance.source_task_order != DConstants::INVALID_INDEX) {
			attempt_metadata.source_task_order = instance.source_task_order;
		}
		if (instance.output_location.empty() || attempt_metadata.output_location != instance.output_location) {
			throw InvalidInputException("finished Flight sink output location does not match instantiated attempt");
		}
		if (!node_id.empty() && !attempt_metadata.node_id.empty() && attempt_metadata.node_id != node_id) {
			throw InvalidInputException("finished Flight sink node does not match its published attempt");
		}
		if (!flight_host.empty() && !attempt_metadata.flight_host.empty() &&
		    attempt_metadata.flight_host != flight_host) {
			throw InvalidInputException("finished Flight sink host does not match its published attempt");
		}
		if (flight_port > 0 && attempt_metadata.flight_port > 0 && attempt_metadata.flight_port != flight_port) {
			throw InvalidInputException("finished Flight sink port does not match its published attempt");
		}
		if (!instance.flight_server_epoch.empty() && !attempt_metadata.flight_server_epoch.empty() &&
		    attempt_metadata.flight_server_epoch != instance.flight_server_epoch) {
			throw InvalidInputException("finished Flight sink epoch does not match its published attempt");
		}
		if (!node_id.empty()) {
			attempt_metadata.node_id = node_id;
		}
		if (!flight_host.empty()) {
			attempt_metadata.flight_host = flight_host;
		}
		if (flight_port > 0) {
			attempt_metadata.flight_port = flight_port;
		}
		if (!instance.flight_server_epoch.empty()) {
			attempt_metadata.flight_server_epoch = instance.flight_server_epoch;
		}
		if (attempt_metadata.mark_join_build_summary.valid && instance.mark_join_build_summary.valid &&
		    !(attempt_metadata.mark_join_build_summary == instance.mark_join_build_summary)) {
			throw InvalidInputException("finished Flight sink MARK join build summary changed for one attempt");
		}
		if (instance.mark_join_build_summary.valid) {
			attempt_metadata.mark_join_build_summary = instance.mark_join_build_summary;
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
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (closed_) {
			throw InvalidInputException("Flight exchange is closed");
		}
	}
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
			// Claim cleanup while holding the coordinator lock. Sink completion
			// and the final barrier may run concurrently, but storage ownership
			// must be exercised by only one caller at a time.
			cleaned_output_locations_.insert(attempt_metadata.output_location);
			attempts.push_back(attempt_metadata);
		}
	}
	return attempts;
}

bool FlightExchange::CleanupAttemptStorage(const SinkAttemptMetadata &attempt_metadata, const char *reason) {
	if (attempt_metadata.output_location.empty()) {
		return true;
	}
	// If the completed attempt belongs to this process, keep its storage under
	// query-scoped cleanup ownership until teardown observes a successful
	// deletion. Remote attempts are not present in this process-local catalog,
	// so the path-based cleanup below remains necessary in either case.
	ShuffleCacheRegistry::Instance().RemoveForDeferredCleanup(attempt_metadata.output_location);
	if (config_.local_dirs.empty() || attempt_metadata.node_id.empty()) {
		return false;
	}

	ShuffleCacheConfig cache_config;
	cache_config.exchange_id = attempt_metadata.output_location;
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
		bool cleaned = false;
		try {
			cleaned = CleanupAttemptStorage(attempt, "unselected");
		} catch (...) {
			std::lock_guard<std::mutex> lock(mutex_);
			cleaned_output_locations_.erase(attempt.output_location);
			throw;
		}
		if (!cleaned) {
			std::lock_guard<std::mutex> lock(mutex_);
			cleaned_output_locations_.erase(attempt.output_location);
		}
	}
}

std::vector<ExchangeSourceHandle> FlightExchange::GetSourceHandles() {
	std::vector<ExchangeSourceHandle> handles;
	std::vector<std::pair<idx_t, SinkAttemptMetadata>> selected_attempts;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (closed_) {
			throw InvalidInputException("Flight exchange is closed");
		}
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
				throw InvalidInputException("selected Flight sink attempt is missing its catalog identity");
			}
			attempt_metadata.task_partition_id = sink_partition_id;
			attempt_metadata.attempt_id = attempt_id;
			selected_attempts.emplace_back(sink_partition_id, std::move(attempt_metadata));
		}
		const bool has_source_order =
		    std::any_of(selected_attempts.begin(), selected_attempts.end(),
		                [](const auto &entry) { return entry.second.source_task_order != DConstants::INVALID_INDEX; });
		if (has_source_order) {
			if (std::any_of(selected_attempts.begin(), selected_attempts.end(), [](const auto &entry) {
				    return entry.second.source_task_order == DConstants::INVALID_INDEX;
			    })) {
				throw InvalidInputException("selected Flight sinks have inconsistent source task ordering metadata");
			}
			std::sort(selected_attempts.begin(), selected_attempts.end(), [](const auto &left, const auto &right) {
				if (left.second.source_task_order != right.second.source_task_order) {
					return left.second.source_task_order < right.second.source_task_order;
				}
				return left.first < right.first;
			});
			for (idx_t entry_idx = 1; entry_idx < selected_attempts.size(); entry_idx++) {
				if (selected_attempts[entry_idx - 1].second.source_task_order ==
				    selected_attempts[entry_idx].second.source_task_order) {
					throw InvalidInputException("selected Flight sinks have duplicate source task ordering metadata");
				}
			}
		}
	}
	if (selected_attempts.empty()) {
		return handles;
	}
	MarkJoinBuildSummary mark_join_build_summary;
	bool has_attempt_without_mark_summary = false;
	for (const auto &entry : selected_attempts) {
		const auto &attempt_summary = entry.second.mark_join_build_summary;
		if (!attempt_summary.IsConsistent()) {
			throw InvalidInputException("selected Flight sink has an invalid MARK join build summary");
		}
		if (!attempt_summary.valid) {
			has_attempt_without_mark_summary = true;
			continue;
		}
		mark_join_build_summary.Merge(attempt_summary);
	}
	if (mark_join_build_summary.valid && has_attempt_without_mark_summary) {
		throw InvalidInputException("selected Flight sink attempts have inconsistent MARK join build summaries");
	}

	auto build_source_file = [&](const SinkAttemptMetadata &attempt_metadata, idx_t partition_id) {
		ExchangeSourceFile file;
		file.path = attempt_metadata.output_location;
		if (file.path.empty() || attempt_metadata.node_id.empty() || config_.local_dirs.empty()) {
			return file;
		}
		try {
			ShuffleCacheConfig cache_config;
			cache_config.exchange_id = attempt_metadata.output_location;
			cache_config.node_id = attempt_metadata.node_id;
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
			const auto source_task_partition_id = entry.first;
			const auto attempt_id = entry.second.attempt_id;
			const auto &attempt_metadata = entry.second;
			ExchangeSourceHandle handle;
			handle.partition_id = partition_id;
			handle.source_task_partition_id = source_task_partition_id;
			handle.attempt_id = attempt_id;
			handle.node_id = attempt_metadata.node_id;
			handle.mark_join_build_summary = mark_join_build_summary;
			if (FlightExchangeUsesNetworkTransport(config_)) {
				handle.flight_host = attempt_metadata.flight_host;
				handle.flight_port = attempt_metadata.flight_port;
				handle.flight_server_epoch = attempt_metadata.flight_server_epoch;
			}
			handle.files.push_back(build_source_file(attempt_metadata, partition_id));
			handles.push_back(std::move(handle));
		}
	}
	return handles;
}

idx_t FlightExchange::GetNumPartitions() const {
	return output_partition_count_;
}

const ExchangeContext &FlightExchange::GetContext() const {
	return ctx_;
}

const std::string &FlightExchange::GetSinkOutputLocationPrefix() const {
	return exchange_instance_id_;
}

void FlightExchange::Close() {
	std::lock_guard<std::mutex> lock(mutex_);
	if (closed_) {
		return;
	}
	closed_ = true;
}

// ─── FlightExchangeSink ─────────────────────────────────

FlightExchangeSink::FlightExchangeSink(std::shared_ptr<ShuffleCache> shuffle_cache,
                                       const ExchangeSinkInstanceHandle &handle, ClientContext *context,
                                       bool descriptor_only_cleanup)
    : shuffle_cache_(std::move(shuffle_cache)), handle_(handle), context_(context) {
	auto retention = descriptor_only_cleanup ? ShuffleCacheRegistry::CacheRetention::DESCRIPTOR_ONLY
	                                         : ShuffleCacheRegistry::CacheRetention::RETAIN;
	auto track_result =
	    ShuffleCacheRegistry::Instance().TrackPending(handle_.output_location, shuffle_cache_, handle_.query_id,
	                                                  handle_.flight_server_epoch, handle_.attempt_id, retention);
	if (track_result.is_err()) {
		throw InvalidInputException("Failed to track Flight exchange sink attempt: %s", track_result.error().what());
	}
	write_lease_ = std::move(track_result.value());
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
	// AddChunk handles memory pressure by flushing retained buffers directly to shuffle storage.
	return false;
}

void FlightExchangeSink::WaitUnblocked() {
	// No-op: AddChunk does not return until any required pressure flush completes.
}

DuckDBResult<void> FlightExchangeSink::Finish() {
	if (finished_) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error("sink already finished"));
	}
	if (!context_) {
		auto result = DuckDBResult<void>::err(DuckDBError::invalid_state_error("sink has no client context"));
		Abort();
		return result;
	}
	// Flush all remaining buffered data to disk IPC files
	auto flush_res = shuffle_cache_->FlushAll(*context_, shuffle_cache_->BufferedNames());
	if (flush_res.is_err()) {
		Abort();
		return flush_res;
	}
	auto manifest_res = shuffle_cache_->WriteAttemptManifest(handle_.sink_handle.task_partition_id, handle_.attempt_id);
	if (manifest_res.is_err()) {
		Abort();
		return manifest_res;
	}
	// Publish only after both data and the committed manifest are durable.
	auto publish_result = ShuffleCacheRegistry::Instance().Publish(
	    handle_.output_location, shuffle_cache_, handle_.query_id, handle_.flight_server_epoch, handle_.attempt_id);
	if (publish_result.is_err()) {
		Abort();
		return publish_result;
	}
	write_lease_.reset();
	finished_ = true;
	return DuckDBResult<void>::ok();
}

DuckDBResult<void> FlightExchangeSink::Abort() {
	if (finished_) {
		return DuckDBResult<void>::ok();
	}
	shuffle_cache_->DiscardBufferedData();
	ShuffleCacheRegistry::Instance().RemoveForDeferredCleanup(handle_.output_location);
	write_lease_.reset();
	finished_ = true;
	return DuckDBResult<void>::ok();
}

size_t FlightExchangeSink::GetMemoryUsage() const {
	return shuffle_cache_->GetBufferedBytes();
}

DuckDBResult<void> FlightExchangeSink::EnsureSchema(ClientContext &context, const vector<LogicalType> &types,
                                                    const vector<string> &names) {
	if (finished_) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error("sink already finished"));
	}
	return shuffle_cache_->EnsureSchemaFile(context, types, names);
}

// ─── FlightExchangeSource ───────────────────────────────

struct FlightExchangeSource::PartitionStreamState {
	enum class Kind : uint8_t { LOCAL_FILES = 1, FLIGHT = 2 };

	~PartitionStreamState() {
		if (flight_interrupter) {
			flight_interrupter->Stop();
		}
		if (flight_data_stream) {
			flight_data_stream->TryCancel();
		}
		flight_reader.reset();
		flight_data_stream.reset();
		if (flight_transport) {
			(void)flight_transport->Close();
		}
	}

	Kind kind = Kind::LOCAL_FILES;
	vector<LogicalType> expected_types;

	std::shared_ptr<ShuffleCache> cache;
	vector<ShufflePartitionFile> files;
	idx_t file_idx = 0;
	std::shared_ptr<arrow::io::InputStream> current_input;
	std::shared_ptr<arrow::ipc::RecordBatchStreamReader> current_ipc_reader;

	std::unique_ptr<arrow::flight::internal::ClientTransport> flight_transport;
	std::shared_ptr<arrow::flight::internal::ClientDataStream> flight_data_stream;
	std::shared_ptr<arrow::ipc::RecordBatchStreamReader> flight_reader;
	unique_ptr<FlightStreamInterrupter> flight_interrupter;

	unique_ptr<ArrowTableSchema> arrow_table;
	vector<LogicalType> arrow_types;
	vector<LogicalType> output_types;
	bool needs_cast = false;
};

FlightExchangeSource::FlightExchangeSource(const FlightExchangeConfig &config, ClientContext *context)
    : config_(config), context_(context) {
}

FlightExchangeSource::~FlightExchangeSource() {
	Close();
}

void FlightExchangeSource::AddSourceHandles(std::vector<ExchangeSourceHandle> handles) {
	handles_.insert(handles_.end(), std::make_move_iterator(handles.begin()), std::make_move_iterator(handles.end()));
}

DuckDBResult<std::unique_ptr<FlightExchangeSource::PartitionStreamState>>
FlightExchangeSource::OpenPartitionStream(const ExchangeSourceHandle &handle) {
	if (!context_) {
		return DuckDBResult<std::unique_ptr<PartitionStreamState>>::err(
		    DuckDBError::invalid_state_error("FlightExchangeSource requires ClientContext"));
	}

	auto partition_id = handle.partition_id;
	if (handle.node_id.empty()) {
		return DuckDBResult<std::unique_ptr<PartitionStreamState>>::err(
		    DuckDBError::invalid_state_error("Flight exchange source handle is missing its producer identity"));
	}
	const auto &source_node_id = handle.node_id;
	if (handle.files.empty() || handle.files[0].path.empty()) {
		return DuckDBResult<std::unique_ptr<PartitionStreamState>>::err(
		    DuckDBError::invalid_state_error("Flight exchange source handle is missing its catalog identity"));
	}
	auto output_location = handle.files[0].path;
	auto source_flight_host = handle.flight_host;
	auto source_flight_port = handle.flight_port;
	const bool uses_object_storage = FlightExchangeUsesObjectStorage(config_);
	const bool uses_network_transport = FlightExchangeUsesNetworkTransport(config_);
	const bool has_flight_endpoint =
	    !source_flight_host.empty() || source_flight_port != 0 || !handle.flight_server_epoch.empty();
	if (!uses_network_transport && has_flight_endpoint) {
		return DuckDBResult<std::unique_ptr<PartitionStreamState>>::err(
		    DuckDBError::invalid_state_error("non-local-disk Flight source handle cannot contain a Flight endpoint"));
	}
	if (uses_network_transport) {
		if (source_flight_host.empty() || source_flight_port <= 0 || handle.flight_server_epoch.empty()) {
			return DuckDBResult<std::unique_ptr<PartitionStreamState>>::err(DuckDBError::invalid_state_error(
			    "local-disk Flight source handle requires producer identity, host, port, and server epoch"));
		}
		auto source_flight_host_res = ParseFlightHost(source_flight_host);
		if (source_flight_host_res.is_err()) {
			return DuckDBResult<std::unique_ptr<PartitionStreamState>>::err(DuckDBError::invalid_state_error(
			    std::string("Flight exchange source handle has an invalid advertised host: ") +
			    source_flight_host_res.error().what()));
		}
		const bool source_flight_host_is_wildcard = source_flight_host_res.value().is_wildcard;
		source_flight_host = std::move(source_flight_host_res.value().host);
		if (source_flight_host_is_wildcard || source_flight_host == "*") {
			return DuckDBResult<std::unique_ptr<PartitionStreamState>>::err(
			    DuckDBError::invalid_state_error("Flight exchange source handle advertises a wildcard host"));
		}
	}

	auto stream = make_uniq<PartitionStreamState>();
	stream->expected_types.insert(stream->expected_types.end(), config_.expected_types.begin(),
	                              config_.expected_types.end());

	DuckDBResult<ShufflePartitionFiles> files_res = DuckDBResult<ShufflePartitionFiles>::err(
	    DuckDBError::invalid_state_error("uninitialized exchange source file result"));
	std::shared_ptr<ShuffleCache> file_cache;
	const bool uses_process_local_registry = !uses_object_storage && source_node_id == config_.node_id;
	auto replay_object_manifest = [&]() {
		if (config_.local_dirs.empty()) {
			return;
		}
		ShuffleCacheConfig cache_config;
		cache_config.exchange_id = output_location;
		cache_config.node_id = source_node_id;
		cache_config.num_partitions = std::max<idx_t>(partition_id + 1, 1);
		cache_config.local_dirs = config_.local_dirs;
		auto manifest_cache = MakeFlightExchangeShuffleCache(cache_config, config_, context_);
		if (!manifest_cache->HasCommittedManifest()) {
			files_res = DuckDBResult<ShufflePartitionFiles>::err(DuckDBError::invalid_state_error(
			    "object-storage shuffle attempt manifest is not committed: " + manifest_cache->ManifestFilePath()));
			return;
		}
		auto manifest_res = manifest_cache->GetPartitionFilesFromManifest(partition_id);
		if (manifest_res.is_ok()) {
			files_res = std::move(manifest_res);
			file_cache = std::move(manifest_cache);
		} else {
			files_res = DuckDBResult<ShufflePartitionFiles>::err(manifest_res.error());
		}
	};

	// Object storage must be replayed from the durable manifest using this
	// task's ClientContext/FileOpener, not the sink task's registry cache.
	if (uses_process_local_registry) {
		auto cache_res = ShuffleCacheRegistry::Instance().Resolve(output_location, handle.flight_server_epoch,
		                                                          source_node_id, handle.attempt_id);
		if (cache_res.is_ok()) {
			auto resolved_cache = std::move(cache_res.value());
			if (!resolved_cache->HasCommittedManifest()) {
				files_res = DuckDBResult<ShufflePartitionFiles>::err(DuckDBError::invalid_state_error(
				    "local Flight exchange attempt is not committed: " + output_location));
			} else {
				files_res = resolved_cache->GetPartitionFiles(partition_id);
			}
			if (files_res.is_ok() && files_res.value().files.empty()) {
				auto manifest_res = resolved_cache->GetPartitionFilesFromManifest(partition_id);
				if (manifest_res.is_ok()) {
					files_res = std::move(manifest_res);
				}
			}
			if (files_res.is_ok()) {
				file_cache = std::move(resolved_cache);
			}
		} else {
			files_res = DuckDBResult<ShufflePartitionFiles>::err(cache_res.error());
		}
	}
	if (files_res.is_err() && uses_object_storage) {
		replay_object_manifest();
	}
	if (files_res.is_ok()) {
		stream->kind = PartitionStreamState::Kind::LOCAL_FILES;
		stream->cache = std::move(file_cache);
		stream->files = std::move(files_res.value().files);
		return DuckDBResult<std::unique_ptr<PartitionStreamState>>::ok(std::move(stream));
	}
	if (uses_process_local_registry) {
		return DuckDBResult<std::unique_ptr<PartitionStreamState>>::err(files_res.error());
	}
	if (uses_object_storage) {
		return DuckDBResult<std::unique_ptr<PartitionStreamState>>::err(DuckDBError::external_error(
		    "object-storage FlightExchange source failed to replay committed manifest for exchange_id=" +
		    output_location + " source_node_id=" + source_node_id + " partition=" + std::to_string(partition_id) +
		    ": " + files_res.error().what()));
	}
	auto location = BuildFlightLocation(source_flight_host, source_flight_port);
	auto transport_res = ConnectFlightExchangeTransport(location);
	if (transport_res.is_err()) {
		return DuckDBResult<std::unique_ptr<PartitionStreamState>>::err(transport_res.error());
	}
	stream->flight_transport = std::move(transport_res.value());

	arrow::flight::Ticket flight_ticket;
	FlightExchangeTicket ticket;
	ticket.server_epoch = handle.flight_server_epoch;
	ticket.exchange_instance_id = output_location;
	ticket.node_id = source_node_id;
	ticket.attempt_id = handle.attempt_id;
	ticket.partition_idx = partition_id;
	flight_ticket.ticket = ticket.Serialize();

	arrow::flight::FlightCallOptions call_options;
	if (config_.flight_timeout_seconds > 0.0) {
		call_options.timeout = arrow::flight::TimeoutDuration(config_.flight_timeout_seconds);
	}
	stream->flight_interrupter = make_uniq<FlightStreamInterrupter>(*context_);
	std::unique_ptr<arrow::flight::internal::ClientDataStream> data_stream;
	auto do_get_status = stream->flight_transport->DoGet(call_options, flight_ticket, &data_stream);
	if (data_stream) {
		stream->flight_data_stream = std::move(data_stream);
		stream->flight_interrupter->SetStream(stream->flight_data_stream.get());
	}
	if (!do_get_status.ok()) {
		return DuckDBResult<std::unique_ptr<PartitionStreamState>>::err(
		    FlightExchangeArrowToError(do_get_status, "flight do_get"));
	}
	if (!stream->flight_data_stream) {
		return DuckDBResult<std::unique_ptr<PartitionStreamState>>::err(
		    DuckDBError::external_error("flight do_get returned no data stream"));
	}
	auto memory_manager = call_options.memory_manager;
	if (!memory_manager) {
		memory_manager = arrow::CPUDevice::Instance()->default_memory_manager();
	}
	auto message_reader = std::make_unique<FlightExchangeIpcMessageReader>(stream->flight_data_stream, memory_manager);
	auto reader_res = arrow::ipc::RecordBatchStreamReader::Open(std::move(message_reader), call_options.read_options);
	if (!reader_res.ok()) {
		return DuckDBResult<std::unique_ptr<PartitionStreamState>>::err(
		    FlightExchangeArrowToError(reader_res.status(), "flight get schema"));
	}
	stream->flight_reader = std::move(reader_res).ValueOrDie();
	auto converter_res = BuildArrowBatchConverter(*context_, stream->flight_reader->schema(), stream->expected_types);
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
			std::shared_ptr<arrow::RecordBatch> batch;
			auto next_status = stream_state_->flight_reader->ReadNext(&batch);
			if (!next_status.ok()) {
				return DuckDBResult<bool>::err(FlightExchangeArrowToError(next_status, "flight read batch"));
			}
			if (!batch) {
				return DuckDBResult<bool>::ok(false);
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
	if (context_ && context_->IsInterrupted()) {
		throw InterruptException();
	}
	if (closed_ || current_handle_idx_ >= handles_.size()) {
		return false;
	}

	while (current_handle_idx_ < handles_.size()) {
		if (!stream_state_) {
			auto stream_res = OpenPartitionStream(handles_[current_handle_idx_]);
			if (stream_res.is_err()) {
				if (context_ && context_->IsInterrupted()) {
					throw InterruptException();
				}
				throw std::runtime_error(stream_res.error().what());
			}
			stream_state_ = std::move(stream_res.value());
		}

		auto read_res = ReadStreamChunk(chunk);
		if (read_res.is_err()) {
			if (context_ && context_->IsInterrupted()) {
				throw InterruptException();
			}
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
	cache_config.exchange_id = handle.output_location;
	cache_config.node_id = config_.node_id;
	cache_config.num_partitions = handle.output_partition_count;
	cache_config.local_dirs = config_.local_dirs;

	if (FlightExchangeUsesNetworkTransport(config_)) {
		auto server_res = EnsureLocalFlightServerStarted(ResolveFlightServiceConfigFromEnv());
		if (server_res.is_err()) {
			throw std::runtime_error(server_res.error().what());
		}
	}
	auto sink_handle = handle;
	sink_handle.flight_server_epoch = GetPublishedFlightServerEpoch();
	auto shuffle_cache = MakeFlightExchangeShuffleCache(cache_config, config_, context_);
	return std::unique_ptr<ExchangeSink>(
	    new FlightExchangeSink(shuffle_cache, sink_handle, context_, FlightExchangeUsesObjectStorage(config_)));
}

std::unique_ptr<ExchangeSource> FlightExchangeManager::CreateSource() {
	RefreshRuntimeNodeId();
	return std::unique_ptr<ExchangeSource>(new FlightExchangeSource(config_, context_));
}

std::string FlightExchangeManager::GetLocalFlightServerHost() {
	return CurrentLocalFlightServerHost();
}

int FlightExchangeManager::GetLocalFlightServerPort() {
	return CurrentLocalFlightServerPort();
}

std::string FlightExchangeManager::GetLocalFlightServerEpoch() {
	return CurrentLocalFlightServerEpoch();
}

std::string FlightExchangeManager::GetPublishedFlightServerHost() const {
	if (!FlightExchangeUsesNetworkTransport(config_)) {
		return std::string();
	}
	return CurrentLocalFlightServerHost();
}

int FlightExchangeManager::GetPublishedFlightServerPort() const {
	if (!FlightExchangeUsesNetworkTransport(config_)) {
		return 0;
	}
	return CurrentLocalFlightServerPort();
}

std::string FlightExchangeManager::GetPublishedFlightServerEpoch() const {
	if (!FlightExchangeUsesNetworkTransport(config_)) {
		return std::string();
	}
	return CurrentLocalFlightServerEpoch();
}

DuckDBResult<void> FlightExchangeManager::EnsureLocalFlightServerStarted(const FlightServiceConfig &config) {
	return EnsureLocalFlightServerStartedInternal(config);
}

DuckDBResult<void> FlightExchangeManager::ShutdownLocalFlightServer() {
	return ShutdownLocalFlightServerInternal();
}

void FlightExchangeManager::Shutdown() {
	// Exchange managers are session-scoped users of the process-local service.
	// The worker runtime owns service shutdown explicitly.
}

} // namespace distributed
} // namespace duckdb
