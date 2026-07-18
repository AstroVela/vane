// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/exchange/flight_server.hpp"

#include "duckdb/execution/distributed/exchange/flight_ticket.hpp"
#include "duckdb/execution/distributed/exchange/shuffle_cache.hpp"
#include "duckdb/execution/distributed/exchange/shuffle_cache_registry.hpp"

#include <arrow/flight/api.h>
#include <arrow/ipc/api.h>
#include <arrow/ipc/dictionary.h>
#include <arrow/io/api.h>
#include <arrow/buffer.h>
#include <algorithm>
#include <sstream>
#include <vector>

namespace duckdb {
namespace distributed {

namespace {

std::string FlightServerSanitizePathComponent(const std::string &value) {
	std::string out = value;
	for (auto &ch : out) {
		if (ch == '/' || ch == '\\') {
			ch = '_';
		}
	}
	return out;
}

DuckDBError FlightServerArrowToError(const arrow::Status &status, const std::string &context) {
	return DuckDBError::external_error(context + ": " + status.ToString());
}

DuckDBResult<arrow::flight::Location> MakeLocation(const FlightServerConfig &config) {
	if (config.bind_host.empty() || config.port < 0) {
		return DuckDBResult<arrow::flight::Location>::err(
		    DuckDBError::value_error("invalid flight server bind address"));
	}
	auto location_res = arrow::flight::Location::ForGrpcTcp(config.bind_host, config.port);
	if (!location_res.ok()) {
		return DuckDBResult<arrow::flight::Location>::err(
		    FlightServerArrowToError(location_res.status(), "create flight location"));
	}
	return DuckDBResult<arrow::flight::Location>::ok(std::move(location_res).ValueOrDie());
}

std::string NodeDirectory(const FlightServerConfig &config, const std::string &shuffle_stage_id,
                          const std::string &node_id) {
	auto stage = FlightServerSanitizePathComponent(shuffle_stage_id);
	auto node = FlightServerSanitizePathComponent(node_id);
	auto base_dir = config.local_dirs.empty() ? std::string() : config.local_dirs[0];
	std::ostringstream ss;
	ss << base_dir << "/shuffle_" << stage << "/node_" << node;
	return ss.str();
}

std::string PartitionDirectory(const FlightServerConfig &config, const std::string &shuffle_stage_id,
                               const std::string &node_id, idx_t partition_idx) {
	auto stage = FlightServerSanitizePathComponent(shuffle_stage_id);
	auto node = FlightServerSanitizePathComponent(node_id);
	auto base_dir =
	    config.local_dirs.empty() ? std::string() : config.local_dirs[partition_idx % config.local_dirs.size()];
	std::ostringstream ss;
	ss << base_dir << "/shuffle_" << stage << "/node_" << node << "/partition_" << partition_idx;
	return ss.str();
}

std::string SchemaFilePath(const FlightServerConfig &config, const std::string &shuffle_stage_id,
                           const std::string &node_id) {
	return NodeDirectory(config, shuffle_stage_id, node_id) + "/schema.arrow";
}

/// Lazy-streaming RecordBatchReader that reads batches on-demand from multiple
/// IPC Stream files. Opens files one-at-a-time, uses RecordBatchStreamReader::ReadNext()
/// for sequential iteration, keeping memory usage proportional to one batch.
class MultiFileRecordBatchReader : public arrow::RecordBatchReader {
public:
	static arrow::Result<std::shared_ptr<MultiFileRecordBatchReader>> Open(std::vector<std::string> files) {
		if (files.empty()) {
			return arrow::Status::Invalid("no files to read");
		}
		auto reader = std::make_shared<MultiFileRecordBatchReader>(std::move(files));
		ARROW_RETURN_NOT_OK(reader->Initialize());
		return reader;
	}

	explicit MultiFileRecordBatchReader(std::vector<std::string> files) : files_(std::move(files)) {
	}

	std::shared_ptr<arrow::Schema> schema() const override {
		return schema_;
	}

	arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *batch) override {
		while (true) {
			// Try to read from current stream reader
			if (current_reader_) {
				ARROW_RETURN_NOT_OK(current_reader_->ReadNext(batch));
				if (*batch) {
					if ((*batch)->num_rows() > 0) {
						return arrow::Status::OK();
					}
					// Skip empty batch, continue
					continue;
				}
				// Current stream exhausted, move to next file
				current_reader_.reset();
				current_file_idx_++;
			}

			// Open next file
			if (current_file_idx_ >= files_.size()) {
				*batch = nullptr;
				return arrow::Status::OK();
			}

			ARROW_ASSIGN_OR_RAISE(auto input, arrow::io::ReadableFile::Open(files_[current_file_idx_]));
			ARROW_ASSIGN_OR_RAISE(current_reader_, arrow::ipc::RecordBatchStreamReader::Open(std::move(input)));
		}
	}

private:
	arrow::Status Initialize() {
		ARROW_ASSIGN_OR_RAISE(auto input, arrow::io::ReadableFile::Open(files_[0]));
		ARROW_ASSIGN_OR_RAISE(auto reader, arrow::ipc::RecordBatchStreamReader::Open(std::move(input)));
		schema_ = reader->schema();
		current_reader_ = std::move(reader);
		current_file_idx_ = 0;
		return arrow::Status::OK();
	}

	std::vector<std::string> files_;
	std::shared_ptr<arrow::Schema> schema_;
	std::shared_ptr<arrow::ipc::RecordBatchStreamReader> current_reader_;
	size_t current_file_idx_ = 0;
};

/// True zero-copy FlightDataStream: reads raw IPC Stream message bytes from disk
/// and packs them directly into FlightPayload without deserializing to RecordBatch.
/// This matches Vane's FlightDataStreamReader optimization.
///
/// IPC Stream format layout:
///   [schema message] [batch1 message] [batch2 message] ... [EOS: 0-length metadata]
///
/// Each message = [continuation: 0xFFFFFFFF (4B)] [metadata_size: i32 LE (4B)]
///                [metadata flatbuf (metadata_size bytes)] [body (body_length bytes)]
/// FlightData.ipc_message = { metadata: flatbuf bytes, body_buffers: [body bytes] }
class ZeroCopyIPCStreamFlightDataStream : public arrow::flight::FlightDataStream {
public:
	static arrow::Result<std::unique_ptr<ZeroCopyIPCStreamFlightDataStream>> Open(std::vector<std::string> files) {
		if (files.empty()) {
			return arrow::Status::Invalid("no files to stream");
		}
		auto stream =
		    std::unique_ptr<ZeroCopyIPCStreamFlightDataStream>(new ZeroCopyIPCStreamFlightDataStream(std::move(files)));
		ARROW_RETURN_NOT_OK(stream->Initialize());
		return stream;
	}

	std::shared_ptr<arrow::Schema> schema() override {
		return schema_;
	}

	arrow::Result<arrow::flight::FlightPayload> GetSchemaPayload() override {
		arrow::flight::FlightPayload payload;
		// Use the schema from the first file's stream reader
		auto dict_memo = arrow::ipc::DictionaryFieldMapper(*schema_);
		ARROW_RETURN_NOT_OK(arrow::ipc::GetSchemaPayload(*schema_, arrow::ipc::IpcWriteOptions::Defaults(), dict_memo,
		                                                 &payload.ipc_message));
		return payload;
	}

	arrow::Result<arrow::flight::FlightPayload> Next() override {
		while (true) {
			if (current_input_) {
				// Read next IPC message directly from the stream
				ARROW_ASSIGN_OR_RAISE(auto message, arrow::ipc::ReadMessage(current_input_.get()));
				if (!message) {
					// End of current stream, move to next file
					current_input_.reset();
					current_file_idx_++;
				} else if (message->type() == arrow::ipc::MessageType::SCHEMA) {
					// Skip schema messages (we already sent schema via GetSchemaPayload)
					continue;
				} else {
					// Pack the raw message directly into FlightPayload
					arrow::flight::FlightPayload payload;
					payload.ipc_message.metadata = message->metadata();
					if (message->body()) {
						payload.ipc_message.body_buffers.push_back(message->body());
						payload.ipc_message.body_length = message->body()->size();
					} else {
						payload.ipc_message.body_length = 0;
					}
					payload.ipc_message.type = message->type();
					return payload;
				}
			}

			if (current_file_idx_ >= files_.size()) {
				// Signal end of stream with empty payload (null metadata)
				return arrow::flight::FlightPayload {};
			}

			ARROW_ASSIGN_OR_RAISE(current_input_, arrow::io::ReadableFile::Open(files_[current_file_idx_]));
		}
	}

	arrow::Status Close() override {
		current_input_.reset();
		return arrow::Status::OK();
	}

private:
	explicit ZeroCopyIPCStreamFlightDataStream(std::vector<std::string> files) : files_(std::move(files)) {
	}

	arrow::Status Initialize() {
		// Open first file to read schema via RecordBatchStreamReader
		ARROW_ASSIGN_OR_RAISE(auto input, arrow::io::ReadableFile::Open(files_[0]));
		ARROW_ASSIGN_OR_RAISE(auto reader, arrow::ipc::RecordBatchStreamReader::Open(input));
		schema_ = reader->schema();
		// Re-open the first file for raw message reading (reader consumed the schema)
		ARROW_ASSIGN_OR_RAISE(current_input_, arrow::io::ReadableFile::Open(files_[0]));
		current_file_idx_ = 0;
		return arrow::Status::OK();
	}

	std::vector<std::string> files_;
	std::shared_ptr<arrow::Schema> schema_;
	std::shared_ptr<arrow::io::ReadableFile> current_input_;
	size_t current_file_idx_ = 0;
};

arrow::Result<std::shared_ptr<arrow::Schema>> ReadSchemaFromFile(const std::string &path) {
	ARROW_ASSIGN_OR_RAISE(auto input, arrow::io::ReadableFile::Open(path));
	ARROW_ASSIGN_OR_RAISE(auto reader, arrow::ipc::RecordBatchStreamReader::Open(std::move(input)));
	return reader->schema();
}

} // namespace

class ShuffleFlightServer final : public arrow::flight::FlightServerBase {
public:
	explicit ShuffleFlightServer(FlightServerConfig config) : config_(std::move(config)) {
	}

	arrow::Status DoGet(const arrow::flight::ServerCallContext &, const arrow::flight::Ticket &request,
	                    std::unique_ptr<arrow::flight::FlightDataStream> *stream) override {
		auto ticket_res = FlightExchangeTicket::Parse(request.ticket);
		if (ticket_res.is_err()) {
			return arrow::Status::Invalid(ticket_res.error().what());
		}
		auto ticket = std::move(ticket_res.value());

		std::vector<std::string> files;

		// Priority 1: Look up ShuffleCacheRegistry (aligned with Vane's do_get)
		auto cache = ShuffleCacheRegistry::Instance().Get(ticket.shuffle_stage_id);
		bool visible_committed_attempt = cache != nullptr;
		if (cache) {
			auto files_res = cache->GetPartitionFiles(ticket.partition_idx);
			if (!files_res.is_err()) {
				for (auto &f : files_res.value().files) {
					files.push_back(f.path);
				}
			}
		}

		// Priority 2: durable manifest recovery (cross-process / registry-loss scenario)
		if (files.empty() && !config_.local_dirs.empty()) {
			ShuffleCacheConfig cache_config;
			cache_config.shuffle_stage_id = ticket.shuffle_stage_id;
			cache_config.node_id = ticket.node_id;
			cache_config.num_partitions = std::max<idx_t>(ticket.partition_idx + 1, 1);
			cache_config.local_dirs = config_.local_dirs;
			ShuffleCache manifest_cache(std::move(cache_config));
			if (manifest_cache.HasCommittedManifest()) {
				visible_committed_attempt = true;
				auto files_res = manifest_cache.GetPartitionFilesFromManifest(ticket.partition_idx);
				if (files_res.is_err()) {
					return arrow::Status::IOError(files_res.error().what());
				}
				for (const auto &f : files_res.value().files) {
					files.push_back(f.path);
				}
			}
		}
		std::sort(files.begin(), files.end());

		if (files.empty() && !visible_committed_attempt) {
			return arrow::Status::Invalid("flight exchange attempt is not committed: " + ticket.shuffle_stage_id);
		}
		if (files.empty()) {
			// Return empty stream with schema only
			if (config_.local_dirs.empty()) {
				return arrow::Status::Invalid("flight server local_dirs is empty and no cache found");
			}
			auto schema_path = SchemaFilePath(config_, ticket.shuffle_stage_id, ticket.node_id);
			ARROW_ASSIGN_OR_RAISE(auto schema, ReadSchemaFromFile(schema_path));
			arrow::RecordBatchVector empty;
			ARROW_ASSIGN_OR_RAISE(auto reader, arrow::RecordBatchReader::Make(empty, schema));
			*stream = std::unique_ptr<arrow::flight::RecordBatchStream>(new arrow::flight::RecordBatchStream(reader));
		} else {
			// True zero-copy: read raw IPC stream bytes directly into FlightPayload
			ARROW_ASSIGN_OR_RAISE(auto zc_stream, ZeroCopyIPCStreamFlightDataStream::Open(std::move(files)));
			*stream = std::move(zc_stream);
		}
		return arrow::Status::OK();
	}

private:
	FlightServerConfig config_;
};

class FlightServerImpl {
public:
	explicit FlightServerImpl(FlightServerConfig config) : config_(std::move(config)) {
	}

	DuckDBResult<void> Start() {
		auto location_res = MakeLocation(config_);
		if (location_res.is_err()) {
			return DuckDBResult<void>::err(location_res.error());
		}
		server_ = std::unique_ptr<ShuffleFlightServer>(new ShuffleFlightServer(config_));
		arrow::flight::FlightServerOptions options(location_res.value());
		auto status = server_->Init(options);
		if (!status.ok()) {
			return DuckDBResult<void>::err(FlightServerArrowToError(status, "init flight server"));
		}
		config_.port = server_->port();
		return DuckDBResult<void>::ok();
	}

	arrow::flight::FlightServerBase *server() const {
		return server_.get();
	}
	int port() const {
		return config_.port;
	}

private:
	FlightServerConfig config_;
	std::unique_ptr<ShuffleFlightServer> server_;
};

FlightServer::FlightServer(FlightServerConfig config)
    : config_(std::move(config)), impl_(std::unique_ptr<FlightServerImpl>(new FlightServerImpl(config_))) {
}

FlightServer::~FlightServer() {
	if (!server_thread_.joinable()) {
		return;
	}
	if (impl_ && impl_->server()) {
		auto status = impl_->server()->Shutdown();
		if (!status.ok()) {
			server_thread_.detach();
			return;
		}
	}
	server_thread_.join();
}

const FlightServerConfig &FlightServer::config() const {
	return config_;
}

int FlightServer::port() const {
	return impl_ ? impl_->port() : config_.port;
}

DuckDBResult<void> FlightServer::StartInternal() {
	if (!impl_) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error("flight server not initialized"));
	}
	if (config_.local_dirs.empty()) {
		return DuckDBResult<void>::err(DuckDBError::value_error("flight server local_dirs is empty"));
	}
	auto start_res = impl_->Start();
	if (start_res.is_err()) {
		return start_res;
	}
	config_.port = impl_->port();
	auto *server = impl_->server();
	if (!server) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error("flight server init failed"));
	}
	server_thread_ = std::thread([server]() { auto status = server->Serve(); });
	return DuckDBResult<void>::ok();
}

DuckDBResult<void> FlightServer::Start() {
	if (server_thread_.joinable()) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error("flight server already started"));
	}
	return StartInternal();
}

DuckDBResult<void> FlightServer::Stop() {
	if (!impl_ || !impl_->server()) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error("flight server not initialized"));
	}
	auto status = impl_->server()->Shutdown();
	if (!status.ok()) {
		return DuckDBResult<void>::err(FlightServerArrowToError(status, "shutdown flight server"));
	}
	if (server_thread_.joinable()) {
		server_thread_.join();
	}
	return DuckDBResult<void>::ok();
}

} // namespace distributed
} // namespace duckdb
