// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "direct_flight.hpp"
#include "arrow_frame.hpp"

#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/main/client_properties.hpp"

#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <arrow/flight/api.h>
#include <arrow/ipc/api.h>
#include <arrow/ipc/dictionary.h>
#include <arrow/ipc/writer.h>
#include <arrow/util/cancel.h>
#include <arrow/util/byte_size.h>
#include <arrow/util/uri.h>

#include <chrono>
#include <condition_variable>
#include <thread>

namespace duckdb {
namespace vane_execution {
namespace {

using Clock = std::chrono::steady_clock;

void Check(const arrow::Status &status, const string &operation = "") {
	if (!status.ok()) {
		throw IOException("direct Flight%s: %s", operation.empty() ? "" : " " + operation, status.ToString());
	}
}

template <class T>
T Unwrap(arrow::Result<T> value, const string &operation = "") {
	Check(value.status(), operation);
	return std::move(value).ValueOrDie();
}

struct Signal : public std::enable_shared_from_this<Signal> {
	std::mutex lock;
	std::condition_variable changed;
	void Wait() {
		std::unique_lock<std::mutex> guard(lock);
		changed.wait_for(guard, std::chrono::milliseconds(10));
	}
	ExchangeWakeup Wakeup() {
		std::weak_ptr<Signal> weak = shared_from_this();
		return [weak]() {
			if (auto signal = weak.lock()) {
				signal->changed.notify_all();
			}
		};
	}
};

struct Endpoint {
	shared_ptr<DirectChannel> channel;
	string consumer;
	std::mutex lock;
	shared_ptr<DirectBatch> lease;
	idx_t sent = 0;
	idx_t acknowledged = 0;
	bool opened = false;
	bool active_stream = false;
	std::condition_variable idle;
	bool finished = false;
	bool closed = false;
	std::shared_ptr<Signal> signal = std::make_shared<Signal>();
};

struct Registry {
	std::mutex lock;
	std::map<string, std::shared_ptr<Endpoint>> endpoints;
	bool stopped = false;
	std::shared_ptr<Endpoint> Find(const string &ticket) {
		std::lock_guard<std::mutex> guard(lock);
		auto entry = endpoints.find(ticket);
		if (stopped || entry == endpoints.end()) {
			throw InvalidInputException("unknown or expired direct Flight capability");
		}
		return entry->second;
	}
};

class Stream : public arrow::flight::FlightDataStream {
public:
	Stream(std::shared_ptr<Endpoint> endpoint_p, const arrow::flight::ServerCallContext &context_p)
	    : endpoint(std::move(endpoint_p)), context(context_p), arrow_schema(ArrowSchemaFor(endpoint->channel->types)) {
	}
	~Stream() override {
		(void)Close();
	}
	std::shared_ptr<arrow::Schema> schema() override {
		return arrow_schema;
	}
	arrow::Result<arrow::flight::FlightPayload> GetSchemaPayload() override {
		arrow::flight::FlightPayload payload;
		arrow::ipc::DictionaryFieldMapper mapper(*arrow_schema);
		ARROW_RETURN_NOT_OK(arrow::ipc::GetSchemaPayload(*arrow_schema, arrow::ipc::IpcWriteOptions::Defaults(), mapper,
		                                                 &payload.ipc_message));
		return payload;
	}
	arrow::Result<arrow::flight::FlightPayload> Next() override {
		try {
			if (sent_finish) {
				return arrow::flight::FlightPayload {};
			}
			while (!context.is_cancelled()) {
				auto snapshot = endpoint->channel->Snapshot();
				if (!snapshot.error.empty()) {
					throw IOException("%s", snapshot.error);
				}
				bool pending;
				{
					std::lock_guard<std::mutex> guard(endpoint->lock);
					if (endpoint->closed) {
						return arrow::Status::Cancelled("direct consumer closed");
					}
					pending = bool(endpoint->lease);
				}
				if (pending) {
					endpoint->signal->Wait();
					continue;
				}
				shared_ptr<DirectBatch> batch;
				auto state = endpoint->channel->Poll(endpoint->consumer, batch, endpoint->signal->Wakeup());
				if (state == DirectRead::BLOCKED) {
					endpoint->signal->Wait();
					continue;
				}
				if (state == DirectRead::CLOSED) {
					return arrow::Status::Cancelled("direct consumer closed");
				}
				DataChunk chunk;
				string metadata;
				{
					std::lock_guard<std::mutex> guard(endpoint->lock);
					if (endpoint->closed) {
						return arrow::Status::Cancelled("direct consumer closed");
					}
					if (state == DirectRead::END) {
						endpoint->finished = sent_finish = true;
						chunk.Initialize(Allocator::DefaultAllocator(), endpoint->channel->types, 0);
						metadata = "F:" + std::to_string(endpoint->sent);
					} else {
						endpoint->lease = batch;
						batch->Reference(chunk);
						metadata = "D:" + std::to_string(++endpoint->sent);
					}
				}
				auto record = Encode(chunk, arrow_schema);
				arrow::flight::FlightPayload payload;
				Check(arrow::ipc::GetRecordBatchPayload(*record, arrow::ipc::IpcWriteOptions::Defaults(),
				                                        &payload.ipc_message));
				payload.app_metadata = arrow::Buffer::FromString(metadata);
				return payload;
			}
			return arrow::Status::Cancelled("direct Flight read canceled");
		} catch (const std::exception &error) {
			endpoint->channel->Abort(error.what());
			return arrow::Status::IOError(error.what());
		}
	}
	arrow::Status Close() override {
		if (!sent_finish) {
			bool closed;
			{
				std::lock_guard<std::mutex> guard(endpoint->lock);
				closed = endpoint->closed;
			}
			if (!closed) {
				endpoint->channel->Abort("direct Flight disconnected before FINISH");
			}
		}
		{
			std::lock_guard<std::mutex> guard(endpoint->lock);
			endpoint->active_stream = false;
			endpoint->idle.notify_all();
		}
		return arrow::Status::OK();
	}

private:
	std::shared_ptr<Endpoint> endpoint;
	const arrow::flight::ServerCallContext &context;
	std::shared_ptr<arrow::Schema> arrow_schema;
	bool sent_finish = false;
};

class Server : public arrow::flight::FlightServerBase {
public:
	explicit Server(std::shared_ptr<Registry> registry_p) : registry(std::move(registry_p)) {
	}
	arrow::Status DoGet(const arrow::flight::ServerCallContext &context, const arrow::flight::Ticket &request,
	                    std::unique_ptr<arrow::flight::FlightDataStream> *stream) override {
		try {
			auto endpoint = registry->Find(request.ticket);
			std::lock_guard<std::mutex> guard(endpoint->lock);
			if (endpoint->opened || endpoint->closed) {
				return arrow::Status::Invalid("direct Flight stream cannot be replayed");
			}
			stream->reset(new Stream(endpoint, context));
			endpoint->opened = endpoint->active_stream = true;
			return arrow::Status::OK();
		} catch (const std::exception &error) {
			return arrow::Status::Invalid(error.what());
		}
	}
	arrow::Status DoAction(const arrow::flight::ServerCallContext &, const arrow::flight::Action &action,
	                       std::unique_ptr<arrow::flight::ResultStream> *result) override {
		try {
			if (!action.body || action.body->size() > 65536) {
				return arrow::Status::Invalid("invalid direct Flight control envelope");
			}
			auto body = action.body->ToString();
			auto split = body.find('\n');
			if (split == string::npos) {
				return arrow::Status::Invalid("invalid direct Flight control envelope");
			}
			auto endpoint = registry->Find(body.substr(split + 1));
			auto argument = body.substr(0, split);
			auto snapshot = endpoint->channel->Snapshot();
			if (!snapshot.error.empty()) {
				return arrow::Status::IOError(snapshot.error);
			}
			shared_ptr<DirectBatch> release;
			string response;
			{
				std::lock_guard<std::mutex> guard(endpoint->lock);
				if (action.type == "vane.direct.ack") {
					idx_t consumed = 0;
					auto sequence = std::stoull(argument, &consumed);
					if (consumed != argument.size() || sequence > endpoint->sent || argument.empty() ||
					    argument[0] == '-') {
						return arrow::Status::Invalid("invalid direct cumulative ACK");
					}
					if (sequence > endpoint->acknowledged) {
						endpoint->acknowledged = sequence;
						release = std::move(endpoint->lease);
					}
				} else if (action.type == "vane.direct.close") {
					endpoint->closed = true;
					release = std::move(endpoint->lease);
				} else if (action.type != "vane.direct.status") {
					return arrow::Status::Invalid("unknown direct Flight action");
				}
				response = endpoint->closed ? "closed" : endpoint->finished ? "finished" : "running";
			}
			release.reset();
			if (action.type == "vane.direct.close") {
				endpoint->channel->CloseConsumer(endpoint->consumer);
			}
			endpoint->signal->changed.notify_all();
			std::vector<arrow::flight::Result> values;
			values.push_back({arrow::Buffer::FromString(response)});
			result->reset(new arrow::flight::SimpleResultStream(std::move(values)));
			return arrow::Status::OK();
		} catch (const std::exception &error) {
			return arrow::Status::Invalid(error.what());
		}
	}

private:
	std::shared_ptr<Registry> registry;
};

struct Link {
	shared_ptr<DirectChannel> channel;
	string producer;
	string ticket;
	std::unique_ptr<arrow::flight::FlightClient> data;
	std::unique_ptr<arrow::flight::FlightClient> control;
	arrow::StopSource stop;
	std::thread reader;
	std::thread watcher;
	std::atomic<bool> done {false};
	std::atomic<bool> ready {false};
	std::shared_ptr<Signal> signal = std::make_shared<Signal>();

	string Action(const string &name, const string &argument = "") {
		arrow::flight::FlightCallOptions options;
		options.timeout = arrow::flight::TimeoutDuration(2);
		arrow::flight::Action action {"vane.direct." + name, arrow::Buffer::FromString(argument + "\n" + ticket)};
		const auto operation = "control " + name;
		auto response = Unwrap(control->DoAction(options, action), operation);
		auto item = Unwrap(response->Next(), operation);
		if (!item || !item->body) {
			throw IOException("direct Flight control reply is missing");
		}
		auto value = item->body->ToString();
		Check(response->Drain(), operation);
		return value;
	}
	void Stop() {
		done = true;
		stop.RequestStop();
		signal->changed.notify_all();
	}
	void Fail(const string &reason) {
		channel->Abort(reason);
		Stop();
	}
	void Read(double timeout, idx_t wire_limit) {
		try {
			arrow::flight::FlightCallOptions options;
			options.timeout = arrow::flight::TimeoutDuration(timeout);
			options.stop_token = stop.token();
			auto input = Unwrap(data->DoGet(options, arrow::flight::Ticket {ticket}), "data open");
			auto schema = Unwrap(input->GetSchema(), "data schema");
			if (!schema->Equals(*ArrowSchemaFor(channel->types))) {
				throw IOException("direct Flight schema mismatch");
			}
			ready = true;
			idx_t sequence = 0;
			bool finished = false;
			while (!done) {
				auto message = Unwrap(input->Next(), "data next");
				if (!message.data) {
					if (!finished || message.app_metadata) {
						throw IOException("direct Flight EOF without FINISH");
					}
					return; // The watcher continues checking persistent errors after EOF.
				}
				if (finished || !message.app_metadata) {
					throw IOException("invalid direct Flight frame ordering");
				}
				auto metadata = message.app_metadata->ToString();
				if (metadata == "F:" + std::to_string(sequence) && message.data->num_rows() == 0) {
					channel->Finish(producer, sequence);
					finished = true;
					continue;
				}
				if (metadata != "D:" + std::to_string(sequence + 1) || message.data->num_rows() <= 0 ||
				    idx_t(message.data->num_rows()) > channel->limits.frame_rows ||
				    arrow::util::TotalBufferSize(*message.data) > int64_t(wire_limit)) {
					throw IOException("invalid direct Flight sequence or frame size");
				}
				Check(message.data->ValidateFull());
				DataChunk chunk;
				Decode(*message.data, channel->types, chunk);
				vector<idx_t> rows;
				for (idx_t row = 0; row < chunk.size(); row++) {
					rows.push_back(row);
				}
				if (channel->FrameRows(chunk, rows, 0) != rows.size()) {
					throw IOException("direct Flight frame exceeds reserved native frame");
				}
				while (!done) {
					auto state =
					    channel->TryWrite(producer, sequence + 1, chunk, rows, 0, rows.size(), signal->Wakeup());
					if (state == DirectWrite::CLOSED) {
						Action("close");
						Stop();
						return;
					}
					if (state == DirectWrite::ACCEPTED) {
						break;
					}
					signal->Wait();
				}
				chunk.Destroy();
				message.data.reset();
				if (!done) {
					Action("ack", std::to_string(++sequence));
				}
			}
		} catch (const std::exception &error) {
			if (!done) {
				Fail(error.what());
			}
		}
	}
	void Watch() {
		try {
			while (!done) {
				if (!channel->Snapshot().error.empty()) {
					Stop();
					return;
				}
				if (!channel->HasConsumers()) {
					Action("close");
					Stop();
					return;
				}
				if (Action("status") == "closed") {
					throw IOException("upstream direct Flight endpoint closed");
				}
				std::unique_lock<std::mutex> guard(signal->lock);
				signal->changed.wait_for(guard, std::chrono::milliseconds(25));
			}
		} catch (const std::exception &error) {
			if (!done) {
				Fail(error.what());
			}
		}
	}
};

} // namespace

struct DirectFlight::Impl {
	std::shared_ptr<Registry> registry = std::make_shared<Registry>();
	Server server {registry};
	std::thread serving;
	std::mutex lock;
	vector<std::shared_ptr<Link>> links;
	const idx_t max_links;
	const idx_t frame_bytes;
	const idx_t wire_limit;
	string location;
	bool closed = false;
	bool canceled = false;

	Impl(const string &host, const string &advertise, idx_t maximum, idx_t staging, idx_t frame, int port,
	     const string &certificate, const string &private_key)
	    : max_links(maximum), frame_bytes(frame), wire_limit(DirectFlight::StagingPerLink(frame) / 4) {
		if (!maximum || maximum > 4096 || staging / DirectFlight::StagingPerLink(frame) < maximum || host.empty() ||
		    advertise.empty() || port < 0 || port > 65535 || certificate.empty() != private_key.empty()) {
			throw InvalidInputException("direct Flight requires finite links and reserved staging capacity");
		}
		auto address = Unwrap(certificate.empty() ? arrow::flight::Location::ForGrpcTcp(host, port)
		                                          : arrow::flight::Location::ForGrpcTls(host, port));
		arrow::flight::FlightServerOptions options(address);
		if (!certificate.empty()) {
			options.tls_certificates.push_back({certificate, private_key});
		}
		Check(server.Init(options));
		location = Unwrap(certificate.empty() ? arrow::flight::Location::ForGrpcTcp(advertise, server.port())
		                                      : arrow::flight::Location::ForGrpcTls(advertise, server.port()))
		               .ToString();
		serving = std::thread([this]() { (void)server.Serve(); });
	}
};

idx_t DirectFlight::StagingPerLink(idx_t frame_bytes) {
	if (!frame_bytes || frame_bytes > (1ULL << 28)) {
		throw InvalidInputException("direct Flight frame_bytes must be between 1 and 256 MiB");
	}
	// Native/Arrow encode or decode buffers plus IPC and bounded gRPC payload.
	// Fixed allowance covers schema/metadata and per-column aligned padding.
	return 16 * frame_bytes + (1 << 18);
}

void DirectFlight::ExportSchema(const vector<LogicalType> &types, const vector<string> &names, ArrowSchema *out) {
	if (types.size() != names.size()) {
		throw InvalidInputException("direct Flight schema name count mismatch");
	}
	Check(arrow::ExportSchema(*ResultSchemaFor(types, names), out));
}

void DirectFlight::ExportBatch(DataChunk &chunk, const vector<string> &names, ArrowArray *array, ArrowSchema *schema) {
	Check(arrow::ExportRecordBatch(*Encode(chunk, ResultSchemaFor(chunk.GetTypes(), names)), array, schema));
}

DirectFlight::DirectFlight(const string &host, const string &advertise, idx_t maximum, idx_t staging, idx_t frame,
                           int port, const string &certificate, const string &private_key)
    : impl(make_uniq<Impl>(host, advertise, maximum, staging, frame, port, certificate, private_key)) {
}

DirectFlight::~DirectFlight() {
	Close();
}

string DirectFlight::Location() const {
	return impl->location;
}

void DirectFlight::Publish(const string &ticket, shared_ptr<DirectChannel> channel, const string &consumer) {
	std::lock_guard<std::mutex> guard(impl->lock);
	std::lock_guard<std::mutex> registry_guard(impl->registry->lock);
	if (impl->closed || impl->canceled || ticket.empty() || ticket.size() > 32768 ||
	    ticket.find('\n') != string::npos || impl->links.size() + impl->registry->endpoints.size() >= impl->max_links ||
	    channel->limits.frame_bytes != impl->frame_bytes || channel->types.size() > 256) {
		throw InvalidInputException("invalid direct Flight publication or exhausted reservation");
	}
	channel->ValidateConsumer(consumer);
	auto endpoint = std::make_shared<Endpoint>();
	endpoint->channel = std::move(channel);
	endpoint->consumer = consumer;
	if (!impl->registry->endpoints.emplace(ticket, endpoint).second) {
		throw InvalidInputException("direct Flight ticket already published");
	}
}

void DirectFlight::Subscribe(const string &location, const string &ticket, shared_ptr<DirectChannel> channel,
                             const string &producer, double timeout, const string &root_certificates) {
	std::lock_guard<std::mutex> guard(impl->lock);
	std::lock_guard<std::mutex> registry_guard(impl->registry->lock);
	if (impl->closed || impl->canceled || impl->links.size() + impl->registry->endpoints.size() >= impl->max_links ||
	    channel->limits.frame_bytes != impl->frame_bytes || channel->types.size() > 256 || !(timeout > 0)) {
		throw InvalidInputException("invalid direct Flight subscription or exhausted reservation");
	}
	channel->ProducerStatus(producer);
	auto link = std::make_shared<Link>();
	link->channel = std::move(channel);
	link->producer = producer;
	link->ticket = ticket;
	auto address = Unwrap(arrow::flight::Location::Parse(location));
	auto options = arrow::flight::FlightClientOptions::Defaults();
	options.tls_root_certs = root_certificates;
	options.generic_options.emplace_back("grpc.max_receive_message_length", int(impl->wire_limit));
	link->data = Unwrap(arrow::flight::FlightClient::Connect(address, options));
	link->control = Unwrap(arrow::flight::FlightClient::Connect(address, options));
	impl->links.push_back(link);
	auto limit = impl->wire_limit;
	link->reader = std::thread([link, timeout, limit]() { link->Read(timeout, limit); });
	link->watcher = std::thread([link]() { link->Watch(); });
}

void DirectFlight::Revoke(const string &ticket) {
	std::shared_ptr<Endpoint> endpoint;
	{
		std::lock_guard<std::mutex> guard(impl->registry->lock);
		auto entry = impl->registry->endpoints.find(ticket);
		if (entry == impl->registry->endpoints.end()) {
			return;
		}
		endpoint = entry->second;
	}
	shared_ptr<DirectBatch> release;
	{
		std::lock_guard<std::mutex> guard(endpoint->lock);
		endpoint->closed = true;
		release = std::move(endpoint->lease);
	}
	endpoint->channel->CloseConsumer(endpoint->consumer);
	endpoint->signal->changed.notify_all();
	{
		std::unique_lock<std::mutex> guard(endpoint->lock);
		if (!endpoint->idle.wait_for(guard, std::chrono::seconds(2), [&]() { return !endpoint->active_stream; })) {
			throw IOException("direct Flight revocation is pending; retry cleanup");
		}
	}
	// Retain its transport reservation until entered streams have exited.
	std::lock_guard<std::mutex> guard(impl->registry->lock);
	auto entry = impl->registry->endpoints.find(ticket);
	if (entry != impl->registry->endpoints.end() && entry->second == endpoint) {
		impl->registry->endpoints.erase(entry);
	}
}

bool DirectFlight::Delivered(const string &ticket) const {
	auto endpoint = impl->registry->Find(ticket);
	std::lock_guard<std::mutex> guard(endpoint->lock);
	auto error = endpoint->channel->Snapshot().error;
	if (!error.empty()) {
		throw IOException("%s", error);
	}
	return endpoint->finished && !endpoint->closed && endpoint->acknowledged == endpoint->sent && !endpoint->lease;
}

void DirectFlight::Cancel(const string &reason) {
	std::lock_guard<std::mutex> guard(impl->lock);
	impl->canceled = true;
	for (auto &link : impl->links) {
		link->Fail(reason);
	}
	std::lock_guard<std::mutex> registry_guard(impl->registry->lock);
	for (auto &item : impl->registry->endpoints) {
		item.second->channel->Abort(reason);
		item.second->signal->changed.notify_all();
	}
}

void DirectFlight::Close() {
	Cancel("direct Flight service closed");
	std::lock_guard<std::mutex> guard(impl->lock);
	if (impl->closed) {
		return;
	}
	impl->closed = true;
	for (auto &link : impl->links) {
		if (link->reader.joinable()) {
			link->reader.join();
		}
		if (link->watcher.joinable()) {
			link->watcher.join();
		}
	}
	{
		std::lock_guard<std::mutex> registry_guard(impl->registry->lock);
		impl->registry->stopped = true;
	}
	auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(2);
	(void)impl->server.Shutdown(&deadline);
	if (impl->serving.joinable()) {
		impl->serving.join();
	}
	impl->links.clear();
	std::lock_guard<std::mutex> registry_guard(impl->registry->lock);
	impl->registry->endpoints.clear();
}

string DirectFlight::Error() const {
	std::lock_guard<std::mutex> guard(impl->lock);
	for (auto &link : impl->links) {
		auto error = link->channel->Snapshot().error;
		if (!error.empty()) {
			return error;
		}
	}
	return string();
}

idx_t DirectFlight::ActiveLinks() const {
	std::lock_guard<std::mutex> guard(impl->lock);
	std::lock_guard<std::mutex> registry_guard(impl->registry->lock);
	return impl->closed ? 0 : impl->links.size() + impl->registry->endpoints.size();
}

bool DirectFlight::Ready() const {
	std::lock_guard<std::mutex> guard(impl->lock);
	if (impl->closed || impl->canceled) {
		return false;
	}
	for (auto &link : impl->links) {
		if (!link->ready || link->done) {
			return false;
		}
	}
	return true;
}

} // namespace vane_execution
} // namespace duckdb
