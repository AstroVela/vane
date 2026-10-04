// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "materialized_exchange.hpp"
#include "arrow_frame.hpp"
#include "fragment_plan.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/local_file_system.hpp"
#include "mbedtls_wrapper.hpp"

#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <chrono>
#include <condition_variable>
#include <thread>

namespace duckdb {
namespace vane_execution {
namespace {

using Hash = duckdb_mbedtls::MbedTlsWrapper::SHA256State;

void Check(const arrow::Status &status) {
	if (!status.ok()) {
		throw IOException("materialized exchange: %s", status.ToString());
	}
}

template <class T>
T Unwrap(arrow::Result<T> value) {
	Check(value.status());
	return std::move(value).ValueOrDie();
}

string Digest(Hash &state) {
	char digest[64];
	state.FinishHex(digest);
	return string(digest, sizeof(digest));
}

string Digest(const char *data, idx_t size) {
	Hash state;
	state.AddBytes(reinterpret_cast<const_data_ptr_t>(data), size);
	return Digest(state);
}

string Number(idx_t value) {
	string result(8, '\0');
	for (idx_t i = 0; i < 8; i++) {
		result[i] = char((value >> (8 * i)) & 255);
	}
	return result;
}

idx_t ParseNumber(const string &bytes) {
	idx_t value = 0;
	for (idx_t i = 0; i < 8; i++) {
		value |= idx_t(uint8_t(bytes[i])) << (8 * i);
	}
	return value;
}

void VerifyHandle(FileHandle &file, const MaterializedObject &expected, const string &schema,
                  const atomic<bool> *stop = nullptr) {
	if (expected.bytes < 40 || expected.bytes > (1ULL << 40) || expected.sha256.size() != 64 ||
	    expected.frames > expected.rows || (expected.frames == 0) != (expected.rows == 0) ||
	    file.GetFileSize() != expected.bytes) {
		throw IOException("materialized object size or identity mismatch");
	}
	Hash hash;
	vector<char> buffer(65536);
	for (idx_t offset = 0; offset < expected.bytes;) {
		if (stop && stop->load()) {
			throw InterruptException();
		}
		auto count = MinValue<idx_t>(buffer.size(), expected.bytes - offset);
		file.Read(buffer.data(), count, offset);
		hash.AddBytes(reinterpret_cast<const_data_ptr_t>(buffer.data()), count);
		offset += count;
	}
	if (Digest(hash) != expected.sha256 || file.GetFileSize() != expected.bytes) {
		throw IOException("materialized object checksum mismatch");
	}
	string header(16, '\0'), footer(24, '\0');
	file.Read(&header[0], header.size(), 0);
	file.Read(&footer[0], footer.size(), expected.bytes - footer.size());
	auto schema_size = ParseNumber(header.substr(8));
	if (header.substr(0, 8) != "VANEMAT1" || schema_size != schema.size() || schema_size > expected.bytes - 40 ||
	    ParseNumber(footer.substr(0, 8)) != 0 || ParseNumber(footer.substr(8, 8)) != expected.frames ||
	    ParseNumber(footer.substr(16, 8)) != expected.rows) {
		throw IOException("materialized object seal mismatch");
	}
	string actual(schema_size, '\0');
	if (schema_size) {
		file.Read(&actual[0], schema_size, 16);
	}
	if (actual != schema) {
		throw IOException("materialized object schema mismatch");
	}
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

} // namespace

struct MaterializedIO::Impl {
	const bool writing;
	const string path;
	shared_ptr<DirectChannel> channel;
	const string identity;
	const idx_t max_bytes;
	const idx_t frame_limit;
	const MaterializedObject expected;
	std::shared_ptr<Signal> signal = std::make_shared<Signal>();
	std::mutex close_lock;
	mutable std::mutex state_lock;
	MaterializedStatus state;
	atomic<bool> stop {false};
	std::thread worker;

	Impl(bool writing_p, const string &path_p, shared_ptr<DirectChannel> channel_p, const string &identity_p,
	     idx_t max_bytes_p, idx_t staging, MaterializedObject expected_p)
	    : writing(writing_p), path(path_p), channel(std::move(channel_p)), identity(identity_p), max_bytes(max_bytes_p),
	      frame_limit(channel ? 4 * channel->limits.frame_bytes + 65536 : 0), expected(std::move(expected_p)) {
		if (!channel || path.empty() || path.find('\0') != string::npos || !max_bytes || max_bytes > (1ULL << 40) ||
		    staging < MaterializedIO::StagingBytes(channel->limits.frame_bytes) || channel->types.empty() ||
		    channel->types.size() > 256) {
			throw InvalidInputException("materialized exchange requires a path and finite object/staging limits");
		}
		(void)ArrowSchemaFor(channel->types);
		if (writing) {
			channel->ValidateConsumer(identity);
		} else {
			channel->ProducerStatus(identity);
			if (expected.bytes > max_bytes) {
				throw InvalidInputException("materialized input exceeds its object reservation");
			}
		}
	}

	void CheckRunning() {
		auto error = channel->Snapshot().error;
		if (!error.empty()) {
			throw IOException("%s", error);
		}
		if (stop.load()) {
			throw InterruptException();
		}
	}

	void Cancel(const string &reason) {
		auto prior_error = channel->Snapshot().error;
		{
			std::lock_guard<std::mutex> guard(state_lock);
			if (state.done) {
				return;
			}
			if (state.error.empty()) {
				state.error = !prior_error.empty() ? prior_error
				              : reason.empty()     ? "materialized I/O canceled"
				                                   : reason;
			}
			stop = true;
		}
		channel->Abort(reason.empty() ? "materialized I/O canceled" : reason);
		signal->changed.notify_all();
	}

	MaterializedObject Write() {
		LocalFileSystem fs;
		auto file = fs.OpenFile(path, FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE |
		                                  FileFlags::FILE_FLAGS_EXCLUSIVE_CREATE | FileFlags::FILE_FLAGS_PRIVATE);
		MaterializedObject result;
		Hash hash;
		auto write = [&](const char *data, idx_t size) {
			CheckRunning();
			if (size > max_bytes - result.bytes) {
				throw OutOfMemoryException("materialized object exceeds reserved storage bytes");
			}
			fs.Write(*file, const_cast<char *>(data), size, result.bytes);
			hash.AddBytes(reinterpret_cast<const_data_ptr_t>(data), size);
			result.bytes += size;
		};
		auto text = [&](const string &value) {
			write(value.data(), value.size());
		};
		text("VANEMAT1");
		auto schema = SerializeSchema(channel->types);
		if (schema.size() > 65536) {
			throw InvalidInputException("materialized schema exceeds metadata limit");
		}
		text(Number(schema.size()));
		text(schema);
		auto arrow_schema = ArrowSchemaFor(channel->types);
		while (true) {
			CheckRunning();
			shared_ptr<DirectBatch> batch;
			auto read = channel->Poll(identity, batch, signal->Wakeup());
			if (read == DirectRead::BLOCKED) {
				signal->Wait();
				continue;
			}
			if (read == DirectRead::CLOSED) {
				throw IOException("materialized writer lost its input before seal");
			}
			if (read == DirectRead::END) {
				break;
			}
			DataChunk chunk;
			batch->Reference(chunk);
			auto record = Encode(chunk, arrow_schema);
			auto output = Unwrap(arrow::io::BufferOutputStream::Create());
			auto writer = Unwrap(arrow::ipc::MakeStreamWriter(output, arrow_schema));
			Check(writer->WriteRecordBatch(*record));
			Check(writer->Close());
			auto encoded = Unwrap(output->Finish());
			if (idx_t(encoded->size()) > frame_limit) {
				throw IOException("materialized frame exceeds staging reservation");
			}
			text(Number(encoded->size()));
			text(Number(chunk.size()));
			text(Digest(reinterpret_cast<const char *>(encoded->data()), encoded->size()));
			write(reinterpret_cast<const char *>(encoded->data()), encoded->size());
			result.frames++;
			result.rows += chunk.size();
		}
		text(Number(0));
		text(Number(result.frames));
		text(Number(result.rows));
		file->Sync();
		file->Close();
		CheckRunning();
		result.sha256 = Digest(hash);
		return result;
	}

	MaterializedObject Read() {
		LocalFileSystem fs;
		auto file = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ | FileLockType::READ_LOCK);
		VerifyHandle(*file, expected, SerializeSchema(channel->types), &stop);
		idx_t offset = 0;
		auto read = [&](idx_t size) {
			CheckRunning();
			if (size > expected.bytes - offset) {
				throw IOException("truncated materialized object");
			}
			string value(size, '\0');
			file->Read(&value[0], size, offset);
			offset += size;
			return value;
		};
		auto number = [&]() {
			auto bytes = read(8);
			idx_t value = 0;
			for (idx_t i = 0; i < 8; i++) {
				value |= idx_t(uint8_t(bytes[i])) << (8 * i);
			}
			return value;
		};
		if (read(8) != "VANEMAT1") {
			throw IOException("unknown materialized object format");
		}
		auto schema_size = number();
		if (!schema_size || schema_size > 65536 || read(schema_size) != SerializeSchema(channel->types)) {
			throw IOException("materialized object schema mismatch");
		}
		idx_t frames = 0, rows = 0;
		while (true) {
			auto size = number();
			if (!size) {
				if (number() != frames || number() != rows || offset != expected.bytes || frames != expected.frames ||
				    rows != expected.rows) {
					throw IOException("materialized object seal mismatch");
				}
				CheckRunning();
				return expected;
			}
			if (size > frame_limit) {
				throw IOException("materialized frame exceeds staging reservation");
			}
			auto count = number();
			if (!count || count > channel->limits.frame_rows || frames >= expected.frames ||
			    count > expected.rows - MinValue(rows, expected.rows)) {
				throw IOException("materialized frame row count mismatch");
			}
			auto digest = read(64);
			auto encoded = read(size);
			if (Digest(encoded.data(), encoded.size()) != digest) {
				throw IOException("materialized frame checksum mismatch");
			}
			auto buffer = arrow::Buffer::Wrap(reinterpret_cast<const uint8_t *>(encoded.data()), encoded.size());
			auto source = std::make_shared<arrow::io::BufferReader>(buffer);
			auto reader = Unwrap(arrow::ipc::RecordBatchStreamReader::Open(source));
			if (!reader->schema()->Equals(*ArrowSchemaFor(channel->types))) {
				throw IOException("materialized Arrow schema mismatch");
			}
			std::shared_ptr<arrow::RecordBatch> batch, extra;
			Check(reader->ReadNext(&batch));
			if (!batch || idx_t(batch->num_rows()) != count) {
				throw IOException("materialized Arrow row count mismatch");
			}
			Check(batch->ValidateFull());
			Check(reader->ReadNext(&extra));
			if (extra) {
				throw IOException("materialized frame has trailing batches");
			}
			DataChunk chunk;
			Decode(*batch, channel->types, chunk);
			vector<idx_t> selection;
			for (idx_t row = 0; row < count; row++) {
				selection.push_back(row);
			}
			if (channel->FrameRows(chunk, selection, 0) != count) {
				throw IOException("materialized frame exceeds reserved native window");
			}
			while (true) {
				CheckRunning();
				auto state = channel->TryWrite(identity, frames + 1, chunk, selection, 0, count, signal->Wakeup());
				if (state == DirectWrite::CLOSED) {
					throw IOException("materialized reader lost its consumers");
				}
				if (state == DirectWrite::ACCEPTED) {
					break;
				}
				signal->Wait();
			}
			frames++;
			rows += count;
		}
	}

	void Run() {
		try {
			auto result = writing ? Write() : Read();
			std::lock_guard<std::mutex> guard(state_lock);
			CheckRunning();
			// Publish EOF and completion under the same cancellation lock.
			// Closing immediately after EOF must not abort successful delivery.
			if (!writing) {
				channel->Finish(identity, result.frames);
			}
			state.object = std::move(result);
			state.done = true;
		} catch (const std::exception &error) {
			Cancel(error.what());
			std::lock_guard<std::mutex> guard(state_lock);
			state.done = true;
		}
	}
};

MaterializedIO::MaterializedIO(bool write, const string &path, shared_ptr<DirectChannel> channel,
                               const string &identity, idx_t max_bytes, idx_t staging, MaterializedObject expected)
    : impl(make_uniq<Impl>(write, path, std::move(channel), identity, max_bytes, staging, std::move(expected))) {
	impl->worker = std::thread([this]() { impl->Run(); });
}

MaterializedIO::~MaterializedIO() {
	Close();
}

idx_t MaterializedIO::StagingBytes(idx_t frame_bytes) {
	if (!frame_bytes || frame_bytes > (1ULL << 28)) {
		throw InvalidInputException("materialized frame_bytes must be between 1 and 256 MiB");
	}
	return 16 * frame_bytes + (1 << 18);
}

void MaterializedIO::Verify(const string &path, const MaterializedObject &expected, const string &schema) {
	if (schema.empty() || schema.size() > 65536) {
		throw InvalidInputException("invalid materialized object schema");
	}
	LocalFileSystem fs;
	auto file = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ | FileLockType::READ_LOCK);
	VerifyHandle(*file, expected, schema);
}

void MaterializedIO::Cancel(const string &reason) {
	impl->Cancel(reason);
}

void MaterializedIO::Close() {
	impl->Cancel("materialized I/O closed");
	std::lock_guard<std::mutex> guard(impl->close_lock);
	if (impl->worker.joinable()) {
		impl->worker.join();
	}
}

MaterializedStatus MaterializedIO::Status() const {
	std::lock_guard<std::mutex> guard(impl->state_lock);
	auto result = impl->state;
	if (result.error.empty()) {
		result.error = impl->channel->Snapshot().error;
	}
	return result;
}

} // namespace vane_execution
} // namespace duckdb
