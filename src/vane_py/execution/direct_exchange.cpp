// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "direct_exchange.hpp"
#include "exchange_types.hpp"
#include "frame_vector.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/vector_buffer.hpp"

#include <cstring>

namespace duckdb {
namespace vane_execution {
struct DirectFrame {
	DirectFrame(shared_ptr<DirectLedger> ledger_p, DataChunk &input, const vector<idx_t> &rows, idx_t offset,
	            idx_t count_p)
	    : ledger(std::move(ledger_p)), types(input.GetTypes()), count(count_p) {
		bytes = Measure(input, rows, offset, count, NumericLimits<idx_t>::Maximum() - 1);
		buffer = Allocator::DefaultAllocator().Allocate(bytes);
		memset(buffer.get(), 0, bytes);
		Measure(input, rows, offset, count, bytes, &columns, buffer.get());
		auto current = ledger->bytes.fetch_add(bytes) + bytes;
		auto peak = ledger->peak_bytes.load();
		while (current > peak && !ledger->peak_bytes.compare_exchange_weak(peak, current)) {
		}
		ledger->frames++;
	}
	~DirectFrame() {
		buffer.Reset();
		ledger->bytes -= bytes;
		ledger->frames--;
	}
	shared_ptr<DirectLedger> ledger;
	vector<LogicalType> types;
	idx_t count;
	idx_t bytes;
	vector<FrameColumn> columns;
	AllocatedData buffer;
};

void DirectLimits::Validate() const {
	if (!window_bytes || !frame_bytes || frame_bytes > window_bytes || !frame_slots || !frame_rows ||
	    frame_rows > STANDARD_VECTOR_SIZE) {
		throw InvalidInputException("direct limits require 0 < frame_bytes <= window_bytes, positive frame_slots and "
		                            "1 <= frame_rows <= STANDARD_VECTOR_SIZE");
	}
}

DirectBatch::DirectBatch(weak_ptr<DirectChannel> channel_p, string consumer_p, string producer_p, idx_t sequence_p,
                         shared_ptr<DirectFrame> frame_p)
    : consumer(std::move(consumer_p)), producer(std::move(producer_p)), sequence(sequence_p),
      channel(std::move(channel_p)), frame(std::move(frame_p)) {
}

DirectBatch::~DirectBatch() {
	auto bytes = Size();
	frame.reset();
	if (active) {
		auto owner = channel.lock();
		if (owner) {
			owner->Release(consumer, producer, bytes);
		}
	}
}

idx_t DirectBatch::Size() const {
	return frame->bytes;
}

void DirectBatch::Reference(DataChunk &target) {
	if (target.ColumnCount() == 0) {
		target.InitializeEmpty(frame->types);
	}
	for (idx_t col = 0; col < frame->types.size(); col++) {
		auto view =
		    ReferenceFrameVector(frame->types[col], frame->columns[col], frame->buffer.get(), shared_from_this());
		target.data[col].Reference(*view);
	}
	target.SetCardinality(frame->count);
}

DirectChannel::DirectChannel(vector<LogicalType> types_p, DirectLimits limits_p, idx_t max_producers_p,
                             const vector<string> &consumer_ids)
    : types(std::move(types_p)), limits(limits_p), max_producers(max_producers_p),
      ledger(make_shared_ptr<DirectLedger>()) {
	limits.Validate();
	if (!max_producers || consumer_ids.empty() || types.empty()) {
		throw InvalidInputException("direct channel requires producers, consumers and a schema");
	}
	idx_t minimum_frame = 0;
	for (auto &type : types) {
		CheckExchangeType(type);
		minimum_frame = AddSize(Align(minimum_frame), sizeof(validity_t) + FrameWidth(type));
	}
	if (limits.frame_bytes < minimum_frame) {
		throw InvalidInputException("direct frame_bytes cannot hold one row of its schema");
	}
	for (auto &id : consumer_ids) {
		if (id.empty() || !consumers.emplace(id, Consumer()).second) {
			throw InvalidInputException("duplicate or empty direct consumer");
		}
	}
}

DirectChannel::~DirectChannel() = default;

void DirectChannel::Notify(std::map<string, ExchangeWakeup> &wakeups) {
	for (auto &entry : wakeups) {
		if (entry.second) {
			entry.second();
		}
	}
}

void DirectChannel::CheckError() const {
	if (!error.empty()) {
		throw InvalidInputException("direct exchange aborted: %s", error);
	}
}

DirectChannel::Producer &DirectChannel::GetProducer(const string &id) {
	auto found = producers.find(id);
	if (found == producers.end()) {
		throw InvalidInputException("unknown direct producer: %s", id);
	}
	return found->second;
}

DirectChannel::Consumer &DirectChannel::GetConsumer(const string &id) {
	auto found = consumers.find(id);
	if (found == consumers.end()) {
		throw InvalidInputException("unknown direct consumer: %s", id);
	}
	return found->second;
}

void DirectChannel::AddProducer(const string &id) {
	lock_guard<mutex> guard(lock);
	CheckError();
	if (sealed || id.empty() || producers.size() == max_producers || !producers.emplace(id, Producer()).second) {
		throw InvalidInputException("invalid direct producer registration (duplicate, full or sealed)");
	}
}

void DirectChannel::SealProducers() {
	std::map<string, ExchangeWakeup> wakeups;
	{
		lock_guard<mutex> guard(lock);
		CheckError();
		sealed = true;
		wakeups.swap(readers);
	}
	Notify(wakeups);
}

void DirectChannel::Finish(const string &id, idx_t last_sequence) {
	std::map<string, ExchangeWakeup> wakeups;
	{
		lock_guard<mutex> guard(lock);
		CheckError();
		auto &producer = GetProducer(id);
		if (producer.sequence != last_sequence) {
			throw InvalidInputException("direct FINISH sequence does not match accepted data");
		}
		producer.finished = true;
		writers.erase(id);
		wakeups.swap(readers);
	}
	Notify(wakeups);
}

bool DirectChannel::AtEnd(const Consumer &consumer) const {
	if (!sealed || !consumer.queue.empty()) {
		return false;
	}
	for (auto &entry : producers) {
		if (!entry.second.finished) {
			return false;
		}
	}
	return true;
}

DirectRead DirectChannel::Poll(const string &id, shared_ptr<DirectBatch> &batch, ExchangeWakeup wakeup) {
	// Release a previous caller reference before taking the credit mutex.
	batch.reset();
	lock_guard<mutex> guard(lock);
	CheckError();
	auto &consumer = GetConsumer(id);
	if (consumer.closed) {
		return DirectRead::CLOSED;
	}
	if (!consumer.queue.empty()) {
		batch = std::move(consumer.queue.front());
		consumer.queue.pop_front();
		readers.erase(id);
		return DirectRead::DATA;
	}
	if (AtEnd(consumer)) {
		readers.erase(id);
		return DirectRead::END;
	}
	// State check and waiter installation share the publication mutex. A data,
	// FINISH or close event cannot fall between them.
	if (wakeup) {
		readers[id] = std::move(wakeup);
	}
	read_blocks++;
	return DirectRead::BLOCKED;
}

idx_t DirectChannel::FrameRows(DataChunk &input, const vector<idx_t> &rows, idx_t offset) const {
	if (input.GetTypes() != types || offset >= rows.size()) {
		throw InvalidInputException("direct frame schema or row selection mismatch");
	}
	auto count = MinValue<idx_t>(limits.frame_rows, rows.size() - offset);
	while (Measure(input, rows, offset, count, limits.frame_bytes) > limits.frame_bytes) {
		if (count == 1) {
			throw InvalidInputException("one row exceeds direct frame_bytes");
		}
		count = (count + 1) / 2;
	}
	return count;
}

DirectWrite DirectChannel::TryWrite(const string &id, idx_t sequence, DataChunk &input, const vector<idx_t> &rows,
                                    idx_t offset, idx_t count, ExchangeWakeup wakeup) {
	if (input.GetTypes() != types || count > limits.frame_rows) {
		throw InvalidInputException("direct frame schema or row count mismatch");
	}
	auto size = Measure(input, rows, offset, count, limits.frame_bytes);
	if (size > limits.frame_bytes) {
		throw InvalidInputException("direct frame exceeds frame_bytes");
	}
	std::map<string, ExchangeWakeup> wakeups;
	{
		lock_guard<mutex> guard(lock);
		CheckError();
		auto &producer = GetProducer(id);
		if (producer.finished || producer.sequence == NumericLimits<idx_t>::Maximum() ||
		    sequence != producer.sequence + 1) {
			throw InvalidInputException("direct data sequence must follow the last accepted sequence, before FINISH");
		}
		idx_t live = 0;
		for (auto &entry : consumers) {
			auto &consumer = entry.second;
			if (consumer.closed) {
				continue;
			}
			live++;
			if (size > limits.window_bytes - consumer.bytes || consumer.outstanding == limits.frame_slots) {
				if (wakeup) {
					writers[id] = std::move(wakeup);
				}
				write_blocks++;
				return DirectWrite::BLOCKED;
			}
		}
		writers.erase(id);
		if (!live) {
			return DirectWrite::CLOSED;
		}
		// Broadcast shares one physical allocation; each consumer has an
		// independent lease/window. Publish only after every enqueue succeeds.
		auto frame = make_shared_ptr<DirectFrame>(ledger, input, rows, offset, count);
		vector<Consumer *> enqueued;
		enqueued.reserve(live);
		try {
			for (auto &entry : consumers) {
				if (!entry.second.closed) {
					entry.second.queue.push_back(
					    make_shared_ptr<DirectBatch>(shared_from_this(), entry.first, id, sequence, frame));
					enqueued.push_back(&entry.second);
				}
			}
		} catch (...) {
			for (auto consumer : enqueued) {
				consumer->queue.pop_back();
			}
			throw;
		}
		for (auto consumer : enqueued) {
			consumer->queue.back()->active = true;
			consumer->bytes += size;
			consumer->outstanding++;
		}
		producer.sequence = sequence;
		producer.outstanding += live;
		accepted_rows += count;
		accepted_frames++;
		wakeups.swap(readers);
	}
	Notify(wakeups);
	return DirectWrite::ACCEPTED;
}

void DirectChannel::Release(const string &consumer_id, const string &producer_id, idx_t bytes) {
	std::map<string, ExchangeWakeup> wakeups;
	{
		lock_guard<mutex> guard(lock);
		auto &consumer = GetConsumer(consumer_id);
		auto &producer = GetProducer(producer_id);
		D_ASSERT(consumer.bytes >= bytes && consumer.outstanding && producer.outstanding);
		consumer.bytes -= bytes;
		consumer.outstanding--;
		producer.outstanding--;
		wakeups.swap(writers);
	}
	Notify(wakeups);
}

void DirectChannel::CloseConsumer(const string &id) {
	std::deque<shared_ptr<DirectBatch>> released;
	std::map<string, ExchangeWakeup> wakeups;
	ExchangeWakeup reader;
	{
		lock_guard<mutex> guard(lock);
		auto &consumer = GetConsumer(id);
		consumer.closed = true;
		released.swap(consumer.queue);
		auto found = readers.find(id);
		if (found != readers.end()) {
			reader = std::move(found->second);
			readers.erase(found);
		}
		wakeups.swap(writers);
	}
	released.clear();
	if (reader) {
		reader();
	}
	Notify(wakeups);
}

void DirectChannel::Abort(const string &reason) {
	std::map<string, ExchangeWakeup> read_wakeups, write_wakeups;
	vector<std::deque<shared_ptr<DirectBatch>>> released;
	{
		lock_guard<mutex> guard(lock);
		if (error.empty()) {
			error = reason.empty() ? "canceled" : reason;
		}
		for (auto &entry : consumers) {
			released.emplace_back();
			released.back().swap(entry.second.queue);
		}
		read_wakeups.swap(readers);
		write_wakeups.swap(writers);
	}
	released.clear();
	Notify(read_wakeups);
	Notify(write_wakeups);
}

idx_t DirectChannel::LastSequence(const string &id) const {
	lock_guard<mutex> guard(lock);
	CheckError();
	auto found = producers.find(id);
	if (found == producers.end()) {
		throw InvalidInputException("unknown direct producer: %s", id);
	}
	return found->second.sequence;
}

DirectProducerStatus DirectChannel::ProducerStatus(const string &id) const {
	lock_guard<mutex> guard(lock);
	auto &producer = producers.at(id);
	DirectProducerStatus result;
	result.finished = producer.finished;
	result.drained = producer.finished && producer.outstanding == 0;
	result.error = error;
	for (auto &entry : consumers) {
		result.has_consumers = result.has_consumers || !entry.second.closed;
	}
	return result;
}

bool DirectChannel::ProducerDrained(const string &id) const {
	lock_guard<mutex> guard(lock);
	CheckError();
	auto &producer = producers.at(id);
	return producer.finished && producer.outstanding == 0;
}

bool DirectChannel::HasConsumers() const {
	lock_guard<mutex> guard(lock);
	CheckError();
	for (auto &entry : consumers) {
		if (!entry.second.closed) {
			return true;
		}
	}
	return false;
}

void DirectChannel::ValidateConsumer(const string &id) const {
	lock_guard<mutex> guard(lock);
	CheckError();
	if (consumers.find(id) == consumers.end()) {
		throw InvalidInputException("unknown direct consumer: %s", id);
	}
}

DirectSnapshot DirectChannel::Snapshot() const {
	lock_guard<mutex> guard(lock);
	DirectSnapshot result;
	result.bytes = ledger->bytes.load();
	result.peak_bytes = ledger->peak_bytes.load();
	result.frames = ledger->frames.load();
	for (auto &entry : consumers) {
		result.leased_bytes += entry.second.bytes;
		result.queued_frames += entry.second.queue.size();
		result.outstanding_frames += entry.second.outstanding;
		result.closed_consumers += entry.second.closed;
	}
	result.producers = producers.size();
	for (auto &entry : producers) {
		result.finished_producers += entry.second.finished;
	}
	result.read_blocks = read_blocks;
	result.write_blocks = write_blocks;
	result.accepted_rows = accepted_rows;
	result.accepted_frames = accepted_frames;
	result.sealed = sealed;
	result.error = error;
	return result;
}

} // namespace vane_execution
} // namespace duckdb
