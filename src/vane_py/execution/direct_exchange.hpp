// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/atomic.hpp"

#include <deque>
#include <functional>
#include <map>

namespace duckdb {
namespace vane_execution {

using ExchangeWakeup = std::function<void()>;

struct DirectLimits {
	idx_t window_bytes;
	idx_t frame_bytes;
	idx_t frame_rows;
	idx_t frame_slots;
	void Validate() const;
};

struct DirectLedger {
	atomic<idx_t> bytes {0};
	atomic<idx_t> peak_bytes {0};
	atomic<idx_t> frames {0};
};

struct DirectFrame;
class DirectChannel;

// Credit follows native vector references, including slices and projections.
// A view never refers to the caller's transient Sink input.
class DirectBatch : public enable_shared_from_this<DirectBatch> {
public:
	DirectBatch(weak_ptr<DirectChannel> channel, string consumer, string producer, idx_t sequence,
	            shared_ptr<DirectFrame> frame);
	~DirectBatch();
	void Reference(DataChunk &target);
	idx_t Size() const;
	const string consumer;
	const string producer;
	const idx_t sequence;
	bool active = false;

private:
	weak_ptr<DirectChannel> channel;
	shared_ptr<DirectFrame> frame;
};

enum class DirectRead { DATA, BLOCKED, END, CLOSED };
enum class DirectWrite { ACCEPTED, BLOCKED, CLOSED };

struct DirectProducerStatus {
	bool finished = false;
	bool drained = false;
	bool has_consumers = false;
	string error;
};

struct DirectSnapshot {
	idx_t bytes = 0;
	idx_t peak_bytes = 0;
	idx_t frames = 0;
	idx_t leased_bytes = 0;
	idx_t queued_frames = 0;
	idx_t outstanding_frames = 0;
	idx_t producers = 0;
	idx_t finished_producers = 0;
	idx_t closed_consumers = 0;
	idx_t read_blocks = 0;
	idx_t write_blocks = 0;
	idx_t accepted_rows = 0;
	idx_t accepted_frames = 0;
	bool sealed = false;
	string error;
};

class DirectChannel : public enable_shared_from_this<DirectChannel> {
	friend class DirectBatch;

public:
	DirectChannel(vector<LogicalType> types, DirectLimits limits, idx_t max_producers, const vector<string> &consumers);
	~DirectChannel();
	void AddProducer(const string &producer);
	void SealProducers();
	void Finish(const string &producer, idx_t last_sequence);
	void Abort(const string &reason);
	void CloseConsumer(const string &consumer);
	DirectRead Poll(const string &consumer, shared_ptr<DirectBatch> &batch, ExchangeWakeup wakeup = {});
	DirectWrite TryWrite(const string &producer, idx_t sequence, DataChunk &input, const vector<idx_t> &rows,
	                     idx_t offset, idx_t count, ExchangeWakeup wakeup = {});
	idx_t FrameRows(DataChunk &input, const vector<idx_t> &rows, idx_t offset) const;
	idx_t LastSequence(const string &producer) const;
	DirectProducerStatus ProducerStatus(const string &producer) const;
	bool ProducerDrained(const string &producer) const;
	bool HasConsumers() const;
	void ValidateConsumer(const string &consumer) const;
	DirectSnapshot Snapshot() const;
	const vector<LogicalType> types;
	const DirectLimits limits;

private:
	struct Producer {
		idx_t sequence = 0;
		idx_t outstanding = 0;
		bool finished = false;
	};
	struct Consumer {
		idx_t bytes = 0;
		idx_t outstanding = 0;
		bool closed = false;
		std::deque<shared_ptr<DirectBatch>> queue;
	};
	void Release(const string &consumer, const string &producer, idx_t bytes);
	void CheckError() const;
	bool AtEnd(const Consumer &consumer) const;
	Producer &GetProducer(const string &producer);
	Consumer &GetConsumer(const string &consumer);
	static void Notify(std::map<string, ExchangeWakeup> &wakeups);
	mutable mutex lock;
	const idx_t max_producers;
	shared_ptr<DirectLedger> ledger;
	std::map<string, Producer> producers;
	std::map<string, Consumer> consumers;
	std::map<string, ExchangeWakeup> readers;
	std::map<string, ExchangeWakeup> writers;
	bool sealed = false;
	string error;
	idx_t read_blocks = 0;
	idx_t write_blocks = 0;
	idx_t accepted_rows = 0;
	idx_t accepted_frames = 0;
};

} // namespace vane_execution
} // namespace duckdb
