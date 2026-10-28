// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/simple_buffered_data.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/buffered_data/buffered_data.hpp"
#include "duckdb/parallel/interrupt.hpp"
#include "duckdb/common/queue.hpp"
#include "duckdb/common/vector_size.hpp"
#include "duckdb/common/types/data_chunk.hpp"

namespace duckdb {

class StreamQueryResult;
class ClientContextLock;

class SimpleBufferedData : public BufferedData {
public:
	static constexpr const BufferedData::Type TYPE = BufferedData::Type::SIMPLE;

public:
	//! A custom result allocator must outlive every returned chunk and its views.
	explicit SimpleBufferedData(ClientContext &context, bool media_flow_receiver = false,
	                            Allocator &result_allocator = Allocator::DefaultAllocator());
	~SimpleBufferedData() override;

public:
	void Append(const DataChunk &chunk);
	//! Copy fitting row prefixes after reserving their payload. Preserve offset
	//! across BLOCKED retries; acknowledge the input only once every row commits.
	SinkResultType AppendMediaFlow(const DataChunk &chunk, idx_t &offset, const InterruptState &interrupt);
	bool IsMediaFlowReceiver() const override {
		return media_flow != nullptr;
	}
	void Close() override;
	struct MediaFlowSnapshot {
		idx_t capacity;
		idx_t queued_bytes;
		idx_t reserved_bytes;
		idx_t peak_bytes;
		idx_t waiting_sinks;
		bool closed;
	};
	MediaFlowSnapshot SnapshotMediaFlow();
	void BlockSink(const InterruptState &blocked_sink);
	bool BufferIsFull();
	void UnblockSinks() override;
	StreamExecutionResult ExecuteTaskInternal(StreamQueryResult &result, ClientContextLock &context_lock) override;
	unique_ptr<DataChunk> Scan() override;
	inline idx_t BufferSize() const {
		return buffer_size;
	}

private:
	bool ChunkReady();
	//! Our handles to reschedule the blocked sink tasks
	queue<InterruptState> blocked_sinks;
	//! The queue of chunks
	queue<unique_ptr<DataChunk>> buffered_chunks;
	//! The current capacity of the buffer (tuples)
	atomic<idx_t> buffered_count;
	//! The amount of tuples we should buffer
	idx_t buffer_size;
	struct MediaFlowState;
	unique_ptr<MediaFlowState> media_flow;
};

} // namespace duckdb
