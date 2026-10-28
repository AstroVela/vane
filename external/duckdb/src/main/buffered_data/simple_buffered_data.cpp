// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/main/buffered_data/simple_buffered_data.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/stream_query_result.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/execution/media_buffer.hpp"
#include "duckdb/common/unordered_map.hpp"

namespace duckdb {

struct SimpleBufferedData::MediaFlowState {
	explicit MediaFlowState(Allocator &allocator_p) : allocator(allocator_p) {
	}
	Allocator &allocator;
	struct Chunk {
		unique_ptr<DataChunk> chunk;
		idx_t bytes;
	};
	queue<Chunk> chunks;
	unordered_map<const void *, InterruptState> blocked_sinks;
	idx_t queued_bytes = 0;
	idx_t reserved_bytes = 0;
	idx_t peak_bytes = 0;
	bool closed = false;
};

SimpleBufferedData::SimpleBufferedData(ClientContext &context, bool media_flow_receiver_p, Allocator &result_allocator)
    : BufferedData(BufferedData::Type::SIMPLE, context),
      media_flow(media_flow_receiver_p ? make_uniq<MediaFlowState>(result_allocator) : nullptr) {
	buffered_count = 0;
	buffer_size = total_buffer_size;
}

SimpleBufferedData::~SimpleBufferedData() {
}

void SimpleBufferedData::BlockSink(const InterruptState &blocked_sink) {
	lock_guard<mutex> lock(glock);
	blocked_sinks.push(blocked_sink);
}

bool SimpleBufferedData::BufferIsFull() {
	if (media_flow) {
		lock_guard<mutex> lock(glock);
		return media_flow->queued_bytes + media_flow->reserved_bytes >= BufferSize();
	}
	return buffered_count >= BufferSize();
}

bool SimpleBufferedData::ChunkReady() {
	if (!media_flow) {
		return BufferIsFull();
	}
	lock_guard<mutex> lock(glock);
	// A short prefix can fill all usable capacity even below the byte limit.
	// Return it now instead of waiting for the producer to reach EOF.
	return !media_flow->chunks.empty();
}

void SimpleBufferedData::UnblockSinks() {
	if (media_flow) {
		unordered_map<const void *, InterruptState> ready;
		{
			lock_guard<mutex> lock(glock);
			auto &state = *media_flow;
			if (!state.closed && state.queued_bytes + state.reserved_bytes >= BufferSize()) {
				return;
			}
			ready.swap(state.blocked_sinks);
		}
		for (auto &entry : ready) {
			try {
				entry.second.Callback();
			} catch (...) {
				// One failing notification must not strand the other receivers.
			}
		}
		return;
	}
	auto cc = context.lock();
	if (!cc) {
		return;
	}
	(void)cc;

	if (buffered_count >= BufferSize()) {
		return;
	}
	// Reschedule enough blocked sinks to populate the buffer
	lock_guard<mutex> lock(glock);
	while (!blocked_sinks.empty()) {
		auto &blocked_sink = blocked_sinks.front();
		if (buffered_count >= BufferSize()) {
			// We have unblocked enough sinks already
			break;
		}
		blocked_sink.Callback();
		blocked_sinks.pop();
	}
}

StreamExecutionResult SimpleBufferedData::ExecuteTaskInternal(StreamQueryResult &result,
                                                              ClientContextLock &context_lock) {
	auto cc = context.lock();
	if (!cc) {
		return StreamExecutionResult::EXECUTION_CANCELLED;
	}
	if (!cc->IsActiveResult(context_lock, result)) {
		return StreamExecutionResult::EXECUTION_CANCELLED;
	}
	// Check for interrupt even if the buffer is full.
	// Without this check, cancel requests would not be detected until the buffer is drained.
	if (cc->interrupted.load(std::memory_order_relaxed)) {
		throw InterruptException();
	}
	if (ChunkReady()) {
		// The buffer isn't empty yet, just return
		return StreamExecutionResult::CHUNK_READY;
	}
	UnblockSinks();
	// Let the executor run until the buffer is no longer empty
	auto execution_result = cc->ExecuteTaskInternal(context_lock, result);
	if (ChunkReady()) {
		return StreamExecutionResult::CHUNK_READY;
	}
	if (execution_result == PendingExecutionResult::BLOCKED ||
	    execution_result == PendingExecutionResult::RESULT_READY) {
		return StreamExecutionResult::BLOCKED;
	}
	if (result.HasError()) {
		Close();
	}
	switch (execution_result) {
	case PendingExecutionResult::NO_TASKS_AVAILABLE:
	case PendingExecutionResult::RESULT_NOT_READY:
		return StreamExecutionResult::CHUNK_NOT_READY;
	case PendingExecutionResult::EXECUTION_FINISHED:
		return StreamExecutionResult::EXECUTION_FINISHED;
	case PendingExecutionResult::EXECUTION_ERROR:
		return StreamExecutionResult::EXECUTION_ERROR;
	default:
		throw InternalException("No conversion from PendingExecutionResult (%s) -> StreamExecutionResult",
		                        EnumUtil::ToString(execution_result));
	}
}

unique_ptr<DataChunk> SimpleBufferedData::Scan() {
	if (media_flow) {
		unique_ptr<DataChunk> chunk;
		{
			lock_guard<mutex> lock(glock);
			auto &state = *media_flow;
			if (state.closed || state.chunks.empty()) {
				return nullptr;
			}
			auto &entry = state.chunks.front();
			state.queued_bytes -= entry.bytes;
			chunk = std::move(entry.chunk);
			state.chunks.pop();
		}
		// Queue -> client handoff. Retained results keep their vector storage,
		// while the bounded queue can accept the next copy.
		UnblockSinks();
		return chunk;
	}
	if (Closed()) {
		return nullptr;
	}

	lock_guard<mutex> lock(glock);
	if (buffered_chunks.empty()) {
		Close();
		return nullptr;
	}
	auto chunk = std::move(buffered_chunks.front());
	buffered_chunks.pop();
	if (chunk) {
		buffered_count -= chunk->GetAllocationSize();
	}
	return chunk;
}

void SimpleBufferedData::Append(const DataChunk &to_append) {
	if (media_flow) {
		throw InternalException("Native media result copy requires receiving capacity reservation");
	}
	auto chunk = make_uniq<DataChunk>();
	chunk->Initialize(Allocator::DefaultAllocator(), to_append.GetTypes());
	to_append.Copy(*chunk, 0);
	auto allocation_size = chunk->GetAllocationSize();

	unique_lock<mutex> lock(glock);
	buffered_chunks.push(std::move(chunk));
	buffered_count += allocation_size;
}

SinkResultType SimpleBufferedData::AppendMediaFlow(const DataChunk &input, idx_t &offset,
                                                   const InterruptState &interrupt) {
	if (!media_flow || offset > input.size()) {
		throw InternalException("Invalid native media result receiving state");
	}
	auto &state = *media_flow;
	if (offset == input.size()) {
		lock_guard<mutex> lock(glock);
		state.blocked_sinks.erase(&offset);
		if (state.closed || context.expired()) {
			throw InterruptException();
		}
	}
	while (offset < input.size()) {
		idx_t rows;
		idx_t bytes;
		shared_ptr<ClientContext> cc;
		{
			lock_guard<mutex> lock(glock);
			state.blocked_sinks.erase(&offset);
			cc = state.closed ? nullptr : context.lock();
			if (!cc) {
				throw InterruptException();
			}
			const auto row_bytes = MaxValue<idx_t>(1, EstimateMediaFlowInputBytes(input, offset, 1));
			if (row_bytes > BufferSize()) {
				throw OutOfMemoryException("Native media streaming result row requires %llu payload bytes, exceeding "
				                           "streaming_buffer_size receiving capacity of %llu bytes",
				                           static_cast<unsigned long long>(row_bytes),
				                           static_cast<unsigned long long>(BufferSize()));
			}
			const auto occupied = state.queued_bytes + state.reserved_bytes;
			D_ASSERT(occupied <= BufferSize());
			const auto available = BufferSize() - occupied;
			rows = available ? MediaFlowRowsWithinByteLimit(input, offset, input.size() - offset, available) : 0;
			if (!rows) {
				// Inspect capacity and arm the retry under the same lock as Scan.
				state.blocked_sinks.emplace(&offset, interrupt);
				return SinkResultType::BLOCKED;
			}
			bytes = MaxValue<idx_t>(1, EstimateMediaFlowInputBytes(input, offset, rows));
			state.reserved_bytes += bytes;
			state.peak_bytes = MaxValue<idx_t>(state.peak_bytes, state.queued_bytes + state.reserved_bytes);
		}
		bool reserved = true;
		try {
			auto chunk = make_uniq<DataChunk>();
			// Returned copies can outlive the result, connection and database.
			// The default allocator has process lifetime, unlike Allocator::Get(*cc).
			chunk->Initialize(state.allocator, input.GetTypes(), rows);
			SelectionVector selected(rows);
			for (idx_t row = 0; row < rows; row++) {
				selected.set_index(row, offset + row);
			}
			input.Copy(*chunk, selected, rows);
			lock_guard<mutex> lock(glock);
			if (state.closed) {
				throw InterruptException();
			}
			state.chunks.push(MediaFlowState::Chunk {std::move(chunk), bytes});
			state.reserved_bytes -= bytes;
			reserved = false;
			state.queued_bytes += bytes;
			offset += rows;
		} catch (...) {
			if (reserved) {
				lock_guard<mutex> lock(glock);
				state.reserved_bytes -= bytes;
			}
			UnblockSinks();
			throw;
		}
	}
	// The receiving queue owns every copied row before production credit is
	// returned. A partial receipt never acknowledges the original batch.
	input.AcknowledgeMediaFlow();
	offset = 0;
	return SinkResultType::NEED_MORE_INPUT;
}

void SimpleBufferedData::Close() {
	if (!media_flow) {
		BufferedData::Close();
		return;
	}
	queue<MediaFlowState::Chunk> retired;
	{
		lock_guard<mutex> lock(glock);
		media_flow->closed = true;
		context.reset();
		retired.swap(media_flow->chunks);
		media_flow->queued_bytes = 0;
	}
	// Reservations for copies in progress remain charged until their rollback.
	// Free chunks and notify tasks outside the queue lock.
	while (!retired.empty()) {
		retired.pop();
	}
	UnblockSinks();
}

SimpleBufferedData::MediaFlowSnapshot SimpleBufferedData::SnapshotMediaFlow() {
	lock_guard<mutex> lock(glock);
	if (!media_flow) {
		throw InvalidInputException("Not a native media result queue");
	}
	auto &state = *media_flow;
	return {BufferSize(),     state.queued_bytes,         state.reserved_bytes,
	        state.peak_bytes, state.blocked_sinks.size(), state.closed};
}

} // namespace duckdb
