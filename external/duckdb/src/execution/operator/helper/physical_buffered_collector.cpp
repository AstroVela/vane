// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/execution/operator/helper/physical_buffered_collector.hpp"
#include "duckdb/main/stream_query_result.hpp"
#include "duckdb/main/client_context.hpp"

namespace duckdb {

PhysicalBufferedCollector::PhysicalBufferedCollector(PhysicalPlan &physical_plan, PreparedStatementData &data,
                                                     bool parallel, bool media_flow_receiver_p)
    : PhysicalResultCollector(physical_plan, data), parallel(parallel), media_flow_receiver(media_flow_receiver_p) {
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class BufferedCollectorGlobalState : public GlobalSinkState {
public:
	mutex glock;
	//! This is weak to avoid creating a cyclical reference
	weak_ptr<ClientContext> context;
	shared_ptr<BufferedData> buffered_data;
};

class BufferedCollectorLocalState : public LocalSinkState {
public:
	idx_t row_offset = 0;
};

SinkResultType PhysicalBufferedCollector::Sink(ExecutionContext &context, DataChunk &chunk,
                                               OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<BufferedCollectorGlobalState>();
	auto &lstate = input.local_state.Cast<BufferedCollectorLocalState>();
	auto &buffered_data = gstate.buffered_data->Cast<SimpleBufferedData>();
	if (media_flow_receiver) {
		return buffered_data.AppendMediaFlow(chunk, lstate.row_offset, input.interrupt_state);
	}

	lock_guard<mutex> l(gstate.glock);

	if (buffered_data.BufferIsFull()) {
		auto callback_state = input.interrupt_state;
		buffered_data.BlockSink(callback_state);
		return SinkResultType::BLOCKED;
	}
	buffered_data.Append(chunk);
	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalBufferedCollector::Combine(ExecutionContext &context,
                                                         OperatorSinkCombineInput &input) const {
	return SinkCombineResultType::FINISHED;
}

unique_ptr<GlobalSinkState> PhysicalBufferedCollector::GetGlobalSinkState(ClientContext &context) const {
	auto state = make_uniq<BufferedCollectorGlobalState>();
	state->context = context.shared_from_this();
	state->buffered_data = make_shared_ptr<SimpleBufferedData>(context, media_flow_receiver);
	return std::move(state);
}

unique_ptr<LocalSinkState> PhysicalBufferedCollector::GetLocalSinkState(ExecutionContext &context) const {
	auto state = make_uniq<BufferedCollectorLocalState>();
	return std::move(state);
}

unique_ptr<QueryResult> PhysicalBufferedCollector::GetResult(GlobalSinkState &state) const {
	auto &gstate = state.Cast<BufferedCollectorGlobalState>();
	lock_guard<mutex> l(gstate.glock);
	// FIXME: maybe we want to check if the execution was successful before creating the StreamQueryResult ?
	auto cc = gstate.context.lock();
	auto result = make_uniq<StreamQueryResult>(statement_type, properties, types, names, cc->GetClientProperties(),
	                                           gstate.buffered_data);
	return std::move(result);
}

bool PhysicalBufferedCollector::ParallelSink() const {
	return parallel;
}

bool PhysicalBufferedCollector::SinkOrderDependent() const {
	return true;
}

} // namespace duckdb
