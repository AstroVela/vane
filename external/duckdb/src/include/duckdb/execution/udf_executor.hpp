// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/udf_executor.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/insertion_order_preserving_map.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/execution/external_block.hpp"
#include "duckdb/main/client_context.hpp"

#include <functional>

namespace duckdb {
class InterruptState;

struct UDFConfig {
	idx_t max_buffer_size = 1024;
};

using RefBlockSlice = ExternalBlockSlice;
using RefBlockMetadata = ExternalBlockMetadata;
using RefBlockDescriptor = ExternalBlockDescriptor;
using LazyRefDataChunk = LazyDataChunk;
using RefBundleBundler = LazyDataChunkBundler;

struct UDFResult {
	unique_ptr<DataChunk> outputs;
	unique_ptr<LazyRefDataChunk> ref_outputs;
	unique_ptr<DataChunk> rows;
	bool submit_complete = true;
	idx_t submit_id = 0;
	std::function<void()> handoff_output_lease;
	std::function<void()> release_output_lease;
};

enum class UDFOutputEventKind { DATA, COMPLETE, ERROR, FINISHED };

struct UDFOutputEvent {
	UDFOutputEventKind kind = UDFOutputEventKind::DATA;
	idx_t submit_id = 0;

	unique_ptr<DataChunk> outputs;
	unique_ptr<LazyRefDataChunk> ref_outputs;
	unique_ptr<DataChunk> rows;

	bool submit_complete = true;
	string error;
	std::function<void()> handoff_output_lease;
	std::function<void()> release_output_lease;
};

struct UDFOutputConsumer {
	std::function<idx_t()> data_capacity;
	std::function<idx_t()> data_byte_capacity;
	std::function<idx_t()> data_item_byte_capacity;
	std::function<void(UDFOutputEvent &&)> accept_event;
	std::function<void(const string &)> accept_error;
	std::function<void()> notify_finished;
};

enum class UDFWakeupRegistrationResult { UNSUPPORTED, ARMED, READY };

class UDFExecutor {
public:
	virtual ~UDFExecutor() = default;

	virtual idx_t Submit(DataChunk &args, DataChunk &rows, ClientContext &context) = 0;
	virtual bool TrySubmit(DataChunk &args, DataChunk &rows, ClientContext &context, idx_t &submit_id) = 0;
	virtual bool TrySubmitWithRetainedBytes(DataChunk &args, DataChunk &rows, ClientContext &context,
	                                        idx_t retained_input_bytes, idx_t &submit_id) = 0;
	virtual bool TrySubmitEnvelope(vector<unique_ptr<DataChunk>> &args, DataChunk &rows, ClientContext &context,
	                               idx_t &submit_id) = 0;
	virtual bool TrySubmitEnvelopeWithRetainedBytes(vector<unique_ptr<DataChunk>> &args, DataChunk &rows,
	                                                ClientContext &context, idx_t retained_input_bytes,
	                                                idx_t &submit_id) = 0;
	virtual bool SupportsRefBundleInput() = 0;
	virtual idx_t SubmitRefBundle(LazyRefDataChunk &bundle, DataChunk &rows, ClientContext &context) = 0;
	virtual bool TrySubmitRefBundle(LazyRefDataChunk &bundle, DataChunk &rows, ClientContext &context,
	                                idx_t &submit_id) = 0;
	virtual bool TrySubmitRefBundleWithRetainedBytes(LazyRefDataChunk &bundle, DataChunk &rows, ClientContext &context,
	                                                 idx_t retained_input_bytes, idx_t &submit_id) = 0;
	virtual std::pair<bool, UDFResult> TakeReadyResult(ClientContext &context) = 0;
	virtual void FinishedSubmitting(ClientContext &context) = 0;
	virtual bool AllTasksFinished(ClientContext &context) = 0;
	virtual bool SupportsAsyncWakeup() = 0;
	virtual UDFWakeupRegistrationResult RegisterWakeup(InterruptState &interrupt_state) = 0;
	virtual void RegisterWakeupCallback(std::function<void()> callback) = 0;
	virtual void EnqueueDeferredWakeup(std::function<void()> callback) = 0;
	virtual bool SupportsOutputConsumer() = 0;
	virtual void RegisterOutputConsumer(UDFOutputConsumer consumer) = 0;
	virtual void NotifyOutputConsumerSpaceAvailable() = 0;
	virtual idx_t DebugSlotId() = 0;
	virtual InsertionOrderPreservingMap<string> Stats() = 0;
};

using udf_executor_factory_t = unique_ptr<UDFExecutor> (*)(ClientContext &context, const Value &payload,
                                                           UDFConfig &config, shared_ptr<void> actor_handles);

DUCKDB_API void SetUDFExecutorFactory(udf_executor_factory_t factory);
DUCKDB_API udf_executor_factory_t GetUDFExecutorFactory();

} // namespace duckdb
