// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file exchange.hpp
 * @brief Abstract Exchange interface — per-stage coordinator.
 *
 * An Exchange object is created for each shuffle stage and manages the
 * lifecycle of sinks and sources. The coordinator uses it to track which sinks
 * have completed and to produce source handles for downstream tasks.
 */

#pragma once

#include "duckdb/common/types.hpp"
#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/execution/distributed/exchange/exchange_handles.hpp"

#include <string>
#include <vector>

namespace duckdb {
namespace distributed {

/// Abstract interface for coordinating a single shuffle stage's exchange.
///
/// Lifecycle (coordinator side):
///   1. CreateExchange() via ExchangeManager
///   2. AddSink() for each input task → get ExchangeSinkHandle
///   3. InstantiateSink() for each handle → get SinkInstanceHandle (sent to worker)
///   4. Workers write data via ExchangeSink::AddChunk()
///   5. SinkFinished() called as each sink completes
///   6. AllRequiredSinksFinished() when all sinks are done
///   7. GetSourceHandles() → produce handles for downstream tasks
///   8. Close()
class Exchange {
public:
	virtual ~Exchange() = default;

	/// Register a new sink for the given task partition.
	/// Called once per input task (coordinator side).
	virtual ExchangeSinkHandle AddSink(idx_t task_partition_id) = 0;

	/// Create a concrete sink instance for the given handle and attempt.
	/// The returned SinkInstanceHandle is sent to the worker to create
	/// the actual ExchangeSink.
	/// Multiple attempts for the same handle support fault tolerance.
	virtual ExchangeSinkInstanceHandle InstantiateSink(const ExchangeSinkHandle &handle, idx_t attempt_id) = 0;

	/// Notify that a sink has finished writing successfully.
	virtual void SinkFinished(const ExchangeSinkHandle &handle, idx_t attempt_id) = 0;

	/// Notify that a sink has finished writing successfully, including the
	/// worker location that can serve the selected attempt to downstream tasks.
	virtual void SinkFinished(const ExchangeSinkHandle &handle, idx_t attempt_id, const std::string &node_id,
	                          int flight_port) {
		SinkFinished(handle, attempt_id);
	}

	/// Notify that all required sinks have finished.
	/// Triggers source handle creation (e.g., listing committed files).
	virtual void AllRequiredSinksFinished() = 0;

	/// Get source handles for downstream tasks to read from.
	/// Only valid after AllRequiredSinksFinished() (for non-streaming
	/// exchanges) or may return partial results incrementally
	/// (for streaming exchanges).
	///
	/// Empty partitions are automatically skipped — no handles are
	/// generated for partitions with no data.
	virtual std::vector<ExchangeSourceHandle> GetSourceHandles() = 0;

	/// Number of output partitions this exchange was created with.
	virtual idx_t GetNumPartitions() const = 0;

	/// Close the exchange and release all resources.
	virtual void Close() = 0;
};

} // namespace distributed
} // namespace duckdb
