// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file exchange.hpp
 * @brief Abstract Exchange interface — per-exchange coordinator.
 *
 * An Exchange object is created for each logical exchange and manages the
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

/// Abstract interface for coordinating one logical exchange.
///
/// Lifecycle (coordinator side):
///   1. CreateExchange() via ExchangeManager
///   2. Copy immutable exchange metadata into each remote sink plan
///   3. The scheduler assigns one stable logical task id to each producer
///   4. The task runner binds that id and the current attempt id before execution
///   5. Workers write data via ExchangeSink::AddChunk()
///   6. AddSink() and SinkFinished() register each completed logical task attempt
///   7. AllRequiredSinksFinished() when all sinks are done
///   8. GetSourceHandles() → produce handles for downstream tasks
///   9. Close()
class Exchange {
public:
	virtual ~Exchange() = default;

	/// Register a new sink for the scheduler-owned task partition.
	/// May be called before submission or when a bound attempt completes.
	virtual ExchangeSinkHandle AddSink(idx_t task_partition_id) = 0;

	/// Create a concrete sink instance for the given handle and attempt.
	/// Coordinator-owned execution paths can use this directly. Serialized
	/// physical plans instead carry immutable exchange metadata and are bound by
	/// their task runner.
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

	/// Notify that a concrete sink instance finished, including transport
	/// incarnation metadata returned by the worker.
	virtual void SinkFinished(const ExchangeSinkInstanceHandle &instance, const std::string &node_id, int flight_port) {
		SinkFinished(instance.sink_handle, instance.attempt_id, node_id, flight_port);
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

	/// Immutable ownership metadata copied into remote sink plans.
	virtual const ExchangeContext &GetContext() const = 0;

	/// Prefix used to construct a concrete sink attempt location.
	virtual const std::string &GetSinkOutputLocationPrefix() const = 0;

	/// Close the exchange and release all resources.
	virtual void Close() = 0;
};

} // namespace distributed
} // namespace duckdb
