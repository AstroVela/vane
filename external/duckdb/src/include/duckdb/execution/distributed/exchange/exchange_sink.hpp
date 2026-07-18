// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file exchange_sink.hpp
 * @brief Abstract ExchangeSink interface for writing data to an exchange.
 *
 * Each worker task creates one ExchangeSink to write partitioned data. The sink
 * supports backpressure via IsBlocked()/WaitUnblocked().
 */

#pragma once

#include "duckdb/common/types.hpp"
#include "duckdb/execution/distributed/common_types.hpp"

namespace duckdb {

class DataChunk;

namespace distributed {

/// Abstract interface for writing exchange data.
///
/// Lifecycle: Created by ExchangeManager::CreateSink() → Add() data →
///            Finish() on success or Abort() on failure.
///
/// Threading: A single ExchangeSink is used by one operator thread.
///            Implementations must be safe for concurrent access from
///            the backpressure mechanism.
class ExchangeSink {
public:
	virtual ~ExchangeSink() = default;

	/// Write a DataChunk to the specified output partition.
	///
	/// @param partition_id  Target partition (0-based, < output_partition_count)
	/// @param chunk         Data to write (will be consumed/moved)
	/// @return Ok on success, error on failure
	virtual DuckDBResult<void> AddChunk(idx_t partition_id, DataChunk &chunk) = 0;

	/// Check whether the sink is currently blocked (backpressure).
	///
	/// When true, the calling operator should yield and call WaitUnblocked()
	/// before attempting to Add() more data.
	/// Mirrors asynchronous backpressure checks in distributed sink APIs.
	virtual bool IsBlocked() const = 0;

	/// Block until the sink is ready to accept more data.
	virtual void WaitUnblocked() = 0;

	/// Signal that no more data will be written. Flushes buffers and
	/// commits the output (e.g., writes committed marker).
	virtual DuckDBResult<void> Finish() = 0;

	/// Signal that the operation was aborted. Cleans up any partial output.
	virtual DuckDBResult<void> Abort() = 0;

	/// Current memory usage in bytes (used for backpressure decisions).
	virtual size_t GetMemoryUsage() const = 0;

	/// Ensure the output schema metadata is written even when no data was
	/// produced (0-row exchange).  Called by PhysicalRemoteExchangeSink::Finalize
	/// after Finish() so that downstream readers can still open the exchange.
	virtual DuckDBResult<void> EnsureSchema(ClientContext &context, const vector<LogicalType> &types,
	                                        const vector<string> &names) {
		return DuckDBResult<void>::ok(); // default no-op
	}
};

} // namespace distributed
} // namespace duckdb
