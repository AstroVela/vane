// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file exchange_source.hpp
 * @brief Abstract ExchangeSource interface for reading data from an exchange.
 *
 * Each downstream task creates one ExchangeSource and adds source handles to
 * it. The source reads data from those handles, possibly from multiple remote
 * nodes.
 */

#pragma once

#include "duckdb/common/types.hpp"
#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/execution/distributed/exchange/exchange_handles.hpp"

#include <vector>

namespace duckdb {

class DataChunk;

namespace distributed {

/// Abstract interface for reading exchange data.
///
/// Lifecycle: Created by ExchangeManager::CreateSource() →
///            AddSourceHandles() → Read loop until IsFinished() → Close().
///
/// Threading: A single ExchangeSource is used by one operator thread.
class ExchangeSource {
public:
	virtual ~ExchangeSource() = default;

	/// Add source handles that this source should read from.
	/// May be called multiple times (incremental handle discovery).
	virtual void AddSourceHandles(std::vector<ExchangeSourceHandle> handles) = 0;

	/// Read the next chunk of data into the provided DataChunk.
	///
	/// @param chunk  Output chunk (will be reset and filled)
	/// @return true if data was returned, false if no data available yet
	///         (call IsBlocked/WaitUnblocked) or source is finished.
	virtual bool ReadChunk(DataChunk &chunk) = 0;

	/// Check whether the source is currently blocked waiting for data.
	virtual bool IsBlocked() const = 0;

	/// Block until data becomes available or the source finishes.
	virtual void WaitUnblocked() = 0;

	/// Whether all data has been read from all source handles.
	virtual bool IsFinished() const = 0;

	/// Current memory usage in bytes.
	virtual size_t GetMemoryUsage() const = 0;

	/// Close the source and release resources.
	virtual void Close() = 0;
};

} // namespace distributed
} // namespace duckdb
