// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file exchange_manager.hpp
 * @brief Abstract ExchangeManager — top-level factory for exchange objects.
 *
 * Main entry point for pluggable exchange implementations. Different
 * ExchangeManager implementations provide different transport strategies:
 *
 *   - FlightExchangeManager:  memory-first + Arrow Flight streaming
 *   - LocalExchangeManager:   intra-process zero-copy
 *   - SpoolingExchangeManager: disk-first + storage SPI (lowest priority)
 */

#pragma once

#include "duckdb/execution/distributed/exchange/exchange.hpp"
#include "duckdb/execution/distributed/exchange/exchange_handles.hpp"
#include "duckdb/execution/distributed/exchange/exchange_sink.hpp"
#include "duckdb/execution/distributed/exchange/exchange_source.hpp"

#include <memory>

namespace duckdb {
namespace distributed {

/// Abstract factory for creating Exchange, ExchangeSink, and ExchangeSource
/// instances.
///
/// Usage:
///   auto mgr = CreateFlightExchangeManager(config);
///   auto exchange = mgr->CreateExchange(ctx, num_partitions);
///   // ... coordinator registers sinks, workers write/read data ...
///
class ExchangeManager {
public:
	virtual ~ExchangeManager() = default;

	/// Create an Exchange for a shuffle stage.
	/// Called once per shuffle stage on the coordinator.
	///
	/// @param ctx                    Context with query/exchange identifiers
	/// @param output_partition_count Number of output partitions for hash/range partitioning
	/// @return Unique pointer to Exchange coordinator object
	virtual std::unique_ptr<Exchange> CreateExchange(const ExchangeContext &ctx, idx_t output_partition_count) = 0;

	/// Create an ExchangeSink instance for a worker task.
	///
	/// @param handle  Instance handle from Exchange::InstantiateSink()
	/// @return Unique pointer to ExchangeSink
	virtual std::unique_ptr<ExchangeSink> CreateSink(const ExchangeSinkInstanceHandle &handle) = 0;

	/// Create an ExchangeSource instance for a downstream task.
	///
	/// @return Unique pointer to ExchangeSource
	virtual std::unique_ptr<ExchangeSource> CreateSource() = 0;

	/// Set the client context (used when manager is created during deserialization
	/// and context is only available later in GetGlobalSinkState).
	virtual void SetContext(ClientContext *ctx) {
	}

	/// Shut down the exchange manager and release global resources.
	virtual void Shutdown() = 0;
};

} // namespace distributed
} // namespace duckdb
