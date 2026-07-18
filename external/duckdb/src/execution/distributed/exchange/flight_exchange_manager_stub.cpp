// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/exchange/flight_exchange_manager.hpp"

#include "duckdb/common/exception.hpp"

#include <utility>

namespace duckdb {
namespace distributed {

namespace {

void ThrowFlightExchangeDisabled() {
	throw NotImplementedException("Flight exchange is disabled. Rebuild with BUILD_DISTRIBUTED_EXCHANGE=ON.");
}

} // namespace

FlightExchangeManager::FlightExchangeManager(FlightExchangeConfig config, ClientContext *context)
    : config_(std::move(config)), context_(context) {
}

FlightExchangeManager::~FlightExchangeManager() {
	Shutdown();
}

void FlightExchangeManager::RefreshRuntimeNodeId() {
}

std::unique_ptr<Exchange> FlightExchangeManager::CreateExchange(const ExchangeContext &, idx_t) {
	ThrowFlightExchangeDisabled();
	return std::unique_ptr<Exchange>();
}

std::unique_ptr<ExchangeSink> FlightExchangeManager::CreateSink(const ExchangeSinkInstanceHandle &) {
	ThrowFlightExchangeDisabled();
	return std::unique_ptr<ExchangeSink>();
}

std::unique_ptr<ExchangeSource> FlightExchangeManager::CreateSource() {
	ThrowFlightExchangeDisabled();
	return std::unique_ptr<ExchangeSource>();
}

int FlightExchangeManager::GetLocalFlightServerPort() {
	return 0;
}

void FlightExchangeManager::Shutdown() {
}

} // namespace distributed
} // namespace duckdb
