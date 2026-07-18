// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/common_types.hpp"

#include <string>
#include <vector>

namespace duckdb {
class ClientContext;
class ColumnDataCollection;
class LogicalType;

namespace distributed {
struct FlightExchangeTicket;

struct FlightClientConfig {
	std::string location;
	double timeout_seconds = 0.0;
};

class FlightClient {
public:
	explicit FlightClient(FlightClientConfig config);

	const FlightClientConfig &config() const;

	DuckDBResult<void> Validate() const;
	DuckDBResult<std::unique_ptr<ColumnDataCollection>>
	FetchPartition(ClientContext &context, const FlightExchangeTicket &ticket,
	               const std::vector<LogicalType> &expected_types) const;

private:
	FlightClientConfig config_;
};

} // namespace distributed
} // namespace duckdb
