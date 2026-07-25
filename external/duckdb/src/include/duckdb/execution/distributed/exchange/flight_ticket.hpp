// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/common_types.hpp"

#include <string>

namespace duckdb {
namespace distributed {

struct FlightExchangeTicket {
	std::string server_epoch;
	std::string exchange_instance_id;
	std::string node_id;
	idx_t attempt_id = 0;
	idx_t partition_idx = 0;

	std::string Serialize() const;
	static DuckDBResult<FlightExchangeTicket> Parse(const std::string &ticket);
};

} // namespace distributed
} // namespace duckdb
