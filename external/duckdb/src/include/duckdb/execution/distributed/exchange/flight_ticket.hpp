// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/common_types.hpp"

#include <string>

namespace duckdb {
namespace distributed {

struct FlightExchangeTicket {
	std::string shuffle_stage_id;
	std::string node_id;
	idx_t partition_idx = 0;

	std::string Serialize() const;
	static DuckDBResult<FlightExchangeTicket> Parse(const std::string &ticket);
};

} // namespace distributed
} // namespace duckdb
