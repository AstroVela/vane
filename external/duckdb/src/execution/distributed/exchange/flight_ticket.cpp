// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/exchange/flight_ticket.hpp"

#include <sstream>
#include <vector>

namespace duckdb {
namespace distributed {

namespace {

constexpr const char *kTicketVersion = "v1";

std::vector<std::string> SplitLines(const std::string &input) {
	std::vector<std::string> parts;
	std::string current;
	current.reserve(input.size());
	for (char ch : input) {
		if (ch == '\n') {
			parts.push_back(current);
			current.clear();
		} else {
			current.push_back(ch);
		}
	}
	parts.push_back(current);
	return parts;
}

} // namespace

std::string FlightExchangeTicket::Serialize() const {
	std::ostringstream ss;
	ss << kTicketVersion << '\n' << shuffle_stage_id << '\n' << node_id << '\n' << partition_idx;
	return ss.str();
}

DuckDBResult<FlightExchangeTicket> FlightExchangeTicket::Parse(const std::string &ticket) {
	auto parts = SplitLines(ticket);
	if (parts.size() != 4) {
		return DuckDBResult<FlightExchangeTicket>::err(DuckDBError::value_error("invalid flight ticket format"));
	}
	if (parts[0] != kTicketVersion) {
		return DuckDBResult<FlightExchangeTicket>::err(DuckDBError::value_error("unsupported flight ticket version"));
	}
	if (parts[1].empty() || parts[2].empty()) {
		return DuckDBResult<FlightExchangeTicket>::err(
		    DuckDBError::value_error("flight ticket missing stage or node id"));
	}

	idx_t partition_idx = 0;
	try {
		auto parsed = std::stoll(parts[3]);
		if (parsed < 0) {
			return DuckDBResult<FlightExchangeTicket>::err(
			    DuckDBError::value_error("flight ticket partition index is negative"));
		}
		partition_idx = static_cast<idx_t>(parsed);
	} catch (const std::exception &) {
		return DuckDBResult<FlightExchangeTicket>::err(
		    DuckDBError::value_error("flight ticket partition index parse failed"));
	}

	FlightExchangeTicket result;
	result.shuffle_stage_id = parts[1];
	result.node_id = parts[2];
	result.partition_idx = partition_idx;
	return DuckDBResult<FlightExchangeTicket>::ok(std::move(result));
}

} // namespace distributed
} // namespace duckdb
