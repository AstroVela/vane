// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/exchange/flight_ticket.hpp"

#include <algorithm>
#include <limits>
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

DuckDBResult<idx_t> ParseIndex(const std::string &value, const std::string &field) {
	if (value.empty() || std::any_of(value.begin(), value.end(), [](char ch) { return ch < '0' || ch > '9'; })) {
		return DuckDBResult<idx_t>::err(DuckDBError::value_error("invalid flight ticket " + field));
	}
	try {
		size_t parsed_chars = 0;
		auto parsed = std::stoull(value, &parsed_chars);
		if (parsed_chars != value.size() ||
		    parsed > static_cast<unsigned long long>(std::numeric_limits<idx_t>::max())) {
			return DuckDBResult<idx_t>::err(DuckDBError::value_error("invalid flight ticket " + field));
		}
		return DuckDBResult<idx_t>::ok(static_cast<idx_t>(parsed));
	} catch (const std::exception &) {
		return DuckDBResult<idx_t>::err(DuckDBError::value_error("invalid flight ticket " + field));
	}
}

} // namespace

std::string FlightExchangeTicket::Serialize() const {
	std::ostringstream ss;
	ss << kTicketVersion << '\n'
	   << server_epoch << '\n'
	   << exchange_instance_id << '\n'
	   << node_id << '\n'
	   << attempt_id << '\n'
	   << partition_idx;
	return ss.str();
}

DuckDBResult<FlightExchangeTicket> FlightExchangeTicket::Parse(const std::string &ticket) {
	auto parts = SplitLines(ticket);
	if (parts.size() != 6) {
		return DuckDBResult<FlightExchangeTicket>::err(DuckDBError::value_error("invalid flight ticket format"));
	}
	if (parts[0] != kTicketVersion) {
		return DuckDBResult<FlightExchangeTicket>::err(DuckDBError::value_error("unsupported flight ticket version"));
	}
	if (parts[1].empty() || parts[2].empty() || parts[3].empty()) {
		return DuckDBResult<FlightExchangeTicket>::err(
		    DuckDBError::value_error("flight ticket missing server epoch, exchange instance, or node id"));
	}

	auto attempt_id = ParseIndex(parts[4], "attempt id");
	if (attempt_id.is_err()) {
		return DuckDBResult<FlightExchangeTicket>::err(attempt_id.error());
	}
	auto partition_idx = ParseIndex(parts[5], "partition index");
	if (partition_idx.is_err()) {
		return DuckDBResult<FlightExchangeTicket>::err(partition_idx.error());
	}

	FlightExchangeTicket result;
	result.server_epoch = parts[1];
	result.exchange_instance_id = parts[2];
	result.node_id = parts[3];
	result.attempt_id = attempt_id.value();
	result.partition_idx = partition_idx.value();
	return DuckDBResult<FlightExchangeTicket>::ok(std::move(result));
}

} // namespace distributed
} // namespace duckdb
