// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/common_types.hpp"

#include <chrono>
#include <string>
#include <memory>
#include <thread>

namespace duckdb {
namespace distributed {

struct FlightServerConfig {
	static constexpr std::chrono::milliseconds DEFAULT_SHUTDOWN_GRACE_PERIOD {5000};

	std::string bind_host;
	int port = 0;
	std::string server_epoch;
	std::chrono::milliseconds shutdown_grace_period {DEFAULT_SHUTDOWN_GRACE_PERIOD};
};

//! Build a Flight TCP URI, adding URI brackets around an IPv6 host.
std::string BuildFlightLocation(const std::string &host, int port);

class FlightServer {
public:
	explicit FlightServer(FlightServerConfig config);
	~FlightServer();

	const FlightServerConfig &config() const;
	int port() const;

	DuckDBResult<void> Start();
	DuckDBResult<void> Stop();

private:
	DuckDBResult<void> StartInternal();

	FlightServerConfig config_;
	std::unique_ptr<class FlightServerImpl> impl_;
	std::thread server_thread_;
};

} // namespace distributed
} // namespace duckdb
