// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/common_types.hpp"

#include <string>
#include <memory>
#include <thread>

namespace duckdb {
namespace distributed {

struct FlightServerConfig {
	std::string bind_host;
	int port = 0;
	std::string server_epoch;
};

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
