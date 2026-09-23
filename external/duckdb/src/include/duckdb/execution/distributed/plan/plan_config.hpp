// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>

#include "duckdb/common/shared_ptr.hpp"
#include "duckdb/execution/distributed/common_types.hpp"

namespace duckdb {
class DatabaseInstance;
namespace distributed {

struct PlanConfig {
	uint16_t query_idx = 0;
	std::string query_id;
	DuckDBExecutionConfigRef config;
	shared_ptr<DatabaseInstance> db;
	size_t num_partitions = 1;
	size_t max_concurrent_tasks = 0;
	//! Metadata collection may represent native scans without a worker-executable scan plan.
	bool native_scan_metadata = false;

	PlanConfig() = default;
	PlanConfig(uint16_t idx, std::string qid, DuckDBExecutionConfigRef cfg, size_t partitions = 1)
	    : query_idx(idx), query_id(std::move(qid)), config(std::move(cfg)), num_partitions(partitions) {
	}
};

} // namespace distributed
} // namespace duckdb
