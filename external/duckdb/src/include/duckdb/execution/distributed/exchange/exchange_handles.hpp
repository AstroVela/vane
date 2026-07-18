// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file exchange_handles.hpp
 * @brief Handle types for the Exchange abstraction layer.
 *
 * Exchange handle types. Handles are lightweight identity objects that travel
 * between coordinator and workers.
 */

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "duckdb/common/types.hpp"

namespace duckdb {
namespace distributed {

// ─── Context ─────────────────────────────────────────────

/// Context for creating an Exchange instance (one per shuffle stage).
struct ExchangeContext {
	std::string query_id;
	std::string exchange_id; // typically shuffle_stage_id
};

// ─── Sink Handles ────────────────────────────────────────

/// Identifies a logical sink within an Exchange (one per task partition).
/// Identifies the logical sink for one task partition.
struct ExchangeSinkHandle {
	idx_t task_partition_id = 0;
};

/// Identifies a concrete sink instance (supports retries via attempt_id).
/// Identifies one concrete sink attempt for a logical sink.
struct ExchangeSinkInstanceHandle {
	ExchangeSinkHandle sink_handle;
	idx_t attempt_id = 0;
	/// Implementation-specific: output directory (Spooling),
	/// Flight server address (Flight), etc.
	std::string output_location;
	idx_t output_partition_count = 0;
};

// ─── Source Handles ──────────────────────────────────────

/// A file/location that an ExchangeSource should read from.
struct ExchangeSourceFile {
	ExchangeSourceFile() = default;
	ExchangeSourceFile(std::string path_p, idx_t rows_p, size_t file_size_p = 0)
	    : path(std::move(path_p)), rows(rows_p), file_size(file_size_p) {
	}

	std::string path; // local path or Flight URI
	idx_t rows = 0;
	size_t file_size = 0;
};

/// Identifies a unit of data for an ExchangeSource to consume.
/// One SourceHandle may cover part of a partition (large partitions are
/// split by target_data_size).
struct ExchangeSourceHandle {
	idx_t partition_id = 0;
	idx_t attempt_id = 0;
	std::string node_id;
	int flight_port = 0;
	std::vector<ExchangeSourceFile> files;
};

} // namespace distributed
} // namespace duckdb
