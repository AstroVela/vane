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

#include <string>
#include <utility>
#include <vector>
#include "duckdb/common/types.hpp"
#include "duckdb/execution/mark_join_build_summary.hpp"

namespace duckdb {
namespace distributed {

// ─── Context ─────────────────────────────────────────────

/// Context for creating one logical Exchange instance.
struct ExchangeContext {
	std::string query_id;
	std::string exchange_id;
};

// ─── Sink Handles ────────────────────────────────────────

/// Identifies the logical sink for one scheduler-owned task partition.
struct ExchangeSinkHandle {
	idx_t task_partition_id = DConstants::INVALID_INDEX;
};

/// Identifies one concrete sink attempt for a logical sink.
struct ExchangeSinkInstanceHandle {
	ExchangeSinkHandle sink_handle;
	idx_t attempt_id = 0;
	/// Logical input sequence for an order-preserving exchange. This is
	/// independent of the scheduler-owned task identity above.
	idx_t source_task_order = DConstants::INVALID_INDEX;
	/// Query that owns this concrete attempt.
	std::string query_id;
	/// Implementation-specific: output directory (Spooling),
	/// Flight server address (Flight), etc.
	std::string output_location;
	idx_t output_partition_count = 0;
	/// Cluster-internal host advertised by the worker that published this attempt.
	std::string flight_host;
	/// Process-local Flight service incarnation that published this attempt.
	std::string flight_server_epoch;
	/// Present only for a MARK-join build shuffle. This is the summary produced
	/// by this concrete sink attempt.
	MarkJoinBuildSummary mark_join_build_summary;
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
	/// Logical upstream sink partition that produced this source handle.
	/// Stable across attempts and independent of worker completion order.
	idx_t source_task_partition_id = DConstants::INVALID_INDEX;
	idx_t attempt_id = 0;
	std::string node_id;
	std::string flight_host;
	int flight_port = 0;
	std::string flight_server_epoch;
	std::vector<ExchangeSourceFile> files;
	/// Global summary reduced across all selected sink attempts. When present,
	/// every source handle for the exchange carries the same value.
	MarkJoinBuildSummary mark_join_build_summary;
};

} // namespace distributed
} // namespace duckdb
