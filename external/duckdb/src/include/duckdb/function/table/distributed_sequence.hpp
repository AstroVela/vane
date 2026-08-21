// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/table/distributed_sequence.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/function/distributed_table_function.hpp"

namespace duckdb {

static constexpr idx_t DISTRIBUTED_SEQUENCE_PROTOCOL_VERSION = 1;
static constexpr idx_t DISTRIBUTED_SEQUENCE_TASK_CODEC_VERSION = 1;
static constexpr const char *DISTRIBUTED_SEQUENCE_TASK_CODEC = "vane.sequence-shard";

//! A half-open slice of a deterministic sequence. The sequence definition
//! itself stays in the serialized worker bind; tasks carry only row offsets.
struct DistributedSequenceShard {
	idx_t ordinal = 0;
	idx_t row_offset = 0;
	idx_t row_count = 0;
	bool exact_count = false;
};

//! Split an exactly counted sequence across the scheduler-selected target, or
//! emit one explicit task for a calendar-sensitive sequence whose count cannot
//! be indexed without walking it.
DUCKDB_API vector<DistributedScanTask> PlanDistributedSequenceTasks(idx_t cardinality, bool exact_count,
                                                                    idx_t target_task_count);

//! Decode and validate the complete assignment for one worker. Empty
//! assignments are valid. Exact assignments may be non-contiguous because the
//! scheduler is free to group elementary tasks.
DUCKDB_API vector<DistributedSequenceShard> DecodeDistributedSequenceTasks(const vector<DistributedScanTask> &tasks,
                                                                           idx_t cardinality, bool exact_count);

//! Exact arithmetic shared by indexable BIGINT, TIMESTAMP, and TIMESTAMPTZ
//! sequences.
DUCKDB_API idx_t ComputeDistributedSequenceCardinality(int64_t start, int64_t end, int64_t increment, bool inclusive);
DUCKDB_API int64_t GetDistributedSequenceValue(int64_t start, int64_t increment, idx_t row_offset);

} // namespace duckdb
