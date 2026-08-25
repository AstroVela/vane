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
static constexpr idx_t DISTRIBUTED_SEQUENCE_SPLIT_CODEC_VERSION = 1;
static constexpr const char *DISTRIBUTED_SEQUENCE_SPLIT_CODEC = "vane.sequence-split";

//! A half-open slice of a deterministic sequence. The sequence definition
//! itself stays in the serialized worker bind; splits carry only row offsets.
struct DistributedSequenceSplit {
	idx_t ordinal = 0;
	idx_t row_offset = 0;
	idx_t row_count = 0;
	bool exact_count = false;
};

//! Split an exactly counted sequence across the scheduler-selected target, or
//! emit one explicit split for a calendar-sensitive sequence whose count cannot
//! be indexed without walking it.
DUCKDB_API vector<DistributedScanSplit> PlanDistributedSequenceSplits(idx_t cardinality, bool exact_count,
                                                                      idx_t target_split_count);

//! Decode and validate the complete assignment for one worker. Empty
//! assignments are valid. Exact assignments may be non-contiguous because the
//! scheduler is free to group elementary splits.
DUCKDB_API vector<DistributedSequenceSplit> DecodeDistributedSequenceSplits(const vector<DistributedScanSplit> &splits,
                                                                            idx_t cardinality, bool exact_count);

DUCKDB_API bool DistributedSequenceSplitsEqual(const vector<DistributedSequenceSplit> &left,
                                               const vector<DistributedSequenceSplit> &right);

//! Exact arithmetic shared by indexable BIGINT and TIMESTAMP sequences.
DUCKDB_API idx_t ComputeDistributedSequenceCardinality(int64_t start, int64_t end, int64_t increment, bool inclusive);
DUCKDB_API int64_t GetDistributedSequenceValue(int64_t start, int64_t increment, idx_t row_offset);

} // namespace duckdb
