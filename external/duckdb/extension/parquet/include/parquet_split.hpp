// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/open_file_info.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {
class ClientContext;
struct MultiFileBindData;

// Preserve each file reference and its options while dividing its row groups
// into independently readable, disjoint ranges. Repeated references stay repeated.
DUCKDB_API vector<OpenFileInfo> PlanParquetRowGroups(ClientContext &context, MultiFileBindData &bind, idx_t partitions);

// Distributed planning already reads the bound file footers to assign splits.
// Cost joins from their row counts, respecting explicit cardinality overrides.
DUCKDB_API idx_t ParquetScanCardinality(ClientContext &context, const MultiFileBindData &bind);

// Prove complete groups from known non-null integral ranges. Disjoint splits
// need no assignment constraint. Otherwise, co-locate each connected component
// of overlapping ranges, provided all partitions can receive balanced work.
// A nonempty assignment is authoritative and covers every split.
DUCKDB_API bool PlanParquetGroupPartitions(ClientContext &context, MultiFileBindData &bind,
                                           const vector<OpenFileInfo> &splits, idx_t column, idx_t partitions,
                                           vector<idx_t> &assignment,
                                           optional_ptr<pair<Value, Value>> bounds = nullptr);

struct ParquetPartitionColumn {
	MultiFileBindData &bind;
	idx_t column;
	vector<OpenFileInfo> splits;
};

// A matching key must have the same owner across both inputs. Publish both
// complete assignments together only when each input is balanced on its own.
DUCKDB_API bool PlanParquetJoinPartitions(ClientContext &context, const ParquetPartitionColumn &left,
                                          const ParquetPartitionColumn &right, idx_t partitions,
                                          vector<idx_t> &left_owners, vector<idx_t> &right_owners);
} // namespace duckdb
