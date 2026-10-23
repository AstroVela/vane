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

// True only when the given integral column has known, non-null, disjoint ranges
// across these exact splits. Every group containing this key is then complete
// within one split, regardless of which worker receives it.
DUCKDB_API bool ParquetSplitRangesDisjoint(ClientContext &context, MultiFileBindData &bind,
                                           const vector<OpenFileInfo> &splits, idx_t column,
                                           optional_ptr<pair<Value, Value>> bounds = nullptr);
} // namespace duckdb
