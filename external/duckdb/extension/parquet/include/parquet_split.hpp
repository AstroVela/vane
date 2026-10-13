// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/open_file_info.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {
class ClientContext;
struct MultiFileBindData;

// Preserve each file reference and its options while dividing its row groups
// into independently readable, disjoint ranges. Repeated references stay repeated.
DUCKDB_API vector<OpenFileInfo> PlanParquetRowGroups(ClientContext &context, MultiFileBindData &bind, idx_t partitions);
} // namespace duckdb
