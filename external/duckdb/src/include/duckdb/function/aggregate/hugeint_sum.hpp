// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/function/aggregate_function.hpp"

namespace duckdb {

// Exact accumulation for a complete distributed group. The result remains
// HUGEINT, with its range checked only after all rows have been combined.
DUCKDB_API AggregateFunction WideHugeintSumFunction();

} // namespace duckdb
