// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/function/aggregate_function.hpp"

namespace duckdb {

// Exact component sums across distributed partitions; preserve native AVG rounding.
DUCKDB_API AggregateFunction WideIntervalAvgFunction();

} // namespace duckdb
