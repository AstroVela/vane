// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/function/aggregate_function.hpp"

namespace duckdb {

// Exact accumulation for complete distributed groups with 128-bit input.
// Finalization preserves the SQL return type and DECIMAL scale.
DUCKDB_API AggregateFunction WideIntegerSumFunction(const LogicalType &input_type);
DUCKDB_API AggregateFunction WideIntegerAvgFunction(const LogicalType &input_type);

} // namespace duckdb
