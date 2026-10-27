// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/execution/physical_plan_generator.hpp"

namespace duckdb {
namespace vane_execution {

bool ValidParquetMaskSize(idx_t words);
// Only unchanged integral keys through projections/filters can reach the scan.
bool CanFilterParquetKey(PhysicalOperator &probe, idx_t key);
// The complete word/mask input must finish before the Parquet source starts.
PhysicalOperator &MakeParquetMask(PhysicalPlan &plan, PhysicalOperator &probe, PhysicalOperator &domain, idx_t key,
                                  idx_t words);

} // namespace vane_execution
} // namespace duckdb
