// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/execution/physical_operator.hpp"

namespace duckdb {
namespace vane_execution {

// Every entry is a complete, independently executable subtree. A valid column
// routes it by the common correlation key; an invalid column broadcasts it.
struct CorrelatedPartition {
	unordered_map<const PhysicalOperator *, optional_idx> inputs;
};

// Prove locality before changing the physical plan or creating any fragments.
// Global/unsupported state keeps its ordinary single-partition native scope.
bool PlanCorrelatedPartition(PhysicalOperator &root, CorrelatedPartition &result);

} // namespace vane_execution
} // namespace duckdb
