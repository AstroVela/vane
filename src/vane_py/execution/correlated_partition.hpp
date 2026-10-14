// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/execution/physical_operator.hpp"

namespace duckdb {
namespace vane_execution {

struct CorrelatedKeySource {
	PhysicalOperator *source = nullptr;
	idx_t column = 0;
};

struct CorrelatedKeyFilter {
	CorrelatedKeySource scan;
	CorrelatedKeySource owner;
};

// Every entry is a complete, independently executable subtree. A valid column
// routes it by the common correlation key; an invalid column broadcasts it.
struct CorrelatedPartition {
	unordered_map<const PhysicalOperator *, optional_idx> inputs;
	// Independent inputs joined directly to this delimiter may be filtered by
	// a superset of its keys, before their partitioning exchange.
	unordered_map<const PhysicalOperator *, CorrelatedKeyFilter> filters;
};

// Prove locality before changing the physical plan or creating any fragments.
// Global/unsupported state keeps its ordinary single-partition native scope.
bool PlanCorrelatedPartition(PhysicalOperator &root, CorrelatedPartition &result);

} // namespace vane_execution
} // namespace duckdb
