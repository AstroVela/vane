// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/projection/physical_pivot.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/common/string_map_set.hpp"
#include "duckdb/planner/tableref/bound_pivotref.hpp"

namespace duckdb {

//! PhysicalPivot implements the physical PIVOT operation
class PhysicalPivot : public PhysicalOperator {
public:
	PhysicalPivot(PhysicalPlan &physical_plan, vector<LogicalType> types, PhysicalOperator &child,
	              BoundPivotInfo bound_pivot);
	PhysicalPivot(PhysicalPlan &physical_plan, vector<LogicalType> types, BoundPivotInfo bound_pivot,
	              idx_t estimated_cardinality);

	BoundPivotInfo bound_pivot;
	//! The map for pivot value -> column index
	string_map_t<idx_t> pivot_map;
	//! The empty aggregate values
	vector<Value> empty_aggregates;

public:
	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate, OperatorState &state) const override;

	bool ParallelOperator() const override {
		return true;
	}

protected:
	void SerializeOperatorData(Serializer &serializer) const override;
};

} // namespace duckdb
