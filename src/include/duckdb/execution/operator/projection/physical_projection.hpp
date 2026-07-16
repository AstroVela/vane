// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/projection/physical_projection.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/expression.hpp"

namespace duckdb {

class PhysicalProjection : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::PROJECTION;

public:
	PhysicalProjection(PhysicalPlan &physical_plan, vector<LogicalType> types,
	                   vector<unique_ptr<Expression>> select_list, idx_t estimated_cardinality);

	vector<unique_ptr<Expression>> select_list;

public:
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;
	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate, OperatorState &state) const override;
	OperatorResultType ExecuteBatch(ExecutionContext &context, ExecutionBatch &input, ExecutionBatch &output,
	                                GlobalOperatorState &gstate, OperatorState &state) const override;

	bool ParallelOperator() const override {
		return true;
	}

	InsertionOrderPreservingMap<string> ParamsToString() const override;

protected:
	//! Serialize projection-specific data (select_list)
	void SerializeOperatorData(Serializer &serializer) const override;
};

} // namespace duckdb
