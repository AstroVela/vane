// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/projection/physical_vllm.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/common/types/value.hpp"

namespace duckdb {

class PhysicalVLLM : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::VLLM_PROJECT;

public:
	PhysicalVLLM(PhysicalPlan &physical_plan, vector<LogicalType> types, unique_ptr<Expression> prompt_expr,
	             string model, Value options, string output_column_name, idx_t estimated_cardinality);

	unique_ptr<Expression> prompt_expr;
	string model;
	Value options;
	string output_column_name;

public:
	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate, OperatorState &state) const override;
	OperatorFinalizeResultType FinalExecute(ExecutionContext &context, DataChunk &chunk, GlobalOperatorState &gstate,
	                                        OperatorState &state) const override;

	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;
	unique_ptr<GlobalOperatorState> GetGlobalOperatorState(ClientContext &context) const override;

	bool RequiresFinalExecute() const override {
		return true;
	}

	OrderPreservationType OperatorOrder() const override {
		return OrderPreservationType::NO_ORDER;
	}

	bool ParallelOperator() const override {
		return true;
	}

	InsertionOrderPreservingMap<string> ParamsToString() const override;

protected:
	void SerializeOperatorData(Serializer &serializer) const override;
};

} // namespace duckdb
