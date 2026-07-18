// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/projection/physical_tableinout_function.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/table_function.hpp"

namespace duckdb {

class PhysicalTableInOutFunction : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::INOUT_FUNCTION;

public:
	PhysicalTableInOutFunction(PhysicalPlan &physical_plan, vector<LogicalType> types, TableFunction function_p,
	                           unique_ptr<FunctionData> bind_data_p, vector<ColumnIndex> column_ids_p,
	                           idx_t estimated_cardinality, vector<column_t> projected_input);

public:
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;
	unique_ptr<GlobalOperatorState> GetGlobalOperatorState(ClientContext &context) const override;
	static void SetOrdinality(DataChunk &chunk, const optional_idx &ordinality_column_idx, const idx_t &ordinality_idx,
	                          const idx_t &ordinality);
	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate, OperatorState &state) const override;
	OperatorResultType ExecuteBatch(ExecutionContext &context, ExecutionBatch &input, ExecutionBatch &output,
	                                GlobalOperatorState &gstate, OperatorState &state) const override;
	OperatorFinalizeResultType FinalExecute(ExecutionContext &context, DataChunk &chunk, GlobalOperatorState &gstate,
	                                        OperatorState &state) const override;
	OperatorFinalizeResultType FinalExecuteBatch(ExecutionContext &context, ExecutionBatch &batch,
	                                             GlobalOperatorState &gstate, OperatorState &state) const override;
	void SerializeOperatorData(Serializer &serializer) const override;
	const TableFunction &GetFunction() const {
		return function;
	}
	const FunctionData *GetBindData() const {
		return bind_data.get();
	}
	const vector<ColumnIndex> &GetColumnIds() const {
		return column_ids;
	}
	const vector<column_t> &GetProjectedInput() const {
		return projected_input;
	}

	bool ParallelOperator() const override {
		return true;
	}

	bool RequiresFinalExecute() const override {
		return function.in_out_function_final;
	}

	InsertionOrderPreservingMap<string> ParamsToString() const override;
	InsertionOrderPreservingMap<string> ExtraOperatorParams(GlobalOperatorState &gstate,
	                                                        OperatorState &state) const override;

	//! Information for WITH ORDINALITY
	optional_idx ordinality_idx;

	OrderPreservationType OperatorOrder() const override {
		return function.order_preservation_type;
	}

private:
	//! The table function
	TableFunction function;
	//! Bind data of the function
	unique_ptr<FunctionData> bind_data;
	//! The set of column ids to fetch
	vector<ColumnIndex> column_ids;
	//! The set of input columns to project out
	vector<column_t> projected_input;
};

} // namespace duckdb
