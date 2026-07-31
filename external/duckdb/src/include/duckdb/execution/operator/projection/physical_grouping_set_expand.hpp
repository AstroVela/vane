// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/parser/group_by_node.hpp"
#include "duckdb/planner/expression.hpp"

namespace duckdb {

//! Expands every input row once per grouping-set occurrence.
//!
//! The output layout is:
//!   input columns, expanded group columns, GROUPING() values,
//!   grouping-set occurrence id, aggregate filter columns.
class PhysicalGroupingSetExpand : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::GROUPING_SET_EXPAND;

public:
	PhysicalGroupingSetExpand(PhysicalPlan &physical_plan, vector<LogicalType> types,
	                          vector<unique_ptr<Expression>> groups, vector<GroupingSet> grouping_sets,
	                          vector<vector<idx_t>> grouping_functions, vector<idx_t> filter_indexes,
	                          idx_t input_column_count, bool emit_empty_grouping_sets, idx_t estimated_cardinality);

	vector<unique_ptr<Expression>> groups;
	vector<GroupingSet> grouping_sets;
	vector<vector<idx_t>> grouping_functions;
	vector<idx_t> filter_indexes;
	idx_t input_column_count;
	bool emit_empty_grouping_sets;

public:
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;
	unique_ptr<GlobalOperatorState> GetGlobalOperatorState(ClientContext &context) const override;

	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate, OperatorState &state) const override;
	OperatorFinalizeResultType FinalExecute(ExecutionContext &context, DataChunk &chunk, GlobalOperatorState &gstate,
	                                        OperatorState &state) const override;

	bool RequiresFinalExecute() const override {
		return true;
	}

	bool ParallelOperator() const override {
		return true;
	}

	InsertionOrderPreservingMap<string> ParamsToString() const override;

protected:
	void SerializeOperatorData(Serializer &serializer) const override;
};

} // namespace duckdb
