// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include "duckdb/common/vector.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"

namespace duckdb {
namespace distributed {

inline void FixHashJoinExpressionReferenceTypes(Expression &expr, const duckdb::vector<LogicalType> &input_types) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_REF) {
		auto &ref = expr.Cast<BoundReferenceExpression>();
		if (ref.index < input_types.size() && ref.return_type != input_types[ref.index]) {
			ref.return_type = input_types[ref.index];
		}
	}
	ExpressionIterator::EnumerateChildren(
	    expr, [&](Expression &child) { FixHashJoinExpressionReferenceTypes(child, input_types); });
}

inline void FixHashJoinConditionTypes(duckdb::vector<JoinCondition> &conditions,
                                      const duckdb::vector<LogicalType> &left_types,
                                      const duckdb::vector<LogicalType> &right_types,
                                      duckdb::vector<LogicalType> *condition_types) {
	if (condition_types) {
		condition_types->clear();
		condition_types->reserve(conditions.size());
	}
	for (auto &cond : conditions) {
		if (cond.left) {
			FixHashJoinExpressionReferenceTypes(*cond.left, left_types);
			if (condition_types) {
				condition_types->push_back(cond.left->return_type);
			}
		} else if (condition_types) {
			condition_types->emplace_back(LogicalTypeId::INVALID);
		}
		if (cond.right) {
			FixHashJoinExpressionReferenceTypes(*cond.right, right_types);
		}
	}
}

inline void FixHashJoinConditionTypes(duckdb::vector<JoinCondition> &conditions,
                                      const duckdb::vector<LogicalType> &left_types,
                                      const duckdb::vector<LogicalType> &right_types,
                                      duckdb::vector<LogicalType> &condition_types) {
	FixHashJoinConditionTypes(conditions, left_types, right_types, &condition_types);
}

inline void FixHashJoinConditionTypes(duckdb::vector<JoinCondition> &conditions,
                                      const duckdb::vector<LogicalType> &left_types,
                                      const duckdb::vector<LogicalType> &right_types) {
	FixHashJoinConditionTypes(conditions, left_types, right_types, nullptr);
}

inline void EnsureHashJoinProjectionColumns(PhysicalHashJoin::JoinProjectionColumns &cols,
                                            const duckdb::vector<LogicalType> &input_types) {
	bool invalid = cols.col_idxs.empty();
	for (auto idx : cols.col_idxs) {
		if (idx >= input_types.size()) {
			invalid = true;
			break;
		}
	}
	if (invalid) {
		cols.col_idxs.clear();
		cols.col_idxs.reserve(input_types.size());
		for (idx_t i = 0; i < input_types.size(); ++i) {
			cols.col_idxs.push_back(i);
		}
	}
	cols.col_types.clear();
	cols.col_types.reserve(cols.col_idxs.size());
	for (auto idx : cols.col_idxs) {
		if (idx < input_types.size()) {
			cols.col_types.push_back(input_types[idx]);
		} else {
			cols.col_types.emplace_back(LogicalTypeId::INVALID);
		}
	}
}

inline void FixHashJoinOutputColumnTypes(PhysicalHashJoin &join, const duckdb::vector<LogicalType> &left_types,
                                         const duckdb::vector<LogicalType> &right_types) {
	EnsureHashJoinProjectionColumns(join.lhs_output_columns, left_types);

	if (join.join_type == JoinType::ANTI || join.join_type == JoinType::SEMI || join.join_type == JoinType::MARK) {
		return;
	}

	EnsureHashJoinProjectionColumns(join.payload_columns, right_types);

	join.rhs_output_columns.col_types.clear();
	join.rhs_output_columns.col_types.reserve(join.rhs_output_columns.col_idxs.size());
	for (auto output_idx : join.rhs_output_columns.col_idxs) {
		if (output_idx < join.conditions.size()) {
			auto &cond = join.conditions[output_idx];
			if (cond.right) {
				join.rhs_output_columns.col_types.push_back(cond.right->return_type);
			} else {
				join.rhs_output_columns.col_types.emplace_back(LogicalTypeId::INVALID);
			}
			continue;
		}

		const auto payload_idx = output_idx - join.conditions.size();
		if (payload_idx < join.payload_columns.col_types.size()) {
			join.rhs_output_columns.col_types.push_back(join.payload_columns.col_types[payload_idx]);
		} else {
			join.rhs_output_columns.col_types.emplace_back(LogicalTypeId::INVALID);
		}
	}
}

inline void RepairHashJoinMetadataAfterChildAttach(PhysicalHashJoin &join,
                                                   const duckdb::vector<LogicalType> &left_types,
                                                   const duckdb::vector<LogicalType> &right_types) {
	FixHashJoinConditionTypes(join.conditions, left_types, right_types, join.condition_types);
	FixHashJoinOutputColumnTypes(join, left_types, right_types);

	duckdb::vector<LogicalType> join_types;
	join_types.reserve(join.lhs_output_columns.col_idxs.size() + join.rhs_output_columns.col_idxs.size() + 1);
	if (join.join_type != JoinType::RIGHT_SEMI && join.join_type != JoinType::RIGHT_ANTI) {
		for (auto idx : join.lhs_output_columns.col_idxs) {
			if (idx < left_types.size()) {
				join_types.push_back(left_types[idx]);
			}
		}
	}
	if (join.join_type != JoinType::ANTI && join.join_type != JoinType::SEMI && join.join_type != JoinType::MARK) {
		for (auto &col_type : join.rhs_output_columns.col_types) {
			join_types.push_back(col_type);
		}
	}
	if (join.join_type == JoinType::MARK) {
		join_types.push_back(LogicalType::BOOLEAN);
	}
	if (!join_types.empty()) {
		join.types = std::move(join_types);
	}
}

} // namespace distributed
} // namespace duckdb
