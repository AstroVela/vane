// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/enums/join_type.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {
namespace distributed {

inline bool JoinOutputsLeft(JoinType join_type) {
	return join_type != JoinType::RIGHT_SEMI && join_type != JoinType::RIGHT_ANTI;
}

inline bool JoinOutputsRight(JoinType join_type) {
	return join_type != JoinType::ANTI && join_type != JoinType::SEMI && join_type != JoinType::MARK;
}

// The side inputs must already contain the projection-resolved output columns.
// This helper applies only the join-type shape: left, right, or left plus MARK.
inline duckdb::vector<LogicalType> BuildJoinOutputTypes(JoinType join_type,
                                                        const duckdb::vector<LogicalType> &left_output_types,
                                                        const duckdb::vector<LogicalType> &right_output_types) {
	duckdb::vector<LogicalType> result;
	result.reserve((JoinOutputsLeft(join_type) ? left_output_types.size() : 0) +
	               (JoinOutputsRight(join_type) ? right_output_types.size() : 0) +
	               (join_type == JoinType::MARK ? 1 : 0));
	if (JoinOutputsLeft(join_type)) {
		result.insert(result.end(), left_output_types.begin(), left_output_types.end());
	}
	if (JoinOutputsRight(join_type)) {
		result.insert(result.end(), right_output_types.begin(), right_output_types.end());
	}
	if (join_type == JoinType::MARK) {
		result.push_back(LogicalType::BOOLEAN);
	}
	return result;
}

} // namespace distributed
} // namespace duckdb
