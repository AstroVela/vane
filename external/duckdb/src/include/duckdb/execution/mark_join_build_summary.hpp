// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

namespace duckdb {

//! Global state required to evaluate an uncorrelated partitioned MARK join.
//! The three valid states are:
//! - no build rows: unmatched probes, including NULL probes, are false;
//! - build rows without NULL: unmatched probes are false and NULL probes are NULL;
//! - build rows with NULL: every unmatched probe is NULL.
struct MarkJoinBuildSummary {
	bool valid = false;
	bool has_rows = false;
	bool has_null = false;

	static MarkJoinBuildSummary Create(bool has_rows_p, bool has_null_p) {
		MarkJoinBuildSummary result;
		result.valid = true;
		result.has_rows = has_rows_p;
		result.has_null = has_null_p;
		return result;
	}

	bool IsValid() const {
		return valid && (!has_null || has_rows);
	}

	void Merge(const MarkJoinBuildSummary &other) {
		if (!other.valid) {
			return;
		}
		if (!valid) {
			*this = other;
			return;
		}
		has_rows = has_rows || other.has_rows;
		has_null = has_null || other.has_null;
	}

	bool operator==(const MarkJoinBuildSummary &other) const {
		return valid == other.valid && has_rows == other.has_rows && has_null == other.has_null;
	}
};

} // namespace duckdb
