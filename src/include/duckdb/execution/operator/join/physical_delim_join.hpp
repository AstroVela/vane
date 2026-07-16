// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/join/physical_delim_join.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/physical_operator.hpp"

namespace duckdb {

class PhysicalHashAggregate;

struct DelimJoinDeserializeTag {
	constexpr DelimJoinDeserializeTag() = default;
};

//! PhysicalDelimJoin represents a join where either the LHS or RHS will be duplicate eliminated and pushed into a
//! PhysicalColumnDataScan in the other side. Implementations are PhysicalLeftDelimJoin and PhysicalRightDelimJoin
class PhysicalDelimJoin : public PhysicalOperator {
public:
	PhysicalDelimJoin(PhysicalPlan &physical_plan, PhysicalOperatorType type, vector<LogicalType> types,
	                  PhysicalOperator &original_join, PhysicalOperator &distinct,
	                  const vector<const_reference<PhysicalOperator>> &delim_scans, idx_t estimated_cardinality,
	                  optional_idx delim_idx);

	PhysicalOperator &join;
	PhysicalHashAggregate &distinct;
	vector<const_reference<PhysicalOperator>> delim_scans;

	optional_idx delim_idx;

public:
	vector<const_reference<PhysicalOperator>> GetChildren() const override;

	bool IsSink() const override {
		return true;
	}
	bool ParallelSink() const override {
		return true;
	}
	OrderPreservationType SourceOrder() const override {
		return OrderPreservationType::NO_ORDER;
	}
	bool SinkOrderDependent() const override {
		return false;
	}

	InsertionOrderPreservingMap<string> ParamsToString() const override;

protected:
	void SerializeOperatorData(Serializer &serializer) const override;
};

} // namespace duckdb
