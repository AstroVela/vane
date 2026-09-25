// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/physical_operator.hpp"

namespace duckdb {

//! Marks and validates a Python DataSink terminal without changing its rows.
//! Local execution validates before result materialization. Distributed
//! translation appends the same bounded validator to every worker task and
//! uses a coordinator-owned collector for the aggregate bound.
class PhysicalDataSink : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::DATA_SINK;

	PhysicalDataSink(PhysicalPlan &physical_plan, vector<LogicalType> types, string operation_id,
	                 idx_t estimated_cardinality);

	string operation_id;

	string GetName() const override {
		return "DATA_SINK";
	}

	InsertionOrderPreservingMap<string> ParamsToString() const override {
		InsertionOrderPreservingMap<string> result;
		result["Operation Id"] = operation_id;
		return result;
	}

	unique_ptr<GlobalOperatorState> GetGlobalOperatorState(ClientContext &context) const override;
	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &global_state, OperatorState &operator_state) const override;

	bool ParallelOperator() const override {
		return true;
	}

protected:
	void SerializeOperatorData(Serializer &serializer) const override;
};

} // namespace duckdb
