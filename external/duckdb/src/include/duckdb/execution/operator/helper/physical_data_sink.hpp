// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/physical_operator.hpp"

namespace duckdb {

//! Marks a Python DataSink terminal without changing local DuckDB results.
//! Distributed translation replaces this passthrough operator with a
//! coordinator-owned result collector; workers execute only its child plan.
class PhysicalDataSink : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::DATA_SINK;

	PhysicalDataSink(PhysicalPlan &physical_plan, vector<LogicalType> types, string operation_id,
	                 idx_t estimated_cardinality)
	    : PhysicalOperator(physical_plan, TYPE, std::move(types), estimated_cardinality),
	      operation_id(std::move(operation_id)) {
		if (this->operation_id.empty() || this->operation_id.size() > 256) {
			throw InvalidInputException("DataSink operation identity must contain 1 to 256 UTF-8 bytes");
		}
	}

	string operation_id;

	string GetName() const override {
		return "DATA_SINK";
	}

	InsertionOrderPreservingMap<string> ParamsToString() const override {
		InsertionOrderPreservingMap<string> result;
		result["Operation Id"] = operation_id;
		return result;
	}

	OperatorResultType Execute(ExecutionContext &, DataChunk &input, DataChunk &chunk, GlobalOperatorState &,
	                           OperatorState &) const override {
		chunk.Reference(input);
		return OperatorResultType::NEED_MORE_INPUT;
	}

	bool ParallelOperator() const override {
		return true;
	}

protected:
	void SerializeOperatorData(Serializer &serializer) const override;
};

} // namespace duckdb
