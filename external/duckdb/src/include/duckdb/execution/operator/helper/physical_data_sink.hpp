// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/physical_operator.hpp"

namespace duckdb {

//! Marker operator for a first-class distributed DataSink terminal. Native
//! execution treats it as a passthrough; the distributed translator replaces
//! it with a coordinator-side DataSinkFinishNode.
class PhysicalDataSink : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::DATA_SINK;

	PhysicalDataSink(PhysicalPlan &physical_plan, vector<LogicalType> types, string operation_id,
	                 idx_t estimated_cardinality)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::DATA_SINK, std::move(types), estimated_cardinality),
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

	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate, OperatorState &state) const override {
		(void)context;
		(void)gstate;
		(void)state;
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
