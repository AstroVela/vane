// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/operator/helper/physical_data_sink.hpp"

#include "duckdb/common/mutex.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/execution/distributed/data_sink.hpp"

namespace duckdb {

namespace {

class DataSinkGlobalOperatorState : public GlobalOperatorState {
public:
	explicit DataSinkGlobalOperatorState(const string &operation_id) : validation(operation_id) {
	}

	mutex lock;
	distributed::DataSinkResultValidationState validation;
};

} // namespace

PhysicalDataSink::PhysicalDataSink(PhysicalPlan &physical_plan, vector<LogicalType> types, string operation_id_p,
                                   idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, TYPE, std::move(types), estimated_cardinality),
      operation_id(std::move(operation_id_p)) {
	distributed::DataSinkResultValidationState validation(operation_id);
	auto operation_res = validation.ValidateOperationId();
	if (operation_res.is_err()) {
		throw InvalidInputException(operation_res.error().what());
	}
	auto schema_res = validation.ValidateSchema(this->types);
	if (schema_res.is_err()) {
		throw InvalidInputException(schema_res.error().what());
	}
}

unique_ptr<GlobalOperatorState> PhysicalDataSink::GetGlobalOperatorState(ClientContext &) const {
	return make_uniq<DataSinkGlobalOperatorState>(operation_id);
}

OperatorResultType PhysicalDataSink::Execute(ExecutionContext &, DataChunk &input, DataChunk &chunk,
                                             GlobalOperatorState &global_state, OperatorState &) const {
	auto &state = global_state.Cast<DataSinkGlobalOperatorState>();
	{
		lock_guard<mutex> guard(state.lock);
		auto validation_res = state.validation.Append(input);
		if (validation_res.is_err()) {
			throw InvalidInputException(validation_res.error().what());
		}
	}
	chunk.Reference(input);
	return OperatorResultType::NEED_MORE_INPUT;
}

void PhysicalDataSink::SerializeOperatorData(Serializer &serializer) const {
	serializer.WriteProperty<string>(103, "operation_id", operation_id);
}

} // namespace duckdb
