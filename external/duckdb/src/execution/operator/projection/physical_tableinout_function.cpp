// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/execution/operator/projection/physical_tableinout_function.hpp"

#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/function/function_serialization.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"

namespace duckdb {

static void ReferenceOrCast(ExecutionContext &context, Vector &target, Vector &source, idx_t position, idx_t count) {
	if (target.GetType() == source.GetType()) {
		ConstantVector::Reference(target, source, position, count);
		return;
	}
	// Cast the selected row into the target type, then mark as constant.
	Vector tmp(source.GetType());
	ConstantVector::Reference(tmp, source, position, count);
	target.SetVectorType(VectorType::FLAT_VECTOR);
	VectorOperations::Cast(context.client, tmp, target, 1);
	target.SetVectorType(VectorType::CONSTANT_VECTOR);
}

class TableInOutLocalState : public OperatorState {
public:
	TableInOutLocalState() : row_index(0), new_row(true) {
	}

	unique_ptr<LocalTableFunctionState> local_state;
	idx_t row_index;
	bool new_row;
	DataChunk input_chunk;
	idx_t current_ordinality_idx = 1;
};

class TableInOutGlobalState : public GlobalOperatorState {
public:
	TableInOutGlobalState() {
	}

	idx_t MaxThreads(idx_t source_max_threads) override {
		// If no state assume maximum parallelism as the source.
		if (!global_state) {
			return source_max_threads;
		}
		return global_state->MaxThreads(source_max_threads);
	}

	void PipelineMaxThreadsResolved(idx_t max_threads) override {
		if (global_state) {
			global_state->PipelineMaxThreadsResolved(max_threads);
		}
	}

	unique_ptr<GlobalTableFunctionState> global_state;
};

PhysicalTableInOutFunction::PhysicalTableInOutFunction(PhysicalPlan &physical_plan, vector<LogicalType> types,
                                                       TableFunction function_p, unique_ptr<FunctionData> bind_data_p,
                                                       vector<ColumnIndex> column_ids_p, idx_t estimated_cardinality,
                                                       vector<column_t> project_input_p)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::INOUT_FUNCTION, std::move(types), estimated_cardinality),
      function(std::move(function_p)), bind_data(std::move(bind_data_p)), column_ids(std::move(column_ids_p)),
      projected_input(std::move(project_input_p)) {
}

unique_ptr<OperatorState> PhysicalTableInOutFunction::GetOperatorState(ExecutionContext &context) const {
	auto &gstate = op_state->Cast<TableInOutGlobalState>();
	auto result = make_uniq<TableInOutLocalState>();
	if (function.init_local) {
		TableFunctionInitInput input(bind_data.get(), column_ids, vector<idx_t>(), nullptr);
		result->local_state = function.init_local(context, input, gstate.global_state.get());
	}
	if (!projected_input.empty()) {
		vector<LogicalType> input_types;
		auto &child_types = children[0].get().GetTypes();
		idx_t input_length = child_types.size() - projected_input.size();
		for (idx_t k = 0; k < input_length; k++) {
			input_types.push_back(child_types[k]);
		}
		for (idx_t k = 0; k < projected_input.size(); k++) {
			D_ASSERT(projected_input[k] >= input_length);
		}
		result->input_chunk.Initialize(context.client, input_types);
	}
	return std::move(result);
}

unique_ptr<GlobalOperatorState> PhysicalTableInOutFunction::GetGlobalOperatorState(ClientContext &context) const {
	auto result = make_uniq<TableInOutGlobalState>();
	if (function.init_global) {
		TableFunctionInitInput input(bind_data.get(), column_ids, vector<idx_t>(), nullptr);
		result->global_state = function.init_global(context, input);
	}
	return std::move(result);
}

void PhysicalTableInOutFunction::SetOrdinality(DataChunk &chunk, const optional_idx &ordinality_column_idx,
                                               const idx_t &ordinality_idx, const idx_t &ordinality) {
	D_ASSERT(ordinality_column_idx.IsValid());
	if (ordinality > 0) {
		constexpr idx_t step = 1;
		chunk.data[ordinality_column_idx.GetIndex()].Sequence(static_cast<int64_t>(ordinality_idx), step, ordinality);
	}
}

OperatorResultType PhysicalTableInOutFunction::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                       GlobalOperatorState &gstate_p, OperatorState &state_p) const {
	auto &gstate = gstate_p.Cast<TableInOutGlobalState>();
	auto &state = state_p.Cast<TableInOutLocalState>();
	TableFunctionInput data(bind_data.get(), state.local_state.get(), gstate.global_state.get());
	if (projected_input.empty()) {
		// straightforward case - no need to project input
		auto result = function.in_out_function(context, data, input, chunk);
		if (this->ordinality_idx.IsValid()) {
			const idx_t ordinality = chunk.size();
			SetOrdinality(chunk, this->ordinality_idx, state.current_ordinality_idx, ordinality);
			state.current_ordinality_idx += ordinality;
		}
		return result;
	}
	// when project_input is set we execute the input function row-by-row
	if (state.new_row) {
		if (state.row_index >= input.size()) {
			// finished processing this chunk
			state.new_row = true;
			state.row_index = 0;
			return OperatorResultType::NEED_MORE_INPUT;
		}
		// we are processing a new row: fetch the data for the current row
		state.input_chunk.Reset();
		// set up the input data to the table in-out function
		for (idx_t col_idx = 0; col_idx < state.input_chunk.ColumnCount(); col_idx++) {
			ReferenceOrCast(context, state.input_chunk.data[col_idx], input.data[col_idx], state.row_index, 1);
		}
		state.input_chunk.SetCardinality(1);
		state.row_index++;
		state.new_row = false;
		state.current_ordinality_idx = 1;
	}
	// set up the output data in "chunk"
	D_ASSERT(chunk.ColumnCount() > projected_input.size());
	D_ASSERT(state.row_index > 0);
	idx_t base_idx = chunk.ColumnCount() - projected_input.size();
	for (idx_t project_idx = 0; project_idx < projected_input.size(); project_idx++) {
		auto source_idx = projected_input[project_idx];
		auto target_idx = base_idx + project_idx;
		ReferenceOrCast(context, chunk.data[target_idx], input.data[source_idx], state.row_index - 1, 1);
	}
	auto result = function.in_out_function(context, data, state.input_chunk, chunk);
	if (this->ordinality_idx.IsValid()) {
		const idx_t ordinality = chunk.size();
		SetOrdinality(chunk, this->ordinality_idx, state.current_ordinality_idx, ordinality);
		state.current_ordinality_idx += ordinality;
	}
	if (result == OperatorResultType::FINISHED) {
		return result;
	}
	if (result == OperatorResultType::BLOCKED) {
		return result;
	}
	if (result == OperatorResultType::NEED_MORE_INPUT) {
		// we finished processing this row: move to the next row
		state.new_row = true;
	}
	return OperatorResultType::HAVE_MORE_OUTPUT;
}

OperatorResultType PhysicalTableInOutFunction::ExecuteBatch(ExecutionContext &context, ExecutionBatch &input,
                                                            ExecutionBatch &output, GlobalOperatorState &gstate_p,
                                                            OperatorState &state_p) const {
	if (!function.in_out_function_batch || !projected_input.empty() || this->ordinality_idx.IsValid()) {
		return PhysicalOperator::ExecuteBatch(context, input, output, gstate_p, state_p);
	}
	auto &gstate = gstate_p.Cast<TableInOutGlobalState>();
	auto &state = state_p.Cast<TableInOutLocalState>();
	TableFunctionInput data(bind_data.get(), state.local_state.get(), gstate.global_state.get());
	return function.in_out_function_batch(context, data, input, output);
}

InsertionOrderPreservingMap<string> PhysicalTableInOutFunction::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	if (function.to_string) {
		TableFunctionToStringInput input(function, bind_data.get());
		auto to_string_result = function.to_string(input);
		for (const auto &it : to_string_result) {
			result[it.first] = it.second;
		}
	} else {
		result["Name"] = function.name;
	}
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

InsertionOrderPreservingMap<string> PhysicalTableInOutFunction::ExtraOperatorParams(GlobalOperatorState &gstate_p,
                                                                                    OperatorState &state_p) const {
	if (!function.dynamic_to_string) {
		return InsertionOrderPreservingMap<string>();
	}
	auto &gstate = gstate_p.Cast<TableInOutGlobalState>();
	auto &state = state_p.Cast<TableInOutLocalState>();
	TableFunctionDynamicToStringInput input(function, bind_data.get(),
	                                        state.local_state ? state.local_state.get() : nullptr,
	                                        gstate.global_state ? gstate.global_state.get() : nullptr);
	return function.dynamic_to_string(input);
}

void PhysicalTableInOutFunction::SerializeOperatorData(Serializer &serializer) const {
	FunctionSerializer::Serialize(serializer, function, bind_data.get());
	serializer.WriteProperty(200, "column_ids", column_ids);
	serializer.WriteProperty(201, "projected_input", projected_input);
	serializer.WritePropertyWithDefault(202, "ordinality_idx", ordinality_idx);
}

OperatorFinalizeResultType PhysicalTableInOutFunction::FinalExecute(ExecutionContext &context, DataChunk &chunk,
                                                                    GlobalOperatorState &gstate_p,
                                                                    OperatorState &state_p) const {
	auto &gstate = gstate_p.Cast<TableInOutGlobalState>();
	auto &state = state_p.Cast<TableInOutLocalState>();
	if (!projected_input.empty()) {
		throw InternalException("FinalExecute not supported for project_input");
	}
	TableFunctionInput data(bind_data.get(), state.local_state.get(), gstate.global_state.get());
	auto result = function.in_out_function_final(context, data, chunk);
	return result;
}

OperatorFinalizeResultType PhysicalTableInOutFunction::FinalExecuteBatch(ExecutionContext &context,
                                                                         ExecutionBatch &batch,
                                                                         GlobalOperatorState &gstate_p,
                                                                         OperatorState &state_p) const {
	if (!function.in_out_function_final_batch || !projected_input.empty() || this->ordinality_idx.IsValid()) {
		return PhysicalOperator::FinalExecuteBatch(context, batch, gstate_p, state_p);
	}
	auto &gstate = gstate_p.Cast<TableInOutGlobalState>();
	auto &state = state_p.Cast<TableInOutLocalState>();
	TableFunctionInput data(bind_data.get(), state.local_state.get(), gstate.global_state.get());
	return function.in_out_function_final_batch(context, data, batch);
}

} // namespace duckdb
