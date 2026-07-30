// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/operator/projection/physical_grouping_set_expand.hpp"

#include "duckdb/common/atomic.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/execution/expression_executor.hpp"

namespace duckdb {

namespace {

class GroupingSetExpandGlobalState : public GlobalOperatorState {
public:
	atomic<bool> seed_owner_claimed {false};
};

class GroupingSetExpandOperatorState : public OperatorState {
public:
	GroupingSetExpandOperatorState(ExecutionContext &context, const vector<unique_ptr<Expression>> &groups)
	    : executor(context.client, groups) {
		vector<LogicalType> group_types;
		group_types.reserve(groups.size());
		for (const auto &group : groups) {
			group_types.push_back(group->return_type);
		}
		group_chunk.Initialize(Allocator::Get(context.client), group_types);
	}

	ExpressionExecutor executor;
	DataChunk group_chunk;
	idx_t next_grouping_set = 0;
	bool input_prepared = false;
	bool owns_seed_output = false;
	idx_t next_seed = 0;
};

int64_t GetGroupingValue(const vector<idx_t> &grouping_function, const GroupingSet &grouping_set) {
	uint64_t value = 0;
	for (const auto group_idx : grouping_function) {
		value <<= 1;
		if (grouping_set.find(group_idx) == grouping_set.end()) {
			value++;
		}
	}
	return UnsafeNumericCast<int64_t>(value);
}

} // namespace

PhysicalGroupingSetExpand::PhysicalGroupingSetExpand(PhysicalPlan &physical_plan, vector<LogicalType> types,
                                                     vector<unique_ptr<Expression>> groups_p,
                                                     vector<GroupingSet> grouping_sets_p,
                                                     vector<vector<idx_t>> grouping_functions_p,
                                                     vector<idx_t> filter_indexes_p, idx_t input_column_count_p,
                                                     bool emit_empty_grouping_sets_p, idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::GROUPING_SET_EXPAND, std::move(types),
                       estimated_cardinality),
      groups(std::move(groups_p)), grouping_sets(std::move(grouping_sets_p)),
      grouping_functions(std::move(grouping_functions_p)), filter_indexes(std::move(filter_indexes_p)),
      input_column_count(input_column_count_p), emit_empty_grouping_sets(emit_empty_grouping_sets_p) {
	const auto expected_column_count =
	    input_column_count + groups.size() + grouping_functions.size() + 1 + filter_indexes.size();
	if (this->types.size() != expected_column_count) {
		throw InternalException("Grouping-set expand expected %llu output columns, got %llu", expected_column_count,
		                        this->types.size());
	}
	for (const auto &grouping_set : grouping_sets) {
		for (const auto group_idx : grouping_set) {
			if (group_idx >= groups.size()) {
				throw InternalException("Grouping-set expand group index %llu is out of range", group_idx);
			}
		}
	}
	for (const auto &grouping_function : grouping_functions) {
		for (const auto group_idx : grouping_function) {
			if (group_idx >= groups.size()) {
				throw InternalException("Grouping-set expand GROUPING argument %llu is out of range", group_idx);
			}
		}
	}
	for (const auto filter_idx : filter_indexes) {
		if (filter_idx != DConstants::INVALID_INDEX && filter_idx >= input_column_count) {
			throw InternalException("Grouping-set expand filter index %llu is out of range", filter_idx);
		}
	}
}

unique_ptr<OperatorState> PhysicalGroupingSetExpand::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<GroupingSetExpandOperatorState>(context, groups);
}

unique_ptr<GlobalOperatorState> PhysicalGroupingSetExpand::GetGlobalOperatorState(ClientContext &context) const {
	return make_uniq<GroupingSetExpandGlobalState>();
}

OperatorResultType PhysicalGroupingSetExpand::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                      GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &state = state_p.Cast<GroupingSetExpandOperatorState>();
	if (grouping_sets.empty()) {
		return OperatorResultType::NEED_MORE_INPUT;
	}
	if (!state.input_prepared) {
		state.group_chunk.Reset();
		if (!groups.empty()) {
			state.executor.Execute(input, state.group_chunk);
		}
		state.input_prepared = true;
	}

	const auto grouping_set_idx = state.next_grouping_set++;
	const auto &grouping_set = grouping_sets[grouping_set_idx];

	for (idx_t column_idx = 0; column_idx < input_column_count; column_idx++) {
		chunk.data[column_idx].Reference(input.data[column_idx]);
	}

	const auto group_offset = input_column_count;
	for (idx_t group_idx = 0; group_idx < groups.size(); group_idx++) {
		auto &result = chunk.data[group_offset + group_idx];
		if (grouping_set.find(group_idx) != grouping_set.end()) {
			result.Reference(state.group_chunk.data[group_idx]);
		} else {
			result.Reference(Value(groups[group_idx]->return_type));
		}
	}

	const auto grouping_function_offset = group_offset + groups.size();
	for (idx_t function_idx = 0; function_idx < grouping_functions.size(); function_idx++) {
		chunk.data[grouping_function_offset + function_idx].Reference(
		    Value::BIGINT(GetGroupingValue(grouping_functions[function_idx], grouping_set)));
	}

	const auto grouping_set_id_offset = grouping_function_offset + grouping_functions.size();
	chunk.data[grouping_set_id_offset].Reference(Value::UBIGINT(grouping_set_idx));

	const auto filter_offset = grouping_set_id_offset + 1;
	for (idx_t filter_idx = 0; filter_idx < filter_indexes.size(); filter_idx++) {
		const auto input_idx = filter_indexes[filter_idx];
		if (input_idx == DConstants::INVALID_INDEX) {
			chunk.data[filter_offset + filter_idx].Reference(Value::BOOLEAN(true));
		} else {
			chunk.data[filter_offset + filter_idx].Reference(input.data[input_idx]);
		}
	}

	chunk.SetCardinality(input.size());
	if (state.next_grouping_set < grouping_sets.size()) {
		return OperatorResultType::HAVE_MORE_OUTPUT;
	}
	state.next_grouping_set = 0;
	state.input_prepared = false;
	return OperatorResultType::NEED_MORE_INPUT;
}

OperatorFinalizeResultType PhysicalGroupingSetExpand::FinalExecute(ExecutionContext &context, DataChunk &chunk,
                                                                   GlobalOperatorState &gstate_p,
                                                                   OperatorState &state_p) const {
	if (!emit_empty_grouping_sets) {
		return OperatorFinalizeResultType::FINISHED;
	}

	// FinalExecute runs once per local operator state. Elect one state so this
	// physical plan contributes only one seed per empty-set occurrence.
	auto &gstate = gstate_p.Cast<GroupingSetExpandGlobalState>();
	auto &state = state_p.Cast<GroupingSetExpandOperatorState>();
	if (!state.owns_seed_output) {
		bool expected = false;
		if (!gstate.seed_owner_claimed.compare_exchange_strong(expected, true)) {
			return OperatorFinalizeResultType::FINISHED;
		}
		state.owns_seed_output = true;
	}

	vector<idx_t> empty_grouping_set_ids;
	for (idx_t grouping_set_idx = state.next_seed; grouping_set_idx < grouping_sets.size(); grouping_set_idx++) {
		if (grouping_sets[grouping_set_idx].empty()) {
			empty_grouping_set_ids.push_back(grouping_set_idx);
		}
	}
	if (empty_grouping_set_ids.empty()) {
		return OperatorFinalizeResultType::FINISHED;
	}

	const auto count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, empty_grouping_set_ids.size());
	const auto group_offset = input_column_count;
	const auto grouping_function_offset = group_offset + groups.size();
	const auto grouping_set_id_offset = grouping_function_offset + grouping_functions.size();
	const auto filter_offset = grouping_set_id_offset + 1;

	for (idx_t column_idx = 0; column_idx < input_column_count; column_idx++) {
		chunk.data[column_idx].Reference(Value(types[column_idx]));
	}
	for (idx_t group_idx = 0; group_idx < groups.size(); group_idx++) {
		chunk.data[group_offset + group_idx].Reference(Value(groups[group_idx]->return_type));
	}
	for (idx_t function_idx = 0; function_idx < grouping_functions.size(); function_idx++) {
		chunk.data[grouping_function_offset + function_idx].SetVectorType(VectorType::FLAT_VECTOR);
	}
	chunk.data[grouping_set_id_offset].SetVectorType(VectorType::FLAT_VECTOR);
	for (idx_t filter_idx = 0; filter_idx < filter_indexes.size(); filter_idx++) {
		chunk.data[filter_offset + filter_idx].Reference(Value::BOOLEAN(false));
	}

	for (idx_t row_idx = 0; row_idx < count; row_idx++) {
		const auto grouping_set_idx = empty_grouping_set_ids[row_idx];
		const auto &grouping_set = grouping_sets[grouping_set_idx];
		for (idx_t function_idx = 0; function_idx < grouping_functions.size(); function_idx++) {
			chunk.SetValue(grouping_function_offset + function_idx, row_idx,
			               Value::BIGINT(GetGroupingValue(grouping_functions[function_idx], grouping_set)));
		}
		chunk.SetValue(grouping_set_id_offset, row_idx, Value::UBIGINT(grouping_set_idx));
	}
	state.next_seed = empty_grouping_set_ids[count - 1] + 1;
	chunk.SetCardinality(count);
	return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
}

InsertionOrderPreservingMap<string> PhysicalGroupingSetExpand::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Grouping Sets"] = std::to_string(grouping_sets.size());
	result["Groups"] = std::to_string(groups.size());
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

void PhysicalGroupingSetExpand::SerializeOperatorData(Serializer &serializer) const {
	serializer.WriteProperty(103, "groups", groups);
	serializer.WriteProperty(104, "grouping_sets", grouping_sets);
	serializer.WriteProperty(105, "grouping_functions", grouping_functions);
	serializer.WriteProperty(106, "filter_indexes", filter_indexes);
	serializer.WriteProperty(107, "input_column_count", input_column_count);
	serializer.WriteProperty(108, "emit_empty_grouping_sets", emit_empty_grouping_sets);
}

} // namespace duckdb
