// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/function/table/range.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/operator/add.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/set.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/function/table/distributed_sequence.hpp"
#include "duckdb/function/table/summary.hpp"
#include "duckdb/function/table_function.hpp"

#include <algorithm>

namespace duckdb {

//===--------------------------------------------------------------------===//
// Distributed sequence task protocol
//===--------------------------------------------------------------------===//
static string EncodeDistributedSequenceShard(const DistributedSequenceShard &shard) {
	MemoryStream stream(Allocator::DefaultAllocator());
	BinarySerializer serializer(stream);
	serializer.Begin();
	serializer.WriteProperty(1, "ordinal", shard.ordinal);
	serializer.WriteProperty(2, "row_offset", shard.row_offset);
	serializer.WriteProperty(3, "row_count", shard.row_count);
	serializer.WriteProperty(4, "exact_count", shard.exact_count);
	serializer.End();
	return string(reinterpret_cast<const char *>(stream.GetData()), stream.GetPosition());
}

static DistributedSequenceShard DecodeDistributedSequenceShard(const string &payload) {
	if (payload.empty()) {
		throw InvalidInputException("distributed sequence shard payload is empty");
	}
	auto data = reinterpret_cast<data_ptr_t>(const_cast<char *>(payload.data()));
	MemoryStream stream(data, payload.size());
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	DistributedSequenceShard result;
	result.ordinal = deserializer.ReadProperty<idx_t>(1, "ordinal");
	result.row_offset = deserializer.ReadProperty<idx_t>(2, "row_offset");
	result.row_count = deserializer.ReadProperty<idx_t>(3, "row_count");
	result.exact_count = deserializer.ReadProperty<bool>(4, "exact_count");
	deserializer.End();
	if (stream.GetPosition() != payload.size()) {
		throw InvalidInputException("distributed sequence shard payload has trailing bytes");
	}
	return result;
}

static idx_t ParseDistributedSequenceTaskId(const string &task_id) {
	if (task_id.empty() || (task_id.size() > 1 && task_id[0] == '0')) {
		throw InvalidInputException("distributed sequence task has non-canonical id '%s'", task_id);
	}
	idx_t result = 0;
	for (auto character : task_id) {
		if (character < '0' || character > '9') {
			throw InvalidInputException("distributed sequence task has non-canonical id '%s'", task_id);
		}
		const auto digit = NumericCast<idx_t>(character - '0');
		if (result > (NumericLimits<idx_t>::Maximum() - digit) / 10) {
			throw InvalidInputException("distributed sequence task id '%s' overflows idx_t", task_id);
		}
		result = result * 10 + digit;
	}
	if (result == NumericLimits<idx_t>::Maximum()) {
		throw InvalidInputException("distributed sequence task id '%s' is reserved", task_id);
	}
	return result;
}

static idx_t SaturatingSequenceByteEstimate(idx_t row_count) {
	constexpr idx_t MAX_VALID_OPTIONAL_INDEX = NumericLimits<idx_t>::Maximum() - 1;
	if (row_count > MAX_VALID_OPTIONAL_INDEX / sizeof(int64_t)) {
		return MAX_VALID_OPTIONAL_INDEX;
	}
	return row_count * sizeof(int64_t);
}

vector<DistributedScanTask> PlanDistributedSequenceTasks(idx_t cardinality, bool exact_count, idx_t target_task_count) {
	vector<DistributedScanTask> result;
	if (!exact_count) {
		DistributedSequenceShard shard;
		shard.exact_count = false;
		DistributedScanTask task;
		task.task_id = "0";
		task.payload = EncodeDistributedSequenceShard(shard);
		task.Validate();
		result.push_back(std::move(task));
		return result;
	}
	if (cardinality == 0) {
		return result;
	}
	if (cardinality == NumericLimits<idx_t>::Maximum()) {
		throw OutOfRangeException("distributed sequence cardinality exceeds the supported index range");
	}

	const auto partition_count = MinValue<idx_t>(cardinality, MaxValue<idx_t>(target_task_count, 1));
	const auto rows_per_partition = cardinality / partition_count;
	const auto remainder = cardinality % partition_count;
	result.reserve(partition_count);
	idx_t row_offset = 0;
	for (idx_t ordinal = 0; ordinal < partition_count; ordinal++) {
		DistributedSequenceShard shard;
		shard.ordinal = ordinal;
		shard.row_offset = row_offset;
		shard.row_count = rows_per_partition + (ordinal < remainder ? 1 : 0);
		shard.exact_count = true;

		DistributedScanTask task;
		task.task_id = std::to_string(ordinal);
		task.payload = EncodeDistributedSequenceShard(shard);
		task.estimated_cardinality = optional_idx(shard.row_count);
		task.estimated_bytes = optional_idx(SaturatingSequenceByteEstimate(shard.row_count));
		task.Validate();
		result.push_back(std::move(task));
		row_offset += shard.row_count;
	}
	D_ASSERT(row_offset == cardinality);
	return result;
}

vector<DistributedSequenceShard> DecodeDistributedSequenceTasks(const vector<DistributedScanTask> &tasks,
                                                                idx_t cardinality, bool exact_count) {
	vector<DistributedSequenceShard> result;
	result.reserve(tasks.size());
	set<idx_t> ordinals;
	for (const auto &task : tasks) {
		task.Validate();
		auto shard = DecodeDistributedSequenceShard(task.payload);
		if (ParseDistributedSequenceTaskId(task.task_id) != shard.ordinal) {
			throw InvalidInputException("distributed sequence task id '%s' does not match payload ordinal %llu",
			                            task.task_id, static_cast<unsigned long long>(shard.ordinal));
		}
		if (!ordinals.insert(shard.ordinal).second) {
			throw InvalidInputException("distributed sequence task ordinal %llu is assigned more than once",
			                            static_cast<unsigned long long>(shard.ordinal));
		}
		if (shard.exact_count != exact_count) {
			throw InvalidInputException("distributed sequence task '%s' count mode does not match its worker bind",
			                            task.task_id);
		}
		if (exact_count) {
			if (shard.row_count == 0 || shard.row_offset >= cardinality ||
			    shard.row_count > cardinality - shard.row_offset) {
				throw InvalidInputException("distributed sequence task '%s' is outside cardinality %llu", task.task_id,
				                            static_cast<unsigned long long>(cardinality));
			}
		} else if (shard.ordinal != 0 || shard.row_offset != 0 || shard.row_count != 0 || tasks.size() != 1) {
			throw InvalidInputException("calendar-sensitive distributed sequence requires exactly task '0'");
		}
		result.push_back(std::move(shard));
	}
	std::sort(result.begin(), result.end(),
	          [](const DistributedSequenceShard &left, const DistributedSequenceShard &right) {
		          if (left.row_offset != right.row_offset) {
			          return left.row_offset < right.row_offset;
		          }
		          return left.ordinal < right.ordinal;
	          });
	if (exact_count) {
		for (idx_t shard_index = 1; shard_index < result.size(); shard_index++) {
			const auto &previous = result[shard_index - 1];
			const auto &current = result[shard_index];
			if (current.row_offset < previous.row_offset + previous.row_count) {
				throw InvalidInputException("distributed sequence task assignments overlap");
			}
		}
	}
	return result;
}

static bool DistributedSequenceShardsEqual(const vector<DistributedSequenceShard> &left,
                                           const vector<DistributedSequenceShard> &right) {
	if (left.size() != right.size()) {
		return false;
	}
	for (idx_t shard_index = 0; shard_index < left.size(); shard_index++) {
		if (left[shard_index].ordinal != right[shard_index].ordinal ||
		    left[shard_index].row_offset != right[shard_index].row_offset ||
		    left[shard_index].row_count != right[shard_index].row_count ||
		    left[shard_index].exact_count != right[shard_index].exact_count) {
			return false;
		}
	}
	return true;
}

static idx_t CastSequenceCardinality(hugeint_t cardinality) {
	idx_t result;
	if (cardinality < 0 || !Hugeint::TryCast<idx_t>(cardinality, result)) {
		throw OutOfRangeException("range cardinality exceeds the supported index range");
	}
	return result;
}

static idx_t ComputeSequenceCardinalityHuge(hugeint_t start, hugeint_t end, hugeint_t increment, bool inclusive) {
	if (increment == 0) {
		throw BinderException("interval cannot be 0!");
	}
	if (increment > 0) {
		if (inclusive ? start > end : start >= end) {
			return 0;
		}
		const auto distance = end - start;
		if (inclusive) {
			return CastSequenceCardinality(distance / increment + 1);
		}
		return CastSequenceCardinality((distance + increment - 1) / increment);
	}
	if (inclusive ? start < end : start <= end) {
		return 0;
	}
	const auto distance = start - end;
	const auto magnitude = -increment;
	if (inclusive) {
		return CastSequenceCardinality(distance / magnitude + 1);
	}
	return CastSequenceCardinality((distance + magnitude - 1) / magnitude);
}

idx_t ComputeDistributedSequenceCardinality(int64_t start, int64_t end, int64_t increment, bool inclusive) {
	return ComputeSequenceCardinalityHuge(hugeint_t(start), hugeint_t(end), hugeint_t(increment), inclusive);
}

int64_t GetDistributedSequenceValue(int64_t start, int64_t increment, idx_t row_offset) {
	const auto value = hugeint_t(start) + hugeint_t(increment) * hugeint_t(NumericCast<uint64_t>(row_offset));
	int64_t result;
	if (!Hugeint::TryCast<int64_t>(value, result)) {
		throw InvalidInputException("distributed sequence shard starts outside the value domain");
	}
	return result;
}

//===--------------------------------------------------------------------===//
// Range (integers)
//===--------------------------------------------------------------------===//
static void GetParameters(const int64_t values[], idx_t value_count, int64_t &start, int64_t &end, int64_t &increment) {
	if (value_count < 2) {
		start = 0;
		end = values[0];
	} else {
		start = values[0];
		end = values[1];
	}
	increment = value_count < 3 ? 1 : values[2];
}

struct RangeFunctionBindData : public TableFunctionData {
	RangeFunctionBindData() = default;

	RangeFunctionBindData(const RangeFunctionBindData &other)
	    : TableFunctionData(other), has_parameters(other.has_parameters), is_null(other.is_null),
	      generate_series(other.generate_series), distributed(other.distributed), start(other.start), end(other.end),
	      increment(other.increment), cardinality(other.cardinality), shards(other.shards) {
	}

	RangeFunctionBindData(const vector<Value> &inputs, bool generate_series_p) : generate_series(generate_series_p) {
		if (inputs.empty()) {
			return;
		}
		has_parameters = true;
		int64_t values[3];
		for (idx_t input_index = 0; input_index < inputs.size(); input_index++) {
			if (inputs[input_index].IsNull()) {
				is_null = true;
				return;
			}
			values[input_index] = inputs[input_index].GetValue<int64_t>();
		}
		GetParameters(values, inputs.size(), start, end, increment);
		cardinality = ComputeDistributedSequenceCardinality(start, end, increment, generate_series);
	}

	bool has_parameters = false;
	bool is_null = false;
	bool generate_series = false;
	bool distributed = false;
	int64_t start = 0;
	int64_t end = 0;
	int64_t increment = 1;
	idx_t cardinality = 0;
	vector<DistributedSequenceShard> shards;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<RangeFunctionBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto other = dynamic_cast<const RangeFunctionBindData *>(&other_p);
		return other && column_ids == other->column_ids && has_parameters == other->has_parameters &&
		       is_null == other->is_null && generate_series == other->generate_series &&
		       distributed == other->distributed && start == other->start && end == other->end &&
		       increment == other->increment && cardinality == other->cardinality &&
		       DistributedSequenceShardsEqual(shards, other->shards);
	}
};

template <bool GENERATE_SERIES>
static unique_ptr<FunctionData> RangeFunctionBind(ClientContext &, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	return_types.emplace_back(LogicalType::BIGINT);
	names.emplace_back(GENERATE_SERIES ? "generate_series" : "range");
	if (input.inputs.size() > 3) {
		return nullptr;
	}
	return make_uniq<RangeFunctionBindData>(input.inputs, GENERATE_SERIES);
}

struct RangeFunctionLocalState : public LocalTableFunctionState {
	bool initialized_row = false;
	idx_t current_input_row = 0;
	hugeint_t current_idx = 0;

	hugeint_t start;
	hugeint_t end;
	hugeint_t increment;
	bool empty_range = false;

	idx_t shard_index = 0;
	idx_t shard_row = 0;
};

static unique_ptr<LocalTableFunctionState> RangeFunctionLocalInit(ExecutionContext &, TableFunctionInitInput &,
                                                                  GlobalTableFunctionState *) {
	return make_uniq<RangeFunctionLocalState>();
}

template <bool GENERATE_SERIES>
static void GenerateRangeParameters(DataChunk &input, idx_t row_id, RangeFunctionLocalState &result) {
	input.Flatten();
	result.empty_range = false;
	for (idx_t column_index = 0; column_index < input.ColumnCount(); column_index++) {
		if (FlatVector::IsNull(input.data[column_index], row_id)) {
			result.start = 0;
			result.end = 0;
			result.increment = 1;
			result.empty_range = true;
			return;
		}
	}
	int64_t values[3];
	for (idx_t column_index = 0; column_index < input.ColumnCount(); column_index++) {
		if (column_index >= 3) {
			throw InternalException("Unsupported parameter count for range function");
		}
		values[column_index] = FlatVector::GetValue<int64_t>(input.data[column_index], row_id);
	}
	int64_t start;
	int64_t end;
	int64_t increment;
	GetParameters(values, input.ColumnCount(), start, end, increment);
	result.start = start;
	result.end = end;
	result.increment = increment;
	if (result.increment == 0) {
		throw BinderException("interval cannot be 0!");
	}
	if (result.start > result.end && result.increment > 0) {
		result.empty_range = true;
	}
	if (result.start < result.end && result.increment < 0) {
		result.empty_range = true;
	}
	if (GENERATE_SERIES) {
		result.end += result.increment < 0 ? -1 : 1;
	}
}

static OperatorResultType DistributedIntegerRangeFunction(const RangeFunctionBindData &bind_data,
                                                          RangeFunctionLocalState &state, DataChunk &output) {
	while (state.shard_index < bind_data.shards.size()) {
		const auto &shard = bind_data.shards[state.shard_index];
		if (state.shard_row == shard.row_count) {
			state.shard_index++;
			state.shard_row = 0;
			continue;
		}
		const auto output_count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, shard.row_count - state.shard_row);
		const auto row_offset = shard.row_offset + state.shard_row;
		const auto current_value = GetDistributedSequenceValue(bind_data.start, bind_data.increment, row_offset);
		output.data[0].Sequence(current_value, bind_data.increment, output_count);
		state.shard_row += output_count;
		output.SetCardinality(output_count);
		return OperatorResultType::HAVE_MORE_OUTPUT;
	}
	output.SetCardinality(0);
	return OperatorResultType::NEED_MORE_INPUT;
}

template <bool GENERATE_SERIES>
static OperatorResultType RangeFunction(ExecutionContext &, TableFunctionInput &data_p, DataChunk &input,
                                        DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<RangeFunctionBindData>();
	auto &state = data_p.local_state->Cast<RangeFunctionLocalState>();
	if (bind_data.distributed) {
		return DistributedIntegerRangeFunction(bind_data, state, output);
	}
	while (true) {
		if (!state.initialized_row) {
			if (state.current_input_row >= input.size()) {
				state.current_input_row = 0;
				state.initialized_row = false;
				return OperatorResultType::NEED_MORE_INPUT;
			}
			GenerateRangeParameters<GENERATE_SERIES>(input, state.current_input_row, state);
			state.initialized_row = true;
			state.current_idx = 0;
		}
		if (state.empty_range) {
			output.SetCardinality(0);
			state.current_input_row++;
			state.initialized_row = false;
			return OperatorResultType::HAVE_MORE_OUTPUT;
		}
		const auto current_value = state.start + state.increment * state.current_idx;
		int64_t current_value_i64;
		if (!Hugeint::TryCast<int64_t>(current_value, current_value_i64)) {
			state.current_input_row++;
			state.initialized_row = false;
			continue;
		}
		const hugeint_t bound_offset = state.increment < 0 ? 1 : -1;
		const auto remaining_huge = (state.end - current_value + (state.increment + bound_offset)) / state.increment;
		idx_t remaining = STANDARD_VECTOR_SIZE;
		if (remaining_huge < hugeint_t(STANDARD_VECTOR_SIZE)) {
			remaining = CastSequenceCardinality(remaining_huge);
		}
		output.data[0].Sequence(current_value_i64, Hugeint::Cast<int64_t>(state.increment), remaining);
		state.current_idx += hugeint_t(NumericCast<uint64_t>(remaining));
		output.SetCardinality(remaining);
		if (remaining == 0) {
			state.current_input_row++;
			state.initialized_row = false;
			continue;
		}
		return OperatorResultType::HAVE_MORE_OUTPUT;
	}
}

static unique_ptr<NodeStatistics> RangeCardinality(ClientContext &, const FunctionData *bind_data_p) {
	if (!bind_data_p) {
		return nullptr;
	}
	auto &bind_data = bind_data_p->Cast<RangeFunctionBindData>();
	if (!bind_data.has_parameters) {
		return nullptr;
	}
	return make_uniq<NodeStatistics>(bind_data.cardinality, bind_data.cardinality);
}

static void RangeFunctionSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data_p,
                                   const TableFunction &) {
	if (!bind_data_p) {
		throw SerializationException("integer sequence function requires bind data");
	}
	auto &bind_data = bind_data_p->Cast<RangeFunctionBindData>();
	serializer.WriteProperty(101, "has_parameters", bind_data.has_parameters);
	serializer.WriteProperty(102, "is_null", bind_data.is_null);
	serializer.WriteProperty(103, "generate_series", bind_data.generate_series);
	serializer.WriteProperty(104, "distributed", bind_data.distributed);
	serializer.WriteProperty(105, "start", bind_data.start);
	serializer.WriteProperty(106, "end", bind_data.end);
	serializer.WriteProperty(107, "increment", bind_data.increment);
}

static bool IsGenerateSeriesFunction(const TableFunction &function) {
	if (function.name == "generate_series") {
		return true;
	}
	if (function.name == "range") {
		return false;
	}
	throw SerializationException("unexpected sequence table function '%s'", function.name);
}

static unique_ptr<FunctionData> RangeFunctionDeserialize(Deserializer &deserializer, TableFunction &function) {
	auto result = make_uniq<RangeFunctionBindData>();
	result->has_parameters = deserializer.ReadProperty<bool>(101, "has_parameters");
	result->is_null = deserializer.ReadProperty<bool>(102, "is_null");
	result->generate_series = deserializer.ReadProperty<bool>(103, "generate_series");
	result->distributed = deserializer.ReadProperty<bool>(104, "distributed");
	result->start = deserializer.ReadProperty<int64_t>(105, "start");
	result->end = deserializer.ReadProperty<int64_t>(106, "end");
	result->increment = deserializer.ReadProperty<int64_t>(107, "increment");
	if (result->generate_series != IsGenerateSeriesFunction(function)) {
		throw SerializationException("serialized integer sequence bind does not match function '%s'", function.name);
	}
	if (result->is_null && !result->has_parameters) {
		throw SerializationException("serialized integer sequence has null state without bound parameters");
	}
	if (result->distributed && !result->has_parameters) {
		throw SerializationException("serialized distributed integer sequence has no bound parameters");
	}
	if (result->has_parameters && !result->is_null) {
		result->cardinality = ComputeDistributedSequenceCardinality(result->start, result->end, result->increment,
		                                                            result->generate_series);
	}
	return std::move(result);
}

static vector<DistributedScanTask> PlanDistributedIntegerRange(const TableFunctionDistributedScanInput &input) {
	auto &bind_data = input.bind_data.Cast<RangeFunctionBindData>();
	if (!bind_data.has_parameters) {
		throw InvalidInputException("distributed integer range requires scalar bound parameters");
	}
	return PlanDistributedSequenceTasks(bind_data.cardinality, true, input.target_task_count);
}

static unique_ptr<FunctionData>
CreateDistributedIntegerRangeWorkerBind(const TableFunctionDistributedScanInput &input) {
	auto &source_bind = input.bind_data.Cast<RangeFunctionBindData>();
	if (!source_bind.has_parameters) {
		throw InvalidInputException("distributed integer range requires scalar bound parameters");
	}
	auto result = make_uniq<RangeFunctionBindData>(source_bind);
	result->distributed = true;
	result->shards.clear();
	return std::move(result);
}

static void ApplyDistributedIntegerRangeTasks(FunctionData &worker_bind_data,
                                              const vector<DistributedScanTask> &tasks) {
	auto &bind_data = worker_bind_data.Cast<RangeFunctionBindData>();
	if (!bind_data.distributed || !bind_data.has_parameters) {
		throw InvalidInputException("distributed integer range tasks require a worker bind");
	}
	bind_data.shards = DecodeDistributedSequenceTasks(tasks, bind_data.cardinality, true);
}

//===--------------------------------------------------------------------===//
// Range (timestamp)
//===--------------------------------------------------------------------===//
static hugeint_t TimestampIntervalMicros(const interval_t &increment) {
	return hugeint_t(static_cast<int64_t>(increment.days)) * hugeint_t(Interval::MICROS_PER_DAY) +
	       hugeint_t(increment.micros);
}

struct RangeDateTimeBindData : public TableFunctionData {
	RangeDateTimeBindData() = default;

	RangeDateTimeBindData(const RangeDateTimeBindData &other)
	    : TableFunctionData(other), has_parameters(other.has_parameters), is_null(other.is_null),
	      generate_series(other.generate_series), distributed(other.distributed), exact_count(other.exact_count),
	      increasing(other.increasing), empty_range(other.empty_range), start(other.start), end(other.end),
	      increment(other.increment), cardinality(other.cardinality), shards(other.shards) {
	}

	RangeDateTimeBindData(const vector<Value> &inputs, bool generate_series_p) : generate_series(generate_series_p) {
		if (inputs.empty()) {
			return;
		}
		has_parameters = true;
		for (const auto &value : inputs) {
			if (value.IsNull()) {
				is_null = true;
				exact_count = true;
				empty_range = true;
				return;
			}
		}
		start = inputs[0].GetValue<timestamp_t>();
		end = inputs[1].GetValue<timestamp_t>();
		increment = inputs[2].GetValue<interval_t>();
		Initialize();
	}

	void Initialize() {
		if (!Timestamp::IsFinite(start) || !Timestamp::IsFinite(end)) {
			throw BinderException("RANGE with infinite bounds is not supported");
		}
		const bool has_positive = increment.months > 0 || increment.days > 0 || increment.micros > 0;
		const bool has_negative = increment.months < 0 || increment.days < 0 || increment.micros < 0;
		if (!has_positive && !has_negative) {
			throw BinderException("interval cannot be 0!");
		}
		if (has_positive && has_negative) {
			throw BinderException("RANGE with composite interval that has mixed signs is not supported");
		}
		increasing = has_positive;
		empty_range = increasing ? start > end : start < end;
		if (empty_range) {
			exact_count = true;
			cardinality = 0;
			return;
		}
		if (start == end) {
			exact_count = true;
			cardinality = generate_series ? 1 : 0;
			empty_range = !generate_series;
			return;
		}
		if (increment.months == 0) {
			exact_count = true;
			cardinality = ComputeSequenceCardinalityHuge(hugeint_t(start.value), hugeint_t(end.value),
			                                             TimestampIntervalMicros(increment), generate_series);
			empty_range = cardinality == 0;
		}
	}

	bool has_parameters = false;
	bool is_null = false;
	bool generate_series = false;
	bool distributed = false;
	bool exact_count = false;
	bool increasing = true;
	bool empty_range = false;
	timestamp_t start = timestamp_t(0);
	timestamp_t end = timestamp_t(0);
	interval_t increment {0, 0, 0};
	idx_t cardinality = 0;
	vector<DistributedSequenceShard> shards;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<RangeDateTimeBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto other = dynamic_cast<const RangeDateTimeBindData *>(&other_p);
		return other && column_ids == other->column_ids && has_parameters == other->has_parameters &&
		       is_null == other->is_null && generate_series == other->generate_series &&
		       distributed == other->distributed && exact_count == other->exact_count &&
		       increasing == other->increasing && empty_range == other->empty_range && start == other->start &&
		       end == other->end && increment == other->increment && cardinality == other->cardinality &&
		       DistributedSequenceShardsEqual(shards, other->shards);
	}
};

template <bool GENERATE_SERIES>
static unique_ptr<FunctionData> RangeDateTimeBind(ClientContext &, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	return_types.push_back(LogicalType::TIMESTAMP);
	names.emplace_back(GENERATE_SERIES ? "generate_series" : "range");
	if (!input.inputs.empty() && input.inputs.size() != 3) {
		return nullptr;
	}
	return make_uniq<RangeDateTimeBindData>(input.inputs, GENERATE_SERIES);
}

static unique_ptr<NodeStatistics> RangeDateTimeCardinality(ClientContext &, const FunctionData *bind_data_p) {
	if (!bind_data_p) {
		return nullptr;
	}
	auto &bind_data = bind_data_p->Cast<RangeDateTimeBindData>();
	if (!bind_data.has_parameters || !bind_data.exact_count) {
		return nullptr;
	}
	return make_uniq<NodeStatistics>(bind_data.cardinality, bind_data.cardinality);
}

struct RangeDateTimeLocalState : public LocalTableFunctionState {
	bool initialized_row = false;
	idx_t current_input_row = 0;
	timestamp_t current_state;

	timestamp_t start;
	timestamp_t end;
	interval_t increment {0, 0, 0};
	bool inclusive_bound = false;
	bool greater_than_check = true;
	bool empty_range = false;

	idx_t shard_index = 0;
	idx_t shard_row = 0;

	bool Finished(timestamp_t current_value) const {
		if (greater_than_check) {
			return inclusive_bound ? current_value > end : current_value >= end;
		}
		return inclusive_bound ? current_value < end : current_value <= end;
	}
};

template <bool GENERATE_SERIES>
static void GenerateRangeDateTimeParameters(DataChunk &input, idx_t row_id, RangeDateTimeLocalState &result) {
	input.Flatten();
	result.empty_range = false;
	for (idx_t column_index = 0; column_index < input.ColumnCount(); column_index++) {
		if (FlatVector::IsNull(input.data[column_index], row_id)) {
			result.start = timestamp_t(0);
			result.end = timestamp_t(0);
			result.increment = interval_t {0, 0, 0};
			result.greater_than_check = true;
			result.inclusive_bound = false;
			result.empty_range = true;
			return;
		}
	}

	result.start = FlatVector::GetValue<timestamp_t>(input.data[0], row_id);
	result.end = FlatVector::GetValue<timestamp_t>(input.data[1], row_id);
	result.increment = FlatVector::GetValue<interval_t>(input.data[2], row_id);
	if (!Timestamp::IsFinite(result.start) || !Timestamp::IsFinite(result.end)) {
		throw BinderException("RANGE with infinite bounds is not supported");
	}
	const bool has_positive = result.increment.months > 0 || result.increment.days > 0 || result.increment.micros > 0;
	const bool has_negative = result.increment.months < 0 || result.increment.days < 0 || result.increment.micros < 0;
	if (!has_positive && !has_negative) {
		throw BinderException("interval cannot be 0!");
	}
	if (has_positive && has_negative) {
		throw BinderException("RANGE with composite interval that has mixed signs is not supported");
	}
	result.greater_than_check = has_positive;
	if ((has_positive && result.start > result.end) || (has_negative && result.start < result.end)) {
		result.empty_range = true;
	}
	result.inclusive_bound = GENERATE_SERIES;
}

static unique_ptr<LocalTableFunctionState> RangeDateTimeLocalInit(ExecutionContext &, TableFunctionInitInput &,
                                                                  GlobalTableFunctionState *) {
	return make_uniq<RangeDateTimeLocalState>();
}

static timestamp_t DistributedTimestampValueAt(const RangeDateTimeBindData &bind_data, idx_t row_offset) {
	const auto value = hugeint_t(bind_data.start.value) +
	                   TimestampIntervalMicros(bind_data.increment) * hugeint_t(NumericCast<uint64_t>(row_offset));
	int64_t result;
	if (!Hugeint::TryCast<int64_t>(value, result)) {
		throw InvalidInputException("distributed timestamp sequence value is outside the timestamp domain");
	}
	return timestamp_t(result);
}

static OperatorResultType DistributedDateTimeRangeFunction(const RangeDateTimeBindData &bind_data,
                                                           RangeDateTimeLocalState &state, DataChunk &output) {
	while (state.shard_index < bind_data.shards.size()) {
		const auto &shard = bind_data.shards[state.shard_index];
		if (!state.initialized_row) {
			state.current_state =
			    shard.exact_count ? DistributedTimestampValueAt(bind_data, shard.row_offset) : bind_data.start;
			state.end = bind_data.end;
			state.increment = bind_data.increment;
			state.greater_than_check = bind_data.increasing;
			state.inclusive_bound = bind_data.generate_series;
			state.shard_row = 0;
			state.initialized_row = true;
		}

		idx_t size = 0;
		auto output_data = FlatVector::GetData<timestamp_t>(output.data[0]);
		if (shard.exact_count) {
			while (state.shard_row < shard.row_count && size < STANDARD_VECTOR_SIZE) {
				output_data[size++] = DistributedTimestampValueAt(bind_data, shard.row_offset + state.shard_row);
				state.shard_row++;
			}
		} else {
			while (!state.Finished(state.current_state) && size < STANDARD_VECTOR_SIZE) {
				output_data[size++] = state.current_state;
				state.current_state =
				    AddOperator::Operation<timestamp_t, interval_t, timestamp_t>(state.current_state, state.increment);
			}
		}

		const bool finished =
		    shard.exact_count ? state.shard_row == shard.row_count : state.Finished(state.current_state);
		if (finished) {
			state.shard_index++;
			state.initialized_row = false;
		}
		if (size > 0) {
			output.SetCardinality(size);
			return OperatorResultType::HAVE_MORE_OUTPUT;
		}
	}
	output.SetCardinality(0);
	return OperatorResultType::NEED_MORE_INPUT;
}

template <bool GENERATE_SERIES>
static OperatorResultType RangeDateTimeFunction(ExecutionContext &, TableFunctionInput &data_p, DataChunk &input,
                                                DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<RangeDateTimeBindData>();
	auto &state = data_p.local_state->Cast<RangeDateTimeLocalState>();
	if (bind_data.distributed) {
		return DistributedDateTimeRangeFunction(bind_data, state, output);
	}
	while (true) {
		if (!state.initialized_row) {
			if (state.current_input_row >= input.size()) {
				state.current_input_row = 0;
				state.initialized_row = false;
				return OperatorResultType::NEED_MORE_INPUT;
			}
			GenerateRangeDateTimeParameters<GENERATE_SERIES>(input, state.current_input_row, state);
			state.initialized_row = true;
			state.current_state = state.start;
		}
		if (state.empty_range) {
			output.SetCardinality(0);
			state.current_input_row++;
			state.initialized_row = false;
			return OperatorResultType::HAVE_MORE_OUTPUT;
		}
		idx_t size = 0;
		auto output_data = FlatVector::GetData<timestamp_t>(output.data[0]);
		while (!state.Finished(state.current_state) && size < STANDARD_VECTOR_SIZE) {
			output_data[size++] = state.current_state;
			state.current_state =
			    AddOperator::Operation<timestamp_t, interval_t, timestamp_t>(state.current_state, state.increment);
		}
		if (size == 0) {
			state.current_input_row++;
			state.initialized_row = false;
			continue;
		}
		output.SetCardinality(size);
		return OperatorResultType::HAVE_MORE_OUTPUT;
	}
}

static void RangeDateTimeSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data_p,
                                   const TableFunction &) {
	if (!bind_data_p) {
		throw SerializationException("timestamp sequence function requires bind data");
	}
	auto &bind_data = bind_data_p->Cast<RangeDateTimeBindData>();
	serializer.WriteProperty(101, "has_parameters", bind_data.has_parameters);
	serializer.WriteProperty(102, "is_null", bind_data.is_null);
	serializer.WriteProperty(103, "generate_series", bind_data.generate_series);
	serializer.WriteProperty(104, "distributed", bind_data.distributed);
	serializer.WriteProperty(105, "start", bind_data.start.value);
	serializer.WriteProperty(106, "end", bind_data.end.value);
	serializer.WriteProperty(107, "increment_months", bind_data.increment.months);
	serializer.WriteProperty(108, "increment_days", bind_data.increment.days);
	serializer.WriteProperty(109, "increment_micros", bind_data.increment.micros);
}

static unique_ptr<FunctionData> RangeDateTimeDeserialize(Deserializer &deserializer, TableFunction &function) {
	auto result = make_uniq<RangeDateTimeBindData>();
	result->has_parameters = deserializer.ReadProperty<bool>(101, "has_parameters");
	result->is_null = deserializer.ReadProperty<bool>(102, "is_null");
	result->generate_series = deserializer.ReadProperty<bool>(103, "generate_series");
	result->distributed = deserializer.ReadProperty<bool>(104, "distributed");
	result->start = timestamp_t(deserializer.ReadProperty<int64_t>(105, "start"));
	result->end = timestamp_t(deserializer.ReadProperty<int64_t>(106, "end"));
	result->increment.months = deserializer.ReadProperty<int32_t>(107, "increment_months");
	result->increment.days = deserializer.ReadProperty<int32_t>(108, "increment_days");
	result->increment.micros = deserializer.ReadProperty<int64_t>(109, "increment_micros");
	if (result->generate_series != IsGenerateSeriesFunction(function)) {
		throw SerializationException("serialized timestamp sequence bind does not match function '%s'", function.name);
	}
	if (result->is_null && !result->has_parameters) {
		throw SerializationException("serialized timestamp sequence has null state without bound parameters");
	}
	if (result->distributed && !result->has_parameters) {
		throw SerializationException("serialized distributed timestamp sequence has no bound parameters");
	}
	if (result->has_parameters) {
		if (result->is_null) {
			result->exact_count = true;
			result->empty_range = true;
		} else {
			result->Initialize();
		}
	}
	return std::move(result);
}

static vector<DistributedScanTask> PlanDistributedDateTimeRange(const TableFunctionDistributedScanInput &input) {
	auto &bind_data = input.bind_data.Cast<RangeDateTimeBindData>();
	if (!bind_data.has_parameters) {
		throw InvalidInputException("distributed timestamp range requires scalar bound parameters");
	}
	return PlanDistributedSequenceTasks(bind_data.cardinality, bind_data.exact_count, input.target_task_count);
}

static unique_ptr<FunctionData>
CreateDistributedDateTimeRangeWorkerBind(const TableFunctionDistributedScanInput &input) {
	auto &source_bind = input.bind_data.Cast<RangeDateTimeBindData>();
	if (!source_bind.has_parameters) {
		throw InvalidInputException("distributed timestamp range requires scalar bound parameters");
	}
	auto result = make_uniq<RangeDateTimeBindData>(source_bind);
	result->distributed = true;
	result->shards.clear();
	return std::move(result);
}

static void ApplyDistributedDateTimeRangeTasks(FunctionData &worker_bind_data,
                                               const vector<DistributedScanTask> &tasks) {
	auto &bind_data = worker_bind_data.Cast<RangeDateTimeBindData>();
	if (!bind_data.distributed || !bind_data.has_parameters) {
		throw InvalidInputException("distributed timestamp range tasks require a worker bind");
	}
	bind_data.shards = DecodeDistributedSequenceTasks(tasks, bind_data.cardinality, bind_data.exact_count);
}

static TableFunctionDistributedScanCallbacks IntegerRangeDistributedCallbacks() {
	TableFunctionDistributedScanCallbacks callbacks;
	callbacks.protocol_version = DISTRIBUTED_SEQUENCE_PROTOCOL_VERSION;
	callbacks.task_codec = {DISTRIBUTED_SEQUENCE_TASK_CODEC, DISTRIBUTED_SEQUENCE_TASK_CODEC_VERSION};
	callbacks.plan = PlanDistributedIntegerRange;
	callbacks.create_worker_bind = CreateDistributedIntegerRangeWorkerBind;
	callbacks.apply_tasks = ApplyDistributedIntegerRangeTasks;
	return callbacks;
}

static TableFunctionDistributedScanCallbacks DateTimeRangeDistributedCallbacks() {
	TableFunctionDistributedScanCallbacks callbacks;
	callbacks.protocol_version = DISTRIBUTED_SEQUENCE_PROTOCOL_VERSION;
	callbacks.task_codec = {DISTRIBUTED_SEQUENCE_TASK_CODEC, DISTRIBUTED_SEQUENCE_TASK_CODEC_VERSION};
	callbacks.plan = PlanDistributedDateTimeRange;
	callbacks.create_worker_bind = CreateDistributedDateTimeRangeWorkerBind;
	callbacks.apply_tasks = ApplyDistributedDateTimeRangeTasks;
	return callbacks;
}

template <bool GENERATE_SERIES>
static TableFunction CreateIntegerRangeFunction(vector<LogicalType> arguments) {
	const string name = GENERATE_SERIES ? "generate_series" : "range";
	TableFunction result(name, arguments, nullptr, RangeFunctionBind<GENERATE_SERIES>, nullptr, RangeFunctionLocalInit);
	result.in_out_function = RangeFunction<GENERATE_SERIES>;
	result.cardinality = RangeCardinality;
	result.serialize = RangeFunctionSerialize;
	result.deserialize = RangeFunctionDeserialize;
	result.SetDistributedScanCallbacks(IntegerRangeDistributedCallbacks());
	result.BindDistributedScanCapability("vane_core");
	return result;
}

template <bool GENERATE_SERIES>
static TableFunction CreateDateTimeRangeFunction() {
	const string name = GENERATE_SERIES ? "generate_series" : "range";
	TableFunction result(name, {LogicalType::TIMESTAMP, LogicalType::TIMESTAMP, LogicalType::INTERVAL}, nullptr,
	                     RangeDateTimeBind<GENERATE_SERIES>, nullptr, RangeDateTimeLocalInit);
	result.in_out_function = RangeDateTimeFunction<GENERATE_SERIES>;
	result.cardinality = RangeDateTimeCardinality;
	result.serialize = RangeDateTimeSerialize;
	result.deserialize = RangeDateTimeDeserialize;
	result.SetDistributedScanCallbacks(DateTimeRangeDistributedCallbacks());
	result.BindDistributedScanCapability("vane_core");
	return result;
}

vector<TableFunction> RangeTableFunction::GetFunctions() {
	vector<TableFunction> result;
	result.push_back(CreateIntegerRangeFunction<false>({LogicalType::BIGINT}));
	result.push_back(CreateIntegerRangeFunction<false>({LogicalType::BIGINT, LogicalType::BIGINT}));
	result.push_back(
	    CreateIntegerRangeFunction<false>({LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT}));
	result.push_back(CreateDateTimeRangeFunction<false>());
	result.push_back(CreateIntegerRangeFunction<true>({LogicalType::BIGINT}));
	result.push_back(CreateIntegerRangeFunction<true>({LogicalType::BIGINT, LogicalType::BIGINT}));
	result.push_back(CreateIntegerRangeFunction<true>({LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT}));
	result.push_back(CreateDateTimeRangeFunction<true>());
	return result;
}

void RangeTableFunction::RegisterFunction(BuiltinFunctions &set) {
	TableFunctionSet range("range");
	TableFunctionSet generate_series("generate_series");
	for (auto &function : GetFunctions()) {
		if (function.name == "range") {
			range.AddFunction(std::move(function));
		} else {
			D_ASSERT(function.name == "generate_series");
			generate_series.AddFunction(std::move(function));
		}
	}
	set.AddFunction(range);
	set.AddFunction(generate_series);
}

void BuiltinFunctions::RegisterTableFunctions() {
	CheckpointFunction::RegisterFunction(*this);
	GlobTableFunction::RegisterFunction(*this);
	RangeTableFunction::RegisterFunction(*this);
	RepeatTableFunction::RegisterFunction(*this);
	SummaryTableFunction::RegisterFunction(*this);
	UnnestTableFunction::RegisterFunction(*this);
	RepeatRowTableFunction::RegisterFunction(*this);
	CSVSnifferFunction::RegisterFunction(*this);
	ReadBlobFunction::RegisterFunction(*this);
	ReadTextFunction::RegisterFunction(*this);
	QueryTableFunction::RegisterFunction(*this);
}

} // namespace duckdb
