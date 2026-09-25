#include "duckdb/function/table/range.hpp"
#include "duckdb/common/algorithm.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/function/table/distributed_sequence.hpp"

namespace duckdb {

struct RepeatFunctionData : public TableFunctionData {
	RepeatFunctionData() = default;

	RepeatFunctionData(Value value, idx_t target_count) : value(std::move(value)), target_count(target_count) {
	}

	Value value;
	idx_t target_count = 0;
	bool distributed = false;
	vector<DistributedSequenceSplit> sequence_splits;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<RepeatFunctionData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto other = dynamic_cast<const RepeatFunctionData *>(&other_p);
		return other && column_ids == other->column_ids && value.type() == other->value.type() &&
		       Value::NotDistinctFrom(value, other->value) && target_count == other->target_count &&
		       distributed == other->distributed &&
		       DistributedSequenceSplitsEqual(sequence_splits, other->sequence_splits);
	}
};

struct RepeatOperatorData : public GlobalTableFunctionState {
	idx_t current_count = 0;
	idx_t sequence_split_index = 0;
	idx_t sequence_split_row = 0;
};

static unique_ptr<FunctionData> RepeatBind(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names) {
	// the repeat function returns the type of the first argument
	auto &inputs = input.inputs;
	return_types.push_back(inputs[0].type());
	names.push_back(inputs[0].ToString());
	if (inputs[1].IsNull()) {
		throw BinderException("Repeat second parameter cannot be NULL");
	}
	auto repeat_count = inputs[1].GetValue<int64_t>();
	if (repeat_count < 0) {
		throw BinderException("Repeat second parameter cannot be be less than 0");
	}
	return make_uniq<RepeatFunctionData>(inputs[0], NumericCast<idx_t>(repeat_count));
}

static unique_ptr<GlobalTableFunctionState> RepeatInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<RepeatOperatorData>();
}

static void RepeatFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<RepeatFunctionData>();
	auto &state = data_p.global_state->Cast<RepeatOperatorData>();

	idx_t remaining;
	if (bind_data.distributed) {
		while (state.sequence_split_index < bind_data.sequence_splits.size() &&
		       state.sequence_split_row == bind_data.sequence_splits[state.sequence_split_index].row_count) {
			state.sequence_split_index++;
			state.sequence_split_row = 0;
		}
		if (state.sequence_split_index == bind_data.sequence_splits.size()) {
			output.SetCardinality(0);
			return;
		}
		auto &sequence_split = bind_data.sequence_splits[state.sequence_split_index];
		remaining = MinValue<idx_t>(sequence_split.row_count - state.sequence_split_row, STANDARD_VECTOR_SIZE);
		state.sequence_split_row += remaining;
	} else {
		remaining = MinValue<idx_t>(bind_data.target_count - state.current_count, STANDARD_VECTOR_SIZE);
		state.current_count += remaining;
	}
	output.data[0].Reference(bind_data.value);
	output.SetCardinality(remaining);
}

static unique_ptr<NodeStatistics> RepeatCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<RepeatFunctionData>();
	return make_uniq<NodeStatistics>(bind_data.target_count, bind_data.target_count);
}

static void RepeatSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data_p,
                            const TableFunction &) {
	if (!bind_data_p) {
		throw SerializationException("repeat requires bind data");
	}
	auto &bind_data = bind_data_p->Cast<RepeatFunctionData>();
	serializer.WriteProperty(101, "value", bind_data.value);
	serializer.WriteProperty(102, "target_count", NumericCast<int64_t>(bind_data.target_count));
	serializer.WriteProperty(103, "distributed", bind_data.distributed);
}

static unique_ptr<FunctionData> RepeatDeserialize(Deserializer &deserializer, TableFunction &) {
	auto result = make_uniq<RepeatFunctionData>();
	result->value = deserializer.ReadProperty<Value>(101, "value");
	auto target_count = deserializer.ReadProperty<int64_t>(102, "target_count");
	if (target_count < 0) {
		throw SerializationException("serialized repeat has a negative target count");
	}
	result->target_count = NumericCast<idx_t>(target_count);
	result->distributed = deserializer.ReadProperty<bool>(103, "distributed");
	return std::move(result);
}

static vector<DistributedScanSplit> PlanDistributedRepeat(const TableFunctionDistributedScanPlanningInput &input) {
	if (!input.bind_data) {
		throw InvalidInputException("distributed repeat requires bind data");
	}
	auto &bind_data = input.bind_data->Cast<RepeatFunctionData>();
	auto result = PlanDistributedSequenceSplits(bind_data.target_count, true, input.target_split_count);
	// The repeated value can be an arbitrarily large nested value. Preserve the
	// exact row estimate, but do not reuse the sequence helper's BIGINT byte
	// estimate as if every output row were eight bytes.
	for (auto &split : result) {
		split.estimated_bytes = optional_idx();
	}
	return result;
}

static unique_ptr<FunctionData> CreateDistributedRepeatWorkerBind(const TableFunctionDistributedScanInput &input) {
	if (!input.bind_data) {
		throw InvalidInputException("distributed repeat requires bind data");
	}
	auto result = make_uniq<RepeatFunctionData>(input.bind_data->Cast<RepeatFunctionData>());
	result->distributed = true;
	result->sequence_splits.clear();
	return std::move(result);
}

static void ApplyDistributedRepeatSplits(optional_ptr<FunctionData> worker_bind_data,
                                         const vector<DistributedScanSplit> &splits) {
	if (!worker_bind_data) {
		throw InvalidInputException("distributed repeat requires worker bind data");
	}
	auto &bind_data = worker_bind_data->Cast<RepeatFunctionData>();
	if (!bind_data.distributed) {
		throw InvalidInputException("distributed repeat splits require a worker bind");
	}
	bind_data.sequence_splits = DecodeDistributedSequenceSplits(splits, bind_data.target_count, true);
}

static TableFunctionDistributedScanCallbacks RepeatDistributedCallbacks() {
	TableFunctionDistributedScanCallbacks callbacks;
	callbacks.protocol_version = DISTRIBUTED_SEQUENCE_PROTOCOL_VERSION;
	callbacks.split_codec = {DISTRIBUTED_SEQUENCE_SPLIT_CODEC, DISTRIBUTED_SEQUENCE_SPLIT_CODEC_VERSION};
	callbacks.plan_splits = PlanDistributedRepeat;
	callbacks.create_worker_bind = CreateDistributedRepeatWorkerBind;
	callbacks.apply_splits = ApplyDistributedRepeatSplits;
	return callbacks;
}

TableFunction RepeatTableFunction::GetFunction() {
	TableFunction repeat("repeat", {LogicalType::ANY, LogicalType::BIGINT}, RepeatFunction, RepeatBind, RepeatInit);
	repeat.cardinality = RepeatCardinality;
	repeat.serialize = RepeatSerialize;
	repeat.deserialize = RepeatDeserialize;
	repeat.SetDistributedScanCallbacks(RepeatDistributedCallbacks());
	repeat.BindDistributedScanCapability("vane_core");
	return repeat;
}

void RepeatTableFunction::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(GetFunction());
}

} // namespace duckdb
