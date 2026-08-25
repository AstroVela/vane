#include "duckdb/function/table/range.hpp"
#include "duckdb/common/algorithm.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/function/table/distributed_sequence.hpp"

namespace duckdb {

struct RepeatRowFunctionData : public TableFunctionData {
	RepeatRowFunctionData() = default;

	RepeatRowFunctionData(vector<Value> values, idx_t target_count)
	    : values(std::move(values)), target_count(target_count) {
	}

	vector<Value> values;
	idx_t target_count = 0;
	bool distributed = false;
	vector<DistributedSequenceSplit> sequence_splits;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<RepeatRowFunctionData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto other = dynamic_cast<const RepeatRowFunctionData *>(&other_p);
		if (!other || column_ids != other->column_ids || values.size() != other->values.size() ||
		    target_count != other->target_count || distributed != other->distributed ||
		    !DistributedSequenceSplitsEqual(sequence_splits, other->sequence_splits)) {
			return false;
		}
		for (idx_t value_index = 0; value_index < values.size(); value_index++) {
			if (values[value_index].type() != other->values[value_index].type() ||
			    !Value::NotDistinctFrom(values[value_index], other->values[value_index])) {
				return false;
			}
		}
		return true;
	}
};

struct RepeatRowOperatorData : public GlobalTableFunctionState {
	idx_t current_count = 0;
	idx_t sequence_split_index = 0;
	idx_t sequence_split_row = 0;
};

static unique_ptr<FunctionData> RepeatRowBind(ClientContext &context, TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types, vector<string> &names) {
	auto &inputs = input.inputs;
	for (idx_t input_idx = 0; input_idx < inputs.size(); input_idx++) {
		return_types.push_back(inputs[input_idx].type());
		names.push_back("column" + std::to_string(input_idx));
	}
	auto entry = input.named_parameters.find("num_rows");
	if (entry == input.named_parameters.end()) {
		throw BinderException("repeat_rows requires num_rows to be specified");
	}
	if (inputs.empty()) {
		throw BinderException("repeat_rows requires at least one column to be specified");
	}
	return make_uniq<RepeatRowFunctionData>(inputs, NumericCast<idx_t>(entry->second.GetValue<int64_t>()));
}

static unique_ptr<GlobalTableFunctionState> RepeatRowInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<RepeatRowOperatorData>();
}

static void RepeatRowFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<RepeatRowFunctionData>();
	auto &state = data_p.global_state->Cast<RepeatRowOperatorData>();

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
	for (idx_t val_idx = 0; val_idx < bind_data.values.size(); val_idx++) {
		output.data[val_idx].Reference(bind_data.values[val_idx]);
	}
	output.SetCardinality(remaining);
}

static unique_ptr<NodeStatistics> RepeatRowCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<RepeatRowFunctionData>();
	return make_uniq<NodeStatistics>(bind_data.target_count, bind_data.target_count);
}

static void RepeatRowSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data_p,
                               const TableFunction &) {
	if (!bind_data_p) {
		throw SerializationException("repeat_row requires bind data");
	}
	auto &bind_data = bind_data_p->Cast<RepeatRowFunctionData>();
	serializer.WriteProperty(101, "values", bind_data.values);
	serializer.WriteProperty(102, "target_count", NumericCast<int64_t>(bind_data.target_count));
	serializer.WriteProperty(103, "distributed", bind_data.distributed);
}

static unique_ptr<FunctionData> RepeatRowDeserialize(Deserializer &deserializer, TableFunction &) {
	auto result = make_uniq<RepeatRowFunctionData>();
	result->values = deserializer.ReadProperty<vector<Value>>(101, "values");
	auto target_count = deserializer.ReadProperty<int64_t>(102, "target_count");
	if (target_count < 0) {
		throw SerializationException("serialized repeat_row has a negative target count");
	}
	result->target_count = NumericCast<idx_t>(target_count);
	result->distributed = deserializer.ReadProperty<bool>(103, "distributed");
	if (result->values.empty()) {
		throw SerializationException("serialized repeat_row has no values");
	}
	return std::move(result);
}

static vector<DistributedScanSplit> PlanDistributedRepeatRow(const TableFunctionDistributedScanPlanningInput &input) {
	if (!input.bind_data) {
		throw InvalidInputException("distributed repeat_row requires bind data");
	}
	auto &bind_data = input.bind_data->Cast<RepeatRowFunctionData>();
	auto result = PlanDistributedSequenceSplits(bind_data.target_count, true, input.target_split_count);
	// Row width depends on every (possibly nested) repeated value. Keep the
	// exact cardinality while leaving byte size unknown to the scheduler.
	for (auto &split : result) {
		split.estimated_bytes = optional_idx();
	}
	return result;
}

static unique_ptr<FunctionData> CreateDistributedRepeatRowWorkerBind(const TableFunctionDistributedScanInput &input) {
	if (!input.bind_data) {
		throw InvalidInputException("distributed repeat_row requires bind data");
	}
	auto result = make_uniq<RepeatRowFunctionData>(input.bind_data->Cast<RepeatRowFunctionData>());
	result->distributed = true;
	result->sequence_splits.clear();
	return std::move(result);
}

static void ApplyDistributedRepeatRowSplits(optional_ptr<FunctionData> worker_bind_data,
                                            const vector<DistributedScanSplit> &splits) {
	if (!worker_bind_data) {
		throw InvalidInputException("distributed repeat_row requires worker bind data");
	}
	auto &bind_data = worker_bind_data->Cast<RepeatRowFunctionData>();
	if (!bind_data.distributed) {
		throw InvalidInputException("distributed repeat_row splits require a worker bind");
	}
	bind_data.sequence_splits = DecodeDistributedSequenceSplits(splits, bind_data.target_count, true);
}

static TableFunctionDistributedScanCallbacks RepeatRowDistributedCallbacks() {
	TableFunctionDistributedScanCallbacks callbacks;
	callbacks.protocol_version = DISTRIBUTED_SEQUENCE_PROTOCOL_VERSION;
	callbacks.split_codec = {DISTRIBUTED_SEQUENCE_SPLIT_CODEC, DISTRIBUTED_SEQUENCE_SPLIT_CODEC_VERSION};
	callbacks.plan_splits = PlanDistributedRepeatRow;
	callbacks.create_worker_bind = CreateDistributedRepeatRowWorkerBind;
	callbacks.apply_splits = ApplyDistributedRepeatRowSplits;
	return callbacks;
}

TableFunction RepeatRowTableFunction::GetFunction() {
	TableFunction repeat_row("repeat_row", {}, RepeatRowFunction, RepeatRowBind, RepeatRowInit);
	repeat_row.varargs = LogicalType::ANY;
	repeat_row.named_parameters["num_rows"] = LogicalType::BIGINT;
	repeat_row.cardinality = RepeatRowCardinality;
	repeat_row.serialize = RepeatRowSerialize;
	repeat_row.deserialize = RepeatRowDeserialize;
	repeat_row.SetDistributedScanCallbacks(RepeatRowDistributedCallbacks());
	repeat_row.BindDistributedScanCapability("vane_core");
	return repeat_row;
}

void RepeatRowTableFunction::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(GetFunction());
}

} // namespace duckdb
