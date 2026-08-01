// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/operator/helper/physical_distributed_reservoir_sample.hpp"

#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/types/hash.hpp"
#include "duckdb/execution/reservoir_sample.hpp"

#include <unordered_set>

namespace duckdb {

namespace {

class DistributedReservoirGlobalState : public GlobalSinkState {
public:
	DistributedReservoirGlobalState(ClientContext &context, const PhysicalDistributedReservoirSample &op)
	    : sample_count(NumericCast<idx_t>(op.options->sample_size.GetValue<int64_t>())),
	      sample(make_uniq<ReservoirSample>(Allocator::Get(context), sample_count, op.GetEffectiveSeed())) {
	}

	mutex lock;
	idx_t sample_count;
	unique_ptr<ReservoirSample> sample;
	std::unordered_set<idx_t> merged_task_indices;
	bool state_emitted = false;
};

static string SerializeReservoirState(const ReservoirSample &sample) {
	MemoryStream stream(Allocator::DefaultAllocator());
	BinarySerializer serializer(stream);
	serializer.Begin();
	sample.Serialize(serializer);
	serializer.End();
	return string(reinterpret_cast<const char *>(stream.GetData()), stream.GetPosition());
}

static unique_ptr<BlockingSample> DeserializeReservoirState(const string &bytes) {
	if (bytes.empty()) {
		throw SerializationException("Cannot deserialize an empty distributed reservoir state");
	}
	auto data = reinterpret_cast<data_ptr_t>(const_cast<char *>(bytes.data()));
	MemoryStream stream(data, bytes.size());
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto sample = BlockingSample::Deserialize(deserializer);
	deserializer.End();
	return sample;
}

} // namespace

PhysicalDistributedReservoirSample::PhysicalDistributedReservoirSample(PhysicalPlan &physical_plan,
                                                                       vector<LogicalType> types,
                                                                       unique_ptr<SampleOptions> options_p,
                                                                       DistributedReservoirSampleStage stage_p,
                                                                       idx_t task_index_p, idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::DISTRIBUTED_RESERVOIR_SAMPLE, std::move(types),
                       estimated_cardinality),
      options(std::move(options_p)), stage(stage_p), task_index(task_index_p) {
	if (!options || options->method != SampleMethod::RESERVOIR_SAMPLE || options->is_percentage) {
		throw InternalException("Distributed reservoir sample requires fixed-row reservoir options");
	}
	if (!options->seed.IsValid()) {
		throw InternalException("Distributed reservoir sample requires a resolved seed");
	}
	if (stage == DistributedReservoirSampleStage::LOCAL) {
		if (this->types.size() != 2 || this->types[0] != LogicalType::UBIGINT || this->types[1] != LogicalType::BLOB) {
			throw InternalException("Local distributed reservoir sample requires UBIGINT, BLOB output");
		}
	} else if (stage != DistributedReservoirSampleStage::FINAL) {
		throw InternalException("Invalid distributed reservoir sample stage");
	}
}

unique_ptr<GlobalSinkState> PhysicalDistributedReservoirSample::GetGlobalSinkState(ClientContext &context) const {
	if (stage == DistributedReservoirSampleStage::LOCAL && task_index == DConstants::INVALID_INDEX) {
		throw InternalException("Local distributed reservoir sample is missing its runtime task identity");
	}
	return make_uniq<DistributedReservoirGlobalState>(context, *this);
}

SinkResultType PhysicalDistributedReservoirSample::Sink(ExecutionContext &context, DataChunk &chunk,
                                                        OperatorSinkInput &input) const {
	auto &state = input.global_state.Cast<DistributedReservoirGlobalState>();
	lock_guard<mutex> guard(state.lock);
	if (stage == DistributedReservoirSampleStage::LOCAL) {
		state.sample->AddToReservoir(chunk);
		return SinkResultType::NEED_MORE_INPUT;
	}

	if (chunk.ColumnCount() != 2) {
		throw InternalException("Final distributed reservoir sample expected two state columns");
	}
	for (idx_t row_idx = 0; row_idx < chunk.size(); row_idx++) {
		auto task_value = chunk.GetValue(0, row_idx);
		auto state_value = chunk.GetValue(1, row_idx);
		if (task_value.IsNull() || state_value.IsNull()) {
			throw InvalidInputException("Distributed reservoir sample received a NULL state");
		}
		const auto task_index = task_value.GetValue<idx_t>();
		if (!state.merged_task_indices.insert(task_index).second) {
			throw InvalidInputException("Distributed reservoir sample received duplicate task state %llu", task_index);
		}

		auto other = DeserializeReservoirState(StringValue::Get(state_value));
		if (other->type != SampleType::RESERVOIR_SAMPLE) {
			throw InvalidInputException("Distributed reservoir sample received an incompatible sample state");
		}
		auto &other_reservoir = other->Cast<ReservoirSample>();
		if (other_reservoir.GetSampleCount() != state.sample_count) {
			throw InvalidInputException("Distributed reservoir sample state has sample count %llu, expected %llu",
			                            other_reservoir.GetSampleCount(), state.sample_count);
		}
		if (other_reservoir.GetActiveSampleCount() > state.sample_count ||
		    other_reservoir.GetTuplesSeen() < other_reservoir.GetActiveSampleCount()) {
			throw InvalidInputException("Distributed reservoir sample received an invalid reservoir state");
		}
		if (other_reservoir.NumSamplesCollected() > 0 && other_reservoir.Chunk().GetTypes() != types) {
			throw InvalidInputException("Distributed reservoir sample state has incompatible row types");
		}
		other_reservoir.PrepareForMerge();
		state.sample->Merge(std::move(other));
	}
	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalDistributedReservoirSample::Combine(ExecutionContext &context,
                                                                  OperatorSinkCombineInput &input) const {
	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalDistributedReservoirSample::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                              OperatorSinkFinalizeInput &input) const {
	auto &state = input.global_state.Cast<DistributedReservoirGlobalState>();
	if (stage == DistributedReservoirSampleStage::LOCAL) {
		state.sample->PrepareForMerge();
	}
	return SinkFinalizeType::READY;
}

SourceResultType PhysicalDistributedReservoirSample::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                                     OperatorSourceInput &input) const {
	auto &state = this->sink_state->Cast<DistributedReservoirGlobalState>();
	lock_guard<mutex> guard(state.lock);
	if (stage == DistributedReservoirSampleStage::LOCAL) {
		if (state.state_emitted) {
			return SourceResultType::FINISHED;
		}
		auto bytes = SerializeReservoirState(*state.sample);
		chunk.SetValue(0, 0, Value::UBIGINT(task_index));
		chunk.SetValue(1, 0, Value::BLOB(reinterpret_cast<const_data_ptr_t>(bytes.data()), bytes.size()));
		chunk.SetCardinality(1);
		state.state_emitted = true;
		return SourceResultType::HAVE_MORE_OUTPUT;
	}

	auto sample_chunk = state.sample->GetChunk();
	if (!sample_chunk) {
		return SourceResultType::FINISHED;
	}
	chunk.Move(*sample_chunk);
	return SourceResultType::HAVE_MORE_OUTPUT;
}

bool PhysicalDistributedReservoirSample::ParallelSink() const {
	return stage == DistributedReservoirSampleStage::LOCAL && !options->repeatable;
}

void PhysicalDistributedReservoirSample::ApplyRuntimeTaskIndex(idx_t runtime_task_index) {
	if (stage != DistributedReservoirSampleStage::LOCAL) {
		throw InternalException("Only a local distributed reservoir sample has a runtime task index");
	}
	if (runtime_task_index == DConstants::INVALID_INDEX) {
		throw InternalException("Invalid runtime task index for local distributed reservoir sample");
	}
	task_index = runtime_task_index;
}

int64_t PhysicalDistributedReservoirSample::GetEffectiveSeed() const {
	const auto seed = options->GetSeed();
	if (stage != DistributedReservoirSampleStage::LOCAL) {
		return seed;
	}
	if (task_index == DConstants::INVALID_INDEX) {
		throw InternalException("Local distributed reservoir sample is missing its runtime task identity");
	}
	const auto task_hash = MurmurHash64(static_cast<uint64_t>(task_index) + 0x9e3779b97f4a7c15ULL);
	const auto derived = MurmurHash64(static_cast<uint64_t>(seed) ^ task_hash);
	return static_cast<int64_t>(derived & static_cast<uint64_t>(NumericLimits<int64_t>::Maximum()));
}

InsertionOrderPreservingMap<string> PhysicalDistributedReservoirSample::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Stage"] = stage == DistributedReservoirSampleStage::LOCAL ? "LOCAL" : "FINAL";
	result["Sample Size"] = options->sample_size.ToString() + " rows";
	return result;
}

void PhysicalDistributedReservoirSample::SerializeOperatorData(Serializer &serializer) const {
	serializer.WriteProperty(103, "sample_options", options);
	serializer.WriteProperty<uint8_t>(104, "stage", static_cast<uint8_t>(stage));
	serializer.WriteProperty(105, "task_index", task_index);
}

} // namespace duckdb
