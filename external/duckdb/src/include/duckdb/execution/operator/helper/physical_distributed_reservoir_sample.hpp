// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/parser/parsed_data/sample_options.hpp"

namespace duckdb {

enum class DistributedReservoirSampleStage : uint8_t {
	LOCAL = 0,
	FINAL = 1,
};

//! Implements the two blocking stages of a distributed fixed-row reservoir
//! sample. LOCAL emits one serialized, mergeable reservoir state. FINAL merges
//! those states and emits the sampled rows.
class PhysicalDistributedReservoirSample : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::DISTRIBUTED_RESERVOIR_SAMPLE;

public:
	PhysicalDistributedReservoirSample(PhysicalPlan &physical_plan, vector<LogicalType> types,
	                                   unique_ptr<SampleOptions> options, DistributedReservoirSampleStage stage,
	                                   idx_t task_index, idx_t estimated_cardinality);

	unique_ptr<SampleOptions> options;
	DistributedReservoirSampleStage stage;
	idx_t task_index;

public:
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;

	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;

	bool IsSink() const override {
		return true;
	}
	bool IsSource() const override {
		return true;
	}
	bool ParallelSink() const override;

	InsertionOrderPreservingMap<string> ParamsToString() const override;

	//! Bind the LOCAL state to the scheduler-owned task partition before execution.
	void ApplyRuntimeTaskIndex(idx_t runtime_task_index);
	int64_t GetEffectiveSeed() const;

protected:
	void SerializeOperatorData(Serializer &serializer) const override;
};

} // namespace duckdb
