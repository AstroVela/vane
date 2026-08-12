// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/function/distributed_write.hpp"

namespace duckdb {

//! Generic worker-side streaming sink for an explicitly registered extension
//! write callback contract. Native DuckDB plans never contain this operator;
//! the distributed translator inserts it below the coordinator-only extension
//! root.
class PhysicalDistributedExtensionWrite : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::DISTRIBUTED_EXTENSION_WRITE;

	PhysicalDistributedExtensionWrite(PhysicalPlan &physical_plan, DistributedExtensionWriteInfo info,
	                                  idx_t estimated_cardinality);

	DistributedExtensionWriteInfo info;
	DistributedWriteTaskContext task_context;

	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;

	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;

	bool IsSink() const override {
		return true;
	}
	bool IsSource() const override {
		return true;
	}
	bool ParallelSink() const override {
		return false;
	}
	bool SinkOrderDependent() const override {
		return true;
	}

	void ApplyRuntimeTaskContext(DistributedWriteTaskContext context);
	InsertionOrderPreservingMap<string> ParamsToString() const override;

protected:
	void SerializeOperatorData(Serializer &serializer) const override;
};

//! Validate the selected operation and FTE task-attempt identities before any
//! worker-plan mutation. Returns the number of target write operators.
DUCKDB_API idx_t ValidateDistributedWriteTaskContextAssignment(const PhysicalPlan &plan,
                                                               const DistributedWriteTaskContext &task_context);

//! Install the task context after plan deserialization and task injection. The
//! function repeats the strict validation when called directly.
DUCKDB_API idx_t ApplyDistributedWriteTaskContext(PhysicalPlan &plan, const DistributedWriteTaskContext &task_context);

} // namespace duckdb
