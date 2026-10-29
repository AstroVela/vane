// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/expression.hpp"

#include <mutex>

namespace duckdb {

struct MediaProjectState;

//! Serial, streaming receive/execute rendezvous. Input is copied to sealed
//! BufferManager storage before acknowledgement; each row runs only after its
//! entire working set has been admitted. No worker waits on a resource mutex.
class PhysicalMediaProject : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::MEDIA_PROJECT;
	PhysicalMediaProject(PhysicalPlan &plan, vector<LogicalType> types, vector<unique_ptr<Expression>> expressions,
	                     idx_t estimated_cardinality);

	bool IsSink() const override {
		return true;
	}
	bool IsSource() const override {
		return true;
	}
	OrderPreservationType SourceOrder() const override {
		return OrderPreservationType::INSERTION_ORDER;
	}
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                 GlobalSourceState &gstate) const override;
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;
	void BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) override;
	vector<const_reference<PhysicalOperator>> GetMaterializedInputChildren(ClientContext &) const override {
		return {};
	}
	void ResetExecutionState();
	InsertionOrderPreservingMap<string> ParamsToString() const override;
	void SerializeOperatorData(Serializer &serializer) const override;

	vector<unique_ptr<Expression>> select_list;

private:
	shared_ptr<MediaProjectState> GetState(ClientContext &context) const;
	mutable std::mutex state_lock;
	mutable shared_ptr<MediaProjectState> state;
};

//! Reject nested/conditional/non-projection managed calls while the experiment
//! is enabled, rather than silently executing an unmanaged scalar callback.
void ValidateManagedMediaPlan(ClientContext &context, LogicalOperator &root);

} // namespace duckdb
