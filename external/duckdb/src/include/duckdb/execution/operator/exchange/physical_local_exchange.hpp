// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/exchange/physical_local_exchange.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/operator/exchange/repartition.hpp"
#include "duckdb/common/types.hpp"
#include <mutex>

namespace duckdb {

struct LocalExchangeState;

//! PhysicalLocalExchange represents a local (intra-process) exchange operator.
//! It repartitions data within a single process using in-memory queues with backpressure.
class PhysicalLocalExchange : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::LOCAL_EXCHANGE;

public:
	PhysicalLocalExchange(PhysicalPlan &physical_plan, vector<LogicalType> types,
	                      std::shared_ptr<RepartitionSpec> repartition_spec, idx_t estimated_cardinality);

	//! The repartition specification
	std::shared_ptr<RepartitionSpec> repartition_spec;
	//! Partition expressions (for hash repartition)
	vector<unique_ptr<Expression>> partition_by;

public:
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;

	bool ParallelOperator() const override {
		return true;
	}

	bool IsSink() const override {
		return true;
	}

	bool IsSource() const override {
		return true;
	}

	bool ParallelSink() const override {
		return true;
	}

	bool ParallelSource() const override {
		return true;
	}

	OrderPreservationType SourceOrder() const override {
		return OrderPreservationType::NO_ORDER;
	}

	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkResultType SinkBatch(ExecutionContext &context, ExecutionBatch &batch, OperatorSinkInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;
	void BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) override;

	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                 GlobalSourceState &gstate) const override;
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;
	SourceResultType GetDataBatch(ExecutionContext &context, ExecutionBatch &batch,
	                              OperatorSourceInput &input) const override;

	InsertionOrderPreservingMap<string> ParamsToString() const override;
	void SerializeOperatorData(Serializer &serializer) const override;

private:
	std::shared_ptr<LocalExchangeState> GetExchangeState(ClientContext &context) const;
	idx_t ResolveNumPartitions(ClientContext &context) const;

	mutable std::shared_ptr<LocalExchangeState> exchange_state;
	mutable std::mutex exchange_lock;

protected:
	PhysicalLocalExchange(PhysicalPlan &physical_plan, PhysicalOperatorType type, vector<LogicalType> types,
	                      std::shared_ptr<RepartitionSpec> repartition_spec, idx_t estimated_cardinality);
};

} // namespace duckdb
