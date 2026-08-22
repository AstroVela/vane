// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/exchange/physical_remote_exchange_sink.hpp
//
// Remote exchange sink that delegates to ExchangeManager SPI.
// Serialized plans carry immutable binding metadata only. A task runner must
// apply one concrete ExchangeSinkInstanceHandle before execution.
// Replaces PhysicalExchangeSink and PhysicalStreamingExchangeSink.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/operator/exchange/repartition.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/distributed/exchange/exchange_manager.hpp"
#include "duckdb/common/optional_idx.hpp"

#include <string>
#include <vector>

namespace duckdb {

class PhysicalRemoteExchangeSink : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXCHANGE_SINK;

public:
	PhysicalRemoteExchangeSink(PhysicalPlan &physical_plan, vector<LogicalType> types, idx_t estimated_cardinality,
	                           std::string exchange_id, idx_t num_partitions, RepartitionSpec::Type repartition_type,
	                           vector<unique_ptr<Expression>> partition_by,
	                           distributed::ExchangeSinkIdentitySource sink_identity_source,
	                           optional_idx plan_task_partition_id, std::string sink_query_id,
	                           std::string sink_output_location_prefix,
	                           std::shared_ptr<distributed::ExchangeManager> exchange_mgr,
	                           vector<string> range_boundaries = {}, vector<string> range_order_modifiers = {});

	bool IsSink() const override {
		return true;
	}

	bool IsSource() const override {
		return true;
	}

	bool ParallelSink() const override {
		return true;
	}

	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;

	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;
	void SerializeOperatorData(Serializer &serializer) const override;

	InsertionOrderPreservingMap<string> ParamsToString() const override;

	const std::string &ExchangeId() const {
		return exchange_id_;
	}
	idx_t NumPartitions() const {
		return num_partitions_;
	}
	distributed::ExchangeSinkIdentitySource SinkIdentitySource() const {
		return sink_identity_source_;
	}
	optional_idx PlanTaskPartitionId() const {
		return plan_task_partition_id_;
	}
	const std::string &SinkQueryId() const {
		return sink_query_id_;
	}
	const std::string &SinkOutputLocationPrefix() const {
		return sink_output_location_prefix_;
	}
	bool HasBoundSinkHandle() const {
		return sink_handle_bound_;
	}
	const distributed::ExchangeSinkInstanceHandle &SinkHandle() const;
	const std::shared_ptr<distributed::ExchangeManager> &GetExchangeManager() const {
		return exchange_mgr_;
	}
	void ApplyRuntimeSinkHandle(distributed::ExchangeSinkInstanceHandle sink_handle) {
		sink_handle_ = std::move(sink_handle);
		sink_handle_bound_ = true;
	}
	const vector<unique_ptr<Expression>> &PartitionBy() const {
		return partition_by_;
	}
	RepartitionSpec::Type RepartitionType() const {
		return repartition_type_;
	}
	const vector<string> &RangeBoundaries() const {
		return range_boundaries_;
	}
	const vector<string> &RangeOrderModifiers() const {
		return range_order_modifiers_;
	}
	void EnableMarkJoinBuildSummary(vector<unique_ptr<Expression>> expressions) {
		collect_mark_join_build_summary_ = true;
		mark_join_build_expressions_ = std::move(expressions);
	}
	bool CollectsMarkJoinBuildSummary() const {
		return collect_mark_join_build_summary_;
	}
	const vector<unique_ptr<Expression>> &MarkJoinBuildExpressions() const {
		return mark_join_build_expressions_;
	}

private:
	static idx_t SelectPartitionHash(const hash_t hash, const idx_t num_partitions);
	static idx_t SelectPartitionRange(const string_t &sort_key, const vector<string> &boundaries,
	                                  const idx_t num_partitions);

	std::string exchange_id_;
	idx_t num_partitions_;
	RepartitionSpec::Type repartition_type_;
	vector<unique_ptr<Expression>> partition_by_;
	distributed::ExchangeSinkIdentitySource sink_identity_source_;
	optional_idx plan_task_partition_id_;
	std::string sink_query_id_;
	std::string sink_output_location_prefix_;
	mutable distributed::ExchangeSinkInstanceHandle sink_handle_;
	mutable bool sink_handle_bound_ = false;
	std::shared_ptr<distributed::ExchangeManager> exchange_mgr_;
	vector<string> range_boundaries_;
	vector<string> range_order_modifiers_;
	bool collect_mark_join_build_summary_ = false;
	vector<unique_ptr<Expression>> mark_join_build_expressions_;
};

} // namespace duckdb
