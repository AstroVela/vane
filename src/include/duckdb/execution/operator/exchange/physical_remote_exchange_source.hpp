// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/exchange/physical_remote_exchange_source.hpp
//
// Remote exchange source that delegates to ExchangeManager SPI.
// Replaces PhysicalExchangeSource and PhysicalStreamingExchangeSource.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/optional_idx.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/distributed/exchange/exchange_manager.hpp"

#include <string>
#include <vector>

namespace duckdb {

namespace distributed {
struct ExchangeSourceTaskDescriptor;
class FteSplitQueue;
} // namespace distributed

class PhysicalRemoteExchangeSource : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXCHANGE_SOURCE;

public:
	PhysicalRemoteExchangeSource(PhysicalPlan &physical_plan, vector<LogicalType> types, idx_t estimated_cardinality,
	                             std::string exchange_id, vector<idx_t> partition_indices,
	                             std::vector<distributed::ExchangeSourceHandle> source_handles,
	                             std::shared_ptr<distributed::ExchangeManager> exchange_mgr,
	                             const vector<std::string> &source_nodes,
	                             optional_idx runtime_source_node_id = optional_idx());

	bool IsSource() const override {
		return true;
	}

	bool ParallelSource() const override {
		return true;
	}

	OrderPreservationType SourceOrder() const override {
		return OrderPreservationType::NO_ORDER;
	}

	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                 GlobalSourceState &gstate) const override;
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;
	void SerializeOperatorData(Serializer &serializer) const override;

	InsertionOrderPreservingMap<string> ParamsToString() const override;

	const std::string &ExchangeId() const {
		return exchange_id_;
	}
	const vector<idx_t> &PartitionIndices() const {
		return partition_indices_;
	}
	const std::vector<distributed::ExchangeSourceHandle> &SourceHandles() const {
		return source_handles_;
	}
	const vector<std::string> &SourceNodes() const {
		return source_nodes_;
	}
	const optional_idx &RuntimeSourceNodeId() const {
		return runtime_source_node_id_;
	}
	void ApplyRuntimeTaskDescriptor(const distributed::ExchangeSourceTaskDescriptor &descriptor);
	void ApplyRuntimeSplitQueue(std::shared_ptr<distributed::FteSplitQueue> queue);

private:
	std::string exchange_id_;
	vector<idx_t> partition_indices_;
	std::vector<distributed::ExchangeSourceHandle> source_handles_;
	std::shared_ptr<distributed::ExchangeManager> exchange_mgr_;
	vector<std::string> source_nodes_;
	optional_idx runtime_source_node_id_;
	std::shared_ptr<distributed::FteSplitQueue> runtime_split_queue_;
	idx_t runtime_source_partition_count_ = 0;
	idx_t runtime_source_task_count_ = 0;
};

} // namespace duckdb
