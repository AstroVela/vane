// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/execution/distributed/common_types.hpp"

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/join/physical_delim_join.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include <iostream>

namespace duckdb {

PhysicalColumnDataScan::PhysicalColumnDataScan(PhysicalPlan &physical_plan, vector<LogicalType> types,
                                               PhysicalOperatorType op_type, idx_t estimated_cardinality,
                                               optionally_owned_ptr<ColumnDataCollection> collection_p)
    : PhysicalOperator(physical_plan, op_type, std::move(types), estimated_cardinality),
      collection(std::move(collection_p)), cte_index(DConstants::INVALID_INDEX) {
}

PhysicalColumnDataScan::PhysicalColumnDataScan(PhysicalPlan &physical_plan, vector<LogicalType> types,
                                               PhysicalOperatorType op_type, idx_t estimated_cardinality,
                                               idx_t cte_index)
    : PhysicalOperator(physical_plan, op_type, std::move(types), estimated_cardinality), collection(nullptr),
      cte_index(cte_index) {
}

class PhysicalColumnDataGlobalScanState : public GlobalSourceState {
public:
	explicit PhysicalColumnDataGlobalScanState(const ColumnDataCollection &collection)
	    : max_threads(MaxValue<idx_t>(collection.ChunkCount(), 1)) {
		collection.InitializeScan(global_scan_state);
	}

	idx_t MaxThreads() override {
		return max_threads;
	}

public:
	ColumnDataParallelScanState global_scan_state;

	const idx_t max_threads;
};

class PhysicalColumnDataLocalScanState : public LocalSourceState {
public:
	ColumnDataLocalScanState local_scan_state;
};

unique_ptr<GlobalSourceState> PhysicalColumnDataScan::GetGlobalSourceState(ClientContext &context) const {
	// DEBUG: Check collection before dereferencing

	if (!collection) {
		throw InternalException("PhysicalColumnDataScan::GetGlobalSourceState - collection is null");
	}

	return make_uniq<PhysicalColumnDataGlobalScanState>(*collection);
}

unique_ptr<LocalSourceState> PhysicalColumnDataScan::GetLocalSourceState(ExecutionContext &,
                                                                         GlobalSourceState &) const {
	return make_uniq<PhysicalColumnDataLocalScanState>();
}

SourceResultType PhysicalColumnDataScan::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                         OperatorSourceInput &input) const {
	auto &gstate = input.global_state.Cast<PhysicalColumnDataGlobalScanState>();
	auto &lstate = input.local_state.Cast<PhysicalColumnDataLocalScanState>();
	collection->Scan(gstate.global_scan_state, lstate.local_scan_state, chunk);
	return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalColumnDataScan::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	// check if there is any additional action we need to do depending on the type
	auto &state = meta_pipeline.GetState();
	switch (type) {
	case PhysicalOperatorType::DELIM_SCAN: {
		auto entry = state.delim_join_dependencies.find(*this);
		if (entry == state.delim_join_dependencies.end()) {
			for (auto &dep : state.delim_join_dependencies) {
			}
		}
		D_ASSERT(entry != state.delim_join_dependencies.end());
		// this chunk scan introduces a dependency to the current pipeline
		// namely a dependency on the duplicate elimination pipeline to finish
		auto delim_dependency = entry->second.get().shared_from_this();
		auto delim_sink = state.GetPipelineSink(*delim_dependency);
		D_ASSERT(delim_sink);
		D_ASSERT(delim_sink->type == PhysicalOperatorType::LEFT_DELIM_JOIN ||
		         delim_sink->type == PhysicalOperatorType::RIGHT_DELIM_JOIN);
		auto &delim_join = delim_sink->Cast<PhysicalDelimJoin>();
		current.AddDependency(delim_dependency);
		state.SetPipelineSource(current, delim_join.distinct.Cast<PhysicalOperator>());
		return;
	}
	case PhysicalOperatorType::CTE_SCAN: {
		auto entry = state.cte_dependencies.find(*this);
		D_ASSERT(entry != state.cte_dependencies.end());
		// this chunk scan introduces a dependency to the current pipeline
		// namely a dependency on the CTE pipeline to finish
		auto cte_dependency = entry->second.get().shared_from_this();
		auto cte_sink = state.GetPipelineSink(*cte_dependency);
		(void)cte_sink;
		D_ASSERT(cte_sink);
		D_ASSERT(cte_sink->type == PhysicalOperatorType::CTE);
		current.AddDependency(cte_dependency);
		state.SetPipelineSource(current, *this);
		return;
	}
	case PhysicalOperatorType::RECURSIVE_RECURRING_CTE_SCAN:
	case PhysicalOperatorType::RECURSIVE_CTE_SCAN:
		if (!meta_pipeline.HasRecursiveCTE()) {
			throw InternalException("Recursive CTE scan found without recursive CTE node");
		}
		break;
	default:
		break;
	}
	D_ASSERT(children.empty());
	state.SetPipelineSource(current, *this);
}

InsertionOrderPreservingMap<string> PhysicalColumnDataScan::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	switch (type) {
	case PhysicalOperatorType::DELIM_SCAN:
		if (delim_index.IsValid()) {
			result["Delim Index"] = StringUtil::Format("%llu", delim_index.GetIndex());
		}
		break;
	case PhysicalOperatorType::RECURSIVE_RECURRING_CTE_SCAN:
	case PhysicalOperatorType::CTE_SCAN:
	case PhysicalOperatorType::RECURSIVE_CTE_SCAN: {
		result["CTE Index"] = StringUtil::Format("%llu", cte_index);
		break;
	}
	default:
		break;
	}
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

void PhysicalColumnDataScan::SerializeOperatorData(Serializer &serializer) const {
	serializer.WriteProperty(103, "cte_index", cte_index);
	serializer.WriteProperty(104, "delim_index", delim_index);
	auto serialize_collection = collection.get() != nullptr && type != PhysicalOperatorType::CTE_SCAN &&
	                            type != PhysicalOperatorType::RECURSIVE_CTE_SCAN &&
	                            type != PhysicalOperatorType::RECURSIVE_RECURRING_CTE_SCAN;
	serializer.WriteProperty(105, "has_collection", serialize_collection);
	if (serialize_collection) {
		serializer.WriteProperty(106, "collection", collection);
	}
	// Persist source_node_id for distributed pset routing (SourceId-based injection)
	serializer.WritePropertyWithDefault(107, "source_node_id", source_node_id);
}

} // namespace duckdb
