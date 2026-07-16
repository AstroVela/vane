// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/scan/physical_column_data_scan.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/optionally_owned_ptr.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/physical_operator.hpp"

namespace duckdb {

//! The PhysicalColumnDataScan scans a ColumnDataCollection
class PhysicalColumnDataScan : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::INVALID;

public:
	PhysicalColumnDataScan(PhysicalPlan &physical_plan, vector<LogicalType> types, PhysicalOperatorType op_type,
	                       idx_t estimated_cardinality, optionally_owned_ptr<ColumnDataCollection> collection);

	PhysicalColumnDataScan(PhysicalPlan &physical_plan, vector<LogicalType> types, PhysicalOperatorType op_type,
	                       idx_t estimated_cardinality, idx_t cte_index);

	//! (optionally owned) column data collection to scan
	optionally_owned_ptr<ColumnDataCollection> collection;

	idx_t cte_index;
	optional_idx delim_index;

	//! Source node ID for distributed pset routing (analogous to PhysicalTableScan::scan_node_id).
	//! When set, the distributed translator preserves this ID for the ScanSourceNode,
	//! enabling key-based worker-side data injection.
	optional_idx source_node_id;

public:
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                 GlobalSourceState &gstate) const override;
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;

	bool IsSource() const override {
		return true;
	}

	InsertionOrderPreservingMap<string> ParamsToString() const override;
	bool ParallelSource() const override {
		return true;
	}

public:
	void BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) override;

protected:
	void SerializeOperatorData(Serializer &serializer) const override;
};

} // namespace duckdb
