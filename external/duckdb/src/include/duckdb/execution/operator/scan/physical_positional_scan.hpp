//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/scan/physical_positional_scan.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/storage/data_table.hpp"

namespace duckdb {

struct PositionalScanDeserializeTag {};

//! Represents a scan of a base table
class PhysicalPositionalScan : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::POSITIONAL_SCAN;

public:
	//! Regular Table Scan
	PhysicalPositionalScan(PhysicalPlan &physical_plan, vector<LogicalType> types, PhysicalOperator &left,
	                       PhysicalOperator &right);
	PhysicalPositionalScan(PhysicalPlan &physical_plan, PositionalScanDeserializeTag, vector<LogicalType> types,
	                       vector<reference<PhysicalOperator>> child_tables, idx_t estimated_cardinality);

	//! The child table functions
	vector<reference<PhysicalOperator>> child_tables;

public:
	bool Equals(const PhysicalOperator &other) const override;
	vector<const_reference<PhysicalOperator>> GetInputChildren() const override;
	vector<const_reference<PhysicalOperator>> GetChildren() const override;

	OrderPreservationType OperatorOrder() const override {
		return OrderPreservationType::FIXED_ORDER;
	}

public:
	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                 GlobalSourceState &gstate) const override;
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;

	ProgressData GetProgress(ClientContext &context, GlobalSourceState &gstate) const override;

	bool IsSource() const override {
		return true;
	}

protected:
	void SerializeOperatorData(Serializer &serializer) const override;
};

} // namespace duckdb
