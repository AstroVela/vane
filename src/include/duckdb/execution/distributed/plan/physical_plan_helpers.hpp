// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

// Minimal helpers for building PhysicalPlans from pipeline node translator and visitors
#pragma once

#include <memory>

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/common/allocator.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/optionally_owned_ptr.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"

namespace duckdb {
class PhysicalPlan;
namespace distributed {

inline DuckPhysicalPlanRef make_physical_plan_from_int_blocks(const duckdb::vector<duckdb::vector<int64_t>> &blocks) {
	Allocator &alloc = Allocator::DefaultAllocator();
	auto plan = std::make_shared<duckdb::PhysicalPlan>(alloc);

	// Single BIGINT column for our integer blocks
	duckdb::vector<duckdb::LogicalType> types = {duckdb::LogicalType::BIGINT};
	auto collection = duckdb::make_uniq<duckdb::ColumnDataCollection>(alloc, types);
	// Initialize append state and add DataChunks for each block
	duckdb::ColumnDataAppendState append_state;
	collection->InitializeAppend(append_state);
	for (const auto &blk : blocks) {
		duckdb::DataChunk chunk;
		chunk.Initialize(alloc, types);
		for (size_t i = 0; i < blk.size(); ++i) {
			chunk.SetValue(0, i, duckdb::Value::BIGINT(blk[i]));
		}
		chunk.SetCardinality((idx_t)blk.size());
		collection->Append(append_state, chunk);
	}

	// Create a PhysicalColumnDataScan operator in the plan using BIGINT types
	plan->Make<duckdb::PhysicalColumnDataScan>(types, duckdb::PhysicalOperatorType::COLUMN_DATA_SCAN, 0,
	                                           std::move(collection));
	return plan;
}

inline DuckPhysicalPlanRef
make_physical_plan_with_identity_projection(const duckdb::vector<duckdb::vector<int64_t>> &blocks) {
	auto plan = make_physical_plan_from_int_blocks(blocks);
	// Compose a projection that selects the first column
	duckdb::vector<duckdb::unique_ptr<Expression>> select_list;
	// Use BoundReferenceExpression to reference column 0
	{
		auto ref = duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0);
		select_list.push_back(duckdb::unique_ptr<Expression>(std::move(ref)));
	}
	duckdb::vector<duckdb::LogicalType> out_types = {duckdb::LogicalType::BIGINT};
	auto &proj = plan->Make<duckdb::PhysicalProjection>(out_types, std::move(select_list), 0);
	plan->SetRoot(proj);
	return plan;
}

} // namespace distributed
} // namespace duckdb
