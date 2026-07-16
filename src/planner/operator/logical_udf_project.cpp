// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/operator/logical_udf_project.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/planner/operator/logical_udf_project.hpp"

namespace duckdb {

LogicalUDFProject::LogicalUDFProject(idx_t table_index_p, unique_ptr<Expression> udf_expr_p,
                                     string output_column_name_p)
    : LogicalOperator(LogicalOperatorType::LOGICAL_UDF_PROJECT), table_index(table_index_p),
      udf_expr(std::move(udf_expr_p)), output_column_name(std::move(output_column_name_p)) {
}

vector<ColumnBinding> LogicalUDFProject::GetColumnBindings() {
	D_ASSERT(!children.empty());
	if (is_flat_map) {
		// flat_map/map_batches mode: single output column (STRUCT or scalar)
		vector<ColumnBinding> bindings;
		bindings.emplace_back(table_index, 0);
		return bindings;
	}
	auto bindings = children[0]->GetColumnBindings();
	bindings.emplace_back(table_index, 0);
	return bindings;
}

void LogicalUDFProject::ResolveTypes() {
	D_ASSERT(!children.empty());
	if (is_flat_map) {
		// flat_map/map_batches mode: output is UDF result only, no passthrough rows.
		// Use the original return type (STRUCT for multi-column, scalar for single).
		// The STRUCT stays intact through the plan — no flattening here.
		types.push_back(udf_expr->return_type);
		return;
	}
	types = children[0]->types;
	types.push_back(udf_expr->return_type);
}

vector<idx_t> LogicalUDFProject::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

string LogicalUDFProject::GetName() const {
	return LogicalOperator::GetName();
}

} // namespace duckdb
