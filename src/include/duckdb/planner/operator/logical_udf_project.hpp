// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/operator/logical_udf_project.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

class LogicalUDFProject : public LogicalOperator {
public:
	static constexpr const LogicalOperatorType TYPE = LogicalOperatorType::LOGICAL_UDF_PROJECT;

public:
	LogicalUDFProject(idx_t table_index, unique_ptr<Expression> udf_expr, string output_column_name);

	idx_t table_index;
	unique_ptr<Expression> udf_expr;
	string output_column_name;
	bool is_flat_map = false;
	bool is_scalar_map = false;
	bool is_row_preserving_batch = false;
	vector<LogicalType> flat_map_output_types;
	vector<string> flat_map_output_names;

public:
	vector<ColumnBinding> GetColumnBindings() override;
	vector<idx_t> GetTableIndex() const override;
	string GetName() const override;
	void Serialize(Serializer &serializer) const override;
	static unique_ptr<LogicalOperator> Deserialize(Deserializer &deserializer);

protected:
	void ResolveTypes() override;
};

} // namespace duckdb
