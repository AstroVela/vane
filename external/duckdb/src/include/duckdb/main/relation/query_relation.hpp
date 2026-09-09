// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/relation/query_relation.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/relation.hpp"
#include "duckdb/parser/query_node.hpp"
#include "duckdb/planner/expression/bound_parameter_data.hpp"

namespace duckdb {
class SelectStatement;

class QueryRelation : public Relation {
public:
	QueryRelation(const shared_ptr<ClientContext> &context, unique_ptr<SelectStatement> select_stmt, string alias,
	              const string &query = "", case_insensitive_map_t<BoundParameterData> parameters = {});
	~QueryRelation() override;
	static void CaptureParameters(unique_ptr<ParsedExpression> &expression,
	                              const case_insensitive_map_t<BoundParameterData> &parameters);

	unique_ptr<SelectStatement> select_stmt;
	string query;
	string alias;
	vector<ColumnDefinition> columns;

public:
	static unique_ptr<SelectStatement> ParseStatement(ClientContext &context, const string &query, const string &error);
	unique_ptr<QueryNode> GetQueryNode() override;
	string GetQuery() override;
	BoundStatement Bind(Binder &binder) override;

	const vector<ColumnDefinition> &Columns() override;
	string ToString(idx_t depth) override;
	string GetAlias() override;

private:
	case_insensitive_map_t<BoundParameterData> parameters;
	unique_ptr<TableRef> GetTableRefInternal() override;
	unique_ptr<SelectStatement> GetSelectStatement();
};

} // namespace duckdb
