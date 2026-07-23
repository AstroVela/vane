// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/main/relation/explain_relation.hpp"
#include "duckdb/parser/statement/explain_statement.hpp"
#include "duckdb/parser/statement/relation_statement.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/parsed_data/create_view_info.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/main/client_context.hpp"

namespace duckdb {

ExplainRelation::ExplainRelation(shared_ptr<Relation> child_p, ExplainType type, ExplainFormat format)
    : Relation(child_p->context, RelationType::EXPLAIN_RELATION), child(std::move(child_p)), type(type),
      format(format) {
	TryBindRelation(columns);
}

BoundStatement ExplainRelation::Bind(Binder &binder) {
	auto relation_stmt = make_uniq<RelationStatement>(child, binder);
	ExplainStatement explain(std::move(relation_stmt), type, format);
	return binder.Bind(explain.Cast<SQLStatement>());
}

unique_ptr<QueryNode> ExplainRelation::GetQueryNode() {
	throw InternalException("Cannot create a query node from an explain relation");
}

string ExplainRelation::GetQuery() {
	return string();
}

const vector<ColumnDefinition> &ExplainRelation::Columns() {
	return columns;
}

string ExplainRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Explain\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb
