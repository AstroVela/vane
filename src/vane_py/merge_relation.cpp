// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/merge_relation.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/parser/statement/merge_into_statement.hpp"
#include "duckdb/planner/binder.hpp"

namespace duckdb {

MergeRelation::MergeRelation(shared_ptr<Relation> source_p, unique_ptr<MergeIntoStatement> statement_p)
    : Relation(source_p->context, RelationType::EXTENSION_RELATION), source(std::move(source_p)),
      statement(std::move(statement_p)) {
	D_ASSERT(source.get() != this);
	if (!statement) {
		throw InternalException("MergeRelation requires a MERGE INTO statement");
	}
	TryBindRelation(columns);
}

MergeRelation::~MergeRelation() = default;

BoundStatement MergeRelation::Bind(Binder &binder) {
	auto statement_copy = unique_ptr_cast<SQLStatement, MergeIntoStatement>(statement->Copy());
	statement_copy->source = BindRelationInput(binder, *source);
	return binder.Bind(statement_copy->Cast<SQLStatement>());
}

unique_ptr<QueryNode> MergeRelation::GetQueryNode() {
	throw InternalException("Cannot create a query node from a merge relation");
}

string MergeRelation::GetQuery() {
	return string();
}

const vector<ColumnDefinition> &MergeRelation::Columns() {
	return columns;
}

string MergeRelation::ToString(idx_t depth) {
	return RenderWhitespace(depth) + "Merge Into [" + statement->target->ToString() + "]\n" +
	       source->ToString(depth + 1);
}

} // namespace duckdb
