// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/main/relation/subquery_relation.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/query_node.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/planner/binder.hpp"

namespace duckdb {

SubqueryRelation::SubqueryRelation(shared_ptr<Relation> child_p, const string &alias_p)
    : Relation(child_p->context, RelationType::SUBQUERY_RELATION, alias_p), child(std::move(child_p)) {
	D_ASSERT(child.get() != this);
	vector<ColumnDefinition> dummy_columns;
	Relation::TryBindRelation(dummy_columns);
}

unique_ptr<QueryNode> SubqueryRelation::GetQueryNode() {
	return child->GetQueryNode();
}

BoundStatement SubqueryRelation::Bind(Binder &binder) {
	if (!RequiresDirectRelationBinding(binder, *child)) {
		return Relation::Bind(binder);
	}
	// An alias is an explicit binding boundary. Bind the child through the
	// transient relation-input path so non-SQL operators remain in the plan.
	// When this relation is used as an input, its projection is exposed under
	// this relation's alias rather than the child's aliases.
	auto select_node = make_uniq<SelectNode>();
	select_node->select_list.push_back(make_uniq<StarExpression>());
	return BindSelectNodeOnChild(binder, *child, std::move(select_node));
}

const vector<ColumnDefinition> &SubqueryRelation::Columns() {
	return child->Columns();
}

string SubqueryRelation::ToString(idx_t depth) {
	return child->ToString(depth);
}

} // namespace duckdb
