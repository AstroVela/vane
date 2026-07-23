// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/main/relation/filter_relation.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/query_node/set_operation_node.hpp"
#include "duckdb/parser/expression/conjunction_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression_binder/where_binder.hpp"

namespace duckdb {

FilterRelation::FilterRelation(shared_ptr<Relation> child_p, unique_ptr<ParsedExpression> condition_p)
    : Relation(child_p->context, RelationType::FILTER_RELATION), condition(std::move(condition_p)),
      child(std::move(child_p)) {
	D_ASSERT(child.get() != this);
	vector<ColumnDefinition> dummy_columns;
	TryBindRelation(dummy_columns);
}

unique_ptr<QueryNode> FilterRelation::GetQueryNode() {
	if (RequiresSQLMultiSourceBinding(*child)) {
		// The child has multiple source bindings: push the filter into its WHERE clause.
		auto child_node = child->GetQueryNode();
		D_ASSERT(child_node->type == QueryNodeType::SELECT_NODE);
		auto &select_node = child_node->Cast<SelectNode>();
		if (!select_node.where_clause) {
			select_node.where_clause = condition->Copy();
		} else {
			select_node.where_clause = make_uniq<ConjunctionExpression>(
			    ExpressionType::CONJUNCTION_AND, std::move(select_node.where_clause), condition->Copy());
		}
		return child_node;
	} else {
		auto result = make_uniq<SelectNode>();
		result->select_list.push_back(make_uniq<StarExpression>());
		result->from_table = GetTableRefForSerialization(*child);
		result->where_clause = condition->Copy();
		return std::move(result);
	}
}

BoundStatement FilterRelation::Bind(Binder &binder) {
	if (!RequiresDirectRelationBinding(binder, *child)) {
		return Relation::Bind(binder);
	}
	auto select_node = make_uniq<SelectNode>();
	select_node->select_list.push_back(make_uniq<StarExpression>());
	select_node->where_clause = condition->Copy();
	return BindSelectNodeOnChild(binder, *child, std::move(select_node));
}

BoundStatement FilterRelation::BindAsInput(Binder &binder) {
	auto child_ref = BindRelationInput(binder, *child);
	auto child_bound = binder.Bind(*child_ref);
	auto condition_copy = condition->Copy();
	ExpandRelationFilter(binder, condition_copy);
	WhereBinder where_binder(binder, binder.context);
	auto bound_condition = where_binder.Bind(condition_copy);
	child_bound.plan = PlanRelationFilter(binder, std::move(bound_condition), std::move(child_bound.plan));
	return child_bound;
}

bool FilterRelation::CanSerializeToQueryNodeInternal(Binder &binder) {
	if (!child->CanSerializeToQueryNodeInternal(binder)) {
		return false;
	}
	if (!child->InheritsColumnBindings() || RequiresSQLMultiSourceBinding(*child)) {
		return true;
	}
	auto serialization_binder = Binder::CreateBinder(binder.context);
	auto serialization_input = BindRelationInput(*serialization_binder, *child);
	return CanSerializeExpressionOnBoundChild(*serialization_binder, *child, *serialization_input, *condition);
}

string FilterRelation::GetAlias() {
	return child->GetAlias();
}

const vector<ColumnDefinition> &FilterRelation::Columns() {
	return child->Columns();
}

string FilterRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Filter [" + condition->ToString() + "]\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb
