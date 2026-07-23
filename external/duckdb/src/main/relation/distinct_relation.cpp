// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/main/relation/distinct_relation.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/query_node.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression_binder/relation_binder.hpp"
#include "duckdb/planner/operator/logical_distinct.hpp"

namespace duckdb {

DistinctRelation::DistinctRelation(shared_ptr<Relation> child_p)
    : Relation(child_p->context, RelationType::DISTINCT_RELATION), child(std::move(child_p)) {
	D_ASSERT(child.get() != this);
	vector<ColumnDefinition> dummy_columns;
	TryBindRelation(dummy_columns);
}

unique_ptr<QueryNode> DistinctRelation::GetQueryNode() {
	auto child_node = child->GetQueryNode();
	bool plain_distinct = child_node->type == QueryNodeType::SELECT_NODE && child_node->modifiers.size() == 1 &&
	                      child_node->modifiers[0]->type == ResultModifierType::DISTINCT_MODIFIER &&
	                      child_node->modifiers[0]->Cast<DistinctModifier>().distinct_on_targets.empty();
	if (child_node->type != QueryNodeType::SELECT_NODE || (!child_node->modifiers.empty() && !plain_distinct)) {
		child_node = WrapQueryNode(std::move(child_node), child->GetAlias(), child->Columns());
	}
	child_node->AddDistinct();
	return child_node;
}

BoundStatement DistinctRelation::Bind(Binder &binder) {
	if (!RequiresDirectRelationBinding(binder, *child)) {
		return Relation::Bind(binder);
	}
	auto select_node = make_uniq<SelectNode>();
	select_node->select_list.push_back(make_uniq<StarExpression>());
	select_node->AddDistinct();
	return BindSelectNodeOnChild(binder, *child, std::move(select_node));
}

BoundStatement DistinctRelation::BindAsInput(Binder &binder) {
	auto child_ref = BindRelationInput(binder, *child);
	auto child_bound = binder.Bind(*child_ref);
	vector<unique_ptr<ParsedExpression>> visible_columns;
	StarExpression star;
	binder.bind_context.GenerateAllColumnExpressions(star, visible_columns);
	RelationBinder relation_binder(binder, binder.context, "distinct");

	vector<unique_ptr<Expression>> targets;
	targets.reserve(visible_columns.size());
	for (auto &column : visible_columns) {
		auto target = relation_binder.Bind(column);
		ExpressionBinder::PushCollation(binder.context, target, target->return_type);
		targets.push_back(std::move(target));
	}
	auto distinct = make_uniq<LogicalDistinct>(std::move(targets), DistinctType::DISTINCT);
	distinct->AddChild(std::move(child_bound.plan));
	child_bound.plan = std::move(distinct);
	return child_bound;
}

string DistinctRelation::GetAlias() {
	return child->GetAlias();
}

const vector<ColumnDefinition> &DistinctRelation::Columns() {
	return child->Columns();
}

string DistinctRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Distinct\n";
	return str + child->ToString(depth + 1);
	;
}

} // namespace duckdb
