// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/main/relation/order_relation.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/query_node.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/planner/binder.hpp"

namespace duckdb {

OrderRelation::OrderRelation(shared_ptr<Relation> child_p, vector<OrderByNode> orders)
    : Relation(child_p->context, RelationType::ORDER_RELATION), orders(std::move(orders)), child(std::move(child_p)) {
	D_ASSERT(child.get() != this);
	// bind the expressions
	TryBindRelation(columns);
}

unique_ptr<QueryNode> OrderRelation::GetQueryNode() {
	if (orders.empty()) {
		return child->GetQueryNode();
	}
	unique_ptr<QueryNode> result;
	if (RequiresSQLMultiSourceBinding(*child)) {
		result = child->GetQueryNode();
	} else {
		auto select = make_uniq<SelectNode>();
		select->from_table = GetTableRefForSerialization(*child);
		select->select_list.push_back(make_uniq<StarExpression>());
		result = std::move(select);
	}
	if (std::any_of(result->modifiers.begin(), result->modifiers.end(), [](const auto &modifier) {
		    return modifier->type == ResultModifierType::ORDER_MODIFIER ||
		           modifier->type == ResultModifierType::LIMIT_MODIFIER ||
		           modifier->type == ResultModifierType::LIMIT_PERCENT_MODIFIER;
	    })) {
		result = WrapQueryNode(std::move(result), child->GetAlias(), child->Columns());
	}
	D_ASSERT(result->type == QueryNodeType::SELECT_NODE);
	auto &select = result->Cast<SelectNode>();
	auto order_node = make_uniq<OrderModifier>();
	for (idx_t i = 0; i < orders.size(); i++) {
		order_node->orders.emplace_back(orders[i].type, orders[i].null_order, orders[i].expression->Copy());
	}
	select.modifiers.push_back(std::move(order_node));
	return result;
}

BoundStatement OrderRelation::Bind(Binder &binder) {
	if (orders.empty()) {
		return child->Bind(binder);
	}
	if (!RequiresDirectRelationBinding(binder, *child)) {
		return Relation::Bind(binder);
	}
	auto select_node = make_uniq<SelectNode>();
	select_node->select_list.push_back(make_uniq<StarExpression>());
	auto order_node = make_uniq<OrderModifier>();
	for (auto &order : orders) {
		order_node->orders.emplace_back(order.type, order.null_order, order.expression->Copy());
	}
	select_node->modifiers.push_back(std::move(order_node));
	return BindSelectNodeOnChild(binder, *child, std::move(select_node));
}

BoundStatement OrderRelation::BindAsInput(Binder &binder) {
	return BindOrderOnChild(binder, *child, orders);
}

bool OrderRelation::CanSerializeToQueryNodeInternal(Binder &binder) {
	if (!ChildCanSerializeToQueryNode(*child, binder)) {
		return false;
	}
	if (orders.empty() || !child->InheritsColumnBindings() || RequiresSQLMultiSourceBinding(*child)) {
		return true;
	}
	auto serialization_binder = Binder::CreateBinder(binder.context);
	auto serialization_input = BindRelationInput(*serialization_binder, *child);
	return std::all_of(orders.begin(), orders.end(), [&](const auto &order) {
		return CanSerializeExpressionOnBoundChild(*serialization_binder, *child, *serialization_input,
		                                          *order.expression);
	});
}

string OrderRelation::GetAlias() {
	return child->GetAlias();
}

const vector<ColumnDefinition> &OrderRelation::Columns() {
	return columns;
}

string OrderRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Order [";
	for (idx_t i = 0; i < orders.size(); i++) {
		if (i != 0) {
			str += ", ";
		}
		str += orders[i].expression->ToString() + (orders[i].type == OrderType::ASCENDING ? " ASC" : " DESC");
	}
	str += "]\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb
