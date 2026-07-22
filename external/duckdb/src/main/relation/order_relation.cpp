// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/main/relation/order_relation.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/parser/query_node.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

namespace duckdb {

static void InlineOrderProjection(unique_ptr<Expression> &expression, const LogicalProjection &projection) {
	if (expression->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &column_ref = expression->Cast<BoundColumnRefExpression>();
		if (column_ref.binding.table_index == projection.table_index) {
			if (column_ref.binding.column_index >= projection.expressions.size()) {
				throw InternalException("Order relation projection reference is out of range");
			}
			expression = projection.expressions[column_ref.binding.column_index]->Copy();
			return;
		}
	}
	ExpressionIterator::EnumerateChildren(
	    *expression, [&](unique_ptr<Expression> &child) { InlineOrderProjection(child, projection); });
}

static bool TryGetOrderProjectionIndex(const Expression &expression, idx_t projection_index, idx_t &column_index) {
	if (expression.type != ExpressionType::BOUND_COLUMN_REF) {
		return false;
	}
	auto &column_ref = expression.Cast<BoundColumnRefExpression>();
	if (column_ref.binding.table_index != projection_index) {
		return false;
	}
	column_index = column_ref.binding.column_index;
	return true;
}

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
		select->from_table = child->GetTableRef();
		select->select_list.push_back(make_uniq<StarExpression>());
		result = std::move(select);
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
	if (!RequiresDirectRelationBinding(*child)) {
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
	if (orders.empty()) {
		auto child_ref = BindRelationInput(binder, *child);
		return binder.Bind(*child_ref);
	}
	auto select_node = make_uniq<SelectNode>();
	select_node->select_list.push_back(make_uniq<StarExpression>());
	auto order_node = make_uniq<OrderModifier>();
	for (auto &order : orders) {
		order_node->orders.emplace_back(order.type, order.null_order, order.expression->Copy());
	}
	select_node->modifiers.push_back(std::move(order_node));
	auto result = BindSelectNodeOnChild(binder, *child, std::move(select_node));

	// The SELECT binder evaluates ORDER BY-only expressions in a temporary
	// projection. Inline those expressions into the LogicalOrder and remove the
	// projection so the child's table bindings remain visible to later relation
	// operators. Window, UNNEST, and subquery plan nodes below it are retained.
	auto root = std::move(result.plan);
	if (root->type == LogicalProjection::TYPE && root->children.size() == 1 &&
	    root->children[0]->type == LogicalOrder::TYPE) {
		root = std::move(root->children[0]);
	}
	if (root->type == LogicalProjection::TYPE) {
		D_ASSERT(root->children.size() == 1);
		result.plan = std::move(root->children[0]);
		return result;
	}
	if (root->type != LogicalOrder::TYPE || root->children.size() != 1 ||
	    root->children[0]->type != LogicalProjection::TYPE) {
		throw InternalException("Unexpected logical plan for an order relation");
	}

	auto &logical_order = root->Cast<LogicalOrder>();
	auto projection_op = std::move(root->children[0]);
	auto &projection = projection_op->Cast<LogicalProjection>();
	D_ASSERT(projection.children.size() == 1);
	unordered_set<idx_t> seen_projection_columns;
	vector<BoundOrderByNode> rewritten_orders;
	rewritten_orders.reserve(logical_order.orders.size());
	for (auto &order : logical_order.orders) {
		idx_t projection_column;
		if (TryGetOrderProjectionIndex(*order.expression, projection.table_index, projection_column) &&
		    !seen_projection_columns.insert(projection_column).second) {
			// The SELECT binder shares identical ORDER BY expressions through one
			// projection slot. A repeated sort key is redundant; dropping it keeps
			// volatile expressions at one evaluation per input row.
			continue;
		}
		InlineOrderProjection(order.expression, projection);
		rewritten_orders.push_back(std::move(order));
	}
	logical_order.orders = std::move(rewritten_orders);
	root->children[0] = std::move(projection.children[0]);
	result.plan = std::move(root);
	return result;
}

bool OrderRelation::CanSerializeToQueryNode() {
	if (!child->CanSerializeToQueryNode()) {
		return false;
	}
	for (auto &order : orders) {
		if (!CanSerializeExpressionOnChild(*child, *order.expression)) {
			return false;
		}
	}
	return true;
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
