// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/main/relation/order_relation.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/parser/query_node.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_order.hpp"

namespace duckdb {

OrderRelation::OrderRelation(shared_ptr<Relation> child_p, vector<OrderByNode> orders)
    : Relation(child_p->context, RelationType::ORDER_RELATION), orders(std::move(orders)), child(std::move(child_p)) {
	D_ASSERT(child.get() != this);
	// bind the expressions
	TryBindRelation(columns);
}

unique_ptr<QueryNode> OrderRelation::GetQueryNode() {
	auto select = make_uniq<SelectNode>();
	select->from_table = child->GetTableRef();
	select->select_list.push_back(make_uniq<StarExpression>());
	auto order_node = make_uniq<OrderModifier>();
	for (idx_t i = 0; i < orders.size(); i++) {
		order_node->orders.emplace_back(orders[i].type, orders[i].null_order, orders[i].expression->Copy());
	}
	select->modifiers.push_back(std::move(order_node));
	return std::move(select);
}

BoundStatement OrderRelation::Bind(Binder &binder) {
	if (!CanMapColumnBindings(*child)) {
		return Relation::Bind(binder);
	}
	bool order_by_all = false;
	for (auto &order : orders) {
		if (order.expression->HasSubquery() || order.expression->IsAggregate() || order.expression->IsWindow()) {
			return Relation::Bind(binder);
		}
		if (order.expression->GetExpressionClass() == ExpressionClass::CONSTANT) {
			auto &constant = order.expression->Cast<ConstantExpression>();
			if (!constant.value.type().IsIntegral()) {
				return Relation::Bind(binder);
			}
		}
		if (order.expression->GetExpressionType() == ExpressionType::STAR) {
			auto &star = order.expression->Cast<StarExpression>();
			if (orders.size() != 1 || !star.exclude_list.empty() || !star.replace_list.empty() || star.expr) {
				return Relation::Bind(binder);
			}
			order_by_all = true;
		}
	}

	auto child_bound = child->Bind(binder);
	auto bindings = child_bound.plan->GetColumnBindings();
	D_ASSERT(bindings.size() == child_bound.names.size());
	D_ASSERT(bindings.size() == child_bound.types.size());

	auto &config = DBConfig::GetConfig(binder.context);
	vector<BoundOrderByNode> bound_orders;
	for (auto &order : orders) {
		auto order_type = config.ResolveOrder(binder.context, order.type);
		auto null_order = config.ResolveNullOrder(binder.context, order_type, order.null_order);
		if (order_by_all) {
			for (idx_t column_idx = 0; column_idx < bindings.size(); column_idx++) {
				unique_ptr<Expression> expression = make_uniq<BoundColumnRefExpression>(
				    child_bound.names[column_idx], child_bound.types[column_idx], bindings[column_idx]);
				ExpressionBinder::PushCollation(binder.context, expression, expression->return_type);
				bound_orders.emplace_back(order_type, null_order, std::move(expression));
			}
			continue;
		}
		if (order.expression->GetExpressionClass() == ExpressionClass::CONSTANT) {
			auto &constant = order.expression->Cast<ConstantExpression>();
			auto order_value = constant.value.GetValue<int64_t>();
			if (order_value <= 0 || NumericCast<idx_t>(order_value) > bindings.size()) {
				throw BinderException(*order.expression, "ORDER term out of range - should be between 1 and %llu",
				                      bindings.size());
			}
			auto column_idx = NumericCast<idx_t>(order_value - 1);
			unique_ptr<Expression> expression = make_uniq<BoundColumnRefExpression>(
			    child_bound.names[column_idx], child_bound.types[column_idx], bindings[column_idx]);
			ExpressionBinder::PushCollation(binder.context, expression, expression->return_type);
			bound_orders.emplace_back(order_type, null_order, std::move(expression));
			continue;
		}
		auto expression = BindExpressionOnBoundRelation(binder, *child, child_bound, order.expression->Copy(), "order");
		ExpressionBinder::PushCollation(binder.context, expression, expression->return_type);
		bound_orders.emplace_back(order_type, null_order, std::move(expression));
	}

	auto logical_order = make_uniq<LogicalOrder>(std::move(bound_orders));
	logical_order->AddChild(std::move(child_bound.plan));
	child_bound.plan = std::move(logical_order);
	return child_bound;
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
