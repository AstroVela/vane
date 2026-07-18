// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/physical_operator_visitor.hpp"

#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/expression/list.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/order/physical_order.hpp"
#include "duckdb/execution/operator/helper/physical_limit.hpp"
#include "duckdb/execution/operator/join/physical_nested_loop_join.hpp"
#include "duckdb/execution/operator/join/physical_blockwise_nl_join.hpp"

namespace duckdb {

void PhysicalOperatorVisitor::VisitOperator(PhysicalOperator &op) {
	VisitOperatorChildren(op);
	VisitOperatorExpressions(op);
}

void PhysicalOperatorVisitor::VisitOperatorChildren(PhysicalOperator &op) {
	for (auto &child : op.children) {
		VisitOperator(child.get());
	}
}

void PhysicalOperatorVisitor::VisitOperatorExpressions(PhysicalOperator &op) {
	PhysicalOperatorVisitor::EnumerateExpressions(op, [&](unique_ptr<Expression> *child) { VisitExpression(child); });
}

void PhysicalOperatorVisitor::VisitExpression(unique_ptr<Expression> *expression) {
	auto &expr = **expression;
	unique_ptr<Expression> result;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::BOUND_AGGREGATE:
		result = VisitReplace(expr.Cast<BoundAggregateExpression>(), expression);
		break;
	case ExpressionClass::BOUND_BETWEEN:
		result = VisitReplace(expr.Cast<BoundBetweenExpression>(), expression);
		break;
	case ExpressionClass::BOUND_CASE:
		result = VisitReplace(expr.Cast<BoundCaseExpression>(), expression);
		break;
	case ExpressionClass::BOUND_CAST:
		result = VisitReplace(expr.Cast<BoundCastExpression>(), expression);
		break;
	case ExpressionClass::BOUND_COLUMN_REF:
		result = VisitReplace(expr.Cast<BoundColumnRefExpression>(), expression);
		break;
	case ExpressionClass::BOUND_COMPARISON:
		result = VisitReplace(expr.Cast<BoundComparisonExpression>(), expression);
		break;
	case ExpressionClass::BOUND_CONJUNCTION:
		result = VisitReplace(expr.Cast<BoundConjunctionExpression>(), expression);
		break;
	case ExpressionClass::BOUND_CONSTANT:
		result = VisitReplace(expr.Cast<BoundConstantExpression>(), expression);
		break;
	case ExpressionClass::BOUND_FUNCTION:
		result = VisitReplace(expr.Cast<BoundFunctionExpression>(), expression);
		break;
	case ExpressionClass::BOUND_SUBQUERY:
		result = VisitReplace(expr.Cast<BoundSubqueryExpression>(), expression);
		break;
	case ExpressionClass::BOUND_OPERATOR:
		result = VisitReplace(expr.Cast<BoundOperatorExpression>(), expression);
		break;
	case ExpressionClass::BOUND_PARAMETER:
		result = VisitReplace(expr.Cast<BoundParameterExpression>(), expression);
		break;
	case ExpressionClass::BOUND_REF:
		result = VisitReplace(expr.Cast<BoundReferenceExpression>(), expression);
		break;
	case ExpressionClass::BOUND_DEFAULT:
		result = VisitReplace(expr.Cast<BoundDefaultExpression>(), expression);
		break;
	case ExpressionClass::BOUND_WINDOW:
		result = VisitReplace(expr.Cast<BoundWindowExpression>(), expression);
		break;
	case ExpressionClass::BOUND_UNNEST:
		result = VisitReplace(expr.Cast<BoundUnnestExpression>(), expression);
		break;
	default:
		throw InternalException("Unrecognized expression type in physical operator visitor");
	}
	if (result) {
		*expression = std::move(result);
	} else {
		// visit the children of this node
		VisitExpressionChildren(expr);
	}
}

void PhysicalOperatorVisitor::VisitExpressionChildren(Expression &expr) {
	ExpressionIterator::EnumerateChildren(expr, [&](unique_ptr<Expression> &expr) { VisitExpression(&expr); });
}

unique_ptr<Expression> PhysicalOperatorVisitor::VisitReplace(BoundAggregateExpression &expr,
                                                             unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}
unique_ptr<Expression> PhysicalOperatorVisitor::VisitReplace(BoundBetweenExpression &expr,
                                                             unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}
unique_ptr<Expression> PhysicalOperatorVisitor::VisitReplace(BoundCaseExpression &expr,
                                                             unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}
unique_ptr<Expression> PhysicalOperatorVisitor::VisitReplace(BoundCastExpression &expr,
                                                             unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}
unique_ptr<Expression> PhysicalOperatorVisitor::VisitReplace(BoundColumnRefExpression &expr,
                                                             unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}
unique_ptr<Expression> PhysicalOperatorVisitor::VisitReplace(BoundComparisonExpression &expr,
                                                             unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}
unique_ptr<Expression> PhysicalOperatorVisitor::VisitReplace(BoundConjunctionExpression &expr,
                                                             unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}
unique_ptr<Expression> PhysicalOperatorVisitor::VisitReplace(BoundConstantExpression &expr,
                                                             unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}
unique_ptr<Expression> PhysicalOperatorVisitor::VisitReplace(BoundDefaultExpression &expr,
                                                             unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}
unique_ptr<Expression> PhysicalOperatorVisitor::VisitReplace(BoundFunctionExpression &expr,
                                                             unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}
unique_ptr<Expression> PhysicalOperatorVisitor::VisitReplace(BoundOperatorExpression &expr,
                                                             unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}
unique_ptr<Expression> PhysicalOperatorVisitor::VisitReplace(BoundReferenceExpression &expr,
                                                             unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}
unique_ptr<Expression> PhysicalOperatorVisitor::VisitReplace(BoundSubqueryExpression &expr,
                                                             unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}
unique_ptr<Expression> PhysicalOperatorVisitor::VisitReplace(BoundParameterExpression &expr,
                                                             unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}
unique_ptr<Expression> PhysicalOperatorVisitor::VisitReplace(BoundWindowExpression &expr,
                                                             unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}
unique_ptr<Expression> PhysicalOperatorVisitor::VisitReplace(BoundUnnestExpression &expr,
                                                             unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

void PhysicalOperatorVisitor::EnumerateExpressions(PhysicalOperator &op,
                                                   const std::function<void(unique_ptr<Expression> *child)> &callback) {
	switch (op.type) {
	case PhysicalOperatorType::FILTER: {
		auto &filter = op.Cast<PhysicalFilter>();
		if (filter.expression) {
			callback(&filter.expression);
		}
		break;
	}
	case PhysicalOperatorType::PROJECTION: {
		auto &proj = op.Cast<PhysicalProjection>();
		for (auto &expr : proj.select_list) {
			callback(&expr);
		}
		break;
	}
	case PhysicalOperatorType::HASH_GROUP_BY: {
		auto &hagg = op.Cast<PhysicalHashAggregate>();
		for (auto &g : hagg.grouped_aggregate_data.groups) {
			callback(&g);
		}
		for (auto &a : hagg.grouped_aggregate_data.aggregates) {
			callback(&a);
		}
		break;
	}
	case PhysicalOperatorType::ORDER_BY: {
		auto &order = op.Cast<PhysicalOrder>();
		for (auto &node : order.orders) {
			if (node.expression) {
				callback(&node.expression);
			}
		}
		break;
	}
	case PhysicalOperatorType::LIMIT: {
		auto &limit = op.Cast<PhysicalLimit>();
		if (limit.limit_val.Type() == LimitNodeType::EXPRESSION_VALUE ||
		    limit.limit_val.Type() == LimitNodeType::EXPRESSION_PERCENTAGE) {
			callback(&const_cast<BoundLimitNode &>(limit.limit_val).GetExpression());
		}
		if (limit.offset_val.Type() == LimitNodeType::EXPRESSION_VALUE ||
		    limit.offset_val.Type() == LimitNodeType::EXPRESSION_PERCENTAGE) {
			callback(&const_cast<BoundLimitNode &>(limit.offset_val).GetExpression());
		}
		break;
	}
	case PhysicalOperatorType::NESTED_LOOP_JOIN: {
		auto &join = op.Cast<PhysicalNestedLoopJoin>();
		if (join.predicate) {
			callback(&join.predicate);
		}
		break;
	}
	case PhysicalOperatorType::BLOCKWISE_NL_JOIN: {
		auto &join = op.Cast<PhysicalBlockwiseNLJoin>();
		if (join.condition) {
			callback(&join.condition);
		}
		break;
	}
	default:
		// Default: no expressions to enumerate
		break;
	}
}

} // namespace duckdb
