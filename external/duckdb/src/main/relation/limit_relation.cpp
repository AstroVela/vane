// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/main/relation/limit_relation.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/query_node.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/common/to_string.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"

namespace duckdb {

LimitRelation::LimitRelation(shared_ptr<Relation> child_p, int64_t limit, int64_t offset)
    : Relation(child_p->context, RelationType::LIMIT_RELATION), limit(limit), offset(offset),
      child(std::move(child_p)) {
	D_ASSERT(child.get() != this);
}

unique_ptr<QueryNode> LimitRelation::GetQueryNode() {
	auto child_node = child->GetQueryNode();
	auto limit_node = make_uniq<LimitModifier>();
	if (limit >= 0) {
		limit_node->limit = make_uniq<ConstantExpression>(Value::BIGINT(limit));
	}
	if (offset > 0) {
		limit_node->offset = make_uniq<ConstantExpression>(Value::BIGINT(offset));
	}

	child_node->modifiers.push_back(std::move(limit_node));
	return child_node;
}

string LimitRelation::GetAlias() {
	return child->GetAlias();
}

const vector<ColumnDefinition> &LimitRelation::Columns() {
	return child->Columns();
}

string LimitRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Limit " + to_string(limit);
	if (offset > 0) {
		str += " Offset " + to_string(offset);
	}
	str += "\n";
	return str + child->ToString(depth + 1);
}

BoundStatement LimitRelation::Bind(Binder &binder) {
	// Bind child directly (NOT through the SQL GetQueryNode() round-trip)
	// to preserve non-SQL-representable child nodes like LocalExchangeRelation.
	auto child_bound = child->Bind(binder);

	BoundLimitNode limit_val_bound;
	BoundLimitNode offset_val_bound;
	if (limit >= 0) {
		limit_val_bound = BoundLimitNode::ConstantValue(limit);
	}
	if (offset > 0) {
		offset_val_bound = BoundLimitNode::ConstantValue(offset);
	}
	auto limit_node = make_uniq<LogicalLimit>(std::move(limit_val_bound), std::move(offset_val_bound));
	limit_node->AddChild(std::move(child_bound.plan));
	child_bound.plan = std::move(limit_node);
	return child_bound;
}

} // namespace duckdb
