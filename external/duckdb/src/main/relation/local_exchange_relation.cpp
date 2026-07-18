// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/main/relation/local_exchange_relation.hpp"
#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/execution/operator/exchange/repartition.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/operator/logical_local_exchange.hpp"

namespace duckdb {

LocalExchangeRelation::LocalExchangeRelation(shared_ptr<Relation> child_p, idx_t num_partitions_p)
    : Relation(child_p->context, RelationType::LOCAL_EXCHANGE_RELATION), num_partitions(num_partitions_p),
      child(std::move(child_p)) {
	D_ASSERT(child.get() != this);
	TryBindRelation(columns);
}

unique_ptr<QueryNode> LocalExchangeRelation::GetQueryNode() {
	// LocalExchange has no SQL representation, but we must NOT return
	// child->GetQueryNode() directly.  Doing so makes this node invisible
	// to downstream Relation operations (e.g. LimitRelation::GetQueryNode())
	// which modify the child query node in-place and silently drop the
	// LOCAL_EXCHANGE from the final plan.
	//
	// Instead, wrap the child as a subquery table-reference so that the
	// child's plan is opaque to the parent.  When this node is actually
	// bound, LocalExchangeRelation::Bind() (not the base-class SQL path)
	// is called and properly inserts LogicalLocalExchange.
	auto select = make_uniq<SelectNode>();
	select->from_table = child->GetTableRef();
	select->select_list.push_back(make_uniq<StarExpression>());
	return std::move(select);
}

string LocalExchangeRelation::GetQuery() {
	return string();
}

string LocalExchangeRelation::GetAlias() {
	return child->GetAlias();
}

const vector<ColumnDefinition> &LocalExchangeRelation::Columns() {
	return columns;
}

BoundStatement LocalExchangeRelation::Bind(Binder &binder) {
	auto child_bound = child->Bind(binder);

	size_t num_partitions_sz = static_cast<size_t>(num_partitions);
	auto spec = RepartitionSpec::create_random(num_partitions_sz);

	auto local_exchange = make_uniq<LogicalLocalExchange>(std::move(spec));
	local_exchange->children.push_back(std::move(child_bound.plan));
	child_bound.plan = std::move(local_exchange);
	return child_bound;
}

string LocalExchangeRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "LocalExchange";
	if (num_partitions) {
		str += " [" + std::to_string(num_partitions) + "]";
	}
	str += "\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb
