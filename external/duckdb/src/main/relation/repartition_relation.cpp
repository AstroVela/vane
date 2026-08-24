// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/main/relation/repartition_relation.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/type_visitor.hpp"
#include "duckdb/execution/operator/exchange/repartition.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression_binder/relation_binder.hpp"
#include "duckdb/planner/operator/logical_repartition.hpp"

namespace duckdb {

RepartitionRelation::RepartitionRelation(shared_ptr<Relation> child_p, idx_t num_partitions_p,
                                         vector<unique_ptr<ParsedExpression>> partition_by_p)
    : Relation(child_p->context, RelationType::REPARTITION_RELATION), num_partitions(num_partitions_p),
      partition_by(std::move(partition_by_p)), child(std::move(child_p)) {
	D_ASSERT(child.get() != this);
	TryBindRelation(columns);
}

unique_ptr<QueryNode> RepartitionRelation::GetQueryNode() {
	throw NotImplementedException(
	    "A repartitioned relation has no SQL query-node representation; converting it would discard the exchange");
}

string RepartitionRelation::GetQuery() {
	return string();
}

string RepartitionRelation::GetAlias() {
	return child->GetAlias();
}

const vector<ColumnDefinition> &RepartitionRelation::Columns() {
	return columns;
}

BoundStatement RepartitionRelation::Bind(Binder &binder) {
	auto select_node = make_uniq<SelectNode>();
	select_node->select_list.push_back(make_uniq<StarExpression>());
	return BindSelectNodeOnChild(binder, *this, std::move(select_node));
}

BoundStatement RepartitionRelation::BindAsInput(Binder &binder) {
	auto child_ref = BindRelationInput(binder, *child);
	auto child_bound = binder.Bind(*child_ref);
	RelationBinder relation_binder(binder, binder.context, "repartition");
	vector<ExprRef> partition_exprs;
	partition_exprs.reserve(partition_by.size());
	vector<unique_ptr<Expression>> bound_partition_exprs;
	bound_partition_exprs.reserve(partition_by.size());
	for (auto &expr : partition_by) {
		auto expr_copy = expr->Copy();
		auto bound_expr = relation_binder.Bind(expr_copy);
		if (bound_expr) {
			if (TypeVisitor::Contains(bound_expr->return_type, FileLogicalType::IsFile)) {
				throw BinderException(bound_expr->GetQueryLocation(), "Repartition keys do not support FILE values");
			}
			partition_exprs.emplace_back(bound_expr->Copy());
			bound_partition_exprs.push_back(std::move(bound_expr));
		}
	}

	size_t num_partitions_sz = static_cast<size_t>(num_partitions);

	std::shared_ptr<RepartitionSpec> spec;
	if (partition_exprs.empty()) {
		spec = RepartitionSpec::create_random(num_partitions_sz);
	} else {
		spec = RepartitionSpec::create_hash(num_partitions_sz, std::move(partition_exprs));
	}

	auto repartition = make_uniq<LogicalRepartition>(std::move(spec));
	repartition->expressions = std::move(bound_partition_exprs);
	repartition->children.push_back(std::move(child_bound.plan));
	child_bound.plan = std::move(repartition);
	return child_bound;
}

string RepartitionRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Repartition";
	if (num_partitions) {
		str += " [" + std::to_string(num_partitions) + "]";
	}
	if (!partition_by.empty()) {
		str += " By [";
		for (idx_t i = 0; i < partition_by.size(); i++) {
			if (i > 0) {
				str += ", ";
			}
			str += partition_by[i]->ToString();
		}
		str += "]";
	}
	str += "\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb
