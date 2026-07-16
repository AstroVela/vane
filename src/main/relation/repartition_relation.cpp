// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/main/relation/repartition_relation.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/operator/exchange/repartition.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression_binder/relation_binder.hpp"
#include "duckdb/planner/operator/logical_repartition.hpp"

#include <unordered_map>

namespace duckdb {

RepartitionRelation::RepartitionRelation(shared_ptr<Relation> child_p, idx_t num_partitions_p,
                                         vector<unique_ptr<ParsedExpression>> partition_by_p)
    : Relation(child_p->context, RelationType::REPARTITION_RELATION), num_partitions(num_partitions_p),
      partition_by(std::move(partition_by_p)), child(std::move(child_p)) {
	D_ASSERT(child.get() != this);
	TryBindRelation(columns);
}

unique_ptr<QueryNode> RepartitionRelation::GetQueryNode() {
	// Repartition is not representable in SQL; fall back to the child query node.
	// This keeps relation->TableFunction (map_batches) working, at the cost of
	// dropping repartition in SQL-based paths.
	return child->GetQueryNode();
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
	auto child_bound = child->Bind(binder);
	auto bindings = child_bound.plan->GetColumnBindings();
	D_ASSERT(bindings.size() == child_bound.names.size());
	D_ASSERT(bindings.size() == child_bound.types.size());

	std::unordered_map<idx_t, vector<string>> names_by_table;
	std::unordered_map<idx_t, vector<LogicalType>> types_by_table;

	for (idx_t i = 0; i < bindings.size(); i++) {
		const auto &binding = bindings[i];
		auto &names = names_by_table[binding.table_index];
		auto &types = types_by_table[binding.table_index];
		if (names.size() <= binding.column_index) {
			names.resize(binding.column_index + 1);
			types.resize(binding.column_index + 1);
		}
		names[binding.column_index] = child_bound.names[i];
		types[binding.column_index] = child_bound.types[i];
	}

	for (auto &entry : names_by_table) {
		auto &names = entry.second;
		for (idx_t i = 0; i < names.size(); i++) {
			if (names[i].empty()) {
				throw InternalException("Failed to build repartition bindings: missing column at index %llu", i);
			}
		}
	}

	auto expr_binder = Binder::CreateBinder(binder.context, binder.shared_from_this());
	bool single_binding = names_by_table.size() == 1;
	for (auto &entry : names_by_table) {
		auto table_index = entry.first;
		auto &names = entry.second;
		auto &types = types_by_table[table_index];
		string alias = single_binding ? child->GetAlias() : StringUtil::Format("__repartition_%llu", table_index);
		expr_binder->bind_context.AddGenericBinding(table_index, alias, names, types);
	}

	RelationBinder relation_binder(*expr_binder, binder.context, "repartition");
	vector<ExprRef> partition_exprs;
	partition_exprs.reserve(partition_by.size());
	vector<unique_ptr<Expression>> bound_partition_exprs;
	bound_partition_exprs.reserve(partition_by.size());
	for (auto &expr : partition_by) {
		auto expr_copy = expr->Copy();
		auto bound_expr = relation_binder.Bind(expr_copy);
		if (bound_expr) {
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
