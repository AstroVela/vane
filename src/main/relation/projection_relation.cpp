// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/main/relation/projection_relation.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/common/exception/parser_exception.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression_binder/relation_binder.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

#include <unordered_map>

namespace duckdb {

ProjectionRelation::ProjectionRelation(shared_ptr<Relation> child_p,
                                       vector<unique_ptr<ParsedExpression>> parsed_expressions, vector<string> aliases)
    : Relation(child_p->context, RelationType::PROJECTION_RELATION), expressions(std::move(parsed_expressions)),
      child(std::move(child_p)) {
	if (!aliases.empty()) {
		if (aliases.size() != expressions.size()) {
			throw ParserException("Aliases list length must match expression list length!");
		}
		for (idx_t i = 0; i < aliases.size(); i++) {
			expressions[i]->SetAlias(aliases[i]);
		}
	}
	// bind the expressions
	TryBindRelation(columns);
}

unique_ptr<QueryNode> ProjectionRelation::GetQueryNode() {
	auto child_ptr = child.get();
	while (child_ptr->InheritsColumnBindings()) {
		child_ptr = child_ptr->ChildRelation();
	}
	unique_ptr<QueryNode> result;
	if (child_ptr->type == RelationType::JOIN_RELATION) {
		// child node is a join: push projection into the child query node
		result = child->GetQueryNode();
	} else {
		// child node is not a join: create a new select node and push the child as a table reference
		auto select = make_uniq<SelectNode>();
		select->from_table = child->GetTableRef();
		result = std::move(select);
	}
	D_ASSERT(result->type == QueryNodeType::SELECT_NODE);
	auto &select_node = result->Cast<SelectNode>();
	select_node.aggregate_handling = AggregateHandling::NO_AGGREGATES_ALLOWED;
	select_node.select_list.clear();
	for (auto &expr : expressions) {
		select_node.select_list.push_back(expr->Copy());
	}
	return result;
}

BoundStatement ProjectionRelation::Bind(Binder &binder) {
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
				throw InternalException("Failed to build projection bindings: missing column at index %llu", i);
			}
		}
	}

	auto expr_binder = Binder::CreateBinder(binder.context, binder.shared_from_this());
	bool single_binding = names_by_table.size() == 1;
	for (auto &entry : names_by_table) {
		auto table_index = entry.first;
		auto &names = entry.second;
		auto &types = types_by_table[table_index];
		string alias = single_binding ? child->GetAlias() : StringUtil::Format("__projection_%llu", table_index);
		expr_binder->bind_context.AddGenericBinding(table_index, alias, names, types);
	}

	vector<unique_ptr<ParsedExpression>> projection_exprs;
	projection_exprs.reserve(expressions.size());
	for (auto &expr : expressions) {
		if (StarExpression::IsStar(*expr) || StarExpression::IsColumns(*expr) ||
		    StarExpression::IsColumnsUnpacked(*expr)) {
			return Relation::Bind(binder);
		}
		projection_exprs.push_back(expr->Copy());
	}

	RelationBinder relation_binder(*expr_binder, binder.context, "projection");
	vector<unique_ptr<Expression>> bound_expressions;
	bound_expressions.reserve(projection_exprs.size());

	BoundStatement result;
	for (auto &expr : projection_exprs) {
		ExpressionBinder::QualifyColumnNames(*expr_binder, expr);
		auto bound_expr = relation_binder.Bind(expr);
		if (!bound_expr) {
			throw InternalException("Failed to bind projection expression");
		}
		result.names.push_back(bound_expr->GetName());
		result.types.push_back(bound_expr->return_type);
		bound_expressions.push_back(std::move(bound_expr));
	}
	if (bound_expressions.empty()) {
		throw BinderException("Projection list is empty");
	}

	auto projection = make_uniq<LogicalProjection>(binder.GenerateTableIndex(), std::move(bound_expressions));
	projection->AddChild(std::move(child_bound.plan));
	result.plan = std::move(projection);

	binder.MoveCorrelatedExpressionsFrom(*expr_binder);
	return result;
}

string ProjectionRelation::GetAlias() {
	return child->GetAlias();
}

const vector<ColumnDefinition> &ProjectionRelation::Columns() {
	return columns;
}

string ProjectionRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Projection [";
	for (idx_t i = 0; i < expressions.size(); i++) {
		if (i != 0) {
			str += ", ";
		}
		str += expressions[i]->ToString() + " as " + expressions[i]->GetAlias();
	}
	str += "]\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb
