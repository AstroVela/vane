// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/main/relation/projection_relation.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/common/exception/parser_exception.hpp"
#include "duckdb/planner/binder.hpp"

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
	unique_ptr<QueryNode> result;
	if (RequiresSQLMultiSourceBinding(*child)) {
		// The child has multiple source bindings: push projection into its query node.
		result = child->GetQueryNode();
	} else {
		// The child has one source binding: create a new select node around its table reference.
		auto select = make_uniq<SelectNode>();
		select->from_table = GetTableRefForSerialization(*child);
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
	if (!RequiresDirectRelationBinding(binder, *child)) {
		return Relation::Bind(binder);
	}
	auto select_node = make_uniq<SelectNode>();
	select_node->aggregate_handling = AggregateHandling::NO_AGGREGATES_ALLOWED;
	select_node->select_list.reserve(expressions.size());
	for (auto &expr : expressions) {
		select_node->select_list.push_back(expr->Copy());
	}
	return BindSelectNodeOnChild(binder, *child, std::move(select_node));
}

bool ProjectionRelation::CanSerializeToQueryNodeInternal(Binder &binder) {
	if (!child->CanSerializeToQueryNodeInternal(binder)) {
		return false;
	}
	if (!child->InheritsColumnBindings() || RequiresSQLMultiSourceBinding(*child)) {
		return true;
	}
	auto serialization_binder = Binder::CreateBinder(binder.context);
	auto serialization_input = BindRelationInput(*serialization_binder, *child);
	return std::all_of(expressions.begin(), expressions.end(), [&](const auto &expression) {
		return CanSerializeExpressionOnBoundChild(*serialization_binder, *child, *serialization_input, *expression);
	});
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
