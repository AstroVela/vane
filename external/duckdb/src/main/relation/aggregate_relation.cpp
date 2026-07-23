// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/main/relation/aggregate_relation.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/planner/binder.hpp"

namespace duckdb {

AggregateRelation::AggregateRelation(shared_ptr<Relation> child_p,
                                     vector<unique_ptr<ParsedExpression>> parsed_expressions)
    : Relation(child_p->context, RelationType::AGGREGATE_RELATION), expressions(std::move(parsed_expressions)),
      child(std::move(child_p)) {
	// bind the expressions
	TryBindRelation(columns);
}

AggregateRelation::AggregateRelation(shared_ptr<Relation> child_p,
                                     vector<unique_ptr<ParsedExpression>> parsed_expressions, GroupByNode groups_p)
    : Relation(child_p->context, RelationType::AGGREGATE_RELATION), expressions(std::move(parsed_expressions)),
      groups(std::move(groups_p)), child(std::move(child_p)) {
	// bind the expressions
	Relation::TryBindRelation(columns);
}

AggregateRelation::AggregateRelation(shared_ptr<Relation> child_p,
                                     vector<unique_ptr<ParsedExpression>> parsed_expressions,
                                     vector<unique_ptr<ParsedExpression>> groups_p)
    : Relation(child_p->context, RelationType::AGGREGATE_RELATION), expressions(std::move(parsed_expressions)),
      child(std::move(child_p)) {
	if (!groups_p.empty()) {
		// explicit groups provided: use standard handling
		GroupingSet grouping_set;
		for (idx_t i = 0; i < groups_p.size(); i++) {
			groups.group_expressions.push_back(std::move(groups_p[i]));
			grouping_set.insert(i);
		}
		groups.grouping_sets.push_back(std::move(grouping_set));
	}
	// bind the expressions
	TryBindRelation(columns);
}

unique_ptr<QueryNode> AggregateRelation::GetQueryNode() {
	unique_ptr<QueryNode> result;
	if (RequiresSQLMultiSourceBinding(*child)) {
		// The child has multiple source bindings: push aggregation into its query node.
		result = child->GetQueryNode();
	} else {
		// The child has one source binding: create a new select node around its table reference.
		auto select = make_uniq<SelectNode>();
		select->from_table = GetTableRefForSerialization(*child);
		result = std::move(select);
	}
	D_ASSERT(result->type == QueryNodeType::SELECT_NODE);
	auto &select_node = result->Cast<SelectNode>();
	if (!groups.group_expressions.empty()) {
		select_node.aggregate_handling = AggregateHandling::STANDARD_HANDLING;
		select_node.groups = groups.Copy();
	} else {
		// no groups provided: automatically figure out groups (if any)
		select_node.aggregate_handling = AggregateHandling::FORCE_AGGREGATES;
	}
	select_node.select_list.clear();
	for (auto &expr : expressions) {
		select_node.select_list.push_back(expr->Copy());
	}
	return result;
}

BoundStatement AggregateRelation::Bind(Binder &binder) {
	if (!RequiresDirectRelationBinding(binder, *child)) {
		return Relation::Bind(binder);
	}
	auto select_node = make_uniq<SelectNode>();
	if (!groups.group_expressions.empty()) {
		select_node->aggregate_handling = AggregateHandling::STANDARD_HANDLING;
		select_node->groups = groups.Copy();
	} else {
		select_node->aggregate_handling = AggregateHandling::FORCE_AGGREGATES;
	}
	for (auto &expr : expressions) {
		select_node->select_list.push_back(expr->Copy());
	}
	return BindSelectNodeOnChild(binder, *child, std::move(select_node));
}

bool AggregateRelation::CanSerializeToQueryNodeInternal(Binder &binder) {
	if (!child->CanSerializeToQueryNodeInternal(binder)) {
		return false;
	}
	if (!child->InheritsColumnBindings() || RequiresSQLMultiSourceBinding(*child)) {
		return true;
	}
	auto serialization_binder = Binder::CreateBinder(binder.context);
	auto serialization_input = BindRelationInput(*serialization_binder, *child);
	for (auto &expression : expressions) {
		if (!CanSerializeExpressionOnBoundChild(*serialization_binder, *child, *serialization_input, *expression)) {
			return false;
		}
	}
	for (auto &group : groups.group_expressions) {
		if (!CanSerializeExpressionOnBoundChild(*serialization_binder, *child, *serialization_input, *group)) {
			return false;
		}
	}
	return true;
}

string AggregateRelation::GetAlias() {
	return child->GetAlias();
}

const vector<ColumnDefinition> &AggregateRelation::Columns() {
	return columns;
}

string AggregateRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Aggregate [";
	for (idx_t i = 0; i < expressions.size(); i++) {
		if (i != 0) {
			str += ", ";
		}
		str += expressions[i]->ToString();
	}
	str += "]\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb
