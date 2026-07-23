// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/main/relation/unnest_relation.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/operator/logical_unnest.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/expression/bound_unnest_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/common/types.hpp"

namespace duckdb {

UnnestRelation::UnnestRelation(shared_ptr<Relation> child_p, string column_name_p)
    : Relation(child_p->context, RelationType::UNNEST_RELATION), column_name(std::move(column_name_p)),
      child(std::move(child_p)) {
	D_ASSERT(child.get() != this);
	TryBindRelation(columns);
}

unique_ptr<QueryNode> UnnestRelation::GetQueryNode() {
	// Fallback AST path (used by default Relation::Bind if custom Bind not called).
	// Builds: SELECT * EXCLUDE(col), unnest(col) AS col FROM (child)
	auto result = make_uniq<SelectNode>();
	result->from_table = GetTableRefForSerialization(*child);

	auto star = make_uniq<StarExpression>();
	star->exclude_list.insert(QualifiedColumnName(column_name));
	result->select_list.push_back(std::move(star));

	vector<unique_ptr<ParsedExpression>> unnest_args;
	unnest_args.push_back(make_uniq<ColumnRefExpression>(column_name));
	auto unnest_func = make_uniq<FunctionExpression>("unnest", std::move(unnest_args));
	unnest_func->SetAlias(column_name);
	result->select_list.push_back(std::move(unnest_func));

	return std::move(result);
}

BoundStatement UnnestRelation::Bind(Binder &binder) {
	// Custom bind: directly build LogicalUnnest + LogicalProjection, skipping SQL/AST entirely.
	// Pattern follows ProjectionRelation::Bind() but produces unnest plan nodes.

	// 1. Bind the child relation
	auto child_bound = child->Bind(binder);
	auto child_bindings = child_bound.plan->GetColumnBindings();
	D_ASSERT(child_bindings.size() == child_bound.names.size());

	// 2. Find the column to unnest
	idx_t unnest_col_idx = DConstants::INVALID_INDEX;
	for (idx_t i = 0; i < child_bound.names.size(); i++) {
		if (child_bound.names[i] == column_name) {
			unnest_col_idx = i;
			break;
		}
	}
	if (unnest_col_idx == DConstants::INVALID_INDEX) {
		throw BinderException("Column \"%s\" not found in child relation for unnest/explode", column_name);
	}

	// 3. Determine element type from the list/array column
	auto &list_type = child_bound.types[unnest_col_idx];
	LogicalType element_type;
	if (list_type.id() == LogicalTypeId::LIST) {
		element_type = ListType::GetChildType(list_type);
	} else if (list_type.id() == LogicalTypeId::ARRAY) {
		element_type = ArrayType::GetChildType(list_type);
	} else {
		throw BinderException("Cannot unnest/explode column \"%s\" of type %s — expected LIST or ARRAY", column_name,
		                      list_type.ToString());
	}

	// 4. Create BoundUnnestExpression: unnest(col_ref) → element_type
	auto col_ref = make_uniq<BoundColumnRefExpression>(column_name, list_type, child_bindings[unnest_col_idx]);
	auto unnest_expr = make_uniq<BoundUnnestExpression>(element_type);
	unnest_expr->child = std::move(col_ref);

	// 5. Create LogicalUnnest node
	idx_t unnest_index = binder.GenerateTableIndex();
	auto logical_unnest = make_uniq<LogicalUnnest>(unnest_index);
	logical_unnest->expressions.push_back(std::move(unnest_expr));
	logical_unnest->AddChild(std::move(child_bound.plan));

	// 6. Create final LogicalProjection
	// Output columns: all child columns, but replace the list column with the unnested scalar.
	// LogicalUnnest::GetColumnBindings() = child_bindings + [ColumnBinding(unnest_index, 0)]
	// So child columns keep their original bindings, and the unnested column is at (unnest_index, 0).
	idx_t proj_index = binder.GenerateTableIndex();
	vector<unique_ptr<Expression>> proj_exprs;
	BoundStatement result;

	for (idx_t i = 0; i < child_bound.names.size(); i++) {
		if (i == unnest_col_idx) {
			// Replace the list column with the unnested scalar
			auto ref = make_uniq<BoundColumnRefExpression>(column_name, element_type, ColumnBinding(unnest_index, 0));
			proj_exprs.push_back(std::move(ref));
			result.names.push_back(column_name);
			result.types.push_back(element_type);
		} else {
			// Pass-through: reference the child's original column binding
			auto ref =
			    make_uniq<BoundColumnRefExpression>(child_bound.names[i], child_bound.types[i], child_bindings[i]);
			proj_exprs.push_back(std::move(ref));
			result.names.push_back(child_bound.names[i]);
			result.types.push_back(child_bound.types[i]);
		}
	}

	auto projection = make_uniq<LogicalProjection>(proj_index, std::move(proj_exprs));
	projection->AddChild(std::move(logical_unnest));
	result.plan = std::move(projection);

	return result;
}

string UnnestRelation::GetAlias() {
	return child->GetAlias();
}

const vector<ColumnDefinition> &UnnestRelation::Columns() {
	return columns;
}

string UnnestRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Unnest [" + column_name + "]\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb
