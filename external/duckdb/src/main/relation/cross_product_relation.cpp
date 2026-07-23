// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/main/relation/cross_product_relation.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/tableref/joinref.hpp"
#include "duckdb/planner/binder.hpp"

namespace duckdb {

CrossProductRelation::CrossProductRelation(shared_ptr<Relation> left_p, shared_ptr<Relation> right_p,
                                           JoinRefType ref_type)
    : Relation(left_p->context, RelationType::CROSS_PRODUCT_RELATION), left(std::move(left_p)),
      right(std::move(right_p)), ref_type(ref_type) {
	if (left->context->GetContext() != right->context->GetContext()) {
		throw InvalidInputException("Cannot combine LEFT and RIGHT relations of different connections!");
	}
	TryBindRelation(columns);
}

unique_ptr<QueryNode> CrossProductRelation::GetQueryNode() {
	auto result = make_uniq<SelectNode>();
	result->select_list.push_back(make_uniq<StarExpression>());
	result->from_table = GetTableRefForSerialization(*this);
	return std::move(result);
}

BoundStatement CrossProductRelation::BindAsInput(Binder &binder) {
	if (ContainsNonSQLRelation()) {
		return Relation::Bind(binder);
	}
	auto table_ref = GetTableRefForSerialization(*this);
	return binder.Bind(*table_ref);
}

unique_ptr<TableRef> CrossProductRelation::GetTableRefInternal() {
	auto cross_product_ref = make_uniq<JoinRef>(ref_type);
	cross_product_ref->left = GetTableRefForSerialization(*left);
	cross_product_ref->right = GetTableRefForSerialization(*right);
	return std::move(cross_product_ref);
}

const vector<ColumnDefinition> &CrossProductRelation::Columns() {
	return this->columns;
}

string CrossProductRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth);
	str = "Cross Product";
	return str + "\n" + left->ToString(depth + 1) + right->ToString(depth + 1);
}

} // namespace duckdb
