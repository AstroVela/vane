// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Parameter capture uses DuckDB's AST traversal; native Relation and Binder
// implementations remain responsible for query binding and source lifetimes.

#include "vane_python/query_parameters.hpp"
#include "duckdb/common/unordered_set.hpp"

#include "duckdb/parser/expression/cast_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/parameter_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/parser/parsed_expression_iterator.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/basetableref.hpp"
#include "duckdb/parser/tableref/joinref.hpp"
#include "duckdb/parser/tableref/pivotref.hpp"
#include "duckdb/parser/tableref/showref.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"

namespace duckdb {

void CaptureParameters(unique_ptr<ParsedExpression> &expression,
                       const case_insensitive_map_t<BoundParameterData> &parameters) {
	if (expression->GetExpressionClass() == ExpressionClass::PARAMETER) {
		auto &parameter = expression->Cast<ParameterExpression>();
		auto entry = parameters.find(parameter.identifier);
		if (entry == parameters.end()) {
			throw InvalidInputException("Value was not provided for parameter %s", parameter.identifier);
		}
		auto &data = entry->second;
		unique_ptr<ParsedExpression> constant = make_uniq<ConstantExpression>(data.GetValue());
		if (data.return_type != data.GetValue().type() && data.return_type.id() != LogicalTypeId::STRING_LITERAL &&
		    data.return_type.id() != LogicalTypeId::INTEGER_LITERAL) {
			constant = make_uniq<CastExpression>(data.return_type, std::move(constant));
		}
		constant->SetAlias(expression->GetAlias());
		constant->SetQueryLocation(expression->GetQueryLocation());
		expression = std::move(constant);
		return;
	}
	if (expression->GetExpressionClass() == ExpressionClass::SUBQUERY) {
		CaptureQueryParameters(*expression->Cast<SubqueryExpression>().subquery->node, parameters);
	}
	ParsedExpressionIterator::EnumerateChildren(
	    *expression, [&](unique_ptr<ParsedExpression> &child) { CaptureParameters(child, parameters); });
}

void CaptureQueryParameters(QueryNode &node, const case_insensitive_map_t<BoundParameterData> &parameters) {
	// An implicit alias on a PIVOT aggregate changes its output naming contract.
	unordered_set<const ParsedExpression *> pivot_aggregates;
	ParsedExpressionIterator::EnumerateQueryNodeChildren(
	    node, [](unique_ptr<ParsedExpression> &) {},
	    [&](TableRef &ref) {
		    if (ref.type == TableReferenceType::PIVOT) {
			    for (auto &aggregate : ref.Cast<PivotRef>().aggregates) {
				    pivot_aggregates.insert(aggregate.get());
			    }
		    }
	    });
	ParsedExpressionIterator::EnumerateQueryNodeChildren(
	    node,
	    [&](unique_ptr<ParsedExpression> &expression) {
		    auto name = expression->GetName();
		    bool has_star = false;
		    ParsedExpressionIterator::VisitExpression<StarExpression>(*expression,
		                                                              [&](const StarExpression &) { has_star = true; });
		    const bool preserve_name = !has_star && !pivot_aggregates.count(expression.get());
		    CaptureParameters(expression, parameters);
		    if (preserve_name && expression->GetAlias().empty()) {
			    expression->SetAlias(std::move(name));
		    }
	    },
	    [&](TableRef &ref) {
		    // Native traversal leaves these syntactic fields to their owning
		    // binder. Capture API parameters without changing engine traversal.
		    switch (ref.type) {
		    case TableReferenceType::BASE_TABLE: {
			    auto &table = ref.Cast<BaseTableRef>();
			    if (table.at_clause) {
				    CaptureParameters(table.at_clause->ExpressionMutable(), parameters);
			    }
			    break;
		    }
		    case TableReferenceType::TABLE_FUNCTION: {
			    auto &function = ref.Cast<TableFunctionRef>();
			    if (function.subquery) {
				    CaptureQueryParameters(*function.subquery->node, parameters);
			    }
			    break;
		    }
		    case TableReferenceType::JOIN:
			    for (auto &expression : ref.Cast<JoinRef>().duplicate_eliminated_columns) {
				    CaptureParameters(expression, parameters);
			    }
			    break;
		    case TableReferenceType::SHOW_REF: {
			    auto &show = ref.Cast<ShowRef>();
			    if (show.query) {
				    CaptureQueryParameters(*show.query, parameters);
			    }
			    break;
		    }
		    case TableReferenceType::PIVOT:
			    for (auto &pivot : ref.Cast<PivotRef>().pivots) {
				    for (auto &expression : pivot.pivot_expressions) {
					    CaptureParameters(expression, parameters);
				    }
				    for (auto &entry : pivot.entries) {
					    if (entry.expr) {
						    CaptureParameters(entry.expr, parameters);
					    }
				    }
				    if (pivot.subquery) {
					    CaptureQueryParameters(*pivot.subquery, parameters);
				    }
			    }
			    break;
		    default:
			    break;
		    }
	    });
}

} // namespace duckdb
