// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/main/relation/query_relation.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/cast_expression.hpp"
#include "duckdb/parser/expression/parameter_expression.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/parser/parsed_expression_iterator.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/parser/tableref/joinref.hpp"
#include "duckdb/parser/tableref/pivotref.hpp"
#include "duckdb/parser/tableref/showref.hpp"
#include "duckdb/parser/tableref/basetableref.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/bound_statement.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/bound_parameter_map.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/planner/query_node/bound_select_node.hpp"
#include "duckdb/parser/common_table_expression_info.hpp"
#include "duckdb/parser/query_node/cte_node.hpp"

namespace duckdb {

using UnpackedColumnNameCaptures = vector<shared_ptr<UnpackedColumnNameCapture>>;

static void CaptureQueryParameters(QueryNode &node, const case_insensitive_map_t<BoundParameterData> &parameters,
                                   optional_ptr<UnpackedColumnNameCaptures> name_captures = nullptr);

QueryRelation::QueryRelation(const shared_ptr<ClientContext> &context, unique_ptr<SelectStatement> select_stmt_p,
                             string alias_p, const string &query_p,
                             case_insensitive_map_t<BoundParameterData> parameters_p)
    : Relation(context, RelationType::QUERY_RELATION), select_stmt(std::move(select_stmt_p)), query(query_p),
      alias(std::move(alias_p)), parameters(std::move(parameters_p)) {
	if (query.empty()) {
		query = select_stmt->ToString();
	}
	UnpackedColumnNameCaptures name_captures;
	if (!parameters.empty()) {
		// Bind an unchanged statement while observing names produced by *COLUMNS.
		// AST copies share the captures; subsequent binds only read the snapshot.
		CaptureQueryParameters(*select_stmt->node, parameters, name_captures);
	}
	TryBindRelation(columns);
	for (auto &capture : name_captures) {
		capture->active = false;
	}
}

QueryRelation::~QueryRelation() {
}

unique_ptr<SelectStatement> QueryRelation::ParseStatement(ClientContext &context, const string &query,
                                                          const string &error) {
	Parser parser(context.GetParserOptions());
	parser.ParseQuery(query);
	if (parser.statements.size() != 1) {
		throw ParserException(error);
	}
	if (parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw ParserException(error);
	}
	return unique_ptr_cast<SQLStatement, SelectStatement>(std::move(parser.statements[0]));
}

static void CaptureExpressionParameters(unique_ptr<ParsedExpression> &expression,
                                        const case_insensitive_map_t<BoundParameterData> &parameters,
                                        optional_ptr<UnpackedColumnNameCaptures> name_captures) {
	if (!name_captures && expression->GetExpressionClass() == ExpressionClass::PARAMETER) {
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
		auto &subquery = expression->Cast<SubqueryExpression>();
		CaptureQueryParameters(*subquery.subquery->node, parameters, name_captures);
	}
	ParsedExpressionIterator::EnumerateChildren(*expression, [&](unique_ptr<ParsedExpression> &child) {
		CaptureExpressionParameters(child, parameters, name_captures);
	});
}

static void CaptureQueryParameters(QueryNode &node, const case_insensitive_map_t<BoundParameterData> &parameters,
                                   optional_ptr<UnpackedColumnNameCaptures> name_captures) {
	unordered_set<const ParsedExpression *> pivot_aggregates;
	ParsedExpressionIterator::EnumerateQueryNodeChildren(
	    node, [](unique_ptr<ParsedExpression> &) {},
	    [&](TableRef &ref) {
		    if (ref.type == TableReferenceType::PIVOT && ref.Cast<PivotRef>().aggregates.size() == 1) {
			    for (auto &aggregate : ref.Cast<PivotRef>().aggregates) {
				    pivot_aggregates.insert(aggregate.get());
			    }
		    }
	    });
	ParsedExpressionIterator::EnumerateQueryNodeChildren(
	    node,
	    [&](unique_ptr<ParsedExpression> &expression) {
		    if (name_captures) {
			    CaptureExpressionParameters(expression, parameters, name_captures);
			    return;
		    }
		    // Keep SQL output names such as "($1 + 1)" stable when the value replaces
		    // its placeholder. Nested expressions retain their own original aliases.
		    auto &capture = expression->unpacked_column_name;
		    const bool has_unpacked_name = capture && !capture->name.empty();
		    auto name = has_unpacked_name ? capture->name : expression->GetName();
		    const bool preserve_name = pivot_aggregates.find(expression.get()) == pivot_aggregates.end();
		    bool has_star = false;
		    ParsedExpressionIterator::VisitExpression<StarExpression>(*expression,
		                                                              [&](const StarExpression &) { has_star = true; });
		    CaptureExpressionParameters(expression, parameters, nullptr);
		    // Giving a PIVOT aggregate an implicit alias changes its output column
		    // names (e.g. "1" becomes "1_count_star()"). Preserve explicit aliases only.
		    // COLUMNS can be nested inside operators, functions or casts. Leave their
		    // aliases implicit so the binder can name each expanded output column.
		    // Argument unpacking keeps one output, named after expanding arguments.
		    if (preserve_name && (!has_star || has_unpacked_name) && expression->GetAlias().empty()) {
			    expression->SetAlias(std::move(name));
		    }
	    },
	    [&](TableRef &ref) {
		    if (ref.type == TableReferenceType::BASE_TABLE) {
			    auto &table = ref.Cast<BaseTableRef>();
			    if (table.at_clause) {
				    CaptureExpressionParameters(table.at_clause->ExpressionMutable(), parameters, name_captures);
			    }
			    return;
		    }
		    if (ref.type == TableReferenceType::TABLE_FUNCTION) {
			    auto &function = ref.Cast<TableFunctionRef>();
			    if (function.subquery) {
				    CaptureQueryParameters(*function.subquery->node, parameters, name_captures);
			    }
			    return;
		    }
		    if (ref.type == TableReferenceType::JOIN) {
			    for (auto &expression : ref.Cast<JoinRef>().duplicate_eliminated_columns) {
				    CaptureExpressionParameters(expression, parameters, name_captures);
			    }
			    return;
		    }
		    if (ref.type == TableReferenceType::SHOW_REF) {
			    auto &show = ref.Cast<ShowRef>();
			    if (show.query) {
				    CaptureQueryParameters(*show.query, parameters, name_captures);
			    }
			    return;
		    }
		    if (ref.type != TableReferenceType::PIVOT) {
			    return;
		    }
		    // The shared iterator omits pivot keys and entries: unlike ordinary
		    // expressions, names in a PIVOT IN list can represent literal values.
		    // Capture parameters here without changing other visitors' treatment of
		    // those names or synthesizing aliases for pivot keys and entries.
		    for (auto &pivot : ref.Cast<PivotRef>().pivots) {
			    for (auto &expression : pivot.pivot_expressions) {
				    CaptureExpressionParameters(expression, parameters, name_captures);
			    }
			    for (auto &entry : pivot.entries) {
				    if (entry.expr) {
					    CaptureExpressionParameters(entry.expr, parameters, name_captures);
				    }
			    }
			    if (pivot.subquery) {
				    CaptureQueryParameters(*pivot.subquery, parameters, name_captures);
			    }
		    }
	    },
	    [&](QueryNode &query_node) {
		    if (!name_captures || query_node.type != QueryNodeType::SELECT_NODE) {
			    return;
		    }
		    for (auto &expression : query_node.Cast<SelectNode>().select_list) {
			    if (!expression->GetAlias().empty()) {
				    continue;
			    }
			    bool has_star = false;
			    ParsedExpressionIterator::VisitExpression<StarExpression>(
			        *expression, [&](const StarExpression &) { has_star = true; });
			    if (has_star) {
				    auto capture = make_shared_ptr<UnpackedColumnNameCapture>();
				    expression->unpacked_column_name = capture;
				    name_captures->push_back(std::move(capture));
			    }
		    }
	    });
}

void QueryRelation::CaptureParameters(unique_ptr<ParsedExpression> &expression,
                                      const case_insensitive_map_t<BoundParameterData> &parameters) {
	CaptureExpressionParameters(expression, parameters, nullptr);
}

unique_ptr<SelectStatement> QueryRelation::GetSelectStatement() {
	auto statement = unique_ptr_cast<SQLStatement, SelectStatement>(select_stmt->Copy());
	if (!parameters.empty()) {
		// Query nodes escape this relation when composing, creating views, or
		// exporting SQL. Capture typed values in the AST so independent relations
		// cannot lose or overwrite each other's parameter bindings.
		CaptureQueryParameters(*statement->node, parameters);
		statement->named_param_map.clear();
	}
	return statement;
}

unique_ptr<QueryNode> QueryRelation::GetQueryNode() {
	auto select = GetSelectStatement();
	return std::move(select->node);
}

string QueryRelation::GetQuery() {
	return query;
}

unique_ptr<TableRef> QueryRelation::GetTableRefInternal() {
	auto subquery_ref = make_uniq<SubqueryRef>(GetSelectStatement(), GetAlias());
	return std::move(subquery_ref);
}

BoundStatement QueryRelation::Bind(Binder &binder) {
	// Keep values on the relation so every bind (including Ray plan serialization)
	// uses DuckDB's parameter type inference without executing the query locally.
	BoundParameterMap parameter_map(parameters);
	struct ParameterScope {
		Binder &binder;
		optional_ptr<BoundParameterMap> previous;
		~ParameterScope() {
			binder.SetParameters(previous);
		}
	} parameter_scope {binder, binder.GetParameters()};
	if (!parameters.empty()) {
		binder.SetParameters(parameter_map);
	}
	auto saved_binding_mode = binder.GetBindingMode();
	binder.SetBindingMode(BindingMode::EXTRACT_REPLACEMENT_SCANS);
	bool first_bind = columns.empty();
	// Validate the original prepared statement with DuckDB's normal parameter
	// rules before any query-node export replaces placeholders with constants.
	BoundStatement result;
	if (parameters.empty()) {
		result = Relation::Bind(binder);
	} else {
		auto statement = select_stmt->Copy();
		result = binder.Bind(*statement);
	}
	auto &replacements = binder.GetReplacementScans();
	if (first_bind) {
		for (auto &kv : replacements) {
			auto &name = kv.first;
			auto &tableref = kv.second;

			if (!tableref->external_dependency) {
				// Only push a CTE for objects that are out of our control (i.e Python)
				// This makes sure replacement scans for files (parquet/csv/json etc) are not transformed into a CTE
				continue;
			}

			auto select = make_uniq<SelectStatement>();
			auto select_node = make_uniq<SelectNode>();
			select_node->select_list.push_back(make_uniq<StarExpression>());
			select_node->from_table = std::move(tableref);
			select->node = std::move(select_node);

			auto cte_info = make_uniq<CommonTableExpressionInfo>();
			cte_info->query = std::move(select);

			auto subquery = make_uniq<SubqueryRef>(std::move(select_stmt), "query_relation");
			auto top_level_select = make_uniq<SelectStatement>();
			auto top_level_select_node = make_uniq<SelectNode>();
			top_level_select_node->select_list.push_back(make_uniq<StarExpression>());
			top_level_select_node->from_table = std::move(subquery);
			auto &cte_map = top_level_select_node->cte_map;
			top_level_select->node = std::move(top_level_select_node);
			cte_map.map[name] = std::move(cte_info);
			select_stmt = std::move(top_level_select);
		}
	}
	replacements.clear();
	binder.SetBindingMode(saved_binding_mode);
	return result;
}

string QueryRelation::GetAlias() {
	return alias;
}

const vector<ColumnDefinition> &QueryRelation::Columns() {
	return columns;
}

string QueryRelation::ToString(idx_t depth) {
	return RenderWhitespace(depth) + "Subquery";
}

} // namespace duckdb
