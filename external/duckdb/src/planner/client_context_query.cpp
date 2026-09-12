// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/planner/client_context_query.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/catalog/catalog_entry_retriever.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression_binder.hpp"

namespace duckdb {
namespace {

bool IsLiteralArgument(const ParsedExpression &expr) {
	return expr.GetExpressionClass() == ExpressionClass::CONSTANT ||
	       expr.GetExpressionClass() == ExpressionClass::PARAMETER;
}

bool IsDirectClientFunction(ClientContext &context, ParsedExpression &expression, bool table_function) {
	if (expression.GetExpressionClass() != ExpressionClass::FUNCTION) {
		return false;
	}
	auto &expr = expression.Cast<FunctionExpression>();
	if (expr.is_operator || expr.distinct || expr.filter || expr.export_state ||
	    (expr.order_bys && !expr.order_bys->orders.empty())) {
		return false;
	}
	for (auto &child : expr.children) {
		if (!IsLiteralArgument(*child)) {
			return false;
		}
	}
	// Metadata support is deliberately limited to zero-argument native readers.
	if (table_function && !expr.children.empty()) {
		return false;
	}
	auto type = table_function ? CatalogType::TABLE_FUNCTION_ENTRY : CatalogType::SCALAR_FUNCTION_ENTRY;
	auto catalog = expr.catalog;
	auto schema = expr.schema;
	Binder::BindSchemaOrCatalog(context, catalog, schema);
	CatalogEntryRetriever retriever(context);
	// Lookup only: do not bind arguments, expand macros or autoload extensions.
	auto entry = Catalog::LookupEntry(retriever, catalog, schema, EntryLookupInfo(type, expr.function_name),
	                                  OnEntryNotFound::RETURN_NULL)
	                 .entry;
	if (!entry || !entry->internal || !entry->ParentCatalog().IsSystemCatalog()) {
		return false;
	}
	if (table_function && entry->type == CatalogType::TABLE_FUNCTION_ENTRY) {
		for (auto &function : entry->Cast<TableFunctionCatalogEntry>().functions.functions) {
			if (!function.IsClientContextRead() || function.bind_replace || function.bind_operator) {
				return false;
			}
		}
		return true;
	}
	if (!table_function && entry->type == CatalogType::SCALAR_FUNCTION_ENTRY) {
		static const unordered_set<string> readers = {
		    "current_setting",       "getvariable",      "current_query",          "current_schema", "current_database",
		    "current_connection_id", "current_query_id", "current_transaction_id", "txid_current",   "now",
		    "transaction_timestamp"};
		for (auto &function : entry->Cast<ScalarFunctionCatalogEntry>().functions.functions) {
			if (!function.IsClientContextRead() || !readers.count(function.name)) {
				return false;
			}
		}
		return true;
	}
	return false;
}

bool IsDirectColumn(const ParsedExpression &expr) {
	if (expr.GetExpressionClass() == ExpressionClass::COLUMN_REF) {
		auto &column = expr.Cast<ColumnRefExpression>();
		// SQL value keywords can bind as state functions when no column matches.
		return column.IsQualified() || GetSQLValueFunctionName(column.GetColumnName()).empty();
	}
	if (expr.GetExpressionClass() == ExpressionClass::STAR) {
		auto &star = expr.Cast<StarExpression>();
		return star.exclude_list.empty() && star.replace_list.empty() && star.rename_list.empty() && !star.expr &&
		       !star.columns;
	}
	return false;
}
} // namespace

bool IsClientContextQuery(ClientContext &context, QueryNode &query) {
	if (query.type != QueryNodeType::SELECT_NODE || !query.cte_map.map.empty() || !query.modifiers.empty()) {
		return false;
	}
	auto &select = query.Cast<SelectNode>();
	if (select.where_clause || select.having || select.qualify || select.sample ||
	    !select.groups.group_expressions.empty() || !select.groups.grouping_sets.empty() ||
	    select.aggregate_handling == AggregateHandling::FORCE_AGGREGATES || !select.from_table ||
	    select.from_table->sample || select.from_table->external_dependency) {
		return false;
	}
	if (select.from_table->type == TableReferenceType::EMPTY_FROM) {
		bool has_reader = false;
		for (auto &expr : select.select_list) {
			if (IsLiteralArgument(*expr)) {
				continue;
			}
			if (!IsDirectClientFunction(context, *expr, false)) {
				return false;
			}
			has_reader = true;
		}
		return has_reader;
	}
	if (select.from_table->type == TableReferenceType::TABLE_FUNCTION) {
		auto &ref = select.from_table->Cast<TableFunctionRef>();
		if (ref.subquery || ref.with_ordinality != OrdinalityType::WITHOUT_ORDINALITY ||
		    !IsDirectClientFunction(context, *ref.function, true)) {
			return false;
		}
		for (auto &expr : select.select_list) {
			if (!IsDirectColumn(*expr)) {
				return false;
			}
		}
		return true;
	}
	return false;
}
} // namespace duckdb
