// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/planner/client_context_query.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry_retriever.hpp"
#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp"
#include "duckdb/function/lambda_functions.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/parser/parsed_expression_iterator.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_binder.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

namespace duckdb {
namespace {

class ClientContextPlanVisitor : public LogicalOperatorVisitor {
public:
	explicit ClientContextPlanVisitor(bool allow_command_results_p) : allow_command_results(allow_command_results_p) {
	}
	bool eligible = true;
	bool has_context = false;
	bool allow_command_results;

	void VisitOperator(LogicalOperator &op) override {
		switch (op.type) {
		case LogicalOperatorType::LOGICAL_GET:
			if (!op.Cast<LogicalGet>().function.IsClientContextRead()) {
				eligible = false;
			} else {
				has_context = true;
			}
			break;
		case LogicalOperatorType::LOGICAL_CHUNK_GET:
			// MaterializedRelation carries already-completed command results.
			// Other scans in a composed plan still need their own capability.
			eligible &= allow_command_results;
			break;
		case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
		case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
		case LogicalOperatorType::LOGICAL_DELIM_GET:
		case LogicalOperatorType::LOGICAL_CTE_REF:
		case LogicalOperatorType::LOGICAL_PROJECTION:
		case LogicalOperatorType::LOGICAL_FILTER:
		case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		case LogicalOperatorType::LOGICAL_WINDOW:
		case LogicalOperatorType::LOGICAL_UNNEST:
		case LogicalOperatorType::LOGICAL_LIMIT:
		case LogicalOperatorType::LOGICAL_ORDER_BY:
		case LogicalOperatorType::LOGICAL_DISTINCT:
		case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		case LogicalOperatorType::LOGICAL_ANY_JOIN:
		case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		case LogicalOperatorType::LOGICAL_DEPENDENT_JOIN:
		case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		case LogicalOperatorType::LOGICAL_UNION:
		case LogicalOperatorType::LOGICAL_EXCEPT:
		case LogicalOperatorType::LOGICAL_INTERSECT:
		case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
			break;
		default:
			eligible = false;
			break;
		}
		for (auto &child : op.children) {
			VisitOperator(*child);
		}
		VisitOperatorExpressions(op);
	}

	unique_ptr<Expression> VisitReplace(BoundFunctionExpression &expr, unique_ptr<Expression> *) override {
		has_context |= expr.function.CanCaptureClientContext();
		if (expr.function.HasModifiedDatabasesCallback() ||
		    (expr.function.RequiresClientContext() && !expr.function.CanCaptureClientContext()) ||
		    (expr.function.GetStability() == FunctionStability::VOLATILE && !expr.function.CanCaptureClientContext())) {
			eligible = false;
		}
		auto lambda = dynamic_cast<ListLambdaBindData *>(expr.bind_info.get());
		if (lambda && lambda->lambda_expr) {
			VisitExpression(&lambda->lambda_expr);
		}
		return nullptr;
	}
};

// This is only an early transaction exemption. Bound-plan admission remains
// authoritative. Unknown syntax/functions retain the existing pre-bind rejection.
class ClientContextSyntaxVisitor {
public:
	explicit ClientContextSyntaxVisitor(ClientContext &context_p) : context(context_p) {
	}
	bool eligible = true;
	bool has_context = false;

	void VisitQuery(QueryNode &node) {
		unordered_set<const ParsedExpression *> table_functions;
		ParsedExpressionIterator::EnumerateQueryNodeChildren(
		    node, [](unique_ptr<ParsedExpression> &) {},
		    [&](TableRef &ref) {
			    switch (ref.type) {
			    case TableReferenceType::TABLE_FUNCTION:
				    table_functions.insert(ref.Cast<TableFunctionRef>().function.get());
				    break;
			    case TableReferenceType::EMPTY_FROM:
			    case TableReferenceType::JOIN:
			    case TableReferenceType::SUBQUERY:
				    break;
			    default:
				    eligible = false;
			    }
		    });
		ParsedExpressionIterator::EnumerateQueryNodeChildren(
		    node,
		    [&](unique_ptr<ParsedExpression> &expr) {
			    VisitExpression(*expr, table_functions.find(expr.get()) != table_functions.end());
		    },
		    [](TableRef &) {});
	}

private:
	ClientContext &context;

	void VisitFunction(FunctionExpression &expr, bool table_function) {
		auto type = table_function ? CatalogType::TABLE_FUNCTION_ENTRY : CatalogType::SCALAR_FUNCTION_ENTRY;
		CatalogEntryRetriever retriever(context);
		auto lookup = Catalog::LookupEntry(retriever, expr.catalog, expr.schema,
		                                   EntryLookupInfo(type, expr.function_name), OnEntryNotFound::RETURN_NULL);
		auto entry = lookup.entry;
		if (!entry || !entry->internal) {
			eligible = false;
			return;
		}
		if (table_function && entry->type == CatalogType::TABLE_FUNCTION_ENTRY) {
			for (auto &function : entry->Cast<TableFunctionCatalogEntry>().functions.functions) {
				eligible &= function.IsClientContextRead() && !function.bind_replace && !function.bind_operator;
			}
			has_context = true;
		} else if (!table_function && entry->type == CatalogType::SCALAR_FUNCTION_ENTRY) {
			for (auto &function : entry->Cast<ScalarFunctionCatalogEntry>().functions.functions) {
				const bool has_bind_callback = function.HasBindCallback() || function.HasBindExtendedCallback() ||
				                               function.HasBindExpressionCallback() || function.HasBindLambdaCallback();
				has_context |= function.CanCaptureClientContext();
				eligible &=
				    !function.HasModifiedDatabasesCallback() &&
				    (!has_bind_callback || function.CanCaptureClientContext()) &&
				    (!function.RequiresClientContext() || function.CanCaptureClientContext()) &&
				    (function.GetStability() != FunctionStability::VOLATILE || function.CanCaptureClientContext());
			}
		} else if (!table_function && entry->type == CatalogType::AGGREGATE_FUNCTION_ENTRY) {
			// Binding callbacks can do more than combine child values. Aggregates
			// have no declared client-context binding capability.
			for (auto &function : entry->Cast<AggregateFunctionCatalogEntry>().functions.functions) {
				eligible &= !function.HasBindCallback();
			}
		} else {
			eligible = false;
		}
	}

	void VisitExpression(ParsedExpression &expr, bool table_function = false) {
		if (expr.GetExpressionClass() == ExpressionClass::FUNCTION) {
			VisitFunction(expr.Cast<FunctionExpression>(), table_function);
		} else if (expr.GetExpressionClass() == ExpressionClass::SUBQUERY) {
			VisitQuery(*expr.Cast<SubqueryExpression>().subquery->node);
		} else if (expr.GetExpressionClass() == ExpressionClass::COLUMN_REF) {
			auto &column = expr.Cast<ColumnRefExpression>();
			if (!column.IsQualified()) {
				auto name = GetSQLValueFunctionName(column.GetColumnName());
				if (!name.empty()) {
					FunctionExpression function(name, vector<unique_ptr<ParsedExpression>>());
					VisitFunction(function, false);
				}
			}
		} else if (expr.GetExpressionClass() == ExpressionClass::CAST ||
		           expr.GetExpressionClass() == ExpressionClass::TYPE ||
		           expr.GetExpressionClass() == ExpressionClass::WINDOW ||
		           expr.GetExpressionClass() == ExpressionClass::BOUND_EXPRESSION) {
			// Cast/type binding can load extensions or evaluate type parameters,
			// neither of which is represented by ordinary expression children.
			eligible = false;
		}
		ParsedExpressionIterator::EnumerateChildren(expr, [&](ParsedExpression &child) { VisitExpression(child); });
	}
};
} // namespace

bool IsClientContextQuery(LogicalOperator &plan, bool captured_client_context, bool allow_command_results) {
	ClientContextPlanVisitor visitor(allow_command_results);
	visitor.has_context = captured_client_context;
	visitor.VisitOperator(plan);
	return visitor.eligible && visitor.has_context;
}

bool IsClientContextQuery(ClientContext &context, QueryNode &query) {
	ClientContextSyntaxVisitor visitor(context);
	visitor.VisitQuery(query);
	return visitor.eligible && visitor.has_context;
}
} // namespace duckdb
