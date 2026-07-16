// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/main/relation/table_function_relation.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_macro_catalog_entry.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/parser/tableref/basetableref.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "duckdb/parser/query_error_context.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/expression/comparison_expression.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/common/shared_ptr.hpp"

namespace duckdb {

void TableFunctionRelation::AddNamedParameter(const string &name, Value argument) {
	named_parameters[name] = std::move(argument);
}

void TableFunctionRelation::RemoveNamedParameterIfExists(const string &name) {
	if (named_parameters.find(name) != named_parameters.end()) {
		named_parameters.erase(name);
	}
}

void TableFunctionRelation::SetNamedParameters(named_parameter_map_t &&options) {
	D_ASSERT(named_parameters.empty());
	named_parameters = std::move(options);
}

TableFunctionRelation::TableFunctionRelation(const shared_ptr<ClientContext> &context, string name_p,
                                             vector<Value> parameters_p, named_parameter_map_t named_parameters,
                                             shared_ptr<Relation> input_relation_p, bool auto_init)
    : Relation(context, RelationType::TABLE_FUNCTION_RELATION), name(std::move(name_p)),
      parameters(std::move(parameters_p)), named_parameters(std::move(named_parameters)),
      input_relation(std::move(input_relation_p)), auto_initialize(auto_init) {
	InitializeColumns();
}

TableFunctionRelation::TableFunctionRelation(const shared_ptr<RelationContextWrapper> &context, string name_p,
                                             vector<Value> parameters_p, named_parameter_map_t named_parameters,
                                             shared_ptr<Relation> input_relation_p, bool auto_init)
    : Relation(context, RelationType::TABLE_FUNCTION_RELATION), name(std::move(name_p)),
      parameters(std::move(parameters_p)), named_parameters(std::move(named_parameters)),
      input_relation(std::move(input_relation_p)), auto_initialize(auto_init) {
	InitializeColumns();
}

TableFunctionRelation::TableFunctionRelation(const shared_ptr<ClientContext> &context, string name_p,
                                             vector<Value> parameters_p, shared_ptr<Relation> input_relation_p,
                                             bool auto_init)
    : Relation(context, RelationType::TABLE_FUNCTION_RELATION), name(std::move(name_p)),
      parameters(std::move(parameters_p)), input_relation(std::move(input_relation_p)), auto_initialize(auto_init) {
	InitializeColumns();
}

void TableFunctionRelation::InitializeColumns() {
	if (!auto_initialize) {
		return;
	}
	TryBindRelation(columns);
}

unique_ptr<QueryNode> TableFunctionRelation::GetQueryNode() {
	auto result = make_uniq<SelectNode>();
	result->select_list.push_back(make_uniq<StarExpression>());
	result->from_table = GetTableRef();
	return std::move(result);
}

unique_ptr<TableRef> TableFunctionRelation::GetTableRef() {
	vector<unique_ptr<ParsedExpression>> children;
	if (input_relation) { // input relation becomes first parameter if present, always
		auto subquery = make_uniq<SubqueryExpression>();
		subquery->subquery = make_uniq<SelectStatement>();
		subquery->subquery->node = input_relation->GetQueryNode();
		subquery->subquery_type = SubqueryType::SCALAR;
		children.push_back(std::move(subquery));
	}
	for (auto &parameter : parameters) {
		children.push_back(make_uniq<ConstantExpression>(parameter));
	}

	for (auto &parameter : named_parameters) {
		// Hackity-hack some comparisons with column refs
		// This is all but pretty, basically the named parameter is the column, the table is empty because that's what
		// the function binder likes
		auto column_ref = make_uniq<ColumnRefExpression>(parameter.first);
		auto constant_value = make_uniq<ConstantExpression>(parameter.second);
		auto comparison = make_uniq<ComparisonExpression>(ExpressionType::COMPARE_EQUAL, std::move(column_ref),
		                                                  std::move(constant_value));
		children.push_back(std::move(comparison));
	}

	auto table_function = make_uniq<TableFunctionRef>();
	auto function = make_uniq<FunctionExpression>(name, std::move(children));
	table_function->function = std::move(function);
	return std::move(table_function);
}

BoundStatement TableFunctionRelation::Bind(Binder &binder) {
	if (!input_relation) {
		return Relation::Bind(binder);
	}

	auto qualified_name = QualifiedName::Parse(name);
	auto catalog = qualified_name.catalog;
	auto schema = qualified_name.schema;
	auto function_name = qualified_name.name;
	Binder::BindSchemaOrCatalog(binder.context, catalog, schema);

	QueryErrorContext error_context;
	EntryLookupInfo table_function_lookup(CatalogType::TABLE_FUNCTION_ENTRY, function_name, error_context);
	auto &func_catalog =
	    *binder.GetCatalogEntry(catalog, schema, table_function_lookup, OnEntryNotFound::THROW_EXCEPTION);
	if (func_catalog.type == CatalogType::TABLE_MACRO_ENTRY) {
		// Fall back to SQL binding for macros.
		return Relation::Bind(binder);
	}
	D_ASSERT(func_catalog.type == CatalogType::TABLE_FUNCTION_ENTRY);
	auto &function_entry = func_catalog.Cast<TableFunctionCatalogEntry>();

	auto child_binder = Binder::CreateBinder(binder.context, &binder);
	child_binder->SetCanContainNulls(true);
	auto child_bound = input_relation->Bind(*child_binder);
	binder.MoveCorrelatedExpressionsFrom(*child_binder);

	vector<LogicalType> arguments;
	vector<Value> bound_parameters;
	arguments.reserve(parameters.size() + 1);
	bound_parameters.reserve(parameters.size() + 1);
	arguments.emplace_back(LogicalTypeId::TABLE);
	bound_parameters.emplace_back(); // placeholder for table parameter

	for (auto &parameter : parameters) {
		arguments.emplace_back(parameter.type());
		bound_parameters.push_back(parameter);
	}

	auto bound_named_parameters = named_parameters;

	ErrorData error;
	FunctionBinder function_binder(binder);
	auto best_function_idx =
	    function_binder.BindFunction(function_entry.name, function_entry.functions, arguments, error);
	if (!best_function_idx.IsValid()) {
		error.Throw();
	}
	auto table_function = function_entry.functions.GetFunctionByOffset(best_function_idx.GetIndex());

	Binder::BindNamedParameters(table_function.named_parameters, bound_named_parameters, error_context,
	                            table_function.name);

	for (idx_t i = 0; i < arguments.size(); i++) {
		auto target_type = i < table_function.arguments.size() ? table_function.arguments[i] : table_function.varargs;
		if (target_type != LogicalType::ANY && target_type != LogicalType::POINTER &&
		    target_type.id() != LogicalTypeId::LIST && target_type.id() != LogicalTypeId::TABLE) {
			bound_parameters[i] = bound_parameters[i].CastAs(binder.context, target_type);
		}
	}

	TableFunctionRef ref;
	ref.alias = name;
	return binder.BindTableFunctionWithInput(table_function, ref, std::move(bound_parameters),
	                                         std::move(bound_named_parameters), child_bound);
}

string TableFunctionRelation::GetAlias() {
	return name;
}

const vector<ColumnDefinition> &TableFunctionRelation::Columns() {
	return columns;
}

string TableFunctionRelation::ToString(idx_t depth) {
	string function_call = name + "(";
	for (idx_t i = 0; i < parameters.size(); i++) {
		if (i > 0) {
			function_call += ", ";
		}
		function_call += parameters[i].ToString();
	}
	function_call += ")";
	return RenderWhitespace(depth) + function_call;
}

} // namespace duckdb
