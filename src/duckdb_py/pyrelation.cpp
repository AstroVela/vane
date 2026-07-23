// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#include "duckdb_python/pybind11/pybind_wrapper.hpp"
#include "duckdb_python/pyrelation.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/pytype.hpp"
#include "duckdb_python/pyresult.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb_python/numpy/numpy_type.hpp"
#include "duckdb/main/relation/query_relation.hpp"
#include "duckdb/main/relation/join_relation.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/main/relation/view_relation.hpp"
#include "duckdb/function/pragma/pragma_functions.hpp"
#include "duckdb/parser/statement/pragma_statement.hpp"
#include "duckdb/common/box_renderer.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/parser/statement/explain_statement.hpp"
#include "duckdb/catalog/default/default_types.hpp"
#include "duckdb/function/scalar/udf_functions.hpp"
#include "duckdb/main/relation/value_relation.hpp"
#include "duckdb/main/relation/filter_relation.hpp"
#include "duckdb/main/relation/unnest_relation.hpp"
#include "duckdb_python/expression/pyexpression.hpp"
#include "duckdb/common/arrow/physical_arrow_collector.hpp"
#include "duckdb_python/arrow/arrow_export_utils.hpp"
#include "duckdb_python/python_udf_utils.hpp"
#include "duckdb_python/python_udf_actor_resources.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb_python/pybind11/gil_wrapper.hpp"

#include <algorithm>

namespace duckdb {

DuckDBPyRelation::DuckDBPyRelation(shared_ptr<Relation> rel_p) : rel(std::move(rel_p)) {
	if (!rel) {
		throw InternalException("DuckDBPyRelation created without a relation");
	}
	this->executed = false;
	auto &columns = rel->Columns();
	for (auto &col : columns) {
		names.push_back(col.GetName());
		types.push_back(col.GetType());
	}
}

bool DuckDBPyRelation::CanBeRegisteredBy(Connection &con) {
	return CanBeRegisteredBy(con.context);
}

bool DuckDBPyRelation::CanBeRegisteredBy(ClientContext &context) {
	if (!rel) {
		// PyRelation without an internal relation can not be registered
		return false;
	}
	auto this_context = rel->context->TryGetContext();
	if (!this_context) {
		return false;
	}
	return &context == this_context.get();
}

bool DuckDBPyRelation::CanBeRegisteredBy(shared_ptr<ClientContext> &con) {
	if (!con) {
		return false;
	}
	return CanBeRegisteredBy(*con);
}

DuckDBPyRelation::~DuckDBPyRelation() {
	D_ASSERT(py::gil_check());
	py::gil_scoped_release gil;
	rel.reset();
}

shared_ptr<Relation> DuckDBPyRelation::GetRelation() {
	if (!rel) {
		throw InternalException("DuckDBPyRelation::GetRelation called on a relation-less DuckDBPyRelation");
	}
	return rel;
}

DuckDBPyRelation::DuckDBPyRelation(shared_ptr<DuckDBPyResult> result_p) : rel(nullptr), result(std::move(result_p)) {
	if (!result) {
		throw InternalException("DuckDBPyRelation created without a result");
	}
	this->executed = true;
	this->types = result->GetTypes();
	this->names = result->GetNames();
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::ProjectFromExpression(const string &expression) {
	auto projected_relation = DeriveRelation(rel->Project(expression));
	for (auto &dep : this->rel->external_dependencies) {
		projected_relation->rel->AddExternalDependency(dep);
	}
	return projected_relation;
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Project(const py::args &args, const string &groups) {
	if (!rel) {
		return nullptr;
	}
	auto arg_count = args.size();
	if (arg_count == 0) {
		return nullptr;
	}
	py::handle first_arg = args[0];
	if (arg_count == 1 && py::isinstance<py::str>(first_arg)) {
		string expr_string = py::str(first_arg);
		return ProjectFromExpression(expr_string);
	} else {
		vector<unique_ptr<ParsedExpression>> expressions;
		for (auto arg : args) {
			shared_ptr<DuckDBPyExpression> py_expr;
			if (!py::try_cast<shared_ptr<DuckDBPyExpression>>(arg, py_expr)) {
				throw InvalidInputException("Please provide arguments of type Expression!");
			}
			auto expr = py_expr->GetExpression().Copy();
			expressions.push_back(std::move(expr));
		}
		vector<string> empty_aliases;
		if (groups.empty()) {
			// No groups provided
			return DeriveRelation(rel->Project(std::move(expressions), empty_aliases));
		}
		return DeriveRelation(rel->Aggregate(std::move(expressions), groups));
	}
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::ProjectFromTypes(const py::object &obj) {
	if (!rel) {
		return nullptr;
	}
	if (!py::isinstance<py::list>(obj)) {
		throw InvalidInputException("'columns_by_type' expects a list containing types");
	}
	auto list = py::list(obj);
	vector<LogicalType> types_filter;
	// Collect the list of types specified that will be our filter
	for (auto &item : list) {
		LogicalType type;
		if (py::isinstance<py::str>(item)) {
			string type_str = py::str(item);
			rel->context->GetContext()->RunFunctionInTransaction(
			    [&]() { type = TransformStringToLogicalType(type_str, *rel->context->GetContext().get()); });
		} else if (py::isinstance<DuckDBPyType>(item)) {
			auto *type_p = item.cast<DuckDBPyType *>();
			type = type_p->Type();
		} else {
			string actual_type = py::str(py::type::of(item));
			throw InvalidInputException("Can only project on objects of type DuckDBPyType or str, not '%s'",
			                            actual_type);
		}
		types_filter.push_back(std::move(type));
	}

	if (types_filter.empty()) {
		throw InvalidInputException("List of types can not be empty!");
	}

	string projection = "";
	for (idx_t i = 0; i < types.size(); i++) {
		auto &type = types[i];
		// Check if any of the types in the filter match the current type
		if (std::find_if(types_filter.begin(), types_filter.end(),
		                 [&](const LogicalType &filter) { return filter == type; }) != types_filter.end()) {
			if (!projection.empty()) {
				projection += ", ";
			}
			projection += KeywordHelper::WriteOptionallyQuoted(names[i]);
		}
	}
	if (projection.empty()) {
		throw InvalidInputException("None of the columns matched the provided type filter!");
	}
	return ProjectFromExpression(projection);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::EmptyResult(const shared_ptr<ClientContext> &context,
                                                           const vector<LogicalType> &types, vector<string> names) {
	vector<Value> dummy_values;
	D_ASSERT(types.size() == names.size());
	dummy_values.reserve(types.size());
	D_ASSERT(!types.empty());
	for (auto &type : types) {
		dummy_values.emplace_back(type);
	}
	vector<vector<Value>> single_row(1, dummy_values);
	auto values_relation =
	    make_uniq<DuckDBPyRelation>(make_shared_ptr<ValueRelation>(context, single_row, std::move(names)));
	// Add a filter on an impossible condition
	return values_relation->FilterFromExpression("true = false");
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::SetAlias(const string &expr) {
	return DeriveRelation(rel->Alias(expr));
}

py::str DuckDBPyRelation::GetAlias() {
	return py::str(string(rel->GetAlias()));
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Filter(const py::object &expr) {
	if (py::isinstance<py::str>(expr)) {
		string expression = py::cast<py::str>(expr);
		return FilterFromExpression(expression);
	}
	shared_ptr<DuckDBPyExpression> expression;
	if (!py::try_cast(expr, expression)) {
		throw InvalidInputException("Please provide either a string or a DuckDBPyExpression object to 'filter'");
	}
	auto expr_p = expression->GetExpression().Copy();
	return DeriveRelation(rel->Filter(std::move(expr_p)));
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::FilterFromExpression(const string &expr) {
	return DeriveRelation(rel->Filter(expr));
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Limit(int64_t n, int64_t offset) {
	return DeriveRelation(rel->Limit(n, offset));
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Repartition(const py::args &args, const py::kwargs &kwargs) {
	if (!rel) {
		return nullptr;
	}
	std::pair<bool, idx_t> num_partitions = std::make_pair(false, idx_t(0));

	if (kwargs) {
		for (auto &item : kwargs) {
			auto key = py::str(item.first).cast<string>();
			if (key != "num_partitions") {
				throw InvalidInputException("Unknown keyword argument '%s' for repartition", key);
			}
		}
		if (kwargs.contains("num_partitions")) {
			auto value = kwargs["num_partitions"];
			if (!value.is_none()) {
				if (!py::isinstance<py::int_>(value)) {
					throw InvalidInputException("num_partitions must be an integer or None");
				}
				num_partitions = std::make_pair(true, value.cast<idx_t>());
			}
		}
	}

	idx_t arg_offset = 0;
	if (!kwargs.contains("num_partitions") && args.size() > 0 && py::isinstance<py::int_>(args[0])) {
		num_partitions = std::make_pair(true, args[0].cast<idx_t>());
		arg_offset = 1;
	}
	if (num_partitions.first && num_partitions.second <= 0) {
		throw InvalidInputException("num_partitions must be greater than zero");
	}

	vector<unique_ptr<ParsedExpression>> partition_by;
	for (idx_t i = arg_offset; i < args.size(); i++) {
		auto arg = args[i];
		if (py::isinstance<py::str>(arg)) {
			auto expr_string = string(py::str(arg));
			auto expressions = Parser::ParseExpressionList(expr_string, rel->context->GetContext()->GetParserOptions());
			if (expressions.size() != 1) {
				throw InvalidInputException("Expected a single expression for repartition");
			}
			partition_by.push_back(std::move(expressions[0]));
			continue;
		}
		shared_ptr<DuckDBPyExpression> py_expr;
		if (!py::try_cast<shared_ptr<DuckDBPyExpression>>(arg, py_expr)) {
			string actual_type = py::str(py::type::of(arg));
			throw InvalidInputException("Expected argument of type Expression or str, received '%s' instead",
			                            actual_type);
		}
		partition_by.push_back(py_expr->GetExpression().Copy());
	}

	return make_uniq<DuckDBPyRelation>(
	    rel->Repartition(num_partitions.first ? num_partitions.second : 0, std::move(partition_by)));
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::LocalExchange(const py::object &num_partitions_obj) {
	idx_t num_partitions = 0;
	if (!num_partitions_obj.is_none()) {
		num_partitions = num_partitions_obj.cast<idx_t>();
	}
	return make_uniq<DuckDBPyRelation>(rel->LocalExchange(num_partitions));
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Order(const string &expr) {
	return DeriveRelation(rel->Order(expr));
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Sort(const py::args &args) {
	vector<OrderByNode> order_nodes;
	order_nodes.reserve(args.size());

	for (auto arg : args) {
		shared_ptr<DuckDBPyExpression> py_expr;
		if (!py::try_cast<shared_ptr<DuckDBPyExpression>>(arg, py_expr)) {
			string actual_type = py::str(py::type::of(arg));
			throw InvalidInputException("Expected argument of type Expression, received '%s' instead", actual_type);
		}
		auto expr = py_expr->GetExpression().Copy();
		order_nodes.emplace_back(py_expr->order_type, py_expr->null_order, std::move(expr));
	}
	if (order_nodes.empty()) {
		throw InvalidInputException("Please provide at least one expression to sort on");
	}
	return DeriveRelation(rel->Order(std::move(order_nodes)));
}

vector<unique_ptr<ParsedExpression>> GetExpressions(ClientContext &context, const py::object &expr) {
	if (py::is_list_like(expr)) {
		vector<unique_ptr<ParsedExpression>> expressions;
		auto aggregate_list = py::list(expr);
		for (auto &item : aggregate_list) {
			shared_ptr<DuckDBPyExpression> py_expr;
			if (!py::try_cast<shared_ptr<DuckDBPyExpression>>(item, py_expr)) {
				throw InvalidInputException("Please provide arguments of type Expression!");
			}
			auto expr = py_expr->GetExpression().Copy();
			expressions.push_back(std::move(expr));
		}
		return expressions;
	} else if (py::isinstance<py::str>(expr)) {
		auto aggregate_list = std::string(py::str(expr));
		return Parser::ParseExpressionList(aggregate_list, context.GetParserOptions());
	} else {
		// A single Expression could be supported here by wrapping it in a vector
		string actual_type = py::str(py::type::of(expr));
		throw InvalidInputException("Please provide either a string or list of Expression objects, not %s",
		                            actual_type);
	}
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Aggregate(const py::object &expr, const string &groups) {
	AssertRelation();
	auto expressions = GetExpressions(*rel->context->GetContext(), expr);
	if (!groups.empty()) {
		return DeriveRelation(rel->Aggregate(std::move(expressions), groups));
	}
	return DeriveRelation(rel->Aggregate(std::move(expressions)));
}

void DuckDBPyRelation::AssertResult() const {
	if (!result) {
		throw InvalidInputException("No open result set");
	}
}

void DuckDBPyRelation::AssertRelation() const {
	if (!rel) {
		throw InvalidInputException("This relation was created from a result");
	}
}

void DuckDBPyRelation::AssertResultOpen() const {
	if (!result || result->IsClosed()) {
		throw InvalidInputException("No open result set");
	}
}

py::list DuckDBPyRelation::Description() {
	return DuckDBPyResult::GetDescription(names, types);
}

Relation &DuckDBPyRelation::GetRel() {
	if (!rel) {
		throw InternalException("DuckDBPyRelation - calling GetRel, but no rel was present");
	}
	return *rel;
}

struct DescribeAggregateInfo {
	explicit DescribeAggregateInfo(string name_p, bool numeric_only = false)
	    : name(std::move(name_p)), numeric_only(numeric_only) {
	}

	string name;
	bool numeric_only;
};

vector<string> CreateExpressionList(const vector<ColumnDefinition> &columns,
                                    const vector<DescribeAggregateInfo> &aggregates) {
	vector<string> expressions;
	expressions.reserve(columns.size());

	string aggr_names = "UNNEST([";
	for (idx_t i = 0; i < aggregates.size(); i++) {
		if (i > 0) {
			aggr_names += ", ";
		}
		aggr_names += "'";
		aggr_names += aggregates[i].name;
		aggr_names += "'";
	}
	aggr_names += "])";
	aggr_names += " AS aggr";
	expressions.push_back(aggr_names);
	for (idx_t c = 0; c < columns.size(); c++) {
		auto &col = columns[c];
		string expr = "UNNEST([";
		for (idx_t i = 0; i < aggregates.size(); i++) {
			if (i > 0) {
				expr += ", ";
			}
			if (aggregates[i].numeric_only && !col.GetType().IsNumeric()) {
				expr += "NULL";
				continue;
			}
			expr += aggregates[i].name;
			expr += "(";
			expr += KeywordHelper::WriteOptionallyQuoted(col.GetName());
			expr += ")";
			if (col.GetType().IsNumeric()) {
				expr += "::DOUBLE";
			} else {
				expr += "::VARCHAR";
			}
		}
		expr += "])";
		expr += " AS " + KeywordHelper::WriteOptionallyQuoted(col.GetName());
		expressions.push_back(expr);
	}
	return expressions;
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Describe() {
	auto &columns = rel->Columns();
	vector<DescribeAggregateInfo> aggregates;
	aggregates = {DescribeAggregateInfo("count"),        DescribeAggregateInfo("mean", true),
	              DescribeAggregateInfo("stddev", true), DescribeAggregateInfo("min"),
	              DescribeAggregateInfo("max"),          DescribeAggregateInfo("median", true)};
	auto expressions = CreateExpressionList(columns, aggregates);
	return DeriveRelation(rel->Aggregate(expressions));
}

string DuckDBPyRelation::ToSQL() {
	if (!rel) {
		// This relation is just a wrapper around a result set, can't figure out what the SQL was
		return "";
	}
	try {
		unique_ptr<QueryNode> query_node;
		{
			D_ASSERT(py::gil_check());
			py::gil_scoped_release release;
			query_node = rel->TryGetSerializableQueryNode();
		}
		if (!query_node) {
			// Some logical operators and binding scopes have no faithful SQL representation.
			return "";
		}
		return query_node->ToString();
	} catch (const std::exception &) {
		return "";
	}
}

string DuckDBPyRelation::GenerateExpressionList(const string &function_name, const string &aggregated_columns,
                                                const string &groups, const string &function_parameter,
                                                bool ignore_nulls, const string &projected_columns,
                                                const string &window_spec) {
	auto input = StringUtil::Split(aggregated_columns, ',');
	return GenerateExpressionList(function_name, std::move(input), groups, function_parameter, ignore_nulls,
	                              projected_columns, window_spec);
}

string DuckDBPyRelation::GenerateExpressionList(const string &function_name, vector<string> input, const string &groups,
                                                const string &function_parameter, bool ignore_nulls,
                                                const string &projected_columns, const string &window_spec) {
	string expr;

	if (StringUtil::CIEquals("count", function_name) && input.empty()) {
		// Insert an artificial '*'
		input.push_back("*");
	}

	if (!projected_columns.empty()) {
		expr = projected_columns + ", ";
	}

	if (input.empty() && !function_parameter.empty()) {
		return expr +=
		       function_name + "(" + function_parameter + ((ignore_nulls) ? " ignore nulls) " : ") ") + window_spec;
	}
	for (idx_t i = 0; i < input.size(); i++) {
		// We parse the input as an expression to validate it.
		auto trimmed_input = input[i];
		StringUtil::Trim(trimmed_input);

		unique_ptr<ParsedExpression> expression;
		try {
			auto expressions = Parser::ParseExpressionList(trimmed_input);
			if (expressions.size() == 1) {
				expression = std::move(expressions[0]);
			}
		} catch (const ParserException &) {
			// First attempt at parsing failed, the input might be a column name that needs quoting.
			auto quoted_input = KeywordHelper::WriteQuoted(trimmed_input, '"');
			auto expressions = Parser::ParseExpressionList(quoted_input);
			if (expressions.size() == 1 && expressions[0]->GetExpressionClass() == ExpressionClass::COLUMN_REF) {
				expression = std::move(expressions[0]);
			}
		}

		if (!expression) {
			throw ParserException("Invalid column expression: %s", trimmed_input);
		}

		// ToString() handles escaping for all expression types
		auto escaped_input = expression->ToString();

		if (function_parameter.empty()) {
			expr += function_name + "(" + escaped_input + ((ignore_nulls) ? " ignore nulls) " : ") ") + window_spec;
		} else {
			expr += function_name + "(" + escaped_input + "," + function_parameter +
			        ((ignore_nulls) ? " ignore nulls) " : ") ") + window_spec;
		}

		if (i < input.size() - 1) {
			expr += ",";
		}
	}
	return expr;
}

/* General aggregate functions */

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::GenericAggregator(const string &function_name,
                                                                 const string &aggregated_columns, const string &groups,
                                                                 const string &function_parameter,
                                                                 const string &projected_columns) {

	//! Construct Aggregation Expression
	auto expr = GenerateExpressionList(function_name, aggregated_columns, groups, function_parameter, false,
	                                   projected_columns, "");
	return Aggregate(py::str(expr), groups);
}

unique_ptr<DuckDBPyRelation>
DuckDBPyRelation::GenericWindowFunction(const string &function_name, const string &function_parameters,
                                        const string &aggr_columns, const string &window_spec, const bool &ignore_nulls,
                                        const string &projected_columns) {
	auto expr = GenerateExpressionList(function_name, aggr_columns, "", function_parameters, ignore_nulls,
	                                   projected_columns, window_spec);
	return DeriveRelation(rel->Project(expr));
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::ApplyAggOrWin(const string &function_name, const string &agg_columns,
                                                             const string &function_parameters, const string &groups,
                                                             const string &window_spec, const string &projected_columns,
                                                             bool ignore_nulls) {
	if (!groups.empty() && !window_spec.empty()) {
		throw InvalidInputException("Either groups or window must be set (can't be both at the same time)");
	}
	if (!window_spec.empty()) {
		return GenericWindowFunction(function_name, function_parameters, agg_columns, window_spec, ignore_nulls,
		                             projected_columns);
	} else {
		return GenericAggregator(function_name, agg_columns, groups, function_parameters, projected_columns);
	}
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::AnyValue(const std::string &column, const std::string &groups,
                                                        const std::string &window_spec,
                                                        const std::string &projected_columns) {
	return ApplyAggOrWin("any_value", column, "", groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::ArgMax(const std::string &arg_column, const std::string &value_column,
                                                      const std::string &groups, const std::string &window_spec,
                                                      const std::string &projected_columns) {
	return ApplyAggOrWin("arg_max", arg_column, value_column, groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::ArgMin(const std::string &arg_column, const std::string &value_column,
                                                      const std::string &groups, const std::string &window_spec,
                                                      const std::string &projected_columns) {
	return ApplyAggOrWin("arg_min", arg_column, value_column, groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Avg(const std::string &column, const std::string &groups,
                                                   const std::string &window_spec,
                                                   const std::string &projected_columns) {
	return ApplyAggOrWin("avg", column, "", groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::BitAnd(const std::string &column, const std::string &groups,
                                                      const std::string &window_spec,
                                                      const std::string &projected_columns) {
	return ApplyAggOrWin("bit_and", column, "", groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::BitOr(const std::string &column, const std::string &groups,
                                                     const std::string &window_spec,
                                                     const std::string &projected_columns) {
	return ApplyAggOrWin("bit_or", column, "", groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::BitXor(const std::string &column, const std::string &groups,
                                                      const std::string &window_spec,
                                                      const std::string &projected_columns) {
	return ApplyAggOrWin("bit_xor", column, "", groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::BitStringAgg(const std::string &column, const Optional<py::object> &min,
                                                            const Optional<py::object> &max, const std::string &groups,
                                                            const std::string &window_spec,
                                                            const std::string &projected_columns) {
	if ((min.is_none() && !max.is_none()) || (!min.is_none() && max.is_none())) {
		throw InvalidInputException("Both min and max values must be set");
	}
	if (!min.is_none()) {
		if (!py::isinstance<py::int_>(min) || !py::isinstance<py::int_>(max)) {
			throw InvalidTypeException("min and max must be of type int");
		}
	}
	auto bitstring_agg_params =
	    min.is_none() ? "" : (std::to_string(min.cast<int>()) + "," + std::to_string(max.cast<int>()));
	return ApplyAggOrWin("bitstring_agg", column, bitstring_agg_params, groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::BoolAnd(const std::string &column, const std::string &groups,
                                                       const std::string &window_spec,
                                                       const std::string &projected_columns) {
	return ApplyAggOrWin("bool_and", column, "", groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::BoolOr(const std::string &column, const std::string &groups,
                                                      const std::string &window_spec,
                                                      const std::string &projected_columns) {
	return ApplyAggOrWin("bool_or", column, "", groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::ValueCounts(const std::string &column, const std::string &groups) {
	return Count(column, groups, "", column);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Count(const std::string &column, const std::string &groups,
                                                     const std::string &window_spec,
                                                     const std::string &projected_columns) {
	return ApplyAggOrWin("count", column, "", groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::FAvg(const std::string &column, const std::string &groups,
                                                    const std::string &window_spec,
                                                    const std::string &projected_columns) {
	return ApplyAggOrWin("favg", column, "", groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::First(const string &column, const std::string &groups,
                                                     const string &projected_columns) {
	return GenericAggregator("first", column, groups, "", projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::FSum(const std::string &column, const std::string &groups,
                                                    const std::string &window_spec,
                                                    const std::string &projected_columns) {
	return ApplyAggOrWin("fsum", column, "", groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::GeoMean(const std::string &column, const std::string &groups,
                                                       const std::string &projected_columns) {
	return GenericAggregator("geomean", column, groups, "", projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Histogram(const std::string &column, const std::string &groups,
                                                         const std::string &window_spec,
                                                         const std::string &projected_columns) {
	return ApplyAggOrWin("histogram", column, "", groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::List(const std::string &column, const std::string &groups,
                                                    const std::string &window_spec,
                                                    const std::string &projected_columns) {
	return ApplyAggOrWin("list", column, "", groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Last(const std::string &column, const std::string &groups,
                                                    const std::string &projected_columns) {
	return GenericAggregator("last", column, groups, "", projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Max(const std::string &column, const std::string &groups,
                                                   const std::string &window_spec,
                                                   const std::string &projected_columns) {
	return ApplyAggOrWin("max", column, "", groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Min(const std::string &column, const std::string &groups,
                                                   const std::string &window_spec,
                                                   const std::string &projected_columns) {
	return ApplyAggOrWin("min", column, "", groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Product(const std::string &column, const std::string &groups,
                                                       const std::string &window_spec,
                                                       const std::string &projected_columns) {
	return ApplyAggOrWin("product", column, "", groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::StringAgg(const std::string &column, const std::string &sep,
                                                         const std::string &groups, const std::string &window_spec,
                                                         const std::string &projected_columns) {
	auto string_agg_params = KeywordHelper::WriteOptionallyQuoted(sep, '\'');
	return ApplyAggOrWin("string_agg", column, string_agg_params, groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Sum(const std::string &column, const std::string &groups,
                                                   const std::string &window_spec,
                                                   const std::string &projected_columns) {
	return ApplyAggOrWin("sum", column, "", groups, window_spec, projected_columns);
}

/* TODO: Approximate aggregate functions */

/* TODO: Statistical aggregate functions */
unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Median(const std::string &column, const std::string &groups,
                                                      const std::string &window_spec,
                                                      const std::string &projected_columns) {
	return ApplyAggOrWin("median", column, "", groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Mode(const std::string &column, const std::string &groups,
                                                    const std::string &window_spec,
                                                    const std::string &projected_columns) {
	return ApplyAggOrWin("mode", column, "", groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::QuantileCont(const std::string &column, const py::object &q,
                                                            const std::string &groups, const std::string &window_spec,
                                                            const std::string &projected_columns) {
	string quantile_params = "";
	if (py::isinstance<py::float_>(q)) {
		quantile_params = std::to_string(q.cast<float>());
	} else if (py::isinstance<py::list>(q)) {
		auto aux = q.cast<std::vector<double>>();
		quantile_params += "[";
		for (idx_t i = 0; i < aux.size(); i++) {
			quantile_params += std::to_string(aux[i]);
			if (i < aux.size() - 1) {
				quantile_params += ",";
			}
		}
		quantile_params += "]";
	} else {
		throw InvalidTypeException("Unsupported type for quantile");
	}
	return ApplyAggOrWin("quantile_cont", column, quantile_params, groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::QuantileDisc(const std::string &column, const py::object &q,
                                                            const std::string &groups, const std::string &window_spec,
                                                            const std::string &projected_columns) {
	string quantile_params = "";
	if (py::isinstance<py::float_>(q)) {
		quantile_params = std::to_string(q.cast<float>());
	} else if (py::isinstance<py::list>(q)) {
		auto aux = q.cast<std::vector<double>>();
		quantile_params += "[";
		for (idx_t i = 0; i < aux.size(); i++) {
			quantile_params += std::to_string(aux[i]);
			if (i < aux.size() - 1) {
				quantile_params += ",";
			}
		}
		quantile_params += "]";
	} else {
		throw InvalidTypeException("Unsupported type for quantile");
	}
	return ApplyAggOrWin("quantile_disc", column, quantile_params, groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::StdPop(const std::string &column, const std::string &groups,
                                                      const std::string &window_spec,
                                                      const std::string &projected_columns) {
	return ApplyAggOrWin("stddev_pop", column, "", groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::StdSamp(const std::string &column, const std::string &groups,
                                                       const std::string &window_spec,
                                                       const std::string &projected_columns) {
	return ApplyAggOrWin("stddev_samp", column, "", groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::VarPop(const std::string &column, const std::string &groups,
                                                      const std::string &window_spec,
                                                      const std::string &projected_columns) {
	return ApplyAggOrWin("var_pop", column, "", groups, window_spec, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::VarSamp(const std::string &column, const std::string &groups,
                                                       const std::string &window_spec,
                                                       const std::string &projected_columns) {
	return ApplyAggOrWin("var_samp", column, "", groups, window_spec, projected_columns);
}

idx_t DuckDBPyRelation::Length() {
	auto aggregate_rel = GenericAggregator("count", "*");
	aggregate_rel->Execute();
	D_ASSERT(aggregate_rel->result);
	auto tmp_res = std::move(aggregate_rel->result);
	return tmp_res->FetchChunk()->GetValue(0, 0).GetValue<idx_t>();
}

py::tuple DuckDBPyRelation::Shape() {
	auto length = Length();
	return py::make_tuple(length, rel->Columns().size());
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Unique(const string &std_columns) {
	return DeriveRelation(rel->Project(std_columns)->Distinct());
}

/* General-purpose window functions */

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::RowNumber(const string &window_spec, const string &projected_columns) {
	return GenericWindowFunction("row_number", "", "*", window_spec, false, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Rank(const string &window_spec, const string &projected_columns) {
	return GenericWindowFunction("rank", "", "*", window_spec, false, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::DenseRank(const string &window_spec, const string &projected_columns) {
	return GenericWindowFunction("dense_rank", "", "*", window_spec, false, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::PercentRank(const string &window_spec, const string &projected_columns) {
	return GenericWindowFunction("percent_rank", "", "*", window_spec, false, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::CumeDist(const string &window_spec, const string &projected_columns) {
	return GenericWindowFunction("cume_dist", "", "*", window_spec, false, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::FirstValue(const string &column, const string &window_spec,
                                                          const string &projected_columns) {
	return GenericWindowFunction("first_value", "", column, window_spec, false, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::NTile(const string &window_spec, const int &num_buckets,
                                                     const string &projected_columns) {
	return GenericWindowFunction("ntile", std::to_string(num_buckets), "", window_spec, false, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Lag(const string &column, const string &window_spec, const int &offset,
                                                   const string &default_value, const bool &ignore_nulls,
                                                   const string &projected_columns) {
	string lag_params = "";
	if (offset != 0) {
		lag_params += std::to_string(offset);
	}
	if (!default_value.empty()) {
		lag_params += "," + default_value;
	}
	return GenericWindowFunction("lag", lag_params, column, window_spec, ignore_nulls, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::LastValue(const std::string &column, const std::string &window_spec,
                                                         const std::string &projected_columns) {
	return GenericWindowFunction("last_value", "", column, window_spec, false, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Lead(const string &column, const string &window_spec, const int &offset,
                                                    const string &default_value, const bool &ignore_nulls,
                                                    const string &projected_columns) {
	string lead_params = "";
	if (offset != 0) {
		lead_params += std::to_string(offset);
	}
	if (!default_value.empty()) {
		lead_params += "," + default_value;
	}
	return GenericWindowFunction("lead", lead_params, column, window_spec, ignore_nulls, projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::NthValue(const string &column, const string &window_spec,
                                                        const int &offset, const bool &ignore_nulls,
                                                        const string &projected_columns) {
	return GenericWindowFunction("nth_value", std::to_string(offset), column, window_spec, ignore_nulls,
	                             projected_columns);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Distinct() {
	return DeriveRelation(rel->Distinct());
}

duckdb::pyarrow::RecordBatchReader DuckDBPyRelation::FetchRecordBatchReader(idx_t rows_per_batch) {
	AssertResult();
	return result->FetchRecordBatchReader(rows_per_batch);
}

// VANE_RUNNER is the single runner-selection surface. Empty or unset defaults to Ray.
static string ResolveRunnerType() {
	return ResolveRunnerTypeFromEnvironment();
}

// ── Per-database runner instances ──────────────────────────────────────
// Each DatabaseInstance gets its own runner (created lazily).
// Replaces the global VANE_RUNNER_PTR singleton for write dispatch.
static std::mutex per_db_runner_mutex;

static bool PyRelationRuntimeUsableForTeardown() {
	if (!Py_IsInitialized()) {
		return false;
	}
	if (PythonIsFinalizing()) {
		return false;
	}
	return true;
}

struct SafeRelationPyObject {
	bool has_value = false;
	py::object obj;

	SafeRelationPyObject() = default;
	explicit SafeRelationPyObject(py::object value) : has_value(true), obj(std::move(value)) {
	}

	SafeRelationPyObject(const SafeRelationPyObject &other) : has_value(false), obj() {
		if (other.has_value && other.obj.ptr() && PyRelationRuntimeUsableForTeardown()) {
			PythonGILWrapper gil;
			obj = py::reinterpret_borrow<py::object>(other.obj);
			has_value = true;
		}
	}

	SafeRelationPyObject &operator=(const SafeRelationPyObject &other) {
		if (this == &other) {
			return *this;
		}
		Reset();
		if (other.has_value && other.obj.ptr() && PyRelationRuntimeUsableForTeardown()) {
			PythonGILWrapper gil;
			obj = py::reinterpret_borrow<py::object>(other.obj);
			has_value = true;
		}
		return *this;
	}

	SafeRelationPyObject(SafeRelationPyObject &&other) noexcept
	    : has_value(other.has_value), obj(std::move(other.obj)) {
		other.has_value = false;
	}

	SafeRelationPyObject &operator=(SafeRelationPyObject &&other) noexcept {
		if (this == &other) {
			return *this;
		}
		Reset();
		has_value = other.has_value;
		obj = std::move(other.obj);
		other.has_value = false;
		return *this;
	}

	~SafeRelationPyObject() {
		Reset();
	}

	void Reset() {
		if (!obj.ptr()) {
			has_value = false;
			return;
		}
		if (!PyRelationRuntimeUsableForTeardown()) {
			obj.release();
			has_value = false;
			return;
		}
		PythonGILWrapper gil;
		auto *ptr = obj.release().ptr();
		Py_DECREF(ptr);
		has_value = false;
	}

	py::object Get() const {
		if (!has_value || !obj.ptr()) {
			return py::none();
		}
		return obj;
	}
};

static std::unordered_map<DatabaseInstance *, SafeRelationPyObject> per_db_runners;

static DatabaseInstance *GetRelationDatabasePtr(const shared_ptr<Relation> &rel) {
	if (!rel || !rel->context) {
		return nullptr;
	}
	auto context = rel->context->GetContext();
	return context->db.get();
}

static void ForgetRunnerForDB(const shared_ptr<Relation> &rel) {
	auto *db_ptr = GetRelationDatabasePtr(rel);
	if (!db_ptr) {
		return;
	}
	std::lock_guard<std::mutex> guard(per_db_runner_mutex);
	per_db_runners.erase(db_ptr);
}

static py::object GetOrCreateRunnerForDB(const shared_ptr<Relation> &rel, const string &runner_type) {
	if (!rel || !rel->context) {
		throw InternalException("Cannot resolve runner: relation has no context");
	}
	auto *db_ptr = GetRelationDatabasePtr(rel);
	if (!db_ptr) {
		throw InternalException("Cannot resolve runner: relation has no database");
	}

	{
		std::lock_guard<std::mutex> guard(per_db_runner_mutex);
		auto it = per_db_runners.find(db_ptr);
		if (it != per_db_runners.end()) {
			auto runner = it->second.Get();
			if (!runner.is_none()) {
				return runner;
			}
		}
	}

	// Create runner outside the lock (Python calls may be slow)
	auto runners_mod = py::module::import("duckdb.runners");
	py::object runner;
	if (runner_type == "ray") {
		// noop_if_initialized=true: reuse existing Ray runner if one was
		// already created by another database instance in this process.
		auto set_fn = runners_mod.attr("set_runner_ray");
		runner = set_fn(py::none(), /*noop_if_initialized=*/true);
	} else if (runner_type == "local") {
		auto set_fn = runners_mod.attr("set_runner_local");
		runner = set_fn();
	} else {
		runner = runners_mod.attr("get_or_create_runner")();
	}

	{
		std::lock_guard<std::mutex> guard(per_db_runner_mutex);
		per_db_runners[db_ptr] = SafeRelationPyObject(runner);
	}
	return runner;
}

// Try to dispatch a write relation to the Python runner.
// Returns true if dispatched, false if this relation should run locally.
static bool TryDispatchToRunner(const shared_ptr<Relation> &write_rel, const py::object &connection_owner) {
	auto runner_type = ResolveRunnerType();
	if (runner_type == "local-fast") {
		return false;
	}
	try {
		auto runner = GetOrCreateRunnerForDB(write_rel, runner_type);
		auto py_write_rel = DuckDBPyRelation(write_rel);
		py_write_rel.SetConnectionOwner(connection_owner);
		auto py_write_rel_obj = py::cast(std::move(py_write_rel));
		runner.attr("run_write")(py_write_rel_obj);
		ForgetRunnerForDB(write_rel);
		return true;
	} catch (py::error_already_set &) {
		throw;
	}
}

static unique_ptr<QueryResult> PyExecuteRelation(const shared_ptr<Relation> &rel, bool stream_result = false) {
	if (!rel) {
		return nullptr;
	}
	auto context = rel->context->GetContext();
	D_ASSERT(py::gil_check());
	ScopedPythonUDFActorResourcePreparation udf_actor_resources(*context);
	py::gil_scoped_release release;
	auto pending_query = context->PendingQuery(rel, stream_result);
	return DuckDBPyConnection::CompletePendingQuery(*pending_query);
}

unique_ptr<QueryResult> DuckDBPyRelation::ExecuteInternal(bool stream_result) {
	this->executed = true;
	return PyExecuteRelation(rel, stream_result);
}

void DuckDBPyRelation::ExecuteOrThrow(bool stream_result) {
	auto query_result = ExecuteInternal(stream_result);
	if (!query_result) {
		throw InternalException("ExecuteOrThrow - no query available to execute");
	}
	if (query_result->HasError()) {
		query_result->ThrowError();
	}
	result = make_uniq<DuckDBPyResult>(std::move(query_result));
}

PandasDataFrame DuckDBPyRelation::FetchDF(bool date_as_object) {
	if (!result) {
		if (!rel) {
			return py::none();
		}
		ExecuteOrThrow();
	}
	if (result->IsClosed()) {
		return py::none();
	}
	auto df = result->FetchDF(date_as_object);
	result = nullptr;
	return df;
}

Optional<py::tuple> DuckDBPyRelation::FetchOne() {
	if (!result) {
		if (!rel) {
			return py::none();
		}
		ExecuteOrThrow(true);
	}
	if (result->IsClosed()) {
		return py::none();
	}
	return result->Fetchone();
}

py::list DuckDBPyRelation::FetchMany(idx_t size) {
	if (!result) {
		if (!rel) {
			return py::list();
		}
		ExecuteOrThrow(true);
		D_ASSERT(result);
	}
	if (result->IsClosed()) {
		return py::list();
	}
	return result->Fetchmany(size);
}

py::list DuckDBPyRelation::FetchAll() {
	if (!result) {
		if (!rel) {
			return py::list();
		}
		ExecuteOrThrow();
	}
	if (result->IsClosed()) {
		return py::list();
	}
	auto res = result->Fetchall();
	result = nullptr;
	return res;
}

py::dict DuckDBPyRelation::FetchNumpy() {
	if (!result) {
		if (!rel) {
			return py::none();
		}
		ExecuteOrThrow();
	}
	if (result->IsClosed()) {
		return py::none();
	}
	auto res = result->FetchNumpy();
	result = nullptr;
	return res;
}

py::dict DuckDBPyRelation::FetchPyTorch() {
	if (!result) {
		if (!rel) {
			return py::none();
		}
		ExecuteOrThrow();
	}
	if (result->IsClosed()) {
		return py::none();
	}
	auto res = result->FetchPyTorch();
	result = nullptr;
	return res;
}

py::dict DuckDBPyRelation::FetchTF() {
	if (!result) {
		if (!rel) {
			return py::none();
		}
		ExecuteOrThrow();
	}
	if (result->IsClosed()) {
		return py::none();
	}
	auto res = result->FetchTF();
	result = nullptr;
	return res;
}

py::dict DuckDBPyRelation::FetchNumpyInternal(bool stream, idx_t vectors_per_chunk) {
	if (!result) {
		if (!rel) {
			return py::none();
		}
		ExecuteOrThrow();
	}
	AssertResultOpen();
	auto res = result->FetchNumpyInternal(stream, vectors_per_chunk);
	result = nullptr;
	return res;
}

//! Should this also keep track of when the result is empty and set result->result_closed accordingly?
PandasDataFrame DuckDBPyRelation::FetchDFChunk(idx_t vectors_per_chunk, bool date_as_object) {
	if (!result) {
		if (!rel) {
			return py::none();
		}
		ExecuteOrThrow(true);
	}
	AssertResultOpen();
	return result->FetchDFChunk(vectors_per_chunk, date_as_object);
}

duckdb::pyarrow::Table DuckDBPyRelation::ToArrowTableInternal(idx_t batch_size, bool to_polars) {
	if (!result) {
		if (!rel) {
			return py::none();
		}
		auto &config = ClientConfig::GetConfig(*rel->context->GetContext());
		ScopedConfigSetting scoped_setting(
		    config,
		    [&batch_size](ClientConfig &config) {
			    config.get_result_collector = [&batch_size](ClientContext &context,
			                                                PreparedStatementData &data) -> PhysicalOperator & {
				    return PhysicalArrowCollector::Create(context, data, batch_size);
			    };
		    },
		    [](ClientConfig &config) { config.get_result_collector = nullptr; });
		ExecuteOrThrow();
	}
	AssertResultOpen();
	auto res = result->FetchArrowTable(batch_size, to_polars);
	result = nullptr;
	return res;
}

duckdb::pyarrow::Table DuckDBPyRelation::ToArrowTable(idx_t batch_size) {
	return ToArrowTableInternal(batch_size, false);
}

py::object DuckDBPyRelation::ToArrowCapsule(const py::object &requested_schema) {
	if (!result) {
		if (!rel) {
			return py::none();
		}
		// The PyCapsule protocol doesn't allow custom parameters, so we use the same
		// default batch size as fetch_arrow_table / fetch_record_batch.
		idx_t batch_size = 1000000;
		auto &config = ClientConfig::GetConfig(*rel->context->GetContext());
		ScopedConfigSetting scoped_setting(
		    config,
		    [&batch_size](ClientConfig &config) {
			    config.get_result_collector = [&batch_size](ClientContext &context,
			                                                PreparedStatementData &data) -> PhysicalOperator & {
				    return PhysicalArrowCollector::Create(context, data, batch_size);
			    };
		    },
		    [](ClientConfig &config) { config.get_result_collector = nullptr; });
		ExecuteOrThrow();
	}
	AssertResultOpen();
	auto res = result->FetchArrowCapsule();
	result = nullptr;
	return res;
}

PolarsDataFrame DuckDBPyRelation::ToPolars(idx_t batch_size, bool lazy) {
	if (!lazy) {
		auto arrow = ToArrowTableInternal(batch_size, true);
		return py::cast<PolarsDataFrame>(
		    pybind11::module_::import("polars").attr("from_arrow")(arrow, py::arg("rechunk") = false));
	}
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	auto lazy_frame_produce = import_cache.duckdb.polars_io.duckdb_source();
	//  We also have to get a polars schema here, for this we can get at empty arrow table
	// We start by extracting the arrow schema
	ArrowSchema arrow_schema;
	auto result_names = names;
	QueryResult::DeduplicateColumns(result_names);
	ClientProperties client_properties;
	if (rel) {
		client_properties = rel->context->GetContext()->GetClientProperties();
	} else if (result) {
		client_properties = result->GetClientProperties();
	} else {
		throw InternalException("DuckDBPyRelation To Polars must have a valid relation or result");
	}
	ArrowConverter::ToArrowSchema(&arrow_schema, types, result_names, client_properties);
	py::list batches;
	// Now we create an empty arrow table
	auto empty_table = pyarrow::ToArrowTable(types, result_names, std::move(batches), client_properties);

	// And we extract the polars schema from the arrow table
	auto polars_df = py::cast<PolarsDataFrame>(pybind11::module_::import("polars").attr("DataFrame")(empty_table));
	auto polars_schema = polars_df.attr("schema");

	return lazy_frame_produce(*this, polars_schema);
}

duckdb::pyarrow::RecordBatchReader DuckDBPyRelation::ToRecordBatch(idx_t batch_size) {
	if (!result) {
		if (!rel) {
			return py::none();
		}
		ExecuteOrThrow(true);
	}
	AssertResultOpen();
	auto res = result->FetchRecordBatchReader(batch_size);
	result = nullptr;
	return res;
}

void DuckDBPyRelation::Close() {
	// We always want to execute the query at least once, for side-effect purposes.
	// if it has already been executed, we don't need to do it again.
	if (!executed && !result) {
		if (!rel) {
			return;
		}
		ExecuteOrThrow();
	}
	if (result) {
		result->Close();
	}
}

bool DuckDBPyRelation::ContainsColumnByName(const string &name) const {
	return std::find_if(names.begin(), names.end(),
	                    [&](const string &item) { return StringUtil::CIEquals(name, item); }) != names.end();
}

void DuckDBPyRelation::SetConnectionOwner(py::object owner) {
	connection_owner = std::move(owner);
}

py::object DuckDBPyRelation::GetConnectionOwner() const {
	return connection_owner;
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::DeriveRelation(shared_ptr<Relation> new_rel) {
	auto result = make_uniq<DuckDBPyRelation>(std::move(new_rel));
	result->connection_owner = connection_owner;
	return result;
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::DeriveRelation(shared_ptr<DuckDBPyResult> result_p) {
	auto result = make_uniq<DuckDBPyRelation>(std::move(result_p));
	result->connection_owner = connection_owner;
	return result;
}

static bool ContainsStructFieldByName(const LogicalType &type, const string &name) {
	if (type.id() != LogicalTypeId::STRUCT) {
		return false;
	}
	auto count = StructType::GetChildCount(type);
	for (idx_t i = 0; i < count; i++) {
		auto &field_name = StructType::GetChildName(type, i);
		if (StringUtil::CIEquals(name, field_name)) {
			return true;
		}
	}
	return false;
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::GetAttribute(const string &name) {
	// TODO: support fetching a result containing only column 'name' from a value_relation
	if (!rel) {
		throw py::attribute_error(
		    StringUtil::Format("This relation does not contain a column by the name of '%s'", name));
	}
	vector<string> column_names;
	if (names.size() == 1 && ContainsStructFieldByName(types[0], name)) {
		// e.g 'rel['my_struct']['my_field']:
		// first 'my_struct' is selected by the bottom condition
		// then 'my_field' is accessed on the result of this
		column_names.push_back(names[0]);
		column_names.push_back(name);
	} else if (ContainsColumnByName(name)) {
		column_names.push_back(name);
	}

	if (column_names.empty()) {
		throw py::attribute_error(
		    StringUtil::Format("This relation does not contain a column by the name of '%s'", name));
	}

	vector<unique_ptr<ParsedExpression>> expressions;
	expressions.push_back(std::move(make_uniq<ColumnRefExpression>(column_names)));
	vector<string> aliases;
	aliases.push_back(name);
	return DeriveRelation(rel->Project(std::move(expressions), aliases));
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Union(DuckDBPyRelation *other) {
	return DeriveRelation(rel->Union(other->rel));
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Except(DuckDBPyRelation *other) {
	return DeriveRelation(rel->Except(other->rel));
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Intersect(DuckDBPyRelation *other) {
	return DeriveRelation(rel->Intersect(other->rel));
}

namespace {
struct SupportedPythonJoinType {
	string name;
	JoinType type;
};
} // namespace

static const SupportedPythonJoinType *GetSupportedJoinTypes(idx_t &length) {
	static const SupportedPythonJoinType SUPPORTED_TYPES[] = {{"left", JoinType::LEFT},   {"right", JoinType::RIGHT},
	                                                          {"outer", JoinType::OUTER}, {"semi", JoinType::SEMI},
	                                                          {"inner", JoinType::INNER}, {"anti", JoinType::ANTI}};
	static const auto SUPPORTED_TYPES_COUNT = sizeof(SUPPORTED_TYPES) / sizeof(SupportedPythonJoinType);
	length = SUPPORTED_TYPES_COUNT;
	return reinterpret_cast<const SupportedPythonJoinType *>(SUPPORTED_TYPES);
}

static JoinType ParseJoinType(const string &type) {
	idx_t supported_types_count;
	auto supported_types = GetSupportedJoinTypes(supported_types_count);
	for (idx_t i = 0; i < supported_types_count; i++) {
		auto &supported_type = supported_types[i];
		if (supported_type.name == type) {
			return supported_type.type;
		}
	}
	return JoinType::INVALID;
}

[[noreturn]] void ThrowUnsupportedJoinTypeError(const string &provided) {
	vector<string> supported_options;
	idx_t length;
	auto supported_types = GetSupportedJoinTypes(length);
	for (idx_t i = 0; i < length; i++) {
		supported_options.push_back(StringUtil::Format("'%s'", supported_types[i].name));
	}
	auto options = StringUtil::Join(supported_options, ", ");
	throw InvalidInputException("Unsupported join type %s, try one of: %s", provided, options);
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Join(DuckDBPyRelation *other, const py::object &condition,
                                                    const string &type) {

	JoinType join_type;
	string type_string = StringUtil::Lower(type);
	StringUtil::Trim(type_string);

	join_type = ParseJoinType(type_string);
	if (join_type == JoinType::INVALID) {
		ThrowUnsupportedJoinTypeError(type);
	}
	auto alias = GetAlias();
	auto other_alias = other->GetAlias();
	if (StringUtil::CIEquals(alias, other_alias)) {
		throw InvalidInputException("Both relations have the same alias, please change the alias of one or both "
		                            "relations using 'rel = rel.set_alias(<new alias>)'");
	}
	if (py::isinstance<py::str>(condition)) {
		auto condition_string = std::string(py::cast<py::str>(condition));
		return DeriveRelation(rel->Join(other->rel, condition_string, join_type));
	}
	vector<string> using_list;
	if (py::is_list_like(condition)) {
		auto using_list_p = py::list(condition);
		for (auto &item : using_list_p) {
			if (!py::isinstance<py::str>(item)) {
				string actual_type = py::str(py::type::of(item));
				throw InvalidInputException("Using clause should be a list of strings, not %s", actual_type);
			}
			using_list.push_back(std::string(py::str(item)));
		}
		if (using_list.empty()) {
			throw InvalidInputException("Please provide at least one string in the condition to create a USING clause");
		}
		auto join_relation = make_shared_ptr<JoinRelation>(rel, other->rel, std::move(using_list), join_type);
		return DeriveRelation(std::move(join_relation));
	}
	shared_ptr<DuckDBPyExpression> condition_expr;
	if (!py::try_cast(condition, condition_expr)) {
		throw InvalidInputException(
		    "Please provide condition as an expression either in string form or as an Expression object");
	}
	vector<unique_ptr<ParsedExpression>> conditions;
	conditions.push_back(condition_expr->GetExpression().Copy());
	return DeriveRelation(rel->Join(other->rel, std::move(conditions), join_type));
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Cross(DuckDBPyRelation *other) {
	return DeriveRelation(rel->CrossProduct(other->rel));
}

static Value NestedDictToStruct(const py::object &dictionary) {
	if (!py::isinstance<py::dict>(dictionary)) {
		throw InvalidInputException("NestedDictToStruct only accepts a dictionary as input");
	}
	py::dict dict_casted = py::dict(dictionary);

	child_list_t<Value> children;
	for (auto item : dict_casted) {
		py::object item_key = item.first.cast<py::object>();
		py::object item_value = item.second.cast<py::object>();

		if (!py::isinstance<py::str>(item_key)) {
			throw InvalidInputException("NestedDictToStruct only accepts a dictionary with string keys");
		}

		if (py::isinstance<py::int_>(item_value)) {
			int32_t item_value_int = py::int_(item_value);
			children.push_back(std::make_pair(py::str(item_key), Value(item_value_int)));
		} else if (py::isinstance<py::dict>(item_value)) {
			children.push_back(std::make_pair(py::str(item_key), NestedDictToStruct(item_value)));
		} else {
			throw InvalidInputException(
			    "NestedDictToStruct only accepts a dictionary with integer values or nested dictionaries");
		}
	}
	return Value::STRUCT(std::move(children));
}

void DuckDBPyRelation::ToParquet(const string &filename, const py::object &compression, const py::object &field_ids,
                                 const py::object &row_group_size_bytes, const py::object &row_group_size,
                                 const py::object &overwrite, const py::object &per_thread_output,
                                 const py::object &use_tmp_file, const py::object &partition_by,
                                 const py::object &write_partition_columns, const py::object &append,
                                 const py::object &filename_pattern, const py::object &file_size_bytes) {
	case_insensitive_map_t<vector<Value>> options;

	if (!py::none().is(compression)) {
		if (!py::isinstance<py::str>(compression)) {
			throw InvalidInputException("to_parquet only accepts 'compression' as a string");
		}
		options["compression"] = {Value(py::str(compression))};
	}

	if (!py::none().is(field_ids)) {
		if (py::isinstance<py::dict>(field_ids)) {
			Value field_ids_value = NestedDictToStruct(field_ids);
			options["field_ids"] = {field_ids_value};
		} else if (py::isinstance<py::str>(field_ids)) {
			options["field_ids"] = {Value(py::str(field_ids))};
		} else {
			throw InvalidInputException("to_parquet only accepts 'field_ids' as a dictionary or 'auto'");
		}
	}

	if (!py::none().is(row_group_size_bytes)) {
		if (py::isinstance<py::int_>(row_group_size_bytes)) {
			int64_t row_group_size_bytes_int = py::int_(row_group_size_bytes);
			options["row_group_size_bytes"] = {Value(row_group_size_bytes_int)};
		} else if (py::isinstance<py::str>(row_group_size_bytes)) {
			options["row_group_size_bytes"] = {Value(py::str(row_group_size_bytes))};
		} else {
			throw InvalidInputException(
			    "to_parquet only accepts 'row_group_size_bytes' as an integer or 'auto' string");
		}
	}

	if (!py::none().is(row_group_size)) {
		if (!py::isinstance<py::int_>(row_group_size)) {
			throw InvalidInputException("to_parquet only accepts 'row_group_size' as an integer");
		}
		int64_t row_group_size_int = py::int_(row_group_size);
		options["row_group_size"] = {Value(row_group_size_int)};
	}

	if (!py::none().is(partition_by)) {
		if (!py::isinstance<py::list>(partition_by)) {
			throw InvalidInputException("to_parquet only accepts 'partition_by' as a list of strings");
		}
		vector<Value> partition_by_values;
		const py::list &partition_fields = partition_by;
		for (auto &field : partition_fields) {
			if (!py::isinstance<py::str>(field)) {
				throw InvalidInputException("to_parquet only accepts 'partition_by' as a list of strings");
			}
			partition_by_values.emplace_back(Value(py::str(field)));
		}
		options["partition_by"] = {partition_by_values};
	}

	if (!py::none().is(write_partition_columns)) {
		if (!py::isinstance<py::bool_>(write_partition_columns)) {
			throw InvalidInputException("to_parquet only accepts 'write_partition_columns' as a boolean");
		}
		options["write_partition_columns"] = {Value::BOOLEAN(py::bool_(write_partition_columns))};
	}

	if (!py::none().is(append)) {
		if (!py::isinstance<py::bool_>(append)) {
			throw InvalidInputException("to_parquet only accepts 'append' as a boolean");
		}
		options["append"] = {Value::BOOLEAN(py::bool_(append))};
	}

	if (!py::none().is(overwrite)) {
		if (!py::isinstance<py::bool_>(overwrite)) {
			throw InvalidInputException("to_parquet only accepts 'overwrite' as a boolean");
		}
		options["overwrite_or_ignore"] = {Value::BOOLEAN(py::bool_(overwrite))};
	}

	if (!py::none().is(per_thread_output)) {
		if (!py::isinstance<py::bool_>(per_thread_output)) {
			throw InvalidInputException("to_parquet only accepts 'per_thread_output' as a boolean");
		}
		options["per_thread_output"] = {Value::BOOLEAN(py::bool_(per_thread_output))};
	}

	if (!py::none().is(use_tmp_file)) {
		if (!py::isinstance<py::bool_>(use_tmp_file)) {
			throw InvalidInputException("to_parquet only accepts 'use_tmp_file' as a boolean");
		}
		options["use_tmp_file"] = {Value::BOOLEAN(py::bool_(use_tmp_file))};
	}

	if (!py::none().is(filename_pattern)) {
		if (!py::isinstance<py::str>(filename_pattern)) {
			throw InvalidInputException("to_parquet only accepts 'filename_pattern' as a string");
		}
		options["filename_pattern"] = {Value(py::str(filename_pattern))};
	}

	if (!py::none().is(file_size_bytes)) {
		if (py::isinstance<py::int_>(file_size_bytes)) {
			int64_t file_size_bytes_int = py::int_(file_size_bytes);
			options["file_size_bytes"] = {Value(file_size_bytes_int)};
		} else if (py::isinstance<py::str>(file_size_bytes)) {
			options["file_size_bytes"] = {Value(py::str(file_size_bytes))};
		} else {
			throw InvalidInputException("to_parquet only accepts 'file_size_bytes' as an integer or string");
		}
	}

	auto write_parquet = rel->WriteParquetRel(filename, std::move(options));
	if (TryDispatchToRunner(write_parquet, connection_owner)) {
		return;
	}
	PyExecuteRelation(write_parquet);
}

void DuckDBPyRelation::ToCSV(const string &filename, const py::object &sep, const py::object &na_rep,
                             const py::object &header, const py::object &quotechar, const py::object &escapechar,
                             const py::object &date_format, const py::object &timestamp_format,
                             const py::object &quoting, const py::object &encoding, const py::object &compression,
                             const py::object &overwrite, const py::object &per_thread_output,
                             const py::object &use_tmp_file, const py::object &partition_by,
                             const py::object &write_partition_columns) {
	case_insensitive_map_t<vector<Value>> options;

	if (!py::none().is(sep)) {
		if (!py::isinstance<py::str>(sep)) {
			throw InvalidInputException("to_csv only accepts 'sep' as a string");
		}
		options["delimiter"] = {Value(py::str(sep))};
	}

	if (!py::none().is(na_rep)) {
		if (!py::isinstance<py::str>(na_rep)) {
			throw InvalidInputException("to_csv only accepts 'na_rep' as a string");
		}
		options["null"] = {Value(py::str(na_rep))};
	}

	if (!py::none().is(header)) {
		if (!py::isinstance<py::bool_>(header)) {
			throw InvalidInputException("to_csv only accepts 'header' as a boolean");
		}
		options["header"] = {Value::BOOLEAN(py::bool_(header))};
	}

	if (!py::none().is(quotechar)) {
		if (!py::isinstance<py::str>(quotechar)) {
			throw InvalidInputException("to_csv only accepts 'quotechar' as a string");
		}
		options["quote"] = {Value(py::str(quotechar))};
	}

	if (!py::none().is(escapechar)) {
		if (!py::isinstance<py::str>(escapechar)) {
			throw InvalidInputException("to_csv only accepts 'escapechar' as a string");
		}
		options["escape"] = {Value(py::str(escapechar))};
	}

	if (!py::none().is(date_format)) {
		if (!py::isinstance<py::str>(date_format)) {
			throw InvalidInputException("to_csv only accepts 'date_format' as a string");
		}
		options["dateformat"] = {Value(py::str(date_format))};
	}

	if (!py::none().is(timestamp_format)) {
		if (!py::isinstance<py::str>(timestamp_format)) {
			throw InvalidInputException("to_csv only accepts 'timestamp_format' as a string");
		}
		options["timestampformat"] = {Value(py::str(timestamp_format))};
	}

	if (!py::none().is(quoting)) {
		// TODO: add list of strings as valid option
		if (py::isinstance<py::str>(quoting)) {
			string quoting_option = StringUtil::Lower(py::str(quoting));
			if (quoting_option != "force" && quoting_option != "all") {
				throw InvalidInputException(
				    "to_csv 'quoting' supported options are ALL or FORCE (both set FORCE_QUOTE=True)");
			}
		} else if (py::isinstance<py::int_>(quoting)) {
			int64_t quoting_value = py::int_(quoting);
			// csv.QUOTE_ALL expands to 1
			static constexpr int64_t QUOTE_ALL = 1;
			if (quoting_value != QUOTE_ALL) {
				throw InvalidInputException("Only csv.QUOTE_ALL is a supported option for 'quoting' currently");
			}
		} else {
			throw InvalidInputException(
			    "to_csv only accepts 'quoting' as a string or a constant from the 'csv' package");
		}
		options["force_quote"] = {Value("*")};
	}

	if (!py::none().is(encoding)) {
		if (!py::isinstance<py::str>(encoding)) {
			throw InvalidInputException("to_csv only accepts 'encoding' as a string");
		}
		string encoding_option = StringUtil::Lower(py::str(encoding));
		if (encoding_option != "utf-8" && encoding_option != "utf8") {
			throw InvalidInputException("The only supported encoding option is 'UTF8");
		}
	}

	if (!py::none().is(compression)) {
		if (!py::isinstance<py::str>(compression)) {
			throw InvalidInputException("to_csv only accepts 'compression' as a string");
		}
		options["compression"] = {Value(py::str(compression))};
	}

	if (!py::none().is(overwrite)) {
		if (!py::isinstance<py::bool_>(overwrite)) {
			throw InvalidInputException("to_csv only accepts 'overwrite' as a boolean");
		}
		options["overwrite_or_ignore"] = {Value::BOOLEAN(py::bool_(overwrite))};
	}

	if (!py::none().is(per_thread_output)) {
		if (!py::isinstance<py::bool_>(per_thread_output)) {
			throw InvalidInputException("to_csv only accepts 'per_thread_output' as a boolean");
		}
		options["per_thread_output"] = {Value::BOOLEAN(py::bool_(per_thread_output))};
	}

	if (!py::none().is(use_tmp_file)) {
		if (!py::isinstance<py::bool_>(use_tmp_file)) {
			throw InvalidInputException("to_csv only accepts 'use_tmp_file' as a boolean");
		}
		options["use_tmp_file"] = {Value::BOOLEAN(py::bool_(use_tmp_file))};
	}

	if (!py::none().is(partition_by)) {
		if (!py::isinstance<py::list>(partition_by)) {
			throw InvalidInputException("to_csv only accepts 'partition_by' as a list of strings");
		}
		vector<Value> partition_by_values;
		const py::list &partition_fields = partition_by;
		for (auto &field : partition_fields) {
			if (!py::isinstance<py::str>(field)) {
				throw InvalidInputException("to_csv only accepts 'partition_by' as a list of strings");
			}
			partition_by_values.emplace_back(Value(py::str(field)));
		}
		options["partition_by"] = {partition_by_values};
	}

	if (!py::none().is(write_partition_columns)) {
		if (!py::isinstance<py::bool_>(write_partition_columns)) {
			throw InvalidInputException("to_csv only accepts 'write_partition_columns' as a boolean");
		}
		options["write_partition_columns"] = {Value::BOOLEAN(py::bool_(write_partition_columns))};
	}

	auto write_csv = rel->WriteCSVRel(filename, std::move(options));
	if (TryDispatchToRunner(write_csv, connection_owner)) {
		return;
	}
	PyExecuteRelation(write_csv);
}

// should this return a rel with the new view?
unique_ptr<DuckDBPyRelation> DuckDBPyRelation::CreateView(const string &view_name, bool replace) {
	rel->CreateView(view_name, replace);
	return DeriveRelation(rel);
}

static bool IsDescribeStatement(SQLStatement &statement) {
	if (statement.type != StatementType::PRAGMA_STATEMENT) {
		return false;
	}
	auto &pragma_statement = statement.Cast<PragmaStatement>();
	if (pragma_statement.info->name != "show") {
		return false;
	}
	return true;
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Query(const string &view_name, const string &sql_query) {
	auto view_relation = CreateView(view_name);
	auto all_dependencies = rel->GetAllDependencies();

	Parser parser(rel->context->GetContext()->GetParserOptions());
	parser.ParseQuery(sql_query);
	if (parser.statements.size() != 1) {
		throw InvalidInputException("'DuckDBPyRelation.query' only accepts a single statement");
	}
	auto &statement = *parser.statements[0];
	if (statement.type == StatementType::SELECT_STATEMENT) {
		auto select_statement = unique_ptr_cast<SQLStatement, SelectStatement>(std::move(parser.statements[0]));
		auto query_relation = make_shared_ptr<QueryRelation>(rel->context->GetContext(), std::move(select_statement),
		                                                     sql_query, "query_relation");
		return DeriveRelation(std::move(query_relation));
	} else if (IsDescribeStatement(statement)) {
		auto query = PragmaShow(view_name);
		return Query(view_name, query);
	}
	{
		D_ASSERT(py::gil_check());
		py::gil_scoped_release release;
		auto query_result = rel->context->GetContext()->Query(std::move(parser.statements[0]), false);
		// Execute it anyways, for creation/altering statements
		// We only care that it succeeds, we can't store the result
		D_ASSERT(query_result);
		if (query_result->HasError()) {
			query_result->ThrowError();
		}
	}
	return nullptr;
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Explode(const string &column) {
	AssertRelation();
	// Create UnnestRelation with custom Bind() that directly builds
	// LogicalUnnest + LogicalProjection — no AST parsing, no catalog ops.
	auto unnest_rel = make_shared_ptr<UnnestRelation>(rel, column);
	return DeriveRelation(std::move(unnest_rel));
}

DuckDBPyRelation &DuckDBPyRelation::Execute() {
	AssertRelation();
	ExecuteOrThrow();
	return *this;
}

void DuckDBPyRelation::InsertInto(const string &table) {
	AssertRelation();
	auto parsed_info = QualifiedName::Parse(table);
	auto insert = rel->InsertRel(parsed_info.schema, parsed_info.name);
	PyExecuteRelation(insert);
}

void DuckDBPyRelation::Update(const py::object &set_p, const py::object &where) {
	AssertRelation();
	unique_ptr<ParsedExpression> condition;
	if (!py::none().is(where)) {
		shared_ptr<DuckDBPyExpression> py_expr;
		if (!py::try_cast<shared_ptr<DuckDBPyExpression>>(where, py_expr)) {
			throw InvalidInputException("Please provide an Expression to 'condition'");
		}
		condition = py_expr->GetExpression().Copy();
	}

	if (!py::is_dict_like(set_p)) {
		throw InvalidInputException("Please provide 'set' as a dictionary of column name to Expression");
	}

	vector<string> names;
	vector<unique_ptr<ParsedExpression>> expressions;

	py::dict set = py::dict(set_p);
	auto arg_count = set.size();
	if (arg_count == 0) {
		throw InvalidInputException("Please provide at least one set expression");
	}

	for (auto item : set) {
		py::object item_key = item.first.cast<py::object>();
		py::object item_value = item.second.cast<py::object>();

		if (!py::isinstance<py::str>(item_key)) {
			throw InvalidInputException("Please provide the column name as the key of the dictionary");
		}
		shared_ptr<DuckDBPyExpression> py_expr;
		if (!py::try_cast<shared_ptr<DuckDBPyExpression>>(item_value, py_expr)) {
			string actual_type = py::str(py::type::of(item_value));
			throw InvalidInputException("Please provide an object of type Expression as the value, not %s",
			                            actual_type);
		}
		names.push_back(std::string(py::str(item_key)));
		expressions.push_back(py_expr->GetExpression().Copy());
	}

	return rel->Update(std::move(names), std::move(expressions), std::move(condition));
}

void DuckDBPyRelation::Insert(const py::object &params) const {
	AssertRelation();
	if (this->rel->type != RelationType::TABLE_RELATION) {
		throw InvalidInputException("'DuckDBPyRelation.insert' can only be used on a table relation");
	}
	vector<vector<Value>> values {DuckDBPyConnection::TransformPythonParamList(params)};

	D_ASSERT(py::gil_check());
	py::gil_scoped_release release;
	rel->Insert(values);
}

void DuckDBPyRelation::Create(const string &table) {
	AssertRelation();
	auto parsed_info = QualifiedName::Parse(table);
	auto create = rel->CreateRel(parsed_info.schema, parsed_info.name, false);
	PyExecuteRelation(create);
}

static bool IsPythonClassCallable(const py::object &fun) {
	auto inspect_module = py::module_::import("inspect");
	return py::cast<bool>(inspect_module.attr("isclass")(fun));
}

static string ResolveUDFExecutionBackend(const Optional<py::object> &execution_backend, const py::object &fun,
                                         const string &runner_type) {
	if (execution_backend.is_none()) {
		const bool use_actor = IsPythonClassCallable(fun);
		if (runner_type == "ray") {
			return use_actor ? "ray_actor" : "ray_task";
		}
		return use_actor ? "subprocess_actor" : "subprocess_task";
	}
	auto backend = StringUtil::Lower(py::cast<string>(execution_backend));
	if (backend != "subprocess_task" && backend != "subprocess_actor" && backend != "ray_task" &&
	    backend != "ray_actor") {
		throw InvalidInputException(
		    "execution_backend must be one of: subprocess_task, subprocess_actor, ray_task, ray_actor");
	}
	return backend;
}

static bool IsSubprocessExecutionBackend(const string &backend) {
	return backend == "subprocess_task" || backend == "subprocess_actor";
}

static string ResolveRayActorThreadPolicy(const Optional<py::object> &thread_policy, const string &execution_backend,
                                          const string &operation_name) {
	if (thread_policy.is_none()) {
		return string();
	}
	if (!py::isinstance<py::str>(thread_policy)) {
		throw InvalidInputException(operation_name + "(ray_actor_thread_policy=...) must be 'managed' or 'ray_native'");
	}
	auto policy = StringUtil::Lower(py::cast<string>(thread_policy));
	if (policy != "managed" && policy != "ray_native") {
		throw InvalidInputException(operation_name + "(ray_actor_thread_policy=...) must be 'managed' or 'ray_native'");
	}
	if (execution_backend != "ray_actor") {
		throw InvalidInputException(operation_name +
		                            "(ray_actor_thread_policy=...) requires execution_backend='ray_actor'");
	}
	return policy;
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::Map(py::function fun, const shared_ptr<DuckDBPyType> &return_type,
                                                   const Optional<py::object> &batch_size,
                                                   const Optional<py::object> &cpus, const Optional<py::object> &gpus,
                                                   const Optional<py::object> &execution_backend,
                                                   const Optional<py::object> &actor_number, bool side_effects) {
	AssertRelation();
	if (!return_type) {
		throw InvalidInputException("map requires return_type");
	}
	auto resolved_execution_backend = ResolveUDFExecutionBackend(execution_backend, fun, ResolveRunnerType());
	auto default_parallelism =
	    static_cast<idx_t>(TaskScheduler::GetScheduler(*rel->context->GetContext()).NumberOfThreads());
	vector<LogicalType> passthrough_types;
	passthrough_types.reserve(rel->Columns().size());
	for (auto &col : rel->Columns()) {
		passthrough_types.push_back(col.GetType());
	}
	auto payload = BuildScalarUDFPayload("map_udf", fun, return_type, resolved_execution_backend, default_parallelism,
	                                     passthrough_types, cpus, gpus, batch_size, actor_number, side_effects);

	{
		auto &existing_children = StructValue::GetChildren(payload);
		auto &existing_type = payload.type();
		child_list_t<Value> new_children;
		for (idx_t i = 0; i < StructType::GetChildCount(existing_type); i++) {
			new_children.emplace_back(StructType::GetChildName(existing_type, i), existing_children[i]);
		}
		vector<Value> input_name_values;
		for (auto &col : rel->Columns()) {
			input_name_values.emplace_back(Value(col.Name()));
		}
		new_children.emplace_back("input_names", Value::LIST(LogicalType::VARCHAR, std::move(input_name_values)));
		payload = Value::STRUCT(std::move(new_children));
	}

	vector<unique_ptr<ParsedExpression>> func_args;
	for (auto &col : rel->Columns()) {
		func_args.push_back(make_uniq<ColumnRefExpression>(col.Name()));
	}
	func_args.push_back(make_uniq<ConstantExpression>(std::move(payload)));
	auto func_expr = make_uniq<FunctionExpression>("udf", std::move(func_args));

	vector<unique_ptr<ParsedExpression>> project_exprs;
	vector<string> aliases;
	for (auto &col : rel->Columns()) {
		project_exprs.push_back(make_uniq<ColumnRefExpression>(col.Name()));
		aliases.push_back(col.Name());
	}
	project_exprs.push_back(std::move(func_expr));
	aliases.push_back("value");
	auto relation = make_uniq<DuckDBPyRelation>(rel->Project(std::move(project_exprs), aliases));
	relation->connection_owner = connection_owner;
	auto rel_dependency = make_uniq<ExternalDependency>();
	rel_dependency->AddDependency("function", PythonDependencyItem::Create(std::move(fun)));
	relation->rel->AddExternalDependency(std::move(rel_dependency));
	return relation;
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::MapBatches(
    py::function fun, Optional<py::object> schema, const Optional<py::object> &batch_size,
    const Optional<py::object> &output_batch_size, const Optional<py::object> &min_task_batch_size,
    const Optional<py::object> &preserve_compute_batch_boundaries, const Optional<py::object> &cpus,
    const Optional<py::object> &gpus, const Optional<py::object> &memory_bytes,
    const Optional<py::object> &execution_backend, const Optional<py::object> &actor_number,
    const Optional<py::object> &ray_actor_thread_policy, const Optional<py::object> &streaming_breaker,
    const Optional<py::object> &target_max_batch_bytes, const Optional<py::object> &task_input_max_bytes,
    const Optional<py::object> &output_target_max_bytes) {
	AssertRelation();
	if (schema.is_none() || !py::isinstance<py::dict>(schema)) {
		throw InvalidInputException("map_batches requires a schema dict");
	}
	auto resolved_execution_backend = ResolveUDFExecutionBackend(execution_backend, fun, ResolveRunnerType());
	auto resolved_ray_actor_thread_policy =
	    ResolveRayActorThreadPolicy(ray_actor_thread_policy, resolved_execution_backend, "map_batches");
	const bool uses_subprocess_backend = IsSubprocessExecutionBackend(resolved_execution_backend);
	if (!gpus.is_none()) {
		try {
			(void)py::cast<double>(gpus);
		} catch (...) {
			throw InvalidInputException("map_batches(gpus=...) must be a number");
		}
	}
	auto default_parallelism =
	    static_cast<idx_t>(TaskScheduler::GetScheduler(*rel->context->GetContext()).NumberOfThreads());
	auto payload = BuildPythonUDFPayload("udf", fun, schema, shared_ptr<DuckDBPyType>(), resolved_execution_backend,
	                                     default_parallelism, cpus, gpus, memory_bytes, batch_size, output_batch_size,
	                                     min_task_batch_size, preserve_compute_batch_boundaries, actor_number,
	                                     target_max_batch_bytes, task_input_max_bytes, output_target_max_bytes,
	                                     /*side_effects=*/false);
	const bool streaming_breaker_requested = streaming_breaker.is_none() || py::cast<bool>(streaming_breaker);
	if (!min_task_batch_size.is_none() && !streaming_breaker_requested) {
		throw InvalidInputException("min_task_batch_size requires streaming_breaker=True");
	}
	if (!preserve_compute_batch_boundaries.is_none() && py::cast<bool>(preserve_compute_batch_boundaries) &&
	    !streaming_breaker_requested) {
		throw InvalidInputException("preserve_compute_batch_boundaries requires streaming_breaker=True");
	}

	// Add input_names to payload for column renaming in Python executors
	auto &existing_children = StructValue::GetChildren(payload);
	auto &existing_type = payload.type();
	child_list_t<Value> new_children;
	for (idx_t i = 0; i < StructType::GetChildCount(existing_type); i++) {
		auto child_name = StructType::GetChildName(existing_type, i);
		new_children.emplace_back(child_name, existing_children[i]);
	}
	if (!resolved_ray_actor_thread_policy.empty()) {
		new_children.emplace_back("ray_actor_thread_policy", Value(resolved_ray_actor_thread_policy));
	}
	vector<Value> input_name_values;
	for (auto &col : rel->Columns()) {
		input_name_values.emplace_back(Value(col.Name()));
	}
	new_children.emplace_back("input_names", Value::LIST(LogicalType::VARCHAR, std::move(input_name_values)));
	new_children.emplace_back("row_preserving", Value::BOOLEAN(false));
	if (streaming_breaker_requested) {
		new_children.emplace_back("streaming_breaker", Value::BOOLEAN(true));
		if (uses_subprocess_backend) {
			new_children.emplace_back("produce_ref_bundle_output", Value::BOOLEAN(true));
			new_children.emplace_back("streaming_output_mode", Value("local_shm_ref_bundle"));
		} else {
			new_children.emplace_back("produce_ray_block_stream", Value::BOOLEAN(true));
		}
		new_children.emplace_back("prebatched_input", Value::BOOLEAN(false));
	}
	payload = Value::STRUCT(std::move(new_children));

	// Build udf(col1, col2, ..., payload) expression
	vector<unique_ptr<ParsedExpression>> func_args;
	for (auto &col : rel->Columns()) {
		func_args.push_back(make_uniq<ColumnRefExpression>(col.Name()));
	}
	func_args.push_back(make_uniq<ConstantExpression>(std::move(payload)));
	auto func_expr = make_uniq<FunctionExpression>("udf", std::move(func_args));

	// Extract output_names from schema for struct flattening
	vector<string> output_names;
	if (!schema.is_none() && py::isinstance<py::dict>(schema)) {
		auto schema_dict = py::reinterpret_borrow<py::dict>(schema);
		for (auto &item : schema_dict) {
			output_names.push_back(std::string(py::str(item.first)));
		}
	}

	vector<unique_ptr<ParsedExpression>> project_exprs;
	vector<string> aliases;

	if (output_names.size() > 1) {
		// Multi-column schema: udf returns a STRUCT.
		// First project: udf(...) AS __udf_result
		project_exprs.push_back(std::move(func_expr));
		aliases.push_back("__udf_result");
		auto udf_relation = rel->Project(std::move(project_exprs), aliases);

		// Second project: struct_extract(__udf_result, 'col_name') AS col_name for each output column
		vector<unique_ptr<ParsedExpression>> flatten_exprs;
		vector<string> flatten_aliases;
		for (auto &name : output_names) {
			vector<unique_ptr<ParsedExpression>> extract_args;
			extract_args.push_back(make_uniq<ColumnRefExpression>("__udf_result"));
			extract_args.push_back(make_uniq<ConstantExpression>(Value(name)));
			flatten_exprs.push_back(make_uniq<FunctionExpression>("struct_extract", std::move(extract_args)));
			flatten_aliases.push_back(name);
		}
		auto relation = make_uniq<DuckDBPyRelation>(udf_relation->Project(std::move(flatten_exprs), flatten_aliases));
		relation->connection_owner = connection_owner;
		auto rel_dependency = make_uniq<ExternalDependency>();
		rel_dependency->AddDependency("function", PythonDependencyItem::Create(std::move(fun)));
		if (!schema.is_none()) {
			rel_dependency->AddDependency("schema", PythonDependencyItem::Create(std::move(schema)));
		}
		relation->rel->AddExternalDependency(std::move(rel_dependency));
		return relation;
	}

	// Single column or no schema: keep the original behavior
	project_exprs.push_back(std::move(func_expr));
	if (output_names.size() == 1) {
		aliases.push_back(output_names[0]);
	}
	auto relation = make_uniq<DuckDBPyRelation>(rel->Project(std::move(project_exprs), aliases));
	relation->connection_owner = connection_owner;
	auto rel_dependency = make_uniq<ExternalDependency>();
	rel_dependency->AddDependency("function", PythonDependencyItem::Create(std::move(fun)));
	if (!schema.is_none()) {
		rel_dependency->AddDependency("schema", PythonDependencyItem::Create(std::move(schema)));
	}
	relation->rel->AddExternalDependency(std::move(rel_dependency));
	return relation;
}

unique_ptr<DuckDBPyRelation> DuckDBPyRelation::FlatMap(
    py::function fun, Optional<py::object> schema, const Optional<py::object> &batch_size,
    const Optional<py::object> &output_batch_size, const Optional<py::object> &min_task_batch_size,
    const Optional<py::object> &preserve_compute_batch_boundaries, const Optional<py::object> &cpus,
    const Optional<py::object> &gpus, const Optional<py::object> &memory_bytes,
    const Optional<py::object> &execution_backend, const Optional<py::object> &actor_number,
    const Optional<py::object> &streaming_breaker, const Optional<py::object> &target_max_batch_bytes,
    const Optional<py::object> &task_input_max_bytes, const Optional<py::object> &output_target_max_bytes) {
	AssertRelation();
	if (schema.is_none() || !py::isinstance<py::dict>(schema)) {
		throw InvalidInputException("flat_map requires a schema dict");
	}
	auto resolved_execution_backend = ResolveUDFExecutionBackend(execution_backend, fun, ResolveRunnerType());
	const bool uses_subprocess_backend = IsSubprocessExecutionBackend(resolved_execution_backend);
	if (!gpus.is_none()) {
		try {
			(void)py::cast<double>(gpus);
		} catch (...) {
			throw InvalidInputException("flat_map(gpus=...) must be a number");
		}
	}
	auto default_parallelism =
	    static_cast<idx_t>(TaskScheduler::GetScheduler(*rel->context->GetContext()).NumberOfThreads());
	auto payload = BuildPythonUDFPayload("udf", fun, schema, shared_ptr<DuckDBPyType>(), resolved_execution_backend,
	                                     default_parallelism, cpus, gpus, memory_bytes, batch_size, output_batch_size,
	                                     min_task_batch_size, preserve_compute_batch_boundaries, actor_number,
	                                     target_max_batch_bytes, task_input_max_bytes, output_target_max_bytes,
	                                     /*side_effects=*/false,
	                                     /*flat_map=*/true);
	const bool streaming_breaker_requested = streaming_breaker.is_none() || py::cast<bool>(streaming_breaker);
	if (!min_task_batch_size.is_none() && !streaming_breaker_requested) {
		throw InvalidInputException("min_task_batch_size requires streaming_breaker=True");
	}
	if (!preserve_compute_batch_boundaries.is_none() && py::cast<bool>(preserve_compute_batch_boundaries) &&
	    !streaming_breaker_requested) {
		throw InvalidInputException("preserve_compute_batch_boundaries requires streaming_breaker=True");
	}

	// Add input_names to payload for column renaming in Python executors
	{
		auto &existing_children = StructValue::GetChildren(payload);
		auto &existing_type = payload.type();
		child_list_t<Value> new_children;
		for (idx_t i = 0; i < StructType::GetChildCount(existing_type); i++) {
			new_children.emplace_back(StructType::GetChildName(existing_type, i), existing_children[i]);
		}
		vector<Value> input_name_values;
		for (auto &col : rel->Columns()) {
			input_name_values.emplace_back(Value(col.Name()));
		}
		new_children.emplace_back("input_names", Value::LIST(LogicalType::VARCHAR, std::move(input_name_values)));
		new_children.emplace_back("row_preserving", Value::BOOLEAN(false));
		if (streaming_breaker_requested) {
			new_children.emplace_back("streaming_breaker", Value::BOOLEAN(true));
			if (uses_subprocess_backend) {
				new_children.emplace_back("produce_ref_bundle_output", Value::BOOLEAN(true));
				new_children.emplace_back("streaming_output_mode", Value("local_shm_ref_bundle"));
			} else {
				new_children.emplace_back("produce_ray_block_stream", Value::BOOLEAN(true));
			}
			new_children.emplace_back("prebatched_input", Value::BOOLEAN(false));
		}
		payload = Value::STRUCT(std::move(new_children));
	}

	// Build udf(col1, col2, ..., payload) expression
	vector<unique_ptr<ParsedExpression>> func_args;
	for (auto &col : rel->Columns()) {
		func_args.push_back(make_uniq<ColumnRefExpression>(col.Name()));
	}
	func_args.push_back(make_uniq<ConstantExpression>(std::move(payload)));
	auto func_expr = make_uniq<FunctionExpression>("udf", std::move(func_args));

	// Extract output_names from schema for struct flattening
	vector<string> output_names;
	if (!schema.is_none() && py::isinstance<py::dict>(schema)) {
		auto schema_dict = py::reinterpret_borrow<py::dict>(schema);
		for (auto &item : schema_dict) {
			output_names.push_back(std::string(py::str(item.first)));
		}
	}

	vector<unique_ptr<ParsedExpression>> project_exprs;
	vector<string> aliases;

	if (output_names.size() > 1) {
		// Multi-column schema: udf returns a STRUCT.
		// First project: udf(...) AS __udf_result
		project_exprs.push_back(std::move(func_expr));
		aliases.push_back("__udf_result");
		auto udf_relation = rel->Project(std::move(project_exprs), aliases);

		// Second project: struct_extract(__udf_result, 'col_name') AS col_name for each output column
		vector<unique_ptr<ParsedExpression>> flatten_exprs;
		vector<string> flatten_aliases;
		for (auto &name : output_names) {
			vector<unique_ptr<ParsedExpression>> extract_args;
			extract_args.push_back(make_uniq<ColumnRefExpression>("__udf_result"));
			extract_args.push_back(make_uniq<ConstantExpression>(Value(name)));
			flatten_exprs.push_back(make_uniq<FunctionExpression>("struct_extract", std::move(extract_args)));
			flatten_aliases.push_back(name);
		}
		auto relation = make_uniq<DuckDBPyRelation>(udf_relation->Project(std::move(flatten_exprs), flatten_aliases));
		relation->connection_owner = connection_owner;
		auto rel_dependency = make_uniq<ExternalDependency>();
		rel_dependency->AddDependency("function", PythonDependencyItem::Create(std::move(fun)));
		if (!schema.is_none()) {
			rel_dependency->AddDependency("schema", PythonDependencyItem::Create(std::move(schema)));
		}
		relation->rel->AddExternalDependency(std::move(rel_dependency));
		return relation;
	}

	// Single column or no schema: keep the original behavior
	project_exprs.push_back(std::move(func_expr));
	auto relation = make_uniq<DuckDBPyRelation>(rel->Project(std::move(project_exprs), aliases));
	relation->connection_owner = connection_owner;
	auto rel_dependency = make_uniq<ExternalDependency>();
	rel_dependency->AddDependency("function", PythonDependencyItem::Create(std::move(fun)));
	if (!schema.is_none()) {
		rel_dependency->AddDependency("schema", PythonDependencyItem::Create(std::move(schema)));
	}
	relation->rel->AddExternalDependency(std::move(rel_dependency));
	return relation;
}

string DuckDBPyRelation::ToStringInternal(const BoxRendererConfig &config, bool invalidate_cache) {
	AssertRelation();
	if (rendered_result.empty() || invalidate_cache) {
		BoxRenderer renderer;
		auto limit = Limit(config.limit, 0);
		auto res = limit->ExecuteInternal();

		auto context = rel->context->GetContext();
		rendered_result = res->ToBox(*context, config);
	}
	return rendered_result;
}

string DuckDBPyRelation::ToString() {
	BoxRendererConfig config;
	config.limit = 10000;
	if (DuckDBPyConnection::IsJupyter()) {
		config.max_width = 10000;
	}
	return ToStringInternal(config);
}

static idx_t IndexFromPyInt(const py::object &object) {
	auto index = py::cast<idx_t>(object);
	return index;
}

bool DuckDBPyRelation::TryPrintDistributed(const BoxRendererConfig &config) {
	AssertRelation();
	if (ResolveRunnerType() != "ray") {
		return false;
	}

	auto runner = GetOrCreateRunnerForDB(rel, "ray");
	auto limited_relation = Limit(static_cast<int64_t>(config.limit), 0);
	auto limited_relation_obj = py::cast(std::move(limited_relation));
	py::list tables;
	try {
		auto table_iter = runner.attr("run_iter_tables")(limited_relation_obj, py::int_(1));
		for (auto table : py::reinterpret_borrow<py::iterable>(table_iter)) {
			tables.append(table);
		}
	} catch (...) {
		ForgetRunnerForDB(rel);
		throw;
	}
	ForgetRunnerForDB(rel);

	unique_ptr<QueryResult> result;
	if (py::len(tables) == 0) {
		auto empty_relation = EmptyResult(rel->context->GetContext(), types, names);
		result = empty_relation->ExecuteInternal();
	} else {
		py::object arrow_table = tables[0];
		if (py::len(tables) > 1) {
			auto pyarrow = py::module_::import("pyarrow");
			arrow_table = pyarrow.attr("concat_tables")(tables);
		}

		// Arrow permits duplicate field names, but DuckDB's Arrow scanner resolves
		// fields by name. Use unique scan names and restore the relation's names in
		// the renderer below.
		py::list scan_names;
		for (idx_t column_idx = 0; column_idx < names.size(); column_idx++) {
			scan_names.append(py::str("__vane_show_column_" + std::to_string(column_idx)));
		}
		arrow_table = arrow_table.attr("rename_columns")(scan_names);

		auto duckdb_module = py::module_::import("duckdb");
		auto local_relation_obj = duckdb_module.attr("from_arrow")(arrow_table);
		auto &local_relation = local_relation_obj.cast<DuckDBPyRelation &>();
		result = local_relation.ExecuteInternal();
	}

	if (!result || result->type != QueryResultType::MATERIALIZED_RESULT) {
		throw InternalException("Distributed show did not produce a materialized result");
	}
	auto &materialized = result->Cast<MaterializedQueryResult>();
	BoxRenderer renderer(config);
	auto context = rel->context->GetContext();
	py::print(py::str(renderer.ToString(*context, names, materialized.Collection())));
	return true;
}

void DuckDBPyRelation::Print(const Optional<py::int_> &max_width, const Optional<py::int_> &max_rows,
                             const Optional<py::int_> &max_col_width, const Optional<py::str> &null_value,
                             const py::object &render_mode) {
	BoxRendererConfig config;
	config.limit = 10000;
	if (DuckDBPyConnection::IsJupyter()) {
		config.max_width = 10000;
	}

	bool invalidate_cache = false;
	if (!py::none().is(max_width)) {
		invalidate_cache = true;
		config.max_width = IndexFromPyInt(max_width);
	}
	if (!py::none().is(max_rows)) {
		invalidate_cache = true;
		config.max_rows = IndexFromPyInt(max_rows);
	}
	if (!py::none().is(max_col_width)) {
		invalidate_cache = true;
		config.max_col_width = IndexFromPyInt(max_col_width);
	}
	if (!py::none().is(null_value)) {
		invalidate_cache = true;
		config.null_value = py::cast<std::string>(null_value);
	}
	if (!py::none().is(render_mode)) {
		invalidate_cache = true;
		if (!py::try_cast(render_mode, config.render_mode)) {
			throw InvalidInputException("'render_mode' accepts either a string, RenderMode or int value");
		}
	}

	if (TryPrintDistributed(config)) {
		return;
	}
	py::print(py::str(ToStringInternal(config, invalidate_cache)));
}

static ExplainFormat GetExplainFormat(ExplainType type) {
	if (DuckDBPyConnection::IsJupyter() && type != ExplainType::EXPLAIN_ANALYZE) {
		return ExplainFormat::HTML;
	} else {
		return ExplainFormat::DEFAULT;
	}
}

static void DisplayHTML(const string &html) {
	PythonGILWrapper gil;
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	auto html_attr = import_cache.IPython.display.HTML();
	auto html_object = html_attr(py::str(html));
	auto display_attr = import_cache.IPython.display.display();
	display_attr(html_object);
}

string DuckDBPyRelation::Explain(ExplainType type) {
	AssertRelation();
	D_ASSERT(py::gil_check());
	py::gil_scoped_release release;

	auto explain_format = GetExplainFormat(type);
	auto res = rel->Explain(type, explain_format);
	D_ASSERT(res->type == duckdb::QueryResultType::MATERIALIZED_RESULT);
	auto &materialized = res->Cast<MaterializedQueryResult>();
	auto &coll = materialized.Collection();
	if (explain_format != ExplainFormat::HTML || !DuckDBPyConnection::IsJupyter()) {
		string result;
		for (auto &row : coll.Rows()) {
			// Skip the first column because it just contains 'physical plan'
			for (idx_t col_idx = 1; col_idx < coll.ColumnCount(); col_idx++) {
				if (col_idx > 1) {
					result += "\t";
				}
				auto val = row.GetValue(col_idx);
				result += val.IsNull() ? "NULL" : StringUtil::Replace(val.ToString(), string("\0", 1), "\\0");
			}
			result += "\n";
		}
		return result;
	}

	auto chunk = materialized.Fetch();
	for (idx_t i = 0; i < chunk->size(); i++) {
		auto plan = chunk->GetValue(1, i);
		auto plan_string = plan.GetValue<string>();
		DisplayHTML(plan_string);
	}

	const string tree_resize_script = R"(
<script>
function toggleDisplay(button) {
    const parentLi = button.closest('li');
    const nestedUl = parentLi.querySelector('ul');
    if (nestedUl) {
        const currentDisplay = getComputedStyle(nestedUl).getPropertyValue('display');
        if (currentDisplay === 'none') {
            nestedUl.classList.toggle('hidden');
            button.textContent = '-';
        } else {
            nestedUl.classList.toggle('hidden');
            button.textContent = '+';
        }
    }
}

function updateTreeHeight(tfTree) {
	if (!tfTree) {
		return;
	}

	const closestElement = tfTree.closest('.lm-Widget.jp-OutputArea.jp-Cell-outputArea');
	if (!closestElement) {
		return;
	}

	console.log(closestElement);

	const height = getComputedStyle(closestElement).getPropertyValue('height');
	tfTree.style.height = height;
}

function resizeTFTree() {
	const tfTrees = document.querySelectorAll('.tf-tree');
	tfTrees.forEach(tfTree => {
		console.log(tfTree);
		if (tfTree) {
			const jupyterViewPort = tfTree.closest('.lm-Widget.jp-OutputArea.jp-Cell-outputArea');
			console.log(jupyterViewPort);
			if (jupyterViewPort) {
				const resizeObserver = new ResizeObserver(() => {
					updateTreeHeight(tfTree);
				});
				resizeObserver.observe(jupyterViewPort);
			}
		}
	});
}

resizeTFTree();

</script>
	)";
	DisplayHTML(tree_resize_script);
	return "";
}

// TODO: RelationType to a python enum
py::str DuckDBPyRelation::Type() {
	if (!rel) {
		return py::str("QUERY_RESULT");
	}
	return py::str(RelationTypeToString(rel->type));
}

py::list DuckDBPyRelation::Columns() {
	AssertRelation();
	py::list res;
	for (auto &col : rel->Columns()) {
		res.append(col.Name());
	}
	return res;
}

py::list DuckDBPyRelation::ColumnTypes() {
	AssertRelation();
	py::list res;
	for (auto &col : rel->Columns()) {
		res.append(DuckDBPyType(col.Type()));
	}
	return res;
}

bool DuckDBPyRelation::IsRelation(const py::object &object) {
	return py::isinstance<DuckDBPyRelation>(object);
}

} // namespace duckdb
