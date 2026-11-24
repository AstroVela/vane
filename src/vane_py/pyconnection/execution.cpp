// SPDX-FileCopyrightText: 2018-2026 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#include "vane_python/pyconnection/pyconnection.hpp"
#include "duckdb/common/enums/profiler_format.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/main/pending_query_result.hpp"
#include "duckdb/main/prepared_statement.hpp"
#include "duckdb/main/relation/materialized_relation.hpp"
#include "duckdb/main/relation/query_relation.hpp"
#include "duckdb/main/stream_query_result.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "vane_python/bound_plan.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"
#include "vane_python/pyresult.hpp"
#include "vane_python/pystatement.hpp"
#include "vane_python/python_conversion.hpp"
#include "vane_python/python_input_callback.hpp"
#include "vane_python/query_parameters.hpp"

namespace duckdb {

py::str DuckDBPyConnection::GetProfilingInformation(const py::str &format) {
	auto query_lock = LockForQuery();
	// We want to expose ProfilerPrintFormat as a string to Python users
	ProfilerPrintFormat format_enum;
	if (format == "query_tree") {
		format_enum = ProfilerPrintFormat::QUERY_TREE;
	} else if (format == "json") {
		format_enum = ProfilerPrintFormat::JSON;
	} else if (format == "query_tree_optimizer") {
		format_enum = ProfilerPrintFormat::QUERY_TREE_OPTIMIZER;
	} else if (format == "no_output") {
		format_enum = ProfilerPrintFormat::NO_OUTPUT;
	} else if (format == "html") {
		format_enum = ProfilerPrintFormat::HTML;
	} else if (format == "graphviz") {
		format_enum = ProfilerPrintFormat::GRAPHVIZ;
	} else {
		throw InvalidInputException(
		    "Invalid ProfilerPrintFormat string: " + std::string(format) +
		    ". Valid options are: query_tree, json, query_tree_optimizer, no_output, html, graphviz.");
	}
	auto &connection = con.GetConnection();
	py::str profiling_info = connection.GetProfilingInformation(format_enum);
	return profiling_info;
}

void DuckDBPyConnection::EnableProfiling() {
	auto query_lock = LockForQuery();
	auto &connection = con.GetConnection();
	connection.EnableProfiling();
}

void DuckDBPyConnection::DisableProfiling() {
	auto query_lock = LockForQuery();
	auto &connection = con.GetConnection();
	connection.DisableProfiling();
}

py::list DuckDBPyConnection::ExtractStatements(const string &query) {
	auto query_lock = LockForQuery();
	unique_lock<std::recursive_mutex> connection_lock;
	{
		py::gil_scoped_release release;
		connection_lock = LockConnection(py_connection_lock);
	}
	py::list result;
	auto context = con.GetConnection().context;
	auto statements = ExtractVaneStatements(*context, query);
	for (auto &statement : statements) {
		result.append(make_uniq<DuckDBPyStatement>(std::move(statement)));
	}
	return result;
}

case_insensitive_map_t<BoundParameterData> TransformPreparedParameters(const py::object &params,
                                                                       optional_ptr<PreparedStatement> prep = {});

shared_ptr<DuckDBPyConnection> DuckDBPyConnection::ExecuteMany(const py::object &query, py::object params_p) {
	PythonGILWrapper gil;
	// A generator's parameter sets can own finalizers. Drop the snapshot only
	// after releasing the cursor lock, including on conversion/execution errors.
	py::list outer_list;
	auto query_lock = LockForQuery();
	con.SetResult(nullptr);
	if (params_p.is_none()) {
		params_p = py::list();
	}

	auto statements = GetStatements(query);
	if (statements.empty()) {
		// TODO: should we throw?
		return nullptr;
	}

	auto last_statement = std::move(statements.back());
	statements.pop_back();
	// First immediately execute any preceding statements (if any)
	// FIXME: DBAPI says to not accept an 'executemany' call with multiple statements
	ExecuteImmediately(std::move(statements));

	{
		PythonInputCallbackScope callback(nullptr);
		if (!py::is_list_like(params_p)) {
			throw InvalidInputException("executemany requires a list of parameter sets to be provided");
		}
		outer_list = py::list(params_p);
	}
	if (outer_list.empty()) {
		throw InvalidInputException("executemany requires a non-empty list of parameter sets to be provided");
	}
	auto interrupt_check = CreateQueryInterruptCheck();
	unique_ptr<PreparedStatement> native_prepared;
	const bool native_execution = GetRunnerType() == "local";
	for (idx_t index = 0; index < outer_list.size(); index++) {
		unique_ptr<DuckDBPyRelation> result;
		if (native_execution && last_statement->type != StatementType::MULTI_STATEMENT) {
			auto params = py::reinterpret_borrow<py::object>(outer_list[index]);
			auto parameters =
			    TransformPreparedParameters(params.is_none() ? py::object(py::list()) : params, native_prepared.get());
			auto context = con.GetConnection().context;
			auto execution = ExecuteWithRunner(context, last_statement->Copy(), nullptr, std::move(parameters),
			                                   CreateWeakOwner(shared_from_this()), interrupt_check, true, nullptr,
			                                   &native_prepared);
			result = CreateConnectionResult(execution.TakeResult());
		} else {
			result = RunStatement(last_statement->Copy(), "", outer_list[index], true, interrupt_check);
		}
		if (result && index + 1 < outer_list.size()) {
			while (py::len(result->FetchMany(STANDARD_VECTOR_SIZE)) != 0) {
			}
		}
		con.SetResult(std::move(result));
	}

	return shared_from_this();
}

unique_ptr<QueryResult> DuckDBPyConnection::CompletePendingQuery(PendingQueryResult &pending_query) {
	PendingExecutionResult execution_result;
	if (pending_query.HasError()) {
		pending_query.ThrowError();
	}
	while (!PendingQueryResult::IsResultReady(execution_result = pending_query.ExecuteTask())) {
		{
			PythonGILWrapper gil;
			if (PyErr_CheckSignals() != 0) {
				throw std::runtime_error("Query interrupted");
			}
		}
		if (execution_result == PendingExecutionResult::BLOCKED) {
			pending_query.WaitForTask();
		}
	}
	if (execution_result == PendingExecutionResult::EXECUTION_ERROR) {
		pending_query.ThrowError();
	}
	return pending_query.Execute();
}

py::list TransformNamedParameters(const case_insensitive_map_t<idx_t> &named_param_map, const py::dict &params) {
	py::list new_params(params.size());

	for (auto &item : params) {
		const std::string &item_name = item.first.cast<std::string>();
		auto entry = named_param_map.find(item_name);
		if (entry == named_param_map.end()) {
			throw InvalidInputException(
			    "Named parameters could not be transformed, because query string is missing named parameter '%s'",
			    item_name);
		}
		auto param_idx = entry->second;
		// Add the value of the named parameter to the list
		new_params[param_idx - 1] = item.second;
	}

	if (named_param_map.size() != params.size()) {
		// One or more named parameters were expected, but not found
		vector<string> missing_params;
		missing_params.reserve(named_param_map.size());
		for (auto &entry : named_param_map) {
			auto &name = entry.first;
			if (!params.contains(name)) {
				missing_params.push_back(name);
			}
		}
		auto message = StringUtil::Join(missing_params, ", ");
		throw InvalidInputException("Not all named parameters have been located, missing: %s", message);
	}

	return new_params;
}

case_insensitive_map_t<BoundParameterData> TransformPreparedParameters(const py::object &params,
                                                                       optional_ptr<PreparedStatement> prep) {
	// Type checks, length/iteration, mapping snapshots and nested value
	// conversions may all invoke Python while the cursor lock is held.
	PythonInputCallbackScope callback(nullptr);
	case_insensitive_map_t<BoundParameterData> named_values;
	if (py::is_list_like(params)) {
		if (prep && prep->named_param_map.size() != py::len(params)) {
			if (py::len(params) == 0) {
				throw InvalidInputException("Expected %d parameters, but none were supplied",
				                            prep->named_param_map.size());
			}
			throw InvalidInputException("Prepared statement needs %d parameters, %d given",
			                            prep->named_param_map.size(), py::len(params));
		}
		auto unnamed_values = DuckDBPyConnection::TransformPythonParamList(params);
		for (idx_t i = 0; i < unnamed_values.size(); i++) {
			auto &value = unnamed_values[i];
			auto identifier = std::to_string(i + 1);
			named_values[identifier] = BoundParameterData(std::move(value));
		}
	} else if (py::is_dict_like(params)) {
		auto dict = py::cast<py::dict>(params);
		named_values = DuckDBPyConnection::TransformPythonParamDict(dict);
	} else {
		throw InvalidInputException("Prepared parameters can only be passed as a list or a dictionary");
	}
	return named_values;
}

vector<unique_ptr<SQLStatement>> DuckDBPyConnection::GetStatements(const py::object &query) {
	unique_lock<std::recursive_mutex> connection_lock;
	{
		py::gil_scoped_release release;
		connection_lock = LockConnection(py_connection_lock);
	}
	con.GetConnection();
	shared_ptr<DuckDBPyStatement> statement_obj;
	if (py::try_cast(query, statement_obj)) {
		vector<unique_ptr<SQLStatement>> result;
		result.push_back(statement_obj->GetStatement());
		return result;
	}
	if (py::isinstance<py::str>(query)) {
		auto sql_query = std::string(py::str(query));
		auto context = con.GetConnection().context;
		auto statements = ExtractVaneStatements(*context, sql_query);
		return std::move(statements);
	}
	throw InvalidInputException("Please provide either a Vane DuckDBPyStatement or a string representing the query");
}

shared_ptr<DuckDBPyConnection> DuckDBPyConnection::ExecuteFromString(const string &query) {
	return Execute(py::str(query));
}

shared_ptr<DuckDBPyConnection> DuckDBPyConnection::Execute(const py::object &query, py::object params) {
	PythonGILWrapper gil;
	auto query_lock = LockForQuery();
	con.SetResult(nullptr);
	con.SetResult(RunQueryInternal(query, "", std::move(params), true));
	return shared_from_this();
}

py::object DuckDBPyConnection::ExecuteResult(const py::object &query, const py::object &params,
                                             const py::object &delivery_timeout, bool stream, idx_t rows_per_batch) {
	if (rows_per_batch == 0) {
		throw InvalidInputException("rows_per_batch must be positive");
	}
	auto query_lock = LockForQuery();
	auto interrupt_check = CreateQueryInterruptCheck();
	auto statements = GetStatements(query);
	if (statements.size() != 1 || statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw InvalidInputException("execute_result requires exactly one read-only SELECT");
	}
	auto parameters = TransformPreparedParameters(params.is_none() ? py::object(py::list()) : params);
	PreparedStatement::VerifyParameters(parameters, statements[0]->named_param_map);
	auto context = con.GetConnection().context;
	auto execution = ExecuteWithRunner(context, std::move(statements[0]), nullptr, std::move(parameters),
	                                   py::cast(shared_from_this()), interrupt_check, stream, nullptr, nullptr,
	                                   &delivery_timeout, rows_per_batch);
	return std::move(execution.managed_result);
}

void DuckDBPyConnection::ExecuteImmediately(vector<unique_ptr<SQLStatement>> statements) {
	ExecutePrecedingStatements(std::move(statements), CreateQueryInterruptCheck());
}

void DuckDBPyConnection::ExecutePrecedingStatements(vector<unique_ptr<SQLStatement>> statements,
                                                    const py::object &interrupt_check) {
	for (auto &statement : statements) {
		if (!statement->named_param_map.empty()) {
			throw NotImplementedException("Prepared parameters are only supported for the last statement, please "
			                              "split your query up into separate calls");
		}
		auto discarded_result = RunStatement(std::move(statement), "", py::none(), true, interrupt_check);
		if (discarded_result) {
			while (py::len(discarded_result->FetchMany(STANDARD_VECTOR_SIZE)) != 0) {
			}
		}
	}
}

unique_ptr<DuckDBPyRelation> DuckDBPyConnection::RunQuery(const py::object &query, string alias, py::object params) {
	return RunQueryInternal(query, std::move(alias), std::move(params), false);
}

unique_ptr<DuckDBPyRelation> DuckDBPyConnection::RunQueryInternal(const py::object &query, string alias,
                                                                  py::object params, bool for_connection) {
	auto query_lock = LockForQuery();
	auto interrupt_check = CreateQueryInterruptCheck();
	auto statements = GetStatements(query);
	if (statements.empty()) {
		return nullptr;
	}
	auto last_statement = std::move(statements.back());
	statements.pop_back();
	ExecutePrecedingStatements(std::move(statements), interrupt_check);
	return RunStatement(std::move(last_statement), std::move(alias), std::move(params), for_connection,
	                    interrupt_check);
}

unique_ptr<DuckDBPyRelation> DuckDBPyConnection::RunStatement(unique_ptr<SQLStatement> statement, string alias,
                                                              py::object params, bool for_connection,
                                                              const py::object &interrupt_check) {
	if (statement->type == StatementType::MULTI_STATEMENT) {
		unique_lock<std::recursive_mutex> connection_lock;
		vector<unique_ptr<SQLStatement>> statements;
		{
			py::gil_scoped_release release;
			connection_lock = LockConnection(py_connection_lock);
			auto context = con.GetConnection().context;
			statements = PreprocessVaneStatement(*context, std::move(statement));
		}
		if (statements.empty()) {
			return nullptr;
		}
		auto last_statement = std::move(statements.back());
		statements.pop_back();
		ExecutePrecedingStatements(std::move(statements), interrupt_check);
		return RunStatement(std::move(last_statement), std::move(alias), std::move(params), for_connection,
		                    interrupt_check);
	}
	auto query = statement->query;
	if (alias.empty()) {
		alias = "unnamed_relation_" + StringUtil::GenerateRandomName(16);
	}
	auto parameters = TransformPreparedParameters(params.is_none() ? py::object(py::list()) : params);
	PreparedStatement::VerifyParameters(parameters, statement->named_param_map);
	// Parameter callbacks cannot reenter connection APIs. Independent control
	// threads can still cancel, so validate the interrupt before execution.
	unique_lock<std::recursive_mutex> execution_lock;
	{
		py::gil_scoped_release release;
		execution_lock = LockConnection(py_connection_lock);
	}
	auto context = con.GetConnection().context;
	interrupt_check();
	if (statement->type == StatementType::SELECT_STATEMENT) {
		shared_ptr<Relation> relation;
		try {
			py::gil_scoped_release release;
			if (for_connection) {
				// QueryRelation binds during construction. Clean up the previous
				// query first, as PendingQuery does, while retaining the relation's
				// source dependencies for the lifetime of the result stream.
				context->CancelTransaction();
			}
			auto select = unique_ptr_cast<SQLStatement, SelectStatement>(std::move(statement));
			auto original_query = select->ToString();
			if (!parameters.empty()) {
				CaptureQueryParameters(*select->node, parameters);
				select->named_param_map.clear();
			}
			relation = CreateVaneQueryRelation(context, std::move(select), alias, original_query);
		} catch (const Exception &exception) {
			ErrorData error(exception);
			context->ProcessError(error, query);
			error.Throw();
		}
		if (!for_connection) {
			return CreateRelation(std::move(relation));
		}
		auto query_relation = make_uniq<DuckDBPyRelation>(std::move(relation));
		query_relation->SetConnectionOwner(CreateWeakOwner(shared_from_this()));
		return CreateConnectionResult(query_relation->ExecuteForConnection(interrupt_check));
	}

	// Retain the execution lock until command sql() streams become relations.
	auto execution = ExecuteWithRunner(context, std::move(statement), nullptr, std::move(parameters),
	                                   CreateWeakOwner(shared_from_this()), interrupt_check, true);
	if (for_connection) {
		return CreateConnectionResult(execution.TakeResult());
	}
	if (execution.return_type != StatementReturnType::QUERY_RESULT) {
		return nullptr;
	}
	if (execution.native_result) {
		auto res = std::move(execution.native_result);
		if (res->type == QueryResultType::STREAM_RESULT) {
			py::gil_scoped_release release;
			res = res->Cast<StreamQueryResult>().Materialize();
		}
		if (res->HasError()) {
			res->ThrowError();
		}
		auto &materialized = res->Cast<MaterializedQueryResult>();
		return CreateRelation(
		    make_shared_ptr<MaterializedRelation>(context, materialized.TakeCollection(), res->names, alias));
	}
	auto result = execution.TakeResult();
	auto collection = make_uniq<ColumnDataCollection>(*context, result->GetTypes());
	while (true) {
		unique_ptr<DataChunk> chunk;
		{
			py::gil_scoped_release release;
			chunk = result->FetchChunk();
		}
		if (!chunk || chunk->size() == 0) {
			break;
		}
		collection->Append(*chunk);
	}
	return CreateRelation(
	    make_shared_ptr<MaterializedRelation>(context, std::move(collection), result->GetNames(), alias));
}

unordered_set<string> DuckDBPyConnection::GetTableNames(const string &query, bool qualified) {
	auto query_lock = LockForQuery();
	auto &connection = con.GetConnection();
	return connection.GetTableNames(query, qualified);
}

shared_ptr<DuckDBPyConnection> DuckDBPyConnection::Begin() {
	ExecuteFromString("BEGIN TRANSACTION");
	return shared_from_this();
}

shared_ptr<DuckDBPyConnection> DuckDBPyConnection::Commit() {
	auto query_lock = LockForQuery();
	auto &connection = con.GetConnection();
	if (connection.context->transaction.IsAutoCommit()) {
		return shared_from_this();
	}
	ExecuteFromString("COMMIT");
	return shared_from_this();
}

shared_ptr<DuckDBPyConnection> DuckDBPyConnection::Rollback() {
	ExecuteFromString("ROLLBACK");
	return shared_from_this();
}

shared_ptr<DuckDBPyConnection> DuckDBPyConnection::Checkpoint() {
	ExecuteFromString("CHECKPOINT");
	return shared_from_this();
}

vector<Value> DuckDBPyConnection::TransformPythonParamList(const py::handle &params) {
	PythonInputCallbackScope callback(nullptr);
	vector<Value> args;
	args.reserve(py::len(params));

	for (auto param : params) {
		args.emplace_back(TransformPythonValue(param, LogicalType::UNKNOWN, false));
	}
	return args;
}

case_insensitive_map_t<BoundParameterData> DuckDBPyConnection::TransformPythonParamDict(const py::dict &params) {
	PythonInputCallbackScope callback(nullptr);
	case_insensitive_map_t<BoundParameterData> args;

	for (auto pair : params) {
		auto &key = pair.first;
		auto &value = pair.second;
		args[std::string(py::str(key))] = BoundParameterData(TransformPythonValue(value, LogicalType::UNKNOWN, false));
	}
	return args;
}

} // namespace duckdb
