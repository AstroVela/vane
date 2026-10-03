// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/pyconnection/pyconnection.hpp"
#include "vane_python/pyresult.hpp"
#include "vane_python/python_replacement_scan.hpp"
#include "vane_python/python_udf_actor_resources.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/pending_query_result.hpp"
#include "duckdb/main/prepared_statement.hpp"

#include <pybind11/stl.h>

namespace duckdb {

case_insensitive_map_t<BoundParameterData> TransformPreparedParameters(const py::object &params,
                                                                       optional_ptr<PreparedStatement> prep);

shared_ptr<DuckDBPyConnection> DuckDBPyConnection::ConnectQuery(const py::object &database, bool read_only,
                                                                const py::dict &config, const py::kwargs &options) {
	CheckCallbackEntry();
	if (options.empty()) {
		return Connect(database, read_only, config);
	}
	for (auto item : options) {
		auto name = py::cast<string>(item.first);
		if (name != "backend" && name != "resources") {
			throw py::value_error("local connections do not accept execution or unknown options");
		}
	}
	if (!options.contains("backend") || !py::isinstance<py::str>(options["backend"]) ||
	    py::cast<string>(options["backend"]) != "local") {
		throw py::value_error("the query API currently requires backend='local'");
	}
	auto resources = options.contains("resources") ? py::reinterpret_borrow<py::object>(options["resources"])
	                                               : py::object(py::none());
	auto runtime = py::module_::import("vane.execution.query_runtime").attr("QueryRuntime")(resources);
	auto connection = ConnectWithRunner(database, read_only, config, "local-fast");
	EnableLocalRuntimeInputPolicy(*connection->con.GetConnection().context);
	connection->vane_session->query_runtime = std::move(runtime);
	return connection;
}

py::object DuckDBPyConnection::GetQueryRuntime() const {
	CheckCallbackEntry();
	if (!vane_session || !vane_session_attached) {
		return py::none();
	}
	lock_guard<mutex> guard(vane_session->lock);
	return vane_session->query_runtime;
}

py::object DuckDBPyConnection::Query(const py::object &sql, const py::object &parameters, const py::object &options,
                                     const py::object &rows_per_batch, const py::kwargs &overrides) {
	auto lock = LockForQuery();
	auto runtime = GetQueryRuntime();
	if (runtime.is_none()) {
		throw InvalidInputException("query() requires a connection created with backend='local'");
	}
	if (local_query_closing) {
		throw ConnectionException("Connection is closing");
	}
	py::object snapshot;
	{
		PythonInputCallbackScope callback(nullptr);
		snapshot = runtime.attr("options")(options, rows_per_batch, overrides);
	}
	const auto batch_size = rows_per_batch.cast<idx_t>();
	auto context = con.GetConnection().context;
	if (!context->transaction.IsAutoCommit()) {
		throw InvalidInputException("query() requires auto-commit mode");
	}
	auto statements = GetStatements(sql);
	if (statements.size() != 1 || statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw InvalidInputException("query() requires exactly one SELECT; use execute() for commands");
	}
	auto native_parameters =
	    TransformPreparedParameters(parameters.is_none() ? py::object(py::list()) : parameters, nullptr);
	PreparedStatement::VerifyParameters(native_parameters, statements[0]->named_param_map);
	con.SetResult(nullptr);
	auto interrupt_check = CreateQueryInterruptCheck();
	const auto generation = InterruptGeneration();
	auto weak_source = weak_ptr<DuckDBPyConnection>(shared_from_this());
	auto publish = py::cpp_function([weak_source, generation](py::object query) {
		if (auto source = weak_source.lock()) {
			source->local_query_request = query;
			source->local_query_thread = query.is_none() ? std::thread::id() : std::this_thread::get_id();
			if (!query.is_none() && (source->local_query_closing || source->InterruptGeneration() != generation)) {
				query.attr("cancel")();
			}
		}
	});
	ScopedPythonReplacementScanFrame caller_frame(*context);
	auto operation = py::cpp_function([&](py::object query) {
		struct NativeCallGuard {
			DuckDBPyConnection &connection;
			~NativeCallGuard() {
				connection.local_query_thread = std::thread::id();
			}
		} native_call_guard {*this};
		auto cursor_lock = py_connection_lock;
		auto close_native = py::cpp_function([context, cursor_lock, weak_source, query](bool retire) {
			if (retire) {
				if (auto source = weak_source.lock()) {
					if (source->local_query_request.is(query)) {
						source->local_query_request = py::none();
						source->local_query_stream = py::none();
						source->local_query_thread = std::thread::id();
					}
				}
			} else {
				auto guard = LockConnection(cursor_lock);
				py::gil_scoped_release release;
				context->CancelTransaction();
			}
		});
		auto guard_native = py::cpp_function([weak_source]() {
			CheckCallbackEntry();
			if (auto source = weak_source.lock()) {
				source->CheckLocalQueryCloseReentrancy();
			}
		});
		// Install the cleanup owner before native preparation can fail.
		query.attr("install_cleanup")(close_native, guard_native);
		interrupt_check();
		ScopedPythonUDFActorResourcePreparation preparation(*context, query);
		unique_ptr<PendingQueryResult> pending;
		{
			py::gil_scoped_release release;
			PendingQueryParameters input;
			input.parameters = native_parameters;
			input.query_parameters = true;
			pending = context->PendingQuery(std::move(statements[0]), input);
		}
		query.attr("started")(py::cpp_function([context]() { context->Interrupt(); }));
		unique_ptr<QueryResult> result;
		{
			py::gil_scoped_release release;
			result = CompletePendingQuery(*pending);
			if (result->HasError()) {
				result->ThrowError();
			}
		}
		auto native = make_uniq<DuckDBPyResult>(std::move(result));
		py::dict schema;
		schema["names"] = py::cast(native->GetNames());
		py::list types;
		for (auto &type : native->GetTypes()) {
			types.append(type.ToString());
		}
		schema["types"] = std::move(types);
		native->SetConnectionLock(cursor_lock, context);
		py::object reader = native->FetchRecordBatchReader(batch_size);
		auto read_native = py::cpp_function([reader, weak_source]() {
			CheckCallbackEntry();
			auto source = weak_source.lock();
			if (!source) {
				throw ConnectionException("Query result's connection is closed");
			}
			auto guard = LockConnection(source->py_connection_lock);
			source->local_query_thread = std::this_thread::get_id();
			try {
				auto batch = reader.attr("read_next_batch")();
				source->local_query_thread = std::thread::id();
				return batch;
			} catch (...) {
				source->local_query_thread = std::thread::id();
				throw;
			}
		});
		query.attr("install_reader")(reader, read_native, schema);
		auto managed = query.attr("_result");
		local_query_stream = py::module_::import("weakref").attr("ref")(managed);
		local_query_thread = std::thread::id();
	});
	return runtime.attr("run")(operation, publish, snapshot);
}

} // namespace duckdb
