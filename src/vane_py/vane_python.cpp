// SPDX-FileCopyrightText: 2018-2026 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#include "module_bindings.hpp"
#include "datasource_function.hpp"

#include "duckdb/common/atomic.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/parser/parser.hpp"

#include "vane_python/python_objects.hpp"
#include "vane_python/datasource_execution_context.hpp"
#include "vane_python/dynamic_extension.hpp"
#include "vane_python/file.hpp"
#include "vane_python/file_reader.hpp"
#include "vane_python/image.hpp"
#include "vane_python/pyconnection/pyconnection.hpp"
#include "vane_python/pystatement.hpp"
#include "vane_python/pyrelation.hpp"
#include "vane_python/expression/pyexpression.hpp"
#include "vane_python/pyresult.hpp"
#include "vane_python/pybind11/exceptions.hpp"
#include "vane_python/typing.hpp"
#include "vane_python/functional.hpp"
#include "vane_python/pybind11/conversions/pyconnection_default.hpp"
#include "duckdb/common/box_renderer.hpp"
#include "duckdb/function/function.hpp"
#include "vane_python/pybind11/conversions/exception_handling_enum.hpp"
#include "vane_python/pybind11/conversions/python_udf_type_enum.hpp"
#include "vane_python/pybind11/conversions/python_csv_line_terminator_enum.hpp"
#include "duckdb/common/enums/statement_type.hpp"
#include "duckdb/common/adbc/adbc-init.hpp"
#include "vane_python/udf_executor.hpp"

#include "duckdb.hpp"
#include "duckdb/execution/distributed/utils/stream.hpp"
#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/execution/operator/projection/physical_udf_inout.hpp"

namespace py = pybind11;

namespace duckdb {

py::dict GetUDFExecutorDebugCounters();
void ResetUDFExecutorDebugCounters();

enum PySQLTokenType : uint8_t {
	PY_SQL_TOKEN_IDENTIFIER = 0,
	PY_SQL_TOKEN_NUMERIC_CONSTANT,
	PY_SQL_TOKEN_STRING_CONSTANT,
	PY_SQL_TOKEN_OPERATOR,
	PY_SQL_TOKEN_KEYWORD,
	PY_SQL_TOKEN_COMMENT
};

static py::list PyTokenize(const string &query) {
	auto tokens = Parser::Tokenize(query);
	py::list result;
	for (auto &token : tokens) {
		auto tuple = py::tuple(2);
		tuple[0] = token.start;
		switch (token.type) {
		case SimplifiedTokenType::SIMPLIFIED_TOKEN_IDENTIFIER:
			tuple[1] = PY_SQL_TOKEN_IDENTIFIER;
			break;
		case SimplifiedTokenType::SIMPLIFIED_TOKEN_NUMERIC_CONSTANT:
			tuple[1] = PY_SQL_TOKEN_NUMERIC_CONSTANT;
			break;
		case SimplifiedTokenType::SIMPLIFIED_TOKEN_STRING_CONSTANT:
			tuple[1] = PY_SQL_TOKEN_STRING_CONSTANT;
			break;
		case SimplifiedTokenType::SIMPLIFIED_TOKEN_OPERATOR:
			tuple[1] = PY_SQL_TOKEN_OPERATOR;
			break;
		case SimplifiedTokenType::SIMPLIFIED_TOKEN_KEYWORD:
			tuple[1] = PY_SQL_TOKEN_KEYWORD;
			break;
		case SimplifiedTokenType::SIMPLIFIED_TOKEN_COMMENT:
			tuple[1] = PY_SQL_TOKEN_COMMENT;
			break;
		default:
			break;
		}
		result.append(tuple);
	}
	return result;
}

static void RegisterStatementType(py::handle &m) {
	auto statement_type = py::enum_<duckdb::StatementType>(m, "StatementType", py::module_local());
	static const duckdb::StatementType TYPES[] = {
	    duckdb::StatementType::INVALID_STATEMENT,       duckdb::StatementType::SELECT_STATEMENT,
	    duckdb::StatementType::INSERT_STATEMENT,        duckdb::StatementType::UPDATE_STATEMENT,
	    duckdb::StatementType::CREATE_STATEMENT,        duckdb::StatementType::DELETE_STATEMENT,
	    duckdb::StatementType::PREPARE_STATEMENT,       duckdb::StatementType::EXECUTE_STATEMENT,
	    duckdb::StatementType::ALTER_STATEMENT,         duckdb::StatementType::TRANSACTION_STATEMENT,
	    duckdb::StatementType::COPY_STATEMENT,          duckdb::StatementType::ANALYZE_STATEMENT,
	    duckdb::StatementType::VARIABLE_SET_STATEMENT,  duckdb::StatementType::CREATE_FUNC_STATEMENT,
	    duckdb::StatementType::EXPLAIN_STATEMENT,       duckdb::StatementType::DROP_STATEMENT,
	    duckdb::StatementType::EXPORT_STATEMENT,        duckdb::StatementType::PRAGMA_STATEMENT,
	    duckdb::StatementType::VACUUM_STATEMENT,        duckdb::StatementType::CALL_STATEMENT,
	    duckdb::StatementType::SET_STATEMENT,           duckdb::StatementType::LOAD_STATEMENT,
	    duckdb::StatementType::RELATION_STATEMENT,      duckdb::StatementType::EXTENSION_STATEMENT,
	    duckdb::StatementType::LOGICAL_PLAN_STATEMENT,  duckdb::StatementType::ATTACH_STATEMENT,
	    duckdb::StatementType::DETACH_STATEMENT,        duckdb::StatementType::MULTI_STATEMENT,
	    duckdb::StatementType::COPY_DATABASE_STATEMENT, duckdb::StatementType::MERGE_INTO_STATEMENT};
	static const idx_t AMOUNT = sizeof(TYPES) / sizeof(duckdb::StatementType);
	for (idx_t i = 0; i < AMOUNT; i++) {
		auto &type = TYPES[i];
		statement_type.value(StatementTypeToString(type).c_str(), type);
	}
	statement_type.export_values();
}

static void RegisterExpectedResultType(py::handle &m) {
	auto expected_return_type = py::enum_<duckdb::StatementReturnType>(m, "ExpectedResultType", py::module_local());
	static const duckdb::StatementReturnType TYPES[] = {duckdb::StatementReturnType::QUERY_RESULT,
	                                                    duckdb::StatementReturnType::CHANGED_ROWS,
	                                                    duckdb::StatementReturnType::NOTHING};
	static const idx_t AMOUNT = sizeof(TYPES) / sizeof(duckdb::StatementReturnType);
	for (idx_t i = 0; i < AMOUNT; i++) {
		auto &type = TYPES[i];
		expected_return_type.value(StatementReturnTypeToString(type).c_str(), type);
	}
	expected_return_type.export_values();
}

// ######################################################################
// Symbol exports
//
// We want to limit the symbols we export to only the absolute minimum.
// This means we compile with -fvisibility=hidden to hide all symbols,
// and then explicitly export only the symbols we want.
//
// Right now we export two Vane-owned symbols only:
// - vane_adbc_init: the entrypoint for our ADBC driver
// - PyInit__native: the entrypoint for the Python extension
//
// All symbols that need exporting must be referenced below and listed in
// CMakeLists.txt.
extern "C" {
AdbcStatusCode vane_adbc_init(int version, void *driver, struct AdbcError *error);
};

PYBIND11_MODULE(_native, m) { // NOLINT
	// DO NOT REMOVE: the below forces that we include all symbols we want to export
	auto *volatile keep_alive = &vane_adbc_init;
	(void)keep_alive;
	// END

	m.def("_datasource_factory_registry_state_for_test", &duckdb::DataSourceFactoryRegistryStateForTest);
	m.def("_clear_datasource_factory_registry", &duckdb::ClearDataSourceFactoryRegistry);
	m.def("_release_datasource_factories_for_query", &duckdb::ReleaseDataSourceFactoriesForQuery, py::arg("query_id"));

	py::enum_<duckdb::ExplainType>(m, "ExplainType", py::module_local())
	    .value("STANDARD", duckdb::ExplainType::EXPLAIN_STANDARD)
	    .value("ANALYZE", duckdb::ExplainType::EXPLAIN_ANALYZE)
	    .export_values();

	RegisterStatementType(m);
	RegisterExpectedResultType(m);
	PythonDataSourceExecutionContext::Initialize(m);

	// Expose experimental Ray bindings as ``vane._native.ray_cxx``.
	extern void register_ray_bindings(py::module_ & m);
	register_ray_bindings(m);

	// Expose runner lifecycle functions directly on ``vane._native``.
	extern void register_vane_runners(py::module_ & m);
	register_vane_runners(m);

	py::enum_<duckdb::PythonCSVLineTerminator::Type>(m, "CSVLineTerminator", py::module_local())
	    .value("LINE_FEED", duckdb::PythonCSVLineTerminator::Type::LINE_FEED)
	    .value("CARRIAGE_RETURN_LINE_FEED", duckdb::PythonCSVLineTerminator::Type::CARRIAGE_RETURN_LINE_FEED)
	    .export_values();

	py::enum_<duckdb::PythonExceptionHandling>(m, "PythonExceptionHandling", py::module_local())
	    .value("DEFAULT", duckdb::PythonExceptionHandling::FORWARD_ERROR)
	    .value("RETURN_NULL", duckdb::PythonExceptionHandling::RETURN_NULL)
	    .export_values();

	py::enum_<duckdb::RenderMode>(m, "RenderMode", py::module_local())
	    .value("ROWS", duckdb::RenderMode::ROWS)
	    .value("COLUMNS", duckdb::RenderMode::COLUMNS)
	    .export_values();

	PythonFile::Initialize(m);
	PythonFileReaderHandle::Initialize(m);
	DuckDBPyTyping::Initialize(m);
	DuckDBPyFunctional::Initialize(m);
	DuckDBPyExpression::Initialize(m);
	DuckDBPyStatement::Initialize(m);
	DuckDBPyRelation::Initialize(m);
	DuckDBPyConnection::Initialize(m);
	InitializeDynamicExtensionBindings(m);
	PythonObject::Initialize();
	RegisterUDFExternalBlockBackend();
	m.def("_udf_executor_debug_counters", &GetUDFExecutorDebugCounters);
	m.def("_reset_udf_executor_debug_counters", &ResetUDFExecutorDebugCounters);
	m.def("_shutdown_udf_executor_dispatcher", &ShutdownUDFExecutorDispatcher);
	m.def("_wake_udf_executor_slots_for_testing", &WakeUDFExecutorSlotsForTesting);
	m.def("_test_streaming_output_capacity_publication", &TestStreamingOutputCapacityPublication,
	      py::call_guard<py::gil_scoped_release>());
	m.def("_pause_udf_dispatcher_wait_for_testing", &PauseUDFDispatcherWaitForTesting);
	try {
		py::module_::import("atexit").attr("register")(m.attr("_shutdown_udf_executor_dispatcher"));
	} catch (const py::error_already_set &) {
		// Explicit runner/query teardown still cleans active slots. This hook is
		// the final guard that joins the native dispatcher before Python state is
		// destroyed.
	}

	py::options pybind_opts;

	m.doc() = "Vane is a distributed, multimodal data engine";
	m.attr("__package__") = "vane";
	m.attr("__version__") = std::string(DuckDB::LibraryVersion()).substr(1);
	m.attr("__standard_vector_size__") = DuckDB::StandardVectorSize();
	m.attr("__git_revision__") = DuckDB::SourceID();
	m.attr("__interactive__") = DuckDBPyConnection::DetectAndGetEnvironment();
	m.attr("__jupyter__") = DuckDBPyConnection::IsJupyter();
	m.attr("__formatted_python_version__") = DuckDBPyConnection::FormattedPythonVersion();
	m.def("default_connection", &DuckDBPyConnection::DefaultConnection,
	      "Retrieve the connection currently registered as the default to be used by the module");
	m.def("set_default_connection", &DuckDBPyConnection::SetDefaultConnection,
	      "Register the provided connection as the default to be used by the module",
	      py::arg("connection").none(false));
	m.attr("apilevel") = "2.0";
	m.attr("threadsafety") = 1;
	m.attr("paramstyle") = "qmark";

	InitializeModuleConnectionBindings(m);
	InitializeModuleTypesBindings(m);
	InitializeModuleExecutionBindings(m);
	InitializeModuleResultsBindings(m);
	InitializeModuleTransactionsBindings(m);
	InitializeModuleRelationsBindings(m);
	InitializeModuleExtensionsBindings(m);
	InitializeModuleRelationalOperationsBindings(m);
	InitializeModuleResultPropertiesBindings(m);

	RegisterExceptions(m);

	m.def("connect", &DuckDBPyConnection::Connect,
	      "Create a DuckDB database instance. Can take a database file name to read/write persistent data and a "
	      "read_only flag if no changes are desired",
	      py::arg("database") = ":memory:", py::arg("read_only") = false, py::arg_v("config", py::dict(), "None"));
	m.def(
	    "_connect_with_runner",
	    [](const string &runner_type) {
		    return DuckDBPyConnection::ConnectWithRunner(py::str(":memory:"), false, py::dict(), runner_type);
	    },
	    py::arg("runner_type"));
	m.def("tokenize", PyTokenize,
	      "Tokenizes a SQL string, returning a list of (position, type) tuples that can be "
	      "used for e.g., syntax highlighting",
	      py::arg("query"));
	py::enum_<PySQLTokenType>(m, "token_type", py::module_local())
	    .value("identifier", PySQLTokenType::PY_SQL_TOKEN_IDENTIFIER)
	    .value("numeric_const", PySQLTokenType::PY_SQL_TOKEN_NUMERIC_CONSTANT)
	    .value("string_const", PySQLTokenType::PY_SQL_TOKEN_STRING_CONSTANT)
	    .value("operator", PySQLTokenType::PY_SQL_TOKEN_OPERATOR)
	    .value("keyword", PySQLTokenType::PY_SQL_TOKEN_KEYWORD)
	    .value("comment", PySQLTokenType::PY_SQL_TOKEN_COMMENT)
	    .export_values();

	// we need this because otherwise we try to remove registered_dfs on shutdown when python is already dead
	auto clean_default_connection = []() {
		DuckDBPyConnection::Cleanup();
	};
	m.add_object("_clean_default_connection", py::capsule(clean_default_connection));
}

} // namespace duckdb
