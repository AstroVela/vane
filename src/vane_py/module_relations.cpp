// SPDX-FileCopyrightText: 2018-2026 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#include "module_bindings.hpp"
#include "vane_python/pyconnection/pyconnection.hpp"
#include "vane_python/pybind11/conversions/pyconnection_default.hpp"
#include "vane_python/pyrelation.hpp"

namespace duckdb {

void InitializeModuleRelationsBindings(py::module_ &m) {
	m.def(
	    "append",
	    [](const string &name, const PandasDataFrame &value, bool by_name,
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Append(name, value, by_name);
	    },
	    "Append the passed DataFrame to the named table", py::arg("table_name"), py::arg("df"), py::kw_only(),
	    py::arg("by_name") = false, py::arg("connection") = py::none());
	m.def(
	    "register",
	    [](const string &name, const py::object &python_object, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->RegisterPythonObject(name, python_object);
	    },
	    "Register the passed Python Object value for querying with a view", py::arg("view_name"),
	    py::arg("python_object"), py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "unregister",
	    [](const string &name, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->UnregisterPythonObject(name);
	    },
	    "Unregister the view name", py::arg("view_name"), py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "table",
	    [](const string &tname, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Table(tname);
	    },
	    "Create a relation object for the named table", py::arg("table_name"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "view",
	    [](const string &vname, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->View(vname);
	    },
	    "Create a relation object for the named view", py::arg("view_name"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "values",
	    [](const py::args &params, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Values(params);
	    },
	    "Create a relation object from the passed values", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "table_function",
	    [](const string &fname, py::object params = py::list(), shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->TableFunction(fname, params);
	    },
	    "Create a relation object from the named table function with given parameters", py::arg("name"),
	    py::arg("parameters") = py::none(), py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "read_json",
	    [](const py::object &name, const Optional<py::object> &columns = py::none(),
	       const Optional<py::object> &sample_size = py::none(), const Optional<py::object> &maximum_depth = py::none(),
	       const Optional<py::str> &records = py::none(), const Optional<py::str> &format = py::none(),
	       const Optional<py::object> &date_format = py::none(),
	       const Optional<py::object> &timestamp_format = py::none(),
	       const Optional<py::object> &compression = py::none(),
	       const Optional<py::object> &maximum_object_size = py::none(),
	       const Optional<py::object> &ignore_errors = py::none(),
	       const Optional<py::object> &convert_strings_to_integers = py::none(),
	       const Optional<py::object> &field_appearance_threshold = py::none(),
	       const Optional<py::object> &map_inference_threshold = py::none(),
	       const Optional<py::object> &maximum_sample_files = py::none(),
	       const Optional<py::object> &filename = py::none(),
	       const Optional<py::object> &hive_partitioning = py::none(),
	       const Optional<py::object> &union_by_name = py::none(), const Optional<py::object> &hive_types = py::none(),
	       const Optional<py::object> &hive_types_autocast = py::none(),
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->ReadJSON(name, columns, sample_size, maximum_depth, records, format, date_format,
		                          timestamp_format, compression, maximum_object_size, ignore_errors,
		                          convert_strings_to_integers, field_appearance_threshold, map_inference_threshold,
		                          maximum_sample_files, filename, hive_partitioning, union_by_name, hive_types,
		                          hive_types_autocast);
	    },
	    "Create a relation object from the JSON file in 'name'", py::arg("path_or_buffer"), py::kw_only(),
	    py::arg("columns") = py::none(), py::arg("sample_size") = py::none(), py::arg("maximum_depth") = py::none(),
	    py::arg("records") = py::none(), py::arg("format") = py::none(), py::arg("date_format") = py::none(),
	    py::arg("timestamp_format") = py::none(), py::arg("compression") = py::none(),
	    py::arg("maximum_object_size") = py::none(), py::arg("ignore_errors") = py::none(),
	    py::arg("convert_strings_to_integers") = py::none(), py::arg("field_appearance_threshold") = py::none(),
	    py::arg("map_inference_threshold") = py::none(), py::arg("maximum_sample_files") = py::none(),
	    py::arg("filename") = py::none(), py::arg("hive_partitioning") = py::none(),
	    py::arg("union_by_name") = py::none(), py::arg("hive_types") = py::none(),
	    py::arg("hive_types_autocast") = py::none(), py::arg("connection") = py::none());
	m.def(
	    "extract_statements",
	    [](const string &query, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->ExtractStatements(query);
	    },
	    "Parse the query string and extract the Statement object(s) produced", py::arg("query"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "sql",
	    [](const py::object &query, string alias = "", py::object params = py::list(),
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->RunQuery(query, alias, params);
	    },
	    "Create a lazy relation for SELECT, capturing positional or named params for execution with the configured "
	    "connection runner when consumed. Writes execute immediately through the same bound-plan entry; "
	    "connection and catalog operations execute on the client.",
	    py::arg("query"), py::kw_only(), py::arg("alias") = "", py::arg("params") = py::none(),
	    py::arg("connection") = py::none());
	m.def(
	    "query",
	    [](const py::object &query, string alias = "", py::object params = py::list(),
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->RunQuery(query, alias, params);
	    },
	    "Create a lazy relation for SELECT, capturing positional or named params for execution with the configured "
	    "connection runner when consumed. Writes execute immediately through the same bound-plan entry; "
	    "connection and catalog operations execute on the client.",
	    py::arg("query"), py::kw_only(), py::arg("alias") = "", py::arg("params") = py::none(),
	    py::arg("connection") = py::none());
	m.def(
	    "from_query",
	    [](const py::object &query, string alias = "", py::object params = py::list(),
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->RunQuery(query, alias, params);
	    },
	    "Create a lazy relation for SELECT, capturing positional or named params for execution with the configured "
	    "connection runner when consumed. Writes execute immediately through the same bound-plan entry; "
	    "connection and catalog operations execute on the client.",
	    py::arg("query"), py::kw_only(), py::arg("alias") = "", py::arg("params") = py::none(),
	    py::arg("connection") = py::none());
	m.def(
	    "read_csv",
	    [](const py::object &name, py::kwargs &kwargs) {
		    auto connection_arg = kwargs.contains("conn") ? kwargs["conn"] : py::none();
		    auto conn = py::cast<shared_ptr<DuckDBPyConnection>>(connection_arg);

		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->ReadCSV(name, kwargs);
	    },
	    "Create a relation object from the CSV file in 'name'", py::arg("path_or_buffer"), py::kw_only());
	m.def(
	    "from_csv_auto",
	    [](const py::object &name, py::kwargs &kwargs) {
		    auto connection_arg = kwargs.contains("conn") ? kwargs["conn"] : py::none();
		    auto conn = py::cast<shared_ptr<DuckDBPyConnection>>(connection_arg);

		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->ReadCSV(name, kwargs);
	    },
	    "Create a relation object from the CSV file in 'name'", py::arg("path_or_buffer"), py::kw_only());
	m.def(
	    "from_df",
	    [](const PandasDataFrame &value, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromDF(value);
	    },
	    "Create a relation object from the DataFrame in df", py::arg("df"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "from_arrow",
	    [](py::object &arrow_object, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromArrow(arrow_object);
	    },
	    "Create a relation object from an Arrow object", py::arg("arrow_object"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "from_parquet",
	    [](const string &file_glob, bool binary_as_string, bool file_row_number, bool filename, bool hive_partitioning,
	       bool union_by_name, const py::object &compression = py::none(),
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromParquet(file_glob, binary_as_string, file_row_number, filename, hive_partitioning,
		                             union_by_name, compression);
	    },
	    "Create a relation object from the Parquet files in file_glob", py::arg("file_glob"),
	    py::arg("binary_as_string") = false, py::kw_only(), py::arg("file_row_number") = false,
	    py::arg("filename") = false, py::arg("hive_partitioning") = false, py::arg("union_by_name") = false,
	    py::arg("compression") = py::none(), py::arg("connection") = py::none());
	m.def(
	    "read_parquet",
	    [](const string &file_glob, bool binary_as_string, bool file_row_number, bool filename, bool hive_partitioning,
	       bool union_by_name, const py::object &compression = py::none(),
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromParquet(file_glob, binary_as_string, file_row_number, filename, hive_partitioning,
		                             union_by_name, compression);
	    },
	    "Create a relation object from the Parquet files in file_glob", py::arg("file_glob"),
	    py::arg("binary_as_string") = false, py::kw_only(), py::arg("file_row_number") = false,
	    py::arg("filename") = false, py::arg("hive_partitioning") = false, py::arg("union_by_name") = false,
	    py::arg("compression") = py::none(), py::arg("connection") = py::none());
	m.def(
	    "from_parquet",
	    [](const vector<string> &file_globs, bool binary_as_string, bool file_row_number, bool filename,
	       bool hive_partitioning, bool union_by_name, const py::object &compression = py::none(),
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromParquets(file_globs, binary_as_string, file_row_number, filename, hive_partitioning,
		                              union_by_name, compression);
	    },
	    "Create a relation object from the Parquet files in file_globs", py::arg("file_globs"),
	    py::arg("binary_as_string") = false, py::kw_only(), py::arg("file_row_number") = false,
	    py::arg("filename") = false, py::arg("hive_partitioning") = false, py::arg("union_by_name") = false,
	    py::arg("compression") = py::none(), py::arg("connection") = py::none());
	m.def(
	    "read_parquet",
	    [](const vector<string> &file_globs, bool binary_as_string, bool file_row_number, bool filename,
	       bool hive_partitioning, bool union_by_name, const py::object &compression = py::none(),
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromParquets(file_globs, binary_as_string, file_row_number, filename, hive_partitioning,
		                              union_by_name, compression);
	    },
	    "Create a relation object from the Parquet files in file_globs", py::arg("file_globs"),
	    py::arg("binary_as_string") = false, py::kw_only(), py::arg("file_row_number") = false,
	    py::arg("filename") = false, py::arg("hive_partitioning") = false, py::arg("union_by_name") = false,
	    py::arg("compression") = py::none(), py::arg("connection") = py::none());
	m.def(
	    "get_table_names",
	    [](const string &query, bool qualified, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->GetTableNames(query, qualified);
	    },
	    "Extract the required table names from a query", py::arg("query"), py::kw_only(), py::arg("qualified") = false,
	    py::arg("connection") = py::none());
}

void InitializeModuleRelationalOperationsBindings(py::module_ &m) {
	m.def(
	    "project",
	    [](const PandasDataFrame &df, const py::args &args, const string &groups = "",
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromDF(df)->Project(args, groups);
	    },
	    "Project the relation object by the projection in project_expr", py::arg("df"), py::kw_only(),
	    py::arg("groups") = "", py::arg("connection") = py::none());
	m.def(
	    "distinct",
	    [](const PandasDataFrame &df, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromDF(df)->Distinct();
	    },
	    "Retrieve distinct rows from this relation object", py::arg("df"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "write_csv",
	    [](const PandasDataFrame &df, const string &filename, const py::object &sep = py::none(),
	       const py::object &na_rep = py::none(), const py::object &header = py::none(),
	       const py::object &quotechar = py::none(), const py::object &escapechar = py::none(),
	       const py::object &date_format = py::none(), const py::object &timestamp_format = py::none(),
	       const py::object &quoting = py::none(), const py::object &encoding = py::none(),
	       const py::object &compression = py::none(), const py::object &overwrite = py::none(),
	       const py::object &per_thread_output = py::none(), const py::object &use_tmp_file = py::none(),
	       const py::object &partition_by = py::none(), const py::object &write_partition_columns = py::none(),
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->FromDF(df)->ToCSV(filename, sep, na_rep, header, quotechar, escapechar, date_format, timestamp_format,
		                            quoting, encoding, compression, overwrite, per_thread_output, use_tmp_file,
		                            partition_by, write_partition_columns);
	    },
	    "Write the relation object to a CSV file in 'file_name'", py::arg("df"), py::arg("filename"), py::kw_only(),
	    py::arg("sep") = py::none(), py::arg("na_rep") = py::none(), py::arg("header") = py::none(),
	    py::arg("quotechar") = py::none(), py::arg("escapechar") = py::none(), py::arg("date_format") = py::none(),
	    py::arg("timestamp_format") = py::none(), py::arg("quoting") = py::none(), py::arg("encoding") = py::none(),
	    py::arg("compression") = py::none(), py::arg("overwrite") = py::none(),
	    py::arg("per_thread_output") = py::none(), py::arg("use_tmp_file") = py::none(),
	    py::arg("partition_by") = py::none(), py::arg("write_partition_columns") = py::none(),
	    py::arg("connection") = py::none());
	m.def(
	    "aggregate",
	    [](const PandasDataFrame &df, const py::object &expr, const string &groups = "",
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromDF(df)->Aggregate(expr, groups);
	    },
	    "Compute the aggregate aggr_expr by the optional groups group_expr on the relation", py::arg("df"),
	    py::arg("aggr_expr"), py::arg("group_expr") = "", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "alias",
	    [](const PandasDataFrame &df, const string &expr, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromDF(df)->SetAlias(expr);
	    },
	    "Rename the relation object to new alias", py::arg("df"), py::arg("alias"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "filter",
	    [](const PandasDataFrame &df, const py::object &expr, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromDF(df)->Filter(expr);
	    },
	    "Filter the relation object by the filter in filter_expr", py::arg("df"), py::arg("filter_expr"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "limit",
	    [](const PandasDataFrame &df, int64_t n, int64_t offset = 0, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromDF(df)->Limit(n, offset);
	    },
	    "Only retrieve the first n rows from this relation object, starting at offset", py::arg("df"), py::arg("n"),
	    py::arg("offset") = 0, py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "order",
	    [](const PandasDataFrame &df, const string &expr, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromDF(df)->Order(expr);
	    },
	    "Reorder the relation object by order_expr", py::arg("df"), py::arg("order_expr"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "query_df",
	    [](const PandasDataFrame &df, const string &view_name, const string &sql_query,
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FromDF(df)->Query(view_name, sql_query);
	    },
	    "Run the given SQL query in sql_query on the view named virtual_table_name that refers to the relation object",
	    py::arg("df"), py::arg("virtual_table_name"), py::arg("sql_query"), py::kw_only(),
	    py::arg("connection") = py::none());
}

} // namespace duckdb
