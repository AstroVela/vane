// SPDX-FileCopyrightText: 2018-2026 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#include "vane_python/pyconnection/pyconnection.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/relation/value_relation.hpp"
#include "duckdb/main/relation/view_relation.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/keyword_helper.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "vane_python/expression/pyexpression.hpp"
#include "vane_python/pyresult.hpp"
#include "vane_python/python_conversion.hpp"
#include "vane_python/python_input_callback.hpp"
#include "vane_python/python_replacement_scan.hpp"

#include "vane_python/pybind11/gil_wrapper.hpp"

namespace duckdb {

unique_ptr<DuckDBPyRelation> DuckDBPyConnection::CreateRelation(shared_ptr<Relation> rel) {
	PythonGILWrapper gil;
	// DuckDBPyRelation owns Python objects, so construction must hold the GIL.
	auto py_rel = make_uniq<DuckDBPyRelation>(std::move(rel));
	py_rel->SetConnectionOwner(py::cast(shared_from_this()));
	return py_rel;
}

unique_ptr<DuckDBPyRelation> DuckDBPyConnection::CreateRelation(shared_ptr<DuckDBPyResult> result) {
	PythonGILWrapper gil;
	// DuckDBPyRelation owns Python objects, so construction must hold the GIL.
	auto py_rel = make_uniq<DuckDBPyRelation>(std::move(result));
	py_rel->SetConnectionOwner(py::cast(shared_from_this()));
	return py_rel;
}

unique_ptr<DuckDBPyRelation> DuckDBPyConnection::CreateConnectionResult(shared_ptr<DuckDBPyResult> result) {
	PythonGILWrapper gil;
	auto py_rel = make_uniq<DuckDBPyRelation>(std::move(result));
	// Every connection-owned result, including prepared executemany results,
	// needs the cursor lock when exported as a live Arrow stream. A weak owner
	// avoids a connection -> result -> connection reference cycle.
	py_rel->SetConnectionOwner(CreateWeakOwner(shared_from_this()));
	return py_rel;
}

shared_ptr<DuckDBPyConnection> DuckDBPyConnection::Append(const string &name, const PandasDataFrame &value,
                                                          bool by_name) {
	RegisterPythonObject("__append_df", value);
	string columns = "";
	if (by_name) {
		auto df_columns = value.attr("columns");
		vector<string> column_names;
		for (auto &column : df_columns) {
			column_names.push_back(std::string(py::str(column)));
		}
		columns += "(";
		for (idx_t i = 0; i < column_names.size(); i++) {
			auto &column = column_names[i];
			if (i != 0) {
				columns += ", ";
			}
			columns += StringUtil::Format("%s", SQLIdentifier(column));
		}
		columns += ")";
	}

	auto sql_query = StringUtil::Format("INSERT INTO %s %s SELECT * FROM __append_df", SQLIdentifier(name), columns);
	return Execute(py::str(sql_query));
}

shared_ptr<DuckDBPyConnection> DuckDBPyConnection::RegisterPythonObject(const string &name,
                                                                        const py::object &python_object) {
	auto query_lock = LockForQuery();
	auto &connection = con.GetConnection();
	auto &client = *connection.context;
	auto object = PythonReplacementScan::ReplacementObject(python_object, name, client);
	auto view_rel = make_shared_ptr<ViewRelation>(connection.context, std::move(object), name);
	bool replace = registered_objects.count(name);
	view_rel->CreateView(name, replace, true);
	registered_objects.insert(name);
	return shared_from_this();
}

unique_ptr<DuckDBPyRelation> DuckDBPyConnection::Table(const string &tname) {
	auto query_lock = LockForQuery();
	auto &connection = con.GetConnection();
	auto qualified_name = QualifiedName::Parse(tname);
	if (qualified_name.schema.empty()) {
		qualified_name.schema = DEFAULT_SCHEMA;
	}
	try {
		return CreateRelation(connection.Table(qualified_name.catalog, qualified_name.schema, qualified_name.name));
	} catch (const CatalogException &) {
		// CatalogException will be of the type '... is not a table'
		// Not a table in the database, make a query relation that can perform replacement scans
		auto sql_query = StringUtil::Format("from %s", KeywordHelper::WriteOptionallyQuoted(tname));
		return RunQuery(py::str(sql_query), tname);
	}
}

static vector<unique_ptr<ParsedExpression>> ValueListFromExpressions(const py::args &expressions) {
	vector<unique_ptr<ParsedExpression>> result;
	auto arg_count = expressions.size();
	if (arg_count == 0) {
		throw InvalidInputException("Please provide a non-empty tuple");
	}

	for (idx_t i = 0; i < arg_count; i++) {
		py::handle arg = expressions[i];
		shared_ptr<DuckDBPyExpression> py_expr;
		if (!py::try_cast<shared_ptr<DuckDBPyExpression>>(arg, py_expr)) {
			throw InvalidInputException("Please provide arguments of type Expression!");
		}
		auto expr = py_expr->GetExpression().Copy();
		result.push_back(std::move(expr));
	}
	return result;
}

static vector<vector<unique_ptr<ParsedExpression>>> ValueListsFromTuples(const py::args &tuples) {
	PythonInputCallbackScope callback(nullptr);
	auto arg_count = tuples.size();
	if (arg_count == 0) {
		throw InvalidInputException("Please provide a non-empty tuple");
	}

	idx_t expected_length = 0;
	vector<vector<unique_ptr<ParsedExpression>>> result;
	for (idx_t i = 0; i < arg_count; i++) {
		py::handle arg = tuples[i];
		if (!py::isinstance<py::tuple>(arg)) {
			string actual_type = py::str(py::type::of(arg));
			throw InvalidInputException("Expected objects of type tuple, not %s", actual_type);
		}
		auto expressions = py::cast<py::args>(arg);
		auto value_list = ValueListFromExpressions(expressions);
		if (i && value_list.size() != expected_length) {
			throw InvalidInputException("Mismatch between length of tuples in input, expected %d but found %d",
			                            expected_length, value_list.size());
		}
		expected_length = value_list.size();
		result.push_back(std::move(value_list));
	}
	return result;
}

unique_ptr<DuckDBPyRelation> DuckDBPyConnection::Values(const py::args &args) {
	auto query_lock = LockForQuery();
	auto &connection = con.GetConnection();

	auto arg_count = args.size();
	if (arg_count == 0) {
		throw InvalidInputException("Could not create a ValueRelation without any inputs");
	}

	D_ASSERT(py::gil_check());
	py::handle first_arg = args[0];
	if (arg_count == 1 && py::isinstance<py::list>(first_arg)) {
		vector<vector<Value>> values {DuckDBPyConnection::TransformPythonParamList(first_arg)};
		return CreateRelation(connection.Values(values));
	} else {
		vector<vector<unique_ptr<ParsedExpression>>> expressions;
		if (py::isinstance<py::tuple>(first_arg)) {
			expressions = ValueListsFromTuples(args);
		} else {
			auto values = ValueListFromExpressions(args);
			expressions.push_back(std::move(values));
		}
		return CreateRelation(connection.Values(std::move(expressions)));
	}
}

unique_ptr<DuckDBPyRelation> DuckDBPyConnection::View(const string &vname) {
	auto query_lock = LockForQuery();
	auto &connection = con.GetConnection();
	return CreateRelation(connection.View(vname));
}

unique_ptr<DuckDBPyRelation> DuckDBPyConnection::TableFunction(const string &fname, py::object params) {
	auto query_lock = LockForQuery();
	auto &connection = con.GetConnection();
	vector<Value> values;
	{
		PythonInputCallbackScope callback(nullptr);
		if (params.is_none()) {
			params = py::list();
		}
		if (!py::is_list_like(params)) {
			throw InvalidInputException("'params' has to be a list of parameters");
		}
		values = DuckDBPyConnection::TransformPythonParamList(params);
	}

	return CreateRelation(connection.TableFunction(fname, values));
}

unique_ptr<DuckDBPyRelation> DuckDBPyConnection::ReadVideoFrames(py::object params, const py::dict &options) {
	auto query_lock = LockForQuery();
	named_parameter_map_t named_parameters;
	for (auto &entry : options) {
		named_parameters[py::cast<string>(entry.first)] = TransformPythonValue(entry.second);
	}
	// Retain constant arguments in the relation so execution and Ray planning
	// see the scan itself.
	auto &connection = con.GetConnection();
	return CreateRelation(
	    connection.TableFunction("read_video_frames", TransformPythonParamList(params), named_parameters));
}

unique_ptr<DuckDBPyRelation> DuckDBPyConnection::FromDF(const PandasDataFrame &value) {
	auto query_lock = LockForQuery();
	auto &connection = con.GetConnection();
	string name = "df_" + StringUtil::GenerateRandomName();
	auto tableref = PythonReplacementScan::ReplacementObject(value, name, *connection.context);
	D_ASSERT(tableref);
	auto rel = make_shared_ptr<ViewRelation>(connection.context, std::move(tableref), name);
	return CreateRelation(std::move(rel));
}

unique_ptr<DuckDBPyRelation> DuckDBPyConnection::FromParquetInternal(Value &&file_param, bool binary_as_string,
                                                                     bool file_row_number, bool filename,
                                                                     bool hive_partitioning, bool union_by_name,
                                                                     const py::object &compression) {
	auto query_lock = LockForQuery();
	PythonInputCallbackScope callback(nullptr);
	auto &connection = con.GetConnection();
	string name = "parquet_" + StringUtil::GenerateRandomName();
	vector<Value> params;
	params.emplace_back(std::move(file_param));
	named_parameter_map_t named_parameters({{"binary_as_string", Value::BOOLEAN(binary_as_string)},
	                                        {"file_row_number", Value::BOOLEAN(file_row_number)},
	                                        {"filename", Value::BOOLEAN(filename)},
	                                        {"hive_partitioning", Value::BOOLEAN(hive_partitioning)},
	                                        {"union_by_name", Value::BOOLEAN(union_by_name)}});

	if (!py::none().is(compression)) {
		if (!py::isinstance<py::str>(compression)) {
			throw InvalidInputException("from_parquet only accepts 'compression' as a string");
		}
		named_parameters["compression"] = Value(py::str(compression));
	}
	D_ASSERT(py::gil_check());
	py::gil_scoped_release gil;
	return CreateRelation(connection.TableFunction("parquet_scan", params, named_parameters)->Alias(name));
}

unique_ptr<DuckDBPyRelation> DuckDBPyConnection::FromParquet(const string &file_glob, bool binary_as_string,
                                                             bool file_row_number, bool filename,
                                                             bool hive_partitioning, bool union_by_name,
                                                             const py::object &compression) {
	auto file_param = Value(file_glob);
	return FromParquetInternal(std::move(file_param), binary_as_string, file_row_number, filename, hive_partitioning,
	                           union_by_name, compression);
}

unique_ptr<DuckDBPyRelation> DuckDBPyConnection::FromParquets(const vector<string> &file_globs, bool binary_as_string,
                                                              bool file_row_number, bool filename,
                                                              bool hive_partitioning, bool union_by_name,
                                                              const py::object &compression) {
	vector<Value> params;
	auto file_globs_as_value = vector<Value>();
	for (const auto &file : file_globs) {
		file_globs_as_value.emplace_back(file);
	}
	auto file_param = Value::LIST(file_globs_as_value);
	return FromParquetInternal(std::move(file_param), binary_as_string, file_row_number, filename, hive_partitioning,
	                           union_by_name, compression);
}

unique_ptr<DuckDBPyRelation> DuckDBPyConnection::FromArrow(py::object &arrow_object) {
	auto query_lock = LockForQuery();
	PythonInputCallbackScope callback(nullptr);
	auto &connection = con.GetConnection();
	string name = "arrow_object_" + StringUtil::GenerateRandomName();
	if (!IsAcceptedArrowObject(arrow_object)) {
		auto py_object_type = string(py::str(py::type::of(arrow_object).attr("__name__")));
		throw InvalidInputException("Python Object Type %s is not an accepted Arrow Object.", py_object_type);
	}
	auto tableref = PythonReplacementScan::ReplacementObject(arrow_object, name, *connection.context, true);
	D_ASSERT(tableref);
	auto rel = make_shared_ptr<ViewRelation>(connection.context, std::move(tableref), name);
	return CreateRelation(std::move(rel));
}

shared_ptr<DuckDBPyConnection> DuckDBPyConnection::UnregisterPythonObject(const string &name) {
	auto query_lock = LockForQuery();
	auto &connection = con.GetConnection();
	if (!registered_objects.count(name)) {
		return shared_from_this();
	}
	D_ASSERT(py::gil_check());
	py::gil_scoped_release release;
	// FIXME: DROP TEMPORARY VIEW? doesn't exist?
	const auto quoted_name = KeywordHelper::WriteOptionallyQuoted(name, '\"');
	connection.Query("DROP VIEW " + quoted_name + "");
	registered_objects.erase(name);
	return shared_from_this();
}

bool DuckDBPyConnection::IsPandasDataframe(const py::object &object) {
	PythonInputCallbackScope callback(nullptr);
	if (!ModuleIsLoaded<PandasCacheItem>()) {
		return false;
	}
	auto &import_cache_py = *DuckDBPyConnection::ImportCache();
	return py::isinstance(object, import_cache_py.pandas.DataFrame());
}

bool IsValidNumpyDimensions(const py::handle &object, int &dim) {
	PythonInputCallbackScope callback(nullptr);
	// check the dimensions of numpy arrays
	// should only be called by IsAcceptedNumpyObject
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	if (!py::isinstance(object, import_cache.numpy.ndarray())) {
		return false;
	}
	auto shape = (py::cast<py::array>(object)).attr("shape");
	if (py::len(shape) != 1) {
		return false;
	}
	int cur_dim = (shape.attr("__getitem__")(0)).cast<int>();
	dim = dim == -1 ? cur_dim : dim;
	return dim == cur_dim;
}
NumpyObjectType DuckDBPyConnection::IsAcceptedNumpyObject(const py::object &object) {
	PythonInputCallbackScope callback(nullptr);
	if (!ModuleIsLoaded<NumpyCacheItem>()) {
		return NumpyObjectType::INVALID;
	}
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	if (py::isinstance(object, import_cache.numpy.ndarray())) {
		auto len = py::len((py::cast<py::array>(object)).attr("shape"));
		switch (len) {
		case 1:
			return NumpyObjectType::NDARRAY1D;
		case 2:
			return NumpyObjectType::NDARRAY2D;
		default:
			return NumpyObjectType::INVALID;
		}
	} else if (py::is_dict_like(object)) {
		int dim = -1;
		for (auto item : py::cast<py::dict>(object)) {
			if (!IsValidNumpyDimensions(item.second, dim)) {
				return NumpyObjectType::INVALID;
			}
		}
		return NumpyObjectType::DICT;
	} else if (py::is_list_like(object)) {
		int dim = -1;
		for (auto item : py::cast<py::list>(object)) {
			if (!IsValidNumpyDimensions(item, dim)) {
				return NumpyObjectType::INVALID;
			}
		}
		return NumpyObjectType::LIST;
	}
	return NumpyObjectType::INVALID;
}

PyArrowObjectType DuckDBPyConnection::GetArrowType(const py::handle &obj) {
	D_ASSERT(py::gil_check());
	// Protocol detection can execute descriptors and __class__ before the
	// replacement scan has installed its own callback scope.
	PythonInputCallbackScope callback(nullptr);

	if (py::isinstance<py::capsule>(obj)) {
		auto capsule = py::reinterpret_borrow<py::capsule>(obj);
		if (string(capsule.name()) != "arrow_array_stream") {
			throw InvalidInputException("Expected a 'arrow_array_stream' PyCapsule, got: %s", string(capsule.name()));
		}
		auto stream = capsule.get_pointer<struct ArrowArrayStream>();
		if (!stream->release) {
			throw InvalidInputException("The ArrowArrayStream was already released");
		}
		return PyArrowObjectType::PyCapsule;
	}

	if (ModuleIsLoaded<PyarrowCacheItem>()) {
		auto &import_cache = *DuckDBPyConnection::ImportCache();
		// MessageReader requires nanoarrow, separate scan function
		if (py::isinstance(obj, import_cache.pyarrow.ipc.MessageReader())) {
			return PyArrowObjectType::MessageReader;
		}

		if (ModuleIsLoaded<PyarrowDatasetCacheItem>()) {
			// Scanner/Dataset don't have __arrow_c_stream__, need dedicated handling
			if (py::isinstance(obj, import_cache.pyarrow.dataset.Scanner())) {
				return PyArrowObjectType::Scanner;
			} else if (py::isinstance(obj, import_cache.pyarrow.dataset.Dataset())) {
				return PyArrowObjectType::Dataset;
			}
		}
	}

	try {
		py::getattr(obj, "__arrow_c_stream__");
		return PyArrowObjectType::PyCapsuleInterface;
	} catch (py::error_already_set &error) {
		if (!error.matches(PyExc_AttributeError)) {
			throw;
		}
	}

	return PyArrowObjectType::Invalid;
}

bool DuckDBPyConnection::IsAcceptedArrowObject(const py::object &object) {
	return DuckDBPyConnection::GetArrowType(object) != PyArrowObjectType::Invalid;
}

} // namespace duckdb
