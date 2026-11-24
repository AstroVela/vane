// SPDX-FileCopyrightText: 2018-2026 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#include "vane_python/pyconnection/pyconnection.hpp"
#include "read_file_options.hpp"
#include "duckdb/catalog/default/default_types.hpp"
#include "duckdb/common/enums/file_compression_type.hpp"
#include "duckdb/function/table/read_csv.hpp"
#include "duckdb/main/relation/read_csv_relation.hpp"
#include "vane_python/filesystem_object.hpp"
#include "vane_python/python_conversion.hpp"
#include <sstream>

namespace duckdb {

static void AcceptableCSVOptions(const string &unkown_parameter) {
	// List of strings to match against
	const unordered_set<string> valid_parameters = {"header",
	                                                "strict_mode",
	                                                "compression",
	                                                "comment"
	                                                "sep",
	                                                "delimiter",
	                                                "files_to_sniff",
	                                                "dtype",
	                                                "na_values",
	                                                "skiprows",
	                                                "quotechar",
	                                                "escapechar",
	                                                "encoding",
	                                                "parallel",
	                                                "date_format",
	                                                "timestamp_format",
	                                                "sample_size",
	                                                "all_varchar",
	                                                "normalize_names",
	                                                "null_padding",
	                                                "names",
	                                                "lineterminator",
	                                                "columns",
	                                                "auto_type_candidates",
	                                                "max_line_size",
	                                                "ignore_errors",
	                                                "store_rejects",
	                                                "rejects_table",
	                                                "rejects_scan",
	                                                "rejects_limit",
	                                                "force_not_null",
	                                                "buffer_size",
	                                                "decimal",
	                                                "allow_quoted_nulls",
	                                                "filename",
	                                                "hive_partitioning",
	                                                "union_by_name",
	                                                "hive_types",
	                                                "hive_types_autocast",
	                                                "thousands"};

	std::ostringstream error;
	error << "The methods read_csv and read_csv_auto do not have the \"" << unkown_parameter << "\" argument." << '\n';
	error << "Possible arguments as suggestions: " << '\n';
	vector<string> parameters(valid_parameters.begin(), valid_parameters.end());
	auto suggestions = StringUtil::TopNJaroWinkler(parameters, unkown_parameter, 3);
	for (auto &suggestion : suggestions) {
		error << "* " << suggestion << '\n';
	}
	throw InvalidInputException(error.str());
}
void ConvertBooleanValue(const py::object &value, string param_name, named_parameter_map_t &bind_parameters) {
	if (!py::none().is(value)) {

		bool value_as_int = py::isinstance<py::int_>(value);
		bool value_as_bool = py::isinstance<py::bool_>(value);

		bool converted_value;
		if (value_as_bool) {
			converted_value = py::bool_(value);
		} else if (value_as_int) {
			if (static_cast<int>(py::int_(value)) != 0) {
				throw InvalidInputException("read_csv only accepts 0 if '%s' is given as an integer", param_name);
			}
			converted_value = true;
		} else {
			throw InvalidInputException("read_csv only accepts '%s' as an integer, or a boolean", param_name);
		}
		bind_parameters[param_name] = Value::BOOLEAN(converted_value);
	}
}

unique_ptr<DuckDBPyRelation> DuckDBPyConnection::ReadCSV(const py::object &name_p, py::kwargs &kwargs) {
	auto query_lock = LockForQuery();
	py::object header = py::none();
	py::object strict_mode = py::none();
	py::object auto_detect = py::none();
	py::object compression = py::none();
	py::object sep = py::none();
	py::object delimiter = py::none();
	py::object files_to_sniff = py::none();
	py::object dtype = py::none();
	py::object na_values = py::none();
	py::object skiprows = py::none();
	py::object quotechar = py::none();
	py::object escapechar = py::none();
	py::object encoding = py::none();
	py::object parallel = py::none();
	py::object date_format = py::none();
	py::object timestamp_format = py::none();
	py::object sample_size = py::none();
	py::object all_varchar = py::none();
	py::object normalize_names = py::none();
	py::object null_padding = py::none();
	py::object names_p = py::none();
	py::object lineterminator = py::none();
	py::object columns = py::none();
	py::object auto_type_candidates = py::none();
	py::object max_line_size = py::none();
	py::object ignore_errors = py::none();
	py::object store_rejects = py::none();
	py::object rejects_table = py::none();
	py::object rejects_scan = py::none();
	py::object rejects_limit = py::none();
	py::object force_not_null = py::none();
	py::object buffer_size = py::none();
	py::object decimal = py::none();
	py::object allow_quoted_nulls = py::none();
	py::object filename = py::none();
	py::object hive_partitioning = py::none();
	py::object union_by_name = py::none();
	py::object hive_types = py::none();
	py::object hive_types_autocast = py::none();
	py::object comment = py::none();
	py::object thousands_separator = py::none();

	for (auto &arg : kwargs) {
		PythonInputCallbackScope callback(nullptr);
		const auto &arg_name = py::str(arg.first).cast<std::string>();
		if (arg_name == "header") {
			header = kwargs[arg_name.c_str()];
		} else if (arg_name == "compression") {
			compression = kwargs[arg_name.c_str()];
		} else if (arg_name == "sep") {
			sep = kwargs[arg_name.c_str()];
		} else if (arg_name == "delimiter") {
			delimiter = kwargs[arg_name.c_str()];
		} else if (arg_name == "files_to_sniff") {
			files_to_sniff = kwargs[arg_name.c_str()];
		} else if (arg_name == "comment") {
			comment = kwargs[arg_name.c_str()];
		} else if (arg_name == "thousands") {
			thousands_separator = kwargs[arg_name.c_str()];
		} else if (arg_name == "dtype") {
			dtype = kwargs[arg_name.c_str()];
		} else if (arg_name == "na_values") {
			na_values = kwargs[arg_name.c_str()];
		} else if (arg_name == "skiprows") {
			skiprows = kwargs[arg_name.c_str()];
		} else if (arg_name == "quotechar") {
			quotechar = kwargs[arg_name.c_str()];
		} else if (arg_name == "escapechar") {
			escapechar = kwargs[arg_name.c_str()];
		} else if (arg_name == "encoding") {
			encoding = kwargs[arg_name.c_str()];
		} else if (arg_name == "parallel") {
			parallel = kwargs[arg_name.c_str()];
		} else if (arg_name == "date_format") {
			date_format = kwargs[arg_name.c_str()];
		} else if (arg_name == "timestamp_format") {
			timestamp_format = kwargs[arg_name.c_str()];
		} else if (arg_name == "sample_size") {
			sample_size = kwargs[arg_name.c_str()];
		} else if (arg_name == "auto_detect") {
			auto_detect = kwargs[arg_name.c_str()];
		} else if (arg_name == "all_varchar") {
			all_varchar = kwargs[arg_name.c_str()];
		} else if (arg_name == "normalize_names") {
			normalize_names = kwargs[arg_name.c_str()];
		} else if (arg_name == "null_padding") {
			null_padding = kwargs[arg_name.c_str()];
		} else if (arg_name == "names") {
			names_p = kwargs[arg_name.c_str()];
		} else if (arg_name == "lineterminator") {
			lineterminator = kwargs[arg_name.c_str()];
		} else if (arg_name == "columns") {
			columns = kwargs[arg_name.c_str()];
		} else if (arg_name == "auto_type_candidates") {
			auto_type_candidates = kwargs[arg_name.c_str()];
		} else if (arg_name == "max_line_size") {
			max_line_size = kwargs[arg_name.c_str()];
		} else if (arg_name == "ignore_errors") {
			ignore_errors = kwargs[arg_name.c_str()];
		} else if (arg_name == "store_rejects") {
			store_rejects = kwargs[arg_name.c_str()];
		} else if (arg_name == "rejects_table") {
			rejects_table = kwargs[arg_name.c_str()];
		} else if (arg_name == "rejects_scan") {
			rejects_scan = kwargs[arg_name.c_str()];
		} else if (arg_name == "rejects_limit") {
			rejects_limit = kwargs[arg_name.c_str()];
		} else if (arg_name == "force_not_null") {
			force_not_null = kwargs[arg_name.c_str()];
		} else if (arg_name == "buffer_size") {
			buffer_size = kwargs[arg_name.c_str()];
		} else if (arg_name == "decimal") {
			decimal = kwargs[arg_name.c_str()];
		} else if (arg_name == "allow_quoted_nulls") {
			allow_quoted_nulls = kwargs[arg_name.c_str()];
		} else if (arg_name == "filename") {
			filename = kwargs[arg_name.c_str()];
		} else if (arg_name == "hive_partitioning") {
			hive_partitioning = kwargs[arg_name.c_str()];
		} else if (arg_name == "union_by_name") {
			union_by_name = kwargs[arg_name.c_str()];
		} else if (arg_name == "hive_types") {
			hive_types = kwargs[arg_name.c_str()];
		} else if (arg_name == "hive_types_autocast") {
			hive_types_autocast = kwargs[arg_name.c_str()];
		} else if (arg_name == "strict_mode") {
			strict_mode = kwargs[arg_name.c_str()];
		} else {
			AcceptableCSVOptions(arg_name);
		}
	}

	auto &connection = con.GetConnection();
	CSVReaderOptions options;
	auto path_like = GetPathLike(name_p);
	auto &name = path_like.files;
	auto file_like_object_wrapper = std::move(path_like.dependency);
	PythonInputCallbackScope callback(connection.context);
	named_parameter_map_t bind_parameters;

	ParseMultiFileOptions(bind_parameters, filename, hive_partitioning, union_by_name, hive_types, hive_types_autocast);

	// First check if the header is explicitly set
	// when false this affects the returned types, so it needs to be known at initialization of the relation
	ConvertBooleanValue(header, "header", bind_parameters);
	ConvertBooleanValue(strict_mode, "strict_mode", bind_parameters);

	if (!py::none().is(compression)) {
		if (!py::isinstance<py::str>(compression)) {
			throw InvalidInputException("read_csv only accepts 'compression' as a string");
		}
		bind_parameters["compression"] = Value(py::str(compression));
	}

	if (!py::none().is(dtype)) {
		if (py::is_dict_like(dtype)) {
			child_list_t<Value> struct_fields;
			py::dict dtype_dict = dtype;
			for (auto &kv : dtype_dict) {
				shared_ptr<DuckDBPyType> sql_type;
				if (!py::try_cast(kv.second, sql_type)) {
					struct_fields.emplace_back(py::str(kv.first), py::str(kv.second));
				} else {
					struct_fields.emplace_back(py::str(kv.first), Value(sql_type->ToString()));
				}
			}
			auto dtype_struct = Value::STRUCT(std::move(struct_fields));
			bind_parameters["dtypes"] = std::move(dtype_struct);
		} else if (py::is_list_like(dtype)) {
			vector<Value> list_values;
			py::list dtype_list = dtype;
			for (auto &child : dtype_list) {
				shared_ptr<DuckDBPyType> sql_type;
				if (!py::try_cast(child, sql_type)) {
					list_values.push_back(Value(py::str(child)));
				} else {
					list_values.push_back(sql_type->ToString());
				}
			}
			bind_parameters["dtypes"] = Value::LIST(LogicalType::VARCHAR, std::move(list_values));
		} else {
			throw InvalidInputException("read_csv only accepts 'dtype' as a dictionary or a list of strings");
		}
	}

	bool has_sep = !py::none().is(sep);
	bool has_delimiter = !py::none().is(delimiter);
	if (has_sep && has_delimiter) {
		throw InvalidInputException("read_csv takes either 'delimiter' or 'sep', not both");
	}
	if (has_sep) {
		bind_parameters["delim"] = Value(py::str(sep));
	} else if (has_delimiter) {
		bind_parameters["delim"] = Value(py::str(delimiter));
	}

	if (!py::none().is(files_to_sniff)) {
		if (!py::isinstance<py::int_>(files_to_sniff)) {
			throw InvalidInputException("read_csv only accepts 'files_to_sniff' as an integer");
		}
		bind_parameters["files_to_sniff"] = Value::INTEGER(py::int_(files_to_sniff));
	}

	if (!py::none().is(names_p)) {
		if (!py::is_list_like(names_p)) {
			throw InvalidInputException("read_csv only accepts 'names' as a list of strings");
		}
		vector<Value> names;
		py::list names_list = names_p;
		for (auto &elem : names_list) {
			if (!py::isinstance<py::str>(elem)) {
				throw InvalidInputException("read_csv 'names' list has to consist of only strings");
			}
			names.push_back(Value(std::string(py::str(elem))));
		}
		bind_parameters["names"] = Value::LIST(LogicalType::VARCHAR, std::move(names));
	}

	if (!py::none().is(na_values)) {
		vector<Value> null_values;
		if (!py::isinstance<py::str>(na_values) && !py::is_list_like(na_values)) {
			throw InvalidInputException("read_csv only accepts 'na_values' as a string or a list of strings");
		} else if (py::isinstance<py::str>(na_values)) {
			null_values.push_back(Value(py::str(na_values)));
		} else {
			py::list null_list = na_values;
			for (auto &elem : null_list) {
				if (!py::isinstance<py::str>(elem)) {
					throw InvalidInputException("read_csv 'na_values' list has to consist of only strings");
				}
				null_values.push_back(Value(std::string(py::str(elem))));
			}
		}
		bind_parameters["nullstr"] = Value::LIST(LogicalType::VARCHAR, std::move(null_values));
	}

	if (!py::none().is(skiprows)) {
		if (!py::isinstance<py::int_>(skiprows)) {
			throw InvalidInputException("read_csv only accepts 'skiprows' as an integer");
		}
		bind_parameters["skip"] = Value::INTEGER(py::int_(skiprows));
	}

	if (!py::none().is(parallel)) {
		if (!py::isinstance<py::bool_>(parallel)) {
			throw InvalidInputException("read_csv only accepts 'parallel' as a boolean");
		}
		bind_parameters["parallel"] = Value::BOOLEAN(py::bool_(parallel));
	}

	if (!py::none().is(quotechar)) {
		if (!py::isinstance<py::str>(quotechar)) {
			throw InvalidInputException("read_csv only accepts 'quotechar' as a string");
		}
		bind_parameters["quote"] = Value(py::str(quotechar));
	}

	if (!py::none().is(comment)) {
		if (!py::isinstance<py::str>(comment)) {
			throw InvalidInputException("read_csv only accepts 'comment' as a string");
		}
		bind_parameters["comment"] = Value(py::str(comment));
	}

	if (!py::none().is(thousands_separator)) {
		if (!py::isinstance<py::str>(thousands_separator)) {
			throw InvalidInputException("read_csv only accepts 'thousands' as a string");
		}
		bind_parameters["thousands"] = Value(py::str(thousands_separator));
	}

	if (!py::none().is(escapechar)) {
		if (!py::isinstance<py::str>(escapechar)) {
			throw InvalidInputException("read_csv only accepts 'escapechar' as a string");
		}
		bind_parameters["escape"] = Value(py::str(escapechar));
	}

	if (!py::none().is(encoding)) {
		if (!py::isinstance<py::str>(encoding)) {
			throw InvalidInputException("read_csv only accepts 'encoding' as a string");
		}
		string encoding_str = StringUtil::Lower(py::str(encoding));
		if (encoding_str != "utf8" && encoding_str != "utf-8") {
			throw BinderException("Copy is only supported for UTF-8 encoded files, ENCODING 'UTF-8'");
		}
	}

	if (!py::none().is(date_format)) {
		if (!py::isinstance<py::str>(date_format)) {
			throw InvalidInputException("read_csv only accepts 'date_format' as a string");
		}
		bind_parameters["dateformat"] = Value(py::str(date_format));
	}

	if (!py::none().is(auto_detect)) {
		bool auto_detect_as_int = py::isinstance<py::int_>(auto_detect);
		bool auto_detect_as_bool = py::isinstance<py::bool_>(auto_detect);
		bool auto_detect_value;
		if (auto_detect_as_bool) {
			auto_detect_value = py::bool_(auto_detect);
		} else if (auto_detect_as_int) {
			if ((int)py::int_(auto_detect) != 0) {
				throw InvalidInputException("read_csv only accepts 0 if 'auto_detect' is given as an integer");
			}
			auto_detect_value = true;
		} else {
			throw InvalidInputException("read_csv only accepts 'auto_detect' as an integer, or a boolean");
		}
		bind_parameters["auto_detect"] = Value::BOOLEAN(auto_detect_value);
	}

	if (!py::none().is(timestamp_format)) {
		if (!py::isinstance<py::str>(timestamp_format)) {
			throw InvalidInputException("read_csv only accepts 'timestamp_format' as a string");
		}
		bind_parameters["timestampformat"] = Value(py::str(timestamp_format));
	}

	if (!py::none().is(sample_size)) {
		if (!py::isinstance<py::int_>(sample_size)) {
			throw InvalidInputException("read_csv only accepts 'sample_size' as an integer");
		}
		bind_parameters["sample_size"] = Value::INTEGER(py::int_(sample_size));
	}

	if (!py::none().is(all_varchar)) {
		if (!py::isinstance<py::bool_>(all_varchar)) {
			throw InvalidInputException("read_csv only accepts 'all_varchar' as a boolean");
		}
		bind_parameters["all_varchar"] = Value::BOOLEAN(py::bool_(all_varchar));
	}

	if (!py::none().is(normalize_names)) {
		if (!py::isinstance<py::bool_>(normalize_names)) {
			throw InvalidInputException("read_csv only accepts 'normalize_names' as a boolean");
		}
		bind_parameters["normalize_names"] = Value::BOOLEAN(py::bool_(normalize_names));
	}

	if (!py::none().is(null_padding)) {
		if (!py::isinstance<py::bool_>(null_padding)) {
			throw InvalidInputException("read_csv only accepts 'null_padding' as a boolean");
		}
		bind_parameters["null_padding"] = Value::BOOLEAN(py::bool_(null_padding));
	}

	if (!py::none().is(lineterminator)) {
		PythonCSVLineTerminator::Type new_line_type;
		if (!py::try_cast<PythonCSVLineTerminator::Type>(lineterminator, new_line_type)) {
			string actual_type = py::str(py::type::of(lineterminator));
			throw BinderException("read_csv only accepts 'lineterminator' as a string or CSVLineTerminator, not '%s'",
			                      actual_type);
		}
		bind_parameters["new_line"] = Value(PythonCSVLineTerminator::ToString(new_line_type));
	}

	if (!py::none().is(max_line_size)) {
		if (!py::isinstance<py::str>(max_line_size) && !py::isinstance<py::int_>(max_line_size)) {
			string actual_type = py::str(py::type::of(max_line_size));
			throw BinderException("read_csv only accepts 'max_line_size' as a string or an integer, not '%s'",
			                      actual_type);
		}
		auto val = TransformPythonValue(max_line_size, LogicalTypeId::VARCHAR);
		bind_parameters["max_line_size"] = val;
	}

	if (!py::none().is(auto_type_candidates)) {
		if (!py::isinstance<py::list>(auto_type_candidates)) {
			string actual_type = py::str(py::type::of(auto_type_candidates));
			throw BinderException("read_csv only accepts 'auto_type_candidates' as a list[str], not '%s'", actual_type);
		}
		auto val = TransformPythonValue(auto_type_candidates, LogicalType::LIST(LogicalTypeId::VARCHAR));
		bind_parameters["auto_type_candidates"] = val;
	}

	if (!py::none().is(ignore_errors)) {
		if (!py::isinstance<py::bool_>(ignore_errors)) {
			string actual_type = py::str(py::type::of(ignore_errors));
			throw BinderException("read_csv only accepts 'ignore_errors' as a bool, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(ignore_errors, LogicalTypeId::BOOLEAN);
		bind_parameters["ignore_errors"] = val;
	}

	if (!py::none().is(store_rejects)) {
		if (!py::isinstance<py::bool_>(store_rejects)) {
			string actual_type = py::str(py::type::of(store_rejects));
			throw BinderException("read_csv only accepts 'store_rejects' as a bool, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(store_rejects, LogicalTypeId::BOOLEAN);
		bind_parameters["store_rejects"] = val;
	}

	if (!py::none().is(rejects_table)) {
		if (!py::isinstance<py::str>(rejects_table)) {
			string actual_type = py::str(py::type::of(rejects_table));
			throw BinderException("read_csv only accepts 'rejects_table' as a string, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(rejects_table, LogicalTypeId::VARCHAR);
		bind_parameters["rejects_table"] = val;
	}

	if (!py::none().is(rejects_scan)) {
		if (!py::isinstance<py::str>(rejects_scan)) {
			string actual_type = py::str(py::type::of(rejects_scan));
			throw BinderException("read_csv only accepts 'rejects_scan' as a string, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(rejects_scan, LogicalTypeId::VARCHAR);
		bind_parameters["rejects_scan"] = val;
	}

	if (!py::none().is(rejects_limit)) {
		if (!py::isinstance<py::int_>(rejects_limit)) {
			string actual_type = py::str(py::type::of(rejects_limit));
			throw BinderException("read_csv only accepts 'rejects_limit' as an int, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(rejects_limit, LogicalTypeId::BIGINT);
		bind_parameters["rejects_limit"] = val;
	}

	if (!py::none().is(force_not_null)) {
		if (!py::isinstance<py::list>(force_not_null)) {
			string actual_type = py::str(py::type::of(force_not_null));
			throw BinderException("read_csv only accepts 'force_not_null' as a list[str], not '%s'", actual_type);
		}
		auto val = TransformPythonValue(force_not_null, LogicalType::LIST(LogicalTypeId::VARCHAR));
		bind_parameters["force_not_null"] = val;
	}

	if (!py::none().is(buffer_size)) {
		if (!py::isinstance<py::int_>(buffer_size)) {
			string actual_type = py::str(py::type::of(buffer_size));
			throw BinderException("read_csv only accepts 'buffer_size' as a list[str], not '%s'", actual_type);
		}
		auto val = TransformPythonValue(buffer_size, LogicalTypeId::UBIGINT);
		bind_parameters["buffer_size"] = val;
	}

	if (!py::none().is(decimal)) {
		if (!py::isinstance<py::str>(decimal)) {
			string actual_type = py::str(py::type::of(decimal));
			throw BinderException("read_csv only accepts 'decimal' as a string, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(decimal, LogicalTypeId::VARCHAR);
		bind_parameters["decimal_separator"] = val;
	}

	if (!py::none().is(allow_quoted_nulls)) {
		if (!py::isinstance<py::bool_>(allow_quoted_nulls)) {
			string actual_type = py::str(py::type::of(allow_quoted_nulls));
			throw BinderException("read_csv only accepts 'allow_quoted_nulls' as a bool, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(allow_quoted_nulls, LogicalTypeId::BOOLEAN);
		bind_parameters["allow_quoted_nulls"] = val;
	}

	if (!py::none().is(columns)) {
		if (!py::is_dict_like(columns)) {
			throw BinderException("read_csv only accepts 'columns' as a dict[str, str]");
		}
		py::dict columns_dict = columns;
		child_list_t<Value> struct_fields;

		for (auto &kv : columns_dict) {
			auto &column_name = kv.first;
			auto &type = kv.second;
			if (!py::isinstance<py::str>(column_name)) {
				string actual_type = py::str(py::type::of(column_name));
				throw BinderException("The provided column name must be a str, not of type '%s'", actual_type);
			}
			if (!py::isinstance<py::str>(type)) {
				string actual_type = py::str(py::type::of(column_name));
				throw BinderException("The provided column type must be a str, not of type '%s'", actual_type);
			}
			struct_fields.emplace_back(py::str(column_name), Value(py::str(type)));
		}
		auto dtype_struct = Value::STRUCT(std::move(struct_fields));
		bind_parameters["columns"] = std::move(dtype_struct);
	}

	// Create the ReadCSV Relation using the 'options'

	D_ASSERT(py::gil_check());
	py::gil_scoped_release gil;
	auto read_csv_p = connection.ReadCSV(name, std::move(bind_parameters));
	auto &read_csv = read_csv_p->Cast<ReadCSVRelation>();
	if (file_like_object_wrapper) {
		read_csv.AddExternalDependency(std::move(file_like_object_wrapper));
	}

	return CreateRelation(read_csv_p->Alias(read_csv.alias));
}

} // namespace duckdb
