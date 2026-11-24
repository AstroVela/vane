// SPDX-FileCopyrightText: 2018-2026 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#include "vane_python/pyconnection/pyconnection.hpp"
#include "read_file_options.hpp"
#include "duckdb/catalog/default/default_types.hpp"
#include "duckdb/common/enums/file_compression_type.hpp"
#include "duckdb/main/relation/read_json_relation.hpp"
#include "vane_python/filesystem_object.hpp"
#include "vane_python/python_conversion.hpp"
#include "vane_python/python_input_callback.hpp"

namespace duckdb {

void ParseMultiFileOptions(named_parameter_map_t &options, const Optional<py::object> &filename,
                           const Optional<py::object> &hive_partitioning, const Optional<py::object> &union_by_name,
                           const Optional<py::object> &hive_types, const Optional<py::object> &hive_types_autocast) {
	PythonInputCallbackScope callback(nullptr);
	if (!py::none().is(filename)) {
		auto val = TransformPythonValue(filename);
		options["filename"] = val;
	}

	if (!py::none().is(hive_types)) {
		auto val = TransformPythonValue(hive_types);
		options["hive_types"] = val;
	}

	if (!py::none().is(hive_partitioning)) {
		if (!py::isinstance<py::bool_>(hive_partitioning)) {
			string actual_type = py::str(py::type::of(hive_partitioning));
			throw BinderException("read_json only accepts 'hive_partitioning' as a boolean, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(hive_partitioning, LogicalTypeId::BOOLEAN);
		options["hive_partitioning"] = val;
	}

	if (!py::none().is(union_by_name)) {
		if (!py::isinstance<py::bool_>(union_by_name)) {
			string actual_type = py::str(py::type::of(union_by_name));
			throw BinderException("read_json only accepts 'union_by_name' as a boolean, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(union_by_name, LogicalTypeId::BOOLEAN);
		options["union_by_name"] = val;
	}

	if (!py::none().is(hive_types_autocast)) {
		if (!py::isinstance<py::bool_>(hive_types_autocast)) {
			string actual_type = py::str(py::type::of(hive_types_autocast));
			throw BinderException("read_json only accepts 'hive_types_autocast' as a boolean, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(hive_types_autocast, LogicalTypeId::BOOLEAN);
		options["hive_types_autocast"] = val;
	}
}

unique_ptr<DuckDBPyRelation> DuckDBPyConnection::ReadJSON(
    const py::object &name_p, const Optional<py::object> &columns, const Optional<py::object> &sample_size,
    const Optional<py::object> &maximum_depth, const Optional<py::str> &records, const Optional<py::str> &format,
    const Optional<py::object> &date_format, const Optional<py::object> &timestamp_format,
    const Optional<py::object> &compression, const Optional<py::object> &maximum_object_size,
    const Optional<py::object> &ignore_errors, const Optional<py::object> &convert_strings_to_integers,
    const Optional<py::object> &field_appearance_threshold, const Optional<py::object> &map_inference_threshold,
    const Optional<py::object> &maximum_sample_files, const Optional<py::object> &filename,
    const Optional<py::object> &hive_partitioning, const Optional<py::object> &union_by_name,
    const Optional<py::object> &hive_types, const Optional<py::object> &hive_types_autocast) {
	auto query_lock = LockForQuery();

	named_parameter_map_t options;

	auto &connection = con.GetConnection();
	auto path_like = GetPathLike(name_p);
	auto &name = path_like.files;
	auto file_like_object_wrapper = std::move(path_like.dependency);
	// Options can contain subclasses with Python conversion methods. Keep the
	// input scope through conversion and binding after internal filesystem setup.
	PythonInputCallbackScope callback(connection.context);

	ParseMultiFileOptions(options, filename, hive_partitioning, union_by_name, hive_types, hive_types_autocast);

	if (!py::none().is(columns)) {
		if (!py::is_dict_like(columns)) {
			throw BinderException("read_json only accepts 'columns' as a dict[str, str]");
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
		options["columns"] = std::move(dtype_struct);
	}

	if (!py::none().is(records)) {
		if (!py::isinstance<py::str>(records)) {
			string actual_type = py::str(py::type::of(records));
			throw BinderException("read_json only accepts 'records' as a string, not '%s'", actual_type);
		}
		auto records_s = py::reinterpret_borrow<py::str>(records);
		auto records_option = std::string(py::str(records_s));
		options["records"] = Value(records_option);
	}

	if (!py::none().is(format)) {
		if (!py::isinstance<py::str>(format)) {
			string actual_type = py::str(py::type::of(format));
			throw BinderException("read_json only accepts 'format' as a string, not '%s'", actual_type);
		}
		auto format_s = py::reinterpret_borrow<py::str>(format);
		auto format_option = std::string(py::str(format_s));
		options["format"] = Value(format_option);
	}

	if (!py::none().is(date_format)) {
		if (!py::isinstance<py::str>(date_format)) {
			string actual_type = py::str(py::type::of(date_format));
			throw BinderException("read_json only accepts 'date_format' as a string, not '%s'", actual_type);
		}
		auto date_format_s = py::reinterpret_borrow<py::str>(date_format);
		auto date_format_option = std::string(py::str(date_format_s));
		options["date_format"] = Value(date_format_option);
	}

	if (!py::none().is(timestamp_format)) {
		if (!py::isinstance<py::str>(timestamp_format)) {
			string actual_type = py::str(py::type::of(timestamp_format));
			throw BinderException("read_json only accepts 'timestamp_format' as a string, not '%s'", actual_type);
		}
		auto timestamp_format_s = py::reinterpret_borrow<py::str>(timestamp_format);
		auto timestamp_format_option = std::string(py::str(timestamp_format_s));
		options["timestamp_format"] = Value(timestamp_format_option);
	}

	if (!py::none().is(compression)) {
		if (!py::isinstance<py::str>(compression)) {
			string actual_type = py::str(py::type::of(compression));
			throw BinderException("read_json only accepts 'compression' as a string, not '%s'", actual_type);
		}
		auto compression_s = py::reinterpret_borrow<py::str>(compression);
		auto compression_option = std::string(py::str(compression_s));
		options["compression"] = Value(compression_option);
	}

	if (!py::none().is(sample_size)) {
		if (!py::isinstance<py::int_>(sample_size)) {
			string actual_type = py::str(py::type::of(sample_size));
			throw BinderException("read_json only accepts 'sample_size' as an integer, not '%s'", actual_type);
		}
		options["sample_size"] = Value::INTEGER(py::int_(sample_size));
	}

	if (!py::none().is(maximum_depth)) {
		if (!py::isinstance<py::int_>(maximum_depth)) {
			string actual_type = py::str(py::type::of(maximum_depth));
			throw BinderException("read_json only accepts 'maximum_depth' as an integer, not '%s'", actual_type);
		}
		options["maximum_depth"] = Value::INTEGER(py::int_(maximum_depth));
	}

	if (!py::none().is(maximum_object_size)) {
		if (!py::isinstance<py::int_>(maximum_object_size)) {
			string actual_type = py::str(py::type::of(maximum_object_size));
			throw BinderException("read_json only accepts 'maximum_object_size' as an unsigned integer, not '%s'",
			                      actual_type);
		}
		auto val = TransformPythonValue(maximum_object_size, LogicalTypeId::UINTEGER);
		options["maximum_object_size"] = val;
	}

	if (!py::none().is(ignore_errors)) {
		if (!py::isinstance<py::bool_>(ignore_errors)) {
			string actual_type = py::str(py::type::of(ignore_errors));
			throw BinderException("read_json only accepts 'ignore_errors' as a boolean, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(ignore_errors, LogicalTypeId::BOOLEAN);
		options["ignore_errors"] = val;
	}

	if (!py::none().is(convert_strings_to_integers)) {
		if (!py::isinstance<py::bool_>(convert_strings_to_integers)) {
			string actual_type = py::str(py::type::of(convert_strings_to_integers));
			throw BinderException("read_json only accepts 'convert_strings_to_integers' as a boolean, not '%s'",
			                      actual_type);
		}
		auto val = TransformPythonValue(convert_strings_to_integers, LogicalTypeId::BOOLEAN);
		options["convert_strings_to_integers"] = val;
	}

	if (!py::none().is(field_appearance_threshold)) {
		if (!py::isinstance<py::float_>(field_appearance_threshold)) {
			string actual_type = py::str(py::type::of(field_appearance_threshold));
			throw BinderException("read_json only accepts 'field_appearance_threshold' as a float, not '%s'",
			                      actual_type);
		}
		auto val = TransformPythonValue(field_appearance_threshold, LogicalTypeId::DOUBLE);
		options["field_appearance_threshold"] = val;
	}

	if (!py::none().is(map_inference_threshold)) {
		if (!py::isinstance<py::int_>(map_inference_threshold)) {
			string actual_type = py::str(py::type::of(map_inference_threshold));
			throw BinderException("read_json only accepts 'map_inference_threshold' as an integer, not '%s'",
			                      actual_type);
		}
		auto val = TransformPythonValue(map_inference_threshold, LogicalTypeId::BIGINT);
		options["map_inference_threshold"] = val;
	}

	if (!py::none().is(maximum_sample_files)) {
		if (!py::isinstance<py::int_>(maximum_sample_files)) {
			string actual_type = py::str(py::type::of(maximum_sample_files));
			throw BinderException("read_json only accepts 'maximum_sample_files' as an integer, not '%s'", actual_type);
		}
		auto val = TransformPythonValue(maximum_sample_files, LogicalTypeId::BIGINT);
		options["maximum_sample_files"] = val;
	}

	bool auto_detect = false;
	if (!options.count("columns")) {
		options["auto_detect"] = Value::BOOLEAN(true);
		auto_detect = true;
	}

	D_ASSERT(py::gil_check());
	py::gil_scoped_release gil;
	auto read_json_relation =
	    make_shared_ptr<ReadJSONRelation>(connection.context, name, std::move(options), auto_detect);
	if (read_json_relation == nullptr) {
		throw BinderException("read_json can only be used when the JSON extension is (statically) loaded");
	}
	if (file_like_object_wrapper) {
		read_json_relation->AddExternalDependency(std::move(file_like_object_wrapper));
	}
	return CreateRelation(std::move(read_json_relation));
}

} // namespace duckdb
