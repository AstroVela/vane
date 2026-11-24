// SPDX-FileCopyrightText: 2018-2026 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#include "module_bindings.hpp"
#include "vane_python/pyconnection/pyconnection.hpp"
#include "vane_python/pybind11/conversions/pyconnection_default.hpp"
#include "duckdb/main/config.hpp"
#include "vane_python/file.hpp"
#include "vane_python/image.hpp"

namespace duckdb {

void InitializeModuleTypesBindings(py::module_ &m) {
	m.def(
	    "sqltype",
	    [](const string &type_str, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Type(type_str);
	    },
	    "Create a type object by parsing the 'type_str' string", py::arg("type_str"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "dtype",
	    [](const string &type_str, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Type(type_str);
	    },
	    "Create a type object by parsing the 'type_str' string", py::arg("type_str"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "type",
	    [](const string &type_str, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Type(type_str);
	    },
	    "Create a type object by parsing the 'type_str' string", py::arg("type_str"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "_parse_serialized_logical_type",
	    [](const string &type_str) { return make_shared_ptr<DuckDBPyType>(DBConfig::ParseLogicalType(type_str)); },
	    "Parse an internal canonical serialized logical type", py::arg("type_str"));
	m.def(
	    "array_type",
	    [](const shared_ptr<DuckDBPyType> &type, idx_t size, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->ArrayType(type, size);
	    },
	    "Create an array type object of 'type'", py::arg("type").none(false), py::arg("size"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "list_type",
	    [](const shared_ptr<DuckDBPyType> &type, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->ListType(type);
	    },
	    "Create a list type object of 'type'", py::arg("type").none(false), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "tensor_type",
	    [](const shared_ptr<DuckDBPyType> &type, const py::object &shape,
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->TensorType(type, shape);
	    },
	    "Create a Tensor type; None shape dimensions vary by row", py::arg("type").none(false),
	    py::arg("shape").none(false), py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "file_type",
	    [](const py::object &media_type) {
		    auto native_media_type = FileMediaType::UNKNOWN;
		    if (!media_type.is_none()) {
			    if (!py::isinstance<PythonFileMediaType>(media_type)) {
				    throw py::type_error("media_type must be vane.MediaType");
			    }
			    native_media_type = py::cast<PythonFileMediaType>(media_type).Type();
		    }
		    return make_shared_ptr<DuckDBPyType>(FileLogicalType::Create(native_media_type));
	    },
	    "Create a logical type in the FILE family", py::arg("media_type") = py::none());
	m.def(
	    "image_type",
	    [](const py::object &mode, const py::object &height, const py::object &width) {
		    if (mode.is_none() && height.is_none() && width.is_none()) {
			    return make_shared_ptr<DuckDBPyType>(ImageLogicalType::Create());
		    }
		    if (py::isinstance<py::str>(mode) && height.is_none() && width.is_none()) {
			    return make_shared_ptr<DuckDBPyType>(ImageLogicalType::Create(py::cast<string>(mode)));
		    }
		    if (!py::isinstance<py::str>(mode) || !py::isinstance<py::int_>(height) ||
		        !py::isinstance<py::int_>(width) || py::isinstance<py::bool_>(height) ||
		        py::isinstance<py::bool_>(width)) {
			    throw py::type_error("image_type requires a string mode and integer height and width together");
		    }
		    return make_shared_ptr<DuckDBPyType>(ImageLogicalType::Create(
		        py::cast<string>(mode), py::cast<uint32_t>(height), py::cast<uint32_t>(width)));
	    },
	    "Create a decoded uint8 IMAGE type, optionally constrained to mode, height and width",
	    py::arg("mode") = py::none(), py::arg("height") = py::none(), py::arg("width") = py::none());
	m.def(
	    "union_type",
	    [](const py::object &members, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->UnionType(members);
	    },
	    "Create a union type object from 'members'", py::arg("members").none(false), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "string_type",
	    [](const string &collation = string(), shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->StringType(collation);
	    },
	    "Create a string type with an optional collation", py::arg("collation") = "", py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "enum_type",
	    [](const string &name, const shared_ptr<DuckDBPyType> &type, const py::list &values_p,
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->EnumType(name, type, values_p);
	    },
	    "Create an enum type of underlying 'type', consisting of the list of 'values'", py::arg("name"),
	    py::arg("type"), py::arg("values"), py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "decimal_type",
	    [](int width, int scale, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->DecimalType(width, scale);
	    },
	    "Create a decimal type with 'width' and 'scale'", py::arg("width"), py::arg("scale"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "struct_type",
	    [](const py::object &fields, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->StructType(fields);
	    },
	    "Create a struct type object from 'fields'", py::arg("fields"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "row_type",
	    [](const py::object &fields, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->StructType(fields);
	    },
	    "Create a struct type object from 'fields'", py::arg("fields"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "map_type",
	    [](const shared_ptr<DuckDBPyType> &key_type, const shared_ptr<DuckDBPyType> &value_type,
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->MapType(key_type, value_type);
	    },
	    "Create a map type object from 'key_type' and 'value_type'", py::arg("key").none(false),
	    py::arg("value").none(false), py::kw_only(), py::arg("connection") = py::none());
}

} // namespace duckdb
