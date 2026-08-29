// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/file.hpp"

#include "duckdb/common/error_data.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "file_value.hpp"
#include "vane_python/pybind11/conversions/pyconnection_default.hpp"
#include "vane_python/pyconnection/pyconnection.hpp"
#include "vane_python/python_objects.hpp"

#include <mutex>
#include <utility>

namespace duckdb {

namespace {

static string PythonTypeName(const py::handle &value) {
	return py::str(py::type::of(value)).cast<string>();
}

static string RequireString(const py::handle &value, const char *field) {
	if (!py::isinstance<py::str>(value)) {
		throw py::type_error(StringUtil::Format("File.%s must be str, not '%s'", field, PythonTypeName(value)));
	}
	return py::cast<string>(value);
}

static std::optional<string> OptionalString(const py::handle &value, const char *field) {
	if (value.is_none()) {
		return std::nullopt;
	}
	return RequireString(value, field);
}

static std::optional<int64_t> OptionalInteger(const py::handle &value, const char *field) {
	if (value.is_none()) {
		return std::nullopt;
	}
	if (py::isinstance<py::bool_>(value) || !py::isinstance<py::int_>(value)) {
		throw py::type_error(StringUtil::Format("File.%s must be int or None, not '%s'", field, PythonTypeName(value)));
	}

	int overflow = 0;
	auto result = PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
	if (overflow != 0) {
		PyErr_Clear();
		auto message = StringUtil::Format("File.%s must fit in signed 64-bit", field);
		PyErr_SetString(PyExc_OverflowError, message.c_str());
		throw py::error_already_set();
	}
	if (result == -1 && PyErr_Occurred()) {
		throw py::error_already_set();
	}
	return result;
}

static Value OptionalStringValue(const std::optional<string> &value) {
	return value ? Value(*value) : Value(LogicalType::VARCHAR);
}

static Value OptionalIntegerValue(const std::optional<int64_t> &value) {
	return value ? Value::BIGINT(*value) : Value(LogicalType::BIGINT);
}

static std::optional<string> OptionalStringFromReference(bool has_value, const string &value) {
	return has_value ? std::optional<string>(value) : std::nullopt;
}

static std::optional<int64_t> OptionalIntegerFromReference(bool has_value, int64_t value) {
	return has_value ? std::optional<int64_t>(value) : std::nullopt;
}

static FileReference MakeReference(const string &url, const std::optional<string> &content_type,
                                   const std::optional<int64_t> &position, const std::optional<int64_t> &size,
                                   const std::optional<string> &checksum) {
	return FileReference::FromFields(Value(url), OptionalStringValue(content_type), OptionalIntegerValue(position),
	                                 OptionalIntegerValue(size), OptionalStringValue(checksum), "File");
}

static py::object ExecuteFileScalar(const PythonFile &file, shared_ptr<DuckDBPyConnection> connection,
                                    const string &query, vector<Value> parameters = {}) {
	if (!connection) {
		connection = DuckDBPyConnection::DefaultConnection();
	}
	parameters.insert(parameters.begin(), file.ToValue());
	Value value;
	ClientProperties client_properties;
	{
		D_ASSERT(py::gil_check());
		py::gil_scoped_release release;
		unique_lock<mutex> lock(connection->py_connection_lock);
		auto &native_connection = connection->con.GetConnection();
		auto pending = native_connection.PendingQuery(query, parameters);
		if (pending->HasError()) {
			pending->ThrowError();
		}
		auto result = DuckDBPyConnection::CompletePendingQuery(*pending);
		if (!result || result->HasError()) {
			if (result) {
				result->ThrowError();
			}
			throw InternalException("FILE metadata query returned no result");
		}
		auto chunk = result->Fetch();
		if (!chunk || chunk->size() != 1 || chunk->ColumnCount() != 1) {
			throw InternalException("FILE metadata query did not return exactly one value");
		}
		value = chunk->GetValue(0, 0);
		client_properties = native_connection.context->GetClientProperties();
	}
	return PythonObject::FromValue(value, value.type(), client_properties);
}

} // namespace

PythonFile::PythonFile(string url_p, std::optional<string> content_type_p, std::optional<int64_t> position_p,
                       std::optional<int64_t> size_p, std::optional<string> checksum_p)
    : url(std::move(url_p)), content_type(std::move(content_type_p)), position(position_p), size(size_p),
      checksum(std::move(checksum_p)) {
	MakeReference(url, content_type, position, size, checksum);
}

void PythonFile::Initialize(py::handle &m) {
	auto file = py::class_<PythonFile>(m, "File", py::module_local(), py::is_final());
	file.def(py::init([](const py::object &url, const py::object &content_type, const py::object &position,
	                     const py::object &size, const py::object &checksum) {
		         return PythonFile::FromPython(url, content_type, position, size, checksum);
	         }),
	         py::arg("url"), py::arg("content_type") = py::none(), py::arg("position") = py::none(),
	         py::arg("size") = py::none(), py::arg("checksum") = py::none());
	file.def_property_readonly("url", &PythonFile::Url);
	file.def_property_readonly("content_type", &PythonFile::ContentType);
	file.def_property_readonly("position", &PythonFile::Position);
	file.def_property_readonly("size", &PythonFile::Size);
	file.def_property_readonly("checksum", &PythonFile::Checksum);
	file.def("exists", &PythonFile::Exists, "Return whether this FILE's logical view is accessible", py::kw_only(),
	         py::arg("connection") = py::none());
	file.def("stat", &PythonFile::Stat, "Return the six-field SQL file_stat value", py::kw_only(),
	         py::arg("connection") = py::none());
	file.def("mime_type", &PythonFile::MimeType, "Return the MIME type selected by SQL file_mime_type",
	         py::arg("detect") = "metadata", py::kw_only(), py::arg("connection") = py::none());
	file.def(
	    "open",
	    [](const PythonFile &value, const py::object &buffer_size, shared_ptr<DuckDBPyConnection> connection) {
		    return py::module_::import("vane._file")
		        .attr("_file_open")(py::cast(value, py::return_value_policy::copy), buffer_size,
		                            py::arg("connection") = std::move(connection));
	    },
	    "Open this FILE as a read-only VaneFileReader", py::arg("buffer_size") = py::none(), py::kw_only(),
	    py::arg("connection") = py::none());
	file.def(
	    "to_tempfile",
	    [](const PythonFile &value, const py::object &buffer_size, shared_ptr<DuckDBPyConnection> connection) {
		    return py::module_::import("vane._file")
		        .attr("_file_to_tempfile")(py::cast(value, py::return_value_policy::copy), buffer_size,
		                                   py::arg("connection") = std::move(connection));
	    },
	    "Copy this FILE's logical view into a temporary binary file", py::arg("buffer_size") = 1024 * 1024,
	    py::kw_only(), py::arg("connection") = py::none());
	file.def("__str__", &PythonFile::ToString);
	file.def("__repr__", &PythonFile::Repr);
	file.def("__eq__", &PythonFile::Equals, py::arg("other"), py::is_operator());
	file.def("__ne__", &PythonFile::NotEquals, py::arg("other"), py::is_operator());
	file.def("__hash__", &PythonFile::Hash);
	file.def(py::pickle([](const PythonFile &value) { return value.State(); },
	                    [](const py::tuple &state) {
		                    if (state.size() != FileLogicalType::FIELD_COUNT) {
			                    throw py::value_error("Invalid File pickle state");
		                    }
		                    return PythonFile::FromPython(
		                        state[FileLogicalType::URL], state[FileLogicalType::CONTENT_TYPE],
		                        state[FileLogicalType::POSITION], state[FileLogicalType::SIZE],
		                        state[FileLogicalType::CHECKSUM]);
	                    }));
}

PythonFile PythonFile::FromPython(const py::handle &url, const py::handle &content_type, const py::handle &position,
                                  const py::handle &size, const py::handle &checksum) {
	auto url_value = RequireString(url, "url");
	auto content_type_value = OptionalString(content_type, "content_type");
	auto position_value = OptionalInteger(position, "position");
	auto size_value = OptionalInteger(size, "size");
	auto checksum_value = OptionalString(checksum, "checksum");
	try {
		return PythonFile(std::move(url_value), std::move(content_type_value), position_value, size_value,
		                  std::move(checksum_value));
	} catch (const InvalidInputException &error) {
		auto error_data = ErrorData(error);
		throw py::value_error(error_data.RawMessage());
	}
}

PythonFile PythonFile::FromValue(const Value &value) {
	auto reference = FileReference::FromValue(value, "FILE materialization");
	return PythonFile(std::move(reference.url),
	                  OptionalStringFromReference(reference.has_content_type, reference.content_type),
	                  OptionalIntegerFromReference(reference.has_range, reference.position),
	                  OptionalIntegerFromReference(reference.has_range, reference.size),
	                  OptionalStringFromReference(reference.has_checksum, reference.checksum));
}

Value PythonFile::ToValue() const {
	return MakeReference(url, content_type, position, size, checksum).ToValue();
}

string PythonFile::ToString() const {
	return url;
}

string PythonFile::Repr() const {
	auto state = State();
	return "File(url=" + py::repr(state[FileLogicalType::URL]).cast<string>() +
	       ", content_type=" + py::repr(state[FileLogicalType::CONTENT_TYPE]).cast<string>() +
	       ", position=" + py::repr(state[FileLogicalType::POSITION]).cast<string>() +
	       ", size=" + py::repr(state[FileLogicalType::SIZE]).cast<string>() +
	       ", checksum=" + py::repr(state[FileLogicalType::CHECKSUM]).cast<string>() + ")";
}

bool PythonFile::Equals(const PythonFile &other) const {
	return url == other.url && content_type == other.content_type && position == other.position && size == other.size &&
	       checksum == other.checksum;
}

bool PythonFile::NotEquals(const PythonFile &other) const {
	return !Equals(other);
}

Py_hash_t PythonFile::Hash() const {
	return py::hash(State());
}

py::tuple PythonFile::State() const {
	return py::make_tuple(url, content_type, position, size, checksum);
}

py::object PythonFile::Exists(shared_ptr<DuckDBPyConnection> connection) const {
	return ExecuteFileScalar(*this, std::move(connection), "SELECT file_exists(?)");
}

py::object PythonFile::Stat(shared_ptr<DuckDBPyConnection> connection) const {
	return ExecuteFileScalar(*this, std::move(connection), "SELECT file_stat(?)");
}

py::object PythonFile::MimeType(const string &detect, shared_ptr<DuckDBPyConnection> connection) const {
	if (detect == "metadata") {
		return ExecuteFileScalar(*this, std::move(connection), "SELECT file_mime_type(?)");
	}
	return ExecuteFileScalar(*this, std::move(connection), "SELECT file_mime_type(?, ?)", {Value(detect)});
}

const string &PythonFile::Url() const {
	return url;
}

const std::optional<string> &PythonFile::ContentType() const {
	return content_type;
}

const std::optional<int64_t> &PythonFile::Position() const {
	return position;
}

const std::optional<int64_t> &PythonFile::Size() const {
	return size;
}

const std::optional<string> &PythonFile::Checksum() const {
	return checksum;
}

} // namespace duckdb
