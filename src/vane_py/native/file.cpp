// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/file.hpp"

#include "duckdb/common/error_data.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "file_value.hpp"

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
