// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/file.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"

#include <limits>
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

static std::optional<string> OptionalStringFromValue(const Value &value) {
	if (value.IsNull()) {
		return std::nullopt;
	}
	return StringValue::Get(value);
}

static std::optional<int64_t> OptionalIntegerFromValue(const Value &value) {
	if (value.IsNull()) {
		return std::nullopt;
	}
	return value.GetValue<int64_t>();
}

} // namespace

PythonFile::PythonFile(string url_p, std::optional<string> content_type_p, std::optional<int64_t> position_p,
                       std::optional<int64_t> size_p, std::optional<string> checksum_p)
    : url(std::move(url_p)), content_type(std::move(content_type_p)), position(position_p), size(size_p),
      checksum(std::move(checksum_p)) {
	Validate(position, size, checksum);
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
	return PythonFile(std::move(url_value), std::move(content_type_value), position_value, size_value,
	                  std::move(checksum_value));
}

PythonFile PythonFile::FromValue(const Value &value, const LogicalType &type) {
	if (!FileLogicalType::IsFile(type)) {
		throw InternalException("Cannot materialize a non-FILE value as vane.File");
	}
	auto &children = StructValue::GetChildren(value);
	if (children.size() != FileLogicalType::FIELD_COUNT) {
		throw InternalException("FILE value has an invalid field count");
	}
	if (children[FileLogicalType::URL].IsNull()) {
		throw py::value_error("File.url cannot be None");
	}
	return PythonFile(StringValue::Get(children[FileLogicalType::URL]),
	                  OptionalStringFromValue(children[FileLogicalType::CONTENT_TYPE]),
	                  OptionalIntegerFromValue(children[FileLogicalType::POSITION]),
	                  OptionalIntegerFromValue(children[FileLogicalType::SIZE]),
	                  OptionalStringFromValue(children[FileLogicalType::CHECKSUM]));
}

Value PythonFile::ToValue() const {
	Validate(position, size, checksum);
	vector<Value> children;
	children.reserve(FileLogicalType::FIELD_COUNT);
	children.emplace_back(url);
	children.push_back(OptionalStringValue(content_type));
	children.push_back(OptionalIntegerValue(position));
	children.push_back(OptionalIntegerValue(size));
	children.push_back(OptionalStringValue(checksum));
	return Value::STRUCT(FileLogicalType::Create(), std::move(children));
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

void PythonFile::Validate(const std::optional<int64_t> &position, const std::optional<int64_t> &size,
                          const std::optional<string> &checksum) {
	if (position.has_value() != size.has_value()) {
		throw py::value_error("File.position and File.size must either both be None or both be non-None");
	}
	if (position) {
		if (*position < 0 || *size < 0) {
			throw py::value_error("File.position and File.size must be non-negative");
		}
		if (*position > std::numeric_limits<int64_t>::max() - *size) {
			throw py::value_error("File byte range exceeds signed 64-bit");
		}
	}
	if (checksum) {
		auto separator = checksum->find(':');
		if (separator == string::npos || separator == 0 || separator + 1 == checksum->size() ||
		    checksum->find(':', separator + 1) != string::npos) {
			throw py::value_error("File.checksum must have the form <algorithm>:<digest>");
		}
	}
}

} // namespace duckdb
