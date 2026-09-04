// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/image.hpp"

#include "duckdb/common/error_data.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {

namespace {

static string PythonTypeName(const py::handle &value) {
	return py::str(py::type::of(value)).cast<string>();
}

static string RequireBytes(const py::handle &value) {
	if (!py::isinstance<py::bytes>(value)) {
		throw py::type_error(StringUtil::Format("Image.data must be bytes, not '%s'", PythonTypeName(value)));
	}
	return py::cast<string>(value);
}

static uint32_t RequireDimension(const py::handle &value, const char *field) {
	if (py::isinstance<py::bool_>(value) || !py::isinstance<py::int_>(value)) {
		throw py::type_error(StringUtil::Format("Image.%s must be int, not '%s'", field, PythonTypeName(value)));
	}
	auto converted = PyLong_AsUnsignedLongLong(value.ptr());
	if (converted == static_cast<unsigned long long>(-1) && PyErr_Occurred()) {
		PyErr_Clear();
		auto message = StringUtil::Format("Image.%s must fit in unsigned 32-bit", field);
		PyErr_SetString(PyExc_OverflowError, message.c_str());
		throw py::error_already_set();
	}
	if (converted > NumericLimits<uint32_t>::Maximum()) {
		auto message = StringUtil::Format("Image.%s must fit in unsigned 32-bit", field);
		PyErr_SetString(PyExc_OverflowError, message.c_str());
		throw py::error_already_set();
	}
	return static_cast<uint32_t>(converted);
}

static string RequireMode(const py::handle &value) {
	if (!py::isinstance<py::str>(value)) {
		throw py::type_error(StringUtil::Format("Image.mode must be str, not '%s'", PythonTypeName(value)));
	}
	return py::cast<string>(value);
}

} // namespace

PythonImage::PythonImage(string data_p, uint32_t width_p, uint32_t height_p, string mode_p)
    : data(std::move(data_p)), width(width_p), height(height_p), channels(ImageLogicalType::ChannelsForMode(mode_p)),
      mode(std::move(mode_p)) {
	ImageLogicalType::ValidateFields(data.size(), width, height, channels, mode, "Image");
}

void PythonImage::Initialize(py::handle &m) {
	auto image = py::class_<PythonImage>(m, "Image", py::module_local(), py::is_final());
	image.def(py::init([](const py::object &data, const py::object &width, const py::object &height,
	                      const py::object &mode) { return FromPython(data, width, height, mode); }),
	          py::arg("data"), py::arg("width"), py::arg("height"), py::arg("mode"));
	image.def_property_readonly("data", &PythonImage::Data);
	image.def_property_readonly("width", &PythonImage::Width);
	image.def_property_readonly("height", &PythonImage::Height);
	image.def_property_readonly("channels", &PythonImage::Channels);
	image.def_property_readonly("mode", &PythonImage::Mode);
	image.def_property_readonly("dtype", &PythonImage::DType);
	image.def("__repr__", &PythonImage::Repr);
	image.def("__eq__", &PythonImage::Equals, py::arg("other"), py::is_operator());
	image.def("__ne__", &PythonImage::NotEquals, py::arg("other"), py::is_operator());
	image.def("__hash__", &PythonImage::Hash);
	image.def(py::pickle([](const PythonImage &value) { return value.State(); },
	                     [](const py::tuple &state) {
		                     if (state.size() != 4) {
			                     throw py::value_error("Invalid Image pickle state");
		                     }
		                     return PythonImage::FromPython(state[0], state[1], state[2], state[3]);
	                     }));
}

PythonImage PythonImage::FromPython(const py::handle &data, const py::handle &width, const py::handle &height,
                                    const py::handle &mode) {
	try {
		return PythonImage(RequireBytes(data), RequireDimension(width, "width"), RequireDimension(height, "height"),
		                   RequireMode(mode));
	} catch (const InvalidInputException &error) {
		throw py::value_error(ErrorData(error).RawMessage());
	}
}

py::object PythonImage::FromValue(const Value &value) {
	ImageLogicalType::ValidateValue(value, "IMAGE materialization");
	auto &children = StructValue::GetChildren(value);
	return py::cast(PythonImage(
	    StringValue::Get(children[ImageLogicalType::DATA]), children[ImageLogicalType::WIDTH].GetValue<uint32_t>(),
	    children[ImageLogicalType::HEIGHT].GetValue<uint32_t>(), children[ImageLogicalType::MODE].GetValue<string>()));
}

Value PythonImage::ToValue() const {
	vector<Value> children;
	children.reserve(ImageLogicalType::FIELD_COUNT);
	children.push_back(Value::BLOB_RAW(data));
	children.push_back(Value::UINTEGER(width));
	children.push_back(Value::UINTEGER(height));
	children.push_back(Value::UTINYINT(channels));
	children.push_back(Value(mode));
	auto result = Value::STRUCT(ImageLogicalType::Create(), std::move(children));
	ImageLogicalType::ValidateValue(result, "Image");
	return result;
}

string PythonImage::Repr() const {
	return StringUtil::Format("Image(data=<%d bytes>, width=%d, height=%d, mode='%s')", data.size(), width, height,
	                          mode);
}

bool PythonImage::Equals(const PythonImage &other) const {
	return data == other.data && width == other.width && height == other.height && mode == other.mode;
}

bool PythonImage::NotEquals(const PythonImage &other) const {
	return !Equals(other);
}

Py_hash_t PythonImage::Hash() const {
	return py::hash(py::make_tuple(py::bytes(data), width, height, mode));
}

py::tuple PythonImage::State() const {
	return py::make_tuple(py::bytes(data), width, height, mode);
}

py::bytes PythonImage::Data() const {
	return py::bytes(data);
}

uint32_t PythonImage::Width() const {
	return width;
}

uint32_t PythonImage::Height() const {
	return height;
}

uint8_t PythonImage::Channels() const {
	return channels;
}

const string &PythonImage::Mode() const {
	return mode;
}

string PythonImage::DType() const {
	return "uint8";
}

} // namespace duckdb
