// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/image.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/image.hpp"

namespace duckdb {

bool PythonImage::IsPIL(const py::handle &value) {
	// PIL inputs have already imported their defining module. Checking the
	// loaded module avoids importing an optional codec for unrelated values.
	auto module = PyDict_GetItemString(PyImport_GetModuleDict(), "PIL.Image");
	if (!module) {
		return false;
	}
	auto image_class = PyObject_GetAttrString(module, "Image");
	if (!image_class) {
		throw py::error_already_set();
	}
	auto type = py::reinterpret_steal<py::object>(image_class);
	return py::isinstance(value, type);
}

static py::array ImagePixels(const py::handle &value, const LogicalType &type, string &mode) {
	py::object input = py::reinterpret_borrow<py::object>(value);
	if (PythonImage::IsPIL(value)) {
		mode = py::cast<string>(value.attr("mode"));
		if (mode == "I;16" || mode == "I;16L" || mode == "I;16B") {
			mode = "L16";
		}
		ImageLogicalType::ChannelsForMode(mode);
		input = py::module_::import("numpy").attr("asarray")(value);
		if (mode == "L16") {
			input = input.attr("astype")(py::dtype::of<uint16_t>(), py::arg("copy") = false);
		}
		if (ImageLogicalType::ChannelsForMode(mode) == 1) {
			auto shape = input.attr("shape").cast<py::tuple>();
			input = input.attr("reshape")(shape[0], shape[1], 1);
		}
	}
	if (!py::isinstance<py::array>(input) ||
	    py::module_::import("numpy").attr("ma").attr("isMaskedArray")(input).cast<bool>()) {
		throw InvalidInputException("IMAGE input must be an HWC ndarray, PIL.Image.Image, or NULL");
	}
	auto array = py::reinterpret_borrow<py::array>(input);
	if (array.ndim() != 3 || array.shape(0) <= 0 || array.shape(1) <= 0 ||
	    array.shape(0) > NumericLimits<uint32_t>::Maximum() || array.shape(1) > NumericLimits<uint32_t>::Maximum() ||
	    array.shape(2) < 1 || array.shape(2) > 4) {
		throw InvalidInputException("IMAGE input requires HWC pixels, positive height/width, and 1 to 4 channels");
	}
	if (mode.empty()) {
		auto channels = uint8_t(array.shape(2));
		if (array.dtype().equal(py::dtype::of<uint8_t>())) {
			mode = ImageLogicalType::ModeName(channels);
		} else if (array.dtype().equal(py::dtype::of<uint16_t>())) {
			mode = ImageLogicalType::ModeName(channels + 4);
		} else if (array.dtype().equal(py::dtype::of<float>()) && channels >= 3) {
			mode = ImageLogicalType::ModeName(channels + 6);
		} else {
			throw InvalidInputException("IMAGE requires UInt8/UInt16 pixels or RGB/RGBA Float32 pixels");
		}
	}
	auto height = uint32_t(array.shape(0));
	auto width = uint32_t(array.shape(1));
	ImageLogicalType::ValidateFields(idx_t(array.size()), width, height, uint16_t(array.shape(2)), mode,
	                                 "Python IMAGE");
	ImageLogicalType::ValidateShape(type, width, height, mode, "Python IMAGE");
	// Accept strided input without changing values, dtype, or channel order.
	auto contiguous = py::array::ensure(array, py::array::c_style);
	if (!contiguous) {
		throw InvalidInputException("Could not copy IMAGE input to contiguous HWC pixels");
	}
	return contiguous;
}

Value PythonImage::FromPython(const py::handle &value, const LogicalType &type) {
	string mode;
	auto pixels = ImagePixels(value, type, mode);
	return ImageVector::FromPixels(const_data_ptr_cast(pixels.data()), idx_t(pixels.size()), uint32_t(pixels.shape(1)),
	                               uint32_t(pixels.shape(0)), mode, type);
}

void PythonImage::ToVector(const py::handle &value, Vector &result, idx_t row) {
	string mode;
	auto pixels = ImagePixels(value, result.GetType(), mode);
	ImageVector::WritePixels(result, row, uint32_t(pixels.shape(1)), uint32_t(pixels.shape(0)), mode,
	                         const_data_ptr_cast(pixels.data()));
}

py::object PythonImage::FromValue(const Value &value) {
	ImageLogicalType::ValidateValue(value, "IMAGE materialization");
	auto layout = ImageVector::Layout(value);
	auto dtype = layout.mode <= 4   ? py::dtype::of<uint8_t>()
	             : layout.mode <= 8 ? py::dtype::of<uint16_t>()
	                                : py::dtype::of<float>();
	py::array array(dtype, {py::ssize_t(layout.height), py::ssize_t(layout.width), py::ssize_t(layout.channels)});
	ImageVector::CopyPixels(value, data_ptr_cast(array.mutable_data()));
	return std::move(array);
}

} // namespace duckdb
