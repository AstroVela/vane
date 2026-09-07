// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/image_functions.hpp"

#include "image_operator_contract.hpp"
#include "media_backend.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/execution/expression_executor_state.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"

namespace duckdb {
namespace {

[[noreturn]] static void RaiseImageHelperError(ClientContext &context, py::error_already_set &error) {
	if (context.IsInterrupted() || !error.matches(PyExc_Exception)) {
		throw InterruptException();
	}
	if (error.matches(PyExc_MemoryError)) {
		throw OutOfMemoryException("Python Image operator ran out of memory");
	}
	if (error.matches(PyExc_OverflowError)) {
		throw OutOfRangeException("Python Image operator exceeded its byte limit: %s", error.what());
	}
	if (error.matches(PyExc_ImportError)) {
		throw InvalidInputException("Python Image operator requires its optional dependencies: %s", error.what());
	}
	throw InternalException("Python Image operator helper failed: %s", error.what());
}

static void CropImage(DataChunk &args, ExpressionState &state, Vector &result) {
	auto constant = args.AllConstant();
	auto count = constant && args.size() ? idx_t(1) : args.size();
	ImageOperatorInput images(args.data[0], count);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	idx_t bytes = 0;
	for (idx_t row = 0; row < count; row++) {
		auto &context = state.GetContext();
		ImageOperatorContract::Interrupt(context);
		ImagePixelView image;
		ImageCropBox box;
		if (images.IsNull(row) || !ImageCropBox::Read(args.data[1], row, box) || !images.Read(row, image)) {
			result.SetValue(row, Value(result.GetType()));
			continue;
		}
		auto size = ImageOperatorContract::CheckSize(box.width, box.height, image.layout.channels,
		                                             ImageOperatorContract::MAX_BYTES - bytes);
		bytes += size;
		auto target =
		    ImageVector::Allocate(result, row, box.width, box.height, ImageLogicalType::ModeName(image.layout.mode));
		PythonGILWrapper gil;
		try {
			auto input = py::memoryview::from_memory(image.data, py::ssize_t(image.layout.Size()));
			auto output = py::memoryview::from_memory(target, py::ssize_t(size), false);
			auto check = py::cpp_function([&context]() { ImageOperatorContract::Interrupt(context); });
			py::module_::import("vane._image_operators")
			    .attr("_crop_image")(input, image.layout.width, image.layout.height, image.layout.channels, box.x,
			                         box.y, box.width, box.height, output, check);
			ImageOperatorContract::Interrupt(context);
		} catch (py::error_already_set &error) {
			RaiseImageHelperError(context, error);
		}
	}
	if (constant && count) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void EncodeImage(DataChunk &args, ExpressionState &state, Vector &result) {
	auto constant = args.AllConstant();
	auto count = constant && args.size() ? idx_t(1) : args.size();
	ImageOperatorInput images(args.data[0], count);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	idx_t bytes = 0;
	for (idx_t row = 0; row < count; row++) {
		auto &context = state.GetContext();
		ImageOperatorContract::Interrupt(context);
		ImagePixelView image;
		if (images.IsNull(row) || !ImageOperatorContract::ReadPNGFormat(args.data[1], row) ||
		    !images.Read(row, image)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		PythonGILWrapper gil;
		try {
			auto input = py::memoryview::from_memory(image.data, py::ssize_t(image.layout.Size()));
			auto check = py::cpp_function([&context]() { ImageOperatorContract::Interrupt(context); });
			auto value =
			    py::module_::import("vane._image_operators")
			        .attr("_encode_image_png")(input, image.layout.width, image.layout.height, image.layout.channels,
			                                   ImageOperatorContract::MAX_BYTES - bytes, check);
			ImageOperatorContract::Interrupt(context);
			if (!py::isinstance<py::bytes>(value)) {
				throw InternalException("Python PNG encoder returned a non-bytes result");
			}
			auto size = idx_t(PyBytes_Size(value.ptr()));
			if (size > ImageOperatorContract::MAX_BYTES - bytes) {
				throw InternalException("Python PNG encoder violated its output byte limit");
			}
			bytes += size;
			FlatVector::GetData<string_t>(result)[row] =
			    StringVector::AddStringOrBlob(result, PyBytes_AsString(value.ptr()), size);
			FlatVector::SetNull(result, row, false);
		} catch (py::error_already_set &error) {
			RaiseImageHelperError(context, error);
		}
	}
	if (constant && count) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static ScalarFunction MakeImageFunction(const string &name, vector<LogicalType> arguments, LogicalType result,
                                        scalar_function_t execute, bind_scalar_function_t bind) {
	ScalarFunction function(name, std::move(arguments), std::move(result), execute, bind);
	function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	function.SetFallible();
	function.SetBindExpressionCallback(
	    [name](FunctionBindExpressionInput &input) { return MediaBackend::BindNative(input, "image", name); });
	return function;
}

} // namespace

ScalarFunctionSet ImageFunctions::GetCropFunctions() {
	ScalarFunctionSet result("crop");
	result.AddFunction(MakeImageFunction("crop", {LogicalType::ANY, LogicalType::ANY}, ImageLogicalType::Create(),
	                                     CropImage, ImageOperatorContract::BindCrop));
	return result;
}

ScalarFunctionSet ImageFunctions::GetEncodeFunctions() {
	ScalarFunctionSet result("encode_image");
	result.AddFunction(MakeImageFunction("encode_image", {LogicalType::ANY, LogicalType::VARCHAR}, LogicalType::BLOB,
	                                     EncodeImage, ImageOperatorContract::BindEncode));
	return result;
}

} // namespace duckdb
