// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/image_functions.hpp"

#include "image_operator_contract.hpp"
#include "image_codec_contract.hpp"
#include "image_transform_contract.hpp"
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
	ImageOperatorInput images(args.data[0], count, &state.GetContext());
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
		auto size = ImageOperatorContract::CheckSize(
		    box.width, box.height, image.layout.channels, ImageOperatorContract::MAX_BYTES - bytes,
		    GetTypeIdSize(ImageLogicalType::StorageType(result.GetType()).InternalType()));
		bytes += size;
		auto layout = image.layout;
		layout.width = box.width;
		layout.height = box.height;
		ImageOperatorOutput output_pixels(result, row, layout);
		auto target = output_pixels.Data();
		PythonGILWrapper gil;
		try {
			auto input = py::memoryview::from_memory(image.data, py::ssize_t(image.layout.Bytes()));
			auto output = py::memoryview::from_memory(target, py::ssize_t(layout.Bytes()), false);
			auto check = py::cpp_function([&context]() { ImageOperatorContract::Interrupt(context); });
			py::module_::import("vane._image_operators")
			    .attr("_crop_image")(input, image.layout.width, image.layout.height, image.layout.channels, box.x,
			                         box.y, box.width, box.height, output, check,
			                         ImageLogicalType::ModeName(image.layout.mode));
			output_pixels.Finish(context);
			ImageOperatorContract::Interrupt(context);
		} catch (py::error_already_set &error) {
			RaiseImageHelperError(context, error);
		}
	}
	if (constant && count) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void TransformImage(DataChunk &args, ExpressionState &state, Vector &result, ImageTransform operation) {
	auto &context = state.GetContext();
	ImageTransformContract::Execute(
	    args, context, result, operation,
	    [&context, operation](const ImagePixelView &image, const ImageLayout &layout, data_ptr_t target,
	                          bool antialias) {
		    PythonGILWrapper gil;
		    try {
			    auto input = py::memoryview::from_memory(image.data, py::ssize_t(image.layout.Bytes()));
			    auto output = py::memoryview::from_memory(target, py::ssize_t(layout.Bytes()), false);
			    auto check = py::cpp_function([&context]() { ImageOperatorContract::Interrupt(context); });
			    auto helpers = py::module_::import("vane._image_operators");
			    if (operation == ImageTransform::RESIZE) {
				    helpers.attr("_resize_image")(input, image.layout.width, image.layout.height, image.layout.channels,
				                                  layout.width, layout.height, output, check,
				                                  ImageLogicalType::ModeName(image.layout.mode), antialias);
			    } else {
				    helpers.attr("_convert_image")(
				        input, image.layout.width, image.layout.height, image.layout.channels, layout.channels, output,
				        check, ImageLogicalType::ModeName(image.layout.mode), ImageLogicalType::ModeName(layout.mode));
			    }
		    } catch (py::error_already_set &error) {
			    RaiseImageHelperError(context, error);
		    }
	    });
}

static void ResizeImage(DataChunk &args, ExpressionState &state, Vector &result) {
	TransformImage(args, state, result, ImageTransform::RESIZE);
}

static void ConvertImage(DataChunk &args, ExpressionState &state, Vector &result) {
	TransformImage(args, state, result, ImageTransform::CONVERT);
}

static void EncodeImage(DataChunk &args, ExpressionState &state, Vector &result) {
	auto constant = args.AllConstant();
	auto count = constant && args.size() ? idx_t(1) : args.size();
	ImageOperatorInput images(args.data[0], count, &state.GetContext());
	result.SetVectorType(VectorType::FLAT_VECTOR);
	idx_t bytes = 0;
	for (idx_t row = 0; row < count; row++) {
		auto &context = state.GetContext();
		ImageOperatorContract::Interrupt(context);
		ImagePixelView image;
		auto format_value = args.data[1].GetValue(row);
		if (images.IsNull(row) || format_value.IsNull() || !images.Read(row, image)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto format = ImageCodecContract::Format(format_value.GetValue<string>());
		ImageCodecContract::CheckEncoding(format, image.layout);
		PythonGILWrapper gil;
		try {
			auto input = py::memoryview::from_memory(image.data, py::ssize_t(image.layout.Bytes()));
			auto check = py::cpp_function([&context]() { ImageOperatorContract::Interrupt(context); });
			auto value = py::module_::import("vane._image_compute")
			                 .attr("_encode_image_bytes")(input, image.layout.width, image.layout.height,
			                                              ImageLogicalType::ModeName(image.layout.mode), format,
			                                              ImageOperatorContract::MAX_BYTES - bytes, check);
			ImageOperatorContract::Interrupt(context);
			if (!py::isinstance<py::bytes>(value)) {
				throw InternalException("Python Image encoder returned a non-bytes result");
			}
			auto size = idx_t(PyBytes_Size(value.ptr()));
			if (size > ImageOperatorContract::MAX_BYTES - bytes) {
				throw InternalException("Python Image encoder violated its output byte limit");
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
                                        scalar_function_t execute, bind_scalar_function_t bind,
                                        function_bind_expression_t bind_expression) {
	ScalarFunction function(name, std::move(arguments), std::move(result), execute, bind);
	function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	function.SetFallible();
	function.SetBindExpressionCallback(bind_expression);
	return function;
}

} // namespace

ScalarFunctionSet ImageFunctions::GetCropFunctions() {
	ScalarFunctionSet result("crop");
	result.AddFunction(MakeImageFunction(
	    "crop", {LogicalType::ANY, LogicalType::ANY}, ImageLogicalType::Create(), CropImage,
	    ImageOperatorContract::BindCrop,
	    [](FunctionBindExpressionInput &input) { return MediaBackend::BindNative(input, "image", "crop"); }));
	return result;
}

ScalarFunctionSet ImageFunctions::GetEncodeFunctions() {
	ScalarFunctionSet result("encode_image");
	result.AddFunction(MakeImageFunction(
	    "encode_image", {LogicalType::ANY, LogicalType::VARCHAR}, LogicalType::BLOB, EncodeImage,
	    ImageOperatorContract::BindEncode,
	    [](FunctionBindExpressionInput &input) { return MediaBackend::BindNative(input, "image", "encode_image"); }));
	return result;
}

ScalarFunctionSet ImageFunctions::GetResizeFunctions() {
	ScalarFunctionSet result("resize");
	for (idx_t count : {3, 4}) {
		result.AddFunction(MakeImageFunction(
		    "resize", vector<LogicalType>(count, LogicalType::ANY), ImageLogicalType::Create(), ResizeImage,
		    ImageTransformContract::BindResize,
		    [](FunctionBindExpressionInput &input) { return MediaBackend::BindNative(input, "image", "resize"); }));
	}
	return result;
}

ScalarFunctionSet ImageFunctions::GetConvertFunctions() {
	ScalarFunctionSet result("convert_image");
	result.AddFunction(MakeImageFunction(
	    "convert_image", {LogicalType::ANY, LogicalType::ANY}, ImageLogicalType::Create(), ConvertImage,
	    ImageTransformContract::BindConvert,
	    [](FunctionBindExpressionInput &input) { return MediaBackend::BindNative(input, "image", "convert_image"); }));
	return result;
}

} // namespace duckdb
