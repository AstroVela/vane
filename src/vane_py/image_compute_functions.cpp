// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/image_functions.hpp"
#include "vane_python/image.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"
#include "image_codec_contract.hpp"
#include "image_hash_contract.hpp"
#include "media_backend.hpp"
#include "duckdb/execution/expression_executor_state.hpp"
#include "duckdb/function/scalar_macro_function.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

namespace duckdb {
namespace {

[[noreturn]] static void HelperError(ClientContext &context, py::error_already_set &error) {
	if (context.IsInterrupted() || !error.matches(PyExc_Exception)) {
		throw InterruptException();
	}
	if (error.matches(PyExc_MemoryError)) {
		throw OutOfMemoryException("Python Image computation ran out of memory");
	}
	if (error.matches(PyExc_OverflowError)) {
		throw OutOfRangeException("Python Image computation exceeded its resource limit: %s", error.what());
	}
	if (error.matches(PyExc_ImportError)) {
		throw InvalidInputException("Python Image computation requires vane-ai[image] dependencies: %s", error.what());
	}
	if (error.matches(PyExc_OSError)) {
		throw IOException("Python Image computation failed: %s", error.what());
	}
	throw InternalException("Python Image computation failed unexpectedly: %s", error.what());
}

static void Decode(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &context = state.GetContext();
	auto constant = args.AllConstant();
	auto count = constant && args.size() ? idx_t(1) : args.size();
	result.SetVectorType(VectorType::FLAT_VECTOR);
	UnifiedVectorFormat encoded, policies, modes;
	args.data[0].ToUnifiedFormat(count, encoded);
	args.data[1].ToUnifiedFormat(count, policies);
	args.data[2].ToUnifiedFormat(count, modes);
	idx_t retained = 0;
	for (idx_t row = 0; row < count; row++) {
		ImageOperatorContract::Interrupt(context);
		auto source_index = encoded.sel->get_index(row), policy_index = policies.sel->get_index(row);
		auto mode_index = modes.sel->get_index(row);
		if (!encoded.validity.RowIsValid(source_index) || !policies.validity.RowIsValid(policy_index)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto policy = UnifiedVectorFormat::GetData<string_t>(policies)[policy_index].GetString();
		auto null_on_error = ImageCodecContract::OnError(policy);
		string mode;
		if (modes.validity.RowIsValid(mode_index)) {
			mode = UnifiedVectorFormat::GetData<string_t>(modes)[mode_index].GetString();
			ImageLogicalType::ModeCode(mode);
		}
		auto input = UnifiedVectorFormat::GetData<string_t>(encoded)[source_index];
		if (input.GetSize() > ImageOperatorContract::MAX_BYTES) {
			throw OutOfRangeException("decode_image exceeds its encoded input byte limit");
		}
		PythonGILWrapper gil;
		py::object content_error;
		try {
			auto module = py::module_::import("vane._image_compute");
			content_error = module.attr("ImageDecodeContentError");
			auto check = py::cpp_function([&context]() { ImageOperatorContract::Interrupt(context); });
			auto bytes = py::memoryview::from_memory(input.GetData(), py::ssize_t(input.GetSize()));
			auto value = module.attr("_decode_image_bytes")(
			    bytes, mode.empty() ? py::none() : py::cast(mode), ImageOperatorContract::MAX_BYTES - retained,
			    GetTypeIdSize(ImageLogicalType::StorageType(result.GetType()).InternalType()), check);
			if (!py::isinstance<py::tuple>(value) || py::len(value) != 2) {
				throw InternalException("Image decoder returned an invalid result");
			}
			auto tuple = value.cast<py::tuple>();
			if (!py::isinstance<py::array>(tuple[0]) || !py::isinstance<py::str>(tuple[1])) {
				throw InternalException("Image decoder returned an invalid pixel payload");
			}
			auto pixels = tuple[0].cast<py::array>();
			auto output_mode = tuple[1].cast<string>();
			auto channels = ImageLogicalType::ChannelsForMode(output_mode);
			auto dtype = ImageLogicalType::ModeCode(output_mode) <= 4   ? py::dtype::of<uint8_t>()
			             : ImageLogicalType::ModeCode(output_mode) <= 8 ? py::dtype::of<uint16_t>()
			                                                            : py::dtype::of<float>();
			if (pixels.ndim() != 3 || pixels.shape(2) != channels || !pixels.dtype().equal(dtype) ||
			    !(pixels.flags() & py::array::c_style) || pixels.shape(0) <= 0 || pixels.shape(1) <= 0 ||
			    pixels.shape(0) > UINT32_MAX || pixels.shape(1) > UINT32_MAX ||
			    (!mode.empty() && output_mode != mode)) {
				throw InternalException("Image decoder violated its pixel dtype or layout contract");
			}
			ImageLayout layout {uint32_t(pixels.shape(1)), uint32_t(pixels.shape(0)), channels,
			                    ImageLogicalType::ModeCode(output_mode)};
			retained +=
			    ImageCodecContract::OutputSize(result.GetType(), layout, ImageOperatorContract::MAX_BYTES - retained);
			ImageOperatorOutput output(result, row, layout);
			for (idx_t offset = 0; offset < layout.Bytes();) {
				ImageOperatorContract::Interrupt(context);
				auto size = MinValue(ImageOperatorContract::COPY_BYTES, layout.Bytes() - offset);
				memcpy(output.Data() + offset, const_data_ptr_cast(pixels.data()) + offset, size);
				offset += size;
			}
			output.Finish(context);
		} catch (py::error_already_set &error) {
			if (context.IsInterrupted()) {
				throw InterruptException();
			}
			if (content_error.ptr() && error.matches(content_error.ptr())) {
				if (null_on_error) {
					FlatVector::SetNull(result, row, true);
					continue;
				}
				throw InvalidInputException("decode_image failed: %s", error.what());
			}
			HelperError(context, error);
		}
	}
	if (constant && count) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void HashImage(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &context = state.GetContext();
	auto &options = state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<ImageHashOptions>();
	auto constant = args.AllConstant();
	auto count = constant && args.size() ? idx_t(1) : args.size();
	result.SetVectorType(VectorType::FLAT_VECTOR);
	ImageOperatorInput images(args.data[0], count, &context);
	for (idx_t row = 0; row < count; row++) {
		ImageOperatorContract::Interrupt(context);
		ImagePixelView image;
		if (!images.Read(row, image)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		if (options.method == "crop_resistant" &&
		    (image.layout.width < options.segments || image.layout.height < options.segments)) {
			throw InvalidInputException("crop_resistant requires at least one pixel per grid segment");
		}
		PythonGILWrapper gil;
		try {
			auto input = py::memoryview::from_memory(image.data, py::ssize_t(image.layout.Bytes()));
			auto check = py::cpp_function([&context]() { ImageOperatorContract::Interrupt(context); });
			auto value = py::module_::import("vane._image_compute")
			                 .attr("_image_hash")(input, image.layout.width, image.layout.height,
			                                      ImageLogicalType::ModeName(image.layout.mode), options.method,
			                                      options.hash_size, options.binbits, options.segments, check);
			if (!py::isinstance<py::bytes>(value) || idx_t(PyBytes_GET_SIZE(value.ptr())) != options.Bytes()) {
				throw InternalException("Python Image hash violated its fixed byte width");
			}
			FlatVector::GetData<string_t>(result)[row] =
			    StringVector::AddStringOrBlob(result, PyBytes_AS_STRING(value.ptr()), options.Bytes());
			FlatVector::SetNull(result, row, false);
		} catch (py::error_already_set &error) {
			HelperError(context, error);
		}
	}
	if (constant && count) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static unique_ptr<CreateMacroInfo> ComputeMacro(const string &name, const vector<string> &names,
                                                const vector<Value> &defaults) {
	vector<unique_ptr<ParsedExpression>> arguments;
	for (auto &parameter : names) {
		arguments.push_back(make_uniq<ColumnRefExpression>(parameter));
	}
	auto macro = make_uniq<ScalarMacroFunction>(make_uniq<FunctionExpression>("_vane_" + name, std::move(arguments)));
	for (idx_t i = 0; i < names.size(); i++) {
		macro->parameters.push_back(make_uniq<ColumnRefExpression>(names[i]));
		macro->types.push_back(LogicalType::UNKNOWN);
		if (i) {
			macro->default_parameters.insert(make_pair(names[i], make_uniq<ConstantExpression>(defaults[i - 1])));
		}
	}
	auto info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	info->schema = DEFAULT_SCHEMA;
	info->name = name;
	info->temporary = true;
	info->internal = true;
	info->macros.push_back(std::move(macro));
	return info;
}

static ScalarFunctionSet ComputeFunction(const string &name, vector<LogicalType> arguments, scalar_function_t execute,
                                         bind_scalar_function_t bind, function_bind_expression_t callback) {
	ScalarFunctionSet set("_vane_" + name);
	ScalarFunction function("_vane_" + name, std::move(arguments), LogicalType::ANY, execute, bind);
	function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	function.SetStability(FunctionStability::VOLATILE);
	function.SetFallible();
	function.SetBindExpressionCallback(callback);
	set.AddFunction(std::move(function));
	return set;
}
} // namespace

ScalarFunctionSet ImageFunctions::GetDecodeFunctions() {
	return ComputeFunction("decode_image", {LogicalType::ANY, LogicalType::VARCHAR, LogicalType::VARCHAR}, Decode,
	                       ImageCodecContract::BindDecode, [](FunctionBindExpressionInput &input) {
		                       return MediaBackend::BindNative(input, "image", "decode_image");
	                       });
}

ScalarFunctionSet ImageFunctions::GetHashFunctions() {
	return ComputeFunction(
	    "image_hash", {LogicalType::ANY, LogicalType::VARCHAR, LogicalType::ANY, LogicalType::ANY, LogicalType::ANY},
	    HashImage, ImageHashOptions::Bind,
	    [](FunctionBindExpressionInput &input) { return MediaBackend::BindNative(input, "image", "image_hash"); });
}

vector<unique_ptr<CreateMacroInfo>> ImageFunctions::GetMacros() {
	vector<unique_ptr<CreateMacroInfo>> result;
	result.push_back(ComputeMacro("decode_image", {"bytes", "on_error", "mode"}, {Value("raise"), Value("RGB")}));
	result.push_back(ComputeMacro("image_hash", {"image", "method", "hash_size", "binbits", "segments"},
	                              {Value("phash"), Value::BIGINT(8), Value::BIGINT(3), Value::BIGINT(3)}));
	return result;
}

} // namespace duckdb
