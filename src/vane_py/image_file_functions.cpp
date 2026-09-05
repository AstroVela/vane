// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/image_file_functions.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression.hpp"
#include "file_resolver.hpp"
#include "file_value.hpp"
#include "media_backend.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"

#include <exception>

namespace duckdb {

namespace {

static constexpr uint64_t DEFAULT_IMAGE_METADATA_BYTES = 1024 * 1024;
static constexpr uint64_t DEFAULT_IMAGE_PIXELS = 100000000;
static constexpr uint64_t MAX_IMAGE_METADATA_BYTES = 64 * 1024 * 1024;
static constexpr uint64_t DEFAULT_IMAGE_MAX_INPUT_BYTES = 256 * 1024 * 1024;
static constexpr uint64_t DEFAULT_IMAGE_MAX_DECODED_BYTES = 512 * 1024 * 1024;
static constexpr uint64_t MAX_IMAGE_BATCH_OUTPUT_BYTES = 256 * 1024 * 1024;
static constexpr idx_t IMAGE_RESULT_COPY_CHUNK_BYTES = 1024 * 1024;

static LogicalType ImageMetadataType() {
	child_list_t<LogicalType> fields;
	fields.emplace_back("width", LogicalType::UINTEGER);
	fields.emplace_back("height", LogicalType::UINTEGER);
	fields.emplace_back("format", LogicalType::VARCHAR);
	fields.emplace_back("mode", LogicalType::VARCHAR);
	return LogicalType::STRUCT(std::move(fields));
}

struct ImageMetadataResult {
	uint32_t width;
	uint32_t height;
	string format;
	string mode;

	Value ToValue() const {
		vector<Value> fields;
		fields.reserve(4);
		fields.push_back(Value::UINTEGER(width));
		fields.push_back(Value::UINTEGER(height));
		fields.push_back(Value(format));
		fields.push_back(Value(mode));
		return Value::STRUCT(ImageMetadataType(), std::move(fields));
	}
};

static unique_ptr<FunctionData> BindImageFileMetadata(ClientContext &, ScalarFunction &bound_function,
                                                      vector<unique_ptr<Expression>> &arguments) {
	auto image_type = FileLogicalType::Create(FileMediaType::IMAGE);
	auto input_type = arguments[0]->return_type;
	if (input_type.id() == LogicalTypeId::UNKNOWN || input_type.id() == LogicalTypeId::SQLNULL) {
		input_type = image_type;
	}
	if (!FileLogicalType::IsFile(input_type) || FileLogicalType::GetMediaType(input_type) != FileMediaType::IMAGE) {
		throw BinderException("image_file_metadata() requires IMAGEFILE, not %s", input_type.ToString());
	}
	bound_function.arguments[0] = std::move(input_type);
	return nullptr;
}

static unique_ptr<FunctionData> BindDecodeImageFile(ClientContext &, ScalarFunction &bound_function,
                                                    vector<unique_ptr<Expression>> &arguments) {
	auto image_file_type = FileLogicalType::Create(FileMediaType::IMAGE);
	auto input_type = arguments[0]->return_type;
	if (input_type.id() == LogicalTypeId::UNKNOWN || input_type.id() == LogicalTypeId::SQLNULL) {
		input_type = image_file_type;
	}
	if (!FileLogicalType::IsFile(input_type) || FileLogicalType::GetMediaType(input_type) != FileMediaType::IMAGE) {
		throw BinderException("decode_image_file() requires IMAGEFILE, not %s", input_type.ToString());
	}
	bound_function.arguments[0] = std::move(input_type);
	return nullptr;
}

enum class ImageDecodeOnError : uint8_t { RAISE, NULL_VALUE };

struct ImageDecodeArguments {
	bool has_mode = false;
	string mode;
	ImageDecodeOnError on_error = ImageDecodeOnError::RAISE;
	uint64_t max_input_bytes = DEFAULT_IMAGE_MAX_INPUT_BYTES;
	uint64_t max_pixels = DEFAULT_IMAGE_PIXELS;
	uint64_t max_decoded_bytes = DEFAULT_IMAGE_MAX_DECODED_BYTES;
};

static bool IsImageResultMode(const string &mode) {
	return mode == "L" || mode == "LA" || mode == "RGB" || mode == "RGBA";
}

static bool GetImageDecodeArguments(DataChunk &args, idx_t row, ImageDecodeArguments &result) {
	if (args.ColumnCount() >= 2) {
		auto mode = args.data[1].GetValue(row);
		if (!mode.IsNull()) {
			result.has_mode = true;
			result.mode = mode.GetValue<string>();
			if (!IsImageResultMode(result.mode)) {
				throw InvalidInputException("decode_image_file() mode must be one of L, LA, RGB, or RGBA");
			}
		}
	}
	if (args.ColumnCount() >= 3) {
		auto on_error = args.data[2].GetValue(row);
		if (on_error.IsNull()) {
			return false;
		}
		auto policy = on_error.GetValue<string>();
		if (policy == "raise") {
			result.on_error = ImageDecodeOnError::RAISE;
		} else if (policy == "null") {
			result.on_error = ImageDecodeOnError::NULL_VALUE;
		} else {
			throw InvalidInputException("decode_image_file() on_error must be 'raise' or 'null'");
		}
	}
	if (args.ColumnCount() == 6) {
		uint64_t *limits[] = {&result.max_input_bytes, &result.max_pixels, &result.max_decoded_bytes};
		const char *names[] = {"max_input_bytes", "max_pixels", "max_decoded_bytes"};
		for (idx_t index = 0; index < 3; index++) {
			auto value = args.data[index + 3].GetValue(row);
			if (value.IsNull()) {
				return false;
			}
			auto limit = value.GetValue<uint64_t>();
			if (limit == 0) {
				throw InvalidInputException("decode_image_file() %s must be greater than zero", names[index]);
			}
			*limits[index] = limit;
		}
	}
	return true;
}

static ImageMetadataResult ProbeImageMetadata(const string &bytes, uint64_t max_pixels, bool truncated,
                                              const FileReference &file, uint64_t max_metadata_bytes) {
	PythonGILWrapper gil;
	py::object image_file_error_type;
	try {
		auto module = py::module_::import("vane._image_file");
		image_file_error_type = module.attr("ImageFileError");
		auto helper = module.attr("_probe_image_metadata");
		py::object content_type = file.has_content_type ? py::cast(file.content_type) : py::none();
		auto value = helper(py::bytes(bytes), py::int_(max_pixels), py::bool_(truncated), std::move(content_type),
		                    py::int_(max_metadata_bytes));
		if (!py::isinstance<py::tuple>(value)) {
			throw InternalException("Image metadata helper returned a non-tuple value");
		}
		auto fields = py::reinterpret_borrow<py::tuple>(value);
		if (fields.size() != 4 || !py::isinstance<py::int_>(fields[0]) || !py::isinstance<py::int_>(fields[1]) ||
		    !py::isinstance<py::str>(fields[2]) || !py::isinstance<py::str>(fields[3])) {
			throw InternalException("Image metadata helper returned an invalid value");
		}
		auto width = py::cast<uint64_t>(fields[0]);
		auto height = py::cast<uint64_t>(fields[1]);
		if (width > NumericLimits<uint32_t>::Maximum() || height > NumericLimits<uint32_t>::Maximum()) {
			throw OutOfRangeException("Image dimensions do not fit the image_file_metadata() result type");
		}
		return {NumericCast<uint32_t>(width), NumericCast<uint32_t>(height), py::cast<string>(fields[2]),
		        py::cast<string>(fields[3])};
	} catch (py::error_already_set &error) {
		if (error.matches(PyExc_KeyboardInterrupt)) {
			throw InterruptException();
		}
		if (error.matches(PyExc_MemoryError)) {
			throw OutOfMemoryException("Image metadata inspection ran out of memory");
		}
		if (error.matches(PyExc_ImportError) ||
		    (image_file_error_type.ptr() && error.matches(image_file_error_type.ptr()))) {
			throw InvalidInputException("image_file_metadata() failed: %s", error.what());
		}
		throw InternalException("image_file_metadata() Python helper failed unexpectedly: %s", error.what());
	}
}

static void RequireImageDependencies(const char *function_name) {
	PythonGILWrapper gil;
	try {
		auto module = py::module_::import("vane._image_file");
		module.attr("_load_pillow")();
	} catch (py::error_already_set &error) {
		if (!error.matches(PyExc_Exception)) {
			throw InterruptException();
		}
		if (error.matches(PyExc_MemoryError)) {
			throw OutOfMemoryException("Image dependency preflight ran out of memory");
		}
		if (error.matches(PyExc_ImportError)) {
			throw InvalidInputException("%s() failed: %s", function_name, error.what());
		}
		throw InternalException("%s() dependency preflight failed unexpectedly: %s", function_name, error.what());
	}
}

[[noreturn]] static void RethrowImageReadError(const std::exception_ptr &read_error) {
	try {
		std::rethrow_exception(read_error);
	} catch (py::error_already_set &error) {
		if (!error.matches(PyExc_Exception)) {
			throw InterruptException();
		}
		if (error.matches(PyExc_MemoryError)) {
			throw OutOfMemoryException("Image source read ran out of memory");
		}
		throw;
	}
}

class PythonImageSpoolGuard {
public:
	explicit PythonImageSpoolGuard(py::object spool_p) : spool(std::move(spool_p)) {
	}

	~PythonImageSpoolGuard() {
		if (closed) {
			return;
		}
		try {
			spool.attr("close")();
		} catch (py::error_already_set &error) {
			error.discard_as_unraisable("closing decoded image spool after failure");
		} catch (...) {
			// Destructors must not replace the active decode exception.
		}
	}

	void Close() {
		spool.attr("close")();
		closed = true;
	}

private:
	py::object spool;
	bool closed = false;
};

static uint64_t ImageSpoolUnsigned(const py::handle &value, const char *field) {
	if (!py::isinstance<py::int_>(value) || py::isinstance<py::bool_>(value)) {
		throw InternalException("Image decode spool returned a non-integer %s", field);
	}
	try {
		return py::cast<uint64_t>(value);
	} catch (py::cast_error &) {
		throw InternalException("Image decode spool returned an out-of-range %s", field);
	}
}

static int64_t ImageSpoolReadCount(const py::handle &value) {
	if (!py::isinstance<py::int_>(value) || py::isinstance<py::bool_>(value)) {
		throw InternalException("Image decode spool returned a non-integer read size");
	}
	try {
		return py::cast<int64_t>(value);
	} catch (py::cast_error &) {
		throw InternalException("Image decode spool returned an out-of-range read size");
	}
}

static bool CopyDecodedImage(ClientContext &context, ResolvedFile &resolved, const FileReference &file,
                             const ImageDecodeArguments &arguments, uint64_t remaining_batch_bytes, idx_t row,
                             Vector &data_result, Vector &width_result, Vector &height_result, Vector &channels_result,
                             Vector &mode_result, uint64_t &output_bytes) {
	PythonGILWrapper gil;
	py::object image_file_error_type;
	py::object image_file_format_error_type;
	std::exception_ptr read_error;
	try {
		auto module = py::module_::import("vane._image_file");
		image_file_error_type = module.attr("ImageFileError");
		image_file_format_error_type = module.attr("ImageFileFormatError");
		auto spool_type = module.attr("_DecodedImageSpool");
		auto helper = module.attr("_decode_image_stream");
		auto read_at = py::cpp_function([&context, &resolved, &read_error](uint64_t offset, uint64_t size) {
			string bytes;
			try {
				if (context.IsInterrupted()) {
					throw InterruptException();
				}
				bytes.resize(NumericCast<idx_t>(size));
				if (size > 0) {
					py::gil_scoped_release release;
					resolved.ReadExact(reinterpret_cast<data_ptr_t>(bytes.data()), size, offset);
				}
				if (context.IsInterrupted()) {
					throw InterruptException();
				}
			} catch (...) {
				read_error = std::current_exception();
				return py::bytes();
			}
			return py::bytes(bytes);
		});
		auto check_interrupted = py::cpp_function([&context]() {
			if (context.IsInterrupted()) {
				throw InterruptException();
			}
		});
		py::object content_type = file.has_content_type ? py::cast(file.content_type) : py::none();
		py::object mode = arguments.has_mode ? py::cast(arguments.mode) : py::none();
		auto value = helper(std::move(read_at), py::int_(resolved.LogicalSize()), std::move(content_type),
		                    std::move(mode), py::int_(arguments.max_input_bytes), py::int_(arguments.max_pixels),
		                    py::int_(arguments.max_decoded_bytes), py::int_(remaining_batch_bytes),
		                    std::move(check_interrupted));
		if (read_error) {
			RethrowImageReadError(read_error);
		}
		if (!py::isinstance(value, spool_type)) {
			throw InternalException("Image decode helper returned an invalid spool");
		}
		PythonImageSpoolGuard spool_guard(value);
		auto width = ImageSpoolUnsigned(value.attr("width"), "width");
		auto height = ImageSpoolUnsigned(value.attr("height"), "height");
		auto data_size = ImageSpoolUnsigned(value.attr("data_size"), "data size");
		auto mode_value = value.attr("mode");
		if (!py::isinstance<py::str>(mode_value)) {
			throw InternalException("Image decode spool returned a non-string mode");
		}
		auto output_mode = py::cast<string>(mode_value);
		if (width == 0 || width > NumericLimits<uint32_t>::Maximum() || height == 0 ||
		    height > NumericLimits<uint32_t>::Maximum() || !IsImageResultMode(output_mode)) {
			throw InternalException("Image decode spool returned out-of-range dimensions or mode");
		}
		auto channels = ImageLogicalType::ChannelsForMode(output_mode);
		if (width > NumericLimits<uint64_t>::Maximum() / height ||
		    width * height > NumericLimits<uint64_t>::Maximum() / channels) {
			throw InternalException("Image decode spool returned overflowing dimensions");
		}
		auto expected_bytes = width * height * channels;
		if (data_size != expected_bytes || data_size > remaining_batch_bytes || data_size > string_t::MAX_STRING_SIZE ||
		    data_size > NumericLimits<idx_t>::Maximum()) {
			throw InternalException("Image decode helper violated its output contract");
		}
		auto result_width = NumericCast<uint32_t>(width);
		auto result_height = NumericCast<uint32_t>(height);
		ImageLogicalType::ValidateFields(NumericCast<idx_t>(data_size), result_width, result_height, channels,
		                                 output_mode, "decode_image_file");

		auto data = StringVector::EmptyString(data_result, NumericCast<idx_t>(data_size));
		auto target_data = data.GetDataWriteable();
		for (uint64_t copied = 0; copied < data_size;) {
			if (context.IsInterrupted()) {
				throw InterruptException();
			}
			auto copy_bytes = MinValue<uint64_t>(data_size - copied, IMAGE_RESULT_COPY_CHUNK_BYTES);
			auto target =
			    py::memoryview::from_memory(target_data + copied, NumericCast<py::ssize_t>(copy_bytes), false);
			auto read_value = value.attr("readinto")(std::move(target));
			auto read_count = ImageSpoolReadCount(read_value);
			if (read_count <= 0 || static_cast<uint64_t>(read_count) > copy_bytes) {
				throw InternalException("Image decode spool ended before the declared output");
			}
			copied += static_cast<uint64_t>(read_count);
		}
		char extra_byte;
		auto extra_target = py::memoryview::from_memory(&extra_byte, 1, false);
		auto extra_value = value.attr("readinto")(std::move(extra_target));
		if (ImageSpoolReadCount(extra_value) != 0) {
			throw InternalException("Image decode spool contains more data than declared");
		}
		spool_guard.Close();
		data.Finalize();

		FlatVector::GetData<string_t>(data_result)[row] = data;
		FlatVector::GetData<uint32_t>(width_result)[row] = result_width;
		FlatVector::GetData<uint32_t>(height_result)[row] = result_height;
		FlatVector::GetData<uint8_t>(channels_result)[row] = channels;
		FlatVector::GetData<string_t>(mode_result)[row] = StringVector::AddString(mode_result, output_mode);
		FlatVector::Validity(data_result).SetValid(row);
		FlatVector::Validity(width_result).SetValid(row);
		FlatVector::Validity(height_result).SetValid(row);
		FlatVector::Validity(channels_result).SetValid(row);
		FlatVector::Validity(mode_result).SetValid(row);
		output_bytes = data_size;
		return true;
	} catch (py::error_already_set &error) {
		if (context.IsInterrupted() || !error.matches(PyExc_Exception)) {
			throw InterruptException();
		}
		if (read_error) {
			RethrowImageReadError(read_error);
		}
		if (error.matches(PyExc_MemoryError)) {
			throw OutOfMemoryException("Image decoding ran out of memory");
		}
		if (image_file_format_error_type.ptr() && error.matches(image_file_format_error_type.ptr()) &&
		    arguments.on_error == ImageDecodeOnError::NULL_VALUE) {
			return false;
		}
		if (error.matches(PyExc_ImportError) ||
		    (image_file_error_type.ptr() && error.matches(image_file_error_type.ptr()))) {
			throw InvalidInputException("decode_image_file() failed: %s", error.what());
		}
		if (error.matches(PyExc_OSError)) {
			throw IOException("decode_image_file() failed: %s", error.what());
		}
		throw InternalException("decode_image_file() Python helper failed unexpectedly: %s", error.what());
	}
}

static void DecodeImageFileFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &children = StructVector::GetEntries(result);
	D_ASSERT(children.size() == ImageLogicalType::FIELD_COUNT);
	auto &data_result = *children[ImageLogicalType::DATA];
	auto &width_result = *children[ImageLogicalType::WIDTH];
	auto &height_result = *children[ImageLogicalType::HEIGHT];
	auto &channels_result = *children[ImageLogicalType::CHANNELS];
	auto &mode_result = *children[ImageLogicalType::MODE];
	for (auto &child : children) {
		child->SetVectorType(VectorType::FLAT_VECTOR);
	}

	bool dependency_ready = false;
	uint64_t batch_output_bytes = 0;
	for (idx_t row = 0; row < args.size(); row++) {
		auto file_value = args.data[0].GetValue(row);
		ImageDecodeArguments arguments;
		if (file_value.IsNull() || !GetImageDecodeArguments(args, row, arguments)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}

		auto &context = state.GetContext();
		try {
			if (!dependency_ready) {
				RequireImageDependencies("decode_image_file");
				dependency_ready = true;
			}
			auto file = FileReference::FromValue(file_value, "decode_image_file");
			auto resolved = ResolvedFile::Open(context, file);
			if (batch_output_bytes > MAX_IMAGE_BATCH_OUTPUT_BYTES) {
				throw InternalException("decode_image_file() exceeded its batch output invariant");
			}
			uint64_t row_output_bytes = 0;
			auto decoded = CopyDecodedImage(
			    context, *resolved, file, arguments, MAX_IMAGE_BATCH_OUTPUT_BYTES - batch_output_bytes, row,
			    data_result, width_result, height_result, channels_result, mode_result, row_output_bytes);
			if (context.IsInterrupted()) {
				throw InterruptException();
			}
			if (!decoded) {
				FlatVector::SetNull(result, row, true);
				continue;
			}
			batch_output_bytes += row_output_bytes;
			FlatVector::SetNull(result, row, false);
		} catch (...) {
			if (context.IsInterrupted()) {
				throw InterruptException();
			}
			throw;
		}
	}
	result.Verify(args.size());
}

static void ImageFileMetadataFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	for (idx_t row = 0; row < args.size(); row++) {
		auto file_value = args.data[0].GetValue(row);
		if (file_value.IsNull()) {
			result.SetValue(row, Value(ImageMetadataType()));
			continue;
		}

		uint64_t max_metadata_bytes = DEFAULT_IMAGE_METADATA_BYTES;
		uint64_t max_pixels = DEFAULT_IMAGE_PIXELS;
		if (args.ColumnCount() == 3) {
			auto max_metadata_value = args.data[1].GetValue(row);
			auto max_pixels_value = args.data[2].GetValue(row);
			if (max_metadata_value.IsNull() || max_pixels_value.IsNull()) {
				result.SetValue(row, Value(ImageMetadataType()));
				continue;
			}
			max_metadata_bytes = max_metadata_value.GetValue<uint64_t>();
			max_pixels = max_pixels_value.GetValue<uint64_t>();
		}
		if (max_metadata_bytes == 0 || max_metadata_bytes > MAX_IMAGE_METADATA_BYTES) {
			throw InvalidInputException("image_file_metadata() max_bytes must be between 1 and %llu",
			                            static_cast<unsigned long long>(MAX_IMAGE_METADATA_BYTES));
		}
		if (max_pixels == 0) {
			throw InvalidInputException("image_file_metadata() max_pixels must be greater than zero");
		}

		auto file = FileReference::FromValue(file_value, "image_file_metadata");
		auto resolved = ResolvedFile::Open(state.GetContext(), file);
		auto logical_size = resolved->LogicalSize();
		auto read_size = MinValue<uint64_t>(logical_size, max_metadata_bytes);
		string bytes(NumericCast<idx_t>(read_size), '\0');
		if (read_size > 0) {
			resolved->ReadExact(reinterpret_cast<data_ptr_t>(bytes.data()), read_size);
		}
		auto metadata = ProbeImageMetadata(bytes, max_pixels, logical_size > read_size, file, max_metadata_bytes);
		if (state.GetContext().IsInterrupted()) {
			throw InterruptException();
		}
		result.SetValue(row, metadata.ToValue());
	}
}

static ScalarFunction MakeImageMetadataFunction(vector<LogicalType> arguments) {
	ScalarFunction function("image_file_metadata", std::move(arguments), ImageMetadataType(), ImageFileMetadataFunction,
	                        BindImageFileMetadata);
	function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	function.SetStability(FunctionStability::VOLATILE);
	function.SetFallible();
	function.SetBindExpressionCallback([](FunctionBindExpressionInput &input) {
		return MediaBackend::BindNative(input, "image", "image_file_metadata");
	});
	return function;
}

static ScalarFunction MakeDecodeImageFileFunction(vector<LogicalType> arguments) {
	ScalarFunction function("decode_image_file", std::move(arguments), ImageLogicalType::Create(),
	                        DecodeImageFileFunction, BindDecodeImageFile);
	function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	function.SetStability(FunctionStability::VOLATILE);
	function.SetFallible();
	function.SetBindExpressionCallback([](FunctionBindExpressionInput &input) {
		return MediaBackend::BindNative(input, "image", "decode_image_file");
	});
	return function;
}

} // namespace

ScalarFunctionSet ImageFileFunctions::GetFunctions() {
	ScalarFunctionSet result("image_file_metadata");
	result.AddFunction(MakeImageMetadataFunction({LogicalType::ANY}));
	result.AddFunction(MakeImageMetadataFunction({LogicalType::ANY, LogicalType::UBIGINT, LogicalType::UBIGINT}));
	return result;
}

ScalarFunctionSet ImageFileFunctions::GetDecodeFunctions() {
	ScalarFunctionSet result("decode_image_file");
	result.AddFunction(MakeDecodeImageFileFunction({LogicalType::ANY}));
	result.AddFunction(MakeDecodeImageFileFunction({LogicalType::ANY, LogicalType::VARCHAR}));
	result.AddFunction(MakeDecodeImageFileFunction({LogicalType::ANY, LogicalType::VARCHAR, LogicalType::VARCHAR}));
	result.AddFunction(MakeDecodeImageFileFunction({LogicalType::ANY, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                                                LogicalType::UBIGINT, LogicalType::UBIGINT, LogicalType::UBIGINT}));
	return result;
}

} // namespace duckdb
