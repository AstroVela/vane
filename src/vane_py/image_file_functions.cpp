// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/image_file_functions.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression.hpp"
#include "file_resolver.hpp"
#include "file_value.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"

namespace duckdb {

namespace {

static constexpr uint64_t DEFAULT_IMAGE_METADATA_BYTES = 1024 * 1024;
static constexpr uint64_t DEFAULT_IMAGE_PIXELS = 100000000;
static constexpr uint64_t MAX_IMAGE_METADATA_BYTES = 64 * 1024 * 1024;

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
			resolved->ReadExact(reinterpret_cast<data_ptr_t>(&bytes[0]), read_size);
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
	return function;
}

} // namespace

ScalarFunctionSet ImageFileFunctions::GetFunctions() {
	ScalarFunctionSet result("image_file_metadata");
	result.AddFunction(MakeImageMetadataFunction({LogicalType::ANY}));
	result.AddFunction(MakeImageMetadataFunction({LogicalType::ANY, LogicalType::UBIGINT, LogicalType::UBIGINT}));
	return result;
}

} // namespace duckdb
