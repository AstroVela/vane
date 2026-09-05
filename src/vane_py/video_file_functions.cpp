// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/video_file_functions.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression.hpp"
#include "file_resolver.hpp"
#include "file_value.hpp"
#include "media_backend.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"

#include <cmath>
#include <exception>

namespace duckdb {

namespace {

static constexpr uint64_t DEFAULT_VIDEO_METADATA_BYTES = 8 * 1024 * 1024;
static constexpr uint64_t MAX_VIDEO_METADATA_BYTES = 64 * 1024 * 1024;

static LogicalType VideoTimeBaseType() {
	child_list_t<LogicalType> fields;
	fields.emplace_back("numerator", LogicalType::BIGINT);
	fields.emplace_back("denominator", LogicalType::BIGINT);
	return LogicalType::STRUCT(std::move(fields));
}

static LogicalType VideoMetadataType() {
	child_list_t<LogicalType> fields;
	fields.emplace_back("width", LogicalType::UINTEGER);
	fields.emplace_back("height", LogicalType::UINTEGER);
	fields.emplace_back("fps", LogicalType::DOUBLE);
	fields.emplace_back("duration", LogicalType::DOUBLE);
	fields.emplace_back("frame_count", LogicalType::BIGINT);
	fields.emplace_back("time_base", VideoTimeBaseType());
	return LogicalType::STRUCT(std::move(fields));
}

struct VideoMetadataResult {
	uint32_t width;
	uint32_t height;
	bool has_fps;
	double fps;
	bool has_duration;
	double duration;
	bool has_frame_count;
	int64_t frame_count;
	int64_t time_base_numerator;
	int64_t time_base_denominator;

	Value ToValue() const {
		vector<Value> time_base_fields;
		time_base_fields.reserve(2);
		time_base_fields.push_back(Value::BIGINT(time_base_numerator));
		time_base_fields.push_back(Value::BIGINT(time_base_denominator));

		vector<Value> fields;
		fields.reserve(6);
		fields.push_back(Value::UINTEGER(width));
		fields.push_back(Value::UINTEGER(height));
		fields.push_back(has_fps ? Value::DOUBLE(fps) : Value(LogicalType::DOUBLE));
		fields.push_back(has_duration ? Value::DOUBLE(duration) : Value(LogicalType::DOUBLE));
		fields.push_back(has_frame_count ? Value::BIGINT(frame_count) : Value(LogicalType::BIGINT));
		fields.push_back(Value::STRUCT(VideoTimeBaseType(), std::move(time_base_fields)));
		return Value::STRUCT(VideoMetadataType(), std::move(fields));
	}
};

static unique_ptr<FunctionData> BindVideoMetadata(ClientContext &, ScalarFunction &bound_function,
                                                  vector<unique_ptr<Expression>> &arguments) {
	auto video_type = FileLogicalType::Create(FileMediaType::VIDEO);
	auto input_type = arguments[0]->return_type;
	if (input_type.id() == LogicalTypeId::UNKNOWN || input_type.id() == LogicalTypeId::SQLNULL) {
		input_type = video_type;
	}
	if (!FileLogicalType::IsFile(input_type) || FileLogicalType::GetMediaType(input_type) != FileMediaType::VIDEO) {
		throw BinderException("video_metadata() requires VIDEOFILE, not %s", input_type.ToString());
	}
	bound_function.arguments[0] = std::move(input_type);
	return nullptr;
}

static void RequireVideoDependency() {
	PythonGILWrapper gil;
	try {
		py::module_::import("vane._video_file").attr("_load_av")();
	} catch (py::error_already_set &error) {
		if (!error.matches(PyExc_Exception)) {
			throw InterruptException();
		}
		if (error.matches(PyExc_MemoryError)) {
			throw OutOfMemoryException("Video dependency preflight ran out of memory");
		}
		if (error.matches(PyExc_ImportError)) {
			throw InvalidInputException("video_metadata() failed: %s", error.what());
		}
		throw InternalException("video_metadata() dependency preflight failed unexpectedly: %s", error.what());
	}
}

[[noreturn]] static void RethrowVideoReadError(const std::exception_ptr &read_error) {
	try {
		std::rethrow_exception(read_error);
	} catch (py::error_already_set &error) {
		if (!error.matches(PyExc_Exception)) {
			throw InterruptException();
		}
		if (error.matches(PyExc_MemoryError)) {
			throw OutOfMemoryException("Video metadata source read ran out of memory");
		}
		throw;
	}
}

static VideoMetadataResult ProbeVideoMetadata(ClientContext &context, ResolvedFile &resolved, const FileReference &file,
                                              uint64_t max_metadata_bytes) {
	PythonGILWrapper gil;
	py::object video_file_error_type;
	std::exception_ptr read_error;
	try {
		auto module = py::module_::import("vane._video_file");
		video_file_error_type = module.attr("VideoFileError");
		auto helper = module.attr("_probe_video_metadata");
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
		py::object content_type = file.has_content_type ? py::cast(file.content_type) : py::none();
		auto value = helper(std::move(read_at), py::int_(resolved.LogicalSize()), std::move(content_type),
		                    py::int_(max_metadata_bytes));
		if (read_error) {
			RethrowVideoReadError(read_error);
		}
		if (!py::isinstance<py::tuple>(value)) {
			throw InternalException("Video metadata helper returned a non-tuple value");
		}
		auto fields = py::reinterpret_borrow<py::tuple>(value);
		if (fields.size() != 7 || !py::isinstance<py::int_>(fields[0]) || !py::isinstance<py::int_>(fields[1]) ||
		    (!fields[2].is_none() && !py::isinstance<py::float_>(fields[2]) && !py::isinstance<py::int_>(fields[2])) ||
		    (!fields[3].is_none() && !py::isinstance<py::float_>(fields[3]) && !py::isinstance<py::int_>(fields[3])) ||
		    (!fields[4].is_none() && !py::isinstance<py::int_>(fields[4])) || !py::isinstance<py::int_>(fields[5]) ||
		    !py::isinstance<py::int_>(fields[6])) {
			throw InternalException("Video metadata helper returned an invalid value");
		}

		auto width = py::cast<int64_t>(fields[0]);
		auto height = py::cast<int64_t>(fields[1]);
		auto has_fps = !fields[2].is_none();
		auto fps = has_fps ? py::cast<double>(fields[2]) : 0;
		auto has_duration = !fields[3].is_none();
		auto duration = has_duration ? py::cast<double>(fields[3]) : 0;
		auto has_frame_count = !fields[4].is_none();
		auto frame_count = has_frame_count ? py::cast<int64_t>(fields[4]) : 0;
		auto time_base_numerator = py::cast<int64_t>(fields[5]);
		auto time_base_denominator = py::cast<int64_t>(fields[6]);
		if (width <= 0 || width > NumericLimits<uint32_t>::Maximum() || height <= 0 ||
		    height > NumericLimits<uint32_t>::Maximum() || (has_fps && (!std::isfinite(fps) || fps <= 0)) ||
		    (has_duration && (!std::isfinite(duration) || duration < 0)) || (has_frame_count && frame_count <= 0) ||
		    time_base_numerator <= 0 || time_base_denominator <= 0) {
			throw InternalException("Video metadata helper returned out-of-range numeric values");
		}
		return {NumericCast<uint32_t>(width),
		        NumericCast<uint32_t>(height),
		        has_fps,
		        fps,
		        has_duration,
		        duration,
		        has_frame_count,
		        frame_count,
		        time_base_numerator,
		        time_base_denominator};
	} catch (py::error_already_set &error) {
		if (!error.matches(PyExc_Exception)) {
			throw InterruptException();
		}
		if (read_error) {
			RethrowVideoReadError(read_error);
		}
		if (error.matches(PyExc_MemoryError)) {
			throw OutOfMemoryException("Video metadata inspection ran out of memory");
		}
		if (error.matches(PyExc_ImportError) ||
		    (video_file_error_type.ptr() && error.matches(video_file_error_type.ptr()))) {
			throw InvalidInputException("video_metadata() failed: %s", error.what());
		}
		throw InternalException("video_metadata() Python helper failed unexpectedly: %s", error.what());
	}
}

static void VideoMetadataFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	bool dependency_ready = false;
	for (idx_t row = 0; row < args.size(); row++) {
		auto file_value = args.data[0].GetValue(row);
		if (file_value.IsNull()) {
			result.SetValue(row, Value(VideoMetadataType()));
			continue;
		}

		uint64_t max_metadata_bytes = DEFAULT_VIDEO_METADATA_BYTES;
		if (args.ColumnCount() == 2) {
			auto max_metadata_value = args.data[1].GetValue(row);
			if (max_metadata_value.IsNull()) {
				result.SetValue(row, Value(VideoMetadataType()));
				continue;
			}
			max_metadata_bytes = max_metadata_value.GetValue<uint64_t>();
		}
		if (max_metadata_bytes == 0 || max_metadata_bytes > MAX_VIDEO_METADATA_BYTES) {
			throw InvalidInputException("video_metadata() max_bytes must be between 1 and %llu",
			                            static_cast<unsigned long long>(MAX_VIDEO_METADATA_BYTES));
		}

		auto &context = state.GetContext();
		try {
			if (!dependency_ready) {
				RequireVideoDependency();
				dependency_ready = true;
			}
			auto file = FileReference::FromValue(file_value, "video_metadata");
			auto resolved = ResolvedFile::Open(context, file);
			auto metadata = ProbeVideoMetadata(context, *resolved, file, max_metadata_bytes);
			if (context.IsInterrupted()) {
				throw InterruptException();
			}
			result.SetValue(row, metadata.ToValue());
		} catch (...) {
			if (context.IsInterrupted()) {
				throw InterruptException();
			}
			throw;
		}
	}
}

static ScalarFunction MakeVideoMetadataFunction(vector<LogicalType> arguments) {
	ScalarFunction function("video_metadata", std::move(arguments), VideoMetadataType(), VideoMetadataFunction,
	                        BindVideoMetadata);
	function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	function.SetStability(FunctionStability::VOLATILE);
	function.SetFallible();
	function.SetBindExpressionCallback(
	    [](FunctionBindExpressionInput &input) { return MediaBackend::BindNative(input, "video", "video_metadata"); });
	return function;
}

} // namespace

ScalarFunctionSet VideoFileFunctions::GetFunctions() {
	ScalarFunctionSet result("video_metadata");
	result.AddFunction(MakeVideoMetadataFunction({LogicalType::ANY}));
	result.AddFunction(MakeVideoMetadataFunction({LogicalType::ANY, LogicalType::UBIGINT}));
	return result;
}

} // namespace duckdb
