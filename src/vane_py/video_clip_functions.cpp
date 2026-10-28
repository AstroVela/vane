// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/video_file_functions.hpp"
#include "vane_python/datasource_execution_context.hpp"
#include "vane_python/file.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"
#include "vane_python/python_conversion.hpp"
#include "duckdb/execution/expression_executor_state.hpp"
#include "duckdb/function/scalar_macro_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/planner/expression.hpp"
#include "file_value.hpp"
#include "media_backend.hpp"

#include <chrono>

namespace duckdb {
namespace {
static constexpr uint64_t CLIP_BATCH_BYTES = 256 * 1024 * 1024;

static vector<string> ClipNames() {
	return {"file",         "start_time",       "end_time",           "include_audio",
	        "max_duration", "max_input_bytes",  "max_decoded_frames", "max_decoded_samples",
	        "max_pixels",   "max_output_bytes", "timeout_seconds"};
}

static vector<LogicalType> ClipArguments() {
	return {LogicalType::ANY,    LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BOOLEAN,
	        LogicalType::DOUBLE, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT,
	        LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::DOUBLE};
}

static LogicalType ClipType() {
	return LogicalType::STRUCT({{"data", LogicalType::BLOB},
	                            {"content_type", LogicalType::VARCHAR},
	                            {"start_time", LogicalType::DOUBLE},
	                            {"end_time", LogicalType::DOUBLE},
	                            {"duration", LogicalType::DOUBLE},
	                            {"frame_count", LogicalType::UBIGINT},
	                            {"has_audio", LogicalType::BOOLEAN}});
}

static unique_ptr<FunctionData> BindClip(ClientContext &, ScalarFunction &function,
                                         vector<unique_ptr<Expression>> &arguments) {
	auto type = arguments[0]->return_type;
	if (type.id() == LogicalTypeId::UNKNOWN || type.id() == LogicalTypeId::SQLNULL) {
		type = FileLogicalType::Create(FileMediaType::VIDEO);
	}
	if (!FileLogicalType::IsFile(type) || FileLogicalType::GetMediaType(type) != FileMediaType::VIDEO) {
		throw BinderException("video_clip() requires VIDEOFILE, not %s", type.ToString());
	}
	function.arguments[0] = type;
	return nullptr;
}

struct ClipContext {
	shared_ptr<PythonDataSourceExecutionContext> token;
	explicit ClipContext(ClientContext &context)
	    : token(make_shared_ptr<PythonDataSourceExecutionContext>(context.shared_from_this())) {
	}
	~ClipContext() {
		token->Invalidate();
	}
};

[[noreturn]] static void ClipError(py::error_already_set &error, ClientContext &context, const py::object &media,
                                   const py::object &native) {
	auto matches = [&](const char *name) {
		return native && error.matches(native.attr(name).ptr());
	};
	if (context.IsInterrupted() || !error.matches(PyExc_Exception) || matches("InterruptException")) {
		throw InterruptException();
	}
	if (error.matches(PyExc_MemoryError) || matches("OutOfMemoryException")) {
		throw OutOfMemoryException("video_clip ran out of memory");
	}
	auto message = py::str(error.value()).cast<string>();
	if (matches("PermissionException")) {
		throw PermissionException("%s", message);
	}
	if (matches("NotImplementedException")) {
		throw NotImplementedException("%s", message);
	}
	if (error.matches(PyExc_OSError) || matches("IOException")) {
		throw IOException("%s", message);
	}
	if ((media && error.matches(media.attr("VideoFileLimitError").ptr())) || matches("OutOfRangeException")) {
		throw OutOfRangeException("%s", message);
	}
	if (error.matches(PyExc_ImportError) || error.matches(PyExc_ValueError) || error.matches(PyExc_TypeError) ||
	    (media && error.matches(media.attr("VideoFileError").ptr())) || matches("InvalidInputException")) {
		throw InvalidInputException("video_clip failed: %s", message);
	}
	throw InternalException("video_clip helper failed unexpectedly: %s", error.what());
}

static void VideoClipFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	uint64_t batch_bytes = 0;
	auto names = ClipNames();
	for (idx_t row = 0; row < args.size(); row++) {
		bool null = false;
		for (auto &column : args.data) {
			null |= column.GetValue(row).IsNull();
		}
		if (null) {
			result.SetValue(row, Value(result.GetType()));
			continue;
		}
		auto &context = state.GetContext();
		auto started = std::chrono::steady_clock::now();
		PythonGILWrapper gil;
		ClipContext scope(context);
		py::object media, native;
		try {
			scope.token->CheckInterrupted();
			media = py::module_::import("vane._video_file");
			native = py::module_::import("vane._native");
			py::dict options;
			for (idx_t index = 1; index < names.size(); index++) {
				auto value = args.data[index].GetValue(row);
				if (index == 3) {
					options[py::str(names[index])] = py::bool_(value.GetValue<bool>());
				} else if (index >= 5 && index <= 9) {
					options[py::str(names[index])] = py::int_(value.GetValue<int64_t>());
				} else {
					options[py::str(names[index])] = py::float_(value.GetValue<double>());
				}
			}
			// Validate caller options before applying the remaining chunk budget.
			auto helper = py::module_::import("vane._video_clip");
			helper.attr("_normalize")(options);
			auto remaining = CLIP_BATCH_BYTES - batch_bytes;
			if (!remaining) {
				throw OutOfRangeException("video_clip exceeds its 256 MiB output budget per chunk");
			}
			auto limit = args.data[9].GetValue(row).GetValue<int64_t>();
			options["max_output_bytes"] = py::int_(MinValue<uint64_t>(uint64_t(limit), remaining));
			auto value = helper.attr("_scalar_video_clip")(PythonFile::FromValue(args.data[0].GetValue(row)), options,
			                                               scope.token);
			auto fields = py::cast<py::tuple>(value);
			if (fields.size() != 7 || !py::isinstance<py::bytes>(fields[0])) {
				throw InternalException("video_clip helper returned invalid fields");
			}
			auto size = uint64_t(PyBytes_Size(fields[0].ptr()));
			if (size > remaining) {
				throw InternalException("video_clip helper exceeded its output reservation");
			}
			batch_bytes += size;
			result.SetValue(row, TransformPythonValue(fields, result.GetType()));
			scope.token->CheckInterrupted();
			if (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >=
			    args.data[10].GetValue(row).GetValue<double>()) {
				throw OutOfRangeException("video_clip exceeded timeout_seconds");
			}
		} catch (py::error_already_set &error) {
			ClipError(error, context, media, native);
		}
	}
}
} // namespace

ScalarFunctionSet VideoFileFunctions::GetClipFunctions() {
	ScalarFunction function("_vane_video_clip", ClipArguments(), ClipType(), VideoClipFunction, BindClip);
	function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	function.SetStability(FunctionStability::VOLATILE);
	function.SetFallible();
	function.SetBindExpressionCallback(
	    [](FunctionBindExpressionInput &input) { return MediaBackend::BindNative(input, "video", "video_clip"); });
	ScalarFunctionSet result("_vane_video_clip");
	result.AddFunction(std::move(function));
	return result;
}

unique_ptr<CreateMacroInfo> VideoFileFunctions::GetClipMacro() {
	auto names = ClipNames();
	auto types = ClipArguments();
	vector<Value> defaults {Value(),
	                        Value(),
	                        Value(),
	                        Value::BOOLEAN(true),
	                        Value::DOUBLE(300),
	                        Value::BIGINT(1024 * 1024 * 1024),
	                        Value::BIGINT(100000),
	                        Value::BIGINT(100000000),
	                        Value::BIGINT(32 * 1024 * 1024),
	                        Value::BIGINT(64 * 1024 * 1024),
	                        Value::DOUBLE(60)};
	vector<unique_ptr<ParsedExpression>> arguments;
	for (auto &name : names) {
		arguments.push_back(make_uniq<ColumnRefExpression>(name));
	}
	auto macro =
	    make_uniq<ScalarMacroFunction>(make_uniq<FunctionExpression>("_vane_video_clip", std::move(arguments)));
	for (idx_t i = 0; i < names.size(); i++) {
		macro->parameters.push_back(make_uniq<ColumnRefExpression>(names[i]));
		macro->types.push_back(i ? types[i] : LogicalType::UNKNOWN);
		if (i >= 3) {
			macro->default_parameters.insert(make_pair(names[i], make_uniq<ConstantExpression>(defaults[i])));
		}
	}
	auto info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	info->schema = DEFAULT_SCHEMA;
	info->name = "video_clip";
	info->temporary = true;
	info->internal = true;
	info->macros.push_back(std::move(macro));
	return info;
}
} // namespace duckdb
