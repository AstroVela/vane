// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "duckdb/common/types/image.hpp"
#include "vane_python/video_file_functions.hpp"
#include "vane_python/datasource_execution_context.hpp"
#include "vane_python/file.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"
#include "vane_python/python_conversion.hpp"
#include "video_frame_contract.hpp"
#include "media_backend.hpp"
#include "duckdb/execution/expression_executor_state.hpp"
#include "duckdb/main/client_context.hpp"

#include <cstring>
#include <exception>

namespace duckdb {
namespace {

struct PythonFrameScope {
	shared_ptr<PythonDataSourceExecutionContext> context;
	py::object generator;
	bool closed = false;

	explicit PythonFrameScope(ClientContext &client)
	    : context(make_shared_ptr<PythonDataSourceExecutionContext>(client.shared_from_this())) {
	}
	~PythonFrameScope() {
		if (!closed && generator) {
			try {
				generator.attr("close")();
			} catch (py::error_already_set &error) {
				error.discard_as_unraisable("closing video frame iterator after failure");
			} catch (...) {
			}
		}
		context->Invalidate();
	}
	void Close() {
		if (generator) {
			generator.attr("close")();
		}
		closed = true;
		context->Invalidate();
	}
};

static py::dict FrameOptionsDictionary(const VideoFrameOptions &options, VideoFrameOperation) {
	py::dict result;
	result["start_time"] = options.start;
	result["end_time"] = options.has_end ? py::cast(options.end) : py::none();
	result["width"] = options.width ? py::cast(options.width) : py::none();
	result["height"] = options.height ? py::cast(options.height) : py::none();
	result["is_key_frame"] = options.has_key ? py::cast(options.key) : py::none();
	result["sample_interval_seconds"] = options.has_interval ? py::cast(options.interval) : py::none();
	result["max_input_bytes"] = options.max_input_bytes;
	result["max_frames"] = options.max_decoded_frames;
	result["max_pixels"] = options.max_pixels;
	result["target_frame_index"] = options.has_target_index ? py::cast(options.target_index) : py::none();
	return result;
}

static py::object PythonIndexValue(const Value &index) {
	if (index.IsNull()) {
		return py::none();
	}
	return py::bytes(StringValue::Get(index));
}

static void RethrowPythonVideoError(py::error_already_set &error, ClientContext &context, const py::object &module,
                                    const py::object &native, bool null_on_error) {
	if (context.IsInterrupted() || !error.matches(PyExc_Exception) ||
	    (native && error.matches(native.attr("InterruptException").ptr()))) {
		throw InterruptException();
	}
	if (error.matches(PyExc_MemoryError) || (native && error.matches(native.attr("OutOfMemoryException").ptr()))) {
		throw OutOfMemoryException("video frame decoding ran out of memory");
	}
	auto message = py::str(error.value()).cast<string>();
	if (native && error.matches(native.attr("PermissionException").ptr())) {
		throw PermissionException("%s", message);
	}
	if (native && error.matches(native.attr("NotImplementedException").ptr())) {
		throw NotImplementedException("%s", message);
	}
	if (module && error.matches(module.attr("VideoFileFormatError").ptr())) {
		if (null_on_error) {
			return;
		}
		throw InvalidInputException("%s", message);
	}
	if (error.matches(PyExc_OSError) || (native && error.matches(native.attr("IOException").ptr()))) {
		throw IOException("%s", message);
	}
	if ((module && error.matches(module.attr("VideoFileLimitError").ptr())) ||
	    (native && error.matches(native.attr("OutOfRangeException").ptr()))) {
		throw OutOfRangeException("%s", message);
	}
	if (error.matches(PyExc_ImportError) || (module && error.matches(module.attr("VideoFileError").ptr())) ||
	    (native && error.matches(native.attr("InvalidInputException").ptr()))) {
		throw InvalidInputException("%s", message);
	}
	throw InternalException("video Python helper failed unexpectedly: %s", error.what());
}

static void CopyPythonFrame(ClientContext &context, const py::tuple &frame, Vector &image, idx_t row, uint32_t width,
                            uint32_t height) {
	if (!py::isinstance<py::bytes>(frame[10])) {
		throw InternalException("video helper returned a non-bytes IMAGE payload");
	}
	char *source;
	py::ssize_t size;
	if (PyBytes_AsStringAndSize(frame[10].ptr(), &source, &size) != 0) {
		throw py::error_already_set();
	}
	if (size < 0 || uint64_t(size) != uint64_t(width) * height * 3) {
		throw InternalException("video helper returned an invalid IMAGE payload size");
	}
	ImageLogicalType::ValidateFields(idx_t(size), width, height, 3, "RGB", "video_frames");
	auto data = ImageVector::Allocate(image, row, width, height, "RGB");
	for (idx_t copied = 0; copied < idx_t(size);) {
		if (context.IsInterrupted()) {
			throw InterruptException();
		}
		auto count = MinValue<idx_t>(VideoFrameContract::MIB, idx_t(size) - copied);
		memcpy(data + copied, source + copied, count);
		copied += count;
	}
}

static void PythonFrameRow(ClientContext &context, const Value &file, const shared_ptr<VideoFrameOptions> &options,
                           const shared_ptr<uint64_t> &batch_bytes, VideoFrameOperation operation, Vector &result,
                           idx_t row, const Value &index) {
	PythonGILWrapper gil;
	PythonFrameScope scope(context);
	py::object module;
	py::object native;
	auto callback_error = make_shared_ptr<std::exception_ptr>();
	auto budget = make_shared_ptr<VideoFrameOutputBudget>(*options, *batch_bytes, file, operation);
	auto dimensions = make_shared_ptr<pair<uint32_t, uint32_t>>(0, 0);
	idx_t written = 0;
	if (operation != VideoFrameOperation::FRAME_BY_INDEX && operation != VideoFrameOperation::SCAN_STATS) {
		FlatVector::GetData<list_entry_t>(result)[row] = list_entry_t(ListVector::GetListSize(result), 0);
		FlatVector::Validity(result).SetValid(row);
	}
	try {
		module = py::module_::import("vane._video_file");
		native = py::module_::import("vane._native");
		if (operation == VideoFrameOperation::SCAN_STATS) {
			auto statistics =
			    py::module_::import("vane._video_index")
			        .attr("_scan_stats")(PythonFile::FromValue(file), FrameOptionsDictionary(*options, operation),
			                             PythonIndexValue(index), scope.context);
			result.SetValue(row, TransformPythonValue(statistics, result.GetType()));
			scope.context->CheckInterrupted();
			scope.Close();
			return;
		}
		auto reserve = py::cpp_function([token = scope.context, options, batch_bytes, budget, dimensions,
		                                 callback_error](uint32_t width, uint32_t height) {
			try {
				token->CheckInterrupted();
				budget->ClaimFrame(width, height);
				*dimensions = {width, height};
			} catch (...) {
				*callback_error = std::current_exception();
				throw;
			}
		});
		scope.generator =
		    py::module_::import("vane._video_expressions")
		        .attr("_scalar_video_frames")(PythonFile::FromValue(file), FrameOptionsDictionary(*options, operation),
		                                      scope.context, std::move(reserve), PythonIndexValue(index));
		for (auto item : scope.generator) {
			scope.context->CheckInterrupted();
			if (*callback_error) {
				std::rethrow_exception(*callback_error);
			}
			if (!py::isinstance<py::tuple>(item) || budget->frame_count != written + 1) {
				throw InternalException("video helper violated its frame reservation contract");
			}
			auto frame = py::reinterpret_borrow<py::tuple>(item);
			if (frame.size() != 11 || !py::isinstance<py::int_>(frame[8]) || !py::isinstance<py::int_>(frame[9]) ||
			    py::cast<uint32_t>(frame[8]) != dimensions->first ||
			    py::cast<uint32_t>(frame[9]) != dimensions->second) {
				throw InternalException("video helper returned invalid frame fields or dimensions");
			}
			idx_t target = row;
			Vector *image = &result;
			if (operation != VideoFrameOperation::FRAME_BY_INDEX) {
				target = ListVector::GetListSize(result);
				ListVector::Reserve(result, target + 1);
				ListVector::SetListSize(result, target + 1);
				image = &ListVector::GetEntry(result);
				if (operation == VideoFrameOperation::FRAMES) {
					auto &children = StructVector::GetEntries(*image);
					children[0]->SetValue(target, file);
					for (idx_t index = 0; index < 8; index++) {
						children[index + 1]->SetValue(
						    target, TransformPythonValue(frame[index], children[index + 1]->GetType()));
					}
					FlatVector::Validity(*image).SetValid(target);
					image = children.back().get();
				}
			}
			CopyPythonFrame(context, frame, *image, target, dimensions->first, dimensions->second);
			written++;
			if (operation != VideoFrameOperation::FRAME_BY_INDEX) {
				FlatVector::GetData<list_entry_t>(result)[row].length++;
			} else if (written != 1 || py::cast<uint64_t>(frame[0]) != options->target_index) {
				throw InternalException("video helper violated exact frame-index selection");
			}
		}
		scope.context->CheckInterrupted();
		if (*callback_error) {
			std::rethrow_exception(*callback_error);
		}
		scope.Close();
		if (operation == VideoFrameOperation::FRAME_BY_INDEX && written == 0) {
			if (!options->null_on_error) {
				throw InvalidInputException("video frame idx %llu is out of range",
				                            static_cast<unsigned long long>(options->target_index));
			}
			result.SetValue(row, Value(result.GetType()));
		}
	} catch (py::error_already_set &error) {
		if (*callback_error) {
			std::rethrow_exception(*callback_error);
		}
		RethrowPythonVideoError(error, context, module, native, options->null_on_error);
		result.SetValue(row, Value(result.GetType()));
	}
}

template <VideoFrameOperation OPERATION>
static void PythonVideoFrames(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto batch_bytes = make_shared_ptr<uint64_t>(0);
	for (idx_t row = 0; row < args.size(); row++) {
		auto &context = state.GetContext();
		if (context.IsInterrupted()) {
			throw InterruptException();
		}
		auto options = make_shared_ptr<VideoFrameOptions>();
		if (!VideoFrameOptions::Read(args, row, OPERATION, *options)) {
			result.SetValue(row, Value(result.GetType()));
			continue;
		}
		PythonFrameRow(context, args.data[0].GetValue(row), options, batch_bytes, OPERATION, result, row,
		               VideoFrameContract::ReadIndex(args.data[14], row));
	}
}

template <VideoFrameOperation OPERATION>
static ScalarFunctionSet FrameFunctions() {
	ScalarFunction function(string("_vane_") + VideoFrameContract::Name(OPERATION), VideoFrameContract::Arguments(),
	                        VideoFrameContract::ResultType(OPERATION), PythonVideoFrames<OPERATION>,
	                        VideoFrameContract::Bind);
	function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	function.SetStability(FunctionStability::VOLATILE);
	function.SetFallible();
	function.SetBindExpressionCallback([](FunctionBindExpressionInput &input) {
		return MediaBackend::BindNative(input, "video", VideoFrameContract::Name(OPERATION));
	});
	ScalarFunctionSet result(function.name);
	result.AddFunction(std::move(function));
	return result;
}
template <bool BUILD>
static void PythonIndexFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &context = state.GetContext();
	uint64_t batch_bytes = 0;
	for (idx_t row = 0; row < args.size(); row++) {
		if (context.IsInterrupted()) {
			throw InterruptException();
		}
		bool null = false;
		for (idx_t column = 0; column < args.ColumnCount(); column++) {
			if (BUILD) {
				null |= args.data[column].GetValue(row).IsNull();
			} else {
				null |= !VideoFrameContract::HasIndex(args.data[column], row);
			}
		}
		if (null) {
			result.SetValue(row, Value(result.GetType()));
			continue;
		}
		VideoFrameOptions options;
		uint64_t limit = 0;
		if (BUILD) {
			options.max_input_bytes = VideoFrameContract::IndexLimit(args.data[1].GetValue(row), "max_input_bytes",
			                                                         16 * 1024 * VideoFrameContract::MIB);
			options.max_decoded_frames =
			    VideoFrameContract::IndexLimit(args.data[2].GetValue(row), "max_decoded_frames", 100000000);
			options.max_pixels = VideoFrameContract::IndexLimit(args.data[3].GetValue(row), "max_pixels",
			                                                    VideoFrameContract::MAX_PIXELS);
			limit = VideoFrameContract::IndexLimit(args.data[4].GetValue(row), "max_index_bytes",
			                                       VideoFrameContract::MAX_INDEX_BYTES);
			limit = MinValue<uint64_t>(limit, VideoFrameContract::MAX_BATCH_BYTES - batch_bytes);
		}
		// Cap BLOBs before allocating a Python copy.
		auto index = BUILD ? Value(LogicalType::BLOB) : VideoFrameContract::ReadIndex(args.data[0], row);
		PythonGILWrapper gil;
		PythonFrameScope scope(context);
		py::object module, native;
		try {
			module = py::module_::import("vane._video_file");
			native = py::module_::import("vane._native");
			auto helper = py::module_::import("vane._video_index");
			py::object output;
			if (BUILD) {
				output = helper.attr("_build_video_index")(PythonFile::FromValue(args.data[0].GetValue(row)),
				                                           FrameOptionsDictionary(options, VideoFrameOperation::FRAMES),
				                                           limit, scope.context);
				if (!py::isinstance<py::bytes>(output) || uint64_t(PyBytes_GET_SIZE(output.ptr())) > limit) {
					throw InternalException("video index helper violated its BLOB budget");
				}
				batch_bytes += uint64_t(PyBytes_GET_SIZE(output.ptr()));
			} else {
				output = helper.attr("_video_index_info")(PythonIndexValue(index), scope.context);
			}
			result.SetValue(row, TransformPythonValue(output, result.GetType()));
			scope.context->CheckInterrupted();
			scope.Close();
		} catch (py::error_already_set &error) {
			RethrowPythonVideoError(error, context, module, native, false);
		}
	}
}

} // namespace

void PythonDataSourceExecutionContext::CaptureVideoError(const py::object &error) {
	shared_ptr<ClientContext> active_context;
	{
		auto guard = LockContext(active_context);
	}
	if (!PyExceptionInstance_Check(error.ptr())) {
		throw InvalidInputException("video error must be an exception instance");
	}
	PyErr_SetObject(reinterpret_cast<PyObject *>(Py_TYPE(error.ptr())), error.ptr());
	py::error_already_set captured;
	std::exception_ptr classified;
	try {
		RethrowPythonVideoError(captured, *active_context, py::module_::import("vane._video_file"),
		                        py::module_::import("vane._native"), false);
	} catch (...) {
		classified = std::current_exception();
	}
	auto guard = LockContext(active_context);
	if (!stream_error) {
		stream_error = std::move(classified);
	}
}

void PythonDataSourceExecutionContext::RethrowStreamError() const {
	shared_ptr<ClientContext> active_context;
	auto guard = LockContext(active_context);
	if (stream_error) {
		if (active_context->IsInterrupted()) {
			throw InterruptException();
		}
		std::rethrow_exception(stream_error);
	}
}

vector<ScalarFunctionSet> VideoFileFunctions::GetFrameFunctions() {
	vector<ScalarFunctionSet> result;
	result.push_back(FrameFunctions<VideoFrameOperation::FRAMES>());
	result.push_back(FrameFunctions<VideoFrameOperation::KEYFRAMES>());
	result.push_back(FrameFunctions<VideoFrameOperation::FRAME_BY_INDEX>());
	result.push_back(FrameFunctions<VideoFrameOperation::SCAN_STATS>());
	ScalarFunction build(
	    "_vane_build_video_index",
	    {LogicalType::ANY, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
	    LogicalType::BLOB, PythonIndexFunction<true>, VideoFrameContract::Bind);
	build.SetStability(FunctionStability::VOLATILE);
	build.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	build.SetFallible();
	build.SetBindExpressionCallback([](FunctionBindExpressionInput &input) {
		return MediaBackend::BindNative(input, "video", "build_video_index");
	});
	result.emplace_back(build);
	ScalarFunction info("video_index_info", {LogicalType::BLOB}, VideoFrameContract::IndexInfoType(),
	                    PythonIndexFunction<false>);
	info.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	info.SetFallible();
	info.SetBindExpressionCallback([](FunctionBindExpressionInput &input) {
		return MediaBackend::BindNative(input, "video", "video_index_info");
	});
	result.emplace_back(info);
	return result;
}
} // namespace duckdb
