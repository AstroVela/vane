// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/audio_file_functions.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression.hpp"
#include "file_resolver.hpp"
#include "file_value.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"

#include <cmath>
#include <exception>

namespace duckdb {

namespace {

static constexpr uint64_t DEFAULT_AUDIO_METADATA_BYTES = 8 * 1024 * 1024;
static constexpr uint64_t MAX_AUDIO_METADATA_BYTES = 64 * 1024 * 1024;

static LogicalType AudioMetadataType() {
	child_list_t<LogicalType> fields;
	fields.emplace_back("sample_rate", LogicalType::BIGINT);
	fields.emplace_back("channels", LogicalType::BIGINT);
	fields.emplace_back("frames", LogicalType::BIGINT);
	fields.emplace_back("duration", LogicalType::DOUBLE);
	fields.emplace_back("format", LogicalType::VARCHAR);
	fields.emplace_back("subtype", LogicalType::VARCHAR);
	return LogicalType::STRUCT(std::move(fields));
}

struct AudioMetadataResult {
	int64_t sample_rate;
	int64_t channels;
	bool has_frames;
	int64_t frames;
	bool has_duration;
	double duration;
	string format;
	bool has_subtype;
	string subtype;

	Value ToValue() const {
		vector<Value> fields;
		fields.reserve(6);
		fields.push_back(Value::BIGINT(sample_rate));
		fields.push_back(Value::BIGINT(channels));
		fields.push_back(has_frames ? Value::BIGINT(frames) : Value(LogicalType::BIGINT));
		fields.push_back(has_duration ? Value::DOUBLE(duration) : Value(LogicalType::DOUBLE));
		fields.push_back(Value(format));
		fields.push_back(has_subtype ? Value(subtype) : Value(LogicalType::VARCHAR));
		return Value::STRUCT(AudioMetadataType(), std::move(fields));
	}
};

static unique_ptr<FunctionData> BindAudioMetadata(ClientContext &, ScalarFunction &bound_function,
                                                  vector<unique_ptr<Expression>> &arguments) {
	auto audio_type = FileLogicalType::Create(FileMediaType::AUDIO);
	auto input_type = arguments[0]->return_type;
	if (input_type.id() == LogicalTypeId::UNKNOWN || input_type.id() == LogicalTypeId::SQLNULL) {
		input_type = audio_type;
	}
	if (!FileLogicalType::IsFile(input_type) || FileLogicalType::GetMediaType(input_type) != FileMediaType::AUDIO) {
		throw BinderException("audio_metadata() requires AUDIOFILE, not %s", input_type.ToString());
	}
	bound_function.arguments[0] = std::move(input_type);
	return nullptr;
}

static void RequireAudioDependency() {
	PythonGILWrapper gil;
	try {
		py::module_::import("vane._audio_file").attr("_load_soundfile")();
	} catch (py::error_already_set &error) {
		if (!error.matches(PyExc_Exception)) {
			throw InterruptException();
		}
		if (error.matches(PyExc_MemoryError)) {
			throw OutOfMemoryException("Audio dependency preflight ran out of memory");
		}
		if (error.matches(PyExc_ImportError)) {
			throw InvalidInputException("audio_metadata() failed: %s", error.what());
		}
		throw InternalException("audio_metadata() dependency preflight failed unexpectedly: %s", error.what());
	}
}

[[noreturn]] static void RethrowAudioReadError(const std::exception_ptr &read_error) {
	try {
		std::rethrow_exception(read_error);
	} catch (py::error_already_set &error) {
		// Registered Python filesystems execute beneath ResolvedFile. Preserve
		// Python control-flow and resource failures when their virtual-I/O
		// exceptions have been parked across the libsndfile callback boundary.
		if (!error.matches(PyExc_Exception)) {
			throw InterruptException();
		}
		if (error.matches(PyExc_MemoryError)) {
			throw OutOfMemoryException("Audio metadata source read ran out of memory");
		}
		throw;
	}
}

static AudioMetadataResult ProbeAudioMetadata(ResolvedFile &resolved, const FileReference &file,
                                              uint64_t max_metadata_bytes) {
	PythonGILWrapper gil;
	py::object audio_file_error_type;
	std::exception_ptr read_error;
	try {
		auto module = py::module_::import("vane._audio_file");
		audio_file_error_type = module.attr("AudioFileError");
		auto helper = module.attr("_probe_audio_metadata");
		auto read_at = py::cpp_function([&resolved, &read_error](uint64_t offset, uint64_t size) {
			string bytes;
			try {
				bytes.resize(NumericCast<idx_t>(size));
				if (size > 0) {
					py::gil_scoped_release release;
					resolved.ReadExact(reinterpret_cast<data_ptr_t>(bytes.data()), size, offset);
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
			RethrowAudioReadError(read_error);
		}
		if (!py::isinstance<py::tuple>(value)) {
			throw InternalException("Audio metadata helper returned a non-tuple value");
		}
		auto fields = py::reinterpret_borrow<py::tuple>(value);
		if (fields.size() != 6 || !py::isinstance<py::int_>(fields[0]) || !py::isinstance<py::int_>(fields[1]) ||
		    (!fields[2].is_none() && !py::isinstance<py::int_>(fields[2])) ||
		    (!fields[3].is_none() && !py::isinstance<py::float_>(fields[3]) && !py::isinstance<py::int_>(fields[3])) ||
		    !py::isinstance<py::str>(fields[4]) || (!fields[5].is_none() && !py::isinstance<py::str>(fields[5]))) {
			throw InternalException("Audio metadata helper returned an invalid value");
		}
		auto sample_rate = py::cast<int64_t>(fields[0]);
		auto channels = py::cast<int64_t>(fields[1]);
		auto has_frames = !fields[2].is_none();
		auto has_duration = !fields[3].is_none();
		if (has_frames != has_duration) {
			throw InternalException("Audio metadata helper returned inconsistent frame and duration values");
		}
		auto frames = has_frames ? py::cast<int64_t>(fields[2]) : 0;
		auto duration = has_duration ? py::cast<double>(fields[3]) : 0;
		if (sample_rate <= 0 || channels <= 0 || frames < 0 || !std::isfinite(duration) || duration < 0) {
			throw InternalException("Audio metadata helper returned out-of-range numeric values");
		}
		if (fields[5].is_none()) {
			return {sample_rate, channels, has_frames, frames, has_duration, duration, py::cast<string>(fields[4]),
			        false,       string()};
		}
		return {sample_rate,
		        channels,
		        has_frames,
		        frames,
		        has_duration,
		        duration,
		        py::cast<string>(fields[4]),
		        true,
		        py::cast<string>(fields[5])};
	} catch (py::error_already_set &error) {
		// A currently propagating interpreter-control exception takes priority
		// over any earlier ordinary connector failure captured by read_at.
		if (!error.matches(PyExc_Exception)) {
			throw InterruptException();
		}
		if (read_error) {
			RethrowAudioReadError(read_error);
		}
		if (error.matches(PyExc_MemoryError)) {
			throw OutOfMemoryException("Audio metadata inspection ran out of memory");
		}
		if (error.matches(PyExc_ImportError) ||
		    (audio_file_error_type.ptr() && error.matches(audio_file_error_type.ptr()))) {
			throw InvalidInputException("audio_metadata() failed: %s", error.what());
		}
		throw InternalException("audio_metadata() Python helper failed unexpectedly: %s", error.what());
	}
}

static void AudioMetadataFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	bool dependency_ready = false;
	for (idx_t row = 0; row < args.size(); row++) {
		auto file_value = args.data[0].GetValue(row);
		if (file_value.IsNull()) {
			result.SetValue(row, Value(AudioMetadataType()));
			continue;
		}

		uint64_t max_metadata_bytes = DEFAULT_AUDIO_METADATA_BYTES;
		if (args.ColumnCount() == 2) {
			auto max_metadata_value = args.data[1].GetValue(row);
			if (max_metadata_value.IsNull()) {
				result.SetValue(row, Value(AudioMetadataType()));
				continue;
			}
			max_metadata_bytes = max_metadata_value.GetValue<uint64_t>();
		}
		if (max_metadata_bytes == 0 || max_metadata_bytes > MAX_AUDIO_METADATA_BYTES) {
			throw InvalidInputException("audio_metadata() max_bytes must be between 1 and %llu",
			                            static_cast<unsigned long long>(MAX_AUDIO_METADATA_BYTES));
		}

		auto &context = state.GetContext();
		try {
			// Fail with the actionable optional-dependency error before resolving a
			// local or remote URL. The Python value API follows the same ordering.
			if (!dependency_ready) {
				RequireAudioDependency();
				dependency_ready = true;
			}
			auto file = FileReference::FromValue(file_value, "audio_metadata");
			auto resolved = ResolvedFile::Open(context, file);
			auto metadata = ProbeAudioMetadata(*resolved, file, max_metadata_bytes);
			if (context.IsInterrupted()) {
				throw InterruptException();
			}
			result.SetValue(row, metadata.ToValue());
		} catch (...) {
			// Cancellation wins over a competing connector, dependency, media, or
			// resource-limit failure raised while this row was being inspected.
			if (context.IsInterrupted()) {
				throw InterruptException();
			}
			throw;
		}
	}
}

static ScalarFunction MakeAudioMetadataFunction(vector<LogicalType> arguments) {
	ScalarFunction function("audio_metadata", std::move(arguments), AudioMetadataType(), AudioMetadataFunction,
	                        BindAudioMetadata);
	function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	function.SetStability(FunctionStability::VOLATILE);
	function.SetFallible();
	return function;
}

} // namespace

ScalarFunctionSet AudioFileFunctions::GetFunctions() {
	ScalarFunctionSet result("audio_metadata");
	result.AddFunction(MakeAudioMetadataFunction({LogicalType::ANY}));
	result.AddFunction(MakeAudioMetadataFunction({LogicalType::ANY, LogicalType::UBIGINT}));
	return result;
}

} // namespace duckdb
