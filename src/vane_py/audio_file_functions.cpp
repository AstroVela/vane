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
static constexpr uint64_t DEFAULT_AUDIO_MAX_INPUT_BYTES = 512 * 1024 * 1024;
static constexpr uint64_t DEFAULT_AUDIO_MAX_FRAMES = 100000000;
static constexpr uint64_t DEFAULT_AUDIO_MAX_DECODED_BYTES = 512 * 1024 * 1024;
static constexpr uint64_t DEFAULT_AUDIO_MAX_OUTPUT_FRAMES = 100000000;
static constexpr uint64_t DEFAULT_AUDIO_MAX_OUTPUT_BYTES = 512 * 1024 * 1024;
static constexpr uint64_t MAX_AUDIO_BATCH_OUTPUT_BYTES = 512 * 1024 * 1024;
static constexpr idx_t AUDIO_RESULT_COPY_CHUNK_BYTES = 1024 * 1024;

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

static LogicalType AudioResampleType() {
	child_list_t<LogicalType> fields;
	fields.emplace_back("samples", LogicalType::LIST(LogicalType::DOUBLE));
	fields.emplace_back("sample_rate", LogicalType::BIGINT);
	fields.emplace_back("frames", LogicalType::BIGINT);
	fields.emplace_back("channels", LogicalType::BIGINT);
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

static unique_ptr<FunctionData> BindAudioResample(ClientContext &, ScalarFunction &bound_function,
                                                  vector<unique_ptr<Expression>> &arguments) {
	auto audio_type = FileLogicalType::Create(FileMediaType::AUDIO);
	auto input_type = arguments[0]->return_type;
	if (input_type.id() == LogicalTypeId::UNKNOWN || input_type.id() == LogicalTypeId::SQLNULL) {
		input_type = audio_type;
	}
	if (!FileLogicalType::IsFile(input_type) || FileLogicalType::GetMediaType(input_type) != FileMediaType::AUDIO) {
		throw BinderException("audio_resample() requires AUDIOFILE, not %s", input_type.ToString());
	}
	bound_function.arguments[0] = std::move(input_type);
	return nullptr;
}

static void RequireAudioDependencies(const char *function_name, bool require_resampler) {
	PythonGILWrapper gil;
	try {
		auto module = py::module_::import("vane._audio_file");
		module.attr("_load_soundfile")();
		if (require_resampler) {
			module.attr("_load_soxr")();
		}
	} catch (py::error_already_set &error) {
		if (!error.matches(PyExc_Exception)) {
			throw InterruptException();
		}
		if (error.matches(PyExc_MemoryError)) {
			throw OutOfMemoryException("Audio dependency preflight ran out of memory");
		}
		if (error.matches(PyExc_ImportError)) {
			throw InvalidInputException("%s() failed: %s", function_name, error.what());
		}
		throw InternalException("%s() dependency preflight failed unexpectedly: %s", function_name, error.what());
	}
}

[[noreturn]] static void RethrowAudioReadError(const std::exception_ptr &read_error, const char *operation) {
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
			throw OutOfMemoryException("Audio %s source read ran out of memory", operation);
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
			RethrowAudioReadError(read_error, "metadata");
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
			RethrowAudioReadError(read_error, "metadata");
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

struct AudioResampleLimits {
	uint64_t max_input_bytes = DEFAULT_AUDIO_MAX_INPUT_BYTES;
	uint64_t max_frames = DEFAULT_AUDIO_MAX_FRAMES;
	uint64_t max_decoded_bytes = DEFAULT_AUDIO_MAX_DECODED_BYTES;
	uint64_t max_output_frames = DEFAULT_AUDIO_MAX_OUTPUT_FRAMES;
	uint64_t max_output_bytes = DEFAULT_AUDIO_MAX_OUTPUT_BYTES;
};

static bool GetAudioResampleArguments(DataChunk &args, idx_t row, int64_t &sample_rate, AudioResampleLimits &limits) {
	for (idx_t index = 1; index < args.ColumnCount(); index++) {
		if (args.data[index].GetValue(row).IsNull()) {
			return false;
		}
	}
	auto sample_rate_value = args.data[1].GetValue(row);
	sample_rate = sample_rate_value.GetValue<int64_t>();
	if (sample_rate <= 0) {
		throw InvalidInputException("audio_resample() sample_rate must be greater than zero");
	}
	if (args.ColumnCount() == 2) {
		return true;
	}

	D_ASSERT(args.ColumnCount() == 7);
	uint64_t *limit_values[] = {&limits.max_input_bytes, &limits.max_frames, &limits.max_decoded_bytes,
	                            &limits.max_output_frames, &limits.max_output_bytes};
	const char *limit_names[] = {"max_input_bytes", "max_frames", "max_decoded_bytes", "max_output_frames",
	                             "max_output_bytes"};
	for (idx_t index = 0; index < 5; index++) {
		auto value = args.data[index + 2].GetValue(row);
		auto limit = value.GetValue<uint64_t>();
		if (limit == 0) {
			throw InvalidInputException("audio_resample() %s must be greater than zero", limit_names[index]);
		}
		*limit_values[index] = limit;
	}
	if (limits.max_frames > NumericLimits<int64_t>::Maximum() ||
	    limits.max_output_frames > NumericLimits<int64_t>::Maximum()) {
		throw InvalidInputException("audio_resample() frame limits must fit in signed 64-bit");
	}
	return true;
}

static void CopyResampledAudio(ClientContext &context, ResolvedFile &resolved, const FileReference &file,
                               int64_t sample_rate, const AudioResampleLimits &limits, idx_t row,
                               Vector &samples_result, Vector &sample_rate_result, Vector &frames_result,
                               Vector &channels_result) {
	auto existing_samples = ListVector::GetListSize(samples_result);
	if (existing_samples > MAX_AUDIO_BATCH_OUTPUT_BYTES / sizeof(double)) {
		throw InternalException("audio_resample() exceeded its batch output invariant");
	}
	auto remaining_batch_bytes = MAX_AUDIO_BATCH_OUTPUT_BYTES - existing_samples * sizeof(double);

	PythonGILWrapper gil;
	py::object audio_file_error_type;
	std::exception_ptr read_error;
	try {
		auto module = py::module_::import("vane._audio_file");
		audio_file_error_type = module.attr("AudioFileError");
		auto resample_spool_type = module.attr("_AudioResampleSpool");
		auto helper = module.attr("_resample_audio_stream");
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
		auto value =
		    helper(std::move(read_at), py::int_(resolved.LogicalSize()), std::move(content_type), py::int_(sample_rate),
		           py::int_(limits.max_input_bytes), py::int_(limits.max_frames), py::int_(limits.max_decoded_bytes),
		           py::int_(limits.max_output_frames), py::int_(limits.max_output_bytes),
		           py::int_(remaining_batch_bytes), std::move(check_interrupted));
		if (read_error) {
			RethrowAudioReadError(read_error, "resample");
		}
		if (!py::isinstance(value, resample_spool_type)) {
			throw InternalException("Audio resample helper returned an invalid spool");
		}
		auto frame_count_value = value.attr("frames");
		auto channel_count_value = value.attr("channels");
		if (!py::isinstance<py::int_>(frame_count_value) || !py::isinstance<py::int_>(channel_count_value)) {
			throw InternalException("Audio resample spool returned invalid dimensions");
		}
		auto frame_count = py::cast<int64_t>(frame_count_value);
		auto channel_count = py::cast<int64_t>(channel_count_value);
		if (frame_count < 0 || channel_count <= 0 || static_cast<uint64_t>(frame_count) > limits.max_output_frames) {
			throw InternalException("Audio resample helper returned out-of-range dimensions");
		}
		if (static_cast<uint64_t>(frame_count) >
		    NumericLimits<uint64_t>::Maximum() / static_cast<uint64_t>(channel_count)) {
			throw InternalException("Audio resample helper returned overflowing dimensions");
		}
		auto sample_count_u64 = static_cast<uint64_t>(frame_count) * static_cast<uint64_t>(channel_count);
		if (sample_count_u64 > remaining_batch_bytes / sizeof(double) ||
		    sample_count_u64 > limits.max_output_bytes / sizeof(double) ||
		    sample_count_u64 > NumericLimits<idx_t>::Maximum()) {
			throw InternalException("Audio resample helper violated its output limit");
		}
		auto sample_count = NumericCast<idx_t>(sample_count_u64);
		ListVector::Reserve(samples_result, existing_samples + sample_count);
		auto &sample_child = ListVector::GetEntry(samples_result);
		auto sample_data = FlatVector::GetData<double>(sample_child);
		if (sample_count > 0) {
			auto sample_bytes = sample_count * sizeof(double);
			for (idx_t copied_bytes = 0; copied_bytes < sample_bytes;) {
				if (context.IsInterrupted()) {
					throw InterruptException();
				}
				auto copy_bytes = MinValue<idx_t>(sample_bytes - copied_bytes, AUDIO_RESULT_COPY_CHUNK_BYTES);
				auto target_data = reinterpret_cast<char *>(sample_data + existing_samples) + copied_bytes;
				auto target = py::memoryview::from_memory(target_data, NumericCast<py::ssize_t>(copy_bytes), false);
				auto read_value = value.attr("readinto")(std::move(target));
				if (!py::isinstance<py::int_>(read_value)) {
					throw InternalException("Audio resample spool returned a non-integer read size");
				}
				auto read_count = py::cast<int64_t>(read_value);
				if (read_count <= 0 || static_cast<uint64_t>(read_count) > copy_bytes) {
					throw InternalException("Audio resample spool ended before the declared output");
				}
				copied_bytes += NumericCast<idx_t>(read_count);
			}
		}
		value.attr("close")();
		FlatVector::Validity(sample_child).SetAllValid(existing_samples + sample_count);
		auto list_entries = FlatVector::GetData<list_entry_t>(samples_result);
		list_entries[row].offset = existing_samples;
		list_entries[row].length = sample_count;
		ListVector::SetListSize(samples_result, existing_samples + sample_count);
		FlatVector::Validity(samples_result).SetValid(row);
		FlatVector::Validity(sample_rate_result).SetValid(row);
		FlatVector::Validity(frames_result).SetValid(row);
		FlatVector::Validity(channels_result).SetValid(row);
		FlatVector::GetData<int64_t>(sample_rate_result)[row] = sample_rate;
		FlatVector::GetData<int64_t>(frames_result)[row] = NumericCast<int64_t>(frame_count);
		FlatVector::GetData<int64_t>(channels_result)[row] = NumericCast<int64_t>(channel_count);
	} catch (py::error_already_set &error) {
		if (!error.matches(PyExc_Exception)) {
			throw InterruptException();
		}
		if (read_error) {
			RethrowAudioReadError(read_error, "resample");
		}
		if (error.matches(PyExc_MemoryError)) {
			throw OutOfMemoryException("Audio resampling ran out of memory");
		}
		if (error.matches(PyExc_ImportError) ||
		    (audio_file_error_type.ptr() && error.matches(audio_file_error_type.ptr()))) {
			throw InvalidInputException("audio_resample() failed: %s", error.what());
		}
		if (error.matches(PyExc_OSError)) {
			throw IOException("audio_resample() failed: %s", error.what());
		}
		throw InternalException("audio_resample() Python helper failed unexpectedly: %s", error.what());
	}
}

static void AudioResampleFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &children = StructVector::GetEntries(result);
	D_ASSERT(children.size() == 4);
	auto &samples_result = *children[0];
	auto &sample_rate_result = *children[1];
	auto &frames_result = *children[2];
	auto &channels_result = *children[3];
	samples_result.SetVectorType(VectorType::FLAT_VECTOR);
	ListVector::SetListSize(samples_result, 0);
	sample_rate_result.SetVectorType(VectorType::FLAT_VECTOR);
	frames_result.SetVectorType(VectorType::FLAT_VECTOR);
	channels_result.SetVectorType(VectorType::FLAT_VECTOR);
	bool dependency_ready = false;

	for (idx_t row = 0; row < args.size(); row++) {
		auto file_value = args.data[0].GetValue(row);
		int64_t sample_rate = 0;
		AudioResampleLimits limits;
		if (file_value.IsNull() || !GetAudioResampleArguments(args, row, sample_rate, limits)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}

		auto &context = state.GetContext();
		try {
			if (!dependency_ready) {
				RequireAudioDependencies("audio_resample", true);
				dependency_ready = true;
			}
			auto file = FileReference::FromValue(file_value, "audio_resample");
			auto resolved = ResolvedFile::Open(context, file);
			CopyResampledAudio(context, *resolved, file, sample_rate, limits, row, samples_result, sample_rate_result,
			                   frames_result, channels_result);
			if (context.IsInterrupted()) {
				throw InterruptException();
			}
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
				RequireAudioDependencies("audio_metadata", false);
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

static ScalarFunction MakeAudioResampleFunction(vector<LogicalType> arguments) {
	ScalarFunction function("audio_resample", std::move(arguments), AudioResampleType(), AudioResampleFunction,
	                        BindAudioResample);
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

ScalarFunctionSet AudioFileFunctions::GetResampleFunctions() {
	ScalarFunctionSet result("audio_resample");
	result.AddFunction(MakeAudioResampleFunction({LogicalType::ANY, LogicalType::BIGINT}));
	result.AddFunction(
	    MakeAudioResampleFunction({LogicalType::ANY, LogicalType::BIGINT, LogicalType::UBIGINT, LogicalType::UBIGINT,
	                               LogicalType::UBIGINT, LogicalType::UBIGINT, LogicalType::UBIGINT}));
	return result;
}

} // namespace duckdb
