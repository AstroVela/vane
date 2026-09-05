// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "media_reader.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/function/distributed_table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include <atomic>
#include <cmath>
#include <limits>
#include "duckdb/common/unordered_set.hpp"

namespace duckdb {
namespace {

static void VideoMetadata(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	for (idx_t row = 0; row < args.size(); row++) {
		auto value = args.data[0].GetValue(row);
		if (value.IsNull() || (args.ColumnCount() == 2 && args.data[1].GetValue(row).IsNull())) {
			result.SetValue(row, Value(result.GetType()));
			continue;
		}
		auto budget = args.ColumnCount() == 2 ? MediaPositive(args.data[1].GetValue(row), "max_bytes", 64 * MEDIA_MIB)
		                                      : MEDIA_METADATA_BYTES;
		MediaReader reader(state.GetContext(), FileReference::FromValue(value, "native_video_metadata"),
		                   AVMEDIA_TYPE_VIDEO, INT64_MAX, budget);
		auto &stream = reader.Stream();
		auto &parameters = *stream.codecpar;
		if (parameters.width <= 0 || parameters.height <= 0) {
			throw MediaFormatException("unknown video dimensions");
		}
		MediaProduct(parameters.width, parameters.height, MEDIA_MAX_PIXELS, "video pixels");
		auto fps = av_guess_frame_rate(&reader.Format(), &stream, nullptr);
		Value duration(LogicalType::DOUBLE), frame_count(LogicalType::BIGINT), rate(LogicalType::DOUBLE);
		if (stream.duration != AV_NOPTS_VALUE && stream.duration >= 0 && stream.time_base.num > 0 &&
		    stream.time_base.den > 0) {
			duration = Value::DOUBLE(stream.duration * av_q2d(stream.time_base));
		}
		if (stream.nb_frames > 0) {
			frame_count = Value::BIGINT(stream.nb_frames);
		}
		if (fps.num > 0 && fps.den > 0) {
			rate = Value::DOUBLE(av_q2d(fps));
		}
		auto time_base_type = StructType::GetChildType(result.GetType(), 5);
		Value time_base(time_base_type);
		if (stream.time_base.num > 0 && stream.time_base.den > 0) {
			time_base = Value::STRUCT(time_base_type,
			                          {Value::BIGINT(stream.time_base.num), Value::BIGINT(stream.time_base.den)});
		}
		result.SetValue(
		    row, Value::STRUCT(result.GetType(), {Value::UINTEGER(parameters.width), Value::UINTEGER(parameters.height),
		                                          rate, duration, frame_count, time_base}));
	}
}

static uint64_t VideoFileBytes(const Value &file) {
	if (file.IsNull() || !FileLogicalType::IsFile(file.type()) ||
	    FileLogicalType::GetMediaType(file.type()) != FileMediaType::VIDEO) {
		throw InvalidInputException("native video source requires non-NULL VIDEOFILE views");
	}
	auto reference = FileReference::FromValue(file, "native_video_frames");
	return reference.url.size() + reference.content_type.size() + reference.checksum.size() + 512;
}

// Parameters are portable values only. Opening files is execution-time work.
struct VideoBind : public TableFunctionData {
	vector<Value> parameters;
	vector<Value> files;
	bool tensor = false;
	bool worker = false;
	bool assigned = false;
	uint64_t row_bytes = 0;
	idx_t TaskCount() const {
		if (files.empty()) {
			return 0;
		}
		if (worker || !parameters[10].IsNull()) {
			return 1;
		}
		return parameters[12].IsNull() ? files.size() : MinValue<idx_t>(files.size(), parameters[12].GetValue<idx_t>());
	}
	pair<idx_t, idx_t> FileRange(idx_t task) const {
		auto count = TaskCount();
		auto size = files.size() / count;
		auto larger_groups = files.size() % count;
		auto start = task * size + MinValue(task, larger_groups);
		return {start, start + size + (task < larger_groups ? 1 : 0)};
	}
	uint64_t RowBytes() const {
		uint64_t file_bytes = 0;
		for (auto &value : files) {
			auto file = FileReference::FromValue(value, "native_video_frames");
			file_bytes =
			    MaxValue<uint64_t>(file_bytes, file.url.size() + file.content_type.size() + file.checksum.size());
		}
		return parameters[0].GetValue<uint64_t>() * parameters[1].GetValue<uint64_t>() * 3 + file_bytes + 160;
	}
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<VideoBind>(*this);
	}
	bool Equals(const FunctionData &other) const override {
		auto value = dynamic_cast<const VideoBind *>(&other);
		return value && parameters == value->parameters && files == value->files && tensor == value->tensor &&
		       worker == value->worker && assigned == value->assigned;
	}
};

static void ValidateVideoBind(VideoBind &bind) {
	auto &p = bind.parameters;
	if (p.size() != 13) {
		throw InvalidInputException("native video requires thirteen scan options");
	}
	const idx_t integer_indices[] = {0, 1, 6, 7, 8, 9};
	const char *names[] = {
	    "height", "width", "max_input_bytes", "max_decoded_frames", "max_pixels", "max_partition_bytes"};
	const uint64_t maxima[] = {100000, 100000, 16 * 1024 * MEDIA_MIB, 100000000, MEDIA_MAX_PIXELS, MEDIA_BATCH_BYTES};
	for (idx_t i = 0; i < 6; i++) {
		MediaPositive(p[integer_indices[i]], names[i], maxima[i]);
	}
	MediaProduct(p[0].GetValue<uint64_t>(), p[1].GetValue<uint64_t>(), p[8].GetValue<uint64_t>(),
	             "video output pixels");
	auto pixels = p[0].GetValue<uint64_t>() * p[1].GetValue<uint64_t>();
	MediaProduct(pixels, 3, p[9].GetValue<uint64_t>(), "video output frame bytes");
	if (p[2].IsNull()) {
		throw InvalidInputException("start_time cannot be NULL");
	}
	for (auto i : {2, 3, 5}) {
		if (!p[i].IsNull() && (!std::isfinite(p[i].GetValue<double>()) || p[i].GetValue<double>() < 0)) {
			throw InvalidInputException("video times must be finite and nonnegative");
		}
	}
	if (!p[3].IsNull() && p[3].GetValue<double>() < p[2].GetValue<double>()) {
		throw InvalidInputException("end_time must be at least start_time");
	}
	if (!p[5].IsNull() && p[5].GetValue<double>() == 0) {
		throw InvalidInputException("sample_interval_seconds must be positive");
	}
	if (!p[10].IsNull() && p[10].GetValue<int64_t>() < 0) {
		throw InvalidInputException("frame_limit must be nonnegative");
	}
	if (p[11].IsNull() || (p[11].GetValue<string>() != "raise" && p[11].GetValue<string>() != "skip")) {
		throw InvalidInputException("video on_error must be raise or skip");
	}
	if (!p[12].IsNull()) {
		MediaPositive(p[12], "read_task_count", NumericLimits<int64_t>::Maximum());
	}
	if (bind.files.size() > 100000) {
		throw OutOfRangeException("native video source exceeds 100000 FILE views");
	}
	uint64_t source_bytes = 0;
	for (auto &file : bind.files) {
		auto bytes = VideoFileBytes(file);
		if (bytes > 64 * MEDIA_MIB - source_bytes) {
			throw OutOfRangeException("native video source metadata exceeds 64 MiB");
		}
		source_bytes += bytes;
	}
	bind.row_bytes = bind.RowBytes();
	if (bind.row_bytes > p[9].GetValue<uint64_t>()) {
		throw OutOfRangeException("native video row exceeds max_partition_bytes");
	}
}

static unique_ptr<FunctionData> BindVideo(ClientContext &, TableFunctionBindInput &input, vector<LogicalType> &types,
                                          vector<string> &names) {
	auto result = make_uniq<VideoBind>();
	result->parameters.assign(input.inputs.begin() + 1, input.inputs.end());
	if (!input.inputs[0].IsNull()) {
		if (input.inputs[0].type().id() != LogicalTypeId::LIST) {
			throw BinderException("native video source requires a list of VIDEOFILE views");
		}
		result->files = ListValue::GetChildren(input.inputs[0]);
	}
	result->tensor = input.table_function.name == "native_video_tensor_frames";
	ValidateVideoBind(*result);
	types.push_back(FileLogicalType::Create(FileMediaType::VIDEO));
	names.push_back("file");
	auto frame_type = MediaVideoFrameType();
	for (auto &field : StructType::GetChildTypes(frame_type)) {
		names.push_back(field.first);
		types.push_back(field.second);
	}
	names.back() = "frame";
	if (result->tensor) {
		types.back() = TensorType::Create(LogicalType::UTINYINT, {result->parameters[0].GetValue<idx_t>(),
		                                                          result->parameters[1].GetValue<idx_t>(), 3});
	}
	return std::move(result);
}

struct VideoGlobal : public GlobalTableFunctionState {
	std::atomic<idx_t> next_task {0};
	idx_t thread_count = 1;
	uint64_t emitted = 0; // Used only in the explicitly serial, globally limited scan.
	idx_t MaxThreads() const override {
		return thread_count;
	}
};
struct VideoLocal : public LocalTableFunctionState {
	unique_ptr<MediaReader> reader;
	idx_t file_index = 0;
	idx_t next_file = 0;
	idx_t end_file = 0;
	uint64_t decoded = 0;
	int64_t origin = AV_NOPTS_VALUE;
	long double next_sample = 0;
	long double last_time = 0;
	bool have_last_time = false;
};
static unique_ptr<GlobalTableFunctionState> InitVideo(ClientContext &, TableFunctionInitInput &input) {
	auto &bind = input.bind_data->Cast<VideoBind>();
	if (bind.worker && !bind.assigned) {
		throw InvalidInputException("native video worker has no split assignment");
	}
	auto result = make_uniq<VideoGlobal>();
	result->thread_count = MaxValue<idx_t>(1, bind.TaskCount());
	return std::move(result);
}
static unique_ptr<LocalTableFunctionState> InitVideoLocal(ExecutionContext &, TableFunctionInitInput &,
                                                          GlobalTableFunctionState *) {
	return make_uniq<VideoLocal>();
}

static void ScanVideo(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
	auto &bind = input.bind_data->Cast<VideoBind>();
	auto &p = bind.parameters;
	auto &global = input.global_state->Cast<VideoGlobal>();
	auto &local = input.local_state->Cast<VideoLocal>();
	auto height = p[0].GetValue<uint32_t>(), width = p[1].GetValue<uint32_t>();
	auto frame_bytes = uint64_t(height) * width * 3;
	auto limit = MinValue<idx_t>(STANDARD_VECTOR_SIZE, p[9].GetValue<uint64_t>() / bind.row_bytes);
	bool limited = !p[10].IsNull();
	if (limited && global.emitted >= p[10].GetValue<uint64_t>()) {
		return;
	}
	idx_t row = 0;
	uint64_t allocated_pixels = 0;
	while (row < limit) {
		MediaInterrupt(context);
		if (limited && global.emitted >= p[10].GetValue<uint64_t>()) {
			local.reader.reset();
			break;
		}
		try {
			if (!local.reader) {
				if (local.next_file == local.end_file) {
					auto task = global.next_task.fetch_add(1);
					if (task >= bind.TaskCount()) {
						break;
					}
					auto range = bind.FileRange(task);
					local.next_file = range.first;
					local.end_file = range.second;
				}
				local.file_index = local.next_file++;
				auto file = FileReference::FromValue(bind.files[local.file_index], "native_video_frames");
				auto input_bytes = p[6].GetValue<uint64_t>();
				local.reader = make_uniq<MediaReader>(context, file, AVMEDIA_TYPE_VIDEO, input_bytes, input_bytes * 4,
				                                      p[8].GetValue<uint64_t>());
				local.origin = local.reader->Stream().start_time;
				if (local.origin == AV_NOPTS_VALUE) {
					local.origin = 0;
				}
				local.have_last_time = false;
				local.decoded = 0;
				local.next_sample = p[2].GetValue<double>();
			}
			if (!local.reader->NextFrame()) {
				local.reader.reset();
				continue;
			}
			if (local.decoded >= p[7].GetValue<uint64_t>()) {
				throw OutOfRangeException("native video exceeds max_decoded_frames");
			}
			auto frame_index = local.decoded++;
			auto &frame = local.reader->Frame();
			auto base = local.reader->Stream().time_base;
			auto pts = frame.pts;
			bool has_time = pts != AV_NOPTS_VALUE && base.num > 0 && base.den > 0;
			long double time = has_time ? (static_cast<long double>(pts) - local.origin) * base.num / base.den : 0;
			if (!has_time && (p[2].GetValue<double>() > 0 || !p[3].IsNull() || !p[5].IsNull())) {
				throw MediaFormatException("video time selection requires presentation timestamps");
			}
			if (has_time) {
				if (local.have_last_time && time < local.last_time) {
					local.next_sample = p[2].GetValue<double>();
				}
				local.last_time = time;
				local.have_last_time = true;
				// A later timestamp discontinuity may return to this inclusive window.
				if (!p[3].IsNull() && double(time) > p[3].GetValue<double>()) {
					continue;
				}
				if (double(time) < p[2].GetValue<double>()) {
					continue;
				}
			}
			bool key = (frame.flags & AV_FRAME_FLAG_KEY) != 0;
			if (!p[4].IsNull() && key != p[4].GetValue<bool>()) {
				continue;
			}
			if (!p[5].IsNull()) {
				auto tolerance =
				    4 * std::numeric_limits<double>::epsilon() *
				    MaxValue<long double>(1, MaxValue<long double>(std::abs(time), std::abs(local.next_sample)));
				if (time + tolerance < local.next_sample) {
					continue;
				}
				auto interval = static_cast<long double>(p[5].GetValue<double>());
				local.next_sample += (std::floor((time + tolerance - local.next_sample) / interval) + 1) * interval;
			}
			output.SetValue(0, row, bind.files[local.file_index]);
			output.SetValue(1, row, Value::BIGINT(NumericCast<int64_t>(frame_index)));
			output.SetValue(2, row, has_time ? Value::DOUBLE(double(time)) : Value(LogicalType::DOUBLE));
			output.SetValue(3, row, base.num > 0 ? Value::BIGINT(base.num) : Value(LogicalType::BIGINT));
			output.SetValue(4, row, base.den > 0 ? Value::BIGINT(base.den) : Value(LogicalType::BIGINT));
			output.SetValue(5, row, pts != AV_NOPTS_VALUE ? Value::BIGINT(pts) : Value(LogicalType::BIGINT));
			output.SetValue(
			    6, row, frame.pkt_dts != AV_NOPTS_VALUE ? Value::BIGINT(frame.pkt_dts) : Value(LogicalType::BIGINT));
			output.SetValue(7, row, frame.duration > 0 ? Value::BIGINT(frame.duration) : Value(LogicalType::BIGINT));
			output.SetValue(8, row, Value::BOOLEAN(key));
			if (bind.tensor) {
				auto &data = ArrayVector::GetEntry(output.data[9]);
				MediaConvertPixels(context, frame, "RGB", width, height,
				                   FlatVector::GetData<uint8_t>(data) + row * frame_bytes);
				FlatVector::SetNull(output.data[9], row, false);
			} else {
				if (frame_bytes > p[9].GetValue<uint64_t>() - allocated_pixels) {
					throw OutOfRangeException("native video failed conversions exhausted the batch byte budget");
				}
				allocated_pixels += frame_bytes;
				MediaWriteImage(context, frame, "RGB", width, height, output.data[9], row, frame_bytes);
			}
			row++;
			if (limited) {
				global.emitted++;
			}
		} catch (const MediaFormatException &) {
			MediaInterrupt(context);
			if (p[11].GetValue<string>() != "skip") {
				throw;
			}
			local.reader.reset();
		}
	}
	output.SetCardinality(row);
}

static string EncodeFiles(const vector<Value> &files) {
	MemoryStream stream;
	BinarySerializer serializer(stream);
	serializer.Begin();
	serializer.WriteProperty(1, "files", files);
	serializer.End();
	return string(const_char_ptr_cast(stream.GetData()), stream.GetPosition());
}
static vector<Value> DecodeFiles(const string &bytes) {
	MemoryStream stream(data_ptr_cast(const_cast<char *>(bytes.data())), bytes.size());
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto files = deserializer.ReadProperty<vector<Value>>(1, "files");
	deserializer.End();
	if (stream.GetPosition() != bytes.size()) {
		throw SerializationException("native video split has trailing bytes");
	}
	return files;
}
static vector<DistributedScanSplit> PlanVideoSplits(const TableFunctionDistributedScanPlanningInput &input) {
	auto &bind = input.bind_data->Cast<VideoBind>();
	vector<DistributedScanSplit> splits;
	auto append = [&](const vector<Value> &files) {
		DistributedScanSplit split;
		split.split_id = std::to_string(splits.size());
		split.payload = EncodeFiles(files);
		split.Validate();
		splits.push_back(std::move(split));
	};
	for (idx_t task = 0; task < bind.TaskCount(); task++) {
		auto range = bind.FileRange(task);
		append(vector<Value>(bind.files.begin() + range.first, bind.files.begin() + range.second));
	}
	return splits;
}
static unique_ptr<FunctionData> WorkerVideoBind(const TableFunctionDistributedScanInput &input) {
	auto result = input.bind_data->Cast<VideoBind>().Copy();
	auto &bind = result->Cast<VideoBind>();
	bind.files.clear();
	bind.worker = true;
	bind.assigned = false;
	return result;
}
static void ApplyVideoSplits(optional_ptr<FunctionData> data, const vector<DistributedScanSplit> &splits) {
	auto &bind = data->Cast<VideoBind>();
	if (!bind.worker) {
		throw SerializationException("native video splits require a worker bind");
	}
	bind.files.clear();
	if (!bind.parameters[10].IsNull() && (splits.size() > 1 || (!splits.empty() && splits[0].split_id != "0"))) {
		throw SerializationException("globally limited native video requires one canonical split");
	}
	unordered_set<string> seen;
	uint64_t source_bytes = 0;
	for (auto &split : splits) {
		split.Validate();
		if (!seen.insert(split.split_id).second) {
			throw SerializationException("duplicate native video split");
		}
		auto files = DecodeFiles(split.payload);
		if (files.empty() || (bind.parameters[10].IsNull() && bind.parameters[12].IsNull() && files.size() != 1)) {
			throw SerializationException("native video split has an invalid FILE group");
		}
		if (files.size() > 100000 - bind.files.size()) {
			throw OutOfRangeException("native video source exceeds 100000 FILE views");
		}
		for (auto &file : files) {
			auto bytes = VideoFileBytes(file);
			if (bytes > 64 * MEDIA_MIB - source_bytes) {
				throw OutOfRangeException("native video source metadata exceeds 64 MiB");
			}
			source_bytes += bytes;
		}
		bind.files.insert(bind.files.end(), files.begin(), files.end());
	}
	ValidateVideoBind(bind);
	bind.assigned = true;
}
static void SerializeVideo(Serializer &serializer, optional_ptr<FunctionData> data, const TableFunction &) {
	auto &bind = data->Cast<VideoBind>();
	serializer.WriteProperty(101, "parameters", bind.parameters);
	serializer.WriteProperty(102, "files", bind.files);
	serializer.WriteProperty(103, "tensor", bind.tensor);
	serializer.WriteProperty(104, "worker", bind.worker);
	serializer.WriteProperty(105, "assigned", bind.assigned);
}
static unique_ptr<FunctionData> DeserializeVideo(Deserializer &deserializer, TableFunction &function) {
	auto result = make_uniq<VideoBind>();
	result->parameters = deserializer.ReadProperty<vector<Value>>(101, "parameters");
	result->files = deserializer.ReadProperty<vector<Value>>(102, "files");
	result->tensor = deserializer.ReadProperty<bool>(103, "tensor");
	if (result->tensor != (function.name == "native_video_tensor_frames")) {
		throw SerializationException("native video output type does not match its registered function");
	}
	result->worker = deserializer.ReadProperty<bool>(104, "worker");
	result->assigned = deserializer.ReadProperty<bool>(105, "assigned");
	if ((!result->worker && result->assigned) || (result->worker && !result->assigned && !result->files.empty())) {
		throw SerializationException("native video contains invalid worker assignment state");
	}
	ValidateVideoBind(*result);
	return std::move(result);
}
} // namespace

void RegisterMediaVideo(ExtensionLoader &loader) {
	ScalarFunctionSet metadata("native_video_metadata");
	metadata.AddFunction(MediaScalar("video_metadata", {LogicalType::ANY}, MediaVideoMetadataType(), VideoMetadata));
	metadata.AddFunction(MediaScalar("video_metadata", {LogicalType::ANY, LogicalType::UBIGINT},
	                                 MediaVideoMetadataType(), VideoMetadata));
	loader.RegisterFunction(metadata);
	for (const auto &name : {"native_video_frames", "native_video_tensor_frames"}) {
		TableFunction function(name,
		                       {LogicalType::ANY, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::DOUBLE,
		                        LogicalType::DOUBLE, LogicalType::BOOLEAN, LogicalType::DOUBLE, LogicalType::BIGINT,
		                        LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT,
		                        LogicalType::VARCHAR, LogicalType::BIGINT},
		                       ScanVideo, BindVideo, InitVideo, InitVideoLocal);
		function.serialize = SerializeVideo;
		function.deserialize = DeserializeVideo;
		TableFunctionDistributedScanCallbacks callbacks;
		callbacks.protocol_version = 1;
		callbacks.split_codec = {"vane.native-video-files", 1};
		callbacks.plan_splits = PlanVideoSplits;
		callbacks.create_worker_bind = WorkerVideoBind;
		callbacks.apply_splits = ApplyVideoSplits;
		function.SetDistributedScanCallbacks(std::move(callbacks));
		loader.RegisterFunction(function);
	}
}
} // namespace duckdb
