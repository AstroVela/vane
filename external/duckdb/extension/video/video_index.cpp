// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "video_index.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/database.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

extern "C" {
#include <libavutil/sha.h>
}

namespace duckdb {
namespace {

using Digest = std::array<uint8_t, 32>;
static constexpr uint64_t INDEX_BLOCK_BYTES = 64 * 1024;
static constexpr uint64_t INDEX_MAX_BYTES = 64 * MEDIA_MIB;
static constexpr uint64_t INDEX_HEADER_BYTES = 152;
static constexpr uint64_t INDEX_FRAME_BYTES = 88;
static constexpr const char *INDEX_MAGIC = "VVIDX001";

class Hash {
public:
	Hash() : hash(av_sha_alloc()) {
		if (!hash) {
			throw OutOfMemoryException("Cannot allocate video index digest state");
		}
		av_sha_init(hash, 256);
	}
	~Hash() {
		av_free(hash);
	}
	void Add(const void *data, uint64_t size) {
		av_sha_update(hash, static_cast<const uint8_t *>(data), size);
	}
	void Number(uint64_t value) {
		uint8_t bytes[8];
		for (unsigned i = 0; i < 8; i++) {
			bytes[i] = uint8_t(value >> (i * 8));
		}
		Add(bytes, 8);
	}
	Digest Finish() {
		Digest result;
		av_sha_final(hash, result.data());
		return result;
	}

private:
	AVSHA *hash;
};

static Digest HashBytes(ClientContext &context, const void *data, uint64_t size) {
	Hash hash;
	auto bytes = static_cast<const uint8_t *>(data);
	for (uint64_t offset = 0; offset < size;) {
		MediaInterrupt(context);
		auto count = MinValue<uint64_t>(MEDIA_MIB, size - offset);
		hash.Add(bytes + offset, count);
		offset += count;
	}
	return hash.Finish();
}

static Digest SourceBinding(ClientContext &context, const FileReference &reference, const ResolvedFile &file) {
	if (reference.url.size() + reference.content_type.size() + reference.checksum.size() > MEDIA_MIB) {
		throw OutOfRangeException("video index FILE metadata exceeds 1 MiB");
	}
	MemoryStream stream;
	BinarySerializer serializer(stream);
	serializer.Begin();
	serializer.WriteProperty(1, "file", reference.ToValue());
	serializer.WriteProperty(2, "stat", file.Stat().ToValue());
	serializer.End();
	return HashBytes(context, stream.GetData(), stream.GetPosition());
}

//! Hash meaningful plane bytes, excluding allocator padding. Decoder metadata
//! affecting pixel conversion is included; seek-dependent packet DTS is not.
static Digest FrameDigest(ClientContext &context, const AVFrame &frame) {
	Hash hash;
	for (auto value : {int64_t(frame.width), int64_t(frame.height), int64_t(frame.format), int64_t(frame.color_range),
	                   int64_t(frame.colorspace), int64_t(frame.color_primaries), int64_t(frame.color_trc),
	                   int64_t(frame.chroma_location)}) {
		hash.Number(uint64_t(value));
	}
	int lines[4] = {};
	MediaCheck(av_image_fill_linesizes(lines, AVPixelFormat(frame.format), frame.width), "index pixel layout");
	ptrdiff_t strides[4] = {lines[0], lines[1], lines[2], lines[3]};
	size_t sizes[4] = {};
	MediaCheck(av_image_fill_plane_sizes(sizes, AVPixelFormat(frame.format), frame.height, strides),
	           "index plane sizes");
	for (unsigned plane = 0; plane < 4; plane++) {
		if (!sizes[plane]) {
			continue;
		}
		if (!frame.data[plane]) {
			throw MediaFormatException("decoded frame has no indexed plane data");
		}
		if (!lines[plane]) { // The fixed-size palette of paletted pixel formats.
			hash.Add(frame.data[plane], sizes[plane]);
			continue;
		}
		if (lines[plane] < 0 || std::abs(int64_t(frame.linesize[plane])) < lines[plane] ||
		    sizes[plane] % uint64_t(lines[plane])) {
			throw MediaFormatException("decoded frame has an invalid indexed plane layout");
		}
		for (uint64_t row = 0; row < sizes[plane] / uint64_t(lines[plane]); row++) {
			MediaInterrupt(context);
			hash.Add(frame.data[plane] + int64_t(row) * frame.linesize[plane], lines[plane]);
		}
	}
	return hash.Finish();
}

struct IndexFrame {
	int64_t pts;
	int64_t dts;
	int64_t duration;
	uint32_t width;
	uint32_t height;
	int format;
	bool key;
	Digest digest;
	uint64_t anchor = 0;
};

struct VideoIndex {
	uint64_t source_size = 0;
	Digest binding;
	AVRational base;
	int64_t origin = 0;
	uint64_t build_bytes = 0;
	vector<Digest> blocks;
	vector<IndexFrame> frames;
	uint64_t SerializedSize() const {
		return INDEX_HEADER_BYTES + 32 + blocks.size() * 32 + frames.size() * INDEX_FRAME_BYTES;
	}
};

class IndexWriter {
public:
	string bytes;
	void Number(uint64_t value) {
		for (unsigned i = 0; i < 8; i++) {
			bytes.push_back(char(value >> (i * 8)));
		}
	}
	void DigestValue(const Digest &digest) {
		bytes.append(reinterpret_cast<const char *>(digest.data()), digest.size());
	}
};

static string EncodeIndex(ClientContext &context, const VideoIndex &index) {
	IndexWriter writer;
	writer.bytes.reserve(index.SerializedSize());
	writer.bytes.append(INDEX_MAGIC, 8);
	for (auto value :
	     {uint64_t(avcodec_version()), uint64_t(avformat_version()), uint64_t(avutil_version()), index.source_size}) {
		writer.Number(value);
	}
	writer.DigestValue(HashBytes(context, DuckDB::SourceID(), strlen(DuckDB::SourceID())));
	writer.DigestValue(index.binding);
	for (auto value : {uint64_t(index.base.num), uint64_t(index.base.den), uint64_t(index.origin), index.build_bytes,
	                   uint64_t(index.blocks.size()), uint64_t(index.frames.size())}) {
		writer.Number(value);
	}
	for (auto &block : index.blocks) {
		MediaInterrupt(context);
		writer.DigestValue(block);
	}
	for (auto &frame : index.frames) {
		MediaInterrupt(context);
		for (auto value : {uint64_t(frame.pts), uint64_t(frame.dts), uint64_t(frame.duration), uint64_t(frame.width),
		                   uint64_t(frame.height), uint64_t(frame.format), uint64_t(frame.key)}) {
			writer.Number(value);
		}
		writer.DigestValue(frame.digest);
	}
	writer.DigestValue(HashBytes(context, writer.bytes.data(), writer.bytes.size()));
	return std::move(writer.bytes);
}

class IndexParser {
public:
	explicit IndexParser(const string &bytes_p) : bytes(bytes_p) {
	}
	uint64_t Number() {
		if (8 > bytes.size() - offset) {
			throw InvalidInputException("truncated video index");
		}
		uint64_t result = 0;
		for (unsigned i = 0; i < 8; i++) {
			result |= uint64_t(uint8_t(bytes[offset++])) << (i * 8);
		}
		return result;
	}
	int64_t Signed() {
		auto value = Number();
		int64_t result;
		memcpy(&result, &value, sizeof(result));
		return result;
	}
	Digest DigestValue() {
		if (32 > bytes.size() - offset) {
			throw InvalidInputException("truncated video index digest");
		}
		Digest result;
		memcpy(result.data(), bytes.data() + offset, 32);
		offset += 32;
		return result;
	}
	const string &bytes;
	uint64_t offset = 8;
};

static unique_ptr<VideoIndex> DecodeIndex(ClientContext &context, const Value &value) {
	auto &bytes = StringValue::Get(value);
	if (bytes.size() > INDEX_MAX_BYTES) {
		throw OutOfRangeException("video index exceeds 64 MiB");
	}
	if (bytes.size() < INDEX_HEADER_BYTES + 32 || memcmp(bytes.data(), INDEX_MAGIC, 8) != 0) {
		throw InvalidInputException("invalid video index format");
	}
	auto expected = HashBytes(context, bytes.data(), bytes.size() - 32);
	if (memcmp(expected.data(), bytes.data() + bytes.size() - 32, 32) != 0) {
		throw InvalidInputException("video index integrity check failed");
	}
	IndexParser parser(bytes);
	if (parser.Number() != avcodec_version() || parser.Number() != avformat_version() ||
	    parser.Number() != avutil_version()) {
		throw InvalidInputException("video index requires the codec build that created it");
	}
	auto result = make_uniq<VideoIndex>();
	result->source_size = parser.Number();
	if (parser.DigestValue() != HashBytes(context, DuckDB::SourceID(), strlen(DuckDB::SourceID()))) {
		throw InvalidInputException("video index requires the engine SourceID that created it");
	}
	result->binding = parser.DigestValue();
	auto numerator = parser.Number(), denominator = parser.Number();
	result->origin = parser.Signed();
	result->build_bytes = parser.Number();
	auto block_count = parser.Number(), frame_count = parser.Number();
	if (!result->source_size || result->source_size > 16 * 1024 * MEDIA_MIB || !numerator || numerator > INT_MAX ||
	    !denominator || denominator > INT_MAX || block_count != (result->source_size - 1) / INDEX_BLOCK_BYTES + 1 ||
	    !frame_count || frame_count > (INDEX_MAX_BYTES - INDEX_HEADER_BYTES - 32) / INDEX_FRAME_BYTES ||
	    INDEX_HEADER_BYTES + 32 + block_count * 32 + frame_count * INDEX_FRAME_BYTES != bytes.size()) {
		throw InvalidInputException("invalid video index dimensions or counts");
	}
	result->base = {int(numerator), int(denominator)};
	result->blocks.reserve(block_count);
	for (uint64_t i = 0; i < block_count; i++) {
		MediaInterrupt(context);
		result->blocks.push_back(parser.DigestValue());
	}
	result->frames.reserve(frame_count);
	uint64_t anchor = 0;
	for (uint64_t i = 0; i < frame_count; i++) {
		MediaInterrupt(context);
		auto pts = parser.Signed(), dts = parser.Signed(), duration = parser.Signed();
		auto width = parser.Number(), height = parser.Number(), format = parser.Number(), key = parser.Number();
		if (pts == AV_NOPTS_VALUE || (i && pts <= result->frames.back().pts) || !width || !height ||
		    width > MEDIA_MAX_PIXELS || height > MEDIA_MAX_PIXELS / width || format >= AV_PIX_FMT_NB || key > 1 ||
		    (!i && !key)) {
			throw InvalidInputException("invalid video index frame record");
		}
		if (key) {
			anchor = i;
		}
		result->frames.push_back({pts, dts, duration, uint32_t(width), uint32_t(height), int(format), bool(key),
		                          parser.DigestValue(), anchor});
	}
	return result;
}

//! Verify whole logical blocks before passing any byte to the container parser.
//! The cache owns one block only; its memory and physical reads are bounded.
class IndexVerifier : public MediaReadVerifier {
public:
	IndexVerifier(ClientContext &context_p, const VideoIndex &index_p, uint64_t limit_p)
	    : context(context_p), index(index_p), limit(limit_p), buffer(INDEX_BLOCK_BYTES) {
	}
	void Read(const ResolvedFile &file, data_ptr_t target, uint64_t size, uint64_t offset) override {
		while (size) {
			MediaInterrupt(context);
			auto block = offset / INDEX_BLOCK_BYTES;
			if (block != cached) {
				auto start = block * INDEX_BLOCK_BYTES;
				auto count = MinValue<uint64_t>(INDEX_BLOCK_BYTES, index.source_size - start);
				if (count > limit - bytes) {
					throw OutOfRangeException("indexed video exceeds its physical read byte budget");
				}
				file.ReadExact(buffer.data(), count, start);
				bytes += count;
				if (HashBytes(context, buffer.data(), count) != index.blocks[block]) {
					throw InvalidInputException("video index source bytes have changed");
				}
				cached = block;
			}
			auto within = offset % INDEX_BLOCK_BYTES;
			auto count = MinValue<uint64_t>(size, INDEX_BLOCK_BYTES - within);
			memcpy(target, buffer.data() + within, count);
			target += count;
			offset += count;
			size -= count;
		}
	}
	uint64_t BytesRead() const override {
		return bytes;
	}

private:
	ClientContext &context;
	const VideoIndex &index;
	uint64_t limit;
	uint64_t bytes = 0;
	uint64_t cached = UINT64_MAX;
	vector<uint8_t> buffer;
};

static long double FrameTime(int64_t pts, AVRational base, int64_t origin) {
	return (static_cast<long double>(pts) - origin) * base.num / base.den;
}

struct Selection {
	explicit Selection(double start) : next_sample(start) {
	}
	long double next_sample;
	long double last_time = 0;
	bool have_time = false;
	bool Select(const VideoFrameOptions &options, uint64_t index, int64_t pts, AVRational base, int64_t origin,
	            bool key) {
		if (options.has_target_index) {
			return index == options.target_index;
		}
		bool has_time = pts != AV_NOPTS_VALUE && base.num > 0 && base.den > 0;
		if (!has_time && (options.start > 0 || options.has_end || options.has_interval)) {
			throw MediaFormatException("video time selection requires presentation timestamps");
		}
		auto time = has_time ? FrameTime(pts, base, origin) : 0;
		if (has_time) {
			if (have_time && time < last_time) {
				next_sample = options.start;
			}
			last_time = time;
			have_time = true;
			if (double(time) < options.start || (options.has_end && double(time) > options.end)) {
				return false;
			}
		}
		if (options.has_key && options.key != key) {
			return false;
		}
		if (options.has_interval) {
			auto tolerance = 4 * std::numeric_limits<double>::epsilon() *
			                 MaxValue<long double>(1, MaxValue<long double>(std::abs(time), std::abs(next_sample)));
			if (time + tolerance < next_sample) {
				return false;
			}
			next_sample += (std::floor((time + tolerance - next_sample) / options.interval) + 1) * options.interval;
		}
		return true;
	}
};

static void BuildVideoIndex(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &context = state.GetContext();
	uint64_t batch_bytes = 0;
	for (idx_t row = 0; row < args.size(); row++) {
		MediaInterrupt(context);
		bool null = false;
		for (idx_t column = 0; column < args.ColumnCount(); column++) {
			null |= args.data[column].GetValue(row).IsNull();
		}
		if (null) {
			result.SetValue(row, Value(LogicalType::BLOB));
			continue;
		}
		auto input_limit = MediaPositive(args.data[1].GetValue(row), "max_input_bytes", 16 * 1024 * MEDIA_MIB);
		auto frame_limit = MediaPositive(args.data[2].GetValue(row), "max_decoded_frames", 100000000);
		auto pixel_limit = MediaPositive(args.data[3].GetValue(row), "max_pixels", VideoFrameContract::MAX_PIXELS);
		auto index_limit = MediaPositive(args.data[4].GetValue(row), "max_index_bytes", INDEX_MAX_BYTES);
		index_limit = MinValue<uint64_t>(index_limit, MEDIA_BATCH_BYTES - batch_bytes);
		auto reference = FileReference::FromValue(args.data[0].GetValue(row), "build_video_index");
		auto resolved = ResolvedFile::Open(context, reference);
		VideoIndex index;
		index.source_size = resolved->LogicalSize();
		if (!index.source_size || index.source_size > input_limit) {
			throw OutOfRangeException("video index input exceeds max_input_bytes or is empty");
		}
		index.binding = SourceBinding(context, reference, *resolved);
		auto blocks = (index.source_size - 1) / INDEX_BLOCK_BYTES + 1;
		if (INDEX_HEADER_BYTES + 32 + blocks * 32 > index_limit) {
			throw OutOfRangeException("video index exceeds max_index_bytes or the 256 MiB batch budget");
		}
		index.blocks.reserve(blocks);
		vector<uint8_t> buffer(INDEX_BLOCK_BYTES);
		for (uint64_t offset = 0; offset < index.source_size;) {
			MediaInterrupt(context);
			auto count = MinValue<uint64_t>(INDEX_BLOCK_BYTES, index.source_size - offset);
			resolved->ReadExact(buffer.data(), count, offset);
			index.blocks.push_back(HashBytes(context, buffer.data(), count));
			offset += count;
		}
		auto handle = resolved.get();
		MediaReader reader(context, reference, std::move(resolved), AVMEDIA_TYPE_VIDEO, input_limit, input_limit * 4,
		                   pixel_limit, MEDIA_MAX_FRAME_BYTES, MEDIA_METADATA_BYTES,
		                   make_uniq<IndexVerifier>(context, index, input_limit * 4));
		index.base = reader.Stream().time_base;
		index.origin = reader.Stream().start_time == AV_NOPTS_VALUE ? 0 : reader.Stream().start_time;
		if (index.base.num <= 0 || index.base.den <= 0) {
			throw NotImplementedException("video indexing requires a known stream time base");
		}
		uint64_t anchor = 0;
		while (reader.NextFrame()) {
			if (index.frames.size() >= frame_limit) {
				throw OutOfRangeException("video indexing exceeds max_decoded_frames");
			}
			if (INDEX_FRAME_BYTES > index_limit - index.SerializedSize()) {
				throw OutOfRangeException("video index exceeds max_index_bytes or the 256 MiB batch budget");
			}
			auto &frame = reader.Frame();
			bool key = (frame.flags & AV_FRAME_FLAG_KEY) != 0;
			if (frame.width <= 0 || frame.height <= 0) {
				throw MediaFormatException("video indexing requires known decoded frame dimensions");
			}
			if (frame.pts == AV_NOPTS_VALUE || (!index.frames.empty() && frame.pts <= index.frames.back().pts) ||
			    (index.frames.empty() && !key)) {
				throw NotImplementedException(
				    "video indexing requires unique increasing presentation timestamps and an initial keyframe");
			}
			if (key) {
				anchor = index.frames.size();
			}
			index.frames.push_back({frame.pts, frame.pkt_dts, frame.duration, uint32_t(frame.width),
			                        uint32_t(frame.height), frame.format, key, FrameDigest(context, frame), anchor});
		}
		reader.CheckIO();
		if (index.frames.empty()) {
			throw MediaFormatException("video index has no decoded frames");
		}
		if (SourceBinding(context, reference, *handle) != index.binding) {
			throw InvalidInputException("video index source metadata changed during indexing");
		}
		index.build_bytes = index.source_size + reader.BytesRead();
		auto bytes = EncodeIndex(context, index);
		batch_bytes += bytes.size();
		result.SetValue(row, Value::BLOB_RAW(bytes));
	}
}

static LogicalType IndexInfoType() {
	return LogicalType::STRUCT({{"frame_count", LogicalType::UBIGINT},
	                            {"keyframe_count", LogicalType::UBIGINT},
	                            {"source_bytes", LogicalType::UBIGINT},
	                            {"index_bytes", LogicalType::UBIGINT},
	                            {"build_bytes_read", LogicalType::UBIGINT},
	                            {"codec_version", LogicalType::VARCHAR}});
}

static void InspectVideoIndex(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	for (idx_t row = 0; row < args.size(); row++) {
		MediaInterrupt(state.GetContext());
		auto value = VideoFrameContract::ReadIndex(args.data[0], row);
		if (value.IsNull()) {
			result.SetValue(row, Value(result.GetType()));
			continue;
		}
		auto index = DecodeIndex(state.GetContext(), value);
		uint64_t keys = 0;
		for (auto &frame : index->frames) {
			keys += frame.key;
		}
		result.SetValue(row, Value::STRUCT(result.GetType(),
		                                   {Value::UBIGINT(index->frames.size()), Value::UBIGINT(keys),
		                                    Value::UBIGINT(index->source_size), Value::UBIGINT(index->SerializedSize()),
		                                    Value::UBIGINT(index->build_bytes), Value(av_version_info())}));
	}
}

} // namespace

struct VideoFrameCursor::State {
	ClientContext &context;
	VideoFrameOptions options;
	Selection selection;
	unique_ptr<VideoIndex> index;
	unique_ptr<MediaReader> reader;
	uint64_t decoded = 0;
	uint64_t selected = 0;
	uint64_t seeks = 0;
	uint64_t candidate = 0;
	uint64_t current = 0;
	bool positioned = false;
	bool finished = false;
	int64_t origin;

	State(ClientContext &context_p, const Value &file, const VideoFrameOptions &options_p, const Value &index_value)
	    : context(context_p), options(options_p), selection(options.start) {
		if (!index_value.IsNull()) {
			index = DecodeIndex(context, index_value);
		}
		auto reference = FileReference::FromValue(file, "video frame selection");
		auto resolved = ResolvedFile::Open(context, reference);
		unique_ptr<MediaReadVerifier> verifier;
		if (index) {
			if (resolved->LogicalSize() != index->source_size ||
			    SourceBinding(context, reference, *resolved) != index->binding) {
				throw InvalidInputException("video index does not match the FILE view or source metadata");
			}
			verifier = make_uniq<IndexVerifier>(context, *index, options.max_input_bytes * 4);
		}
		reader = make_uniq<MediaReader>(context, reference, std::move(resolved), AVMEDIA_TYPE_VIDEO,
		                                options.max_input_bytes, options.max_input_bytes * 4, options.max_pixels,
		                                MEDIA_MAX_FRAME_BYTES, MEDIA_METADATA_BYTES, std::move(verifier));
		auto stream_origin = reader->Stream().start_time;
		origin = stream_origin == AV_NOPTS_VALUE ? 0 : stream_origin;
		if (index && (index->base.num != reader->Stream().time_base.num ||
		              index->base.den != reader->Stream().time_base.den || index->origin != origin)) {
			throw InvalidInputException("video index stream metadata does not match");
		}
	}

	bool Decode() {
		if (!reader->NextFrame()) {
			return false;
		}
		if (decoded >= options.max_decoded_frames) {
			throw OutOfRangeException("video exceeds max_decoded_frames");
		}
		decoded++;
		return true;
	}

	bool NextIndexed() {
		uint64_t target = 0;
		bool found = false;
		if (options.has_target_index) {
			target = options.target_index;
			finished = true;
			found = target < index->frames.size();
		} else {
			while (candidate < index->frames.size()) {
				MediaInterrupt(context);
				target = candidate++;
				auto &frame = index->frames[target];
				if (options.has_end && double(duckdb::FrameTime(frame.pts, index->base, origin)) > options.end) {
					finished = true;
					break;
				}
				if (selection.Select(options, target, frame.pts, index->base, origin, frame.key)) {
					found = true;
					break;
				}
			}
		}
		if (!found) {
			finished = true;
			return false;
		}
		auto anchor = index->frames[target].anchor;
		if (!positioned || anchor > current + 1) {
			try {
				reader->Seek(index->frames[anchor].pts);
			} catch (const MediaFormatException &error) {
				throw NotImplementedException("indexed video keyframe seek is unavailable: %s", error.what());
			}
			seeks++;
			positioned = false;
		}
		for (;;) {
			try {
				if (!Decode()) {
					throw InvalidInputException("indexed video ended before the requested frame");
				}
			} catch (const MediaFormatException &error) {
				throw NotImplementedException("video keyframe seek cannot decode the indexed frame: %s", error.what());
			}
			auto &frame = reader->Frame();
			if (!positioned) {
				auto entry = std::lower_bound(index->frames.begin(), index->frames.end(), frame.pts,
				                              [](const IndexFrame &record, int64_t pts) { return record.pts < pts; });
				current = uint64_t(entry - index->frames.begin());
				positioned = true;
			} else {
				current++;
			}
			if (current > target || current >= index->frames.size() || frame.pts != index->frames[current].pts) {
				throw NotImplementedException("video keyframe seek cannot reproduce the indexed presentation order");
			}
			auto &expected = index->frames[current];
			if (FrameDigest(context, frame) != expected.digest ||
			    ((frame.flags & AV_FRAME_FLAG_KEY) != 0) != expected.key) {
				throw NotImplementedException("video keyframe seek cannot reproduce the indexed frame");
			}
			if (current == target) {
				// These describe the original presentation-order decode, including
				// packets which were drained differently after a keyframe seek.
				frame.pkt_dts = expected.dts;
				frame.duration = expected.duration;
				return true;
			}
		}
	}

	bool Next() {
		MediaInterrupt(context);
		if (finished) {
			return false;
		}
		if (index) {
			if (!NextIndexed()) {
				return false;
			}
		} else {
			for (;;) {
				if (!Decode()) {
					finished = true;
					return false;
				}
				current = decoded - 1;
				auto &frame = reader->Frame();
				if (selection.Select(options, current, frame.pts, reader->Stream().time_base, origin,
				                     (frame.flags & AV_FRAME_FLAG_KEY) != 0)) {
					finished = options.has_target_index;
					break;
				}
			}
		}
		selected++;
		reader->CheckIO();
		return true;
	}
};

VideoFrameCursor::VideoFrameCursor(ClientContext &context, const Value &file, const VideoFrameOptions &options,
                                   VideoFrameOperation, const Value &index)
    : state(make_uniq<State>(context, file, options, index)) {
}
VideoFrameCursor::~VideoFrameCursor() = default;
bool VideoFrameCursor::Next() {
	return state->Next();
}
MediaReader &VideoFrameCursor::Reader() {
	return *state->reader;
}
uint64_t VideoFrameCursor::FrameIndex() const {
	return state->current;
}
Value VideoFrameCursor::FrameTime() const {
	auto pts = state->reader->Frame().pts;
	auto base = state->reader->Stream().time_base;
	return pts != AV_NOPTS_VALUE && base.num > 0 && base.den > 0
	           ? Value::DOUBLE(double(duckdb::FrameTime(pts, base, state->origin)))
	           : Value(LogicalType::DOUBLE);
}
Value VideoFrameCursor::Statistics() const {
	return Value::STRUCT(VideoFrameContract::ResultType(VideoFrameOperation::SCAN_STATS),
	                     {Value::UBIGINT(state->reader->BytesRead()), Value::UBIGINT(state->decoded),
	                      Value::UBIGINT(state->seeks), Value::UBIGINT(state->selected)});
}

void RegisterVideoIndexFunctions(ExtensionLoader &loader) {
	loader.RegisterFunction(MediaScalar(
	    "build_video_index",
	    {LogicalType::ANY, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
	    LogicalType::BLOB, BuildVideoIndex));
	ScalarFunction info("video_index_info", {LogicalType::BLOB}, IndexInfoType(), InspectVideoIndex);
	info.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	info.SetFallible();
	loader.RegisterFunction(std::move(info));
}

} // namespace duckdb
