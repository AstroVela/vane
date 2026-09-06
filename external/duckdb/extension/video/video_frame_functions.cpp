// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "media_reader.hpp"
#include "video_frame_contract.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include <limits>

namespace duckdb {
namespace {

//! Presentation-order decoding is the exact-index baseline, including B frames.
struct ScalarFrameSelection {
	uint64_t decoded = 0;
	long double next_sample;
	long double last_time = 0;
	bool have_last_time = false;

	explicit ScalarFrameSelection(double start) : next_sample(start) {
	}

	bool Select(MediaReader &reader, const VideoFrameOptions &options, VideoFrameOperation operation, uint64_t &index,
	            Value &timestamp) {
		if (decoded >= options.max_decoded_frames) {
			throw OutOfRangeException("video exceeds max_decoded_frames");
		}
		index = decoded++;
		auto &frame = reader.Frame();
		auto base = reader.Stream().time_base;
		auto origin = reader.Stream().start_time;
		if (origin == AV_NOPTS_VALUE) {
			origin = 0;
		}
		bool has_time = frame.pts != AV_NOPTS_VALUE && base.num > 0 && base.den > 0;
		long double time = has_time ? (static_cast<long double>(frame.pts) - origin) * base.num / base.den : 0;
		timestamp = has_time ? Value::DOUBLE(double(time)) : Value(LogicalType::DOUBLE);
		if (operation == VideoFrameOperation::FRAME_BY_INDEX) {
			return index == options.target_index;
		}
		if (!has_time && (options.start > 0 || options.has_end || options.has_interval)) {
			throw MediaFormatException("video time selection requires presentation timestamps");
		}
		if (has_time) {
			if (have_last_time && time < last_time) {
				next_sample = options.start;
			}
			last_time = time;
			have_last_time = true;
			if (double(time) < options.start || (options.has_end && double(time) > options.end)) {
				return false;
			}
		}
		if (options.has_key && ((frame.flags & AV_FRAME_FLAG_KEY) != 0) != options.key) {
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

static void WriteFrameRecord(Vector &result, idx_t row, const Value &file, uint64_t index, const Value &time,
                             MediaReader &reader) {
	auto &children = StructVector::GetEntries(result);
	auto &frame = reader.Frame();
	auto base = reader.Stream().time_base;
	children[0]->SetValue(row, file);
	children[1]->SetValue(row, Value::BIGINT(int64_t(index)));
	children[2]->SetValue(row, time);
	children[3]->SetValue(row, base.num > 0 ? Value::BIGINT(base.num) : Value(LogicalType::BIGINT));
	children[4]->SetValue(row, base.den > 0 ? Value::BIGINT(base.den) : Value(LogicalType::BIGINT));
	children[5]->SetValue(row, frame.pts != AV_NOPTS_VALUE ? Value::BIGINT(frame.pts) : Value(LogicalType::BIGINT));
	children[6]->SetValue(row,
	                      frame.pkt_dts != AV_NOPTS_VALUE ? Value::BIGINT(frame.pkt_dts) : Value(LogicalType::BIGINT));
	children[7]->SetValue(row, frame.duration > 0 ? Value::BIGINT(frame.duration) : Value(LogicalType::BIGINT));
	children[8]->SetValue(row, Value::BOOLEAN((frame.flags & AV_FRAME_FLAG_KEY) != 0));
	FlatVector::Validity(result).SetValid(row);
}

template <VideoFrameOperation OPERATION>
static void VideoFramesScalar(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	uint64_t batch_bytes = 0;
	auto &context = state.GetContext();
	for (idx_t row = 0; row < args.size(); row++) {
		MediaInterrupt(context);
		VideoFrameOptions options;
		if (!VideoFrameOptions::Read(args, row, OPERATION, options)) {
			result.SetValue(row, Value(result.GetType()));
			continue;
		}
		auto file = args.data[0].GetValue(row);
		VideoFrameOutputBudget budget(options, batch_bytes, file, OPERATION);
		if (OPERATION != VideoFrameOperation::FRAME_BY_INDEX) {
			FlatVector::GetData<list_entry_t>(result)[row] = list_entry_t(ListVector::GetListSize(result), 0);
			FlatVector::Validity(result).SetValid(row);
		}
		try {
			MediaReader reader(context, FileReference::FromValue(file, VideoFrameContract::Name(OPERATION)),
			                   AVMEDIA_TYPE_VIDEO, options.max_input_bytes, options.max_input_bytes * 4,
			                   options.max_pixels);
			ScalarFrameSelection selection(options.start);
			bool found = false;
			while (reader.NextFrame()) {
				uint64_t index;
				Value timestamp;
				if (!selection.Select(reader, options, OPERATION, index, timestamp)) {
					continue;
				}
				auto &frame = reader.Frame();
				auto width = options.width ? options.width : uint32_t(frame.width);
				auto height = options.height ? options.height : uint32_t(frame.height);
				auto bytes = budget.ClaimFrame(width, height);
				Vector *image = &result;
				idx_t target = row;
				if (OPERATION != VideoFrameOperation::FRAME_BY_INDEX) {
					target = ListVector::GetListSize(result);
					ListVector::Reserve(result, target + 1);
					ListVector::SetListSize(result, target + 1);
					image = &ListVector::GetEntry(result);
					if (OPERATION == VideoFrameOperation::FRAMES) {
						WriteFrameRecord(*image, target, file, index, timestamp, reader);
						image = StructVector::GetEntries(*image).back().get();
					}
				}
				MediaWriteImage(context, frame, "RGB", width, height, *image, target, bytes);
				found = true;
				if (OPERATION == VideoFrameOperation::FRAME_BY_INDEX) {
					break;
				}
				FlatVector::GetData<list_entry_t>(result)[row].length++;
			}
			reader.CheckIO();
			MediaInterrupt(context);
			if (OPERATION == VideoFrameOperation::FRAME_BY_INDEX && !found) {
				if (!options.null_on_error) {
					throw InvalidInputException("video frame idx %llu is out of range",
					                            static_cast<unsigned long long>(options.target_index));
				}
				result.SetValue(row, Value(result.GetType()));
			}
		} catch (const MediaFormatException &) {
			MediaInterrupt(context);
			if (!options.null_on_error) {
				throw;
			}
			// Retain the child high-water mark and its charge; a NULL parent never
			// exposes those partially written children to callers or Flight.
			result.SetValue(row, Value(result.GetType()));
		}
	}
}

template <VideoFrameOperation OPERATION>
static void RegisterFrameFunction(ExtensionLoader &loader) {
	ScalarFunction function(
	    string("native_") + VideoFrameContract::Name(OPERATION), VideoFrameContract::Arguments(),
	    VideoFrameContract::ResultType(OPERATION),
	    [](DataChunk &args, ExpressionState &state, Vector &output) {
		    try {
			    VideoFramesScalar<OPERATION>(args, state, output);
		    } catch (...) {
			    MediaInterrupt(state.GetContext());
			    throw;
		    }
	    },
	    VideoFrameContract::Bind);
	function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	function.SetStability(FunctionStability::VOLATILE);
	function.SetFallible();
	loader.RegisterFunction(std::move(function));
}
} // namespace

void RegisterMediaVideoFrameFunctions(ExtensionLoader &loader) {
	RegisterFrameFunction<VideoFrameOperation::FRAMES>(loader);
	RegisterFrameFunction<VideoFrameOperation::KEYFRAMES>(loader);
	RegisterFrameFunction<VideoFrameOperation::FRAME_BY_INDEX>(loader);
}
} // namespace duckdb
