// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "video_index.hpp"
#include "video_frame_contract.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include <limits>

namespace duckdb {
namespace {

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
		if (OPERATION != VideoFrameOperation::FRAME_BY_INDEX && OPERATION != VideoFrameOperation::SCAN_STATS) {
			FlatVector::GetData<list_entry_t>(result)[row] = list_entry_t(ListVector::GetListSize(result), 0);
			FlatVector::Validity(result).SetValid(row);
		}
		try {
			VideoFrameCursor cursor(context, file, options, OPERATION,
			                        VideoFrameContract::ReadIndex(args.data[14], row));
			auto &reader = cursor.Reader();
			bool found = false;
			while (cursor.Next()) {
				if (OPERATION == VideoFrameOperation::SCAN_STATS) {
					continue;
				}
				auto index = cursor.FrameIndex();
				auto timestamp = cursor.FrameTime();
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
			if (OPERATION == VideoFrameOperation::SCAN_STATS) {
				result.SetValue(row, cursor.Statistics());
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
	RegisterVideoIndexFunctions(loader);
	RegisterFrameFunction<VideoFrameOperation::SCAN_STATS>(loader);
	RegisterFrameFunction<VideoFrameOperation::FRAMES>(loader);
	RegisterFrameFunction<VideoFrameOperation::KEYFRAMES>(loader);
	RegisterFrameFunction<VideoFrameOperation::FRAME_BY_INDEX>(loader);
}
} // namespace duckdb
