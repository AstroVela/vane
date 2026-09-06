// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "file_value.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/planner/expression.hpp"

#include <cmath>

namespace duckdb {

enum class VideoFrameOperation : uint8_t { FRAMES, KEYFRAMES, FRAME_BY_INDEX, SCAN_STATS };

//! Codec-independent contracts shared by the Python bridge and video extension.
struct VideoFrameContract {
	static constexpr uint64_t MIB = 1024 * 1024;
	static constexpr uint64_t MAX_BATCH_BYTES = 256 * MIB;
	static constexpr uint64_t MAX_PIXELS = 32 * MIB;
	static constexpr uint64_t MAX_INDEX_BYTES = 64 * MIB;

	static bool HasIndex(Vector &input, idx_t row) {
		UnifiedVectorFormat format;
		input.ToUnifiedFormat(row + 1, format);
		return format.validity.RowIsValid(format.sel->get_index(row));
	}

	//! Reject oversized serialized indexes before GetValue could copy the BLOB.
	static Value ReadIndex(Vector &input, idx_t row) {
		UnifiedVectorFormat format;
		input.ToUnifiedFormat(row + 1, format);
		auto selected = format.sel->get_index(row);
		if (!format.validity.RowIsValid(selected)) {
			return Value(LogicalType::BLOB);
		}
		auto value = UnifiedVectorFormat::GetData<string_t>(format)[selected];
		if (value.GetSize() > MAX_INDEX_BYTES) {
			throw OutOfRangeException("video index exceeds 64 MiB");
		}
		return Value::BLOB(const_data_ptr_cast(value.GetData()), value.GetSize());
	}

	static const char *Name(VideoFrameOperation operation) {
		switch (operation) {
		case VideoFrameOperation::FRAMES:
			return "video_frames";
		case VideoFrameOperation::KEYFRAMES:
			return "video_keyframes";
		case VideoFrameOperation::SCAN_STATS:
			return "video_scan_stats";
		default:
			return "get_video_frame_by_idx";
		}
	}

	static LogicalType FrameType() {
		child_list_t<LogicalType> fields;
		fields.emplace_back("file", FileLogicalType::Create(FileMediaType::VIDEO));
		fields.emplace_back("frame_index", LogicalType::BIGINT);
		fields.emplace_back("frame_time", LogicalType::DOUBLE);
		fields.emplace_back("frame_time_base_numerator", LogicalType::BIGINT);
		fields.emplace_back("frame_time_base_denominator", LogicalType::BIGINT);
		fields.emplace_back("frame_pts", LogicalType::BIGINT);
		fields.emplace_back("frame_dts", LogicalType::BIGINT);
		fields.emplace_back("frame_duration", LogicalType::BIGINT);
		fields.emplace_back("is_key_frame", LogicalType::BOOLEAN);
		fields.emplace_back("data", ImageLogicalType::Create());
		return LogicalType::STRUCT(std::move(fields));
	}

	static LogicalType ResultType(VideoFrameOperation operation) {
		if (operation == VideoFrameOperation::SCAN_STATS) {
			return LogicalType::STRUCT({{"bytes_read", LogicalType::UBIGINT},
			                            {"decoded_frames", LogicalType::UBIGINT},
			                            {"seeks", LogicalType::UBIGINT},
			                            {"selected_frames", LogicalType::UBIGINT}});
		}
		if (operation == VideoFrameOperation::FRAME_BY_INDEX) {
			return ImageLogicalType::Create();
		}
		return LogicalType::LIST(operation == VideoFrameOperation::FRAMES ? FrameType() : ImageLogicalType::Create());
	}

	// file, start, end, width, height, key, interval, on_error, input/decoded/pixel
	// limits, output byte/frame limits, exact frame index, optional seek index.
	// Public SQL macros fill defaults.
	static vector<LogicalType> Arguments() {
		return {LogicalType::ANY,    LogicalType::DOUBLE,  LogicalType::DOUBLE, LogicalType::BIGINT,
		        LogicalType::BIGINT, LogicalType::BOOLEAN, LogicalType::DOUBLE, LogicalType::VARCHAR,
		        LogicalType::BIGINT, LogicalType::BIGINT,  LogicalType::BIGINT, LogicalType::BIGINT,
		        LogicalType::BIGINT, LogicalType::BIGINT,  LogicalType::BLOB};
	}

	static unique_ptr<FunctionData> Bind(ClientContext &, ScalarFunction &function,
	                                     vector<unique_ptr<Expression>> &arguments) {
		auto type = arguments[0]->return_type;
		if (type.id() == LogicalTypeId::SQLNULL || type.id() == LogicalTypeId::UNKNOWN) {
			type = FileLogicalType::Create(FileMediaType::VIDEO);
		}
		if (!FileLogicalType::IsFile(type) || FileLogicalType::GetMediaType(type) != FileMediaType::VIDEO) {
			throw BinderException("%s requires VIDEOFILE, not %s", function.name, type);
		}
		function.arguments[0] = std::move(type);
		return nullptr;
	}
};

struct VideoFrameOptions {
	double start = 0;
	double end = 0;
	double interval = 0;
	bool has_end = false;
	bool has_interval = false;
	bool has_key = false;
	bool key = false;
	bool null_on_error = false;
	uint32_t width = 0;
	uint32_t height = 0;
	uint64_t max_input_bytes = 0;
	uint64_t max_decoded_frames = 0;
	uint64_t max_pixels = 0;
	uint64_t max_output_bytes = 0;
	uint64_t max_output_frames = 0;
	uint64_t target_index = 0;
	bool has_target_index = false;

	static bool Read(DataChunk &args, idx_t row, VideoFrameOperation operation, VideoFrameOptions &result) {
		if (args.data[0].GetValue(row).IsNull()) {
			return false;
		}
		for (auto index : {1, 7, 8, 9, 10, 11, 12}) {
			if (args.data[index].GetValue(row).IsNull()) {
				return false;
			}
		}
		if (operation == VideoFrameOperation::FRAME_BY_INDEX && args.data[13].GetValue(row).IsNull()) {
			return false;
		}
		auto positive = [&](idx_t index, uint64_t maximum, const char *name) {
			auto value = args.data[index].GetValue(row).GetValue<int64_t>();
			if (value <= 0 || uint64_t(value) > maximum) {
				throw InvalidInputException("%s must be between 1 and %llu", name,
				                            static_cast<unsigned long long>(maximum));
			}
			return uint64_t(value);
		};
		result.max_input_bytes = positive(8, 16 * 1024 * VideoFrameContract::MIB, "max_input_bytes");
		result.max_decoded_frames = positive(9, 100000000, "max_decoded_frames");
		result.max_pixels = positive(10, VideoFrameContract::MAX_PIXELS, "max_pixels");
		result.max_output_bytes = positive(11, VideoFrameContract::MAX_BATCH_BYTES, "max_output_bytes");
		result.max_output_frames = positive(12, 100000, "max_output_frames");
		if (!args.data[3].GetValue(row).IsNull()) {
			result.width = uint32_t(positive(3, 100000, "width"));
		}
		if (!args.data[4].GetValue(row).IsNull()) {
			result.height = uint32_t(positive(4, 100000, "height"));
		}
		if ((result.width == 0) != (result.height == 0)) {
			throw InvalidInputException("width and height must be provided together");
		}
		if (uint64_t(result.width) * result.height > result.max_pixels) {
			throw OutOfRangeException("video output dimensions exceed max_pixels");
		}
		auto time = [&](idx_t index, const char *name) {
			auto value = args.data[index].GetValue(row).GetValue<double>();
			if (!std::isfinite(value) || value < 0) {
				throw InvalidInputException("%s must be finite and nonnegative", name);
			}
			return value;
		};
		result.start = time(1, "start_time");
		result.has_end = !args.data[2].GetValue(row).IsNull();
		if (result.has_end) {
			result.end = time(2, "end_time");
			if (result.end < result.start) {
				throw InvalidInputException("end_time must be at least start_time");
			}
		}
		result.has_key = operation == VideoFrameOperation::KEYFRAMES || !args.data[5].GetValue(row).IsNull();
		result.key = operation == VideoFrameOperation::KEYFRAMES ||
		             (result.has_key && args.data[5].GetValue(row).GetValue<bool>());
		result.has_interval = !args.data[6].GetValue(row).IsNull();
		if (result.has_interval) {
			result.interval = time(6, "sample_interval_seconds");
			if (result.interval == 0) {
				throw InvalidInputException("sample_interval_seconds must be positive");
			}
		}
		auto policy = args.data[7].GetValue(row).GetValue<string>();
		if (policy != "raise" && policy != "null") {
			throw InvalidInputException("on_error must be 'raise' or 'null'");
		}
		result.null_on_error = policy == "null";
		result.has_target_index =
		    operation == VideoFrameOperation::FRAME_BY_INDEX ||
		    (operation == VideoFrameOperation::SCAN_STATS && !args.data[13].GetValue(row).IsNull());
		if (result.has_target_index) {
			auto index = args.data[13].GetValue(row).GetValue<int64_t>();
			if (index < 0) {
				throw InvalidInputException("video frame idx must be nonnegative");
			}
			result.target_index = uint64_t(index);
			if (!VideoFrameContract::HasIndex(args.data[14], row) && result.target_index >= result.max_decoded_frames) {
				throw OutOfRangeException("video frame idx exceeds max_decoded_frames");
			}
		}
		return true;
	}
};

//! Count retained vector allocations, including rows later nulled by on_error.
struct VideoFrameOutputBudget {
	const VideoFrameOptions &options;
	uint64_t &batch_bytes;
	uint64_t row_bytes = 0;
	uint64_t frame_count = 0;
	uint64_t metadata_bytes = 128;

	VideoFrameOutputBudget(const VideoFrameOptions &options_p, uint64_t &batch_bytes_p, const Value &file,
	                       VideoFrameOperation operation)
	    : options(options_p), batch_bytes(batch_bytes_p) {
		if (operation == VideoFrameOperation::FRAMES) {
			auto reference = FileReference::FromValue(file, "video_frames");
			metadata_bytes = 512 + reference.url.size() + reference.content_type.size() + reference.checksum.size();
		}
		Charge(64);
	}

	void Charge(uint64_t bytes) {
		if (bytes > options.max_output_bytes - row_bytes) {
			throw OutOfRangeException("video scalar result exceeds max_output_bytes; use read_video_frames to stream");
		}
		if (bytes > VideoFrameContract::MAX_BATCH_BYTES - batch_bytes) {
			throw OutOfRangeException("video scalar batch exceeds 256 MiB; use read_video_frames to stream");
		}
		row_bytes += bytes;
		batch_bytes += bytes;
	}

	uint64_t ClaimFrame(uint32_t width, uint32_t height) {
		if (width == 0 || height == 0 || uint64_t(width) * height > options.max_pixels) {
			throw OutOfRangeException("video output dimensions exceed max_pixels");
		}
		if (frame_count >= options.max_output_frames) {
			throw OutOfRangeException("video scalar result exceeds max_output_frames; use read_video_frames to stream");
		}
		auto pixels = uint64_t(width) * height * 3;
		Charge(pixels + metadata_bytes);
		frame_count++;
		return pixels;
	}
};

} // namespace duckdb
