// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "media_reader.hpp"
#include "duckdb/common/numeric_utils.hpp"
extern "C" {
#include <libswscale/swscale.h>
}

namespace duckdb {

void MediaConvertPixels(ClientContext &context, const AVFrame &frame, const string &mode, uint32_t width,
                        uint32_t height, data_ptr_t destination) {
	MediaInterrupt(context);
	AVPixelFormat pixel_format;
	if (mode == "L") {
		pixel_format = AV_PIX_FMT_GRAY8;
	} else if (mode == "LA") {
		pixel_format = AV_PIX_FMT_RGBA;
	} else if (mode == "RGB") {
		pixel_format = AV_PIX_FMT_RGB24;
	} else if (mode == "RGBA") {
		pixel_format = AV_PIX_FMT_RGBA;
	} else {
		throw InvalidInputException("native image mode must be L, LA, RGB, or RGBA");
	}
	if (!width || !height || width > INT_MAX || height > INT_MAX || frame.width <= 0 || frame.height <= 0) {
		throw MediaFormatException("invalid decoded image dimensions");
	}
	auto size = av_image_get_buffer_size(pixel_format, NumericCast<int>(width), NumericCast<int>(height), 1);
	if (size < 0) {
		throw OutOfRangeException("native output image buffer size is not representable");
	}
	auto padded_size = av_image_get_buffer_size(pixel_format, NumericCast<int>(width), NumericCast<int>(height), 32);
	if (padded_size < 0) {
		throw OutOfRangeException("native aligned output image size is not representable");
	}
	MediaProduct(1, uint64_t(padded_size) + 32, MEDIA_MAX_FRAME_BYTES, "pixel conversion buffer bytes");
	if (!sws_isSupportedInput(AVPixelFormat(frame.format)) || !sws_isSupportedOutput(pixel_format)) {
		throw MediaFormatException("unsupported native pixel format");
	}
	auto converter = sws_getContext(frame.width, frame.height, AVPixelFormat(frame.format), NumericCast<int>(width),
	                                NumericCast<int>(height), pixel_format, SWS_BILINEAR, nullptr, nullptr, nullptr);
	if (!converter) {
		throw OutOfMemoryException("Cannot allocate native pixel converter");
	}
	uint8_t *planes[4] = {};
	int strides[4] = {};
	try {
		// swscale SIMD requires padded, aligned planes. Never let it write past
		// an engine BLOB/ARRAY allocation, including for narrow images.
		MediaCheck(av_image_alloc(planes, strides, width, height, pixel_format, 32),
		           "allocate pixel conversion buffer");
		auto coefficients =
		    sws_getCoefficients(frame.colorspace == AVCOL_SPC_UNSPECIFIED ? SWS_CS_DEFAULT : frame.colorspace);
		MediaCheck(sws_setColorspaceDetails(converter, coefficients, frame.color_range == AVCOL_RANGE_JPEG,
		                                    coefficients, 1, 0, 1 << 16, 1 << 16),
		           "configure image colorspace");
		auto rows = sws_scale(converter, frame.data, frame.linesize, 0, frame.height, planes, strides);
		MediaInterrupt(context);
		if (rows != NumericCast<int>(height)) {
			throw MediaFormatException("pixel conversion returned an incomplete image");
		}
		if (mode == "LA") {
			for (uint32_t y = 0; y < height; y++) {
				MediaInterrupt(context);
				auto source = planes[0] + uint64_t(y) * strides[0];
				auto target = destination + uint64_t(y) * width * 2;
				for (uint32_t x = 0; x < width; x++) {
					target[2 * x] =
					    uint8_t((299 * source[4 * x] + 587 * source[4 * x + 1] + 114 * source[4 * x + 2] + 500) / 1000);
					target[2 * x + 1] = source[4 * x + 3];
				}
			}
		} else {
			const uint8_t *source[4] = {planes[0], planes[1], planes[2], planes[3]};
			MediaCheck(av_image_copy_to_buffer(destination, size, source, strides, pixel_format, width, height, 1),
			           "copy image pixels");
		}
	} catch (...) {
		av_freep(&planes[0]);
		sws_freeContext(converter);
		throw;
	}
	av_freep(&planes[0]);
	sws_freeContext(converter);
}

uint64_t MediaWriteImage(ClientContext &context, const AVFrame &frame, const string &mode, uint32_t width,
                         uint32_t height, Vector &result, idx_t row, uint64_t remaining_bytes) {
	auto channels = ImageLogicalType::ChannelsForMode(mode);
	auto pixels = MediaProduct(width, height, MEDIA_MAX_PIXELS, "output pixels");
	auto size = MediaProduct(pixels, channels, MinValue<uint64_t>(remaining_bytes, string_t::MAX_STRING_SIZE),
	                         "output image bytes");
	auto &children = StructVector::GetEntries(result);
	auto &data = *children[ImageLogicalType::DATA];
	auto bytes = StringVector::EmptyString(data, NumericCast<idx_t>(size));
	MediaConvertPixels(context, frame, mode, width, height, reinterpret_cast<data_ptr_t>(bytes.GetDataWriteable()));
	bytes.Finalize();
	FlatVector::GetData<string_t>(data)[row] = bytes;
	FlatVector::GetData<uint32_t>(*children[ImageLogicalType::WIDTH])[row] = width;
	FlatVector::GetData<uint32_t>(*children[ImageLogicalType::HEIGHT])[row] = height;
	FlatVector::GetData<uint8_t>(*children[ImageLogicalType::CHANNELS])[row] = channels;
	children[ImageLogicalType::MODE]->SetValue(row, Value(mode));
	FlatVector::SetNull(result, row, false);
	for (auto &child : children) {
		FlatVector::SetNull(*child, row, false);
	}
	return size;
}
} // namespace duckdb
