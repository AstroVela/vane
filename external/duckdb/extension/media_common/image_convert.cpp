// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/common/types/image.hpp"
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

void MediaConvertVideoPixels(ClientContext &context, const AVFrame &frame, const string &mode, uint32_t width,
                             uint32_t height, data_ptr_t destination) {
	MediaInterrupt(context);
	if (mode != "RGB") {
		throw InternalException("video pixel conversion requires RGB output");
	}
	if (!width || !height || width > INT_MAX || height > INT_MAX || frame.width <= 0 || frame.height <= 0) {
		throw MediaFormatException("invalid decoded video dimensions");
	}
	// av_frame_get_buffer pads rows as well as strides. Bound that allocation
	// before invoking FFmpeg, including for very narrow or short RGB outputs.
	auto padded_width = (uint64_t(width) + 31) & ~uint64_t(31);
	auto padded_height = (uint64_t(height) + 31) & ~uint64_t(31);
	MediaProduct(padded_width * 3, padded_height, MEDIA_MAX_FRAME_BYTES - 4 * AV_INPUT_BUFFER_PADDING_SIZE,
	             "video conversion buffer bytes");
	auto size = av_image_get_buffer_size(AV_PIX_FMT_RGB24, int(width), int(height), 1);
	VideoCheck(size, "calculate video RGB buffer size");
	auto output = av_frame_alloc();
	auto converter = sws_alloc_context();
	if (!output || !converter) {
		av_frame_free(&output);
		sws_freeContext(converter);
		throw OutOfMemoryException("Cannot allocate video pixel converter");
	}
	try {
		// Keep the decoder-owned frame and its provenance unchanged. Match
		// _frame_to_image: preserve the YUV matrix/range and chroma location,
		// but do not request a transfer-function or color-primary conversion.
		AVFrame source = frame;
		source.color_trc = AVCOL_TRC_UNSPECIFIED;
		source.color_primaries = AVCOL_PRI_UNSPECIFIED;
		// PyAV copies frame properties before allocating the destination. In
		// particular, keep interlaced field layout and color side data intact.
		VideoCheck(av_frame_copy_props(output, &source), "copy video frame properties");
		output->format = AV_PIX_FMT_RGB24;
		output->width = int(width);
		output->height = int(height);
		VideoCheck(av_frame_get_buffer(output, 32), "allocate video RGB buffer");
		converter->flags = SWS_BILINEAR;
		converter->threads = 1;
		// An uninitialized context selects FFmpeg's frame-aware scaler, as
		// PyAV does. Initializing with sws_getContext instead selects the
		// legacy scaler and changes resized chroma pixels for the same flags.
		VideoCheck(sws_scale_frame(converter, output, &source), "convert video RGB pixels");
		MediaInterrupt(context);
		const uint8_t *planes[4] = {output->data[0], output->data[1], output->data[2], output->data[3]};
		VideoCheck(av_image_copy_to_buffer(destination, size, planes, output->linesize, AV_PIX_FMT_RGB24, int(width),
		                                   int(height), 1),
		           "copy video RGB pixels");
	} catch (...) {
		av_frame_free(&output);
		sws_freeContext(converter);
		throw;
	}
	av_frame_free(&output);
	sws_freeContext(converter);
}

uint64_t MediaWriteImage(ClientContext &context, const AVFrame &frame, const string &mode, uint32_t width,
                         uint32_t height, Vector &result, idx_t row, uint64_t remaining_bytes,
                         media_pixel_converter_t convert) {
	auto channels = ImageLogicalType::ChannelsForMode(mode);
	auto pixels = MediaProduct(width, height, MEDIA_MAX_PIXELS, "output pixels");
	auto size = MediaProduct(pixels, channels, MinValue<uint64_t>(remaining_bytes, string_t::MAX_STRING_SIZE),
	                         "output image bytes");
	auto pixels_out = ImageVector::Allocate(result, row, width, height, mode);
	convert(context, frame, mode, width, height, pixels_out);
	return size;
}
} // namespace duckdb
