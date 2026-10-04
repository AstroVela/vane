// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "image_codec.hpp"

#include <webp/decode.h>
#include <webp/demux.h>

namespace duckdb {

struct ImageWebPHeader {
	uint32_t width, height;
	bool alpha;

	template <class READ>
	static ImageWebPHeader Read(READ read, idx_t size) {
		auto header = read(0, 20);
		auto number = [](const string &s, idx_t offset, idx_t count) {
			uint32_t value = 0;
			for (idx_t i = 0; i < count; i++) {
				value |= uint32_t(uint8_t(s[offset + i])) << (8 * i);
			}
			return value;
		};
		auto riff = number(header, 4, 4), chunk = number(header, 16, 4);
		if (header.substr(0, 4) != "RIFF" || header.substr(8, 4) != "WEBP" || riff < 12 || uint64_t(riff) + 8 > size ||
		    uint64_t(chunk) + (chunk & 1) > riff - 12) {
			throw MediaFormatException("invalid WebP RIFF header");
		}
		auto kind = header.substr(12, 4);
		if (kind == "VP8X") {
			if (chunk != 10) {
				throw MediaFormatException("invalid WebP extended header length");
			}
			auto bytes = read(20, 10);
			if ((uint8_t(bytes[0]) & 0xc1) || number(bytes, 1, 3)) {
				throw MediaFormatException("invalid WebP extended header flags");
			}
			return {number(bytes, 4, 3) + 1, number(bytes, 7, 3) + 1, bool(uint8_t(bytes[0]) & 0x10)};
		}
		if (kind == "VP8L" && chunk >= 5) {
			auto bytes = read(20, 5);
			auto bits = number(bytes, 1, 4);
			if (uint8_t(bytes[0]) != 0x2f || (bits >> 29)) {
				throw MediaFormatException("invalid lossless WebP header");
			}
			return {(bits & 0x3fff) + 1, ((bits >> 14) & 0x3fff) + 1, bool((bits >> 28) & 1)};
		}
		if (kind == "VP8 " && chunk >= 10) {
			auto bytes = read(20, 10);
			if ((uint8_t(bytes[0]) & 1) || bytes.substr(3, 3) != string("\x9d\x01\x2a", 3)) {
				throw MediaFormatException("invalid lossy WebP frame header");
			}
			auto width = number(bytes, 6, 2) & 0x3fff, height = number(bytes, 8, 2) & 0x3fff;
			if (!width || !height) {
				throw MediaFormatException("invalid WebP dimensions");
			}
			return {width, height, false};
		}
		throw MediaFormatException("unsupported WebP header");
	}
};

class ImageWebPDecoder {
public:
	~ImageWebPDecoder() {
		WebPIDelete(decoder);
		WebPDemuxReleaseIterator(&frame);
		WebPDemuxDelete(demux);
		WebPFreeDecBuffer(&config.output);
	}
	DecodedImagePixels Decode(ClientContext &context, const_data_ptr_t data, idx_t size, idx_t max_pixels,
	                          idx_t max_bytes, uint16_t output_channels, idx_t output_width, idx_t storage_width,
	                          idx_t remaining) {
		auto header = ImageWebPHeader::Read(
		    [&](idx_t offset, idx_t count) {
			    if (offset > size || count > size - offset) {
				    throw MediaFormatException("truncated WebP header");
			    }
			    return string(const_char_ptr_cast(data + offset), count);
		    },
		    size);
		auto channels = uint16_t(header.alpha ? 4 : 3);
		MediaProduct(header.width, header.height, max_pixels, "decoded image pixels");
		ImageCodecContract::CheckDecodedBytes(header.width, header.height, 4,
		                                      output_channels ? output_channels : channels,
		                                      output_width ? output_width : 1, storage_width, max_bytes);
		ImageOperatorContract::CheckSize(header.width, header.height, output_channels ? output_channels : channels,
		                                 remaining, storage_width);
		WebPData input {data, size};
		demux = WebPDemux(&input);
		if (!demux || !WebPDemuxGetFrame(demux, 1, &frame) || !frame.complete) {
			throw MediaFormatException("WebP contains no complete first frame");
		}
		if (WebPDemuxGetI(demux, WEBP_FF_CANVAS_WIDTH) != header.width ||
		    WebPDemuxGetI(demux, WEBP_FF_CANVAS_HEIGHT) != header.height || frame.x_offset < 0 || frame.y_offset < 0 ||
		    frame.width <= 0 || frame.height <= 0 || uint64_t(frame.x_offset) + frame.width > header.width ||
		    uint64_t(frame.y_offset) + frame.height > header.height) {
			throw MediaFormatException("invalid WebP frame rectangle");
		}
		if (!WebPInitDecoderConfig(&config)) {
			throw InvalidInputException("WebP decoder ABI mismatch");
		}
		Check(WebPGetFeatures(frame.fragment.bytes, frame.fragment.size, &config.input));
		if (config.input.width != frame.width || config.input.height != frame.height) {
			throw MediaFormatException("WebP frame dimensions differ from its header");
		}
		ImageLayout layout {header.width, header.height, channels, uint8_t(channels)};
		DecodedImagePixels result {layout, string(layout.Bytes(), '\0')};
		auto offset = (idx_t(frame.y_offset) * header.width + frame.x_offset) * channels;
		config.output.colorspace = header.alpha ? MODE_RGBA : MODE_RGB;
		config.output.is_external_memory = 1;
		config.output.u.RGBA.rgba = data_ptr_cast(&result.data[0]) + offset;
		config.output.u.RGBA.stride = int(header.width * channels);
		config.output.u.RGBA.size = result.data.size() - offset;
		config.options.use_threads = 0;
		decoder = WebPIDecode(nullptr, 0, &config);
		if (!decoder) {
			throw OutOfMemoryException("Cannot allocate WebP decoder");
		}
		auto status = VP8_STATUS_SUSPENDED;
		for (idx_t begin = 0; begin < frame.fragment.size && status == VP8_STATUS_SUSPENDED;) {
			MediaInterrupt(context);
			auto count = MinValue<idx_t>(65536, frame.fragment.size - begin);
			status = WebPIAppend(decoder, frame.fragment.bytes + begin, count);
			begin += count;
		}
		Check(status);
		MediaInterrupt(context);
		return result;
	}

private:
	static void Check(VP8StatusCode status) {
		if (status == VP8_STATUS_OUT_OF_MEMORY) {
			throw OutOfMemoryException("WebP allocation failed");
		}
		if (status != VP8_STATUS_OK) {
			throw MediaFormatException("invalid or truncated WebP pixels");
		}
	}
	WebPDemuxer *demux = nullptr;
	WebPIterator frame {};
	WebPDecoderConfig config {};
	WebPIDecoder *decoder = nullptr;
};

} // namespace duckdb
