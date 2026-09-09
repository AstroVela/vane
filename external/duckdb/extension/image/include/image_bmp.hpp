// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "media_reader.hpp"

namespace duckdb {

//! BMP mode discovery reads only the bounded DIB header and color table.
struct ImageBMPHeader {
	uint32_t width, height;
	string mode;

	template <class READ>
	static ImageBMPHeader Read(READ read) {
		auto le16 = [](const string &data, idx_t at) {
			return uint32_t(uint8_t(data[at])) | (uint32_t(uint8_t(data[at + 1])) << 8);
		};
		auto le32 = [&](const string &data, idx_t at) {
			return le16(data, at) | (le16(data, at + 2) << 16);
		};
		auto size = le32(read(14, 4), 0);
		if (size != 12 && size != 40 && size != 52 && size != 56 && size != 64 && size != 108 && size != 124) {
			throw MediaFormatException("unsupported BMP header");
		}
		auto dib = read(14, size);
		auto width = size == 12 ? le16(dib, 4) : le32(dib, 4);
		auto signed_height = size == 12 ? int64_t(le16(dib, 6)) : int64_t(int32_t(le32(dib, 8)));
		auto height = uint32_t(signed_height < 0 ? -signed_height : signed_height);
		auto bits = le16(dib, size == 12 ? 10 : 14);
		auto planes = le16(dib, size == 12 ? 8 : 12);
		if (!width || width > INT32_MAX || !height || planes != 1 ||
		    (bits != 1 && bits != 4 && bits != 8 && bits != 16 && bits != 24 && bits != 32)) {
			throw MediaFormatException("invalid BMP dimensions, planes or pixel depth");
		}
		ImageBMPHeader result {width, height, "RGB"};
		if (bits <= 8) {
			auto colors = size == 12 ? uint32_t(0) : le32(dib, 32);
			colors = colors ? colors : uint32_t(1) << bits;
			if (colors > (uint32_t(1) << bits)) {
				throw MediaFormatException("invalid BMP color table size");
			}
			auto stride = size == 12 ? 3 : 4;
			auto palette = read(14 + size, colors * stride);
			bool gray = true;
			for (uint32_t i = 0; i < colors; i++) {
				auto expected = colors == 2 ? i * 255 : i;
				for (idx_t channel = 0; channel < 3; channel++) {
					gray &= uint8_t(palette[i * stride + channel]) == expected;
				}
			}
			result.mode = gray ? (colors == 2 ? "1" : "L") : "P";
		} else if (bits == 32 && size >= 56 && le32(dib, 16) == 3) {
			if (le32(dib, 52) || !(le32(dib, 40) | le32(dib, 44) | le32(dib, 48))) {
				result.mode = "RGBA";
			}
		}
		return result;
	}
};

} // namespace duckdb
