// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "media_reader.hpp"

namespace duckdb {

struct ImageGIFHeader {
	uint32_t width, height;

	template <class READ>
	static ImageGIFHeader Read(READ read) {
		auto screen = read(6, 7);
		auto byte = [&](idx_t at) {
			return uint32_t(uint8_t(screen[at]));
		};
		ImageGIFHeader header {byte(0) | (byte(1) << 8), byte(2) | (byte(3) << 8)};
		if (!header.width || !header.height) {
			throw MediaFormatException("invalid GIF logical screen dimensions");
		}
		if (byte(4) & 0x80) {
			// The declared global color table immediately follows the complete
			// screen descriptor; header probing never reads compressed pixels.
			auto colors = idx_t(1) << ((byte(4) & 7) + 1);
			read(13, colors * 3);
		}
		return header;
	}
};

} // namespace duckdb
