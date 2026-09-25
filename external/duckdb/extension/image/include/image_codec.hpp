// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "image_codec_contract.hpp"
#include "media_reader.hpp"

namespace duckdb {

struct DecodedImagePixels {
	ImageLayout layout;
	string data;
};

struct NativeImageCodec {
	static DecodedImagePixels Decode(ClientContext &context, const_data_ptr_t data, idx_t size,
	                                 const LogicalType &output_type, const string &output_mode, idx_t remaining,
	                                 idx_t max_pixels = ImageOperatorContract::MAX_PIXELS,
	                                 idx_t max_bytes = MEDIA_MAX_FRAME_BYTES);
	static idx_t Write(ClientContext &context, const DecodedImagePixels &image, const string &mode, Vector &result,
	                   idx_t row, idx_t remaining);
	static string Encode(ClientContext &context, const ImagePixelView &image, const string &format, idx_t limit);
	static ImageLayout TIFFMetadata(ClientContext &context, ResolvedFile &file, const string &prefix, idx_t budget,
	                                idx_t max_pixels);
};

void RegisterImagePixelFunctions(ExtensionLoader &loader);
void RegisterImageComputeFunctions(ExtensionLoader &loader);

} // namespace duckdb
