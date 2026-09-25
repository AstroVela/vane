// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "image_codec.hpp"

#include <csetjmp>
#include <cstdio>
#include <jpeglib.h>
#include <jerror.h>

namespace duckdb {

//! Match Pillow's full-size libjpeg decode: accurate integer IDCT, fancy
//! chroma upsampling, and the inverted CMYK convention used by its JPEG loader.
//! State that survives a C error jump belongs to this heap-allocated object.
class ImageJPEGDecoder {
public:
	ImageJPEGDecoder(ClientContext &context) : context(context) {
		codec.err = jpeg_std_error(&manager);
		manager.error_exit = Error;
		manager.output_message = Quiet;
		codec.client_data = this;
		source.init_source = Noop;
		source.fill_input_buffer = EndOfInput;
		source.skip_input_data = Skip;
		source.resync_to_restart = jpeg_resync_to_restart;
		source.term_source = Noop;
	}
	~ImageJPEGDecoder() {
		if (codec.mem) {
			jpeg_destroy_decompress(&codec);
		}
	}
	DecodedImagePixels Decode(const_data_ptr_t data, idx_t size, idx_t max_pixels, idx_t max_bytes,
	                          uint16_t output_channels, idx_t output_width, idx_t storage_width, idx_t remaining) {
		if (setjmp(jump)) {
			if (manager.msg_code == JERR_OUT_OF_MEMORY) {
				throw OutOfMemoryException("JPEG allocation failed: %s", message);
			}
			if (manager.msg_code == JERR_NO_BACKING_STORE) {
				throw OutOfRangeException("JPEG decoder exceeded its memory limit");
			}
			if (manager.msg_code == JERR_TFILE_CREATE || manager.msg_code == JERR_TFILE_READ ||
			    manager.msg_code == JERR_TFILE_SEEK || manager.msg_code == JERR_TFILE_WRITE) {
				throw IOException("JPEG backing store failed: %s", message);
			}
			throw MediaFormatException(string("JPEG decoding failed: ") + message);
		}
		jpeg_create_decompress(&codec);
		source.next_input_byte = data;
		source.bytes_in_buffer = size;
		codec.src = &source;
		jpeg_read_header(&codec, TRUE);
		if (!codec.image_width || !codec.image_height || codec.data_precision != 8 ||
		    (codec.num_components != 1 && codec.num_components != 3 && codec.num_components != 4)) {
			throw MediaFormatException("unsupported JPEG pixel layout");
		}
		MediaProduct(codec.image_width, codec.image_height, max_pixels, "decoded image pixels");
		cmyk = codec.num_components == 4;
		codec.out_color_space = cmyk ? JCS_CMYK : codec.num_components == 1 ? JCS_GRAYSCALE : JCS_RGB;
		codec.dct_method = JDCT_ISLOW;
		codec.do_fancy_upsampling = TRUE;
		codec.mem->max_memory_to_use = long(max_bytes);
		layout = {codec.image_width, codec.image_height, uint16_t(codec.num_components == 1 ? 1 : 3),
		          uint8_t(codec.num_components == 1 ? 1 : 3)};
		ImageCodecContract::CheckDecodedBytes(layout.width, layout.height, 4,
		                                      output_channels ? output_channels : layout.channels,
		                                      output_width ? output_width : 1, storage_width, max_bytes);
		ImageOperatorContract::CheckSize(layout.width, layout.height,
		                                 output_channels ? output_channels : layout.channels, remaining, storage_width);
		pixels.resize(layout.Bytes());
		if (cmyk) {
			row_buffer.resize(idx_t(layout.width) * 4);
		}
		MediaInterrupt(context);
		jpeg_start_decompress(&codec);
		while (codec.output_scanline < codec.output_height) {
			MediaInterrupt(context);
			auto target = data_ptr_cast(&pixels[0]) + idx_t(codec.output_scanline) * layout.width * layout.channels;
			JSAMPROW row = cmyk ? data_ptr_cast(&row_buffer[0]) : target;
			jpeg_read_scanlines(&codec, &row, 1);
			if (cmyk) {
				for (idx_t x = 0; x < layout.width; x++) {
					for (idx_t channel = 0; channel < 3; channel++) {
						target[x * 3 + channel] =
						    uint8_t((uint32_t(row[x * 4 + channel]) * row[x * 4 + 3] + 127) / 255);
					}
				}
			}
		}
		jpeg_finish_decompress(&codec);
		MediaInterrupt(context);
		return {layout, std::move(pixels)};
	}

private:
	static void Error(j_common_ptr codec) {
		auto &self = *static_cast<ImageJPEGDecoder *>(codec->client_data);
		codec->err->format_message(codec, self.message);
		std::longjmp(self.jump, 1);
	}
	static void Quiet(j_common_ptr) {
	}
	static void Noop(j_decompress_ptr) {
	}
	static boolean EndOfInput(j_decompress_ptr codec) {
		codec->err->msg_code = JERR_INPUT_EOF;
		Error(j_common_ptr(codec));
		return FALSE;
	}
	static void Skip(j_decompress_ptr codec, long count) {
		if (count > 0) {
			if (uint64_t(count) > codec->src->bytes_in_buffer) {
				EndOfInput(codec);
			}
			codec->src->next_input_byte += count;
			codec->src->bytes_in_buffer -= count;
		}
	}
	ClientContext &context;
	jpeg_decompress_struct codec {};
	jpeg_error_mgr manager {};
	jpeg_source_mgr source {};
	std::jmp_buf jump;
	char message[JMSG_LENGTH_MAX] {};
	ImageLayout layout {};
	bool cmyk = false;
	string pixels, row_buffer;
};

} // namespace duckdb
