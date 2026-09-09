// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "image_codec.hpp"
#include "image_bmp.hpp"
#include "image_transform.hpp"

#include <csetjmp>
#include <cstdarg>
#include <cstdio>
#include <exception>
#include <jpeglib.h>
#include <tiffio.h>

// jerror's enum depends on configuration macros defined by jpeglib.
#include <jerror.h>

extern "C" {
#include <libswscale/swscale.h>
}

namespace duckdb {
namespace {

static uint8_t DecodedMode(AVPixelFormat format) {
	auto descriptor = av_pix_fmt_desc_get(format);
	if (!descriptor) {
		throw MediaFormatException("unsupported decoded pixel format");
	}
	bool palette = descriptor->flags & AV_PIX_FMT_FLAG_PAL;
	bool alpha = palette || (descriptor->flags & AV_PIX_FMT_FLAG_ALPHA);
	bool gray = !palette && !(descriptor->flags & AV_PIX_FMT_FLAG_RGB) && descriptor->nb_components <= 2;
	bool wide = false;
	for (idx_t i = 0; i < descriptor->nb_components; i++) {
		wide |= descriptor->comp[i].depth > 8;
	}
	return uint8_t((gray ? (alpha ? 2 : 1) : (alpha ? 4 : 3)) + (wide ? 4 : 0));
}

//! Library callbacks only see this bounded byte source/sink. A TIFF library
//! never opens paths, resolves URLs, or handles credentials.
class TIFFBytes {
public:
	TIFFBytes(ClientContext &context, const_data_ptr_t input, idx_t size, idx_t limit, bool writing = false,
	          ResolvedFile *file = nullptr, idx_t read_budget = ImageOperatorContract::MAX_BYTES)
	    : context(context), input(input), size(size), limit(limit), writing(writing), file(file),
	      read_budget(read_budget), deadline(std::chrono::steady_clock::now() + std::chrono::seconds(30)) {
		auto options = TIFFOpenOptionsAlloc();
		if (!options) {
			throw OutOfMemoryException("Cannot allocate TIFF options");
		}
		TIFFOpenOptionsSetMaxSingleMemAlloc(options, tmsize_t(ImageOperatorContract::MAX_BYTES));
		TIFFOpenOptionsSetMaxCumulatedMemAlloc(options, tmsize_t(ImageOperatorContract::MAX_BYTES));
		TIFFOpenOptionsSetErrorHandlerExtR(options, Error, this);
		TIFFOpenOptionsSetWarningHandlerExtR(options, Warning, this);
		handle = TIFFClientOpenExt("Vane Image bytes", writing ? "wl" : "rmO", this, Read, Write, Seek, Close, Size,
		                           Map, Unmap, options);
		TIFFOpenOptionsFree(options);
		try {
			Check(handle != nullptr);
		} catch (...) {
			if (handle) {
				TIFFClose(handle);
				handle = nullptr;
			}
			throw;
		}
	}
	~TIFFBytes() {
		if (handle) {
			TIFFClose(handle);
		}
	}
	TIFFBytes(const TIFFBytes &) = delete;
	TIFFBytes &operator=(const TIFFBytes &) = delete;

	void Check(bool success = true) {
		MediaInterrupt(context);
		if (error) {
			std::rethrow_exception(error);
		}
		if (std::chrono::steady_clock::now() > deadline) {
			throw OutOfRangeException("TIFF operation exceeded its time budget");
		}
		if (!success || message[0]) {
			if (writing) {
				throw IOException("TIFF encoding failed: %s", message);
			}
			throw MediaFormatException(string("TIFF decoding failed: ") + message);
		}
	}
	string Finish() {
		Check(TIFFWriteDirectory(handle) == 1);
		TIFFClose(handle);
		handle = nullptr;
		Check();
		return std::move(output);
	}

	TIFF *handle = nullptr;

private:
	static int Error(TIFF *, void *opaque, const char *, const char *format, va_list args) noexcept {
		auto &self = *static_cast<TIFFBytes *>(opaque);
		vsnprintf(self.message, sizeof(self.message), format, args);
		// libtiff reports allocation failures through its error callback rather
		// than a distinct return code. Preserve them as resource errors.
		if (!self.error && (strstr(self.message, "alloc") || strstr(self.message, "memory") ||
		                    strstr(self.message, "Memory") || strstr(self.message, "No space"))) {
			try {
				throw OutOfMemoryException("TIFF allocation failed: %s", self.message);
			} catch (...) {
				self.error = std::current_exception();
			}
		}
		return 1;
	}
	static int Warning(TIFF *, void *, const char *, const char *, va_list) noexcept {
		return 1;
	}
	static tmsize_t Read(thandle_t opaque, void *target, tmsize_t requested) noexcept {
		auto &self = *static_cast<TIFFBytes *>(opaque);
		try {
			self.Check();
			if (requested < 0) {
				throw MediaFormatException("negative TIFF read size");
			}
			auto length = self.writing ? self.output.size() : self.size;
			auto count = self.position > length ? idx_t(0) : MinValue(idx_t(requested), length - self.position);
			if (count > self.read_budget - self.read_bytes) {
				throw OutOfRangeException("TIFF operation exceeded its read byte budget");
			}
			if (count) {
				if (self.file) {
					self.file->ReadExact(data_ptr_cast(target), count, self.position);
				} else {
					memcpy(target,
					       (self.writing ? const_data_ptr_cast(self.output.data()) : self.input) + self.position,
					       count);
				}
			}
			self.read_bytes += count;
			self.position += count;
			return tmsize_t(count);
		} catch (...) {
			self.error = std::current_exception();
			return -1;
		}
	}
	static tmsize_t Write(thandle_t opaque, void *data, tmsize_t requested) noexcept {
		auto &self = *static_cast<TIFFBytes *>(opaque);
		try {
			self.Check();
			if (!self.writing || requested < 0 || self.position > self.limit ||
			    idx_t(requested) > self.limit - self.position) {
				throw OutOfRangeException("TIFF encoding exceeds its output byte limit");
			}
			auto end = self.position + idx_t(requested);
			if (end > self.output.size()) {
				self.output.resize(end);
			}
			if (requested) {
				memcpy(&self.output[0] + self.position, data, idx_t(requested));
			}
			self.position = end;
			return requested;
		} catch (...) {
			self.error = std::current_exception();
			return -1;
		}
	}
	static toff_t Seek(thandle_t opaque, toff_t offset, int whence) noexcept {
		auto &self = *static_cast<TIFFBytes *>(opaque);
		try {
			self.Check();
			auto length = self.writing ? self.output.size() : self.size;
			auto maximum = self.writing ? self.limit : length;
			auto base = whence == SEEK_SET ? idx_t(0) : whence == SEEK_CUR ? self.position : length;
			if (whence != SEEK_SET && whence != SEEK_CUR && whence != SEEK_END) {
				throw MediaFormatException("invalid TIFF seek origin");
			}
			// TIFF uses an unsigned offset type for signed relative offsets.
			if (whence != SEEK_SET && offset > uint64_t(INT64_MAX)) {
				auto distance = uint64_t(0) - offset;
				if (distance > base) {
					throw MediaFormatException("TIFF seek precedes the input");
				}
				self.position = base - distance;
			} else {
				if (base > maximum || offset > maximum - base) {
					if (self.writing) {
						throw OutOfRangeException("TIFF encoding seek exceeds its output byte limit");
					}
					throw MediaFormatException("TIFF seek exceeds the encoded input");
				}
				self.position = base + offset;
			}
			return self.position;
		} catch (...) {
			self.error = std::current_exception();
			return toff_t(-1);
		}
	}
	static int Close(thandle_t) {
		return 0;
	}
	static toff_t Size(thandle_t opaque) {
		auto &self = *static_cast<TIFFBytes *>(opaque);
		return self.writing ? self.output.size() : self.size;
	}
	static int Map(thandle_t, void **, toff_t *) {
		return 0;
	}
	static void Unmap(thandle_t, void *, toff_t) {
	}

	ClientContext &context;
	const_data_ptr_t input;
	idx_t size, limit;
	bool writing;
	ResolvedFile *file;
	idx_t read_budget, read_bytes = 0, position = 0;
	std::chrono::steady_clock::time_point deadline;
	string output;
	char message[512] = {};
	std::exception_ptr error;
};

struct TIFFLayout {
	ImageLayout image;
	uint16_t planar, photometric;
};

static TIFFLayout ReadTIFFLayout(TIFFBytes &bytes, idx_t max_pixels, idx_t max_bytes) {
	auto tif = bytes.handle;
	uint32_t width = 0, height = 0, depth = 1;
	uint16_t samples = 1, bits = 1, sample_format = SAMPLEFORMAT_UINT, photo = 0, planar = 1, orientation = 1;
	bytes.Check(TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width) && TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height));
	TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samples);
	TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &bits);
	TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &sample_format);
	TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &planar);
	TIFFGetFieldDefaulted(tif, TIFFTAG_ORIENTATION, &orientation);
	TIFFGetFieldDefaulted(tif, TIFFTAG_IMAGEDEPTH, &depth);
	bytes.Check(TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photo) == 1);
	if (!width || !height || depth != 1 || samples < 1 || samples > 4 || orientation != ORIENTATION_TOPLEFT ||
	    (planar != PLANARCONFIG_CONTIG && planar != PLANARCONFIG_SEPARATE) || TIFFIsTiled(tif) ||
	    (photo != PHOTOMETRIC_MINISBLACK && photo != PHOTOMETRIC_MINISWHITE && photo != PHOTOMETRIC_RGB) ||
	    (photo == PHOTOMETRIC_RGB ? samples < 3 : samples > 2)) {
		throw MediaFormatException("unsupported TIFF layout, photometric interpretation, orientation or tiling");
	}
	uint8_t mode;
	if (sample_format == SAMPLEFORMAT_UINT && (bits == 8 || bits == 16)) {
		mode = uint8_t(samples + (bits == 16 ? 4 : 0));
	} else if (sample_format == SAMPLEFORMAT_IEEEFP && bits == 32 && samples >= 3) {
		mode = uint8_t(samples + 6);
	} else {
		throw MediaFormatException("unsupported TIFF pixel dtype");
	}
	uint16_t count = 0, *extras = nullptr;
	TIFFGetFieldDefaulted(tif, TIFFTAG_EXTRASAMPLES, &count, &extras);
	if (count != ((samples == 2 || samples == 4) ? 1 : 0) ||
	    (count && (!extras || extras[0] != EXTRASAMPLE_UNASSALPHA))) {
		throw MediaFormatException("TIFF requires unassociated alpha");
	}
	MediaProduct(width, height, max_pixels, "TIFF pixels");
	MediaProduct(uint64_t(width) * height, uint64_t(samples) * bits / 8, max_bytes, "TIFF decoded bytes");
	bytes.Check();
	return {{width, height, samples, mode}, planar, photo};
}

static DecodedImagePixels DecodeTIFF(ClientContext &context, const_data_ptr_t data, idx_t size, idx_t max_pixels,
                                     idx_t max_bytes, uint16_t output_channels, idx_t output_width, idx_t storage_width,
                                     idx_t remaining) {
	TIFFBytes source(context, data, size, 0, false, nullptr, size * 4);
	auto layout = ReadTIFFLayout(source, max_pixels, max_bytes);
	auto element_size = ImageLogicalType::ElementSize(ImageLogicalType::ModeName(layout.image.mode));
	ImageCodecContract::CheckDecodedBytes(layout.image.width, layout.image.height, layout.image.channels * element_size,
	                                      output_channels ? output_channels : layout.image.channels,
	                                      output_width ? output_width : element_size, storage_width, max_bytes);
	ImageOperatorContract::CheckSize(layout.image.width, layout.image.height,
	                                 output_channels ? output_channels : layout.image.channels, remaining,
	                                 storage_width);
	DecodedImagePixels result {layout.image, string(layout.image.Bytes(), '\0')};
	auto stride = idx_t(layout.image.width) * layout.image.channels * element_size;
	auto scanline_size = TIFFScanlineSize64(source.handle);
	auto expected = layout.planar == PLANARCONFIG_CONTIG ? stride : idx_t(layout.image.width) * element_size;
	if (scanline_size != expected) {
		throw MediaFormatException("TIFF scanline size does not match its pixel layout");
	}
	string scanline(expected, '\0');
	auto planes = layout.planar == PLANARCONFIG_CONTIG ? 1 : layout.image.channels;
	for (uint16_t channel = 0; channel < planes; channel++) {
		for (uint32_t row = 0; row < layout.image.height; row++) {
			source.Check();
			source.Check(TIFFReadScanline(source.handle, &scanline[0], row, channel) == 1);
			auto target = data_ptr_cast(&result.data[0]) + row * stride;
			if (planes == 1) {
				memcpy(target, scanline.data(), expected);
			} else {
				for (idx_t x = 0; x < layout.image.width; x++) {
					memcpy(target + (x * layout.image.channels + channel) * element_size,
					       scanline.data() + x * element_size, element_size);
				}
			}
		}
	}
	if (layout.photometric == PHOTOMETRIC_MINISWHITE) {
		for (idx_t i = 0; i < layout.image.Size(); i += layout.image.channels) {
			if (i % 16384 == 0) {
				source.Check();
			}
			auto pixels = data_ptr_cast(&result.data[0]);
			ImageStore(pixels, layout.image.mode, i,
			           ImageRange(layout.image.mode) - ImageSample(pixels, layout.image.mode, i));
		}
	}
	return result;
}

static string EncodeTIFF(ClientContext &context, const ImagePixelView &image, idx_t limit) {
	TIFFBytes output(context, nullptr, 0, limit, true);
	auto tif = output.handle;
	auto mode = ImageLogicalType::ModeName(image.layout.mode);
	auto width = image.layout.width, height = image.layout.height;
	auto channels = image.layout.channels;
	auto element_size = ImageLogicalType::ElementSize(mode);
	auto stride = idx_t(width) * channels * element_size;
	output.Check(
	    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width) && TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height) &&
	    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, channels) &&
	    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, uint16_t(element_size * 8)) &&
	    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, image.layout.mode <= 8 ? SAMPLEFORMAT_UINT : SAMPLEFORMAT_IEEEFP) &&
	    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, channels < 3 ? PHOTOMETRIC_MINISBLACK : PHOTOMETRIC_RGB) &&
	    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG) &&
	    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE) &&
	    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, uint32_t(MaxValue<idx_t>(1, 65536 / stride))));
	uint16_t alpha = EXTRASAMPLE_UNASSALPHA;
	if (channels == 2 || channels == 4) {
		output.Check(TIFFSetField(tif, TIFFTAG_EXTRASAMPLES, 1, &alpha) == 1);
	}
	// libtiff may byte-swap its writable scanline argument. Keep engine pixels immutable.
	string scanline(stride, '\0');
	for (uint32_t row = 0; row < height; row++) {
		output.Check();
		memcpy(&scanline[0], image.data + idx_t(row) * stride, stride);
		output.Check(TIFFWriteScanline(tif, &scanline[0], row, 0) == 1);
	}
	return output.Finish();
}

struct CodecState {
	ClientContext &context;
	idx_t max_pixels, max_bytes, output_limit;
	uint16_t output_channels = 0;
	uint8_t source_mode = 0;
	idx_t output_width = 0, storage_width = 1, remaining = ImageOperatorContract::MAX_BYTES, decoded_buffer_bytes = 0;
	std::exception_ptr error;
	AVCodecContext *codec = nullptr;
	AVFrame *frame = nullptr;
	AVPacket *packet = nullptr;

	CodecState(ClientContext &context, idx_t max_pixels, idx_t max_bytes, idx_t output_limit)
	    : context(context), max_pixels(max_pixels), max_bytes(max_bytes), output_limit(output_limit) {
	}
	~CodecState() {
		av_packet_free(&packet);
		av_frame_free(&frame);
		avcodec_free_context(&codec);
	}
	void Check(int code, const char *operation) {
		MediaInterrupt(context);
		if (error) {
			std::rethrow_exception(error);
		}
		MediaCheck(code, operation);
	}
	static int DecodeBuffer(AVCodecContext *codec, AVFrame *frame, int flags) noexcept {
		auto &self = *static_cast<CodecState *>(codec->opaque);
		try {
			MediaInterrupt(self.context);
			if (frame->width <= 0 || frame->height <= 0) {
				throw MediaFormatException("invalid decoder dimensions");
			}
			MediaProduct(frame->width, frame->height, self.max_pixels, "decoded image pixels");
			auto mode = ImageLogicalType::ModeName(self.source_mode ? self.source_mode
			                                                        : DecodedMode(AVPixelFormat(frame->format)));
			auto channels = ImageLogicalType::ChannelsForMode(mode);
			auto element_size = ImageLogicalType::ElementSize(mode);
			ImageCodecContract::CheckDecodedBytes(frame->width, frame->height, channels * element_size,
			                                      self.output_channels ? self.output_channels : channels,
			                                      self.output_width ? self.output_width : element_size,
			                                      self.storage_width, self.max_bytes);
			ImageOperatorContract::CheckSize(frame->width, frame->height,
			                                 self.output_channels ? self.output_channels
			                                                      : ImageLogicalType::ChannelsForMode(mode),
			                                 self.remaining, self.storage_width);
			int width = frame->width, height = frame->height, alignments[AV_NUM_DATA_POINTERS] = {};
			avcodec_align_dimensions2(codec, &width, &height, alignments);
			int alignment = 1;
			for (auto value : alignments) {
				alignment = MaxValue(alignment, value);
			}
			auto bytes = av_image_get_buffer_size(AVPixelFormat(frame->format), width, height, alignment);
			self.Check(bytes, "calculate image decoder buffer");
			MediaProduct(1, uint64_t(bytes) + 4 * AV_INPUT_BUFFER_PADDING_SIZE, self.max_bytes,
			             "aligned image decoder bytes");
			self.decoded_buffer_bytes = idx_t(bytes) + 4 * AV_INPUT_BUFFER_PADDING_SIZE;
			return avcodec_default_get_buffer2(codec, frame, flags);
		} catch (...) {
			self.error = std::current_exception();
			return AVERROR_EXTERNAL;
		}
	}
	static int EncodeBuffer(AVCodecContext *codec, AVPacket *packet, int flags) noexcept {
		auto &self = *static_cast<CodecState *>(codec->opaque);
		try {
			MediaInterrupt(self.context);
			if (packet->size < 0 || idx_t(packet->size) > self.output_limit) {
				throw OutOfRangeException("Image encoding exceeds its output byte limit");
			}
			return avcodec_default_get_encode_buffer(codec, packet, flags);
		} catch (...) {
			self.error = std::current_exception();
			return AVERROR_EXTERNAL;
		}
	}
};

static AVCodecID ImageCodecID(ClientContext &context, const_data_ptr_t data, idx_t size, idx_t &width, idx_t &height) {
	auto require = [&](idx_t offset, idx_t count) {
		if (offset > size || count > size - offset) {
			throw MediaFormatException("truncated image header");
		}
	};
	auto be16 = [&](idx_t offset) {
		require(offset, 2);
		return idx_t(data[offset]) * 256 + data[offset + 1];
	};
	auto le16 = [&](idx_t offset) {
		require(offset, 2);
		return idx_t(data[offset]) + idx_t(data[offset + 1]) * 256;
	};
	auto le32 = [&](idx_t offset) {
		require(offset, 4);
		return le16(offset) + (le16(offset + 2) << 16);
	};
	if (size >= 8 && memcmp(data, "\x89PNG\r\n\x1a\n", 8) == 0) {
		require(8, 25);
		width = (be16(16) << 16) + be16(18);
		height = (be16(20) << 16) + be16(22);
		return AV_CODEC_ID_PNG;
	}
	if (size >= 6 && (memcmp(data, "GIF87a", 6) == 0 || memcmp(data, "GIF89a", 6) == 0)) {
		width = le16(6);
		height = le16(8);
		return AV_CODEC_ID_GIF;
	}
	if (size >= 2 && data[0] == 'B' && data[1] == 'M') {
		auto header = le32(14);
		if (header == 12) {
			width = le16(18);
			height = le16(20);
		} else if (header >= 40) {
			width = le32(18);
			auto signed_height = int64_t(int32_t(le32(22)));
			height = idx_t(signed_height < 0 ? -signed_height : signed_height);
		} else {
			throw MediaFormatException("unsupported BMP header");
		}
		return AV_CODEC_ID_BMP;
	}
	if (size >= 2 && data[0] == 0xff && data[1] == 0xd8) {
		idx_t position = 2;
		while (position < size) {
			MediaInterrupt(context);
			if (data[position++] != 0xff) {
				throw MediaFormatException("invalid JPEG marker");
			}
			while (position < size && data[position] == 0xff) {
				if (position % 16384 == 0) {
					MediaInterrupt(context);
				}
				position++;
			}
			require(position, 1);
			auto marker = data[position++];
			if (marker == 0xda || marker == 0xd9) {
				break;
			}
			if (marker == 1 || (marker >= 0xd0 && marker <= 0xd7)) {
				continue;
			}
			auto length = be16(position);
			if (length < 2) {
				throw MediaFormatException("invalid JPEG segment length");
			}
			require(position, length);
			if (marker >= 0xc0 && marker <= 0xcf && marker != 0xc4 && marker != 0xc8 && marker != 0xcc) {
				if (length < 8) {
					throw MediaFormatException("invalid JPEG frame header");
				}
				height = be16(position + 3);
				width = be16(position + 5);
				return AV_CODEC_ID_MJPEG;
			}
			position += length;
		}
		throw MediaFormatException("JPEG contains no frame dimensions");
	}
	throw MediaFormatException("Image bytes must contain PNG, JPEG, TIFF, GIF or BMP");
}

static DecodedImagePixels DecodeCodec(ClientContext &context, const_data_ptr_t data, idx_t size, idx_t max_pixels,
                                      idx_t max_bytes, uint16_t output_channels, idx_t output_width,
                                      idx_t storage_width, idx_t remaining) {
	idx_t width = 0, height = 0;
	auto id = ImageCodecID(context, data, size, width, height);
	if (!width || !height || width > INT_MAX || height > INT_MAX) {
		throw MediaFormatException("invalid image dimensions");
	}
	MediaProduct(width, height, max_pixels, "decoded image pixels");
	// Header preflight classifies impossible allocations before codec-specific
	// dimension checks can turn them into suppressible content errors.
	MediaProduct(width * height, id == AV_CODEC_ID_PNG && data[24] == 16 ? 8 : 4, max_bytes, "decoded image bytes");
	CodecState state(context, max_pixels, max_bytes, 0);
	state.output_channels = output_channels;
	state.output_width = output_width;
	state.storage_width = storage_width;
	state.remaining = remaining;
	if (id == AV_CODEC_ID_BMP) {
		auto header = ImageBMPHeader::Read([&](idx_t offset, idx_t count) {
			if (offset > size || count > size - offset) {
				throw MediaFormatException("truncated BMP header or color table");
			}
			return string(const_char_ptr_cast(data + offset), count);
		});
		state.source_mode = ImageLogicalType::ModeCode(header.mode == "P"   ? "RGBA"
		                                               : header.mode == "1" ? "L"
		                                                                    : header.mode);
	}
	auto implementation = avcodec_find_decoder(id);
	if (!implementation) {
		throw InvalidInputException("Required native Image decoder is unavailable");
	}
	state.codec = avcodec_alloc_context3(implementation);
	state.frame = av_frame_alloc();
	state.packet = av_packet_alloc();
	if (!state.codec || !state.frame || !state.packet) {
		throw OutOfMemoryException("Cannot allocate Image decoder");
	}
	state.codec->thread_count = 1;
	state.codec->opaque = &state;
	state.codec->get_buffer2 = CodecState::DecodeBuffer;
	state.codec->err_recognition = AV_EF_CRCCHECK | AV_EF_BITSTREAM | AV_EF_BUFFER | AV_EF_EXPLODE;
	// Visible pixels are bounded before allocation in DecodeBuffer. FFmpeg also
	// applies max_pixels to its padded buffer dimensions, which can exceed a
	// valid caller limit for small images; aligned bytes have their own budget.
	state.codec->max_pixels = NumericLimits<int64_t>::Maximum();
	state.Check(avcodec_open2(state.codec, implementation, nullptr), "open Image decoder");
	state.Check(av_new_packet(state.packet, int(size)), "allocate Image input packet");
	memcpy(state.packet->data, data, size);
	state.Check(avcodec_send_packet(state.codec, state.packet), "decode Image packet");
	state.Check(avcodec_receive_frame(state.codec, state.frame), "receive decoded Image");
	auto &frame = *state.frame;
	if (frame.width != int(width) || frame.height != int(height)) {
		throw MediaFormatException("decoded image dimensions differ from its header");
	}
	auto code = state.source_mode ? state.source_mode : DecodedMode(AVPixelFormat(frame.format));

	auto mode = ImageLogicalType::ModeName(code);
	ImageLayout layout {uint32_t(width), uint32_t(height), ImageLogicalType::ChannelsForMode(mode), code};
	MediaProduct(layout.Size(), ImageLogicalType::ElementSize(mode), max_bytes - state.decoded_buffer_bytes,
	             "Image decoder max_decoded_bytes including aligned frame");
	DecodedImagePixels result {layout, string(layout.Bytes(), '\0')};
	MediaConvertPixels(context, frame, mode, uint32_t(width), uint32_t(height), data_ptr_cast(&result.data[0]));
	return result;
}

//! libavcodec's MJPEG encoder exposes only three-component YUV formats.
//! Encode L directly with libjpeg so grayscale mode survives round-trips.
class GrayJPEGEncoder {
public:
	GrayJPEGEncoder(ClientContext &context, idx_t limit) : context(context), limit(limit) {
		codec.err = jpeg_std_error(&manager);
		manager.error_exit = Error;
		codec.client_data = this;
		destination.init_destination = Init;
		destination.empty_output_buffer = Flush;
		destination.term_destination = Finish;
	}
	~GrayJPEGEncoder() {
		if (codec.mem) {
			jpeg_destroy_compress(&codec);
		}
	}
	string Encode(const ImagePixelView &image) {
		// All state changed across a libjpeg error jump lives in this heap
		// object. No C++ object with a destructor is bypassed by longjmp.
		if (setjmp(jump)) {
			if (error) {
				std::rethrow_exception(error);
			}
			if (manager.msg_code == JERR_OUT_OF_MEMORY) {
				throw OutOfMemoryException("JPEG allocation failed: %s", message);
			}
			throw IOException("JPEG encoding failed: %s", message);
		}
		jpeg_create_compress(&codec);
		codec.dest = &destination;
		codec.image_width = image.layout.width;
		codec.image_height = image.layout.height;
		codec.input_components = 1;
		codec.in_color_space = JCS_GRAYSCALE;
		jpeg_set_defaults(&codec);
		jpeg_set_quality(&codec, 95, TRUE);
		jpeg_start_compress(&codec, TRUE);
		while (codec.next_scanline < codec.image_height) {
			MediaInterrupt(context);
			JSAMPROW row = const_cast<JSAMPROW>(image.data + idx_t(codec.next_scanline) * image.layout.width);
			jpeg_write_scanlines(&codec, &row, 1);
		}
		jpeg_finish_compress(&codec);
		MediaInterrupt(context);
		return std::move(output);
	}

private:
	static GrayJPEGEncoder &Self(j_common_ptr codec) {
		return *static_cast<GrayJPEGEncoder *>(codec->client_data);
	}
	static void Error(j_common_ptr codec) {
		auto &self = Self(codec);
		codec->err->format_message(codec, self.message);
		std::longjmp(self.jump, 1);
	}
	static void Init(j_compress_ptr codec) {
		auto &self = Self(j_common_ptr(codec));
		self.destination.next_output_byte = self.buffer;
		self.destination.free_in_buffer = sizeof(self.buffer);
	}
	static boolean Flush(j_compress_ptr codec) {
		auto &self = Self(j_common_ptr(codec));
		self.Append(sizeof(self.buffer));
		Init(codec);
		return TRUE;
	}
	static void Finish(j_compress_ptr codec) {
		auto &self = Self(j_common_ptr(codec));
		self.Append(sizeof(self.buffer) - self.destination.free_in_buffer);
	}
	void Append(idx_t count) noexcept {
		try {
			MediaInterrupt(context);
			if (count > limit - output.size()) {
				throw OutOfRangeException("JPEG encoding exceeds its output byte limit");
			}
			output.append(const_char_ptr_cast(buffer), count);
			return;
		} catch (...) {
			error = std::current_exception();
		}
		std::longjmp(jump, 1);
	}

	ClientContext &context;
	idx_t limit;
	jpeg_compress_struct codec {};
	jpeg_error_mgr manager {};
	jpeg_destination_mgr destination {};
	std::jmp_buf jump;
	std::exception_ptr error;
	char message[JMSG_LENGTH_MAX] {};
	JOCTET buffer[65536];
	string output;
};

static string EncodeCodec(ClientContext &context, const ImagePixelView &image, const string &format, idx_t limit) {
	AVCodecID id = format == "JPEG" ? AV_CODEC_ID_MJPEG : format == "GIF" ? AV_CODEC_ID_GIF : AV_CODEC_ID_BMP;
	auto implementation = avcodec_find_encoder(id);
	if (!implementation) {
		throw InvalidInputException("Required native Image encoder is unavailable");
	}
	CodecState state(context, ImageOperatorContract::MAX_PIXELS, ImageOperatorContract::MAX_BYTES, limit);
	state.codec = avcodec_alloc_context3(implementation);
	state.frame = av_frame_alloc();
	state.packet = av_packet_alloc();
	if (!state.codec || !state.frame || !state.packet) {
		throw OutOfMemoryException("Cannot allocate Image encoder");
	}
	auto &codec = *state.codec;
	codec.width = int(image.layout.width);
	codec.height = int(image.layout.height);
	codec.thread_count = 1;
	codec.time_base = {1, 1};
	codec.pix_fmt = format == "JPEG" ? AV_PIX_FMT_YUVJ444P : format == "GIF" ? AV_PIX_FMT_PAL8 : AV_PIX_FMT_BGR24;
	codec.color_range = AVCOL_RANGE_JPEG;
	codec.opaque = &state;
	codec.get_encode_buffer = CodecState::EncodeBuffer;
	if (format == "JPEG") {
		codec.flags |= AV_CODEC_FLAG_QSCALE;
		codec.global_quality = FF_QP2LAMBDA * 2;
	}
	state.Check(avcodec_open2(&codec, implementation, nullptr), "open Image encoder");
	auto &frame = *state.frame;
	frame.width = codec.width;
	frame.height = codec.height;
	frame.format = codec.pix_fmt;
	frame.quality = codec.global_quality;
	frame.pts = 0;
	frame.color_range = AVCOL_RANGE_JPEG;
	auto padded_width = (uint64_t(frame.width) + 31) & ~uint64_t(31);
	auto padded_height = (uint64_t(frame.height) + 31) & ~uint64_t(31);
	MediaProduct(padded_width * 4, padded_height, ImageOperatorContract::MAX_BYTES, "Image encoder frame bytes");
	state.Check(av_frame_get_buffer(&frame, 32), "allocate Image encoder frame");
	if (format == "GIF") {
		for (idx_t i = 0; i < 256; i++) {
			uint32_t color;
			if (image.layout.channels == 1) {
				color = uint32_t(i) * 0x010101U;
			} else {
				color = uint32_t(((i >> 5) * 255 / 7) << 16) | uint32_t((((i >> 2) & 7) * 255 / 7) << 8) |
				        uint32_t((i & 3) * 255 / 3);
			}
			color |= 0xff000000U;
			memcpy(frame.data[1] + i * 4, &color, sizeof(color));
		}
		for (idx_t y = 0; y < image.layout.height; y++) {
			MediaInterrupt(context);
			for (idx_t x = 0; x < image.layout.width; x++) {
				auto source = image.data + (y * image.layout.width + x) * image.layout.channels;
				frame.data[0][y * frame.linesize[0] + x] =
				    image.layout.channels == 1
				        ? source[0]
				        : uint8_t((source[0] & 0xe0) | ((source[1] >> 3) & 0x1c) | (source[2] >> 6));
			}
		}
	} else {
		auto byte = [](double value) {
			return uint8_t(std::floor(MaxValue(0.0, MinValue(255.0, value)) + 0.5));
		};
		for (idx_t y = 0; y < image.layout.height; y++) {
			MediaInterrupt(context);
			for (idx_t x = 0; x < image.layout.width; x++) {
				auto source = image.data + (y * image.layout.width + x) * image.layout.channels;
				double red = source[0], green = image.layout.channels == 1 ? red : source[1],
				       blue = image.layout.channels == 1 ? red : source[2];
				if (format == "BMP") {
					auto target = frame.data[0] + y * frame.linesize[0] + x * 3;
					target[0] = uint8_t(blue);
					target[1] = uint8_t(green);
					target[2] = uint8_t(red);
				} else {
					frame.data[0][y * frame.linesize[0] + x] = byte(.299 * red + .587 * green + .114 * blue);
					frame.data[1][y * frame.linesize[1] + x] = byte(-.168736 * red - .331264 * green + .5 * blue + 128);
					frame.data[2][y * frame.linesize[2] + x] = byte(.5 * red - .418688 * green - .081312 * blue + 128);
				}
			}
		}
	}
	state.Check(avcodec_send_frame(&codec, &frame), "encode Image frame");
	state.Check(avcodec_receive_packet(&codec, state.packet), "receive encoded Image");
	if (state.packet->size < 0 || idx_t(state.packet->size) > limit) {
		throw OutOfRangeException("Image encoding exceeds its output byte limit");
	}
	string output(const_char_ptr_cast(state.packet->data), idx_t(state.packet->size));
	if (format == "GIF" && (output.empty() || uint8_t(output.back()) != 0x3b)) {
		if (output.size() == limit) {
			throw OutOfRangeException("GIF trailer exceeds its output byte limit");
		}
		output.push_back(char(0x3b));
	}
	return output;
}

} // namespace

ImageLayout NativeImageCodec::TIFFMetadata(ClientContext &context, ResolvedFile &file, idx_t budget, idx_t max_pixels) {
	TIFFBytes source(context, nullptr, file.LogicalSize(), 0, false, &file, budget);
	return ReadTIFFLayout(source, max_pixels, NumericLimits<idx_t>::Maximum()).image;
}

DecodedImagePixels NativeImageCodec::Decode(ClientContext &context, const_data_ptr_t data, idx_t size,
                                            const LogicalType &output_type, const string &output_mode, idx_t remaining,
                                            idx_t max_pixels, idx_t max_bytes) {
	MediaInterrupt(context);
	if (size > ImageOperatorContract::MAX_BYTES) {
		throw OutOfRangeException("Image encoded input exceeds 256 MiB");
	}
	bool tiff = size >= 4 && ((data[0] == 'I' && data[1] == 'I' && (data[2] == 42 || data[2] == 43) && !data[3]) ||
	                          (data[0] == 'M' && data[1] == 'M' && !data[2] && (data[3] == 42 || data[3] == 43)));
	auto channels = output_mode.empty() ? uint16_t(0) : ImageLogicalType::ChannelsForMode(output_mode);
	auto output_width = output_mode.empty() ? idx_t(0) : ImageLogicalType::ElementSize(output_mode);
	auto storage_width = GetTypeIdSize(ImageLogicalType::StorageType(output_type).InternalType());
	auto result =
	    tiff
	        ? DecodeTIFF(context, data, size, max_pixels, max_bytes, channels, output_width, storage_width, remaining)
	        : DecodeCodec(context, data, size, max_pixels, max_bytes, channels, output_width, storage_width, remaining);
	auto mode = ImageLogicalType::ModeName(result.layout.mode);
	try {
		ImageVector::ValidatePixels(const_data_ptr_cast(result.data.data()), ImageLogicalType::PixelType(mode),
		                            result.layout.Size(), mode, "decoded Image");
	} catch (const InvalidInputException &error) {
		throw MediaFormatException(error.what());
	}
	MediaInterrupt(context);
	return result;
}

idx_t NativeImageCodec::Write(ClientContext &context, const DecodedImagePixels &image, const string &mode,
                              Vector &result, idx_t row, idx_t remaining) {
	auto layout = image.layout;
	if (!mode.empty()) {
		layout.mode = ImageLogicalType::ModeCode(mode);
		layout.channels = ImageLogicalType::ChannelsForMode(mode);
	}
	auto size = ImageCodecContract::OutputSize(result.GetType(), layout, remaining);
	ImageOperatorOutput output(result, row, layout);
	ImagePixelView source {image.layout, const_data_ptr_cast(image.data.data())};
	ConvertImagePixels(source, layout, output.Data(), [&context]() { MediaInterrupt(context); });
	output.Finish(context);
	return size;
}

string NativeImageCodec::Encode(ClientContext &context, const ImagePixelView &image, const string &format,
                                idx_t limit) {
	ImageCodecContract::CheckEncoding(format, image.layout);
	if (format == "JPEG" && image.layout.channels == 1) {
		auto encoder = make_uniq<GrayJPEGEncoder>(context, limit);
		return encoder->Encode(image);
	}
	return format == "TIFF" ? EncodeTIFF(context, image, limit) : EncodeCodec(context, image, format, limit);
}

} // namespace duckdb
