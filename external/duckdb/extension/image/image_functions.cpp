// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "media_reader.hpp"
#include "image_codec.hpp"
#include "image_bmp.hpp"
#include "image_gif.hpp"
#include "image_webp.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
extern "C" {
#include <libavutil/crc.h>
}

namespace duckdb {
namespace {

struct ImageHeader {
	uint32_t width;
	uint32_t height;
	string format;
	string mode;
};

// Inspect encoded headers only. Large ancillary chunks cannot turn metadata
// inspection into a full download or pixel decode.
static ImageHeader ReadHeader(ClientContext &context, ResolvedFile &input, const FileReference &file, uint64_t budget,
                              uint64_t max_pixels) {
	uint64_t consumed = 0;
	uint64_t buffer_offset = 0;
	string buffer;
	bool buffered = false;
	auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
	auto read = [&](uint64_t offset, uint64_t size) {
		MediaInterrupt(context);
		if (offset > input.LogicalSize() || size > input.LogicalSize() - offset) {
			throw MediaFormatException("truncated image header");
		}
		if (offset < buffer_offset || offset - buffer_offset > buffer.size() ||
		    size > buffer.size() - (offset - buffer_offset)) {
			if (std::chrono::steady_clock::now() >= deadline) {
				throw OutOfRangeException("native image metadata probe exceeded its time budget");
			}
			auto cached = offset >= buffer_offset && offset - buffer_offset <= buffer.size()
			                  ? uint64_t(buffer.size()) - (offset - buffer_offset)
			                  : uint64_t(0);
			auto missing = size - cached;
			if (missing > budget - consumed) {
				throw OutOfRangeException("native image metadata exceeds max_bytes");
			}
			// JPEG marker bytes share bounded range reads. PNG's fixed header
			// still uses exact small reads without fetching encoded pixels.
			// Keep an overlapping prefix and fetch only its missing tail.
			// Drop older bytes so the cache stays one bounded read window.
			auto read_offset = offset + cached;
			auto count = buffered ? MinValue<uint64_t>(64 * 1024, input.LogicalSize() - read_offset) : missing;
			count = MinValue<uint64_t>(count, budget - consumed);
			if (cached) {
				buffer.erase(0, NumericCast<idx_t>(offset - buffer_offset));
			} else {
				buffer.clear();
			}
			buffer.resize(NumericCast<idx_t>(cached + count));
			input.ReadExact(reinterpret_cast<data_ptr_t>(&buffer[NumericCast<idx_t>(cached)]), count, read_offset);
			buffer_offset = offset;
			consumed += count;
			if (std::chrono::steady_clock::now() >= deadline) {
				throw OutOfRangeException("native image metadata probe exceeded its time budget");
			}
		}
		return buffer.substr(NumericCast<idx_t>(offset - buffer_offset), NumericCast<idx_t>(size));
	};
	auto byte = [](const string &s, idx_t i) {
		return uint32_t(uint8_t(s[i]));
	};
	auto be16 = [&](const string &s, idx_t i) {
		return (byte(s, i) << 8) | byte(s, i + 1);
	};
	auto be32 = [&](const string &s, idx_t i) {
		return (be16(s, i) << 16) | be16(s, i + 2);
	};
	auto signature = read(0, 8);
	ImageHeader result;
	if (signature == string("\x89PNG\r\n\x1a\n", 8)) {
		auto ihdr = read(8, 25);
		if (be32(ihdr, 0) != 13 || ihdr.substr(4, 4) != "IHDR" || byte(ihdr, 18) || byte(ihdr, 19) ||
		    byte(ihdr, 20) > 1) {
			throw MediaFormatException("invalid PNG IHDR");
		}
		auto depth = byte(ihdr, 16), color = byte(ihdr, 17);
		bool valid_depth = depth == 8 || (depth == 16 && color != 3) ||
		                   ((color == 0 || color == 3) && (depth == 1 || depth == 2 || depth == 4));
		auto crc = av_crc(av_crc_get_table(AV_CRC_32_IEEE_LE), UINT32_MAX,
		                  reinterpret_cast<const uint8_t *>(ihdr.data() + 4), 17) ^
		           UINT32_MAX;
		if (!valid_depth || crc != be32(ihdr, 21)) {
			throw MediaFormatException("invalid PNG bit depth or IHDR checksum");
		}
		result = {be32(ihdr, 8), be32(ihdr, 12), "PNG", ""};
		switch (byte(ihdr, 17)) {
		case 0:
			result.mode = depth == 16 ? "L16" : depth == 1 ? "1" : "L";
			break;
		case 2:
			result.mode = depth == 16 ? "RGB16" : "RGB";
			break;
		case 3:
			result.mode = "P";
			break;
		case 4:
			result.mode = depth == 16 ? "LA16" : "LA";
			break;
		case 6:
			result.mode = depth == 16 ? "RGBA16" : "RGBA";
			break;
		default:
			throw MediaFormatException("unsupported PNG color type");
		}
		MediaValidateMIME(file, "image/png");
	} else if (byte(signature, 0) == 0xff && byte(signature, 1) == 0xd8) {
		buffered = true;
		uint64_t offset = 2;
		for (;;) {
			auto marker = read(offset++, 1);
			if (byte(marker, 0) != 0xff) {
				throw MediaFormatException("invalid JPEG marker");
			}
			do {
				marker = read(offset++, 1);
			} while (byte(marker, 0) == 0xff);
			auto code = byte(marker, 0);
			if (!code || code == 0xda || code == 0xd9) {
				throw MediaFormatException("JPEG is missing its frame header");
			}
			if (code == 0x01 || (code >= 0xd0 && code <= 0xd7)) {
				continue;
			}
			auto length = be16(read(offset, 2), 0);
			if (length < 2 || offset > input.LogicalSize() || length > input.LogicalSize() - offset) {
				throw MediaFormatException("invalid JPEG segment length");
			}
			if (code >= 0xc0 && code <= 0xcf && code != 0xc4 && code != 0xc8 && code != 0xcc) {
				if (length < 8) {
					throw MediaFormatException("invalid JPEG frame header");
				}
				auto sof = read(offset + 2, 6);
				auto components = byte(sof, 5);
				if ((components != 1 && components != 3 && components != 4) || length != 8 + 3 * components) {
					throw MediaFormatException("unsupported JPEG components");
				}
				auto precision = byte(sof, 0);
				bool lossless = code == 0xc3 || code == 0xc7 || code == 0xcb || code == 0xcf;
				// T.81 frame precision: baseline DCT is 8-bit, other DCT
				// processes are 8/12-bit, and lossless processes are 2-16-bit.
				bool valid_precision =
				    lossless ? precision >= 2 && precision <= 16 : precision == 8 || (code != 0xc0 && precision == 12);
				if (!valid_precision || (components == 4 && precision > 8)) {
					throw MediaFormatException("unsupported JPEG sample precision");
				}
				auto mode = components == 1   ? (precision > 8 ? "L16" : "L")
				            : components == 4 ? "CMYK"
				                              : (precision > 8 ? "RGB16" : "RGB");
				result = {be16(sof, 3), be16(sof, 1), "JPEG", mode};
				break;
			}
			offset += length;
		}
		MediaValidateMIME(file, "image/jpeg");
	} else if (signature.substr(0, 6) == "GIF87a" || signature.substr(0, 6) == "GIF89a") {
		auto header = ImageGIFHeader::Read(read);
		result = {header.width, header.height, "GIF", "P"};
		MediaValidateMIME(file, "image/gif");
	} else if (signature.substr(0, 2) == "BM") {
		auto header = ImageBMPHeader::Read(read);
		result = {header.width, header.height, "BMP", header.mode};
		MediaValidateMIME(file, "image/bmp");
	} else if (signature.substr(0, 4) == "RIFF") {
		auto header = ImageWebPHeader::Read(read, input.LogicalSize());
		result = {header.width, header.height, "WEBP", header.alpha ? "RGBA" : "RGB"};
		MediaValidateMIME(file, "image/webp");
	} else if ((signature.substr(0, 2) == "II" && (byte(signature, 2) == 42 || byte(signature, 2) == 43) &&
	            !byte(signature, 3)) ||
	           (signature.substr(0, 2) == "MM" && !byte(signature, 2) &&
	            (byte(signature, 3) == 42 || byte(signature, 3) == 43))) {
		auto layout = NativeImageCodec::TIFFMetadata(context, input, signature, budget - consumed, max_pixels);
		result = {layout.width, layout.height, "TIFF", ImageLogicalType::ModeName(layout.mode)};
		MediaValidateMIME(file, "image/tiff");
	} else {
		throw MediaFormatException("native image supports PNG, JPEG, TIFF, GIF, BMP and WebP encoded files");
	}
	if (!result.width || !result.height) {
		throw MediaFormatException("invalid image dimensions");
	}
	MediaProduct(result.width, result.height, max_pixels, "image pixels");
	MediaInterrupt(context);
	return result;
}

static void ImageMetadata(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	for (idx_t row = 0; row < args.size(); row++) {
		auto file = args.data[0].GetValue(row);
		if (file.IsNull() ||
		    (args.ColumnCount() == 3 && (args.data[1].GetValue(row).IsNull() || args.data[2].GetValue(row).IsNull()))) {
			result.SetValue(row, Value(result.GetType()));
			continue;
		}
		auto budget = args.ColumnCount() == 3 ? MediaPositive(args.data[1].GetValue(row), "max_bytes", 64 * MEDIA_MIB)
		                                      : MEDIA_MIB;
		auto pixels = args.ColumnCount() == 3
		                  ? MediaPositive(args.data[2].GetValue(row), "max_pixels", NumericLimits<uint64_t>::Maximum())
		                  : MEDIA_MAX_PIXELS;
		auto reference = FileReference::FromValue(file, "native_image_file_metadata");
		auto resolved = ResolvedFile::Open(state.GetContext(), reference);
		auto header = ReadHeader(state.GetContext(), *resolved, reference, budget, pixels);
		result.SetValue(row,
		                Value::STRUCT(result.GetType(), {Value::UINTEGER(header.width), Value::UINTEGER(header.height),
		                                                 Value(header.format), Value(header.mode)}));
	}
}

static void DecodeImage(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	for (auto &child : StructVector::GetEntries(result)) {
		child->SetVectorType(VectorType::FLAT_VECTOR);
	}
	uint64_t batch_bytes = 0;
	for (idx_t row = 0; row < args.size(); row++) {
		auto value = args.data[0].GetValue(row);
		bool null = value.IsNull();
		if (args.ColumnCount() == 6) {
			for (idx_t col = 3; col < 6; col++) {
				null = null || args.data[col].GetValue(row).IsNull();
			}
		}
		if (args.ColumnCount() >= 3 && args.data[2].GetValue(row).IsNull()) {
			null = true;
		}
		if (null) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		string mode;
		bool has_mode = args.ColumnCount() >= 2 && !args.data[1].GetValue(row).IsNull();
		if (has_mode) {
			mode = args.data[1].GetValue(row).GetValue<string>();
		}
		if (has_mode) {
			ImageLogicalType::ModeCode(mode);
		}
		auto on_error = args.ColumnCount() >= 3 ? args.data[2].GetValue(row).GetValue<string>() : "raise";
		if (on_error != "raise" && on_error != "null") {
			throw InvalidInputException("on_error must be raise or null");
		}
		auto maximum = NumericLimits<uint64_t>::Maximum();
		auto input_bytes = args.ColumnCount() == 6
		                       ? MediaPositive(args.data[3].GetValue(row), "max_input_bytes", maximum)
		                       : 256 * MEDIA_MIB;
		auto pixels = args.ColumnCount() == 6 ? MediaPositive(args.data[4].GetValue(row), "max_pixels", maximum)
		                                      : MEDIA_MAX_PIXELS;
		// Caller budgets are positive UBIGINT values; independent operator
		// limits still constrain the actual input, dimensions and output.
		pixels = MinValue<uint64_t>(pixels, MEDIA_MAX_PIXELS);
		auto output_bytes = args.ColumnCount() == 6
		                        ? MediaPositive(args.data[5].GetValue(row), "max_decoded_bytes", maximum)
		                        : MEDIA_MAX_FRAME_BYTES;
		auto &context = state.GetContext();
		try {
			auto file = FileReference::FromValue(value, "native_decode_image_file");
			auto resolved = ResolvedFile::Open(context, file);
			if (resolved->LogicalSize() > input_bytes) {
				throw OutOfRangeException("native image exceeds max_input_bytes");
			}
			if (resolved->LogicalSize() > ImageOperatorContract::MAX_BYTES) {
				throw OutOfRangeException("native image encoded input exceeds 256 MiB");
			}
			auto header = ReadHeader(context, *resolved, file, MinValue<uint64_t>(input_bytes, 64 * MEDIA_MIB), pixels);
			string encoded(idx_t(resolved->LogicalSize()), '\0');
			for (idx_t offset = 0; offset < encoded.size();) {
				MediaInterrupt(context);
				auto count = MinValue(ImageOperatorContract::COPY_BYTES, encoded.size() - offset);
				resolved->ReadExact(data_ptr_cast(&encoded[0]) + offset, count, offset);
				offset += count;
			}
			auto decoded =
			    NativeImageCodec::Decode(context, const_data_ptr_cast(encoded.data()), encoded.size(), result.GetType(),
			                             mode, MEDIA_BATCH_BYTES - batch_bytes, pixels, output_bytes);
			if (decoded.layout.width != header.width || decoded.layout.height != header.height) {
				throw MediaFormatException("decoded dimensions differ from image header");
			}
			batch_bytes +=
			    NativeImageCodec::Write(context, decoded, mode, result, row, MEDIA_BATCH_BYTES - batch_bytes);
		} catch (const MediaFormatException &) {
			MediaInterrupt(context);
			if (on_error != "null") {
				throw;
			}
			FlatVector::SetNull(result, row, true);
		}
	}
}
} // namespace

void RegisterMediaImages(ExtensionLoader &loader) {
	ScalarFunctionSet metadata("native_image_file_metadata");
	metadata.AddFunction(
	    MediaScalar("image_file_metadata", {LogicalType::ANY}, MediaImageMetadataType(), ImageMetadata));
	metadata.AddFunction(MediaScalar("image_file_metadata",
	                                 {LogicalType::ANY, LogicalType::UBIGINT, LogicalType::UBIGINT},
	                                 MediaImageMetadataType(), ImageMetadata));
	loader.RegisterFunction(metadata);
	ScalarFunctionSet decode("native_decode_image_file");
	for (auto &args :
	     vector<vector<LogicalType>> {{LogicalType::ANY},
	                                  {LogicalType::ANY, LogicalType::VARCHAR},
	                                  {LogicalType::ANY, LogicalType::VARCHAR, LogicalType::VARCHAR},
	                                  {LogicalType::ANY, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                                   LogicalType::UBIGINT, LogicalType::UBIGINT, LogicalType::UBIGINT}}) {
		decode.AddFunction(MediaScalar("decode_image_file", args, ImageLogicalType::Create(), DecodeImage));
	}
	loader.RegisterFunction(decode);
}
} // namespace duckdb
