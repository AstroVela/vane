// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "media_reader.hpp"
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
	auto read = [&](uint64_t offset, uint64_t size) {
		if (size > budget - consumed) {
			throw OutOfRangeException("native image metadata exceeds max_bytes");
		}
		if (offset > input.LogicalSize() || size > input.LogicalSize() - offset) {
			throw MediaFormatException("truncated image header");
		}
		string bytes(NumericCast<idx_t>(size), '\0');
		input.ReadExact(reinterpret_cast<data_ptr_t>(&bytes[0]), size, offset);
		consumed += size;
		return bytes;
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
			result.mode = byte(ihdr, 16) == 16 ? "I;16" : "L";
			break;
		case 2:
			result.mode = "RGB";
			break;
		case 3:
			result.mode = "P";
			break;
		case 4:
			result.mode = "LA";
			break;
		case 6:
			result.mode = "RGBA";
			break;
		default:
			throw MediaFormatException("unsupported PNG color type");
		}
		MediaValidateMIME(file, "image/png");
	} else if (byte(signature, 0) == 0xff && byte(signature, 1) == 0xd8) {
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
				result = {be16(sof, 3), be16(sof, 1), "JPEG", components == 1 ? "L" : components == 4 ? "CMYK" : "RGB"};
				break;
			}
			offset += length;
		}
		MediaValidateMIME(file, "image/jpeg");
	} else {
		throw MediaFormatException("native image supports PNG and JPEG encoded files");
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
		                  ? MediaPositive(args.data[2].GetValue(row), "max_pixels", MEDIA_MAX_PIXELS)
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
		if (has_mode && mode != "L" && mode != "LA" && mode != "RGB" && mode != "RGBA") {
			throw InvalidInputException("native image mode must be L, LA, RGB, or RGBA");
		}
		auto on_error = args.ColumnCount() >= 3 ? args.data[2].GetValue(row).GetValue<string>() : "raise";
		if (on_error != "raise" && on_error != "null") {
			throw InvalidInputException("on_error must be raise or null");
		}
		auto input_bytes = args.ColumnCount() == 6
		                       ? MediaPositive(args.data[3].GetValue(row), "max_input_bytes", 4 * 1024 * MEDIA_MIB)
		                       : 256 * MEDIA_MIB;
		auto pixels = args.ColumnCount() == 6
		                  ? MediaPositive(args.data[4].GetValue(row), "max_pixels", MEDIA_MAX_PIXELS)
		                  : MEDIA_MAX_PIXELS;
		auto output_bytes = args.ColumnCount() == 6
		                        ? MediaPositive(args.data[5].GetValue(row), "max_decoded_bytes", MEDIA_MAX_FRAME_BYTES)
		                        : MEDIA_MAX_FRAME_BYTES;
		auto &context = state.GetContext();
		try {
			auto file = FileReference::FromValue(value, "native_decode_image_file");
			auto resolved = ResolvedFile::Open(context, file);
			if (resolved->LogicalSize() > input_bytes) {
				throw OutOfRangeException("native image exceeds max_input_bytes");
			}
			auto header = ReadHeader(context, *resolved, file, MinValue<uint64_t>(input_bytes, 64 * MEDIA_MIB), pixels);
			// Enforce the decoder's conservative budget before FFmpeg can report
			// an internal pixel-limit rejection as an encoded-format error.
			MediaProduct(uint64_t(header.width) * header.height, 8, output_bytes, "decoded frame bytes");
			MediaReader reader(context, file, std::move(resolved), AVMEDIA_TYPE_VIDEO, input_bytes, input_bytes * 4,
			                   pixels, output_bytes);
			if (!reader.NextFrame()) {
				throw MediaFormatException("image has no decodable frame");
			}
			auto &frame = reader.Frame();
			if (frame.width != int64_t(header.width) || frame.height != int64_t(header.height)) {
				throw MediaFormatException("decoded dimensions differ from image header");
			}
			if (mode.empty()) {
				auto descriptor = av_pix_fmt_desc_get(AVPixelFormat(frame.format));
				mode = (descriptor && (descriptor->flags & AV_PIX_FMT_FLAG_ALPHA)) || header.mode == "P" ? "RGBA"
				       : header.mode == "L"                                                              ? "L"
				                                                                                         : "RGB";
			}
			auto bytes =
			    MediaProduct(uint64_t(header.width) * header.height, ImageLogicalType::ChannelsForMode(mode),
			                 MinValue<uint64_t>(output_bytes, MEDIA_BATCH_BYTES - batch_bytes), "image batch bytes");
			// Charge the allocation even if pixel conversion later fails under on_error=null.
			batch_bytes += bytes;
			MediaWriteImage(context, frame, mode, header.width, header.height, result, row, bytes);
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
