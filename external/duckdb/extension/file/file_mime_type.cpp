// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// file_mime_type.cpp
//
//===----------------------------------------------------------------------===//

#include "file_mime_type.hpp"

#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"

#include <cstring>

namespace duckdb {

namespace {

struct ExtensionMimeType {
	const char *extension;
	const char *mime_type;
};

struct IsoBmffMimeCandidate {
	const char *mime_type = nullptr;
	uint8_t priority = 0;
};

static bool HasBytes(const_data_ptr_t data, idx_t size, idx_t offset, const char *expected, idx_t expected_size) {
	return offset <= size && expected_size <= size - offset && memcmp(data + offset, expected, expected_size) == 0;
}

static bool SetMimeType(string &result, const char *mime_type) {
	result = mime_type;
	return true;
}

static uint16_t ReadLittleEndianUInt16(const_data_ptr_t data) {
	return static_cast<uint16_t>(data[0]) | (static_cast<uint16_t>(data[1]) << 8);
}

static uint32_t ReadLittleEndianUInt32(const_data_ptr_t data) {
	return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) |
	       (static_cast<uint32_t>(data[2]) << 16) | (static_cast<uint32_t>(data[3]) << 24);
}

static uint32_t ReadBigEndianUInt32(const_data_ptr_t data) {
	return (static_cast<uint32_t>(data[0]) << 24) | (static_cast<uint32_t>(data[1]) << 16) |
	       (static_cast<uint32_t>(data[2]) << 8) | static_cast<uint32_t>(data[3]);
}

static uint64_t ReadBigEndianUInt64(const_data_ptr_t data) {
	return (static_cast<uint64_t>(ReadBigEndianUInt32(data)) << 32) | ReadBigEndianUInt32(data + 4);
}

static bool IsGifHeader(const_data_ptr_t data, idx_t size) {
	static constexpr idx_t LOGICAL_SCREEN_DESCRIPTOR_END = 13;
	return size >= LOGICAL_SCREEN_DESCRIPTOR_END &&
	       (HasBytes(data, size, 0, "GIF87a", 6) || HasBytes(data, size, 0, "GIF89a", 6)) &&
	       ReadLittleEndianUInt16(data + 6) != 0 && ReadLittleEndianUInt16(data + 8) != 0;
}

static bool IsRiffForm(const_data_ptr_t data, idx_t size, const char *form_type, uint32_t minimum_payload_size = 4) {
	return size >= 12 && HasBytes(data, size, 0, "RIFF", 4) &&
	       ReadLittleEndianUInt32(data + 4) >= minimum_payload_size && HasBytes(data, size, 8, form_type, 4);
}

static bool IsWebpHeader(const_data_ptr_t data, idx_t size) {
	return size >= 16 && IsRiffForm(data, size, "WEBP", 12) &&
	       (HasBytes(data, size, 12, "VP8 ", 4) || HasBytes(data, size, 12, "VP8L", 4) ||
	        HasBytes(data, size, 12, "VP8X", 4));
}

static bool IsOggPageHeader(const_data_ptr_t data, idx_t size) {
	static constexpr idx_t FIXED_HEADER_SIZE = 27;
	if (size < FIXED_HEADER_SIZE || !HasBytes(data, size, 0, "OggS", 4) || data[4] != 0 || (data[5] & 0xf8) != 0) {
		return false;
	}
	return data[26] <= size - FIXED_HEADER_SIZE;
}

static bool IsMpegAudioFrameHeader(const_data_ptr_t data, idx_t size) {
	if (size < 4 || data[0] != 0xff || (data[1] & 0xe0) != 0xe0) {
		return false;
	}
	auto version = (data[1] >> 3) & 0x03;
	auto layer = (data[1] >> 1) & 0x03;
	auto bitrate = (data[2] >> 4) & 0x0f;
	auto sample_rate = (data[2] >> 2) & 0x03;
	auto emphasis = data[3] & 0x03;
	return version != 0x01 && layer != 0x00 && bitrate != 0x00 && bitrate != 0x0f && sample_rate != 0x03 &&
	       emphasis != 0x02;
}

static bool IsId3v2Header(const_data_ptr_t data, idx_t size, bool complete_input) {
	if (size < 10 || !HasBytes(data, size, 0, "ID3", 3)) {
		return false;
	}
	auto major_version = data[3];
	auto revision = data[4];
	auto flags = data[5];
	if (major_version < 2 || major_version > 4 || revision == 0xff) {
		return false;
	}

	static constexpr uint8_t RESERVED_FLAGS[] = {0, 0, 0x3f, 0x1f, 0x0f};
	if ((flags & RESERVED_FLAGS[major_version]) != 0) {
		return false;
	}

	uint64_t payload_size = 0;
	for (idx_t offset = 6; offset < 10; offset++) {
		if ((data[offset] & 0x80) != 0) {
			return false;
		}
		payload_size = (payload_size << 7) | data[offset];
	}
	if (payload_size == 0) {
		return false;
	}
	auto footer_size = major_version == 4 && (flags & 0x10) != 0 ? 10 : 0;
	return !complete_input || payload_size + footer_size <= size - 10;
}

static bool IsPdfHeader(const_data_ptr_t data, idx_t size) {
	return size >= 8 && HasBytes(data, size, 0, "%PDF-", 5) && data[5] >= '0' && data[5] <= '9' && data[6] == '.' &&
	       data[7] >= '0' && data[7] <= '9';
}

static bool IsZipHeader(const_data_ptr_t data, idx_t size) {
	if (HasBytes(data, size, 0, "PK\x03\x04", 4)) {
		return size >= 30;
	}
	if (HasBytes(data, size, 0, "PK\x05\x06", 4)) {
		return size >= 22;
	}
	return size >= 12 && HasBytes(data, size, 0, "PK\x07\x08", 4) && HasBytes(data, size, 8, "PK\x03\x04", 4);
}

static IsoBmffMimeCandidate MimeCandidateFromIsoBmffBrand(const_data_ptr_t brand) {
	if (HasBytes(brand, 4, 0, "avif", 4)) {
		return {"image/avif", 3};
	}
	if (HasBytes(brand, 4, 0, "avis", 4)) {
		return {"image/avif-sequence", 3};
	}
	if (HasBytes(brand, 4, 0, "heic", 4) || HasBytes(brand, 4, 0, "heix", 4) || HasBytes(brand, 4, 0, "hevc", 4) ||
	    HasBytes(brand, 4, 0, "hevx", 4)) {
		return {"image/heic", 3};
	}
	if (HasBytes(brand, 4, 0, "mif1", 4)) {
		return {"image/heif", 2};
	}
	if (HasBytes(brand, 4, 0, "msf1", 4)) {
		return {"image/heif-sequence", 2};
	}
	if (HasBytes(brand, 4, 0, "M4A ", 4) || HasBytes(brand, 4, 0, "M4B ", 4)) {
		return {"audio/mp4", 3};
	}
	if (HasBytes(brand, 4, 0, "qt  ", 4)) {
		return {"video/quicktime", 3};
	}
	return {};
}

static void ConsiderIsoBmffBrand(const_data_ptr_t brand, IsoBmffMimeCandidate &best_candidate) {
	auto candidate = MimeCandidateFromIsoBmffBrand(brand);
	// The first equally specific brand wins. A concrete compatible brand still
	// overrides a generic HEIF brand such as mif1 or msf1.
	if (candidate.priority > best_candidate.priority) {
		best_candidate = candidate;
	}
}

static bool MimeTypeFromIsoBmff(const_data_ptr_t data, idx_t size, string &result) {
	if (size < 16 || !HasBytes(data, size, 4, "ftyp", 4)) {
		return false;
	}

	uint64_t box_size = ReadBigEndianUInt32(data);
	idx_t brand_offset = 8;
	idx_t compatible_brand_offset = 16;
	if (box_size == 1) {
		if (size < 24) {
			return false;
		}
		box_size = ReadBigEndianUInt64(data + 8);
		brand_offset = 16;
		compatible_brand_offset = 24;
	} else if (box_size == 0) {
		box_size = size;
	}
	if (box_size < compatible_brand_offset || (box_size - compatible_brand_offset) % 4 != 0) {
		return false;
	}

	IsoBmffMimeCandidate best_candidate;
	ConsiderIsoBmffBrand(data + brand_offset, best_candidate);
	auto available_box_size = NumericCast<idx_t>(MinValue<uint64_t>(box_size, size));
	for (idx_t offset = compatible_brand_offset; offset + 4 <= available_box_size; offset += 4) {
		ConsiderIsoBmffBrand(data + offset, best_candidate);
	}
	// Daft treats an otherwise unknown ftyp box as MP4. Keep that best-effort
	// fallback after Vane's image/audio-specific routing has inspected all
	// available compatible brands.
	return SetMimeType(result, best_candidate.mime_type ? best_candidate.mime_type : "video/mp4");
}

static bool IsHtmlPrefix(const_data_ptr_t data, idx_t size) {
	return HasBytes(data, size, 0, "<!DOCTYPE", 9) || HasBytes(data, size, 0, "<html", 5) ||
	       HasBytes(data, size, 0, "<HTML", 5);
}

static bool IsUriStyleLocator(const string &path) {
	auto separator = path.find("://");
	if (separator == string::npos || separator == 0 || !StringUtil::CharacterIsAlpha(path[0])) {
		return false;
	}
#ifdef _WIN32
	if (separator == 1) {
		return false;
	}
#endif
	for (idx_t index = 1; index < separator; index++) {
		auto character = path[index];
		if (!StringUtil::CharacterIsAlpha(character) && !StringUtil::CharacterIsDigit(character) && character != '+' &&
		    character != '-' && character != '.') {
			return false;
		}
	}
	return StringUtil::Lower(path.substr(0, separator)) != "file";
}

static string ExtensionFromPath(const string &path) {
	auto is_uri = IsUriStyleLocator(path);
	auto scheme_separator = is_uri ? path.find("://") : string::npos;
	auto scheme = is_uri ? StringUtil::Lower(path.substr(0, scheme_separator)) : string();
	auto delimiter = is_uri ? path.find_first_of("?#", scheme_separator + 3) : string::npos;
	if (delimiter != string::npos && scheme != "http" && scheme != "https") {
		return string();
	}
	auto clean_path = path.substr(0, delimiter);
	if (is_uri && clean_path.find('/', scheme_separator + 3) == string::npos) {
		return string();
	}
	auto slash = clean_path.find_last_of("/\\");
	auto dot = clean_path.find_last_of('.');
	if (dot == string::npos || (slash != string::npos && dot < slash) || dot + 1 == clean_path.size()) {
		return string();
	}
	return StringUtil::Lower(clean_path.substr(dot + 1));
}

} // namespace

bool FileMimeType::FromPath(const string &path, string &result) {
	static constexpr ExtensionMimeType MAPPINGS[] = {
	    {"txt", "text/plain"},
	    {"text", "text/plain"},
	    {"log", "text/plain"},
	    {"csv", "text/csv"},
	    {"tsv", "text/tab-separated-values"},
	    {"md", "text/markdown"},
	    {"html", "text/html"},
	    {"htm", "text/html"},
	    {"json", "application/json"},
	    {"jsonl", "application/x-ndjson"},
	    {"ndjson", "application/x-ndjson"},
	    {"xml", "application/xml"},
	    {"pdf", "application/pdf"},
	    {"zip", "application/zip"},
	    {"gz", "application/gzip"},
	    {"gzip", "application/gzip"},
	    {"png", "image/png"},
	    {"jpg", "image/jpeg"},
	    {"jpeg", "image/jpeg"},
	    {"gif", "image/gif"},
	    {"webp", "image/webp"},
	    {"bmp", "image/bmp"},
	    {"tif", "image/tiff"},
	    {"tiff", "image/tiff"},
	    {"avif", "image/avif"},
	    {"avifs", "image/avif-sequence"},
	    {"heic", "image/heic"},
	    {"heif", "image/heif"},
	    {"heifs", "image/heif-sequence"},
	    {"mp3", "audio/mpeg"},
	    {"wav", "audio/wav"},
	    {"flac", "audio/flac"},
	    {"aac", "audio/aac"},
	    {"ogg", "audio/ogg"},
	    {"oga", "audio/ogg"},
	    {"opus", "audio/ogg"},
	    {"m4a", "audio/mp4"},
	    {"mp4", "video/mp4"},
	    {"m4v", "video/mp4"},
	    {"mov", "video/quicktime"},
	    {"webm", "video/webm"},
	    {"mkv", "video/x-matroska"},
	    {"avi", "video/x-msvideo"},
	    {"h5", "application/vnd.hdfgroup.hdf5"},
	    {"hdf5", "application/vnd.hdfgroup.hdf5"},
	    {"he5", "application/vnd.hdfgroup.hdf5"},
	};
	auto extension = ExtensionFromPath(path);
	for (auto &mapping : MAPPINGS) {
		if (extension == mapping.extension) {
			return SetMimeType(result, mapping.mime_type);
		}
	}
	return false;
}

bool FileMimeType::IsHdf5Signature(const_data_ptr_t data, idx_t size) {
	static constexpr char HDF5[] = "\x89HDF\r\n\x1a\n";
	return HasBytes(data, size, 0, HDF5, sizeof(HDF5) - 1);
}

bool FileMimeType::FromBytes(const_data_ptr_t data, idx_t size, string &result, bool complete_input) {
	// This is deliberately a bounded, best-effort magic-byte sniffer, not a
	// decoder or a structural validator. Consumers must validate the file when
	// they decode it. The compact rule set follows Daft's file MIME semantics,
	// with Vane-specific ISO-BMFF routing and HDF5 user-block support.
	static constexpr char PNG[] = "\x89PNG\r\n\x1a\n";
	if (HasBytes(data, size, 0, PNG, sizeof(PNG) - 1)) {
		return SetMimeType(result, "image/png");
	}
	if (size >= 3 && data[0] == 0xff && data[1] == 0xd8 && data[2] == 0xff) {
		return SetMimeType(result, "image/jpeg");
	}
	if (IsGifHeader(data, size)) {
		return SetMimeType(result, "image/gif");
	}
	if (IsWebpHeader(data, size)) {
		return SetMimeType(result, "image/webp");
	}
	if (MimeTypeFromIsoBmff(data, size, result)) {
		return true;
	}
	if (IsId3v2Header(data, size, complete_input) || IsMpegAudioFrameHeader(data, size)) {
		return SetMimeType(result, "audio/mpeg");
	}
	if (IsRiffForm(data, size, "WAVE")) {
		return SetMimeType(result, "audio/wav");
	}
	if (IsOggPageHeader(data, size)) {
		return SetMimeType(result, "audio/ogg");
	}
	if (HasBytes(data, size, 0, "\x00\x00\x01\xba", 4)) {
		return SetMimeType(result, "video/mpeg");
	}
	static constexpr idx_t HDF5_SIGNATURE_SIZE = 8;
	for (idx_t offset = 0; offset + HDF5_SIGNATURE_SIZE <= size; offset = offset == 0 ? 512 : offset * 2) {
		if (IsHdf5Signature(data + offset, size - offset)) {
			return SetMimeType(result, "application/vnd.hdfgroup.hdf5");
		}
		if (offset > size / 2) {
			break;
		}
	}
	if (IsPdfHeader(data, size)) {
		return SetMimeType(result, "application/pdf");
	}
	if (IsZipHeader(data, size)) {
		return SetMimeType(result, "application/zip");
	}
	if (IsHtmlPrefix(data, size)) {
		return SetMimeType(result, "text/html");
	}
	return false;
}

} // namespace duckdb
