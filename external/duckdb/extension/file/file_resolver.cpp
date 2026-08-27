// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// file_resolver.cpp
//
//===----------------------------------------------------------------------===//

#include "file_resolver.hpp"

#include "file_mime_type.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/opener_file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "duckdb/main/client_context.hpp"
#include "mbedtls_wrapper.hpp"

#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace duckdb {

namespace {

static Value OptionalStringValue(bool has_value, const string &value) {
	return has_value ? Value(value) : Value(LogicalType::VARCHAR);
}

#ifndef _WIN32
static bool IsNativeLocalNonRegularPath(FileSystem &file_system, const string &path) {
	auto scheme_separator = path.find("://");
	if (scheme_separator != string::npos && !StringUtil::StartsWith(StringUtil::Lower(path), "file://")) {
		return false;
	}
	auto local_path = file_system.ExpandPath(path);
	struct stat status;
	return stat(local_path.c_str(), &status) == 0 && !S_ISREG(status.st_mode);
}
#endif

static bool TryGetMetadataString(const FileMetadata &metadata, const string &name, string &result) {
	auto entry = metadata.extended_file_info.find(name);
	if (entry == metadata.extended_file_info.end() || entry->second.IsNull() ||
	    entry->second.type().id() == LogicalTypeId::BLOB) {
		return false;
	}
	result = entry->second.DefaultCastAs(LogicalType::VARCHAR).GetValue<string>();
	return !result.empty();
}

} // namespace

LogicalType FileStatValue::Type() {
	child_list_t<LogicalType> fields;
	fields.emplace_back("url", LogicalType::VARCHAR);
	fields.emplace_back("object_size", LogicalType::UBIGINT);
	fields.emplace_back("last_modified", LogicalType::TIMESTAMP);
	fields.emplace_back("version", LogicalType::VARCHAR);
	fields.emplace_back("etag", LogicalType::VARCHAR);
	fields.emplace_back("content_type", LogicalType::VARCHAR);
	return LogicalType::STRUCT(std::move(fields));
}

Value FileStatValue::ToValue() const {
	vector<Value> fields;
	fields.reserve(6);
	fields.emplace_back(url);
	fields.push_back(has_object_size ? Value::UBIGINT(object_size) : Value(LogicalType::UBIGINT));
	fields.push_back(has_last_modified ? Value::TIMESTAMP(last_modified) : Value(LogicalType::TIMESTAMP));
	fields.push_back(OptionalStringValue(has_version, version));
	fields.push_back(OptionalStringValue(has_etag, etag));
	fields.push_back(OptionalStringValue(has_content_type, content_type));
	return Value::STRUCT(Type(), std::move(fields));
}

ResolvedFile::ResolvedFile(ClientContext &context_p, unique_ptr<FileHandle> handle_p, FileReference file_p,
                           uint64_t object_size_p, uint64_t logical_position_p, uint64_t logical_size_p)
    : context(context_p), handle(std::move(handle_p)), file(std::move(file_p)), object_size(object_size_p),
      logical_position(logical_position_p), logical_size(logical_size_p) {
}

unique_ptr<ResolvedFile> ResolvedFile::Open(ClientContext &context, const FileReference &file) {
	file.Validate("FILE resolver");
	if (context.IsInterrupted()) {
		throw InterruptException();
	}
	auto &file_system = FileSystem::GetFileSystem(context);
#ifdef _WIN32
	file_system.Cast<OpenerFileSystem>().VerifyCanAccessFile(file.url);
	if (file_system.IsPipe(file.url)) {
		throw IOException({{"file_kind", "not_regular"}}, "FILE resolver requires a regular file");
	}
#endif
	auto flags = FileFlags::FILE_FLAGS_READ | FileFlags::FILE_FLAGS_PARALLEL_ACCESS | FileFlags::FILE_FLAGS_NONBLOCKING;
	unique_ptr<FileHandle> handle;
	try {
		handle = file_system.OpenFile(file.url, flags);
	} catch (const IOException &) {
#ifndef _WIN32
		if (IsNativeLocalNonRegularPath(file_system, file.url)) {
			throw IOException({{"file_kind", "not_regular"}}, "FILE resolver requires a regular file");
		}
#endif
		throw;
	}
	if (!handle) {
		throw IOException("FILE resolver could not open the requested file");
	}
	auto file_type = handle->file_system.GetFileType(*handle);
	// VirtualFileSystem wraps native FIFOs in PipeFileSystem, whose generic file type is INVALID.
	// Consult the opened handle in that case so the wrapper cannot be mistaken for an untyped remote file.
	if (file_type != FileType::FILE_TYPE_REGULAR && (file_type != FileType::FILE_TYPE_INVALID || handle->IsPipe())) {
		throw IOException({{"file_kind", "not_regular"}}, "FILE resolver requires a regular file");
	}
	auto signed_object_size = handle->file_system.GetFileSize(*handle);
	if (signed_object_size < 0) {
		throw IOException("FILE resolver could not determine the object size");
	}
	auto object_size = NumericCast<uint64_t>(signed_object_size);
	auto logical_position = file.has_range ? NumericCast<uint64_t>(file.position) : 0;
	auto logical_size = file.has_range ? NumericCast<uint64_t>(file.size) : object_size;
	if (logical_position > object_size || logical_size > object_size - logical_position) {
		throw IOException({{"file_range", "out_of_bounds"}},
		                  "FILE byte range [%d, %d) exceeds the backing object size %d", logical_position,
		                  logical_position + logical_size, object_size);
	}
	return unique_ptr<ResolvedFile>(
	    new ResolvedFile(context, std::move(handle), file, object_size, logical_position, logical_size));
}

uint64_t ResolvedFile::ObjectSize() const {
	return object_size;
}

uint64_t ResolvedFile::LogicalSize() const {
	return logical_size;
}

FileStatValue ResolvedFile::Stat() const {
	FileStatValue result;
	result.url = file.url;
	result.has_object_size = true;
	result.object_size = object_size;
	auto is_local_file = handle->file_system.IsLocalFileSystem();
	FileMetadata metadata;
	bool has_metadata = false;

	try {
		metadata = handle->file_system.Stats(*handle);
		has_metadata = true;
	} catch (const NotImplementedException &) {
	}

	if (has_metadata) {
		auto last_modified = metadata.last_modification_time;
		if (Timestamp::IsFinite(last_modified)) {
			result.has_last_modified = true;
			result.last_modified = last_modified;
		}
		result.has_version = TryGetMetadataString(metadata, "version", result.version) ||
		                     TryGetMetadataString(metadata, "version_id", result.version);
		result.has_etag = TryGetMetadataString(metadata, "etag", result.etag);
	}

	if (!result.has_last_modified) {
		try {
			auto last_modified = handle->file_system.GetLastModifiedTime(*handle);
			if (Timestamp::IsFinite(last_modified) && !(!is_local_file && last_modified == timestamp_t())) {
				result.has_last_modified = true;
				result.last_modified = last_modified;
			}
		} catch (const NotImplementedException &) {
		}
	}

	if (!is_local_file && !result.has_version && !result.has_etag) {
		try {
			auto version_tag = handle->file_system.GetVersionTag(*handle);
			if (!version_tag.empty()) {
				result.has_etag = true;
				result.etag = std::move(version_tag);
			}
		} catch (const NotImplementedException &) {
		}
	}

	if (file.has_content_type) {
		result.has_content_type = true;
		result.content_type = file.content_type;
	} else if (has_metadata && (TryGetMetadataString(metadata, "content_type", result.content_type) ||
	                            TryGetMetadataString(metadata, "mime_type", result.content_type))) {
		result.has_content_type = true;
	} else {
		result.has_content_type = FileMimeType::FromPath(file.url, result.content_type);
	}
	return result;
}

bool ResolvedFile::MimeTypeFromResolvedMetadata(string &result) const {
	if (file.has_content_type) {
		result = file.content_type;
		return true;
	}
	try {
		auto metadata = handle->file_system.Stats(*handle);
		if (TryGetMetadataString(metadata, "content_type", result) ||
		    TryGetMetadataString(metadata, "mime_type", result)) {
			return true;
		}
	} catch (const NotImplementedException &) {
	}
	return FileMimeType::FromPath(file.url, result);
}

void ResolvedFile::ReadExact(data_ptr_t target, uint64_t size, uint64_t logical_offset) const {
	if (logical_offset > logical_size || size > logical_size - logical_offset) {
		throw OutOfRangeException("FILE read [%d, %d) exceeds the logical view size %d", logical_offset,
		                          logical_offset + size, logical_size);
	}
	if (size == 0) {
		return;
	}
	if (context.IsInterrupted()) {
		throw InterruptException();
	}
	auto read_size = NumericCast<idx_t>(size);
	auto absolute_offset = NumericCast<idx_t>(logical_position + logical_offset);
	try {
		handle->Read(QueryContext(context), target, read_size, absolute_offset);
		if (context.IsInterrupted()) {
			throw InterruptException();
		}
		return;
	} catch (const NotImplementedException &) {
		// Some registered file systems only implement seek plus sequential reads.
	}

	if (context.IsInterrupted()) {
		throw InterruptException();
	}
	handle->Seek(absolute_offset);
	idx_t total_read = 0;
	while (total_read < read_size) {
		if (context.IsInterrupted()) {
			throw InterruptException();
		}
		auto remaining = read_size - total_read;
		auto bytes_read = handle->Read(QueryContext(context), target + total_read, remaining);
		if (bytes_read <= 0) {
			throw IOException("FILE resolver encountered a short read: expected %d bytes, received %d", read_size,
			                  total_read);
		}
		auto read_count = NumericCast<idx_t>(bytes_read);
		if (read_count > remaining) {
			throw IOException("FILE resolver received %d bytes after requesting at most %d", read_count, remaining);
		}
		total_read += read_count;
	}
}

string ResolvedFile::Sha256() const {
	duckdb_mbedtls::MbedTlsWrapper::SHA256State state;
	static constexpr idx_t BUFFER_SIZE = 1024 * 1024;
	vector<data_t> buffer(MinValue<uint64_t>(logical_size, BUFFER_SIZE));
	uint64_t offset = 0;
	while (offset < logical_size) {
		if (context.IsInterrupted()) {
			throw InterruptException();
		}
		auto next_size = MinValue<uint64_t>(logical_size - offset, buffer.size());
		ReadExact(buffer.data(), next_size, offset);
		if (context.IsInterrupted()) {
			throw InterruptException();
		}
		state.AddBytes(buffer.data(), next_size);
		offset += next_size;
	}
	string result(duckdb_mbedtls::MbedTlsWrapper::SHA256_HASH_LENGTH_TEXT, '\0');
	state.FinishHex(&result[0]);
	return result;
}

bool ResolvedFile::GuessMimeType(string &result, idx_t maximum_bytes) const {
	auto probe_size = MinValue<uint64_t>(logical_size, maximum_bytes);
	if (probe_size == 0) {
		return false;
	}
	vector<data_t> prefix(NumericCast<idx_t>(probe_size));
	ReadExact(prefix.data(), probe_size);
	if (FileMimeType::FromBytes(prefix.data(), prefix.size(), result, probe_size == logical_size)) {
		return true;
	}

	static constexpr idx_t HDF5_SIGNATURE_SIZE = 8;
	vector<data_t> hdf5_signature(HDF5_SIGNATURE_SIZE);
	for (uint64_t offset = 0; offset <= logical_size && HDF5_SIGNATURE_SIZE <= logical_size - offset;) {
		// FromBytes already inspected every legal offset wholly contained in the
		// prefix, so only the remaining offsets require another range read.
		if (offset + HDF5_SIGNATURE_SIZE > prefix.size()) {
			ReadExact(hdf5_signature.data(), HDF5_SIGNATURE_SIZE, offset);
			if (FileMimeType::IsHdf5Signature(hdf5_signature.data(), hdf5_signature.size())) {
				result = "application/vnd.hdfgroup.hdf5";
				return true;
			}
		}
		if (offset == 0) {
			offset = 512;
		} else {
			if (offset > logical_size / 2) {
				break;
			}
			offset *= 2;
		}
	}
	return false;
}

} // namespace duckdb
