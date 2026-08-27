// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// file_resolver.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "file_value.hpp"

#include "duckdb/common/file_system.hpp"

namespace duckdb {

class ClientContext;

struct FileStatValue {
	string url;
	uint64_t object_size = 0;
	timestamp_t last_modified;
	string version;
	string etag;
	string content_type;
	bool has_object_size = false;
	bool has_last_modified = false;
	bool has_version = false;
	bool has_etag = false;
	bool has_content_type = false;

	static LogicalType Type();
	Value ToValue() const;
};

class ResolvedFile {
public:
	static unique_ptr<ResolvedFile> Open(ClientContext &context, const FileReference &file);

	uint64_t ObjectSize() const;
	uint64_t LogicalSize() const;
	FileStatValue Stat() const;
	bool MimeTypeFromResolvedMetadata(string &result) const;
	void ReadExact(data_ptr_t target, uint64_t size, uint64_t logical_offset = 0) const;
	string Sha256() const;
	bool GuessMimeType(string &result, idx_t maximum_bytes = 4096 + 8) const;

private:
	ResolvedFile(ClientContext &context, unique_ptr<FileHandle> handle, FileReference file, uint64_t object_size,
	             uint64_t logical_position, uint64_t logical_size);

	ClientContext &context;
	unique_ptr<FileHandle> handle;
	FileReference file;
	uint64_t object_size;
	uint64_t logical_position;
	uint64_t logical_size;
};

} // namespace duckdb
