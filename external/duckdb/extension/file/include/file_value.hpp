// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// file_value.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types/value.hpp"

namespace duckdb {

//! A validated, extension-owned view of the five fields in a FILE value.
struct FileReference {
	FileMediaType media_type = FileMediaType::UNKNOWN;
	string url;
	string content_type;
	int64_t position = 0;
	int64_t size = 0;
	string checksum;
	bool has_content_type = false;
	bool has_range = false;
	bool has_checksum = false;

	static FileReference FromFields(const Value &url, const Value &content_type, const Value &position,
	                                const Value &size, const Value &checksum, const string &function_name,
	                                FileMediaType media_type = FileMediaType::UNKNOWN);
	static FileReference FromValue(const Value &value, const string &function_name);
	static void ValidateFields(const string *url, bool has_position, int64_t position, bool has_size, int64_t size,
	                           const string *checksum, const string &function_name);

	void Validate(const string &function_name) const;
	Value ToValue() const;
};

struct FileIdentity {
	static string LocatorId(const FileReference &file);
	static string NormalizeChecksum(const string &checksum);
	static Value ContentId(const FileReference &file);
};

} // namespace duckdb
