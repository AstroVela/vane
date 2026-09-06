// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// file_value.cpp
//
//===----------------------------------------------------------------------===//

#include "file_value.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"
#include "mbedtls_wrapper.hpp"

namespace duckdb {

namespace {

static Value OptionalStringValue(bool has_value, const string &value) {
	return has_value ? Value(value) : Value(LogicalType::VARCHAR);
}

static Value OptionalBigintValue(bool has_value, int64_t value) {
	return has_value ? Value::BIGINT(value) : Value(LogicalType::BIGINT);
}

static void AppendBigEndianUInt64(string &target, uint64_t value) {
	for (idx_t byte_index = 0; byte_index < sizeof(uint64_t); byte_index++) {
		auto shift = NumericCast<uint8_t>((sizeof(uint64_t) - byte_index - 1) * 8);
		target.push_back(static_cast<char>((value >> shift) & 0xff));
	}
}

static string Sha256Hex(const string &input) {
	duckdb_mbedtls::MbedTlsWrapper::SHA256State state;
	state.AddBytes(const_data_ptr_cast(input.data()), input.size());
	string result(duckdb_mbedtls::MbedTlsWrapper::SHA256_HASH_LENGTH_TEXT, '\0');
	state.FinishHex(&result[0]);
	return result;
}

} // namespace

FileReference FileReference::FromFields(const Value &url_value, const Value &content_type_value,
                                        const Value &position_value, const Value &size_value,
                                        const Value &checksum_value, const string &function_name,
                                        FileMediaType media_type) {
	FileReference result;
	result.media_type = media_type;
	string url;
	const string *url_ptr = nullptr;
	if (!url_value.IsNull()) {
		url = url_value.GetValue<string>();
		url_ptr = &url;
	}
	auto has_position = !position_value.IsNull();
	auto has_size = !size_value.IsNull();
	auto position = has_position ? position_value.GetValue<int64_t>() : 0;
	auto size = has_size ? size_value.GetValue<int64_t>() : 0;
	string checksum;
	const string *checksum_ptr = nullptr;
	if (!checksum_value.IsNull()) {
		checksum = checksum_value.GetValue<string>();
		checksum_ptr = &checksum;
	}
	ValidateFields(url_ptr, has_position, position, has_size, size, checksum_ptr, function_name);

	result.url = std::move(url);
	if (!content_type_value.IsNull()) {
		result.has_content_type = true;
		result.content_type = content_type_value.GetValue<string>();
	}
	if (has_position) {
		result.has_range = true;
		result.position = position;
		result.size = size;
	}
	if (checksum_ptr) {
		result.has_checksum = true;
		result.checksum = std::move(checksum);
	}
	return result;
}

FileReference FileReference::FromValue(const Value &value, const string &function_name) {
	if (value.IsNull()) {
		throw InternalException("FileReference::FromValue called with NULL");
	}
	if (!FileLogicalType::IsFile(value.type())) {
		throw InvalidInputException("%s() requires an exact FILE-family value", function_name);
	}
	const auto &children = StructValue::GetChildren(value);
	if (children.size() != FileLogicalType::FIELD_COUNT) {
		throw InvalidInputException("%s() received a malformed FILE value", function_name);
	}
	return FromFields(children[FileLogicalType::URL], children[FileLogicalType::CONTENT_TYPE],
	                  children[FileLogicalType::POSITION], children[FileLogicalType::SIZE],
	                  children[FileLogicalType::CHECKSUM], function_name, FileLogicalType::GetMediaType(value.type()));
}

void FileReference::ValidateFields(const string *url, bool has_position, int64_t position, bool has_size, int64_t size,
                                   const string *checksum, const string &function_name) {
	FileLogicalType::ValidateFields(url, has_position, position, has_size, size, checksum, function_name);
}

void FileReference::Validate(const string &function_name) const {
	ValidateFields(&url, has_range, position, has_range, size, has_checksum ? &checksum : nullptr, function_name);
}

Value FileReference::ToValue() const {
	Validate("FILE output");
	vector<Value> fields;
	fields.reserve(FileLogicalType::FIELD_COUNT);
	fields.emplace_back(url);
	fields.push_back(OptionalStringValue(has_content_type, content_type));
	fields.push_back(OptionalBigintValue(has_range, position));
	fields.push_back(OptionalBigintValue(has_range, size));
	fields.push_back(OptionalStringValue(has_checksum, checksum));
	return Value::STRUCT(FileLogicalType::Create(media_type), std::move(fields));
}

string FileIdentity::LocatorId(const FileReference &file) {
	file.Validate("file_locator_id");
	string canonical = "vane:file-locator:v1";
	canonical.push_back('\0');
	AppendBigEndianUInt64(canonical, file.url.size());
	canonical.append(file.url);
	canonical.push_back(file.has_range ? '\1' : '\0');
	if (file.has_range) {
		AppendBigEndianUInt64(canonical, NumericCast<uint64_t>(file.position));
		AppendBigEndianUInt64(canonical, NumericCast<uint64_t>(file.size));
	}
	return "file-locator-v1:sha256:" + Sha256Hex(canonical);
}

string FileIdentity::NormalizeChecksum(const string &checksum) {
	auto separator = checksum.find(':');
	D_ASSERT(separator != string::npos);
	return StringUtil::Lower(checksum.substr(0, separator)) + checksum.substr(separator);
}

Value FileIdentity::ContentId(const FileReference &file) {
	file.Validate("file_content_id");
	if (file.has_range && file.size == 0) {
		return Value("file-content-v1:empty");
	}
	if (!file.has_checksum) {
		return Value(LogicalType::VARCHAR);
	}
	return Value("file-content-v1:checksum:" + NormalizeChecksum(file.checksum));
}

} // namespace duckdb
