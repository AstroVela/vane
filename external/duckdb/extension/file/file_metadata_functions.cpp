// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// file_metadata_functions.cpp
//
//===----------------------------------------------------------------------===//

#include "file_metadata_functions.hpp"

#include "file_mime_type.hpp"
#include "file_resolver.hpp"
#include "file_value.hpp"

#include "duckdb/common/error_data.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"

#include <cerrno>

namespace duckdb {

namespace {

static Value NullValue(const LogicalType &type) {
	return Value(type);
}

static bool TryGetFile(DataChunk &args, idx_t argument, idx_t row, const string &function_name, FileReference &result) {
	auto value = args.data[argument].GetValue(row);
	if (value.IsNull()) {
		return false;
	}
	result = FileReference::FromValue(value, function_name);
	return true;
}

static bool IsRecoverableAccessError(const ErrorData &error) {
	switch (error.Type()) {
	case ExceptionType::IO:
	case ExceptionType::HTTP:
	case ExceptionType::NETWORK:
	case ExceptionType::CONNECTION:
	case ExceptionType::PERMISSION:
	case ExceptionType::MISSING_EXTENSION:
	case ExceptionType::AUTOLOAD:
	case ExceptionType::INVALID_CONFIGURATION:
	case ExceptionType::NOT_IMPLEMENTED:
		return true;
	default:
		return false;
	}
}

static bool IsNotFoundError(const ErrorData &error) {
	auto status = error.ExtraInfo().find("status_code");
	if (status != error.ExtraInfo().end() && (status->second == "404" || status->second == "410")) {
		return true;
	}
	auto error_number = error.ExtraInfo().find("errno");
	return error_number != error.ExtraInfo().end() &&
	       (error_number->second == std::to_string(ENOENT) || error_number->second == std::to_string(ENOTDIR));
}

static bool IsNotRegularFileError(const ErrorData &error) {
	auto file_kind = error.ExtraInfo().find("file_kind");
	return file_kind != error.ExtraInfo().end() && file_kind->second == "not_regular";
}

static bool IsFileRangeOutOfBoundsError(const ErrorData &error) {
	auto file_range = error.ExtraInfo().find("file_range");
	return file_range != error.ExtraInfo().end() && file_range->second == "out_of_bounds";
}

static FileReference ToFileReference(ClientContext &context, const string &path) {
	FileReference result;
	result.url = path;
	auto resolved = ResolvedFile::Open(context, result);
	result.has_range = true;
	result.position = 0;
	result.size = NumericCast<int64_t>(resolved->ObjectSize());
	result.has_content_type = resolved->MimeTypeFromResolvedMetadata(result.content_type);
	result.Validate("to_file");
	return result;
}

template <bool TRY>
static void ToFileFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	for (idx_t row = 0; row < args.size(); row++) {
		auto path_value = args.data[0].GetValue(row);
		if (path_value.IsNull()) {
			result.SetValue(row, NullValue(FileLogicalType::Create()));
			continue;
		}
		try {
			result.SetValue(row, ToFileReference(state.GetContext(), path_value.GetValue<string>()).ToValue());
		} catch (const std::exception &exception) {
			if (!TRY) {
				throw;
			}
			ErrorData error(exception);
			if (!IsRecoverableAccessError(error)) {
				throw;
			}
			result.SetValue(row, NullValue(FileLogicalType::Create()));
		}
	}
}

static void FilePathFunction(DataChunk &args, ExpressionState &, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	for (idx_t row = 0; row < args.size(); row++) {
		FileReference file;
		if (!TryGetFile(args, 0, row, "file_path", file)) {
			result.SetValue(row, NullValue(LogicalType::VARCHAR));
			continue;
		}
		result.SetValue(row, Value(file.url));
	}
}

static void FileSizeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	for (idx_t row = 0; row < args.size(); row++) {
		FileReference file;
		if (!TryGetFile(args, 0, row, "file_size", file)) {
			result.SetValue(row, NullValue(LogicalType::UBIGINT));
			continue;
		}
		auto size = file.has_range ? NumericCast<uint64_t>(file.size)
		                           : ResolvedFile::Open(state.GetContext(), file)->LogicalSize();
		result.SetValue(row, Value::UBIGINT(size));
	}
}

static void FileExistsFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	for (idx_t row = 0; row < args.size(); row++) {
		FileReference file;
		if (!TryGetFile(args, 0, row, "file_exists", file)) {
			result.SetValue(row, NullValue(LogicalType::BOOLEAN));
			continue;
		}
		try {
			ResolvedFile::Open(state.GetContext(), file);
			result.SetValue(row, Value::BOOLEAN(true));
		} catch (const std::exception &exception) {
			ErrorData error(exception);
			if (IsNotFoundError(error) || IsNotRegularFileError(error) || IsFileRangeOutOfBoundsError(error)) {
				result.SetValue(row, Value::BOOLEAN(false));
			} else if (IsRecoverableAccessError(error)) {
				result.SetValue(row, NullValue(LogicalType::BOOLEAN));
			} else {
				throw;
			}
		}
	}
}

static void FileStatFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	for (idx_t row = 0; row < args.size(); row++) {
		FileReference file;
		if (!TryGetFile(args, 0, row, "file_stat", file)) {
			result.SetValue(row, NullValue(FileStatValue::Type()));
			continue;
		}
		result.SetValue(row, ResolvedFile::Open(state.GetContext(), file)->Stat().ToValue());
	}
}

enum class MimeDetectionMode : uint8_t { METADATA, CONTENT, AUTO };

static MimeDetectionMode ParseMimeDetectionMode(const Value &value) {
	auto mode = StringUtil::Lower(value.GetValue<string>());
	if (mode == "metadata") {
		return MimeDetectionMode::METADATA;
	}
	if (mode == "content") {
		return MimeDetectionMode::CONTENT;
	}
	if (mode == "auto") {
		return MimeDetectionMode::AUTO;
	}
	throw InvalidInputException("file_mime_type() detect must be 'metadata', 'content', or 'auto'");
}

static void FileMimeTypeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	for (idx_t row = 0; row < args.size(); row++) {
		FileReference file;
		if (!TryGetFile(args, 0, row, "file_mime_type", file)) {
			result.SetValue(row, NullValue(LogicalType::VARCHAR));
			continue;
		}
		MimeDetectionMode mode = MimeDetectionMode::METADATA;
		if (args.ColumnCount() == 2) {
			auto detect = args.data[1].GetValue(row);
			if (detect.IsNull()) {
				result.SetValue(row, NullValue(LogicalType::VARCHAR));
				continue;
			}
			mode = ParseMimeDetectionMode(detect);
		}

		string mime_type;
		bool detected = false;
		if (mode == MimeDetectionMode::METADATA) {
			if (file.has_content_type) {
				mime_type = file.content_type;
				detected = true;
			} else {
				detected = FileMimeType::FromPath(file.url, mime_type);
			}
		} else if (mode == MimeDetectionMode::AUTO && file.has_content_type) {
			mime_type = file.content_type;
			detected = true;
		}
		if (!detected && mode != MimeDetectionMode::METADATA) {
			auto resolved = ResolvedFile::Open(state.GetContext(), file);
			if (mode == MimeDetectionMode::AUTO) {
				detected = resolved->MimeTypeFromResolvedMetadata(mime_type);
			}
			if (!detected) {
				detected = resolved->GuessMimeType(mime_type);
			}
		}
		result.SetValue(row, detected ? Value(mime_type) : NullValue(LogicalType::VARCHAR));
	}
}

static void GuessMimeTypeFunction(DataChunk &args, ExpressionState &, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	for (idx_t row = 0; row < args.size(); row++) {
		auto bytes = args.data[0].GetValue(row);
		if (bytes.IsNull()) {
			result.SetValue(row, NullValue(LogicalType::VARCHAR));
			continue;
		}
		auto &data = StringValue::Get(bytes);
		string mime_type;
		auto detected = FileMimeType::FromBytes(const_data_ptr_cast(data.data()), data.size(), mime_type, true);
		result.SetValue(row, detected ? Value(mime_type) : NullValue(LogicalType::VARCHAR));
	}
}

static void FileEnrichFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	for (idx_t row = 0; row < args.size(); row++) {
		FileReference file;
		auto fields_value = args.data[1].GetValue(row);
		if (!TryGetFile(args, 0, row, "file_enrich", file) || fields_value.IsNull()) {
			result.SetValue(row, NullValue(FileLogicalType::Create()));
			continue;
		}
		bool enrich_size = false;
		bool enrich_content_type = false;
		bool enrich_checksum = false;
		for (const auto &field_value : ListValue::GetChildren(fields_value)) {
			if (field_value.IsNull()) {
				throw InvalidInputException("file_enrich() fields cannot contain NULL");
			}
			auto field = StringUtil::Lower(field_value.GetValue<string>());
			if (field == "size") {
				enrich_size = true;
			} else if (field == "content_type") {
				enrich_content_type = true;
			} else if (field == "checksum") {
				enrich_checksum = true;
			} else {
				throw InvalidInputException("file_enrich() field '%s' is not supported", field);
			}
		}

		unique_ptr<ResolvedFile> resolved;
		auto resolve = [&]() -> ResolvedFile & {
			if (!resolved) {
				resolved = ResolvedFile::Open(state.GetContext(), file);
			}
			return *resolved;
		};
		if (enrich_size && !file.has_range) {
			auto object_size = resolve().ObjectSize();
			file.has_range = true;
			file.position = 0;
			file.size = NumericCast<int64_t>(object_size);
		}
		if (enrich_content_type && !file.has_content_type) {
			file.has_content_type = resolve().MimeTypeFromResolvedMetadata(file.content_type);
			if (!file.has_content_type) {
				file.has_content_type = resolve().GuessMimeType(file.content_type);
			}
		}
		if (enrich_checksum && !file.has_checksum) {
			auto checksum = resolve().Sha256();
			file.has_checksum = true;
			file.checksum = "sha256:" + checksum;
		}
		result.SetValue(row, file.ToValue());
	}
}

static void FileSameLocationFunction(DataChunk &args, ExpressionState &, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	for (idx_t row = 0; row < args.size(); row++) {
		FileReference left;
		FileReference right;
		if (!TryGetFile(args, 0, row, "file_same_location", left) ||
		    !TryGetFile(args, 1, row, "file_same_location", right)) {
			result.SetValue(row, NullValue(LogicalType::BOOLEAN));
			continue;
		}
		if (left.url != right.url) {
			result.SetValue(row, Value::BOOLEAN(false));
		} else if (!left.has_range && !right.has_range) {
			result.SetValue(row, Value::BOOLEAN(true));
		} else if (left.has_range != right.has_range) {
			result.SetValue(row, NullValue(LogicalType::BOOLEAN));
		} else {
			result.SetValue(row, Value::BOOLEAN(left.position == right.position && left.size == right.size));
		}
	}
}

static void FileSameContentFunction(DataChunk &args, ExpressionState &, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	for (idx_t row = 0; row < args.size(); row++) {
		FileReference left;
		FileReference right;
		if (!TryGetFile(args, 0, row, "file_same_content", left) ||
		    !TryGetFile(args, 1, row, "file_same_content", right)) {
			result.SetValue(row, NullValue(LogicalType::BOOLEAN));
			continue;
		}
		if (left.has_range && right.has_range && left.size != right.size) {
			result.SetValue(row, Value::BOOLEAN(false));
			continue;
		}
		if (left.has_range && right.has_range && left.size == 0) {
			result.SetValue(row, Value::BOOLEAN(true));
			continue;
		}
		if (!left.has_checksum || !right.has_checksum) {
			result.SetValue(row, NullValue(LogicalType::BOOLEAN));
			continue;
		}
		auto left_separator = left.checksum.find(':');
		auto right_separator = right.checksum.find(':');
		auto left_algorithm = StringUtil::Lower(left.checksum.substr(0, left_separator));
		auto right_algorithm = StringUtil::Lower(right.checksum.substr(0, right_separator));
		if (left_algorithm != right_algorithm) {
			result.SetValue(row, NullValue(LogicalType::BOOLEAN));
			continue;
		}
		result.SetValue(row, Value::BOOLEAN(left.checksum.substr(left_separator + 1) ==
		                                    right.checksum.substr(right_separator + 1)));
	}
}

static void FileLocatorIdFunction(DataChunk &args, ExpressionState &, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	for (idx_t row = 0; row < args.size(); row++) {
		FileReference file;
		if (!TryGetFile(args, 0, row, "file_locator_id", file)) {
			result.SetValue(row, NullValue(LogicalType::VARCHAR));
			continue;
		}
		result.SetValue(row, Value(FileIdentity::LocatorId(file)));
	}
}

static void FileContentIdFunction(DataChunk &args, ExpressionState &, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	for (idx_t row = 0; row < args.size(); row++) {
		FileReference file;
		if (!TryGetFile(args, 0, row, "file_content_id", file)) {
			result.SetValue(row, NullValue(LogicalType::VARCHAR));
			continue;
		}
		result.SetValue(row, FileIdentity::ContentId(file));
	}
}

static ScalarFunction MakeScalarFunction(const string &name, vector<LogicalType> arguments, LogicalType return_type,
                                         scalar_function_t function, bool accesses_storage = false) {
	ScalarFunction result(name, std::move(arguments), std::move(return_type), std::move(function));
	result.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	result.SetFallible();
	if (accesses_storage) {
		result.SetStability(FunctionStability::VOLATILE);
	}
	return result;
}

} // namespace

vector<ScalarFunction> FileMetadataFunctions::GetFunctions() {
	auto file_type = FileLogicalType::Create();
	vector<ScalarFunction> result;
	result.push_back(MakeScalarFunction("to_file", {LogicalType::VARCHAR}, file_type, ToFileFunction<false>, true));
	result.push_back(MakeScalarFunction("try_to_file", {LogicalType::VARCHAR}, file_type, ToFileFunction<true>, true));
	result.push_back(MakeScalarFunction("file_enrich", {file_type, LogicalType::LIST(LogicalType::VARCHAR)}, file_type,
	                                    FileEnrichFunction, true));
	result.push_back(MakeScalarFunction("file_path", {file_type}, LogicalType::VARCHAR, FilePathFunction));
	result.push_back(MakeScalarFunction("file_size", {file_type}, LogicalType::UBIGINT, FileSizeFunction, true));
	result.push_back(MakeScalarFunction("file_exists", {file_type}, LogicalType::BOOLEAN, FileExistsFunction, true));
	result.push_back(MakeScalarFunction("file_stat", {file_type}, FileStatValue::Type(), FileStatFunction, true));
	result.push_back(MakeScalarFunction("file_mime_type", {file_type}, LogicalType::VARCHAR, FileMimeTypeFunction));
	result.push_back(MakeScalarFunction("file_mime_type", {file_type, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                                    FileMimeTypeFunction, true));
	result.push_back(
	    MakeScalarFunction("guess_mime_type", {LogicalType::BLOB}, LogicalType::VARCHAR, GuessMimeTypeFunction));
	result.push_back(MakeScalarFunction("file_same_location", {file_type, file_type}, LogicalType::BOOLEAN,
	                                    FileSameLocationFunction));
	result.push_back(
	    MakeScalarFunction("file_same_content", {file_type, file_type}, LogicalType::BOOLEAN, FileSameContentFunction));
	result.push_back(MakeScalarFunction("file_locator_id", {file_type}, LogicalType::VARCHAR, FileLocatorIdFunction));
	result.push_back(MakeScalarFunction("file_content_id", {file_type}, LogicalType::VARCHAR, FileContentIdFunction));
	return result;
}

} // namespace duckdb
