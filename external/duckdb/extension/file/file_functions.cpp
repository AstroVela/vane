// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// file_functions.cpp
//
//===----------------------------------------------------------------------===//

#include "file_functions.hpp"
#include "file_resolver.hpp"
#include "file_value.hpp"
#include "image_tensor.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/string.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/image.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/planner/expression.hpp"

namespace duckdb {

namespace {

static void ValidateFileArguments(DataChunk &args) {
	UnifiedVectorFormat url_data;
	UnifiedVectorFormat position_data;
	UnifiedVectorFormat size_data;
	UnifiedVectorFormat checksum_data;
	args.data[FileLogicalType::URL].ToUnifiedFormat(args.size(), url_data);
	args.data[FileLogicalType::POSITION].ToUnifiedFormat(args.size(), position_data);
	args.data[FileLogicalType::SIZE].ToUnifiedFormat(args.size(), size_data);
	args.data[FileLogicalType::CHECKSUM].ToUnifiedFormat(args.size(), checksum_data);

	auto positions = UnifiedVectorFormat::GetData<int64_t>(position_data);
	auto sizes = UnifiedVectorFormat::GetData<int64_t>(size_data);
	auto checksums = UnifiedVectorFormat::GetData<string_t>(checksum_data);
	auto urls = UnifiedVectorFormat::GetData<string_t>(url_data);
	for (idx_t row = 0; row < args.size(); row++) {
		auto url_index = url_data.sel->get_index(row);
		auto position_index = position_data.sel->get_index(row);
		auto size_index = size_data.sel->get_index(row);
		auto checksum_index = checksum_data.sel->get_index(row);
		auto has_position = position_data.validity.RowIsValid(position_index);
		auto has_size = size_data.validity.RowIsValid(size_index);
		auto position = has_position ? positions[position_index] : 0;
		auto size = has_size ? sizes[size_index] : 0;
		string url;
		const string *url_ptr = nullptr;
		if (url_data.validity.RowIsValid(url_index)) {
			url = urls[url_index].GetString();
			url_ptr = &url;
		}

		string checksum;
		const string *checksum_ptr = nullptr;
		if (checksum_data.validity.RowIsValid(checksum_index)) {
			checksum = checksums[checksum_index].GetString();
			checksum_ptr = &checksum;
		}
		FileReference::ValidateFields(url_ptr, has_position, position, has_size, size, checksum_ptr, "file");
	}
}

static void FileConstructorFunction(DataChunk &args, ExpressionState &, Vector &result) {
	D_ASSERT(args.ColumnCount() == FileLogicalType::FIELD_COUNT);
	ValidateFileArguments(args);

	bool all_constant = true;
	auto &children = StructVector::GetEntries(result);
	for (idx_t index = 0; index < args.ColumnCount(); index++) {
		if (args.data[index].GetVectorType() != VectorType::CONSTANT_VECTOR) {
			all_constant = false;
		}
		children[index]->Reference(args.data[index]);
	}
	result.SetVectorType(all_constant ? VectorType::CONSTANT_VECTOR : VectorType::FLAT_VECTOR);
	result.Verify(args.size());
}

static void ImageConstructorFunction(DataChunk &args, ExpressionState &, Vector &result) {
	const auto all_constant = args.AllConstant();
	const auto count = all_constant && args.size() ? idx_t(1) : args.size();
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &children = StructVector::GetEntries(result);
	for (auto &child : children) {
		child->SetVectorType(VectorType::FLAT_VECTOR);
	}
	ListVector::SetListSize(*children[ImageLogicalType::DATA], 0);
	for (idx_t row = 0; row < count; row++) {
		vector<Value> fields;
		bool is_null = false;
		for (auto &arg : args.data) {
			fields.push_back(arg.GetValue(row));
			is_null |= fields.back().IsNull();
		}
		if (is_null) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto &bytes = StringValue::Get(fields[0]);
		auto width = fields[1].GetValue<uint32_t>();
		auto height = fields[2].GetValue<uint32_t>();
		auto mode = fields[4].GetValue<string>();
		auto element_size = ImageLogicalType::ElementSize(mode);
		if (bytes.size() % element_size) {
			throw InvalidInputException("image pixel bytes are not aligned with its mode");
		}
		ImageLogicalType::ValidateFields(bytes.size() / element_size, width, height, fields[3].GetValue<uint8_t>(),
		                                 mode, "image");
		ImageVector::WritePixels(result, row, width, height, mode, const_data_ptr_cast(bytes.data()));
	}
	if (all_constant) {
		// SetVectorType recursively marks every STRUCT child constant as well.
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static unique_ptr<FunctionData> BindImageAttribute(ClientContext &, ScalarFunction &function,
                                                   vector<unique_ptr<Expression>> &arguments) {
	auto &type = arguments[0]->return_type;
	if (type.id() == LogicalTypeId::UNKNOWN) {
		throw ParameterNotResolvedException();
	}
	if (type.id() != LogicalTypeId::SQLNULL && !ImageLogicalType::IsImage(type)) {
		throw BinderException("%s() requires IMAGE, got %s", function.name, type);
	}
	function.arguments[0] = type;
	return nullptr;
}

static bool ImageRowIsNull(const Vector &input, idx_t row) {
	if (input.GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		return ImageRowIsNull(DictionaryVector::Child(input), DictionaryVector::SelVector(input).get_index(row));
	}
	return input.GetVectorType() == VectorType::CONSTANT_VECTOR ? ConstantVector::IsNull(input)
	                                                            : FlatVector::IsNull(input, row);
}

static ImageLayout ImageAttributeLayout(Vector &input, idx_t row) {
	if (ImageLogicalType::IsFixedShape(input.GetType())) {
		return ImageVector::Layout(input, row);
	}
	// Vector::Slice pre-slices STRUCT children, including repeated dictionary
	// selections. Their logical row is already selected; remapping the parent
	// index here would apply the dictionary selection twice.
	auto &fields = StructVector::GetEntries(input);
	for (auto &field : fields) {
		if (ImageRowIsNull(*field, row)) {
			throw InvalidInputException("Non-NULL IMAGE values cannot contain NULL fields");
		}
	}
	return {fields[ImageLogicalType::WIDTH]->GetValue(row).GetValue<uint32_t>(),
	        fields[ImageLogicalType::HEIGHT]->GetValue(row).GetValue<uint32_t>(),
	        fields[ImageLogicalType::CHANNELS]->GetValue(row).GetValue<uint16_t>(),
	        fields[ImageLogicalType::MODE]->GetValue(row).GetValue<uint8_t>()};
}

template <int PROPERTY>
static void ImageAttributeFunction(DataChunk &args, ExpressionState &, Vector &result) {
	const auto all_constant = args.AllConstant();
	const auto count = all_constant && args.size() ? idx_t(1) : args.size();
	// Inspect only parent validity and metadata. Flattening an Image also
	// expands its pixel children, which attributes never need to read.
	auto &input = args.data[0];
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto data = FlatVector::GetData<uint32_t>(result);
	for (idx_t row = 0; row < count; row++) {
		if (ImageRowIsNull(input, row)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto property = PROPERTY;
		if (PROPERTY == 0) {
			auto value = args.data[1].GetValue(row);
			if (value.IsNull()) {
				FlatVector::SetNull(result, row, true);
				continue;
			}
			auto name = StringValue::Get(value);
			if (name == "height") {
				property = 1;
			} else if (name == "width") {
				property = 2;
			} else if (name == "channel") {
				property = 3;
			} else if (name == "mode") {
				property = 4;
			} else {
				throw InvalidInputException("image_attribute() property must be height, width, channel, or mode");
			}
		}
		auto layout = ImageAttributeLayout(input, row);
		data[row] = property == 1   ? layout.height
		            : property == 2 ? layout.width
		            : property == 3 ? layout.channels
		                            : layout.mode;
		FlatVector::SetNull(result, row, false);
	}
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

template <int PROPERTY>
static ScalarFunction GetImageAttribute(const string &name) {
	vector<LogicalType> arguments {LogicalType::ANY};
	if (PROPERTY == 0) {
		arguments.push_back(LogicalType::VARCHAR);
	}
	ScalarFunction function(name, arguments, LogicalType::UINTEGER, ImageAttributeFunction<PROPERTY>,
	                        BindImageAttribute);
	function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	function.SetFallible();
	return function;
}

static bool MimeTypeMatches(FileMediaType media_type, const string &mime_type) {
	switch (media_type) {
	case FileMediaType::IMAGE:
		return StringUtil::CIStartsWith(mime_type, "image/");
	case FileMediaType::AUDIO:
		return StringUtil::CIStartsWith(mime_type, "audio/");
	case FileMediaType::VIDEO:
		return StringUtil::CIStartsWith(mime_type, "video/");
	case FileMediaType::UNKNOWN:
	default:
		throw InternalException("Cannot verify an unknown FILE media type");
	}
}

static void VerifyMediaFile(ClientContext &context, FileReference &file, const string &function_name) {
	if (file.has_content_type && !MimeTypeMatches(file.media_type, file.content_type)) {
		throw InvalidInputException("%s() expected %s content but content_type is '%s'", function_name,
		                            FileLogicalType::GetTypeName(file.media_type), file.content_type);
	}

	string detected_type;
	auto resolved = ResolvedFile::Open(context, file);
	if (!resolved->GuessMimeType(detected_type) || !MimeTypeMatches(file.media_type, detected_type)) {
		throw InvalidInputException("%s() could not verify %s content at '%s'", function_name,
		                            FileLogicalType::GetTypeName(file.media_type), file.url);
	}
	if (!file.has_content_type) {
		file.has_content_type = true;
		file.content_type = std::move(detected_type);
	}
}

template <FileMediaType MEDIA_TYPE>
static unique_ptr<FunctionData> BindMediaFileConstructor(ClientContext &, ScalarFunction &bound_function,
                                                         vector<unique_ptr<Expression>> &arguments) {
	auto input_type = arguments[0]->return_type;
	if (input_type.id() == LogicalTypeId::UNKNOWN) {
		// The URL form is the only interpretation available for an unresolved
		// parameter. Preserve exact FILE-family types when the input type is
		// already known, but retain normal VARCHAR inference for PREPARE.
		input_type = LogicalType::VARCHAR;
	}
	auto valid_input = (input_type.id() == LogicalTypeId::VARCHAR && !input_type.IsJSONType()) ||
	                   input_type.id() == LogicalTypeId::STRING_LITERAL || input_type.id() == LogicalTypeId::SQLNULL;
	if (FileLogicalType::IsFile(input_type)) {
		auto input_media_type = FileLogicalType::GetMediaType(input_type);
		valid_input = input_media_type == FileMediaType::UNKNOWN || input_media_type == MEDIA_TYPE;
	}
	if (!valid_input) {
		throw BinderException("%s() requires VARCHAR, FILE, or %s, not %s",
		                      FileLogicalType::GetConstructorName(MEDIA_TYPE), FileLogicalType::GetTypeName(MEDIA_TYPE),
		                      input_type.ToString());
	}
	bound_function.arguments[0] = input_type.id() == LogicalTypeId::STRING_LITERAL ? LogicalType::VARCHAR : input_type;
	return nullptr;
}

template <FileMediaType MEDIA_TYPE>
static void MediaFileConstructorFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto function_name = FileLogicalType::GetConstructorName(MEDIA_TYPE);
	for (idx_t row = 0; row < args.size(); row++) {
		auto input = args.data[0].GetValue(row);
		auto verify = args.ColumnCount() == 2 ? args.data[1].GetValue(row) : Value::BOOLEAN(false);
		if (input.IsNull() || verify.IsNull()) {
			result.SetValue(row, Value(result.GetType()));
			continue;
		}

		FileReference file;
		if (input.type().id() == LogicalTypeId::VARCHAR) {
			file.url = input.GetValue<string>();
			file.media_type = MEDIA_TYPE;
			file.Validate(function_name);
		} else {
			file = FileReference::FromValue(input, function_name);
			file.media_type = MEDIA_TYPE;
		}
		if (verify.GetValue<bool>()) {
			VerifyMediaFile(state.GetContext(), file, function_name);
		}
		result.SetValue(row, file.ToValue());
	}
}

template <bool NEGATE>
static void FileComparisonFunction(DataChunk &args, ExpressionState &, Vector &result) {
	D_ASSERT(args.ColumnCount() == 2);
	auto &left_children = StructVector::GetEntries(args.data[0]);
	auto &right_children = StructVector::GetEntries(args.data[1]);
	D_ASSERT(left_children.size() == FileLogicalType::FIELD_COUNT);
	D_ASSERT(right_children.size() == FileLogicalType::FIELD_COUNT);

	for (idx_t index = 0; index < FileLogicalType::FIELD_COUNT; index++) {
		Vector field_equal(LogicalType::BOOLEAN);
		VectorOperations::Equals(*left_children[index], *right_children[index], field_equal, args.size());
		if (index == 0) {
			result.Reference(field_equal);
		} else {
			Vector conjunction(LogicalType::BOOLEAN);
			VectorOperations::And(field_equal, result, conjunction, args.size());
			result.Reference(conjunction);
		}
	}

	if (NEGATE) {
		Vector negated_result(LogicalType::BOOLEAN);
		VectorOperations::Not(result, negated_result, args.size());
		result.Reference(negated_result);
	}

	UnifiedVectorFormat left_data;
	UnifiedVectorFormat right_data;
	args.data[0].ToUnifiedFormat(args.size(), left_data);
	args.data[1].ToUnifiedFormat(args.size(), right_data);
	if (!left_data.validity.AllValid() || !right_data.validity.AllValid()) {
		result.Flatten(args.size());
		auto &result_validity = FlatVector::Validity(result);
		for (idx_t row = 0; row < args.size(); row++) {
			auto left_index = left_data.sel->get_index(row);
			auto right_index = right_data.sel->get_index(row);
			if (!left_data.validity.RowIsValid(left_index) || !right_data.validity.RowIsValid(right_index)) {
				result_validity.SetInvalid(row);
			}
		}
	}
}

static ScalarFunction GetFileConstructor() {
	vector<LogicalType> arguments {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT,
	                               LogicalType::VARCHAR};
	ScalarFunction function("file", std::move(arguments), FileLogicalType::Create(), FileConstructorFunction);
	function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	function.SetFallible();
	return function;
}

static ScalarFunction GetImageConstructor() {
	vector<LogicalType> arguments {LogicalType::BLOB, LogicalType::UINTEGER, LogicalType::UINTEGER,
	                               LogicalType::UTINYINT, LogicalType::VARCHAR};
	ScalarFunction function(ImageLogicalType::CONSTRUCTOR_NAME, std::move(arguments), ImageLogicalType::Create(),
	                        ImageConstructorFunction);
	function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	function.SetFallible();
	return function;
}

template <FileMediaType MEDIA_TYPE>
static ScalarFunction GetMediaFileConstructor(bool with_verify) {
	vector<LogicalType> arguments {LogicalType::ANY};
	if (with_verify) {
		arguments.push_back(LogicalType::BOOLEAN);
	}
	ScalarFunction function(FileLogicalType::GetConstructorName(MEDIA_TYPE), std::move(arguments),
	                        FileLogicalType::Create(MEDIA_TYPE), MediaFileConstructorFunction<MEDIA_TYPE>,
	                        BindMediaFileConstructor<MEDIA_TYPE>);
	function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	function.SetFallible();
	if (with_verify) {
		function.SetStability(FunctionStability::VOLATILE);
	}
	return function;
}

template <bool NEGATE>
static ScalarFunction GetFileComparison(FileMediaType media_type) {
	auto file_type = FileLogicalType::Create(media_type);
	auto name = NEGATE ? FileLogicalType::NOT_EQUAL_FUNCTION_NAME : FileLogicalType::EQUAL_FUNCTION_NAME;
	return ScalarFunction(name, {file_type, file_type}, LogicalType::BOOLEAN, FileComparisonFunction<NEGATE>);
}

} // namespace

vector<ScalarFunction> FileFunctions::GetFunctions() {
	vector<ScalarFunction> result;
	result.push_back(GetFileConstructor());
	result.push_back(GetImageConstructor());
	result.push_back(ImageToTensor::Function());
	result.push_back(GetImageAttribute<0>("image_attribute"));
	result.push_back(GetImageAttribute<1>("image_height"));
	result.push_back(GetImageAttribute<2>("image_width"));
	result.push_back(GetImageAttribute<3>("image_channel"));
	result.push_back(GetImageAttribute<4>("image_mode"));
	result.push_back(GetMediaFileConstructor<FileMediaType::IMAGE>(false));
	result.push_back(GetMediaFileConstructor<FileMediaType::IMAGE>(true));
	result.push_back(GetMediaFileConstructor<FileMediaType::AUDIO>(false));
	result.push_back(GetMediaFileConstructor<FileMediaType::AUDIO>(true));
	result.push_back(GetMediaFileConstructor<FileMediaType::VIDEO>(false));
	result.push_back(GetMediaFileConstructor<FileMediaType::VIDEO>(true));
	for (auto media_type : FileLogicalType::MEDIA_TYPES) {
		result.push_back(GetFileComparison<false>(media_type));
		result.push_back(GetFileComparison<true>(media_type));
	}
	return result;
}

} // namespace duckdb
