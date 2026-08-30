// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/ai_sql_functions.hpp"

#include "file_resolver.hpp"
#include "file_value.hpp"
#include "vane_python/python_conversion.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/function/scalar/nested_functions.hpp"
#include "duckdb/function/scalar_macro_function.hpp"
#include "duckdb/function/scalar/vllm_functions.hpp"
#include "duckdb/function/scalar/udf_functions.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"
#include "vane_python/python_objects.hpp"
#include "vane_python/python_udf_utils.hpp"

namespace duckdb {

namespace {

enum class AISQLKind : uint8_t { PROMPT, EMBED };
enum class PromptInputKind : uint8_t { TEXT, BLOB, BLOB_LIST, FILE, FILE_LIST };

static constexpr const char *HIDDEN_EMBED_FUNCTION = "__vane_ai_embed";
static constexpr const char *HIDDEN_PROMPT_FUNCTION = "__vane_ai_prompt";
static constexpr const char *HIDDEN_PROMPT_PACK_FUNCTION = "__vane_ai_prompt_pack";
static constexpr idx_t PROMPT_PACK_MEDIA_SUPPORTED_INDEX = 0;
static constexpr idx_t PROMPT_PACK_SUPPORTED_MIME_TYPES_INDEX = 1;
static constexpr idx_t PROMPT_PACK_SINGLE_MESSAGE_INDEX = 2;
static constexpr idx_t PROMPT_PACK_MESSAGE_OFFSET = 3;

// Prompt providers receive media inline. Keep both an individual FILE and the
// materialized input vector bounded before allocating its bytes. The vector
// limit matches the default UDF transport target; the per-FILE limit matches
// the largest common inline request limit among the initial providers.
static constexpr uint64_t MAX_PROMPT_FILE_BYTES = 20ULL * 1024ULL * 1024ULL;
static constexpr uint64_t MAX_PROMPT_MEDIA_VECTOR_BYTES = 128ULL * 1024ULL * 1024ULL;

static constexpr const char *PROMPT_MEDIA_ERROR_UNSUPPORTED = "unsupported";
static constexpr const char *PROMPT_MEDIA_ERROR_READ = "read";
static constexpr const char *PROMPT_MEDIA_ERROR_MIME = "mime";
static constexpr const char *PROMPT_MEDIA_ERROR_INVALID_MIME = "invalid_mime";
static constexpr const char *PROMPT_MEDIA_ERROR_UNSUPPORTED_MIME = "unsupported_mime";
static constexpr const char *PROMPT_MEDIA_ERROR_EMPTY = "empty";
static constexpr const char *PROMPT_MEDIA_ERROR_TOO_LARGE = "too_large";
static constexpr const char *PROMPT_MEDIA_ERROR_VECTOR_TOO_LARGE = "vector_too_large";

static LogicalType PromptMediaLogicalType() {
	child_list_t<LogicalType> fields;
	fields.emplace_back("content_type", LogicalType::VARCHAR);
	fields.emplace_back("data", LogicalType::BLOB);
	fields.emplace_back("error", LogicalType::VARCHAR);
	return LogicalType::STRUCT(std::move(fields));
}

static bool IsPromptMimeTokenCharacter(char value) {
	return (value >= '0' && value <= '9') || (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z') ||
	       value == '!' || value == '#' || value == '$' || value == '%' || value == '&' || value == '\'' ||
	       value == '*' || value == '+' || value == '-' || value == '.' || value == '^' || value == '_' ||
	       value == '`' || value == '|' || value == '~';
}

static bool NormalizePromptContentType(const string &value, string &result) {
	auto separator = value.find(';');
	result = value.substr(0, separator);
	StringUtil::Trim(result);
	result = StringUtil::Lower(result);
	auto slash = result.find('/');
	if (slash == string::npos || slash == 0 || slash + 1 == result.size() ||
	    result.find('/', slash + 1) != string::npos) {
		return false;
	}
	auto media_type = result.substr(0, slash);
	auto media_subtype = result.substr(slash + 1);
	if (media_type == "*" || media_subtype == "*") {
		return false;
	}
	for (auto character : result) {
		if (character != '/' && !IsPromptMimeTokenCharacter(character)) {
			return false;
		}
	}
	return true;
}

static bool PromptContentTypeSupported(const string &content_type, const vector<string> &supported_mime_types) {
	if (supported_mime_types.empty()) {
		return true;
	}
	for (auto &supported : supported_mime_types) {
		if (supported == content_type) {
			return true;
		}
	}
	return false;
}

static bool MustPropagatePromptFileError(const ErrorData &error) {
	switch (error.Type()) {
	case ExceptionType::INTERRUPT:
	case ExceptionType::OUT_OF_MEMORY:
	case ExceptionType::FATAL:
	case ExceptionType::INTERNAL:
	case ExceptionType::NULL_POINTER:
		return true;
	default:
		return false;
	}
}

struct PreparedPromptMedia {
	bool is_null = false;
	bool has_content_type = false;
	string content_type;
	string error;
	uint64_t size = 0;
	unique_ptr<ResolvedFile> resolved;

	Value Materialize() {
		auto media_type = PromptMediaLogicalType();
		if (is_null) {
			return Value(media_type);
		}

		Value data(LogicalType::BLOB);
		if (error.empty()) {
			try {
				string bytes(NumericCast<idx_t>(size), '\0');
				resolved->ReadExact(reinterpret_cast<data_ptr_t>(bytes.data()), size);
				data = Value::BLOB_RAW(bytes);
			} catch (const std::exception &exception) {
				ErrorData error_data(exception);
				if (MustPropagatePromptFileError(error_data)) {
					throw;
				}
				// FILE validation already ran before opening. Everything left at
				// this boundary is an execution-time resolver/read failure, which
				// must reach the Python row policy without exposing its details.
				error = PROMPT_MEDIA_ERROR_READ;
			}
		}
		resolved.reset();

		vector<Value> fields;
		fields.reserve(3);
		fields.push_back(has_content_type ? Value(content_type) : Value(LogicalType::VARCHAR));
		fields.push_back(std::move(data));
		fields.push_back(error.empty() ? Value(LogicalType::VARCHAR) : Value(error));
		return Value::STRUCT(media_type, std::move(fields));
	}
};

static void PreparePromptMedia(ClientContext &context, const Value &value, bool media_supported,
                               const vector<string> &supported_mime_types, uint64_t &vector_bytes,
                               PreparedPromptMedia &result) {
	if (value.IsNull()) {
		result.is_null = true;
		return;
	}

	// Validate the canonical FILE contract even when the selected provider
	// cannot consume media. This is pure and intentionally precedes all I/O.
	auto file = FileReference::FromValue(value, "ai_prompt");
	if (!media_supported) {
		result.error = PROMPT_MEDIA_ERROR_UNSUPPORTED;
		return;
	}
	if (file.has_content_type) {
		result.has_content_type = NormalizePromptContentType(file.content_type, result.content_type);
		if (!result.has_content_type) {
			result.error = PROMPT_MEDIA_ERROR_INVALID_MIME;
			return;
		}
		if (!PromptContentTypeSupported(result.content_type, supported_mime_types)) {
			result.error = PROMPT_MEDIA_ERROR_UNSUPPORTED_MIME;
			return;
		}
	}

	try {
		result.resolved = ResolvedFile::Open(context, file);
		result.size = result.resolved->LogicalSize();
		if (result.size == 0) {
			result.error = PROMPT_MEDIA_ERROR_EMPTY;
			result.resolved.reset();
			return;
		}
		if (result.size > MAX_PROMPT_FILE_BYTES) {
			result.error = PROMPT_MEDIA_ERROR_TOO_LARGE;
			result.resolved.reset();
			return;
		}
		if (result.size > MAX_PROMPT_MEDIA_VECTOR_BYTES - vector_bytes) {
			result.error = PROMPT_MEDIA_ERROR_VECTOR_TOO_LARGE;
			result.resolved.reset();
			return;
		}
		if (!file.has_content_type) {
			result.has_content_type = result.resolved->GuessMimeType(result.content_type);
			if (!result.has_content_type) {
				result.error = PROMPT_MEDIA_ERROR_MIME;
				result.resolved.reset();
				return;
			}
			if (!PromptContentTypeSupported(result.content_type, supported_mime_types)) {
				result.error = PROMPT_MEDIA_ERROR_UNSUPPORTED_MIME;
				result.resolved.reset();
				return;
			}
		}
		vector_bytes += result.size;
	} catch (const std::exception &exception) {
		ErrorData error(exception);
		if (MustPropagatePromptFileError(error)) {
			throw;
		}
		result.error = PROMPT_MEDIA_ERROR_READ;
		result.resolved.reset();
	}
}

static bool IsPromptFileList(const LogicalType &type);

struct MaterializedPromptFileInput {
	Value value;
	bool has_error;
};

static MaterializedPromptFileInput MaterializePromptFileInput(ClientContext &context, const Value &value,
                                                              const LogicalType &input_type, bool media_supported,
                                                              const vector<string> &supported_mime_types,
                                                              uint64_t &vector_bytes) {
	auto media_type = PromptMediaLogicalType();
	if (value.IsNull()) {
		return {Value(FileLogicalType::IsFile(input_type) ? media_type : LogicalType::LIST(media_type)), false};
	}
	if (FileLogicalType::IsFile(input_type)) {
		PreparedPromptMedia item;
		PreparePromptMedia(context, value, media_supported, supported_mime_types, vector_bytes, item);
		auto materialized = item.Materialize();
		return {std::move(materialized), !item.error.empty()};
	}
	if (!IsPromptFileList(input_type)) {
		throw InternalException("ai_prompt pack received an unexpected input type");
	}

	auto &children = ListValue::GetChildren(value);
	vector<Value> values;
	values.reserve(children.size());
	for (auto &child : children) {
		PreparedPromptMedia item;
		PreparePromptMedia(context, child, media_supported, supported_mime_types, vector_bytes, item);
		values.push_back(item.Materialize());
		if (!item.error.empty()) {
			// Only the first error is observable by the row wrapper. Drop bytes
			// already read for this list and do not open its remaining FILEs.
			for (idx_t index = 0; index + 1 < values.size(); index++) {
				values[index] = Value(media_type);
			}
			while (values.size() < children.size()) {
				values.push_back(Value(media_type));
			}
			return {Value::LIST(media_type, std::move(values)), true};
		}
	}
	return {Value::LIST(media_type, std::move(values)), false};
}

static bool PromptPackBooleanArgument(DataChunk &args, idx_t argument_index, idx_t row_index,
                                      const char *argument_name) {
	auto value = args.data[argument_index].GetValue(row_index);
	if (value.IsNull()) {
		throw InvalidInputException("ai_prompt %s cannot be NULL", argument_name);
	}
	return BooleanValue::Get(value);
}

static vector<string> PromptPackMimeTypesArgument(DataChunk &args, idx_t row_index) {
	auto value = args.data[PROMPT_PACK_SUPPORTED_MIME_TYPES_INDEX].GetValue(row_index);
	if (value.IsNull()) {
		throw InvalidInputException("ai_prompt supported MIME types cannot be NULL");
	}
	vector<string> result;
	auto &children = ListValue::GetChildren(value);
	result.reserve(children.size());
	for (auto &child : children) {
		string normalized;
		if (child.IsNull() || !NormalizePromptContentType(StringValue::Get(child), normalized)) {
			throw InvalidInputException("ai_prompt supported MIME types must contain valid MIME strings");
		}
		result.push_back(std::move(normalized));
	}
	return result;
}

static void PromptPackFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &expression = state.expr.Cast<BoundFunctionExpression>();
	if (!expression.bind_info) {
		throw InternalException("ai_prompt pack is missing bind data");
	}
	auto &bind_data = expression.bind_info->Cast<VariableReturnBindData>();
	auto &return_children = StructType::GetChildTypes(bind_data.stype);
	if (args.ColumnCount() < PROMPT_PACK_MESSAGE_OFFSET + 1 ||
	    return_children.size() != args.ColumnCount() - PROMPT_PACK_MESSAGE_OFFSET) {
		throw InternalException("ai_prompt pack received an invalid argument envelope");
	}
	auto input_types = args.GetTypes();

	result.SetVectorType(VectorType::FLAT_VECTOR);
	uint64_t vector_bytes = 0;
	for (idx_t row_index = 0; row_index < args.size(); row_index++) {
		auto media_supported =
		    PromptPackBooleanArgument(args, PROMPT_PACK_MEDIA_SUPPORTED_INDEX, row_index, "media policy");
		auto supported_mime_types = PromptPackMimeTypesArgument(args, row_index);
		auto single_message =
		    PromptPackBooleanArgument(args, PROMPT_PACK_SINGLE_MESSAGE_INDEX, row_index, "single-message policy");
		auto first_value = args.data[PROMPT_PACK_MESSAGE_OFFSET].GetValue(row_index);
		if (single_message && first_value.IsNull()) {
			// SQL FILE overloads treat the text prompt as the primary input. A
			// row-varying NULL must short-circuit before any sibling FILE opens.
			result.SetValue(row_index, Value(bind_data.stype));
			continue;
		}

		auto row_start_bytes = vector_bytes;
		vector<Value> fields;
		fields.reserve(return_children.size());
		vector<idx_t> successful_file_fields;
		bool row_has_media_error = false;
		for (idx_t argument_index = PROMPT_PACK_MESSAGE_OFFSET; argument_index < args.ColumnCount(); argument_index++) {
			auto value = argument_index == PROMPT_PACK_MESSAGE_OFFSET ? std::move(first_value)
			                                                          : args.data[argument_index].GetValue(row_index);
			auto &input_type = input_types[argument_index];
			if (FileLogicalType::IsFile(input_type) || IsPromptFileList(input_type)) {
				auto field_index = fields.size();
				if (row_has_media_error) {
					fields.push_back(Value(return_children[field_index].second));
					continue;
				}
				auto materialized = MaterializePromptFileInput(state.GetContext(), value, input_type, media_supported,
				                                               supported_mime_types, vector_bytes);
				fields.push_back(std::move(materialized.value));
				if (materialized.has_error) {
					// A failed row never reaches the provider. Release successful
					// FILE payloads from this row and restore their vector budget so
					// later rows are not rejected because of discarded bytes.
					for (auto successful_index : successful_file_fields) {
						fields[successful_index] = Value(return_children[successful_index].second);
					}
					vector_bytes = row_start_bytes;
					row_has_media_error = true;
				} else {
					successful_file_fields.push_back(field_index);
				}
			} else {
				fields.push_back(std::move(value));
			}
		}
		result.SetValue(row_index, Value::STRUCT(bind_data.stype, std::move(fields)));
	}
}

static unique_ptr<FunctionData> PromptPackBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() < PROMPT_PACK_MESSAGE_OFFSET + 1) {
		throw BinderException("ai_prompt pack requires at least one message");
	}
	for (auto policy_index : {PROMPT_PACK_MEDIA_SUPPORTED_INDEX, PROMPT_PACK_SINGLE_MESSAGE_INDEX}) {
		auto &policy = *arguments[policy_index];
		if (!policy.IsFoldable()) {
			throw BinderException("ai_prompt pack policies must be constant BOOLEAN values");
		}
		if (policy.HasParameter()) {
			throw ParameterNotResolvedException();
		}
		auto value = ExpressionExecutor::EvaluateScalar(context, policy);
		if (value.IsNull() || value.type().id() != LogicalTypeId::BOOLEAN) {
			throw BinderException("ai_prompt pack policies must be non-NULL BOOLEAN values");
		}
	}
	auto &mime_policy = *arguments[PROMPT_PACK_SUPPORTED_MIME_TYPES_INDEX];
	if (!mime_policy.IsFoldable()) {
		throw BinderException("ai_prompt pack supported MIME types must be constant");
	}
	if (mime_policy.HasParameter()) {
		throw ParameterNotResolvedException();
	}
	auto mime_types = ExpressionExecutor::EvaluateScalar(context, mime_policy);
	if (mime_types.IsNull() || mime_types.type() != LogicalType::LIST(LogicalType::VARCHAR)) {
		throw BinderException("ai_prompt pack supported MIME types must be a non-NULL VARCHAR[]");
	}
	for (auto &mime_type : ListValue::GetChildren(mime_types)) {
		string normalized;
		if (mime_type.IsNull() || !NormalizePromptContentType(StringValue::Get(mime_type), normalized)) {
			throw BinderException("ai_prompt pack supported MIME types must contain valid MIME strings");
		}
	}
	auto media_supported =
	    BooleanValue::Get(ExpressionExecutor::EvaluateScalar(context, *arguments[PROMPT_PACK_MEDIA_SUPPORTED_INDEX]));

	child_list_t<LogicalType> fields;
	fields.reserve(arguments.size() - PROMPT_PACK_MESSAGE_OFFSET);
	for (idx_t argument_index = PROMPT_PACK_MESSAGE_OFFSET; argument_index < arguments.size(); argument_index++) {
		auto &argument = arguments[argument_index];
		auto input_type = argument->return_type;
		if (input_type.id() == LogicalTypeId::UNKNOWN) {
			throw ParameterNotResolvedException();
		}
		if (input_type.id() == LogicalTypeId::SQLNULL) {
			argument = BoundCastExpression::AddCastToType(context, std::move(argument), LogicalType::VARCHAR);
			input_type = LogicalType::VARCHAR;
		}

		LogicalType output_type;
		if (input_type == LogicalType::VARCHAR) {
			output_type = LogicalType::VARCHAR;
		} else if (input_type == LogicalType::BLOB) {
			if (!media_supported) {
				throw BinderException("Prompt messages for this provider must have type VARCHAR, FILE, or FILE[]");
			}
			output_type = LogicalType::BLOB;
		} else if (input_type == LogicalType::LIST(LogicalType::BLOB)) {
			if (!media_supported) {
				throw BinderException("Prompt messages for this provider must have type VARCHAR, FILE, or FILE[]");
			}
			output_type = input_type;
		} else if (FileLogicalType::IsFile(input_type)) {
			output_type = PromptMediaLogicalType();
		} else if (IsPromptFileList(input_type)) {
			output_type = LogicalType::LIST(PromptMediaLogicalType());
		} else {
			throw BinderException("Prompt messages must have type VARCHAR, BLOB, BLOB[], FILE, or FILE[]");
		}
		fields.emplace_back(StringUtil::Format("message_%d", argument_index - PROMPT_PACK_MESSAGE_OFFSET),
		                    std::move(output_type));
	}

	auto return_type = LogicalType::STRUCT(std::move(fields));
	bound_function.SetReturnType(return_type);
	return make_uniq<VariableReturnBindData>(std::move(return_type));
}

static bool IsPromptFileList(const LogicalType &type) {
	return type.id() == LogicalTypeId::LIST && FileLogicalType::IsFile(ListType::GetChildType(type));
}

static const char *PromptInputKindName(PromptInputKind kind) {
	switch (kind) {
	case PromptInputKind::TEXT:
		return "text";
	case PromptInputKind::BLOB:
		return "blob";
	case PromptInputKind::BLOB_LIST:
		return "blob_list";
	case PromptInputKind::FILE:
		return "file";
	case PromptInputKind::FILE_LIST:
		return "file_list";
	}
	throw InternalException("unknown ai_prompt input kind");
}

struct NativeVLLMSpec {
	string model;
	Value options;
	Value system_message;
};

struct NativeVLLMAISQLFunctionData : public FunctionData {
	NativeVLLMAISQLFunctionData(string model_p, Value options_p, Value validation_payload_p, LogicalType return_type_p)
	    : model(std::move(model_p)), options(std::move(options_p)), validation_payload(std::move(validation_payload_p)),
	      return_type(std::move(return_type_p)) {
	}

	string model;
	Value options;
	Value validation_payload;
	LogicalType return_type;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<NativeVLLMAISQLFunctionData>(model, options, validation_payload, return_type);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<NativeVLLMAISQLFunctionData>();
		return model == other.model && options == other.options && validation_payload == other.validation_payload &&
		       return_type == other.return_type;
	}
};

static void ThrowIfNotConstant(const Expression &arg, const string &name) {
	if (!arg.IsFoldable()) {
		throw BinderException("ai SQL: argument '%s' must be constant", name);
	}
}

static Value EvaluateConstant(ClientContext &context, Expression &arg) {
	if (arg.HasParameter()) {
		throw ParameterNotResolvedException();
	}
	return ExpressionExecutor::EvaluateScalar(context, arg);
}

static bool IsFoldableNull(ClientContext &context, const Expression &arg) {
	Value value;
	return arg.IsFoldable() && ExpressionExecutor::TryEvaluateScalar(context, arg, value) && value.IsNull();
}

static vector<string> ParseInputNames(const py::object &input_names) {
	if (!py::isinstance<py::list>(input_names) && !py::isinstance<py::tuple>(input_names)) {
		throw BinderException("ai SQL helper returned invalid input_names");
	}
	auto names = py::list(input_names);
	if (names.empty()) {
		throw BinderException("ai SQL helper returned empty input_names");
	}
	vector<string> result;
	result.reserve(names.size());
	for (auto &name_obj : names) {
		if (!py::isinstance<py::str>(name_obj)) {
			throw BinderException("ai SQL helper returned non-string input_names");
		}
		result.push_back(py::cast<string>(name_obj));
	}
	return result;
}

static vector<string> ParsePromptMediaMimeTypes(const py::object &mime_types) {
	if (!py::isinstance<py::list>(mime_types) && !py::isinstance<py::tuple>(mime_types)) {
		throw BinderException("ai SQL helper returned invalid supported_media_mime_types");
	}
	vector<string> result;
	auto values = py::list(mime_types);
	result.reserve(values.size());
	for (auto &value : values) {
		if (!py::isinstance<py::str>(value)) {
			throw BinderException("ai SQL helper returned a non-string supported media MIME type");
		}
		string normalized;
		if (!NormalizePromptContentType(py::cast<string>(value), normalized)) {
			throw BinderException("ai SQL helper returned an invalid supported media MIME type");
		}
		result.push_back(std::move(normalized));
	}
	return result;
}

static py::object DictGetOrNone(const py::dict &dict, const char *key) {
	auto py_key = py::str(key);
	if (!dict.contains(py_key)) {
		return py::none();
	}
	return py::reinterpret_borrow<py::object>(dict[py_key]);
}

static idx_t OptionsArgumentIndex(AISQLKind kind, idx_t argument_count) {
	if (kind == AISQLKind::EMBED) {
		if (argument_count == 6) {
			return 5;
		}
		throw BinderException("%s requires six arguments supplied by the ai_embed macro", HIDDEN_EMBED_FUNCTION);
	}
	if (argument_count == 8) {
		return 7;
	}
	if (argument_count == 9) {
		return 8;
	}
	throw BinderException("%s requires eight or nine arguments supplied by the ai_prompt macro",
	                      HIDDEN_PROMPT_FUNCTION);
}

static py::object OptionsToPython(ClientContext &context, vector<unique_ptr<Expression>> &arguments,
                                  idx_t options_index, bool require_struct) {
	if (options_index >= arguments.size()) {
		return py::none();
	}
	auto &options_arg = *arguments[options_index];
	ThrowIfNotConstant(options_arg, "options");
	auto options = EvaluateConstant(context, options_arg);
	if (options.IsNull()) {
		return py::none();
	}
	if (require_struct && options.type().id() != LogicalTypeId::STRUCT) {
		throw BinderException("ai SQL options must be NULL or a foldable STRUCT, not %s", options.type().ToString());
	}
	return PythonObject::FromValue(options, options.type(), context.GetClientProperties());
}

static py::object ConstantArgumentToPython(ClientContext &context, vector<unique_ptr<Expression>> &arguments,
                                           idx_t index, const string &name) {
	auto &argument = *arguments[index];
	ThrowIfNotConstant(argument, name);
	auto value = EvaluateConstant(context, argument);
	if (value.IsNull()) {
		return py::none();
	}
	return PythonObject::FromValue(value, value.type(), context.GetClientProperties());
}

static py::dict BuildAISQLSpec(AISQLKind kind, ClientContext &context, vector<unique_ptr<Expression>> &arguments,
                               idx_t options_index, PromptInputKind prompt_input_kind) {
	auto sql_module = py::module_::import("vane.ai._sql");
	auto py_options = OptionsToPython(context, arguments, options_index, true);
	if (kind == AISQLKind::PROMPT) {
		auto has_media_input = prompt_input_kind != PromptInputKind::TEXT;
		auto constant_offset = has_media_input ? idx_t(2) : idx_t(1);
		auto return_format = ConstantArgumentToPython(context, arguments, constant_offset, "return_format");
		auto system_message = ConstantArgumentToPython(context, arguments, constant_offset + 1, "system_message");
		auto provider = ConstantArgumentToPython(context, arguments, constant_offset + 2, "provider");
		auto model = ConstantArgumentToPython(context, arguments, constant_offset + 3, "model");
		auto return_raw_response =
		    ConstantArgumentToPython(context, arguments, constant_offset + 4, "return_raw_response");
		auto on_error = ConstantArgumentToPython(context, arguments, constant_offset + 5, "on_error");
		return py::cast<py::dict>(sql_module.attr("build_ai_prompt_sql_spec")(
		    provider, model, system_message, on_error, py_options, PromptInputKindName(prompt_input_kind),
		    return_format, return_raw_response));
	}
	auto provider = ConstantArgumentToPython(context, arguments, 1, "provider");
	auto model = ConstantArgumentToPython(context, arguments, 2, "model");
	auto dimensions = ConstantArgumentToPython(context, arguments, 3, "dimensions");
	auto on_error = ConstantArgumentToPython(context, arguments, 4, "on_error");
	return py::cast<py::dict>(
	    sql_module.attr("build_ai_embed_sql_spec")(provider, model, dimensions, on_error, py_options));
}

static string ParseExecutionKind(const py::dict &spec) {
	auto execution_kind = DictGetOrNone(spec, "execution_kind");
	if (execution_kind.is_none()) {
		return "expression_udf";
	}
	if (!py::isinstance<py::str>(execution_kind)) {
		throw BinderException("ai SQL helper returned invalid execution_kind");
	}
	return py::cast<string>(execution_kind);
}

static NativeVLLMSpec ParseNativeVLLMSpec(const py::dict &spec) {
	auto model_obj = DictGetOrNone(spec, "model");
	auto options_obj = DictGetOrNone(spec, "options");
	if (!py::isinstance<py::str>(model_obj) || !py::isinstance<py::dict>(options_obj)) {
		throw BinderException("ai SQL native vLLM helper returned invalid model or options envelope");
	}
	auto system_message_obj = DictGetOrNone(spec, "system_message");
	Value system_message;
	if (!system_message_obj.is_none()) {
		if (!py::isinstance<py::str>(system_message_obj)) {
			throw BinderException("ai SQL native vLLM helper returned invalid system_message");
		}
		system_message = Value(py::cast<string>(system_message_obj));
	}
	return {py::cast<string>(model_obj), TransformPythonValue(options_obj), std::move(system_message)};
}

static Value BuildAISQLPayload(ClientContext &context, const py::dict &spec) {
	auto expression_helpers = py::module_::import("vane._expression_udf");
	auto normalize_schema = expression_helpers.attr("_normalize_schema");

	auto name = py::cast<string>(spec[py::str("name")]);
	auto udf = py::cast<py::function>(spec[py::str("function")]);
	auto input_names = ParseInputNames(py::reinterpret_borrow<py::object>(spec[py::str("input_names")]));
	auto schema = py::reinterpret_borrow<py::object>(normalize_schema(spec[py::str("schema")]));
	auto batch_size = DictGetOrNone(spec, "batch_size");
	auto gpus = DictGetOrNone(spec, "gpus");
	auto actor_number = DictGetOrNone(spec, "actor_number");
	auto dimensions = DictGetOrNone(spec, "dimensions");
	auto provider = py::cast<string>(spec[py::str("provider")]);
	auto model = py::cast<string>(spec[py::str("model")]);
	auto return_type = py::cast<string>(spec[py::str("return_type")]);

	auto default_parallelism = static_cast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads());
	auto payload =
	    BuildExpressionMapBatchesUDFPayload(name, udf, schema, "subprocess_actor", default_parallelism, input_names,
	                                        batch_size, /*row_preserving=*/true, gpus, actor_number, py::none());
	return AddAISQLPayloadMetadata(payload, provider, model, return_type, dimensions);
}

static unique_ptr<Expression> BindScalarFunction(ClientContext &context, const string &name,
                                                 vector<unique_ptr<Expression>> children) {
	FunctionBinder binder(context);
	ErrorData error;
	auto result = binder.BindScalarFunction(DEFAULT_SCHEMA, name, std::move(children), error);
	if (!result) {
		error.Throw();
	}
	return result;
}

static unique_ptr<Expression> BindPromptPack(ClientContext &context, vector<unique_ptr<Expression>> messages,
                                             bool media_supported, const vector<string> &supported_mime_types,
                                             bool single_message) {
	vector<unique_ptr<Expression>> children;
	children.reserve(PROMPT_PACK_MESSAGE_OFFSET + messages.size());
	children.push_back(make_uniq<BoundConstantExpression>(Value::BOOLEAN(media_supported)));
	vector<Value> mime_type_values;
	mime_type_values.reserve(supported_mime_types.size());
	for (auto &mime_type : supported_mime_types) {
		mime_type_values.emplace_back(mime_type);
	}
	children.push_back(
	    make_uniq<BoundConstantExpression>(Value::LIST(LogicalType::VARCHAR, std::move(mime_type_values))));
	children.push_back(make_uniq<BoundConstantExpression>(Value::BOOLEAN(single_message)));
	for (auto &message : messages) {
		children.push_back(std::move(message));
	}
	return BindScalarFunction(context, HIDDEN_PROMPT_PACK_FUNCTION, std::move(children));
}

static unique_ptr<Expression> CastPromptOutput(ClientContext &context, unique_ptr<Expression> result,
                                               const LogicalType &target_type) {
	if (result->return_type == target_type) {
		return result;
	}
	if (target_type.id() == LogicalTypeId::STRUCT) {
		// Structured Prompt UDFs return validated JSON text, not DuckDB's STRUCT literal syntax.
		result = BoundCastExpression::AddCastToType(context, std::move(result), LogicalType::JSON());
	}
	return BoundCastExpression::AddCastToType(context, std::move(result), target_type);
}

static unique_ptr<Expression> BuildNativeVLLMPromptArgument(ClientContext &context, unique_ptr<Expression> prompt,
                                                            const Value &system_message) {
	if (system_message.IsNull() || StringValue::Get(system_message).empty()) {
		return prompt;
	}

	vector<unique_ptr<Expression>> concat_arguments;
	auto prefix = StringValue::Get(system_message) + "\n\n";
	concat_arguments.push_back(make_uniq<BoundConstantExpression>(Value(std::move(prefix))));
	concat_arguments.push_back(std::move(prompt));
	return BindScalarFunction(context, "||", std::move(concat_arguments));
}

static unique_ptr<Expression> LowerNativeVLLMPrompt(FunctionBindExpressionInput &input) {
	if (!input.bind_data) {
		throw BinderException("native vLLM ai_prompt is missing bind data");
	}
	auto &data = input.bind_data->Cast<NativeVLLMAISQLFunctionData>();
	if (input.children.size() != 1) {
		throw BinderException("native vLLM ai_prompt expected one runtime argument");
	}

	vector<unique_ptr<Expression>> children;
	children.reserve(3);
	children.push_back(std::move(input.children[0]));
	children.push_back(make_uniq<BoundConstantExpression>(Value(data.model)));
	children.push_back(make_uniq<BoundConstantExpression>(data.options));
	auto result = BindScalarFunction(input.context, "vllm", std::move(children));
	if (!data.validation_payload.IsNull()) {
		vector<unique_ptr<Expression>> validation_children;
		validation_children.reserve(2);
		validation_children.push_back(std::move(result));
		validation_children.push_back(make_uniq<BoundConstantExpression>(data.validation_payload));
		result = BindScalarFunction(input.context, UDFFunction::Name, std::move(validation_children));
	}
	return CastPromptOutput(input.context, std::move(result), data.return_type);
}

static unique_ptr<FunctionData> AISQLBind(ClientContext &context, ScalarFunction &bound_function,
                                          vector<unique_ptr<Expression>> &arguments, AISQLKind kind) {
	auto options_index = OptionsArgumentIndex(kind, arguments.size());
	auto has_media_input = kind == AISQLKind::PROMPT && arguments.size() == 9;
	auto runtime_argument_count = has_media_input ? idx_t(2) : idx_t(1);
	auto prompt_input_kind = PromptInputKind::TEXT;
	auto input_type_id = arguments[0]->return_type.id();
	if (input_type_id != LogicalTypeId::VARCHAR && input_type_id != LogicalTypeId::SQLNULL) {
		throw BinderException("ai SQL input argument must be VARCHAR");
	}
	if (has_media_input) {
		auto media_type = arguments[1]->return_type;
		if (media_type.id() == LogicalTypeId::SQLNULL && bound_function.arguments.size() > 1) {
			media_type = bound_function.arguments[1];
		}
		if (media_type == LogicalType::BLOB) {
			prompt_input_kind = PromptInputKind::BLOB;
		} else if (media_type == LogicalType::LIST(LogicalType::BLOB)) {
			prompt_input_kind = PromptInputKind::BLOB_LIST;
		} else if (FileLogicalType::IsFile(media_type)) {
			prompt_input_kind = PromptInputKind::FILE;
		} else if (IsPromptFileList(media_type)) {
			prompt_input_kind = PromptInputKind::FILE_LIST;
		} else {
			throw BinderException("ai_prompt media argument must be BLOB, BLOB[], FILE, or FILE[]");
		}
	}
	Value payload;
	Value native_validation_payload;
	LogicalType public_return_type;
	unique_ptr<NativeVLLMSpec> native_vllm;
	bool supports_prompt_media = true;
	vector<string> supported_prompt_media_mime_types;
	{
		PythonGILWrapper acquire;
		auto spec = BuildAISQLSpec(kind, context, arguments, options_index, prompt_input_kind);
		auto return_type_obj = DictGetOrNone(spec, "return_type");
		if (!py::isinstance<py::str>(return_type_obj)) {
			throw BinderException("ai SQL helper returned invalid return_type");
		}
		public_return_type = TransformStringToLogicalType(py::cast<string>(return_type_obj), context);
		auto execution_kind = ParseExecutionKind(spec);
		if (execution_kind == "native_vllm") {
			if (kind != AISQLKind::PROMPT) {
				throw BinderException("native vLLM execution is only valid for ai_prompt");
			}
			native_vllm = make_uniq<NativeVLLMSpec>(ParseNativeVLLMSpec(spec));
			auto validation_spec = DictGetOrNone(spec, "validation_spec");
			if (!validation_spec.is_none()) {
				if (!py::isinstance<py::dict>(validation_spec)) {
					throw BinderException("ai SQL native vLLM helper returned invalid validation_spec");
				}
				native_validation_payload =
				    BuildAISQLPayload(context, py::reinterpret_borrow<py::dict>(validation_spec));
			}
		} else if (execution_kind == "expression_udf") {
			if (prompt_input_kind == PromptInputKind::FILE || prompt_input_kind == PromptInputKind::FILE_LIST) {
				auto supports_media_obj = DictGetOrNone(spec, "supports_media_inputs");
				if (!py::isinstance<py::bool_>(supports_media_obj)) {
					throw BinderException("ai SQL helper returned invalid supports_media_inputs");
				}
				supports_prompt_media = py::cast<bool>(supports_media_obj);
				supported_prompt_media_mime_types =
				    ParsePromptMediaMimeTypes(DictGetOrNone(spec, "supported_media_mime_types"));
			}
			payload = BuildAISQLPayload(context, spec);
		} else {
			throw BinderException("ai SQL helper returned unknown execution_kind '%s'", execution_kind);
		}
	}
	if (prompt_input_kind == PromptInputKind::FILE || prompt_input_kind == PromptInputKind::FILE_LIST) {
		if (arguments[1]->return_type.id() == LogicalTypeId::SQLNULL) {
			auto file_type = FileLogicalType::Create();
			auto target_type =
			    prompt_input_kind == PromptInputKind::FILE_LIST ? LogicalType::LIST(file_type) : file_type;
			arguments[1] = BoundCastExpression::AddCastToType(context, std::move(arguments[1]), target_type);
		}
		vector<unique_ptr<Expression>> messages;
		messages.reserve(2);
		messages.push_back(std::move(arguments[0]));
		messages.push_back(std::move(arguments[1]));
		arguments[0] = BindPromptPack(context, std::move(messages), supports_prompt_media,
		                              supported_prompt_media_mime_types, true);
		bound_function.arguments[0] = arguments[0]->return_type;
		Function::EraseArgument(bound_function, arguments, 1);
		runtime_argument_count = 1;
	}

	if (native_vllm) {
		arguments[0] = BuildNativeVLLMPromptArgument(context, std::move(arguments[0]), native_vllm->system_message);
		for (idx_t index = arguments.size(); index-- > runtime_argument_count;) {
			Function::EraseArgument(bound_function, arguments, index);
		}
		bound_function.SetReturnType(public_return_type);
		bound_function.SetBindExpressionCallback(LowerNativeVLLMPrompt);
		return make_uniq<NativeVLLMAISQLFunctionData>(std::move(native_vllm->model), std::move(native_vllm->options),
		                                              std::move(native_validation_payload),
		                                              std::move(public_return_type));
	}
	auto internal_return_type = udf_helpers::ResolvePayloadReturnType(payload);
	bound_function.SetReturnType(public_return_type);
	if (kind == AISQLKind::EMBED) {
		// The public macro forwards five call-level constants after the text
		// expression. They are fully consumed by this binder and must not become
		// row inputs to the lowered expression UDF.
		for (idx_t index = arguments.size(); index-- > 1;) {
			Function::EraseArgument(bound_function, arguments, index);
		}
	} else {
		// Prompt call-level constants are consumed by the binder. FILE inputs
		// are one packed runtime value; eager BLOB inputs remain two columns.
		for (idx_t index = arguments.size(); index-- > runtime_argument_count;) {
			Function::EraseArgument(bound_function, arguments, index);
		}
	}
	bound_function.SetExtraFunctionInfo(make_shared_ptr<RegisteredUDFFunctionInfo>(payload));
	return make_uniq<UDFFunctionData>(std::move(payload), std::move(internal_return_type));
}

static unique_ptr<FunctionData> AISQLPromptBind(ClientContext &context, ScalarFunction &bound_function,
                                                vector<unique_ptr<Expression>> &arguments) {
	return AISQLBind(context, bound_function, arguments, AISQLKind::PROMPT);
}

static unique_ptr<FunctionData> AISQLEmbedBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {
	return AISQLBind(context, bound_function, arguments, AISQLKind::EMBED);
}

static void AISQLExecute(DataChunk &, ExpressionState &, Vector &) {
	throw InvalidInputException(
	    "ai SQL functions can only be used in a projection and must be planned as UDF operators");
}

static unique_ptr<Expression> LowerAISQLPromptExpressionUDF(FunctionBindExpressionInput &input) {
	if (!input.bind_data) {
		throw BinderException("registered expression UDF is missing bind payload");
	}
	auto &registered_data = input.bind_data->Cast<UDFFunctionData>();
	if (input.children.empty() || input.children.size() > 2) {
		throw BinderException("ai_prompt expected one or two runtime arguments");
	}
	if (IsFoldableNull(input.context, *input.children[0])) {
		auto public_type = UDFPayloadStringField(registered_data.payload, "ai_return_type");
		if (!public_type.first) {
			throw BinderException("ai_prompt payload is missing ai_return_type");
		}
		return make_uniq<BoundConstantExpression>(
		    Value(TransformStringToLogicalType(public_type.second, input.context)));
	}
	auto result = LowerRegisteredExpressionUDFPreservingFoldableNulls(input);
	auto public_type = UDFPayloadStringField(registered_data.payload, "ai_return_type");
	if (!public_type.first) {
		throw BinderException("ai_prompt payload is missing ai_return_type");
	}
	auto target_type = TransformStringToLogicalType(public_type.second, input.context);
	return CastPromptOutput(input.context, std::move(result), target_type);
}

static unique_ptr<Expression> LowerAISQLEmbedExpressionUDF(FunctionBindExpressionInput &input) {
	if (!input.bind_data) {
		throw BinderException("registered expression UDF is missing bind payload");
	}
	if (input.children.size() != 1) {
		throw BinderException("ai_embed expected one runtime argument");
	}
	if (IsFoldableNull(input.context, *input.children[0])) {
		auto &registered_data = input.bind_data->Cast<UDFFunctionData>();
		return make_uniq<BoundConstantExpression>(Value(registered_data.return_type));
	}
	return LowerRegisteredExpressionUDF(input);
}

static unique_ptr<Expression> LowerAIEmbedTextInput(FunctionBindExpressionInput &input) {
	if (input.children.size() != 1) {
		throw BinderException("ai_embed text validation expected one runtime argument");
	}
	auto &text = input.children[0];
	auto input_type_id = text->return_type.id();
	if (input_type_id == LogicalTypeId::UNKNOWN) {
		throw ParameterNotResolvedException();
	}
	if (input_type_id == LogicalTypeId::SQLNULL) {
		return make_uniq<BoundConstantExpression>(Value(LogicalType::VARCHAR));
	}
	if (input_type_id != LogicalTypeId::VARCHAR) {
		throw BinderException("ai SQL input argument must be VARCHAR");
	}
	return std::move(text);
}

static unique_ptr<Expression> LowerAIPromptInput(FunctionBindExpressionInput &input) {
	if (input.children.size() != 1) {
		throw BinderException("ai_prompt input validation expected one runtime argument");
	}
	auto &message = input.children[0];
	auto &input_type = message->return_type;
	if (input_type.id() == LogicalTypeId::UNKNOWN) {
		throw ParameterNotResolvedException();
	}
	if (input_type.id() == LogicalTypeId::SQLNULL) {
		return make_uniq<BoundConstantExpression>(Value(LogicalType::VARCHAR));
	}
	if (input_type != LogicalType::VARCHAR && input_type != LogicalType::BLOB &&
	    input_type != LogicalType::LIST(LogicalType::BLOB) && !FileLogicalType::IsFile(input_type) &&
	    !IsPromptFileList(input_type)) {
		throw BinderException("Prompt messages must have type VARCHAR, BLOB, BLOB[], FILE, or FILE[]");
	}
	return std::move(message);
}

static unique_ptr<Expression> LowerAIPromptTextInput(FunctionBindExpressionInput &input) {
	if (input.children.size() != 2) {
		throw BinderException("ai_prompt text validation expected two runtime arguments");
	}
	ThrowIfNotConstant(*input.children[1], "media_policy");
	auto policy = EvaluateConstant(input.context, *input.children[1]);
	if (policy.IsNull() || policy.type().id() != LogicalTypeId::BOOLEAN) {
		throw BinderException("ai_prompt media policy must be a constant BOOLEAN");
	}
	auto text_only = BooleanValue::Get(policy);
	auto &message = input.children[0];
	auto input_type_id = message->return_type.id();
	if (input_type_id == LogicalTypeId::UNKNOWN) {
		throw ParameterNotResolvedException();
	}
	if (input_type_id == LogicalTypeId::SQLNULL) {
		return make_uniq<BoundConstantExpression>(Value(LogicalType::VARCHAR));
	}
	if (input_type_id != LogicalTypeId::VARCHAR &&
	    (text_only || (!FileLogicalType::IsFile(message->return_type) && !IsPromptFileList(message->return_type)))) {
		if (text_only) {
			throw BinderException("Native Prompt messages must have type VARCHAR");
		}
		throw BinderException("Prompt messages for this provider must have type VARCHAR, FILE, or FILE[]");
	}
	return std::move(message);
}

} // namespace

ScalarFunctionSet AISQLFunction::GetPromptPackFunctions() {
	ScalarFunctionSet set(HIDDEN_PROMPT_PACK_FUNCTION);
	auto pack = ScalarFunction({LogicalType::BOOLEAN, LogicalType::LIST(LogicalType::VARCHAR), LogicalType::BOOLEAN},
	                           LogicalTypeId::STRUCT, PromptPackFunction, PromptPackBind);
	pack.varargs = LogicalType::ANY;
	pack.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	pack.SetStability(FunctionStability::VOLATILE);
	pack.SetFallible();
	pack.SetSerializeCallback(VariableReturnBindData::Serialize);
	pack.SetDeserializeCallback(VariableReturnBindData::Deserialize);
	set.AddFunction(std::move(pack));
	return set;
}

ScalarFunctionSet AISQLFunction::GetPromptImplementationFunctions() {
	ScalarFunctionSet set(HIDDEN_PROMPT_FUNCTION);
	auto prompt_input = ScalarFunction({LogicalType::ANY}, LogicalType::ANY, AISQLExecute);
	prompt_input.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	prompt_input.SetBindExpressionCallback(LowerAIPromptInput);
	set.AddFunction(std::move(prompt_input));

	auto prompt_text_input =
	    ScalarFunction({LogicalType::ANY, LogicalType::BOOLEAN}, LogicalType::VARCHAR, AISQLExecute);
	prompt_text_input.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	prompt_text_input.SetBindExpressionCallback(LowerAIPromptTextInput);
	set.AddFunction(std::move(prompt_text_input));

	auto add_implementation = [&](vector<LogicalType> arguments) {
		auto implementation =
		    ScalarFunction(std::move(arguments), LogicalType::VARCHAR, AISQLExecute, AISQLPromptBind, nullptr, nullptr,
		                   nullptr, LogicalType::INVALID, FunctionStability::VOLATILE);
		implementation.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
		implementation.SetBindExpressionCallback(LowerAISQLPromptExpressionUDF);
		set.AddFunction(std::move(implementation));
	};

	add_implementation({LogicalType::VARCHAR, LogicalType::JSON(), LogicalType::VARCHAR, LogicalType::VARCHAR,
	                    LogicalType::VARCHAR, LogicalType::BOOLEAN, LogicalType::VARCHAR, LogicalType::ANY});
	add_implementation({LogicalType::VARCHAR, LogicalType::BLOB, LogicalType::JSON(), LogicalType::VARCHAR,
	                    LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BOOLEAN, LogicalType::VARCHAR,
	                    LogicalType::ANY});
	add_implementation({LogicalType::VARCHAR, LogicalType::LIST(LogicalType::BLOB), LogicalType::JSON(),
	                    LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BOOLEAN,
	                    LogicalType::VARCHAR, LogicalType::ANY});
	auto file_type = FileLogicalType::Create();
	add_implementation({LogicalType::VARCHAR, file_type, LogicalType::JSON(), LogicalType::VARCHAR,
	                    LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BOOLEAN, LogicalType::VARCHAR,
	                    LogicalType::ANY});
	add_implementation({LogicalType::VARCHAR, LogicalType::LIST(file_type), LogicalType::JSON(), LogicalType::VARCHAR,
	                    LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BOOLEAN, LogicalType::VARCHAR,
	                    LogicalType::ANY});
	return set;
}

unique_ptr<CreateMacroInfo> AISQLFunction::GetPromptMacro() {
	auto make_function = [&](const LogicalType *media_type, const char *media_parameter) {
		auto macro_arguments =
		    media_type
		        ? StringUtil::Format("prompt, %s, return_format, system_message, provider, model, return_raw_response, "
		                             "on_error, options",
		                             media_parameter)
		        : "prompt, return_format, system_message, provider, model, return_raw_response, on_error, options";
		auto expressions =
		    Parser::ParseExpressionList(StringUtil::Format("%s(%s)", HIDDEN_PROMPT_FUNCTION, macro_arguments));
		if (expressions.size() != 1) {
			throw InternalException("Expected one ai_prompt macro expression");
		}
		auto function = make_uniq<ScalarMacroFunction>(std::move(expressions[0]));

		auto add_parameter = [&](const string &name, const LogicalType &type, const char *default_sql) {
			function->parameters.push_back(make_uniq<ColumnRefExpression>(name));
			function->types.push_back(type);
			if (!default_sql) {
				return;
			}
			auto defaults = Parser::ParseExpressionList(default_sql);
			if (defaults.size() != 1) {
				throw InternalException("Expected one default expression for ai_prompt parameter '%s'", name);
			}
			function->default_parameters.insert(make_pair(name, std::move(defaults[0])));
		};

		add_parameter("prompt", LogicalType::VARCHAR, nullptr);
		if (media_type) {
			add_parameter(media_parameter, *media_type, nullptr);
		}
		// JSON registers a zero-cost implicit cast from BLOB, which makes a
		// positional image ambiguous with the text overload's return_format.
		// VARCHAR accepts JSON with a low-cost cast while excluding BLOB; the
		// hidden implementation then casts the value back to JSON.
		add_parameter("return_format", LogicalType::VARCHAR, "NULL");
		add_parameter("system_message", LogicalType::VARCHAR, "NULL");
		add_parameter("provider", LogicalType::VARCHAR, "'openai'");
		add_parameter("model", LogicalType::VARCHAR, "NULL");
		add_parameter("return_raw_response", LogicalType::BOOLEAN, "FALSE");
		add_parameter("on_error", LogicalType::VARCHAR, "'raise'");
		add_parameter("options", LogicalType::UNKNOWN, "NULL");
		return function;
	};

	auto info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	info->schema = DEFAULT_SCHEMA;
	info->name = "ai_prompt";
	info->temporary = true;
	info->internal = true;
	info->macros.push_back(make_function(nullptr, nullptr));
	LogicalType blob_type(LogicalTypeId::BLOB);
	info->macros.push_back(make_function(&blob_type, "image"));
	auto blob_list_type = LogicalType::LIST(LogicalType::BLOB);
	info->macros.push_back(make_function(&blob_list_type, "images"));
	auto file_type = FileLogicalType::Create();
	info->macros.push_back(make_function(&file_type, "file"));
	auto file_list_type = LogicalType::LIST(file_type);
	info->macros.push_back(make_function(&file_list_type, "files"));
	return info;
}

ScalarFunctionSet AISQLFunction::GetEmbedImplementationFunctions() {
	ScalarFunctionSet set(HIDDEN_EMBED_FUNCTION);
	auto text_input = ScalarFunction({LogicalType::ANY}, LogicalType::VARCHAR, AISQLExecute);
	text_input.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	text_input.SetBindExpressionCallback(LowerAIEmbedTextInput);
	set.AddFunction(std::move(text_input));

	auto implementation = ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                                      LogicalType::INTEGER, LogicalType::VARCHAR, LogicalType::ANY},
	                                     LogicalType::ANY, AISQLExecute, AISQLEmbedBind, nullptr, nullptr, nullptr,
	                                     LogicalType::INVALID, FunctionStability::VOLATILE);
	// model, dimensions, and options legitimately default to NULL. The binder
	// must still run so it can consume those call-level constants, resolve the
	// fixed output type, and preserve it for a NULL text input.
	implementation.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	implementation.SetBindExpressionCallback(LowerAISQLEmbedExpressionUDF);
	set.AddFunction(std::move(implementation));
	return set;
}

unique_ptr<CreateMacroInfo> AISQLFunction::GetEmbedMacro() {
	auto expressions = Parser::ParseExpressionList(
	    StringUtil::Format("%s(text, provider, model, dimensions, on_error, options)", HIDDEN_EMBED_FUNCTION));
	if (expressions.size() != 1) {
		throw InternalException("Expected one ai_embed macro expression");
	}
	auto function = make_uniq<ScalarMacroFunction>(std::move(expressions[0]));

	auto add_parameter = [&](const string &name, const LogicalType &type, const char *default_sql) {
		function->parameters.push_back(make_uniq<ColumnRefExpression>(name));
		function->types.push_back(type);
		if (!default_sql) {
			return;
		}
		auto defaults = Parser::ParseExpressionList(default_sql);
		if (defaults.size() != 1) {
			throw InternalException("Expected one default expression for ai_embed parameter '%s'", name);
		}
		function->default_parameters.insert(make_pair(name, std::move(defaults[0])));
	};

	add_parameter("text", LogicalType::VARCHAR, nullptr);
	add_parameter("provider", LogicalType::VARCHAR, "'openai'");
	add_parameter("model", LogicalType::VARCHAR, "NULL");
	add_parameter("dimensions", LogicalType::INTEGER, "NULL");
	add_parameter("on_error", LogicalType::VARCHAR, "'raise'");
	// Macro parameters cannot use ANY. UNKNOWN leaves options uncast so the
	// hidden binder can require a foldable STRUCT (or NULL) precisely.
	add_parameter("options", LogicalType::UNKNOWN, "NULL");

	auto info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	info->schema = DEFAULT_SCHEMA;
	info->name = "ai_embed";
	info->temporary = true;
	info->internal = true;
	info->macros.push_back(std::move(function));
	return info;
}

} // namespace duckdb
