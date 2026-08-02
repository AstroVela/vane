// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "duckdb_python/ai_sql_functions.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/function/scalar_macro_function.hpp"
#include "duckdb/function/scalar/vllm_functions.hpp"
#include "duckdb/function/scalar/udf_functions.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb_python/pybind11/gil_wrapper.hpp"
#include "duckdb_python/python_objects.hpp"
#include "duckdb_python/python_udf_utils.hpp"

namespace duckdb {

namespace {

enum class AISQLKind : uint8_t { PROMPT, EMBED };

static constexpr const char *HIDDEN_EMBED_FUNCTION = "__vane_ai_embed";
static constexpr const char *HIDDEN_PROMPT_FUNCTION = "__vane_ai_prompt";

struct NativeVLLMSpec {
	string model;
	string options_json;
	Value system_message;
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
	if (argument_count == 6) {
		return 5;
	}
	if (argument_count == 7) {
		return 6;
	}
	throw BinderException("%s requires six or seven arguments supplied by the ai_prompt macro", HIDDEN_PROMPT_FUNCTION);
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
                               idx_t options_index, bool image_input) {
	auto sql_module = py::module_::import("vane.ai._sql");
	auto py_options = OptionsToPython(context, arguments, options_index, true);
	if (kind == AISQLKind::PROMPT) {
		auto constant_offset = image_input ? idx_t(2) : idx_t(1);
		auto system_message = ConstantArgumentToPython(context, arguments, constant_offset, "system_message");
		auto provider = ConstantArgumentToPython(context, arguments, constant_offset + 1, "provider");
		auto model = ConstantArgumentToPython(context, arguments, constant_offset + 2, "model");
		auto on_error = ConstantArgumentToPython(context, arguments, constant_offset + 3, "on_error");
		return py::cast<py::dict>(sql_module.attr("build_ai_prompt_sql_spec")(provider, model, system_message, on_error,
		                                                                      py_options, image_input));
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
	auto options_obj = DictGetOrNone(spec, "options_json");
	if (!py::isinstance<py::str>(model_obj) || !py::isinstance<py::str>(options_obj)) {
		throw BinderException("ai SQL native vLLM helper returned invalid model or options_json");
	}
	auto system_message_obj = DictGetOrNone(spec, "system_message");
	Value system_message;
	if (!system_message_obj.is_none()) {
		if (!py::isinstance<py::str>(system_message_obj)) {
			throw BinderException("ai SQL native vLLM helper returned invalid system_message");
		}
		system_message = Value(py::cast<string>(system_message_obj));
	}
	return {py::cast<string>(model_obj), py::cast<string>(options_obj), std::move(system_message)};
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
	auto payload = BuildExpressionMapBatchesUDFPayload(name, udf, schema, "subprocess_actor", default_parallelism,
	                                                   input_names, batch_size, /*row_preserving=*/true, gpus,
	                                                   actor_number, /*stateful=*/false);
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
	auto &data = input.bind_data->Cast<VLLMFunctionData>();
	if (input.children.size() != 1) {
		throw BinderException("native vLLM ai_prompt expected one runtime argument");
	}

	vector<unique_ptr<Expression>> children;
	children.reserve(3);
	children.push_back(std::move(input.children[0]));
	children.push_back(make_uniq<BoundConstantExpression>(Value(data.model)));
	children.push_back(make_uniq<BoundConstantExpression>(data.options));
	return BindScalarFunction(input.context, "vllm", std::move(children));
}

static unique_ptr<FunctionData> AISQLBind(ClientContext &context, ScalarFunction &bound_function,
                                          vector<unique_ptr<Expression>> &arguments, AISQLKind kind) {
	auto options_index = OptionsArgumentIndex(kind, arguments.size());
	auto image_input = kind == AISQLKind::PROMPT && arguments.size() == 7;
	auto runtime_argument_count = image_input ? idx_t(2) : idx_t(1);
	auto input_type_id = arguments[0]->return_type.id();
	if (input_type_id != LogicalTypeId::VARCHAR && input_type_id != LogicalTypeId::SQLNULL) {
		throw BinderException("ai SQL input argument must be VARCHAR");
	}
	if (image_input && arguments[1]->return_type != LogicalType::BLOB &&
	    arguments[1]->return_type != LogicalType::LIST(LogicalType::BLOB) &&
	    arguments[1]->return_type.id() != LogicalTypeId::SQLNULL) {
		throw BinderException("ai_prompt image argument must be BLOB or BLOB[]");
	}
	Value payload;
	unique_ptr<NativeVLLMSpec> native_vllm;
	{
		PythonGILWrapper acquire;
		auto spec = BuildAISQLSpec(kind, context, arguments, options_index, image_input);
		auto execution_kind = ParseExecutionKind(spec);
		if (execution_kind == "native_vllm") {
			if (kind != AISQLKind::PROMPT) {
				throw BinderException("native vLLM execution is only valid for ai_prompt");
			}
			native_vllm = make_uniq<NativeVLLMSpec>(ParseNativeVLLMSpec(spec));
		} else if (execution_kind == "expression_udf") {
			payload = BuildAISQLPayload(context, spec);
		} else {
			throw BinderException("ai SQL helper returned unknown execution_kind '%s'", execution_kind);
		}
	}

	if (native_vllm) {
		arguments[0] = BuildNativeVLLMPromptArgument(context, std::move(arguments[0]), native_vllm->system_message);
		for (idx_t index = arguments.size(); index-- > runtime_argument_count;) {
			Function::EraseArgument(bound_function, arguments, index);
		}
		bound_function.SetReturnType(LogicalType::VARCHAR);
		bound_function.SetBindExpressionCallback(LowerNativeVLLMPrompt);
		return make_uniq<VLLMFunctionData>(std::move(native_vllm->model), Value(std::move(native_vllm->options_json)));
	}
	auto return_type = udf_helpers::ResolvePayloadReturnType(payload);
	bound_function.SetReturnType(return_type);
	if (kind == AISQLKind::EMBED) {
		// The public macro forwards five call-level constants after the text
		// expression. They are fully consumed by this binder and must not become
		// row inputs to the lowered expression UDF.
		for (idx_t index = arguments.size(); index-- > 1;) {
			Function::EraseArgument(bound_function, arguments, index);
		}
	} else {
		// Prompt call-level constants are consumed by the binder. Only the
		// prompt and optional image expression become row inputs.
		for (idx_t index = arguments.size(); index-- > runtime_argument_count;) {
			Function::EraseArgument(bound_function, arguments, index);
		}
	}
	bound_function.SetExtraFunctionInfo(make_shared_ptr<RegisteredUDFFunctionInfo>(payload));
	return make_uniq<UDFFunctionData>(std::move(payload), std::move(return_type));
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
		return make_uniq<BoundConstantExpression>(Value(registered_data.return_type));
	}
	return LowerRegisteredExpressionUDFPreservingFoldableNulls(input);
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

} // namespace

ScalarFunctionSet AISQLFunction::GetPromptImplementationFunctions() {
	ScalarFunctionSet set(HIDDEN_PROMPT_FUNCTION);
	auto add_implementation = [&](vector<LogicalType> arguments) {
		auto implementation =
		    ScalarFunction(std::move(arguments), LogicalType::VARCHAR, AISQLExecute, AISQLPromptBind, nullptr, nullptr,
		                   nullptr, LogicalType::INVALID, FunctionStability::VOLATILE);
		implementation.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
		implementation.SetBindExpressionCallback(LowerAISQLPromptExpressionUDF);
		set.AddFunction(std::move(implementation));
	};

	add_implementation({LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                    LogicalType::VARCHAR, LogicalType::ANY});
	add_implementation({LogicalType::VARCHAR, LogicalType::BLOB, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                    LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::ANY});
	add_implementation({LogicalType::VARCHAR, LogicalType::LIST(LogicalType::BLOB), LogicalType::VARCHAR,
	                    LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::ANY});
	return set;
}

unique_ptr<CreateMacroInfo> AISQLFunction::GetPromptMacro() {
	auto make_function = [&](const LogicalType *image_type) {
		auto macro_arguments = image_type ? "prompt, image, system_message, provider, model, on_error, options"
		                                  : "prompt, system_message, provider, model, on_error, options";
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
		if (image_type) {
			add_parameter("image", *image_type, nullptr);
		}
		add_parameter("system_message", LogicalType::VARCHAR, "NULL");
		add_parameter("provider", LogicalType::VARCHAR, "'openai'");
		add_parameter("model", LogicalType::VARCHAR, "NULL");
		add_parameter("on_error", LogicalType::VARCHAR, "'raise'");
		add_parameter("options", LogicalType::UNKNOWN, "NULL");
		return function;
	};

	auto info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	info->schema = DEFAULT_SCHEMA;
	info->name = "ai_prompt";
	info->temporary = true;
	info->internal = true;
	info->macros.push_back(make_function(nullptr));
	LogicalType blob_type(LogicalTypeId::BLOB);
	info->macros.push_back(make_function(&blob_type));
	auto blob_list_type = LogicalType::LIST(LogicalType::BLOB);
	info->macros.push_back(make_function(&blob_list_type));
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
