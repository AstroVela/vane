// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "duckdb_python/ai_sql_functions.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/function/scalar/vllm_functions.hpp"
#include "duckdb/function/scalar/udf_functions.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb_python/pybind11/gil_wrapper.hpp"
#include "duckdb_python/python_objects.hpp"
#include "duckdb_python/python_udf_utils.hpp"

namespace duckdb {

namespace {

enum class AISQLKind : uint8_t { PROMPT, EMBED };

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

static py::object OptionsToPython(ClientContext &context, vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() == 1) {
		return py::none();
	}
	if (arguments.size() != 2) {
		throw BinderException("ai SQL functions require one or two arguments");
	}
	auto &options_arg = *arguments[1];
	ThrowIfNotConstant(options_arg, "options");
	auto options = EvaluateConstant(context, options_arg);
	if (options.IsNull()) {
		return py::none();
	}
	return PythonObject::FromValue(options, options.type(), context.GetClientProperties());
}

static py::dict BuildAISQLSpec(AISQLKind kind, const py::object &py_options) {
	auto sql_module = py::module_::import("vane.ai._sql");
	auto builder = kind == AISQLKind::PROMPT ? sql_module.attr("build_ai_prompt_sql_spec")
	                                         : sql_module.attr("build_ai_embed_sql_spec");
	return py::cast<py::dict>(builder(py_options));
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
	vector<unique_ptr<Expression>> concat_arguments;
	if (!system_message.IsNull() && !StringValue::Get(system_message).empty()) {
		auto prefix = StringValue::Get(system_message) + "\n\n";
		concat_arguments.push_back(make_uniq<BoundConstantExpression>(Value(std::move(prefix))));
	}
	concat_arguments.push_back(std::move(prompt));
	// concat ignores NULL arguments. The empty suffix preserves non-NULL text
	// while matching the previous Python batch wrapper's NULL-to-empty policy.
	concat_arguments.push_back(make_uniq<BoundConstantExpression>(Value("")));
	return BindScalarFunction(context, "concat", std::move(concat_arguments));
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
	if (arguments.empty() || arguments.size() > 2) {
		throw BinderException("ai SQL functions require one or two arguments");
	}
	if (arguments[0]->return_type.id() != LogicalTypeId::VARCHAR) {
		throw BinderException("ai SQL input argument must be VARCHAR");
	}

	Value payload;
	unique_ptr<NativeVLLMSpec> native_vllm;
	{
		PythonGILWrapper acquire;
		auto py_options = OptionsToPython(context, arguments);
		auto spec = BuildAISQLSpec(kind, py_options);
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
		Function::EraseArgument(bound_function, arguments, 1);
		bound_function.SetReturnType(LogicalType::VARCHAR);
		bound_function.SetBindExpressionCallback(LowerNativeVLLMPrompt);
		return make_uniq<VLLMFunctionData>(std::move(native_vllm->model), Value(std::move(native_vllm->options_json)));
	}
	auto return_type = udf_helpers::ResolvePayloadReturnType(payload);
	bound_function.SetReturnType(return_type);
	if (arguments.size() == 2) {
		Function::EraseArgument(bound_function, arguments, 1);
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

static void AddAISQLFunctions(ScalarFunctionSet &set, bind_scalar_function_t bind) {
	auto base = ScalarFunction({LogicalType::VARCHAR}, LogicalType::ANY, AISQLExecute, bind, nullptr, nullptr, nullptr,
	                           LogicalType::INVALID, FunctionStability::VOLATILE);
	base.SetBindExpressionCallback(LowerRegisteredExpressionUDF);
	set.AddFunction(std::move(base));

	auto with_options = ScalarFunction({LogicalType::VARCHAR, LogicalType::ANY}, LogicalType::ANY, AISQLExecute, bind,
	                                   nullptr, nullptr, nullptr, LogicalType::ANY, FunctionStability::VOLATILE);
	with_options.SetBindExpressionCallback(LowerRegisteredExpressionUDF);
	set.AddFunction(std::move(with_options));
}

} // namespace

ScalarFunctionSet AISQLFunction::GetPromptFunctions() {
	ScalarFunctionSet set("ai_prompt");
	AddAISQLFunctions(set, AISQLPromptBind);
	return set;
}

ScalarFunctionSet AISQLFunction::GetEmbedFunctions() {
	ScalarFunctionSet set("ai_embed");
	AddAISQLFunctions(set, AISQLEmbedBind);
	return set;
}

} // namespace duckdb
