// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "duckdb_python/ai_sql_functions.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/function.hpp"
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
		if (argument_count == 1) {
			return argument_count;
		}
		if (argument_count == 2) {
			return 1;
		}
		throw BinderException("ai_embed requires one or two arguments");
	}
	if (argument_count == 1) {
		return argument_count;
	}
	if (argument_count == 2) {
		return 1;
	}
	if (argument_count == 3) {
		return 2;
	}
	throw BinderException("ai_prompt requires one, two, or three arguments");
}

static py::object OptionsToPython(ClientContext &context, vector<unique_ptr<Expression>> &arguments,
                                  idx_t options_index) {
	if (options_index >= arguments.size()) {
		return py::none();
	}
	auto &options_arg = *arguments[options_index];
	ThrowIfNotConstant(options_arg, "options");
	auto options = EvaluateConstant(context, options_arg);
	if (options.IsNull()) {
		return py::none();
	}
	return PythonObject::FromValue(options, options.type(), context.GetClientProperties());
}

static Value BuildAISQLPayload(ClientContext &context, AISQLKind kind, const py::object &py_options, bool image_input) {
	auto sql_module = py::module_::import("vane.ai._sql");
	auto expression_helpers = py::module_::import("vane._expression_udf");
	auto normalize_schema = expression_helpers.attr("_normalize_schema");

	auto builder = kind == AISQLKind::PROMPT ? sql_module.attr("build_ai_prompt_sql_spec")
	                                         : sql_module.attr("build_ai_embed_sql_spec");
	auto spec = py::cast<py::dict>(kind == AISQLKind::PROMPT ? builder(py_options, image_input) : builder(py_options));
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

static unique_ptr<FunctionData> AISQLBind(ClientContext &context, ScalarFunction &bound_function,
                                          vector<unique_ptr<Expression>> &arguments, AISQLKind kind) {
	auto options_index = OptionsArgumentIndex(kind, arguments.size());
	auto image_input = kind == AISQLKind::PROMPT && arguments.size() == 3;
	auto input_type_id = arguments[0]->return_type.id();
	if (input_type_id != LogicalTypeId::VARCHAR && !(image_input && input_type_id == LogicalTypeId::SQLNULL)) {
		throw BinderException("ai SQL input argument must be VARCHAR");
	}
	if (image_input && arguments[1]->return_type != LogicalType::BLOB &&
	    arguments[1]->return_type != LogicalType::LIST(LogicalType::BLOB) &&
	    arguments[1]->return_type.id() != LogicalTypeId::SQLNULL) {
		throw BinderException("ai_prompt image argument must be BLOB or BLOB[]");
	}
	if (image_input && IsFoldableNull(context, *arguments[0])) {
		auto return_type = LogicalType::VARCHAR;
		bound_function.SetReturnType(return_type);
		// A NULL payload is a local bind-state marker consumed by the image lowerer.
		return make_uniq<UDFFunctionData>(Value(LogicalType::SQLNULL), std::move(return_type));
	}

	Value payload;
	{
		PythonGILWrapper acquire;
		auto py_options = OptionsToPython(context, arguments, options_index);
		payload = BuildAISQLPayload(context, kind, py_options, image_input);
	}
	auto return_type = udf_helpers::ResolvePayloadReturnType(payload);
	bound_function.SetReturnType(return_type);
	if (options_index < arguments.size()) {
		Function::EraseArgument(bound_function, arguments, options_index);
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

static unique_ptr<Expression> LowerAISQLImageExpressionUDF(FunctionBindExpressionInput &input) {
	if (!input.bind_data) {
		throw BinderException("registered expression UDF is missing bind payload");
	}
	auto &registered_data = input.bind_data->Cast<UDFFunctionData>();
	if (registered_data.payload.IsNull()) {
		return make_uniq<BoundConstantExpression>(Value(registered_data.return_type));
	}
	return LowerRegisteredExpressionUDFPreservingFoldableNulls(input);
}

static void AddAISQLFunctions(ScalarFunctionSet &set, bind_scalar_function_t bind, bool include_image_inputs) {
	auto base = ScalarFunction({LogicalType::VARCHAR}, LogicalType::ANY, AISQLExecute, bind, nullptr, nullptr, nullptr,
	                           LogicalType::INVALID, FunctionStability::VOLATILE);
	base.SetBindExpressionCallback(LowerRegisteredExpressionUDF);
	set.AddFunction(std::move(base));

	auto with_options = ScalarFunction({LogicalType::VARCHAR, LogicalType::ANY}, LogicalType::ANY, AISQLExecute, bind,
	                                   nullptr, nullptr, nullptr, LogicalType::INVALID, FunctionStability::VOLATILE);
	with_options.SetBindExpressionCallback(LowerRegisteredExpressionUDF);
	set.AddFunction(std::move(with_options));

	if (!include_image_inputs) {
		return;
	}
	auto with_image =
	    ScalarFunction({LogicalType::VARCHAR, LogicalType::BLOB, LogicalType::ANY}, LogicalType::ANY, AISQLExecute,
	                   bind, nullptr, nullptr, nullptr, LogicalType::INVALID, FunctionStability::VOLATILE);
	with_image.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	with_image.SetBindExpressionCallback(LowerAISQLImageExpressionUDF);
	set.AddFunction(std::move(with_image));

	auto with_images = ScalarFunction({LogicalType::VARCHAR, LogicalType::LIST(LogicalType::BLOB), LogicalType::ANY},
	                                  LogicalType::ANY, AISQLExecute, bind, nullptr, nullptr, nullptr,
	                                  LogicalType::INVALID, FunctionStability::VOLATILE);
	with_images.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	with_images.SetBindExpressionCallback(LowerAISQLImageExpressionUDF);
	set.AddFunction(std::move(with_images));
}

} // namespace

ScalarFunctionSet AISQLFunction::GetPromptFunctions() {
	ScalarFunctionSet set("ai_prompt");
	AddAISQLFunctions(set, AISQLPromptBind, true);
	return set;
}

ScalarFunctionSet AISQLFunction::GetEmbedFunctions() {
	ScalarFunctionSet set("ai_embed");
	AddAISQLFunctions(set, AISQLEmbedBind, false);
	return set;
}

} // namespace duckdb
