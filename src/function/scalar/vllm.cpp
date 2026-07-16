// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/function/scalar/vllm_functions.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

namespace duckdb {

VLLMFunctionData::VLLMFunctionData(string model_p, Value options_p)
    : model(std::move(model_p)), options(std::move(options_p)) {
}

unique_ptr<FunctionData> VLLMFunctionData::Copy() const {
	return make_uniq<VLLMFunctionData>(model, options);
}

bool VLLMFunctionData::Equals(const FunctionData &other_p) const {
	auto &other = other_p.Cast<VLLMFunctionData>();
	return model == other.model && options == other.options;
}

namespace {

void ThrowIfNotConstant(const Expression &arg, const string &name) {
	if (!arg.IsFoldable()) {
		throw BinderException("vllm: argument '%s' must be constant", name);
	}
}

Value EvaluateScalar(ClientContext &context, Expression &arg) {
	if (arg.HasParameter()) {
		throw ParameterNotResolvedException();
	}
	return ExpressionExecutor::EvaluateScalar(context, arg);
}

unique_ptr<FunctionData> VLLMBind(ClientContext &context, ScalarFunction &bound_function,
                                  vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() < 2 || arguments.size() > 3) {
		throw BinderException("vllm requires 2 or 3 arguments: vllm(prompt, model, options)");
	}
	if (arguments[0]->return_type.id() != LogicalTypeId::VARCHAR) {
		throw BinderException("vllm: prompt argument must be VARCHAR");
	}

	auto &model_arg = *arguments[1];
	ThrowIfNotConstant(model_arg, "model");
	if (model_arg.return_type.id() != LogicalTypeId::VARCHAR) {
		throw BinderException("vllm: model argument must be VARCHAR");
	}
	auto model_value = EvaluateScalar(context, model_arg);
	if (model_value.IsNull()) {
		throw BinderException("vllm: model argument cannot be NULL");
	}
	string model = StringValue::Get(model_value);

	Value options;
	if (arguments.size() == 3) {
		auto &options_arg = *arguments[2];
		ThrowIfNotConstant(options_arg, "options");
		options = EvaluateScalar(context, options_arg);
	} else {
		options = Value();
	}

	// Remove model/options from runtime arguments, they are stored in bind info.
	if (arguments.size() == 3) {
		Function::EraseArgument(bound_function, arguments, 2);
	}
	Function::EraseArgument(bound_function, arguments, 1);

	bound_function.SetReturnType(LogicalType::VARCHAR);
	return make_uniq<VLLMFunctionData>(std::move(model), std::move(options));
}

void VLLMExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	throw InvalidInputException("vllm can only be used in a projection and must be planned as a vLLM operator");
}

void VLLMSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data, const ScalarFunction &) {
	if (!bind_data) {
		serializer.WriteProperty<bool>(100, "has_bind_data", false);
		return;
	}
	auto &data = bind_data->Cast<VLLMFunctionData>();
	serializer.WriteProperty<bool>(100, "has_bind_data", true);
	serializer.WriteProperty(101, "model", data.model);
	serializer.WriteProperty(102, "options", data.options);
}

unique_ptr<FunctionData> VLLMDeserialize(Deserializer &deserializer, ScalarFunction &) {
	auto has_bind_data = deserializer.ReadProperty<bool>(100, "has_bind_data");
	if (!has_bind_data) {
		return nullptr;
	}
	auto model = deserializer.ReadProperty<string>(101, "model");
	auto options = deserializer.ReadProperty<Value>(102, "options");
	return make_uniq<VLLMFunctionData>(std::move(model), std::move(options));
}

} // namespace

ScalarFunctionSet VLLMFunction::GetFunctions() {
	ScalarFunctionSet set("vllm");
	auto vllm_base =
	    ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR, VLLMExecute, VLLMBind,
	                   nullptr, nullptr, nullptr, LogicalType::INVALID, FunctionStability::VOLATILE);
	vllm_base.serialize = VLLMSerialize;
	vllm_base.deserialize = VLLMDeserialize;
	set.AddFunction(std::move(vllm_base));

	auto vllm_with_options =
	    ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::ANY}, LogicalType::VARCHAR,
	                   VLLMExecute, VLLMBind, nullptr, nullptr, nullptr, LogicalType::ANY, FunctionStability::VOLATILE);
	vllm_with_options.serialize = VLLMSerialize;
	vllm_with_options.deserialize = VLLMDeserialize;
	set.AddFunction(std::move(vllm_with_options));
	return set;
}

} // namespace duckdb
