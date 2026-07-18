// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/scalar/udf.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/function/scalar/udf_functions.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/optional_ptr.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

namespace duckdb {

UDFFunctionData::UDFFunctionData(Value payload_p, LogicalType return_type_p, shared_ptr<void> actor_handles_p)
    : payload(std::move(payload_p)), return_type(std::move(return_type_p)), actor_handles(std::move(actor_handles_p)) {
}

unique_ptr<FunctionData> UDFFunctionData::Copy() const {
	return make_uniq<UDFFunctionData>(payload, return_type, actor_handles);
}

bool UDFFunctionData::Equals(const FunctionData &other_p) const {
	auto &other = other_p.Cast<UDFFunctionData>();
	return payload == other.payload && return_type == other.return_type;
}

namespace udf_helpers {

void ThrowIfNotConstant(const Expression &arg, const string &name) {
	if (!arg.IsFoldable()) {
		throw BinderException("udf: argument '%s' must be constant", name);
	}
}

Value EvaluateScalar(ClientContext &context, Expression &arg) {
	if (arg.HasParameter()) {
		throw ParameterNotResolvedException();
	}
	return ExpressionExecutor::EvaluateScalar(context, arg);
}

optional_ptr<const Value> GetStructField(const Value &payload, const string &name) {
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		throw BinderException("udf: payload must be a STRUCT");
	}
	auto &children = StructValue::GetChildren(payload);
	auto &payload_type = payload.type();
	auto child_count = StructType::GetChildCount(payload_type);
	for (idx_t i = 0; i < child_count; i++) {
		if (StringUtil::CIEquals(StructType::GetChildName(payload_type, i), name)) {
			return &children[i];
		}
	}
	return nullptr;
}

string GetStructStringField(const Value &payload, const string &name) {
	auto field = GetStructField(payload, name);
	if (!field || field->IsNull()) {
		throw BinderException("udf: payload missing required field '%s'", name);
	}
	if (field->type().id() != LogicalTypeId::VARCHAR) {
		throw BinderException("udf: payload field '%s' must be VARCHAR", name);
	}
	return StringValue::Get(*field);
}

bool TryGetStructStringField(const Value &payload, const string &name, string &result) {
	auto field = GetStructField(payload, name);
	if (!field || field->IsNull()) {
		return false;
	}
	if (field->type().id() != LogicalTypeId::VARCHAR) {
		throw BinderException("udf: payload field '%s' must be VARCHAR", name);
	}
	result = StringValue::Get(*field);
	return true;
}

vector<idx_t> ParseOutputSchemaTensorShape(const Value &entry) {
	auto shape_field = GetStructField(entry, "shape");
	if (!shape_field || shape_field->IsNull()) {
		throw BinderException("udf: output_schema tensor entry is missing shape");
	}
	if (shape_field->type().id() != LogicalTypeId::LIST) {
		throw BinderException("udf: output_schema tensor shape must be LIST<BIGINT>");
	}
	vector<idx_t> shape;
	auto &values = ListValue::GetChildren(*shape_field);
	shape.reserve(values.size());
	for (auto &value : values) {
		if (value.IsNull()) {
			throw BinderException("udf: output_schema tensor shape cannot contain NULL");
		}
		auto dim = value.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
		if (dim <= 0) {
			throw BinderException("udf: output_schema tensor shape dimensions must be positive");
		}
		shape.push_back(NumericCast<idx_t>(dim));
	}
	if (shape.empty()) {
		throw BinderException("udf: output_schema tensor shape must be non-empty");
	}
	return shape;
}

child_list_t<LogicalType> ParseOutputSchemaChildren(const Value &payload) {
	auto field = GetStructField(payload, "output_schema");
	if (!field || field->IsNull()) {
		return {};
	}
	if (field->type().id() != LogicalTypeId::LIST) {
		throw BinderException("udf: payload field 'output_schema' must be a LIST");
	}
	child_list_t<LogicalType> output_children;
	auto &entries = ListValue::GetChildren(*field);
	output_children.reserve(entries.size());
	for (idx_t i = 0; i < entries.size(); i++) {
		auto &entry = entries[i];
		if (entry.IsNull() || entry.type().id() != LogicalTypeId::STRUCT) {
			throw BinderException("udf: output_schema entries must be STRUCT values");
		}
		string name;
		if (!TryGetStructStringField(entry, "name", name) || name.empty()) {
			throw BinderException("udf: output_schema entry is missing name");
		}
		string kind;
		if (!TryGetStructStringField(entry, "kind", kind) || kind.empty() ||
		    StringUtil::CIEquals(kind, "duckdb_type")) {
			string type_name;
			if (!TryGetStructStringField(entry, "type", type_name) || type_name.empty()) {
				throw BinderException("udf: output_schema duckdb_type entry is missing type");
			}
			output_children.emplace_back(name, DBConfig::ParseLogicalType(type_name));
			continue;
		}
		if (StringUtil::CIEquals(kind, "tensor")) {
			string dtype;
			if (!TryGetStructStringField(entry, "dtype", dtype) || dtype.empty()) {
				throw BinderException("udf: output_schema tensor entry is missing dtype");
			}
			auto shape = ParseOutputSchemaTensorShape(entry);
			output_children.emplace_back(name, TensorType::Create(DBConfig::ParseLogicalType(dtype), shape));
			continue;
		}
		throw BinderException("udf: unsupported output_schema kind '%s'", kind);
	}
	return output_children;
}

static void ValidatePayloadVersion(const Value &payload) {
	auto version_field = GetStructField(payload, "payload_version");
	if (!version_field || version_field->IsNull()) {
		throw BinderException("udf: payload missing required payload_version");
	}
	auto version = version_field->DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
	if (version != 1) {
		throw BinderException("udf: unsupported payload_version %lld (expected 1)", version);
	}
}

LogicalType ResolvePayloadReturnType(const Value &payload) {
	ValidatePayloadVersion(payload);
	auto return_type_field = udf_helpers::GetStructField(payload, "method_return_type");
	if (return_type_field && !return_type_field->IsNull()) {
		auto return_type_str = StringValue::Get(*return_type_field);
		return DBConfig::ParseLogicalType(return_type_str);
	}
	auto output_children = ParseOutputSchemaChildren(payload);
	if (!output_children.empty()) {
		if (output_children.size() == 1) {
			return output_children[0].second;
		}
		return LogicalType::STRUCT(std::move(output_children));
	}

	throw BinderException("udf: payload missing method_return_type or output_schema");
}

} // namespace udf_helpers

unique_ptr<FunctionData> UDFBind(ClientContext &context, ScalarFunction &bound_function,
                                 vector<unique_ptr<Expression>> &arguments) {
	if (arguments.empty()) {
		throw BinderException("udf requires at least one argument (payload)");
	}
	auto payload_index = arguments.size() - 1;
	auto &payload_arg = *arguments[payload_index];
	udf_helpers::ThrowIfNotConstant(payload_arg, "payload");
	auto payload = udf_helpers::EvaluateScalar(context, payload_arg);
	if (payload.IsNull()) {
		throw BinderException("udf: payload cannot be NULL");
	}

	auto return_type = udf_helpers::ResolvePayloadReturnType(payload);

	// Remove payload from arguments, manually handling varargs case
	// (Function::EraseArgument requires arguments.size() == bound_function.arguments.size(),
	// which fails for varargs when input has multiple columns)
	if (bound_function.original_arguments.empty()) {
		bound_function.original_arguments = bound_function.arguments;
	}
	arguments.erase_at(payload_index);
	// Rebuild bound_function.arguments to match remaining expression types
	bound_function.arguments.clear();
	for (auto &arg : arguments) {
		bound_function.arguments.push_back(arg->return_type);
	}
	bound_function.SetReturnType(return_type);
	return make_uniq<UDFFunctionData>(std::move(payload), std::move(return_type));
}

void UDFExecute(DataChunk &args, ExpressionState &state, Vector &result) {
	throw InvalidInputException("udf can only be used in a projection and must be planned as a UDF operator");
}

void UDFSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data, const ScalarFunction &) {
	if (!bind_data) {
		serializer.WriteProperty<bool>(100, "has_bind_data", false);
		return;
	}
	auto &data = bind_data->Cast<UDFFunctionData>();
	serializer.WriteProperty<bool>(100, "has_bind_data", true);
	serializer.WriteProperty(101, "payload", data.payload);
	serializer.WriteProperty(102, "return_type", data.return_type);
}

unique_ptr<FunctionData> UDFDeserialize(Deserializer &deserializer, ScalarFunction &function) {
	auto has_bind_data = deserializer.ReadProperty<bool>(100, "has_bind_data");
	if (!has_bind_data) {
		return nullptr;
	}
	auto payload = deserializer.ReadProperty<Value>(101, "payload");
	auto return_type = deserializer.ReadProperty<LogicalType>(102, "return_type");
	auto payload_return_type = udf_helpers::ResolvePayloadReturnType(payload);
	if (payload_return_type != return_type) {
		throw SerializationException("udf: serialized return type '%s' does not match payload return type '%s'",
		                             return_type.ToString(), payload_return_type.ToString());
	}
	function.SetReturnType(return_type);
	return make_uniq<UDFFunctionData>(std::move(payload), std::move(return_type));
}

ScalarFunctionSet UDFFunction::GetFunctions() {
	ScalarFunctionSet set("udf");
	auto udf = ScalarFunction({LogicalType::ANY}, LogicalType::ANY, UDFExecute, UDFBind, nullptr, nullptr, nullptr,
	                          LogicalType::ANY, FunctionStability::VOLATILE);
	udf.serialize = UDFSerialize;
	udf.deserialize = UDFDeserialize;
	set.AddFunction(std::move(udf));
	return set;
}

} // namespace duckdb
