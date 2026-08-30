// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/physical_plan/plan_udf_project.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/execution/operator/projection/physical_udf_inout.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/scalar/udf_functions.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_udf_project.hpp"

namespace duckdb {

namespace {

static bool HasRegisteredUDFInfo(const ScalarFunction &func) {
	if (!func.function_info) {
		return false;
	}
	return dynamic_cast<RegisteredUDFFunctionInfo *>(func.function_info.get()) != nullptr;
}

static BoundFunctionExpression &GetUDFFunction(Expression &expr) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		throw InternalException("udf operator expected a bound function expression");
	}
	auto &bound = expr.Cast<BoundFunctionExpression>();
	if (!StringUtil::CIEquals(bound.function.name, "udf") && !HasRegisteredUDFInfo(bound.function)) {
		throw InternalException("udf operator expected a udf-backed function expression");
	}
	return bound;
}

static bool PayloadBoolField(const Value &payload, const string &name) {
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		return false;
	}
	auto &children = StructValue::GetChildren(payload);
	auto child_count = StructType::GetChildCount(payload.type());
	for (idx_t i = 0; i < child_count; i++) {
		if (StructType::GetChildName(payload.type(), i) != name || i >= children.size() || children[i].IsNull()) {
			continue;
		}
		return children[i].type().id() == LogicalTypeId::BOOLEAN && BooleanValue::Get(children[i]);
	}
	return false;
}

static double PayloadNumericField(const Value &payload, const string &name) {
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		return 0.0;
	}
	auto &children = StructValue::GetChildren(payload);
	auto child_count = StructType::GetChildCount(payload.type());
	for (idx_t i = 0; i < child_count; i++) {
		if (StructType::GetChildName(payload.type(), i) != name || i >= children.size() || children[i].IsNull()) {
			continue;
		}
		switch (children[i].type().id()) {
		case LogicalTypeId::DOUBLE:
			return DoubleValue::Get(children[i]);
		case LogicalTypeId::FLOAT:
			return FloatValue::Get(children[i]);
		case LogicalTypeId::INTEGER:
			return IntegerValue::Get(children[i]);
		case LogicalTypeId::BIGINT:
			return BigIntValue::Get(children[i]);
		default:
			throw InvalidInputException("UDF payload field '%s' must be numeric", name);
		}
	}
	return 0.0;
}

static bool PayloadStringEquals(const Value &payload, const string &name, const string &expected) {
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		return false;
	}
	auto &children = StructValue::GetChildren(payload);
	auto child_count = StructType::GetChildCount(payload.type());
	for (idx_t i = 0; i < child_count; i++) {
		if (StructType::GetChildName(payload.type(), i) != name || i >= children.size() || children[i].IsNull()) {
			continue;
		}
		if (children[i].type().id() != LogicalTypeId::VARCHAR) {
			return false;
		}
		return StringValue::Get(children[i]) == expected;
	}
	return false;
}

static bool PayloadHasField(const Value &payload, const string &name) {
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		return false;
	}
	auto child_count = StructType::GetChildCount(payload.type());
	for (idx_t i = 0; i < child_count; i++) {
		if (StructType::GetChildName(payload.type(), i) == name) {
			return true;
		}
	}
	return false;
}

static pair<bool, idx_t> PayloadIdxField(const Value &payload, const string &name) {
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		return make_pair(false, idx_t(0));
	}
	auto &children = StructValue::GetChildren(payload);
	auto child_count = StructType::GetChildCount(payload.type());
	for (idx_t i = 0; i < child_count; i++) {
		if (StructType::GetChildName(payload.type(), i) != name || i >= children.size() || children[i].IsNull()) {
			continue;
		}
		switch (children[i].type().id()) {
		case LogicalTypeId::INTEGER:
			return make_pair(true, static_cast<idx_t>(IntegerValue::Get(children[i])));
		case LogicalTypeId::BIGINT:
			return make_pair(true, static_cast<idx_t>(BigIntValue::Get(children[i])));
		default:
			return make_pair(false, idx_t(0));
		}
	}
	return make_pair(false, idx_t(0));
}

static pair<bool, idx_t> PayloadListSizeField(const Value &payload, const string &name) {
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		return make_pair(false, idx_t(0));
	}
	auto &children = StructValue::GetChildren(payload);
	auto child_count = StructType::GetChildCount(payload.type());
	for (idx_t i = 0; i < child_count; i++) {
		if (StructType::GetChildName(payload.type(), i) != name || i >= children.size() || children[i].IsNull()) {
			continue;
		}
		if (children[i].type().id() != LogicalTypeId::LIST) {
			throw InvalidInputException("UDF payload field '%s' must be a LIST", name);
		}
		return make_pair(true, ListValue::GetChildren(children[i]).size());
	}
	return make_pair(false, idx_t(0));
}

static bool IsRowPreservingUDFPayload(const Value &payload) {
	return UDFModePreservesRows(ClassifyUDFMode(payload));
}

static bool IsRowPreservingLayoutPayload(const Value &payload) {
	return IsRowPreservingUDFPayload(payload) && PayloadHasField(payload, "scalar_arg_count") &&
	       PayloadHasField(payload, "ref_output_types");
}

static vector<Value> UDFLogicalTypesToStringValues(const vector<LogicalType> &types) {
	vector<Value> values;
	values.reserve(types.size());
	for (auto &type : types) {
		values.emplace_back(Value(type.ToString()));
	}
	return values;
}

static vector<Value> UDFLogicalContractValues(const vector<LogicalType> &types) {
	vector<Value> values;
	values.reserve(types.size());
	for (auto &type : types) {
		values.emplace_back(Value(udf_helpers::SerializableContractType(type).ToString()));
	}
	return values;
}

static vector<Value> UDFLogicalOutputContractValues(const Value &payload, const LogicalType &return_type) {
	if (PayloadHasField(payload, "method_return_type")) {
		return UDFLogicalContractValues(vector<LogicalType> {return_type});
	}
	auto output_count = PayloadListSizeField(payload, "output_schema");
	if (!output_count.first || output_count.second == 0) {
		throw InvalidInputException("UDF payload is missing output type information");
	}
	if (output_count.second == 1) {
		return UDFLogicalContractValues(vector<LogicalType> {return_type});
	}
	if (return_type.id() != LogicalTypeId::STRUCT || StructType::GetChildCount(return_type) != output_count.second) {
		throw InternalException("UDF output_schema does not match its bound return type");
	}
	vector<LogicalType> output_types;
	output_types.reserve(output_count.second);
	for (auto &child : StructType::GetChildTypes(return_type)) {
		output_types.push_back(child.second);
	}
	return UDFLogicalContractValues(output_types);
}

static bool UDFTypeContainsSupportedBit(const LogicalType &type) {
	if (type.id() == LogicalTypeId::BIT) {
		return true;
	}
	switch (type.id()) {
	case LogicalTypeId::STRUCT:
		for (auto &child : StructType::GetChildTypes(type)) {
			if (UDFTypeContainsSupportedBit(child.second)) {
				return true;
			}
		}
		return false;
	case LogicalTypeId::LIST:
		return UDFTypeContainsSupportedBit(ListType::GetChildType(type));
	case LogicalTypeId::ARRAY:
		return UDFTypeContainsSupportedBit(ArrayType::GetChildType(type));
	case LogicalTypeId::MAP:
		return UDFTypeContainsSupportedBit(MapType::KeyType(type)) ||
		       UDFTypeContainsSupportedBit(MapType::ValueType(type));
	case LogicalTypeId::UNION:
		// Python FILE contracts do not yet materialize UNION tags, so leave
		// generic UNION inputs unchanged rather than partially annotating them.
		return false;
	default:
		return false;
	}
}

static bool UDFTypeContainsFile(const LogicalType &type) {
	if (FileLogicalType::IsFile(type)) {
		return true;
	}
	switch (type.id()) {
	case LogicalTypeId::STRUCT:
		for (auto &child : StructType::GetChildTypes(type)) {
			if (UDFTypeContainsFile(child.second)) {
				return true;
			}
		}
		return false;
	case LogicalTypeId::UNION:
		for (idx_t index = 0; index < UnionType::GetMemberCount(type); index++) {
			if (UDFTypeContainsFile(UnionType::GetMemberType(type, index))) {
				return true;
			}
		}
		return false;
	case LogicalTypeId::LIST:
		return UDFTypeContainsFile(ListType::GetChildType(type));
	case LogicalTypeId::ARRAY:
		return UDFTypeContainsFile(ArrayType::GetChildType(type));
	case LogicalTypeId::MAP:
		return UDFTypeContainsFile(MapType::KeyType(type)) || UDFTypeContainsFile(MapType::ValueType(type));
	default:
		return false;
	}
}

static LogicalType UDFInputContractType(const LogicalType &type) {
	if (FileLogicalType::IsFile(type)) {
		return FileLogicalType::Create();
	}
	if (type.id() == LogicalTypeId::BIT) {
		return LogicalType::BIT;
	}
	// Only FILE and supported BIT positions matter to the Python boundary.
	// Collapsing unrelated subtrees makes the descriptor catalog-independent.
	if (!UDFTypeContainsFile(type) && !UDFTypeContainsSupportedBit(type)) {
		return LogicalType::VARCHAR;
	}
	if (TensorType::IsTensor(type)) {
		return TensorType::Create(UDFInputContractType(TensorType::GetChildType(type)), TensorType::GetShape(type));
	}

	switch (type.id()) {
	case LogicalTypeId::STRUCT: {
		child_list_t<LogicalType> children;
		for (auto &child : StructType::GetChildTypes(type)) {
			children.emplace_back(child.first, UDFInputContractType(child.second));
		}
		return LogicalType::STRUCT(std::move(children));
	}
	case LogicalTypeId::UNION: {
		child_list_t<LogicalType> members;
		for (idx_t index = 0; index < UnionType::GetMemberCount(type); index++) {
			members.emplace_back(UnionType::GetMemberName(type, index),
			                     UDFInputContractType(UnionType::GetMemberType(type, index)));
		}
		return LogicalType::UNION(std::move(members));
	}
	case LogicalTypeId::LIST:
		return LogicalType::LIST(UDFInputContractType(ListType::GetChildType(type)));
	case LogicalTypeId::ARRAY: {
		auto size = ArrayType::IsAnySize(type) ? optional_idx() : optional_idx(ArrayType::GetSize(type));
		return LogicalType::ARRAY(UDFInputContractType(ArrayType::GetChildType(type)), size);
	}
	case LogicalTypeId::MAP:
		return LogicalType::MAP(UDFInputContractType(MapType::KeyType(type)),
		                        UDFInputContractType(MapType::ValueType(type)));
	default:
		return LogicalType::VARCHAR;
	}
}

static vector<Value> UDFLogicalInputContractValues(const vector<LogicalType> &types) {
	vector<Value> values;
	values.reserve(types.size());
	for (auto &type : types) {
		if (!UDFTypeContainsFile(type) && !UDFTypeContainsSupportedBit(type)) {
			values.emplace_back(LogicalType::VARCHAR);
			continue;
		}
		values.emplace_back(UDFInputContractType(type).ToString());
	}
	return values;
}

static Value AddUDFTypePayload(const Value &payload, const vector<LogicalType> &argument_types,
                               const vector<LogicalType> &passthrough_types, const LogicalType &return_type,
                               bool include_row_preserving_layout) {
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		throw InternalException("udf layout payload must be a struct");
	}

	vector<LogicalType> ref_output_types = passthrough_types;
	ref_output_types.push_back(return_type);

	auto make_field = [&](const string &name) -> Value {
		if (name == "scalar_arg_count") {
			return Value::BIGINT(static_cast<int64_t>(argument_types.size()));
		}
		if (name == "input_types") {
			return Value::LIST(LogicalType::VARCHAR, UDFLogicalTypesToStringValues(argument_types));
		}
		if (name == "input_contract_types") {
			return Value::LIST(LogicalType::VARCHAR, UDFLogicalInputContractValues(argument_types));
		}
		if (name == "output_contract_types") {
			return Value::LIST(LogicalType::VARCHAR, UDFLogicalOutputContractValues(payload, return_type));
		}
		if (name == "ref_output_types") {
			return Value::LIST(LogicalType::VARCHAR, UDFLogicalContractValues(ref_output_types));
		}
		throw InternalException("unknown udf layout payload field");
	};

	auto &children = StructValue::GetChildren(payload);
	auto &payload_type = payload.type();
	child_list_t<Value> new_children;
	vector<string> fields {"input_types", "input_contract_types", "output_contract_types"};
	if (include_row_preserving_layout) {
		fields = {"scalar_arg_count", "input_types", "input_contract_types", "output_contract_types",
		          "ref_output_types"};
	}

	for (idx_t i = 0; i < StructType::GetChildCount(payload_type); i++) {
		auto child_name = StructType::GetChildName(payload_type, i);
		bool replace = false;
		for (auto &field_name : fields) {
			if (child_name == field_name) {
				replace = true;
				break;
			}
		}
		if (replace) {
			new_children.emplace_back(child_name, make_field(child_name));
		} else {
			new_children.emplace_back(child_name, children[i]);
		}
	}

	for (auto &field_name : fields) {
		bool found = false;
		for (idx_t i = 0; i < StructType::GetChildCount(payload_type); i++) {
			if (StructType::GetChildName(payload_type, i) == field_name) {
				found = true;
				break;
			}
		}
		if (!found) {
			new_children.emplace_back(field_name, make_field(field_name));
		}
	}

	return Value::STRUCT(std::move(new_children));
}

static Value PayloadWithBoolField(const Value &payload, const string &name, bool value) {
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		throw InternalException("udf ref handoff payload must be a struct");
	}
	auto &children = StructValue::GetChildren(payload);
	auto &payload_type = payload.type();
	child_list_t<Value> new_children;
	bool found = false;
	auto child_count = StructType::GetChildCount(payload_type);
	for (idx_t i = 0; i < child_count; i++) {
		auto child_name = StructType::GetChildName(payload_type, i);
		if (child_name == name) {
			new_children.emplace_back(child_name, Value::BOOLEAN(value));
			found = true;
		} else {
			new_children.emplace_back(child_name, children[i]);
		}
	}
	if (!found) {
		new_children.emplace_back(name, Value::BOOLEAN(value));
	}
	return Value::STRUCT(std::move(new_children));
}

static Value PayloadWithStringFields(const Value &payload, child_list_t<Value> fields) {
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		throw InternalException("udf payload must be a struct");
	}
	auto &children = StructValue::GetChildren(payload);
	auto &payload_type = payload.type();
	vector<bool> consumed(fields.size(), false);
	child_list_t<Value> new_children;
	auto child_count = StructType::GetChildCount(payload_type);
	for (idx_t i = 0; i < child_count; i++) {
		auto child_name = StructType::GetChildName(payload_type, i);
		bool replaced = false;
		for (idx_t field_idx = 0; field_idx < fields.size(); field_idx++) {
			if (fields[field_idx].first == child_name) {
				new_children.emplace_back(child_name, fields[field_idx].second);
				consumed[field_idx] = true;
				replaced = true;
				break;
			}
		}
		if (!replaced) {
			new_children.emplace_back(child_name, children[i]);
		}
	}
	for (idx_t field_idx = 0; field_idx < fields.size(); field_idx++) {
		if (!consumed[field_idx]) {
			new_children.emplace_back(fields[field_idx].first, fields[field_idx].second);
		}
	}
	return Value::STRUCT(std::move(new_children));
}

static string StreamingOutputModeForRunner(const string &runner_type) {
	return runner_type == "ray" ? "ray_block_stream" : "local_shm_ref_bundle";
}

static bool PayloadUsesActorBackend(const Value &payload) {
	return PayloadStringEquals(payload, "execution_backend", "ray_actor") ||
	       PayloadStringEquals(payload, "execution_backend", "subprocess_actor") ||
	       PayloadHasField(payload, "actor_number");
}

static Value PayloadWithResolvedExpressionBackend(const Value &payload) {
	auto runner_type = ResolveRunnerTypeFromEnvironment();
	if (runner_type != "ray" && PayloadNumericField(payload, "gpus") > 0.0) {
		throw InvalidInputException("GPU resources require VANE_RUNNER=ray");
	}
	const bool uses_actor_backend = PayloadUsesActorBackend(payload);
	const bool local_ref_bundle_output = runner_type != "ray";
	child_list_t<Value> fields;
	fields.emplace_back("execution_backend",
	                    Value(ExpressionUDFExecutionBackendForRunner(runner_type, uses_actor_backend)));
	if (uses_actor_backend && runner_type == "ray") {
		auto actor_number = PayloadIdxField(payload, "actor_number");
		if (!actor_number.first || actor_number.second == 0) {
			throw InternalException("ray actor expression UDF payload is missing actor_number");
		}
		fields.emplace_back("actor_pool_size", Value::BIGINT(static_cast<int64_t>(actor_number.second)));
	}
	fields.emplace_back("streaming_output_mode", Value(StreamingOutputModeForRunner(runner_type)));
	auto result = PayloadWithStringFields(payload, std::move(fields));
	if (runner_type == "ray") {
		result = PayloadWithBoolField(result, "produce_ray_block_stream", true);
		result = PayloadWithBoolField(result, "produce_ref_bundle_output", false);
	} else {
		result = PayloadWithBoolField(result, "produce_ray_block_stream", false);
		result = PayloadWithBoolField(result, "produce_ref_bundle_output", local_ref_bundle_output);
	}
	return result;
}

static PhysicalOperator &MaybeWrapRowPreservingLayoutProjection(PhysicalPlanGenerator &generator,
                                                                PhysicalOperator &plan,
                                                                vector<unique_ptr<Expression>> &args,
                                                                const Value &payload, idx_t estimated_cardinality) {
	if (!IsRowPreservingLayoutPayload(payload)) {
		return plan;
	}
	auto arg_count = PayloadIdxField(payload, "scalar_arg_count");
	if (!arg_count.first) {
		throw InvalidInputException("row-preserving udf payload requires scalar_arg_count");
	}
	auto child_types = plan.GetTypes();
	if (arg_count.second != args.size()) {
		throw InvalidInputException(
		    "row-preserving udf payload scalar_arg_count %llu does not match UDF argument count %llu",
		    static_cast<unsigned long long>(arg_count.second), static_cast<unsigned long long>(args.size()));
	}

	vector<unique_ptr<Expression>> layout_exprs;
	vector<LogicalType> layout_types;
	layout_exprs.reserve(arg_count.second + child_types.size());
	layout_types.reserve(arg_count.second + child_types.size());
	for (auto &arg : args) {
		layout_types.push_back(arg->return_type);
		layout_exprs.push_back(std::move(arg));
	}
	for (idx_t i = 0; i < child_types.size(); i++) {
		layout_types.push_back(child_types[i]);
		layout_exprs.push_back(make_uniq<BoundReferenceExpression>(child_types[i], i));
	}
	auto &layout_projection =
	    generator.Make<PhysicalProjection>(std::move(layout_types), std::move(layout_exprs), estimated_cardinality);
	layout_projection.children.push_back(plan);
	return layout_projection;
}

} // namespace

PhysicalOperator &PhysicalPlanGenerator::CreatePlan(LogicalUDFProject &op) {
	D_ASSERT(op.children.size() == 1);
	auto udf_expr = std::move(op.udf_expr);
	if (!udf_expr) {
		throw InternalException("udf operator is missing expression");
	}
	auto &bound = GetUDFFunction(*udf_expr);
	if (!bound.bind_info) {
		throw InternalException("udf function is missing bind data");
	}

	auto &bind_data = bound.bind_info->Cast<UDFFunctionData>();
	if (PayloadBoolField(bind_data.payload, "expression_udf")) {
		bind_data.payload = PayloadWithResolvedExpressionBackend(bind_data.payload);
	}
	bind_data.payload = RequireUDFStreamingOutput(std::move(bind_data.payload));

	auto &plan = CreatePlan(*op.children[0]);
	auto child_types = plan.GetTypes();
	vector<LogicalType> argument_types;
	argument_types.reserve(bound.children.size());
	for (auto &child : bound.children) {
		argument_types.push_back(child->return_type);
	}
	const bool include_row_preserving_layout = op.is_scalar_map || op.is_row_preserving_batch;
	bind_data.payload = AddUDFTypePayload(bind_data.payload, argument_types, child_types, bind_data.return_type,
	                                      include_row_preserving_layout);

	// Build the serializable TableFunction descriptor consumed by
	// PhysicalStreamingUDF.
	// Always output 1 column with the original return type (STRUCT for multi-column,
	// scalar for single).  This must match BoundStatement.types set at bind time.
	vector<LogicalType> return_types;
	return_types.push_back(bind_data.return_type);
	vector<string> return_names;
	return_names.push_back(op.output_column_name);
	if (op.is_scalar_map || op.is_row_preserving_batch) {
		auto child_types = op.children[0]->types;
		return_types.clear();
		return_names.clear();
		return_types.reserve(child_types.size() + 1);
		return_names.reserve(child_types.size() + 1);
		for (idx_t i = 0; i < child_types.size(); i++) {
			return_types.push_back(child_types[i]);
			return_names.push_back(StringUtil::Format("c%d", i));
		}
		return_types.push_back(bind_data.return_type);
		return_names.push_back(op.output_column_name);
	}
	auto table_function = MakeUDFTableFunction(bind_data.payload, return_types, return_names);

	auto udf_bind_data = std::move(bound.bind_info);

	// Single output column
	vector<ColumnIndex> column_ids;
	column_ids.emplace_back(0);
	auto &input_plan = MaybeWrapRowPreservingLayoutProjection(*this, plan, bound.children, bind_data.payload,
	                                                          op.estimated_cardinality);

	auto &streaming_op =
	    Make<PhysicalStreamingUDF>(op.types, std::move(table_function), std::move(udf_bind_data), std::move(column_ids),
	                               op.estimated_cardinality, vector<column_t>());
	streaming_op.children.push_back(input_plan);
	return streaming_op;
}

} // namespace duckdb
