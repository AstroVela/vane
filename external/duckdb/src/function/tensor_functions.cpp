// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/function/built_in_functions.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression.hpp"

namespace duckdb {
namespace {

unique_ptr<FunctionData> BindTensor(ClientContext &context, ScalarFunction &function,
                                    vector<unique_ptr<Expression>> &arguments) {
	auto &data = arguments[0]->return_type;
	if (data.HasAlias() || (data.id() != LogicalTypeId::LIST && data.id() != LogicalTypeId::ARRAY)) {
		throw BinderException("tensor() requires a typed numeric list and a shape");
	}
	auto child = data.id() == LogicalTypeId::LIST ? ListType::GetChildType(data) : ArrayType::GetChildType(data);
	auto &shape_type = arguments[1]->return_type;
	idx_t rank = 0;
	if (shape_type.id() == LogicalTypeId::ARRAY && !shape_type.HasAlias()) {
		rank = ArrayType::GetSize(shape_type);
	} else if (arguments[1]->IsFoldable() && shape_type.id() == LogicalTypeId::LIST) {
		auto shape = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
		if (!shape.IsNull()) {
			rank = ListValue::GetChildren(shape).size();
		}
	}
	if (rank == 0 || rank > TensorType::MAX_VARIABLE_RANK) {
		throw BinderException("tensor() shape requires a constant list or an INTEGER array with rank 1..32");
	}
	auto shape_child = shape_type.id() == LogicalTypeId::LIST ? ListType::GetChildType(shape_type)
	                                                          : ArrayType::GetChildType(shape_type);
	if (!shape_child.IsIntegral() && shape_child.id() != LogicalTypeId::SQLNULL) {
		throw BinderException("tensor() shape dimensions must be integers");
	}
	function.arguments[0] = LogicalType::LIST(child);
	function.arguments[1] = LogicalType::ARRAY(LogicalType::INTEGER, rank);
	function.return_type = TensorType::Create(child, vector<idx_t>(rank, TensorType::VARIABLE_DIMENSION));
	return nullptr;
}

void ConstructTensor(DataChunk &args, ExpressionState &, Vector &result) {
	args.Flatten();
	result.SetVectorType(VectorType::FLAT_VECTOR);
	FlatVector::Validity(result).SetAllValid(args.size());
	auto &fields = StructVector::GetEntries(result);
	fields[0]->Reference(args.data[0]);
	fields[1]->Reference(args.data[1]);
	vector<idx_t> rows;
	rows.reserve(args.size());
	for (idx_t row = 0; row < args.size(); row++) {
		if (FlatVector::IsNull(args.data[0], row) || FlatVector::IsNull(args.data[1], row)) {
			FlatVector::SetNull(result, row, true);
		} else {
			rows.push_back(row);
		}
	}
	TensorType::ValidateRows(result, rows, "tensor()");
}

template <idx_t FIELD>
unique_ptr<FunctionData> BindTensorField(ClientContext &, ScalarFunction &function,
                                         vector<unique_ptr<Expression>> &arguments) {
	auto &type = arguments[0]->return_type;
	if (!TensorType::IsVariableShapeTensor(type)) {
		throw BinderException("%s() requires a variable shape TENSOR", function.name);
	}
	function.arguments[0] = type;
	function.return_type = StructType::GetChildType(type, FIELD);
	return nullptr;
}

template <idx_t FIELD>
void TensorField(DataChunk &args, ExpressionState &, Vector &result) {
	auto &input = args.data[0];
	input.Flatten(args.size());
	result.Reference(*StructVector::GetEntries(input)[FIELD]);
	result.Flatten(args.size());
	FlatVector::Validity(result).EnsureWritable();
	for (idx_t row = 0; row < args.size(); row++) {
		if (FlatVector::IsNull(input, row)) {
			FlatVector::SetNull(result, row, true);
		}
	}
}

} // namespace

void BuiltinFunctions::RegisterTensorFunctions() {
	ScalarFunction construct("tensor", {LogicalType::ANY, LogicalType::ANY}, LogicalType::ANY, ConstructTensor,
	                         BindTensor);
	construct.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	construct.SetFallible();
	AddFunction(std::move(construct));
	AddFunction(
	    ScalarFunction("tensor_data", {LogicalType::ANY}, LogicalType::ANY, TensorField<0>, BindTensorField<0>));
	AddFunction(
	    ScalarFunction("tensor_shape", {LogicalType::ANY}, LogicalType::ANY, TensorField<1>, BindTensorField<1>));
}

} // namespace duckdb
