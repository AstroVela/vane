// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/scalar/file.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/function/scalar/file_functions.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/type_visitor.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"

namespace duckdb {

namespace {

static void FileConstructorFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == FileLogicalType::FIELD_COUNT);

	bool all_constant = true;
	auto &children = StructVector::GetEntries(result);
	for (idx_t index = 0; index < args.ColumnCount(); index++) {
		if (args.data[index].GetVectorType() != VectorType::CONSTANT_VECTOR) {
			all_constant = false;
		}
		children[index]->Reference(args.data[index]);
	}
	result.SetVectorType(all_constant ? VectorType::CONSTANT_VECTOR : VectorType::FLAT_VECTOR);
	FileLogicalType::Validate(result, args.size(), "file()");
	result.Verify(args.size());
}

template <bool NEGATE>
static void FileComparisonFunction(DataChunk &args, ExpressionState &state, Vector &result) {
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

template <bool NEGATE>
static ScalarFunction GetFileComparison() {
	auto file_type = FileLogicalType::Create();
	auto name = NEGATE ? FileLogicalType::NOT_EQUAL_FUNCTION_NAME : FileLogicalType::EQUAL_FUNCTION_NAME;
	return ScalarFunction(name, {file_type, file_type}, LogicalType::BOOLEAN, FileComparisonFunction<NEGATE>);
}

} // namespace

unique_ptr<FunctionData> BindFileCollectionSearch(ClientContext &, ScalarFunction &,
                                                  vector<unique_ptr<Expression>> &arguments) {
	for (auto &argument : arguments) {
		if (TypeVisitor::Contains(argument->return_type, FileLogicalType::IsFile)) {
			throw BinderException("Collection search functions do not support FILE values");
		}
	}
	return nullptr;
}

unique_ptr<FunctionData> BindFileMapSearch(ClientContext &, ScalarFunction &,
                                           vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(arguments.size() == 2);
	auto &map_type = arguments[0]->return_type;
	if ((map_type.id() == LogicalTypeId::MAP &&
	     TypeVisitor::Contains(MapType::KeyType(map_type), FileLogicalType::IsFile)) ||
	    TypeVisitor::Contains(arguments[1]->return_type, FileLogicalType::IsFile)) {
		throw BinderException("Collection search functions do not support FILE values");
	}
	return nullptr;
}

vector<ScalarFunction> FileFunctions::GetFunctions() {
	vector<ScalarFunction> result;
	result.push_back(GetFileConstructor());
	result.push_back(GetFileComparison<false>());
	result.push_back(GetFileComparison<true>());
	return result;
}

} // namespace duckdb
