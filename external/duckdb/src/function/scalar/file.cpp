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
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"

namespace duckdb {

namespace {

static void ValidateFileFields(bool url_is_valid, bool position_is_valid, int64_t position, bool size_is_valid,
                               int64_t size, const string *checksum) {
	if (!url_is_valid) {
		throw InvalidInputException("file() url cannot be NULL");
	}
	if (position_is_valid != size_is_valid) {
		throw InvalidInputException("file() position and size must either both be NULL or both be non-NULL");
	}
	if (position_is_valid) {
		if (position < 0 || size < 0) {
			throw InvalidInputException("file() position and size must be non-negative");
		}
		if (position > NumericLimits<int64_t>::Maximum() - size) {
			throw InvalidInputException("file() byte range exceeds BIGINT");
		}
	}
	if (checksum) {
		auto separator = checksum->find(':');
		if (separator == string::npos || separator == 0 || separator + 1 == checksum->size() ||
		    checksum->find(':', separator + 1) != string::npos) {
			throw InvalidInputException("file() checksum must have the form <algorithm>:<digest>");
		}
	}
}

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
	for (idx_t row = 0; row < args.size(); row++) {
		auto url_index = url_data.sel->get_index(row);
		auto position_index = position_data.sel->get_index(row);
		auto size_index = size_data.sel->get_index(row);
		auto checksum_index = checksum_data.sel->get_index(row);
		auto position_is_valid = position_data.validity.RowIsValid(position_index);
		auto size_is_valid = size_data.validity.RowIsValid(size_index);
		auto position = position_is_valid ? positions[position_index] : 0;
		auto size = size_is_valid ? sizes[size_index] : 0;

		string checksum;
		const string *checksum_ptr = nullptr;
		if (checksum_data.validity.RowIsValid(checksum_index)) {
			checksum = checksums[checksum_index].GetString();
			checksum_ptr = &checksum;
		}
		ValidateFileFields(url_data.validity.RowIsValid(url_index), position_is_valid, position, size_is_valid, size,
		                   checksum_ptr);
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

template <bool NEGATE>
static ScalarFunction GetFileComparison() {
	auto file_type = FileLogicalType::Create();
	auto name = NEGATE ? FileLogicalType::NOT_EQUAL_FUNCTION_NAME : FileLogicalType::EQUAL_FUNCTION_NAME;
	return ScalarFunction(name, {file_type, file_type}, LogicalType::BOOLEAN, FileComparisonFunction<NEGATE>);
}

} // namespace

vector<ScalarFunction> FileFunctions::GetFunctions() {
	vector<ScalarFunction> result;
	result.push_back(GetFileConstructor());
	result.push_back(GetFileComparison<false>());
	result.push_back(GetFileComparison<true>());
	return result;
}

} // namespace duckdb
