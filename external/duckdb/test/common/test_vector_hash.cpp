// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"

#include <algorithm>

using namespace duckdb; // NOLINT

// Use the unchanged primitive Hash/CombineHash operations as the reference,
// independent of how nested vectors batch their child hashes.
static hash_t NestedHashReference(const Value &value, bool combine) {
	Vector result(LogicalType::HASH, 1);
	Vector seed(Value::UBIGINT(combine ? 123 : 0));
	VectorOperations::Hash(seed, result, 1);
	bool initialized = combine || value.type().id() == LogicalTypeId::ARRAY;
	const auto &children = value.IsNull()                              ? duckdb::vector<Value>()
	                       : value.type().id() == LogicalTypeId::ARRAY ? ArrayValue::GetChildren(value)
	                                                                   : ListValue::GetChildren(value);
	if (value.IsNull() || children.empty()) {
		if (!combine) {
			Vector null {Value(LogicalType::INTEGER)};
			VectorOperations::Hash(null, result, 1);
		}
	} else {
		for (auto &value : children) {
			Vector child(value);
			if (initialized) {
				VectorOperations::CombineHash(result, child, 1);
			} else {
				VectorOperations::Hash(child, result, 1);
				initialized = true;
			}
		}
	}
	return result.GetValue(0).GetValue<hash_t>();
}

static void CheckNestedHashes(Vector &input, const duckdb::vector<Value> &values) {
	const idx_t count = values.size();
	REQUIRE(count == 5);
	SelectionVector selection(3);
	selection.set_index(0, 4);
	selection.set_index(1, 1);
	selection.set_index(2, 3);
	const auto seed = Value::UBIGINT(123).Hash();
	for (bool combine : {false, true}) {
		for (bool selected : {false, true}) {
			Vector result(Value::UBIGINT(seed));
			result.Flatten(count);
			if (selected) {
				if (combine) {
					VectorOperations::CombineHash(result, input, selection, 3);
				} else {
					VectorOperations::Hash(input, result, selection, 3);
				}
			} else if (combine) {
				VectorOperations::CombineHash(result, input, count);
			} else {
				VectorOperations::Hash(input, result, count);
			}
			for (idx_t i = 0; i < count; i++) {
				INFO("row=" << i << " combine=" << combine << " selected=" << selected);
				auto expected = selected && (i == 0 || i == 2) ? seed : NestedHashReference(values[i], combine);
				REQUIRE(result.GetValue(i).GetValue<hash_t>() == expected);
			}
		}
	}
}

TEST_CASE("Nested hashes retain values across child batch boundaries and selections", "[vector][hash]") {
	for (bool array : {false, true}) {
		for (idx_t length :
		     {STANDARD_VECTOR_SIZE - 1, STANDARD_VECTOR_SIZE, STANDARD_VECTOR_SIZE + 1, 2 * STANDARD_VECTOR_SIZE + 7}) {
			INFO("array=" << array << " length=" << length);
			duckdb::vector<Value> children;
			for (idx_t i = 0; i < length; i++) {
				children.push_back(i % 31 == 0 ? Value(LogicalType::INTEGER) : Value::INTEGER(int32_t(i)));
			}
			auto value =
			    array ? Value::ARRAY(LogicalType::INTEGER, children) : Value::LIST(LogicalType::INTEGER, children);
			auto repeated = children;
			std::fill(repeated.begin(), repeated.end(), Value::INTEGER(7));
			auto other =
			    array ? Value::ARRAY(LogicalType::INTEGER, repeated) : Value::LIST(LogicalType::INTEGER, repeated);
			duckdb::vector<Value> values {value, Value(value.type()),
			                              array ? other : Value::LIST(LogicalType::INTEGER, {}), other, value};
			Vector input(value.type(), values.size());
			for (idx_t i = 0; i < values.size(); i++) {
				input.SetValue(i, values[i]);
			}
			CheckNestedHashes(input, values);

			SelectionVector selection(5);
			idx_t positions[] = {3, 1, 0, 4, 2};
			duckdb::vector<Value> selected;
			for (idx_t i = 0; i < 5; i++) {
				selection.set_index(i, positions[i]);
				selected.push_back(values[positions[i]]);
			}
			Vector dictionary(input, selection, 5);
			CheckNestedHashes(dictionary, selected);

			Vector constant(value);
			CheckNestedHashes(constant, duckdb::vector<Value>(5, value));
			// Child hashing may itself produce a constant vector, including a
			// shorter final batch. Reusing the buffer must flatten that result.
			auto &child = array ? ArrayVector::GetEntry(constant) : ListVector::GetEntry(constant);
			child.Reference(Value::INTEGER(7));
			CheckNestedHashes(constant, duckdb::vector<Value>(5, other));
		}
	}
}
