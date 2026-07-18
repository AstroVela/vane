// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/allocator.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/execution/operator/projection/physical_udf_inout.hpp"
#include "duckdb/function/scalar/udf_functions.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"

using namespace duckdb;

namespace {

Value MakeTestPayload(const LogicalType &return_type = LogicalType::BIGINT) {
	child_list_t<Value> children;
	children.emplace_back("payload_version", Value::BIGINT(1));
	children.emplace_back("udf_name", Value("test_udf"));
	children.emplace_back("call_mode", Value("map"));
	children.emplace_back("execution_backend", Value("subprocess_task"));
	children.emplace_back("function_pickle", Value::BLOB_RAW("fake_pickle_data"));
	children.emplace_back("function_pickle_size_bytes", Value::BIGINT(16));
	children.emplace_back("method_return_type", Value(return_type.ToString()));
	children.emplace_back("ref_output_types", Value::LIST(LogicalType::VARCHAR, {Value(return_type.ToString())}));
	return Value::STRUCT(std::move(children));
}

} // namespace

TEST_CASE("udf table function built-in registration", "[execution][udf][serialization]") {
	SECTION("GetUDFBuiltinTableFunction returns valid function with all required callbacks") {
		auto tf = GetUDFBuiltinTableFunction();
		REQUIRE(tf.name == "udf");
		REQUIRE(tf.in_out_function != nullptr);
		REQUIRE(tf.in_out_function_final != nullptr);
		REQUIRE(tf.init_local != nullptr);
		REQUIRE(tf.serialize != nullptr);
		REQUIRE(tf.deserialize != nullptr);
	}

	SECTION("udf is findable in catalog after database init") {
		DuckDB db(nullptr);
		Connection con(db);

		// Verify udf is registered by querying duckdb_functions()
		auto result = con.Query(
		    "SELECT function_name FROM duckdb_functions() WHERE function_name = 'udf' AND function_type = 'table'");
		REQUIRE(!result->HasError());
		REQUIRE(result->RowCount() > 0);
	}
}

TEST_CASE("MakeUDFTableFunction has serialize hooks", "[execution][udf][serialization]") {
	auto payload = MakeTestPayload();
	auto tf = MakeUDFTableFunction(std::move(payload), {LogicalType::BIGINT}, {"result"});

	REQUIRE(tf.name == "udf");
	REQUIRE(tf.serialize != nullptr);
	REQUIRE(tf.deserialize != nullptr);
	REQUIRE(tf.in_out_function != nullptr);
	REQUIRE(tf.in_out_function_final != nullptr);
}

TEST_CASE("udf UDFFunctionData serialization round-trip", "[execution][udf][serialization]") {
	SECTION("round-trip preserves payload and return_type") {
		Allocator allocator;

		// Create bind data
		auto payload = MakeTestPayload();
		auto return_type = LogicalType::BIGINT;
		auto original = make_uniq<UDFFunctionData>(payload, return_type);

		auto tf = GetUDFBuiltinTableFunction();

		// Serialize
		MemoryStream stream(allocator);
		SerializationOptions options;
		BinarySerializer serializer(stream, options);
		serializer.Begin();
		tf.serialize(serializer, original.get(), tf);
		serializer.End();

		auto serialized_size = stream.GetPosition();
		REQUIRE(serialized_size > 0);

		// Deserialize
		stream.Rewind();
		BinaryDeserializer deserializer(stream);
		deserializer.Begin();
		auto deserialized = tf.deserialize(deserializer, tf);
		deserializer.End();

		REQUIRE(deserialized != nullptr);
		auto &result = deserialized->Cast<UDFFunctionData>();
		REQUIRE(result.return_type == LogicalType::BIGINT);
		REQUIRE(result.payload.type().id() == LogicalTypeId::STRUCT);

		// Verify specific payload fields survived round-trip
		auto &children = StructValue::GetChildren(result.payload);
		auto child_count = StructType::GetChildCount(result.payload.type());
		bool found_udf_name = false;
		for (idx_t i = 0; i < child_count; i++) {
			if (StructType::GetChildName(result.payload.type(), i) == "udf_name") {
				REQUIRE(children[i].ToString() == "test_udf");
				found_udf_name = true;
			}
		}
		REQUIRE(found_udf_name);
	}

	SECTION("round-trip with null bind_data") {
		Allocator allocator;
		auto tf = GetUDFBuiltinTableFunction();

		MemoryStream stream(allocator);
		SerializationOptions options;
		BinarySerializer serializer(stream, options);
		serializer.Begin();
		tf.serialize(serializer, nullptr, tf);
		serializer.End();

		stream.Rewind();
		BinaryDeserializer deserializer(stream);
		deserializer.Begin();
		auto deserialized = tf.deserialize(deserializer, tf);
		deserializer.End();

		REQUIRE(deserialized == nullptr);
	}

	SECTION("round-trip with STRUCT return type") {
		Allocator allocator;

		child_list_t<LogicalType> struct_children;
		struct_children.emplace_back("col_a", LogicalType::INTEGER);
		struct_children.emplace_back("col_b", LogicalType::VARCHAR);
		auto return_type = LogicalType::STRUCT(std::move(struct_children));
		auto payload = MakeTestPayload(return_type);
		auto original = make_uniq<UDFFunctionData>(payload, return_type);

		auto tf = GetUDFBuiltinTableFunction();

		MemoryStream stream(allocator);
		SerializationOptions options;
		BinarySerializer serializer(stream, options);
		serializer.Begin();
		tf.serialize(serializer, original.get(), tf);
		serializer.End();

		stream.Rewind();
		BinaryDeserializer deserializer(stream);
		deserializer.Begin();
		auto deserialized = tf.deserialize(deserializer, tf);
		deserializer.End();

		REQUIRE(deserialized != nullptr);
		auto &result = deserialized->Cast<UDFFunctionData>();
		REQUIRE(result.return_type.id() == LogicalTypeId::STRUCT);
		REQUIRE(StructType::GetChildCount(result.return_type) == 2);
		REQUIRE(StructType::GetChildName(result.return_type, 0) == "col_a");
		REQUIRE(StructType::GetChildName(result.return_type, 1) == "col_b");
		REQUIRE(StructType::GetChildType(result.return_type, 0) == LogicalType::INTEGER);
		REQUIRE(StructType::GetChildType(result.return_type, 1) == LogicalType::VARCHAR);
	}
}
