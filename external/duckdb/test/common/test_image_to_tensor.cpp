// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"
#include "duckdb.hpp"
#include "duckdb.h"
#include "duckdb/common/types/vector_cache.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/parser/tableref/column_data_ref.hpp"
#include "image_tensor.hpp"

using namespace duckdb; // NOLINT

TEST_CASE("Image to Tensor retains constant storage and its owner", "[image][tensor]") {
	DuckDB database(nullptr);
	Connection connection(database);
	for (auto fixed : {false, true}) {
		auto type = fixed ? ImageLogicalType::Create("RGBA", 2, 3) : ImageLogicalType::Create("RGBA");
		Vector result(ImageToTensor::ResultType(type));
		const_data_ptr_t pixels = nullptr;
		{
			uint8_t bytes[24];
			for (idx_t i = 0; i < 24; i++) {
				bytes[i] = uint8_t(i);
			}
			DataChunk args;
			args.Initialize(Allocator::DefaultAllocator(), {type});
			args.data[0].Reference(ImageVector::FromPixels(bytes, 24, 3, 2, "RGBA", type));
			args.SetCardinality(STANDARD_VECTOR_SIZE);
			ImageOperatorInput reader(args.data[0], 1);
			ImagePixelView view;
			REQUIRE(reader.Read(0, view));
			pixels = view.data;
			ImageToTensor::Execute(args, *connection.context, result);
		}
		REQUIRE(result.GetVectorType() == VectorType::CONSTANT_VECTOR);
		auto &elements =
		    fixed ? ArrayVector::GetEntry(result) : ListVector::GetEntry(*StructVector::GetEntries(result)[0]);
		REQUIRE(FlatVector::GetData<uint8_t>(elements) == pixels);
		for (idx_t i = 0; i < 24; i++) {
			REQUIRE(pixels[i] == i);
		}
		if (fixed) {
			REQUIRE(ArrayVector::GetTotalSize(result) == 24);
		} else {
			auto value = result.GetValue(0);
			auto &fields = StructValue::GetChildren(value);
			REQUIRE(ListValue::GetChildren(fields[0]).size() == 24);
			auto &shape = ArrayValue::GetChildren(fields[1]);
			REQUIRE(shape[0].GetValue<int32_t>() == 2);
			REQUIRE(shape[1].GetValue<int32_t>() == 3);
			REQUIRE(shape[2].GetValue<int32_t>() == 4);
		}
	}
}

TEST_CASE("Image to Tensor handles dictionary selections and inactive payloads", "[image][tensor]") {
	DuckDB database(nullptr);
	Connection connection(database);
	for (auto fixed : {false, true}) {
		auto type = fixed ? ImageLogicalType::Create("RGB", 2, 3) : ImageLogicalType::Create();
		Vector source(type);
		for (idx_t row = 0; row < 4; row++) {
			if (row == 2) {
				source.SetValue(row, Value(type));
				continue;
			}
			auto mode = fixed ? "RGB" : row == 0 ? "L" : row == 1 ? "LA" : "RGBA";
			auto width = fixed ? 3 : uint32_t(row + 1);
			auto size = width * 2 * ImageLogicalType::ChannelsForMode(mode);
			memset(ImageVector::Allocate(source, row, width, 2, mode), 30 + row, size);
		}
		if (!fixed) {
			// Garbage beneath a NULL parent must neither be dereferenced nor propagated.
			auto &data = *StructVector::GetEntries(source)[0];
			FlatVector::GetData<list_entry_t>(data)[2] = list_entry_t(NumericLimits<idx_t>::Maximum(), 123);
		}
		SelectionVector first(5);
		idx_t original[] = {3, 2, 0, 1, 3};
		for (idx_t row = 0; row < 5; row++) {
			first.set_index(row, original[row]);
		}
		Vector selected(source, first, 5);
		SelectionVector second(4);
		idx_t positions[] = {4, 1, 3, 2};
		for (idx_t row = 0; row < 4; row++) {
			second.set_index(row, positions[row]);
		}
		selected.Slice(second, 4);
		DataChunk args;
		args.Initialize(Allocator::DefaultAllocator(), {type});
		args.data[0].Reference(selected);
		args.SetCardinality(4);
		Vector result(ImageToTensor::ResultType(type));
		ImageToTensor::Execute(args, *connection.context, result);
		auto &input_pixels =
		    fixed ? ArrayVector::GetEntry(source) : ListVector::GetEntry(*StructVector::GetEntries(source)[0]);
		auto &output_pixels =
		    fixed ? ArrayVector::GetEntry(result) : ListVector::GetEntry(*StructVector::GetEntries(result)[0]);
		REQUIRE(FlatVector::GetData<uint8_t>(input_pixels) == FlatVector::GetData<uint8_t>(output_pixels));
		REQUIRE(result.GetValue(1).IsNull());
		for (idx_t row : {idx_t(0), idx_t(2), idx_t(3)}) {
			auto value = result.GetValue(row);
			auto &data =
			    fixed ? ArrayValue::GetChildren(value) : ListValue::GetChildren(StructValue::GetChildren(value)[0]);
			for (auto &element : data) {
				REQUIRE(element.GetValue<uint8_t>() == 30 + original[positions[row]]);
			}
		}
		if (!fixed) {
			auto &data = *StructVector::GetEntries(result)[0];
			auto entry = FlatVector::GetData<list_entry_t>(data)[1];
			REQUIRE(entry.offset == 0);
			REQUIRE(entry.length == 0);
		}
	}
}

TEST_CASE("Image to Tensor rejects active corruption and observes interruption", "[image][tensor]") {
	DuckDB database(nullptr);
	Connection connection(database);
	for (auto fixed : {false, true}) {
		auto type = fixed ? ImageLogicalType::Create("RGB", 1, 1) : ImageLogicalType::Create();
		DataChunk args;
		args.Initialize(Allocator::DefaultAllocator(), {type});
		memset(ImageVector::Allocate(args.data[0], 0, 1, 1, "RGB"), 1, 3);
		args.SetCardinality(1);
		Vector result(ImageToTensor::ResultType(type));
		auto &pixels = fixed ? ArrayVector::GetEntry(args.data[0])
		                     : ListVector::GetEntry(*StructVector::GetEntries(args.data[0])[0]);
		FlatVector::SetNull(pixels, 1, true);
		REQUIRE_THROWS_AS(ImageToTensor::Execute(args, *connection.context, result), InvalidInputException);
		FlatVector::SetNull(pixels, 1, false);
		connection.Interrupt();
		REQUIRE_THROWS_AS(ImageToTensor::Execute(args, *connection.context, result), InterruptException);
		connection.context->interrupted = false;
	}
}

TEST_CASE("Fixed numeric Tensor storage grows by rows and copies element validity", "[tensor][array]") {
	for (auto &element_type : duckdb::vector<LogicalType> {
	         LogicalType::BOOLEAN, LogicalType::TINYINT, LogicalType::SMALLINT, LogicalType::INTEGER,
	         LogicalType::BIGINT, LogicalType::UTINYINT, LogicalType::USMALLINT, LogicalType::UINTEGER,
	         LogicalType::UBIGINT, LogicalType::FLOAT, LogicalType::DOUBLE}) {
		auto type = TensorType::Create(element_type, {2, 3});
		VectorCache cache(Allocator::DefaultAllocator(), type);
		Vector input(type);
		for (idx_t iteration = 0; iteration < 2; iteration++) {
			input.ResetFromCache(cache);
			REQUIRE(ArrayVector::GetTotalSize(input) == 0);
			auto &data = ArrayVector::GetEntryForWrite(input, 3);
			for (idx_t i = 0; i < 18; i++) {
				data.SetValue(i, Value::INTEGER(int32_t(i % 2)).DefaultCastAs(element_type));
			}
			FlatVector::SetNull(input, 1, true);
			FlatVector::SetNull(data, 2, true);
			SelectionVector selected(4);
			selected.set_index(0, 2);
			selected.set_index(1, 1);
			selected.set_index(2, 0);
			selected.set_index(3, 2);
			Vector output(type);
			output.SetValue(0, Value(type));
			VectorOperations::Copy(input, output, selected, 4, 0, 1);
			REQUIRE(ArrayVector::GetTotalSize(output) == 30);
			REQUIRE(output.GetValue(1) == input.GetValue(2));
			REQUIRE(output.GetValue(2).IsNull());
			REQUIRE(output.GetValue(3) == input.GetValue(0));
			REQUIRE(output.GetValue(4) == input.GetValue(2));
			Vector sliced(output, 3, 4);
			ArrayVector::GetEntryForWrite(sliced, 2).SetValue(6, Value::INTEGER(1).DefaultCastAs(element_type));
			REQUIRE(output.GetValue(4) == input.GetValue(2));
			REQUIRE(sliced.GetValue(0) == input.GetValue(0));
		}
		// The C API still promises writable child storage for the full requested capacity.
		auto capi_type = reinterpret_cast<duckdb_logical_type>(&type);
		auto capi_vector = duckdb_create_vector(capi_type, 5);
		REQUIRE(capi_vector != nullptr);
		REQUIRE(ArrayVector::GetTotalSize(*reinterpret_cast<Vector *>(capi_vector)) == 30);
		auto child = duckdb_array_vector_get_child(capi_vector);
		REQUIRE(duckdb_vector_get_data(child) != nullptr);
		duckdb_destroy_vector(&capi_vector);
	}
	REQUIRE_FALSE(ArrayVector::UsesDeferredStorage(LogicalType::ARRAY(LogicalType::DOUBLE, 6)));
	REQUIRE_FALSE(ArrayVector::UsesDeferredStorage(TensorType::Create(LogicalType::VARCHAR, {2})));
}

TEST_CASE("Materialized Tensor query descriptions use schema instead of element values", "[tensor]") {
	auto type = TensorType::Create(LogicalType::UTINYINT, {1, 1, 3});
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), {type});
	chunk.SetValue(0, 0,
	               Value::ARRAY(LogicalType::UTINYINT, {Value::UTINYINT(97), Value::UTINYINT(98), Value::UTINYINT(99)})
	                   .DefaultCastAs(type));
	chunk.SetCardinality(1);
	ColumnDataCollection collection(Allocator::DefaultAllocator(), {type});
	collection.Append(chunk);
	ColumnDataRef reference(&collection, {"pixels"});
	auto description = reference.ToString();
	REQUIRE(description.find("1 rows") != string::npos);
	REQUIRE(description.find(type.ToString()) != string::npos);
	REQUIRE(description.find("97") == string::npos);
	REQUIRE(description.find("FLAT") == string::npos);
	REQUIRE(collection.Count() == 1);
}

TEST_CASE("CASE and COALESCE merge selected fixed Tensor branches without overwriting storage", "[tensor][array]") {
	DuckDB database(nullptr);
	Connection connection(database);
	for (auto &element_type : duckdb::vector<LogicalType> {
	         LogicalType::BOOLEAN, LogicalType::TINYINT, LogicalType::SMALLINT, LogicalType::INTEGER,
	         LogicalType::BIGINT, LogicalType::UTINYINT, LogicalType::USMALLINT, LogicalType::UINTEGER,
	         LogicalType::UBIGINT, LogicalType::FLOAT, LogicalType::DOUBLE}) {
		auto type = TensorType::Create(element_type, {2, 2});
		auto name = type.ToString();
		auto result =
		    connection.Query("WITH tensors AS (SELECT i, "
		                     "(CASE WHEN i%3=0 THEN [0,1,NULL,1] END)::" +
		                     name +
		                     " AS a, "
		                     "(CASE WHEN i%3=1 THEN [1,NULL,0,1] END)::" +
		                     name +
		                     " AS b "
		                     "FROM range(4101) t(i) WHERE i%7<>0) "
		                     "SELECT i, CASE WHEN i%2=0 THEN a ELSE b END, COALESCE(a,b) FROM tensors ORDER BY i");
		REQUIRE_FALSE(result->HasError());
		REQUIRE(result->types[1] == type);
		REQUIRE(result->types[2] == type);
		idx_t rows = 0;
		while (auto chunk = result->Fetch()) {
			for (idx_t row = 0; row < chunk->size(); row++) {
				auto index = chunk->GetValue(0, row).GetValue<int64_t>();
				for (idx_t column : {idx_t(1), idx_t(2)}) {
					auto branch = index % 3;
					auto valid = branch != 2 && (column == 2 || branch == index % 2);
					auto value = chunk->GetValue(column, row);
					REQUIRE(value.IsNull() == !valid);
					if (!valid) {
						continue;
					}
					auto &elements = ArrayValue::GetChildren(value);
					REQUIRE(elements.size() == 4);
					for (idx_t element = 0; element < 4; element++) {
						auto is_null = element == (branch == 0 ? 2 : 1);
						REQUIRE(elements[element].IsNull() == is_null);
						if (!is_null) {
							auto expected = element == (branch == 0 ? 0 : 2) ? 0 : 1;
							REQUIRE(elements[element] == Value::INTEGER(expected).DefaultCastAs(element_type));
						}
					}
				}
				rows++;
			}
		}
		REQUIRE(rows == 3515);
	}
}
