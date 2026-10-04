// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "arrow_frame.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/main/client_properties.hpp"
#include <arrow/c/bridge.h>

namespace duckdb {
namespace vane_execution {
std::shared_ptr<arrow::Schema> ArrowSchemaFor(const vector<LogicalType> &types, const vector<string> &names) {
	std::vector<std::shared_ptr<arrow::Field>> fields;
	for (idx_t i = 0; i < types.size(); i++) {
		std::shared_ptr<arrow::DataType> type;
		switch (types[i].id()) {
		case LogicalTypeId::BOOLEAN:
			type = arrow::boolean();
			break;
		case LogicalTypeId::TINYINT:
			type = arrow::int8();
			break;
		case LogicalTypeId::SMALLINT:
			type = arrow::int16();
			break;
		case LogicalTypeId::INTEGER:
			type = arrow::int32();
			break;
		case LogicalTypeId::BIGINT:
			type = arrow::int64();
			break;
		case LogicalTypeId::UTINYINT:
			type = arrow::uint8();
			break;
		case LogicalTypeId::USMALLINT:
			type = arrow::uint16();
			break;
		case LogicalTypeId::UINTEGER:
			type = arrow::uint32();
			break;
		case LogicalTypeId::UBIGINT:
			type = arrow::uint64();
			break;
		case LogicalTypeId::FLOAT:
			type = arrow::float32();
			break;
		case LogicalTypeId::DOUBLE:
			type = arrow::float64();
			break;
		case LogicalTypeId::VARCHAR:
			type = arrow::utf8();
			break;
		case LogicalTypeId::SQLNULL:
			type = arrow::null();
			break;
		default:
			throw NotImplementedException("unsupported direct Flight type");
		}
		fields.push_back(arrow::field(names.empty() ? "c" + std::to_string(i) : names[i], type));
	}
	return arrow::schema(std::move(fields));
}

std::shared_ptr<arrow::RecordBatch> Encode(DataChunk &chunk, const std::shared_ptr<arrow::Schema> &schema) {
	ClientProperties properties;
	ArrowArray array;
	array.Init();
	unordered_map<idx_t, const shared_ptr<ArrowTypeExtensionData>> extensions;
	ArrowConverter::ToArrowArray(chunk, &array, properties, extensions);
	auto result = arrow::ImportRecordBatch(&array, schema);
	if (!result.ok()) {
		throw IOException("native Arrow frame: %s", result.status().ToString());
	}
	return std::move(result).ValueOrDie();
}

template <class Array, class T>
void DecodePrimitive(const arrow::Array &source, Vector &target, idx_t count) {
	auto &values = static_cast<const Array &>(source);
	auto data = FlatVector::GetData<T>(target);
	for (idx_t row = 0; row < count; row++) {
		data[row] = values.Value(row);
	}
}

// The declared basic-types profile has no dictionaries or nested values. String
// references borrow the received Arrow batch until TryWrite copies into its
// reserved native frame. There is no unbounded Python or row-value staging.
void Decode(const arrow::RecordBatch &batch, const vector<LogicalType> &types, DataChunk &chunk) {
	const auto count = idx_t(batch.num_rows());
	chunk.Initialize(Allocator::DefaultAllocator(), types, count);
	chunk.SetCardinality(count);
	for (idx_t col = 0; col < types.size(); col++) {
		auto &source = *batch.column(col);
		auto &target = chunk.data[col];
		switch (types[col].id()) {
		case LogicalTypeId::BOOLEAN:
			DecodePrimitive<arrow::BooleanArray, bool>(source, target, count);
			break;
		case LogicalTypeId::TINYINT:
			DecodePrimitive<arrow::Int8Array, int8_t>(source, target, count);
			break;
		case LogicalTypeId::SMALLINT:
			DecodePrimitive<arrow::Int16Array, int16_t>(source, target, count);
			break;
		case LogicalTypeId::INTEGER:
			DecodePrimitive<arrow::Int32Array, int32_t>(source, target, count);
			break;
		case LogicalTypeId::BIGINT:
			DecodePrimitive<arrow::Int64Array, int64_t>(source, target, count);
			break;
		case LogicalTypeId::UTINYINT:
			DecodePrimitive<arrow::UInt8Array, uint8_t>(source, target, count);
			break;
		case LogicalTypeId::USMALLINT:
			DecodePrimitive<arrow::UInt16Array, uint16_t>(source, target, count);
			break;
		case LogicalTypeId::UINTEGER:
			DecodePrimitive<arrow::UInt32Array, uint32_t>(source, target, count);
			break;
		case LogicalTypeId::UBIGINT:
			DecodePrimitive<arrow::UInt64Array, uint64_t>(source, target, count);
			break;
		case LogicalTypeId::FLOAT:
			DecodePrimitive<arrow::FloatArray, float>(source, target, count);
			break;
		case LogicalTypeId::DOUBLE:
			DecodePrimitive<arrow::DoubleArray, double>(source, target, count);
			break;
		case LogicalTypeId::VARCHAR: {
			auto &values = static_cast<const arrow::StringArray &>(source);
			auto data = FlatVector::GetData<string_t>(target);
			for (idx_t row = 0; row < count; row++) {
				auto view = values.GetView(row);
				data[row] = string_t(view.data(), view.size());
			}
			break;
		}
		case LogicalTypeId::SQLNULL:
			break;
		default:
			throw NotImplementedException("unsupported direct Flight type");
		}
		for (idx_t row = 0; row < count; row++) {
			if (source.IsNull(row)) {
				FlatVector::Validity(target).SetInvalid(row);
			}
		}
	}
}

} // namespace vane_execution
} // namespace duckdb
