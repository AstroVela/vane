// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "arrow_frame.hpp"
#include "exchange_types.hpp"
#include "duckdb/common/types/hugeint.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/main/client_properties.hpp"
#include <arrow/c/bridge.h>

namespace duckdb {
namespace vane_execution {
namespace {
std::shared_ptr<arrow::DataType> ArrowTypeFor(const LogicalType &type) {
	CheckExchangeType(type);
	switch (type.id()) {
	case LogicalTypeId::BOOLEAN:
		return arrow::boolean();
	case LogicalTypeId::TINYINT:
		return arrow::int8();
	case LogicalTypeId::SMALLINT:
		return arrow::int16();
	case LogicalTypeId::INTEGER:
		return arrow::int32();
	case LogicalTypeId::BIGINT:
		return arrow::int64();
	case LogicalTypeId::UTINYINT:
		return arrow::uint8();
	case LogicalTypeId::USMALLINT:
		return arrow::uint16();
	case LogicalTypeId::UINTEGER:
		return arrow::uint32();
	case LogicalTypeId::UBIGINT:
		return arrow::uint64();
	case LogicalTypeId::FLOAT:
		return arrow::float32();
	case LogicalTypeId::DOUBLE:
		return arrow::float64();
	case LogicalTypeId::VARCHAR:
		return arrow::utf8();
	case LogicalTypeId::BLOB:
		return arrow::binary();
	case LogicalTypeId::SQLNULL:
		return arrow::null();
	case LogicalTypeId::HUGEINT:
		return arrow::decimal128(38, 0);
	case LogicalTypeId::DATE:
		return arrow::date32();
	case LogicalTypeId::TIME:
		return arrow::time64(arrow::TimeUnit::MICRO);
	case LogicalTypeId::TIMESTAMP:
		return arrow::timestamp(arrow::TimeUnit::MICRO);
	case LogicalTypeId::TIMESTAMP_SEC:
		return arrow::timestamp(arrow::TimeUnit::SECOND);
	case LogicalTypeId::TIMESTAMP_MS:
		return arrow::timestamp(arrow::TimeUnit::MILLI);
	case LogicalTypeId::TIMESTAMP_NS:
		return arrow::timestamp(arrow::TimeUnit::NANO);
	case LogicalTypeId::TIMESTAMP_TZ:
		return arrow::timestamp(arrow::TimeUnit::MICRO, "UTC");
	case LogicalTypeId::INTERVAL:
		return arrow::month_day_nano_interval();
	case LogicalTypeId::DECIMAL:
		return arrow::decimal128(DecimalType::GetWidth(type), DecimalType::GetScale(type));
	case LogicalTypeId::STRUCT: {
		std::vector<std::shared_ptr<arrow::Field>> fields;
		for (auto &child : StructType::GetChildTypes(type)) {
			fields.push_back(arrow::field(child.first, ArrowTypeFor(child.second)));
		}
		return arrow::struct_(std::move(fields));
	}
	case LogicalTypeId::LIST:
		return arrow::list(arrow::field("l", ArrowTypeFor(ListType::GetChildType(type))));
	case LogicalTypeId::MAP:
		return arrow::map(ArrowTypeFor(MapType::KeyType(type)), ArrowTypeFor(MapType::ValueType(type)));
	case LogicalTypeId::ARRAY:
		return arrow::fixed_size_list(ArrowTypeFor(ArrayType::GetChildType(type)), ArrayType::GetSize(type));
	default:
		throw NotImplementedException("unsupported exchange Arrow type: %s", type.ToString());
	}
}
} // namespace

std::shared_ptr<arrow::Schema> ArrowSchemaFor(const vector<LogicalType> &types, const vector<string> &names) {
	std::vector<std::shared_ptr<arrow::Field>> fields;
	for (idx_t i = 0; i < types.size(); i++) {
		fields.push_back(arrow::field(names.empty() ? "c" + std::to_string(i) : names[i], ArrowTypeFor(types[i])));
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

// String and blob references borrow the Arrow batch until the channel copies
// them into its reserved frame. Nested vectors are decoded column by column;
// the staging reservation covers the complete received batch.
void DecodeVector(const arrow::Array &source, Vector &target) {
	const auto count = idx_t(source.length());
	const auto &type = target.GetType();
	switch (type.id()) {
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
	case LogicalTypeId::DATE:
		DecodePrimitive<arrow::Date32Array, int32_t>(source, target, count);
		break;
	case LogicalTypeId::TIME:
		DecodePrimitive<arrow::Time64Array, int64_t>(source, target, count);
		break;
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::TIMESTAMP_TZ:
		DecodePrimitive<arrow::TimestampArray, int64_t>(source, target, count);
		break;
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::DECIMAL: {
		auto &values = static_cast<const arrow::Decimal128Array &>(source);
		for (idx_t row = 0; row < count; row++) {
			if (source.IsNull(row)) {
				continue;
			}
			arrow::Decimal128 value(values.GetValue(row));
			hugeint_t native(value.high_bits(), value.low_bits());
			switch (type.InternalType()) {
			case PhysicalType::INT16:
				FlatVector::GetData<int16_t>(target)[row] = Hugeint::Cast<int16_t>(native);
				break;
			case PhysicalType::INT32:
				FlatVector::GetData<int32_t>(target)[row] = Hugeint::Cast<int32_t>(native);
				break;
			case PhysicalType::INT64:
				FlatVector::GetData<int64_t>(target)[row] = Hugeint::Cast<int64_t>(native);
				break;
			default:
				FlatVector::GetData<hugeint_t>(target)[row] = native;
				break;
			}
		}
		break;
	}
	case LogicalTypeId::INTERVAL: {
		auto &values = static_cast<const arrow::MonthDayNanoIntervalArray &>(source);
		auto data = FlatVector::GetData<interval_t>(target);
		for (idx_t row = 0; row < count; row++) {
			auto value = values.Value(row);
			if (!source.IsNull(row) && value.nanoseconds % 1000) {
				throw IOException("exchange interval loses microsecond precision");
			}
			data[row] = {value.months, value.days, value.nanoseconds / 1000};
		}
		break;
	}
	case LogicalTypeId::VARCHAR:
	case LogicalTypeId::BLOB: {
		auto &values = static_cast<const arrow::BinaryArray &>(source);
		auto data = FlatVector::GetData<string_t>(target);
		for (idx_t row = 0; row < count; row++) {
			auto view = values.GetView(row);
			data[row] = string_t(view.data(), view.size());
		}
		break;
	}
	case LogicalTypeId::STRUCT: {
		auto &values = static_cast<const arrow::StructArray &>(source);
		auto &children = StructVector::GetEntries(target);
		for (idx_t i = 0; i < children.size(); i++) {
			DecodeVector(*values.field(i), *children[i]);
		}
		break;
	}
	case LogicalTypeId::LIST:
	case LogicalTypeId::MAP: {
		auto &values = static_cast<const arrow::ListArray &>(source);
		auto start = values.value_offset(0);
		auto length = values.value_offset(count) - start;
		ListVector::Reserve(target, length);
		ListVector::SetListSize(target, length);
		DecodeVector(*values.values()->Slice(start, length), ListVector::GetEntry(target));
		auto entries = FlatVector::GetData<list_entry_t>(target);
		for (idx_t row = 0; row < count; row++) {
			entries[row] = {idx_t(values.value_offset(row) - start), idx_t(values.value_length(row))};
		}
		break;
	}
	case LogicalTypeId::ARRAY: {
		auto &values = static_cast<const arrow::FixedSizeListArray &>(source);
		auto size = ArrayType::GetSize(type);
		DecodeVector(*values.values()->Slice(values.value_offset(0), count * size), ArrayVector::GetEntry(target));
		break;
	}
	case LogicalTypeId::SQLNULL:
		break;
	default:
		throw NotImplementedException("unsupported exchange Arrow type: %s", type.ToString());
	}
	for (idx_t row = 0; row < count; row++) {
		if (source.IsNull(row)) {
			FlatVector::Validity(target).SetInvalid(row);
		}
	}
}

void Decode(const arrow::RecordBatch &batch, const vector<LogicalType> &types, DataChunk &chunk) {
	if (!batch.schema()->Equals(*ArrowSchemaFor(types))) {
		throw IOException("exchange Arrow schema mismatch");
	}
	auto validation = batch.ValidateFull();
	if (!validation.ok()) {
		throw IOException("invalid exchange Arrow frame: %s", validation.ToString());
	}
	const auto count = idx_t(batch.num_rows());
	chunk.Initialize(Allocator::DefaultAllocator(), types, MaxValue<idx_t>(count, 1));
	chunk.SetCardinality(count);
	for (idx_t col = 0; col < types.size(); col++) {
		DecodeVector(*batch.column(col), chunk.data[col]);
	}
}

} // namespace vane_execution
} // namespace duckdb
