// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "arrow_frame.hpp"
#include "exchange_types.hpp"
#include "duckdb/common/types/hugeint.hpp"
#include "duckdb/common/types/vector_buffer.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/main/client_properties.hpp"
#include <arrow/c/bridge.h>

namespace duckdb {
namespace vane_execution {
namespace {
LogicalType StorageTypeFor(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::TIME:
		return LogicalType::BIGINT;
	case LogicalTypeId::INTERVAL:
		return LogicalType::STRUCT(
		    {{"months", LogicalType::INTEGER}, {"days", LogicalType::INTEGER}, {"micros", LogicalType::BIGINT}});
	case LogicalTypeId::STRUCT: {
		child_list_t<LogicalType> children;
		for (auto &child : StructType::GetChildTypes(type)) {
			children.emplace_back(child.first, StorageTypeFor(child.second));
		}
		return LogicalType::STRUCT(std::move(children));
	}
	case LogicalTypeId::LIST:
		return LogicalType::LIST(StorageTypeFor(ListType::GetChildType(type)));
	case LogicalTypeId::MAP:
		return LogicalType::MAP(StorageTypeFor(MapType::KeyType(type)), StorageTypeFor(MapType::ValueType(type)));
	case LogicalTypeId::ARRAY:
		return LogicalType::ARRAY(StorageTypeFor(ArrayType::GetChildType(type)), ArrayType::GetSize(type));
	default:
		return type;
	}
}

// Work on a private, compact flat vector. In particular INTERVAL must be
// transformed before the native Arrow exporter multiplies micros by 1000.
Vector StorageVector(Vector &source, idx_t count, const LogicalType &storage_type) {
	if (source.GetType() == storage_type) {
		return Vector(source);
	}
	Vector result(storage_type, count);
	FlatVector::SetValidity(result, FlatVector::Validity(source));
	switch (source.GetType().id()) {
	case LogicalTypeId::TIME:
		result.Reinterpret(source);
		break;
	case LogicalTypeId::INTERVAL: {
		auto data = FlatVector::GetData<interval_t>(source);
		auto &children = StructVector::GetEntries(result);
		for (idx_t row = 0; row < count; row++) {
			auto value = FlatVector::Validity(source).RowIsValid(row) ? data[row] : interval_t {0, 0, 0};
			FlatVector::GetData<int32_t>(*children[0])[row] = value.months;
			FlatVector::GetData<int32_t>(*children[1])[row] = value.days;
			FlatVector::GetData<int64_t>(*children[2])[row] = value.micros;
		}
		break;
	}
	case LogicalTypeId::STRUCT: {
		auto &inputs = StructVector::GetEntries(source);
		auto &outputs = StructVector::GetEntries(result);
		for (idx_t i = 0; i < inputs.size(); i++) {
			auto child = StorageVector(*inputs[i], count, outputs[i]->GetType());
			outputs[i]->Reference(child);
		}
		break;
	}
	case LogicalTypeId::LIST:
	case LogicalTypeId::MAP: {
		auto child_count = ListVector::GetListSize(source);
		ListVector::Reserve(result, child_count);
		auto child = StorageVector(ListVector::GetEntry(source), child_count, ListVector::GetEntry(result).GetType());
		ListVector::GetEntry(result).Reference(child);
		ListVector::SetListSize(result, child_count);
		auto entries = FlatVector::GetData<list_entry_t>(source);
		for (idx_t row = 0; row < count; row++) {
			FlatVector::GetData<list_entry_t>(result)[row] =
			    FlatVector::Validity(source).RowIsValid(row) ? entries[row] : list_entry_t {0, 0};
		}
		break;
	}
	case LogicalTypeId::ARRAY: {
		auto child = StorageVector(ArrayVector::GetEntry(source), count * ArrayType::GetSize(source.GetType()),
		                           ArrayVector::GetEntry(result).GetType());
		ArrayVector::GetEntry(result).Reference(child);
		break;
	}
	default:
		throw InternalException("unexpected exchange storage type");
	}
	return result;
}

std::shared_ptr<arrow::DataType> ArrowTypeFor(const LogicalType &type, bool native_layout = false) {
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
		return native_layout ? arrow::decimal128(38, 0) : arrow::decimal256(39, 0);
	case LogicalTypeId::DATE:
		return arrow::date32();
	case LogicalTypeId::TIME:
		return arrow::int64(); // Microseconds since midnight, including 24:00:00.
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
		return ArrowTypeFor(StorageTypeFor(type), native_layout);
	case LogicalTypeId::DECIMAL:
		return arrow::decimal128(DecimalType::GetWidth(type), DecimalType::GetScale(type));
	case LogicalTypeId::STRUCT: {
		std::vector<std::shared_ptr<arrow::Field>> fields;
		for (auto &child : StructType::GetChildTypes(type)) {
			fields.push_back(arrow::field(child.first, ArrowTypeFor(child.second, native_layout)));
		}
		return arrow::struct_(std::move(fields));
	}
	case LogicalTypeId::LIST:
		return arrow::list(arrow::field("l", ArrowTypeFor(ListType::GetChildType(type), native_layout)));
	case LogicalTypeId::MAP:
		return arrow::map(ArrowTypeFor(MapType::KeyType(type), native_layout),
		                  ArrowTypeFor(MapType::ValueType(type), native_layout));
	case LogicalTypeId::ARRAY:
		return arrow::fixed_size_list(ArrowTypeFor(ArrayType::GetChildType(type), native_layout),
		                              ArrayType::GetSize(type));
	default:
		throw NotImplementedException("unsupported exchange Arrow type: %s", type.ToString());
	}
}

void CheckArrow(const arrow::Status &status) {
	if (!status.ok()) {
		throw IOException("native Arrow frame: %s", status.ToString());
	}
}

std::shared_ptr<arrow::ArrayData> WidenHugeInts(const std::shared_ptr<arrow::ArrayData> &data,
                                                const std::shared_ptr<arrow::DataType> &type) {
	if (data->type->Equals(type)) {
		return data;
	}
	if (type->id() == arrow::Type::DECIMAL256) {
		// The native exporter owns a signed 128-bit buffer. Its decimal128(38)
		// schema cannot describe the entire HUGEINT domain. Widen the values,
		// including nested ones, before publishing a valid Arrow batch.
		arrow::Decimal128Array values(data);
		arrow::Decimal256Builder builder(type);
		CheckArrow(builder.Reserve(values.length()));
		for (int64_t row = 0; row < values.length(); row++) {
			if (values.IsNull(row)) {
				builder.UnsafeAppendNull();
			} else {
				builder.UnsafeAppend(arrow::Decimal256(arrow::Decimal128(values.GetValue(row))));
			}
		}
		std::shared_ptr<arrow::Array> result;
		CheckArrow(builder.Finish(&result));
		return result->data();
	}
	auto result = data->Copy();
	result->type = type;
	for (idx_t i = 0; i < data->child_data.size(); i++) {
		result->child_data[i] = WidenHugeInts(data->child_data[i], type->field(i)->type());
	}
	return result;
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
	DataChunk storage;
	vector<LogicalType> storage_types;
	for (auto &column : chunk.data) {
		storage_types.push_back(StorageTypeFor(column.GetType()));
	}
	storage.InitializeEmpty(storage_types);
	storage.SetCardinality(chunk.size());
	for (idx_t i = 0; i < chunk.ColumnCount(); i++) {
		if (storage_types[i] == chunk.data[i].GetType()) {
			storage.data[i].Reference(chunk.data[i]);
		} else {
			Vector flat(chunk.data[i].GetType(), chunk.size());
			VectorOperations::Copy(chunk.data[i], flat, chunk.size(), 0, 0);
			auto converted = StorageVector(flat, chunk.size(), storage_types[i]);
			storage.data[i].Reference(converted);
		}
	}
	ClientProperties properties;
	ArrowArray array;
	array.Init();
	unordered_map<idx_t, const shared_ptr<ArrowTypeExtensionData>> extensions;
	ArrowConverter::ToArrowArray(storage, &array, properties, extensions);
	std::vector<std::shared_ptr<arrow::Field>> native_fields;
	for (idx_t i = 0; i < chunk.ColumnCount(); i++) {
		native_fields.push_back(schema->field(i)->WithType(ArrowTypeFor(storage_types[i], true)));
	}
	auto result = arrow::ImportRecordBatch(&array, arrow::schema(std::move(native_fields)));
	if (!result.ok()) {
		throw IOException("native Arrow frame: %s", result.status().ToString());
	}
	auto native = std::move(result).ValueOrDie();
	std::vector<std::shared_ptr<arrow::Array>> columns;
	for (idx_t i = 0; i < chunk.ColumnCount(); i++) {
		columns.push_back(arrow::MakeArray(WidenHugeInts(native->column(i)->data(), schema->field(i)->type())));
	}
	return arrow::RecordBatch::Make(schema, chunk.size(), std::move(columns));
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
	case LogicalTypeId::TIME: {
		auto &values = static_cast<const arrow::Int64Array &>(source);
		for (idx_t row = 0; row < count; row++) {
			if (!source.IsNull(row) && (values.Value(row) < 0 || values.Value(row) > Interval::MICROS_PER_DAY)) {
				throw IOException("exchange value is outside the TIME range");
			}
		}
		DecodePrimitive<arrow::Int64Array, int64_t>(source, target, count);
		break;
	}
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::TIMESTAMP_TZ:
		DecodePrimitive<arrow::TimestampArray, int64_t>(source, target, count);
		break;
	case LogicalTypeId::HUGEINT: {
		auto &values = static_cast<const arrow::Decimal256Array &>(source);
		for (idx_t row = 0; row < count; row++) {
			if (source.IsNull(row)) {
				continue;
			}
			arrow::Decimal256 value(values.GetValue(row));
			auto words = value.little_endian_array();
			hugeint_t native(int64_t(words[1]), words[0]);
			if (value != arrow::Decimal256(arrow::Decimal128(native.upper, native.lower))) {
				throw IOException("exchange value is outside the HUGEINT range");
			}
			FlatVector::GetData<hugeint_t>(target)[row] = native;
		}
		break;
	}
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
		auto &values = static_cast<const arrow::StructArray &>(source);
		auto &months = static_cast<const arrow::Int32Array &>(*values.field(0));
		auto &days = static_cast<const arrow::Int32Array &>(*values.field(1));
		auto &micros = static_cast<const arrow::Int64Array &>(*values.field(2));
		auto data = FlatVector::GetData<interval_t>(target);
		for (idx_t row = 0; row < count; row++) {
			if (source.IsNull(row)) {
				continue;
			}
			if (months.IsNull(row) || days.IsNull(row) || micros.IsNull(row)) {
				throw IOException("exchange INTERVAL has a null component");
			}
			data[row] = {months.Value(row), days.Value(row), micros.Value(row)};
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
		// Parent container capacity may exceed the decoded element count.
		// Publish this ARRAY's actual length, rather than the record batch's
		// cardinality, before recursively decoding its child values.
		target.GetAuxiliary()->Cast<VectorArrayBuffer>().SetSize(count);
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
