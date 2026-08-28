// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "arrow_parquet_writer.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/serializer/buffered_file_writer.hpp"
#include "duckdb/function/copy_function.hpp"
#include "duckdb/parser/keyword_helper.hpp"

#include <arrow/array.h>
#include <arrow/c/bridge.h>
#include <arrow/extension_type.h>
#include <arrow/io/interfaces.h>
#include <arrow/record_batch.h>
#include <arrow/scalar.h>
#include <arrow/status.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/schema.h>
#include <parquet/arrow/writer.h>
#include <parquet/metadata.h>
#include <parquet/properties.h>
#include <parquet/schema.h>
#include <parquet/statistics.h>
#include <parquet/types.h>

#include <cmath>
#include <limits>
#include <memory>

namespace duckdb {

namespace {

static string ArrowError(const ::arrow::Status &status) {
	return status.ToString();
}

static void ThrowArrowInputError(const ::arrow::Status &status, const char *action) {
	throw InvalidInputException("Failed to %s for Arrow-native Parquet COPY: %s", action, ArrowError(status));
}

static void ThrowArrowWriterError(const ::arrow::Status &status, const char *action) {
	throw IOException("Failed to %s for Arrow-native Parquet COPY: %s", action, ArrowError(status));
}

template <class PARQUET_TYPE>
static void MergeTypedStatistics(std::shared_ptr<::parquet::Statistics> &target,
                                 const std::shared_ptr<::parquet::Statistics> &source) {
	if (!target) {
		target = source;
		return;
	}
	auto typed_target = std::static_pointer_cast<::parquet::TypedStatistics<PARQUET_TYPE>>(target);
	auto typed_source = std::static_pointer_cast<::parquet::TypedStatistics<PARQUET_TYPE>>(source);
	typed_target->Merge(*typed_source);
}

static bool MergeStatistics(std::shared_ptr<::parquet::Statistics> &target,
                            const std::shared_ptr<::parquet::Statistics> &source) {
	switch (source->physical_type()) {
	case ::parquet::Type::BOOLEAN:
		MergeTypedStatistics<::parquet::BooleanType>(target, source);
		return true;
	case ::parquet::Type::INT32:
		MergeTypedStatistics<::parquet::Int32Type>(target, source);
		return true;
	case ::parquet::Type::INT64:
		MergeTypedStatistics<::parquet::Int64Type>(target, source);
		return true;
	case ::parquet::Type::FLOAT:
		MergeTypedStatistics<::parquet::FloatType>(target, source);
		return true;
	case ::parquet::Type::DOUBLE:
		MergeTypedStatistics<::parquet::DoubleType>(target, source);
		return true;
	case ::parquet::Type::BYTE_ARRAY:
		MergeTypedStatistics<::parquet::ByteArrayType>(target, source);
		return true;
	case ::parquet::Type::FIXED_LEN_BYTE_ARRAY:
		MergeTypedStatistics<::parquet::FLBAType>(target, source);
		return true;
	default:
		return false;
	}
}

static string BinaryScalarValue(const ::arrow::BaseBinaryScalar &scalar) {
	if (!scalar.value || scalar.value->size() == 0) {
		return string();
	}
	return string(reinterpret_cast<const char *>(scalar.value->data()), NumericCast<idx_t>(scalar.value->size()));
}

static string UUIDScalarToString(const ::arrow::BaseBinaryScalar &scalar) {
	if (!scalar.value || scalar.value->size() != 16) {
		return string();
	}
	static constexpr char UUID_DIGITS[] = "0123456789abcdef";
	string result;
	result.reserve(36);
	for (idx_t byte_idx = 0; byte_idx < 16; byte_idx++) {
		if (byte_idx == 4 || byte_idx == 6 || byte_idx == 8 || byte_idx == 10) {
			result += "-";
		}
		auto byte = scalar.value->data()[byte_idx];
		result += UUID_DIGITS[byte >> 4];
		result += UUID_DIGITS[byte & 0x0F];
	}
	return result;
}

static string StatisticsScalarToString(const ::arrow::Scalar &scalar, const LogicalType &sql_type) {
	if (scalar.type->id() == ::arrow::Type::BOOL) {
		return static_cast<const ::arrow::BooleanScalar &>(scalar).value ? "1" : "0";
	}
	if (scalar.type->id() == ::arrow::Type::STRING || scalar.type->id() == ::arrow::Type::LARGE_STRING) {
		return scalar.ToString();
	}
	auto binary = dynamic_cast<const ::arrow::BaseBinaryScalar *>(&scalar);
	if (!binary) {
		return Value(scalar.ToString()).DefaultCastAs(sql_type).ToString();
	}
	if (sql_type.id() == LogicalTypeId::BLOB) {
		return binary->value ? binary->value->ToHexString() : string();
	}
	if (sql_type.id() == LogicalTypeId::UUID) {
		return UUIDScalarToString(*binary);
	}
	return BinaryScalarValue(*binary);
}

static bool CanReturnMinMax(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::BOOLEAN:
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
	case LogicalTypeId::DATE:
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_NS:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::TIMESTAMP_TZ:
	case LogicalTypeId::BLOB:
	case LogicalTypeId::VARCHAR:
	case LogicalTypeId::UUID:
		return true;
	default:
		return false;
	}
}

static string ChildStatisticsName(const string &base_name, const string &child_name) {
	return base_name + "." + KeywordHelper::WriteQuoted(child_name, '"');
}

static void AppendLeafMetadata(const LogicalType &type, const string &name, vector<LogicalType> &leaf_types,
                               vector<string> &leaf_names) {
	switch (type.id()) {
	case LogicalTypeId::STRUCT:
	case LogicalTypeId::UNION:
		for (const auto &child : StructType::GetChildTypes(type)) {
			AppendLeafMetadata(child.second, ChildStatisticsName(name, child.first), leaf_types, leaf_names);
		}
		return;
	case LogicalTypeId::LIST:
		AppendLeafMetadata(ListType::GetChildType(type), ChildStatisticsName(name, "element"), leaf_types, leaf_names);
		return;
	case LogicalTypeId::ARRAY:
		AppendLeafMetadata(ArrayType::GetChildType(type), ChildStatisticsName(name, "element"), leaf_types, leaf_names);
		return;
	case LogicalTypeId::MAP:
		AppendLeafMetadata(MapType::KeyType(type), ChildStatisticsName(name, "key"), leaf_types, leaf_names);
		AppendLeafMetadata(MapType::ValueType(type), ChildStatisticsName(name, "value"), leaf_types, leaf_names);
		return;
	default:
		leaf_types.push_back(type);
		leaf_names.push_back(name);
		return;
	}
}

static std::shared_ptr<::arrow::Array> ExtensionStorage(std::shared_ptr<::arrow::Array> array) {
	while (array->type_id() == ::arrow::Type::EXTENSION) {
		array = static_cast<const ::arrow::ExtensionArray &>(*array).storage();
	}
	return array;
}

static std::shared_ptr<::arrow::DataType> ExtensionStorageType(std::shared_ptr<::arrow::DataType> type) {
	while (type->id() == ::arrow::Type::EXTENSION) {
		type = static_cast<const ::arrow::ExtensionType &>(*type).storage_type();
	}
	return type;
}

static void ValidateArrowNativeParquetEncoding(std::shared_ptr<::arrow::DataType> type) {
	type = ExtensionStorageType(std::move(type));
	if (type->id() == ::arrow::Type::TIMESTAMP &&
	    static_cast<const ::arrow::TimestampType &>(*type).unit() == ::arrow::TimeUnit::NANO) {
		throw NotImplementedException("Nanosecond Arrow timestamps are not supported by Arrow-native Parquet V1 COPY");
	}
	if (type->id() == ::arrow::Type::RUN_END_ENCODED) {
		throw NotImplementedException("Run-end encoded arrays are not supported by Arrow-native Parquet COPY");
	}
	if (type->id() == ::arrow::Type::LIST_VIEW || type->id() == ::arrow::Type::LARGE_LIST_VIEW) {
		throw NotImplementedException("List-view arrays are not supported by Arrow-native Parquet COPY");
	}
	if (type->id() == ::arrow::Type::DICTIONARY) {
		auto value_type = ExtensionStorageType(static_cast<const ::arrow::DictionaryType &>(*type).value_type());
		if (value_type->id() == ::arrow::Type::DICTIONARY || value_type->id() == ::arrow::Type::RUN_END_ENCODED ||
		    value_type->num_fields() != 0) {
			throw NotImplementedException(
			    "Dictionary-encoded nested values are not supported by Arrow-native Parquet COPY");
		}
		ValidateArrowNativeParquetEncoding(std::move(value_type));
		return;
	}
	for (const auto &field : type->fields()) {
		ValidateArrowNativeParquetEncoding(field->type());
	}
}

template <class ARRAY_TYPE>
static std::shared_ptr<::arrow::Array> FlattenListArray(const ::arrow::Array &array) {
	auto flattened = static_cast<const ARRAY_TYPE &>(array).Flatten();
	if (!flattened.ok()) {
		ThrowArrowInputError(flattened.status(), "flatten a nested Arrow array for RETURN_STATS");
	}
	return std::move(flattened).ValueUnsafe();
}

static std::shared_ptr<::arrow::Array> FlattenListArray(std::shared_ptr<::arrow::Array> array) {
	array = ExtensionStorage(std::move(array));
	switch (array->type_id()) {
	case ::arrow::Type::LIST:
	case ::arrow::Type::MAP:
		return FlattenListArray<::arrow::ListArray>(*array);
	case ::arrow::Type::LARGE_LIST:
		return FlattenListArray<::arrow::LargeListArray>(*array);
	case ::arrow::Type::FIXED_SIZE_LIST:
		return FlattenListArray<::arrow::FixedSizeListArray>(*array);
	default:
		throw InvalidInputException("Arrow-native Parquet COPY expected a list array for RETURN_STATS, received %s",
		                            array->type()->ToString());
	}
}

template <class ARRAY_TYPE>
static bool PrimitiveFloatingArrayHasNaN(const ::arrow::Array &array) {
	auto &values = static_cast<const ARRAY_TYPE &>(array);
	for (int64_t row_idx = 0; row_idx < values.length(); row_idx++) {
		if (values.IsValid(row_idx) && std::isnan(values.Value(row_idx))) {
			return true;
		}
	}
	return false;
}

template <class ARRAY_TYPE>
static bool DictionaryFloatingArrayHasNaN(const ::arrow::DictionaryArray &encoded, const ::arrow::Array &dictionary) {
	auto &values = static_cast<const ARRAY_TYPE &>(dictionary);
	vector<uint8_t> nan_state(NumericCast<idx_t>(values.length()), 0);
	for (int64_t row_idx = 0; row_idx < encoded.length(); row_idx++) {
		if (!encoded.IsValid(row_idx)) {
			continue;
		}
		auto dictionary_idx = encoded.GetValueIndex(row_idx);
		if (dictionary_idx < 0 || dictionary_idx >= values.length()) {
			throw InvalidInputException("Arrow dictionary index is outside the dictionary for RETURN_STATS");
		}
		auto &state = nan_state[NumericCast<idx_t>(dictionary_idx)];
		if (state == 0) {
			state = values.IsValid(dictionary_idx) && std::isnan(values.Value(dictionary_idx)) ? 2 : 1;
		}
		if (state == 2) {
			return true;
		}
	}
	return false;
}

static bool FloatingArrayHasNaN(std::shared_ptr<::arrow::Array> array) {
	array = ExtensionStorage(std::move(array));
	switch (array->type_id()) {
	case ::arrow::Type::FLOAT:
		return PrimitiveFloatingArrayHasNaN<::arrow::FloatArray>(*array);
	case ::arrow::Type::DOUBLE:
		return PrimitiveFloatingArrayHasNaN<::arrow::DoubleArray>(*array);
	case ::arrow::Type::DICTIONARY: {
		auto &dictionary_array = static_cast<const ::arrow::DictionaryArray &>(*array);
		auto dictionary = ExtensionStorage(dictionary_array.dictionary());
		switch (dictionary->type_id()) {
		case ::arrow::Type::FLOAT:
			return DictionaryFloatingArrayHasNaN<::arrow::FloatArray>(dictionary_array, *dictionary);
		case ::arrow::Type::DOUBLE:
			return DictionaryFloatingArrayHasNaN<::arrow::DoubleArray>(dictionary_array, *dictionary);
		default:
			throw InvalidInputException(
			    "Arrow-native Parquet COPY expected a FLOAT or DOUBLE dictionary for RETURN_STATS, received %s",
			    dictionary->type()->ToString());
		}
	}
	default:
		throw InvalidInputException(
		    "Arrow-native Parquet COPY expected FLOAT or DOUBLE data for RETURN_STATS, received %s",
		    array->type()->ToString());
	}
}

static void AccumulateNaNs(std::shared_ptr<::arrow::Array> array, const LogicalType &type, vector<bool> &has_nan,
                           idx_t &leaf_idx) {
	switch (type.id()) {
	case LogicalTypeId::STRUCT:
	case LogicalTypeId::UNION: {
		array = ExtensionStorage(std::move(array));
		if (array->type_id() != ::arrow::Type::STRUCT) {
			throw InvalidInputException(
			    "Arrow-native Parquet COPY expected a STRUCT array for RETURN_STATS, received %s",
			    array->type()->ToString());
		}
		auto flattened = static_cast<const ::arrow::StructArray &>(*array).Flatten();
		if (!flattened.ok()) {
			ThrowArrowInputError(flattened.status(), "flatten a struct Arrow array for RETURN_STATS");
		}
		auto children = std::move(flattened).ValueUnsafe();
		auto &child_types = StructType::GetChildTypes(type);
		if (children.size() != child_types.size()) {
			throw InvalidInputException("Arrow STRUCT child count changed during Arrow-native Parquet COPY");
		}
		for (idx_t child_idx = 0; child_idx < child_types.size(); child_idx++) {
			AccumulateNaNs(std::move(children[child_idx]), child_types[child_idx].second, has_nan, leaf_idx);
		}
		return;
	}
	case LogicalTypeId::LIST:
		AccumulateNaNs(FlattenListArray(std::move(array)), ListType::GetChildType(type), has_nan, leaf_idx);
		return;
	case LogicalTypeId::ARRAY:
		AccumulateNaNs(FlattenListArray(std::move(array)), ArrayType::GetChildType(type), has_nan, leaf_idx);
		return;
	case LogicalTypeId::MAP: {
		auto entries = ExtensionStorage(FlattenListArray(std::move(array)));
		if (entries->type_id() != ::arrow::Type::STRUCT) {
			throw InvalidInputException("Arrow-native Parquet COPY expected MAP entries to be a STRUCT array");
		}
		auto flattened = static_cast<const ::arrow::StructArray &>(*entries).Flatten();
		if (!flattened.ok()) {
			ThrowArrowInputError(flattened.status(), "flatten map entries for RETURN_STATS");
		}
		auto children = std::move(flattened).ValueUnsafe();
		if (children.size() != 2) {
			throw InvalidInputException("Arrow MAP entry child count changed during Arrow-native Parquet COPY");
		}
		AccumulateNaNs(std::move(children[0]), MapType::KeyType(type), has_nan, leaf_idx);
		AccumulateNaNs(std::move(children[1]), MapType::ValueType(type), has_nan, leaf_idx);
		return;
	}
	default:
		if (leaf_idx >= has_nan.size()) {
			throw InternalException("Arrow-native Parquet RETURN_STATS leaf count exceeds the bound schema");
		}
		if (!has_nan[leaf_idx] && (type.id() == LogicalTypeId::FLOAT || type.id() == LogicalTypeId::DOUBLE)) {
			has_nan[leaf_idx] = FloatingArrayHasNaN(std::move(array));
		}
		leaf_idx++;
		return;
	}
}

class DuckDBArrowOutputStream : public ::arrow::io::OutputStream {
public:
	DuckDBArrowOutputStream(FileSystem &fs, const string &file_path)
	    : writer(fs, file_path, FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE_NEW) {
		set_mode(::arrow::io::FileMode::WRITE);
	}

	~DuckDBArrowOutputStream() override {
		if (!is_closed) {
			(void)Close();
		}
	}

	::arrow::Status Close() override {
		if (is_closed) {
			return ::arrow::Status::OK();
		}
		try {
			writer.Close();
			is_closed = true;
			return ::arrow::Status::OK();
		} catch (const std::exception &ex) {
			return ::arrow::Status::IOError(ex.what());
		}
	}

	bool closed() const override {
		return is_closed;
	}

	::arrow::Result<int64_t> Tell() const override {
		try {
			return NumericCast<int64_t>(writer.GetTotalWritten());
		} catch (const std::exception &ex) {
			return ::arrow::Status::IOError(ex.what());
		}
	}

	::arrow::Status Write(const void *data, int64_t nbytes) override {
		if (is_closed) {
			return ::arrow::Status::Invalid("cannot write to a closed DuckDB output stream");
		}
		if (nbytes < 0) {
			return ::arrow::Status::Invalid("cannot write a negative number of bytes");
		}
		if (nbytes == 0) {
			return ::arrow::Status::OK();
		}
		if (!data) {
			return ::arrow::Status::Invalid("cannot write from a null buffer");
		}
		try {
			writer.WriteData(reinterpret_cast<const_data_ptr_t>(data), NumericCast<idx_t>(nbytes));
			return ::arrow::Status::OK();
		} catch (const std::exception &ex) {
			return ::arrow::Status::IOError(ex.what());
		}
	}

	::arrow::Status Flush() override {
		if (is_closed) {
			return ::arrow::Status::Invalid("cannot flush a closed DuckDB output stream");
		}
		try {
			writer.Flush();
			return ::arrow::Status::OK();
		} catch (const std::exception &ex) {
			return ::arrow::Status::IOError(ex.what());
		}
	}

	idx_t Size() const {
		return writer.GetTotalWritten();
	}

private:
	BufferedFileWriter writer;
	bool is_closed = false;
};

static ::parquet::Compression::type ArrowCompression(duckdb_parquet::CompressionCodec::type codec) {
	switch (codec) {
	case duckdb_parquet::CompressionCodec::UNCOMPRESSED:
		return ::parquet::Compression::UNCOMPRESSED;
	case duckdb_parquet::CompressionCodec::SNAPPY:
		return ::parquet::Compression::SNAPPY;
	case duckdb_parquet::CompressionCodec::GZIP:
		return ::parquet::Compression::GZIP;
	case duckdb_parquet::CompressionCodec::ZSTD:
		return ::parquet::Compression::ZSTD;
	case duckdb_parquet::CompressionCodec::BROTLI:
		return ::parquet::Compression::BROTLI;
	case duckdb_parquet::CompressionCodec::LZ4_RAW:
		return ::parquet::Compression::LZ4;
	default:
		throw NotImplementedException("Compression codec is not supported by Arrow-native Parquet COPY");
	}
}

static std::shared_ptr<::parquet::WriterProperties>
BuildWriterProperties(const std::shared_ptr<::arrow::Schema> &schema, const ArrowParquetWriterOptions &options,
                      const std::shared_ptr<::parquet::ArrowWriterProperties> &arrow_properties) {
	::parquet::WriterProperties::Builder builder;
	builder.version(::parquet::ParquetVersion::PARQUET_1_0);
	builder.compression(ArrowCompression(options.codec));
	builder.max_row_group_length(NumericCast<int64_t>(options.row_group_size));
	// Keep Arrow's bounded 1 MiB dictionary fallback. DuckDB's string-only page limit is not
	// equivalent: Arrow applies this property to every column and has no separate cardinality limit.
	if (options.disable_dictionary) {
		builder.disable_dictionary();
	}
	if (options.codec == duckdb_parquet::CompressionCodec::ZSTD) {
		builder.compression_level(NumericCast<int>(options.compression_level));
	}

	if (options.enable_bloom_filters) {
		auto base_properties = builder.build();
		std::shared_ptr<::parquet::SchemaDescriptor> parquet_schema;
		auto status =
		    ::parquet::arrow::ToParquetSchema(schema.get(), *base_properties, *arrow_properties, &parquet_schema);
		if (!status.ok()) {
			ThrowArrowInputError(status, "convert the Arrow schema");
		}
		::parquet::BloomFilterOptions bloom_options;
		bloom_options.ndv = NumericCast<int32_t>(
		    MinValue<idx_t>(options.row_group_size, NumericCast<idx_t>(std::numeric_limits<int32_t>::max())));
		bloom_options.fpp = options.bloom_filter_false_positive_ratio;
		for (int column_idx = 0; column_idx < parquet_schema->num_columns(); column_idx++) {
			auto column = parquet_schema->Column(column_idx);
			if (column->physical_type() != ::parquet::Type::BOOLEAN && column->max_repetition_level() == 0) {
				builder.enable_bloom_filter(column->path(), bloom_options);
			}
		}
	}
	return builder.build();
}

} // namespace

struct ArrowParquetLocalState::Impl {
	std::shared_ptr<::arrow::Schema> schema;
	std::shared_ptr<::arrow::RecordBatch> record_batch;
};

ArrowParquetLocalState::ArrowParquetLocalState() : impl(make_uniq<Impl>()) {
}

ArrowParquetLocalState::~ArrowParquetLocalState() = default;

void ArrowParquetLocalState::ImportSchema(ArrowSchema &schema, const vector<string> &names) {
	auto imported = ::arrow::ImportSchema(&schema);
	if (!imported.ok()) {
		ThrowArrowInputError(imported.status(), "import the Arrow schema");
	}
	std::vector<std::string> normalized_names(names.begin(), names.end());
	auto imported_schema = std::move(imported).ValueUnsafe();
	auto normalized = imported_schema->WithNames(normalized_names);
	if (!normalized.ok()) {
		ThrowArrowInputError(normalized.status(), "normalize the Arrow schema names");
	}
	impl->schema = std::move(normalized).ValueUnsafe();
	impl->record_batch.reset();
}

void ArrowParquetLocalState::ImportRecordBatch(ArrowArray &array) {
	if (!impl->schema) {
		throw InternalException("Arrow-native Parquet COPY received a record batch before its schema");
	}
	auto imported = ::arrow::ImportRecordBatch(&array, impl->schema);
	if (!imported.ok()) {
		ThrowArrowInputError(imported.status(), "import the Arrow record batch");
	}
	impl->record_batch = std::move(imported).ValueUnsafe();
}

struct ArrowParquetWriter::Impl {
	Impl(FileSystem &fs, const string &file_path, std::shared_ptr<::arrow::Schema> schema_p,
	     const ArrowParquetWriterOptions &options_p)
	    : schema(std::move(schema_p)), options(options_p) {
		if (!schema) {
			throw InternalException("Arrow-native Parquet writer was initialized without a schema");
		}
		if (options.sql_types.size() != NumericCast<idx_t>(schema->num_fields())) {
			throw InternalException("Arrow-native Parquet writer SQL and Arrow schemas have different column counts");
		}
		for (idx_t column_idx = 0; column_idx < options.sql_types.size(); column_idx++) {
			auto field = schema->field(NumericCast<int>(column_idx));
			ValidateArrowNativeParquetEncoding(field->type());
			auto name = KeywordHelper::WriteQuoted(field->name(), '"');
			AppendLeafMetadata(options.sql_types[column_idx], name, leaf_types, leaf_names);
		}
		has_nan.resize(leaf_types.size(), false);

		::parquet::ArrowWriterProperties::Builder arrow_builder;
		arrow_builder.set_use_threads(false);
		auto arrow_properties = arrow_builder.build();
		auto properties = BuildWriterProperties(schema, options, arrow_properties);
		std::shared_ptr<::parquet::SchemaDescriptor> parquet_schema;
		auto schema_status =
		    ::parquet::arrow::ToParquetSchema(schema.get(), *properties, *arrow_properties, &parquet_schema);
		if (!schema_status.ok()) {
			ThrowArrowInputError(schema_status, "convert the Arrow schema");
		}
		if (NumericCast<idx_t>(parquet_schema->num_columns()) != leaf_types.size()) {
			throw InvalidInputException("Arrow and SQL schemas have different Parquet leaf counts");
		}

		// Validate the complete schema before creating the output file.
		output = std::make_shared<DuckDBArrowOutputStream>(fs, file_path);
		auto opened = ::parquet::arrow::FileWriter::Open(*schema, ::arrow::default_memory_pool(), output, properties,
		                                                 arrow_properties);
		if (!opened.ok()) {
			ThrowArrowWriterError(opened.status(), "open the Parquet writer");
		}
		writer = std::move(opened).ValueUnsafe();
		if (!options.key_value_metadata.empty()) {
			std::vector<std::string> keys;
			std::vector<std::string> values;
			keys.reserve(options.key_value_metadata.size());
			values.reserve(options.key_value_metadata.size());
			for (const auto &entry : options.key_value_metadata) {
				keys.push_back(entry.first);
				values.push_back(entry.second);
			}
			auto metadata = std::make_shared<::arrow::KeyValueMetadata>(std::move(keys), std::move(values));
			auto status = writer->AddKeyValueMetadata(metadata);
			if (!status.ok()) {
				ThrowArrowWriterError(status, "add Parquet key-value metadata");
			}
		}
	}

	std::shared_ptr<::arrow::Schema> schema;
	ArrowParquetWriterOptions options;
	std::shared_ptr<DuckDBArrowOutputStream> output;
	std::unique_ptr<::parquet::arrow::FileWriter> writer;
	optional_ptr<CopyFunctionFileStatistics> written_statistics;
	vector<LogicalType> leaf_types;
	vector<string> leaf_names;
	vector<bool> has_nan;
	idx_t rows_written = 0;
	idx_t rows_in_current_row_group = 0;
	idx_t completed_row_groups = 0;
	bool finalized = false;
};

ArrowParquetWriter::ArrowParquetWriter(FileSystem &fs, const string &file_path, ArrowParquetLocalState &local_state,
                                       const ArrowParquetWriterOptions &options)
    : impl(make_uniq<Impl>(fs, file_path, local_state.impl->schema, options)) {
}

ArrowParquetWriter::~ArrowParquetWriter() = default;

void ArrowParquetWriter::Write(ArrowParquetLocalState &local_state, idx_t offset, idx_t cardinality) {
	if (impl->finalized) {
		throw InternalException("Cannot write to a finalized Arrow-native Parquet writer");
	}
	auto &local = *local_state.impl;
	if (!local.schema || !local.record_batch) {
		throw InternalException("Arrow-native Parquet COPY has no imported record batch");
	}
	if (!impl->schema->Equals(*local.schema, false)) {
		throw InvalidInputException("Arrow schema changed during Arrow-native Parquet COPY");
	}
	const auto batch_rows = NumericCast<idx_t>(local.record_batch->num_rows());
	if (offset > batch_rows || cardinality > batch_rows - offset) {
		throw InvalidInputException("Arrow Parquet input slice exceeds its record batch");
	}
	if (cardinality > impl->options.row_group_size - impl->rows_in_current_row_group) {
		throw InternalException("Arrow-native Parquet input crossed a row-group boundary");
	}
	if (cardinality == 0) {
		return;
	}
	auto slice = local.record_batch->Slice(NumericCast<int64_t>(offset), NumericCast<int64_t>(cardinality));
	if (impl->written_statistics) {
		idx_t leaf_idx = 0;
		for (idx_t column_idx = 0; column_idx < impl->options.sql_types.size(); column_idx++) {
			AccumulateNaNs(slice->column(NumericCast<int>(column_idx)), impl->options.sql_types[column_idx],
			               impl->has_nan, leaf_idx);
		}
		if (leaf_idx != impl->has_nan.size()) {
			throw InternalException("Arrow-native Parquet RETURN_STATS leaf count changed while writing");
		}
	}
	auto status = impl->writer->WriteRecordBatch(*slice);
	if (!status.ok()) {
		ThrowArrowWriterError(status, "write the Arrow record batch");
	}
	impl->rows_written += cardinality;

	idx_t remaining = cardinality;
	while (remaining > 0) {
		auto available = impl->options.row_group_size - impl->rows_in_current_row_group;
		auto count = MinValue<idx_t>(remaining, available);
		impl->rows_in_current_row_group += count;
		remaining -= count;
		if (impl->rows_in_current_row_group == impl->options.row_group_size) {
			impl->rows_in_current_row_group = 0;
			impl->completed_row_groups++;
		}
	}
	if (cardinality == batch_rows - offset) {
		local.record_batch.reset();
	}
}

void ArrowParquetWriter::SetWrittenStatistics(CopyFunctionFileStatistics &statistics) {
	if (impl->finalized) {
		throw InternalException("Cannot attach written statistics to a finalized Arrow-native Parquet writer");
	}
	if (impl->rows_written != 0) {
		throw InternalException("Arrow-native Parquet RETURN_STATS must be attached before writing rows");
	}
	impl->written_statistics = statistics;
}

static void GatherWrittenStatistics(const ::parquet::FileMetaData &metadata, idx_t file_size,
                                    const vector<LogicalType> &leaf_types, const vector<string> &leaf_names,
                                    const vector<bool> &has_nan, CopyFunctionFileStatistics &result) {
	if (NumericCast<idx_t>(metadata.num_columns()) != leaf_types.size() || leaf_types.size() != leaf_names.size() ||
	    leaf_types.size() != has_nan.size()) {
		throw InternalException("Arrow-native Parquet RETURN_STATS schema does not match the file metadata");
	}
	result.row_count = NumericCast<idx_t>(metadata.num_rows());
	result.file_size_bytes = file_size;
	result.footer_size_bytes = Value::UBIGINT(metadata.SerializeToString().size());
	result.column_statistics.clear();

	for (int column_idx = 0; column_idx < metadata.num_columns(); column_idx++) {
		idx_t column_size_bytes = 0;
		idx_t num_values = 0;
		idx_t null_count = 0;
		bool all_null_counts_set = true;
		bool all_min_max_set = CanReturnMinMax(leaf_types[NumericCast<idx_t>(column_idx)]);
		std::shared_ptr<::parquet::Statistics> merged_statistics;

		for (int row_group_idx = 0; row_group_idx < metadata.num_row_groups(); row_group_idx++) {
			auto row_group = metadata.RowGroup(row_group_idx);
			auto column = row_group->ColumnChunk(column_idx);
			column_size_bytes += NumericCast<idx_t>(column->total_compressed_size());
			num_values += NumericCast<idx_t>(column->num_values());
			if (!column->is_stats_set()) {
				all_null_counts_set = false;
				all_min_max_set = false;
				continue;
			}
			auto statistics = column->statistics();
			if (statistics->HasNullCount()) {
				null_count += NumericCast<idx_t>(statistics->null_count());
			} else {
				all_null_counts_set = false;
			}
			if (all_min_max_set && (!statistics->HasMinMax() || !MergeStatistics(merged_statistics, statistics))) {
				all_min_max_set = false;
			}
		}

		case_insensitive_map_t<Value> column_statistics;
		column_statistics["column_size_bytes"] = Value::UBIGINT(column_size_bytes);
		column_statistics["num_values"] = Value::UBIGINT(num_values);
		if (all_null_counts_set) {
			column_statistics["null_count"] = Value::UBIGINT(null_count);
		}
		auto &leaf_type = leaf_types[NumericCast<idx_t>(column_idx)];
		if (leaf_type.id() == LogicalTypeId::FLOAT || leaf_type.id() == LogicalTypeId::DOUBLE) {
			column_statistics["has_nan"] = Value::BOOLEAN(has_nan[NumericCast<idx_t>(column_idx)]);
		}
		if (all_min_max_set && merged_statistics) {
			std::shared_ptr<::arrow::Scalar> minimum;
			std::shared_ptr<::arrow::Scalar> maximum;
			auto status = ::parquet::arrow::StatisticsAsScalars(*merged_statistics, &minimum, &maximum);
			if (!status.ok()) {
				ThrowArrowWriterError(status, "convert Parquet column statistics");
			}
			if (minimum && minimum->is_valid) {
				column_statistics["min"] = StatisticsScalarToString(*minimum, leaf_type);
			}
			if (maximum && maximum->is_valid) {
				column_statistics["max"] = StatisticsScalarToString(*maximum, leaf_type);
			}
		}

		result.column_statistics.emplace(leaf_names[NumericCast<idx_t>(column_idx)], std::move(column_statistics));
	}
}

void ArrowParquetWriter::Finalize() {
	if (impl->finalized) {
		return;
	}
	auto writer_status = impl->writer->Close();
	auto output_status = impl->output->Close();
	impl->finalized = true;
	if (!writer_status.ok()) {
		ThrowArrowWriterError(writer_status, "finalize the Parquet writer");
	}
	if (!output_status.ok()) {
		ThrowArrowWriterError(output_status, "close the Parquet output");
	}
	if (impl->written_statistics) {
		auto metadata = impl->writer->metadata();
		if (!metadata) {
			throw InternalException("Arrow-native Parquet writer finalized without file metadata");
		}
		GatherWrittenStatistics(*metadata, impl->output->Size(), impl->leaf_types, impl->leaf_names, impl->has_nan,
		                        *impl->written_statistics);
	}
}

idx_t ArrowParquetWriter::FileSize() const {
	return impl->output->Size();
}

idx_t ArrowParquetWriter::NumberOfRowGroups() const {
	return impl->completed_row_groups;
}

idx_t ArrowParquetWriter::RowsUntilRowGroupBoundary() const {
	if (impl->finalized) {
		throw InternalException("Cannot inspect a finalized Arrow-native Parquet writer");
	}
	return impl->options.row_group_size - impl->rows_in_current_row_group;
}

} // namespace duckdb
