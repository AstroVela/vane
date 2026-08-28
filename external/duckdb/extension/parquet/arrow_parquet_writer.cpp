// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "arrow_parquet_writer.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/serializer/buffered_file_writer.hpp"
#include "duckdb/function/copy_function.hpp"
#include "duckdb/parser/keyword_helper.hpp"

#include <arrow/c/bridge.h>
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

static string BinaryScalarToHex(const ::arrow::BaseBinaryScalar &scalar) {
	return scalar.value ? scalar.value->ToHexString() : string();
}

static string StatisticsScalarToString(const ::arrow::Scalar &scalar) {
	if (scalar.type->id() == ::arrow::Type::BOOL) {
		return static_cast<const ::arrow::BooleanScalar &>(scalar).value ? "1" : "0";
	}
	if (scalar.type->id() == ::arrow::Type::STRING || scalar.type->id() == ::arrow::Type::LARGE_STRING) {
		return scalar.ToString();
	}
	auto binary = dynamic_cast<const ::arrow::BaseBinaryScalar *>(&scalar);
	return binary ? BinaryScalarToHex(*binary) : scalar.ToString();
}

static string ColumnStatisticsName(const ::parquet::ColumnDescriptor &descriptor) {
	string result;
	auto path = descriptor.path();
	for (const auto &part : path->ToDotVector()) {
		if (!result.empty()) {
			result += ".";
		}
		result += KeywordHelper::WriteQuoted(part, '"');
	}
	return result;
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
	builder.compression(ArrowCompression(options.codec));
	builder.max_row_group_length(NumericCast<int64_t>(options.row_group_size));
	builder.dictionary_pagesize_limit(NumericCast<int64_t>(options.dictionary_page_size_limit));
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
			if (column->physical_type() != ::parquet::Type::BOOLEAN) {
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
	    : schema(std::move(schema_p)), options(options_p),
	      output(std::make_shared<DuckDBArrowOutputStream>(fs, file_path)) {
		if (!schema) {
			throw InternalException("Arrow-native Parquet writer was initialized without a schema");
		}
		::parquet::ArrowWriterProperties::Builder arrow_builder;
		arrow_builder.set_use_threads(false);
		auto arrow_properties = arrow_builder.build();
		auto properties = BuildWriterProperties(schema, options, arrow_properties);
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
	if (!impl->schema->Equals(*local.schema, true)) {
		throw InvalidInputException("Arrow schema changed during Arrow-native Parquet COPY");
	}
	const auto batch_rows = NumericCast<idx_t>(local.record_batch->num_rows());
	if (offset > batch_rows || cardinality > batch_rows - offset) {
		throw InvalidInputException("Arrow Parquet input slice exceeds its record batch");
	}
	if (cardinality == 0) {
		return;
	}
	auto slice = local.record_batch->Slice(NumericCast<int64_t>(offset), NumericCast<int64_t>(cardinality));
	auto status = impl->writer->WriteRecordBatch(*slice);
	if (!status.ok()) {
		ThrowArrowWriterError(status, "write the Arrow record batch");
	}

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
	impl->written_statistics = statistics;
}

static void GatherWrittenStatistics(const ::parquet::FileMetaData &metadata, idx_t file_size,
                                    CopyFunctionFileStatistics &result) {
	result.row_count = NumericCast<idx_t>(metadata.num_rows());
	result.file_size_bytes = file_size;
	result.footer_size_bytes = Value::UBIGINT(metadata.size());
	result.column_statistics.clear();

	for (int column_idx = 0; column_idx < metadata.num_columns(); column_idx++) {
		idx_t column_size_bytes = 0;
		idx_t num_values = 0;
		idx_t null_count = 0;
		bool all_null_counts_set = true;
		bool all_min_max_set = true;
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
			if (!statistics->HasMinMax() || !MergeStatistics(merged_statistics, statistics)) {
				all_min_max_set = false;
			}
		}

		case_insensitive_map_t<Value> column_statistics;
		column_statistics["column_size_bytes"] = Value::UBIGINT(column_size_bytes);
		column_statistics["num_values"] = Value::UBIGINT(num_values);
		if (all_null_counts_set) {
			column_statistics["null_count"] = Value::UBIGINT(null_count);
		}
		if (all_min_max_set && merged_statistics) {
			std::shared_ptr<::arrow::Scalar> minimum;
			std::shared_ptr<::arrow::Scalar> maximum;
			auto status = ::parquet::arrow::StatisticsAsScalars(*merged_statistics, &minimum, &maximum);
			if (!status.ok()) {
				ThrowArrowWriterError(status, "convert Parquet column statistics");
			}
			if (minimum && minimum->is_valid) {
				column_statistics["min"] = StatisticsScalarToString(*minimum);
			}
			if (maximum && maximum->is_valid) {
				column_statistics["max"] = StatisticsScalarToString(*maximum);
			}
		}

		auto descriptor = metadata.schema()->Column(column_idx);
		result.column_statistics.emplace(ColumnStatisticsName(*descriptor), std::move(column_statistics));
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
		GatherWrittenStatistics(*metadata, impl->output->Size(), *impl->written_statistics);
	}
}

idx_t ArrowParquetWriter::FileSize() const {
	return impl->output->Size();
}

idx_t ArrowParquetWriter::NumberOfRowGroups() const {
	return impl->completed_row_groups;
}

} // namespace duckdb
