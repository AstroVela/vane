#include "duckdb/function/table/read_csv.hpp"
#include "duckdb/function/table/read_duckdb.hpp"

#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/multi_file/union_by_name.hpp"
#include "duckdb/execution/operator/csv_scanner/global_csv_state.hpp"
#include "duckdb/execution/operator/csv_scanner/csv_error.hpp"
#include "duckdb/execution/operator/csv_scanner/sniffer/csv_sniffer.hpp"
#include "duckdb/execution/operator/persistent/csv_rejects_table.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_data.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/execution/operator/csv_scanner/csv_file_scanner.hpp"
#include "duckdb/execution/operator/csv_scanner/base_scanner.hpp"

#include "duckdb/execution/operator/csv_scanner/string_value_scanner.hpp"

#include <limits>
#include "duckdb/execution/operator/csv_scanner/csv_schema.hpp"
#include "duckdb/common/multi_file/multi_file_function.hpp"
#include "duckdb/execution/operator/csv_scanner/csv_multi_file_info.hpp"

#include <algorithm>

namespace duckdb {

SerializedCSVReaderOptions::SerializedCSVReaderOptions(CSVReaderOptions options_p, MultiFileOptions file_options_p)
    : options(std::move(options_p)), file_options(std::move(file_options_p)) {
}

SerializedCSVReaderOptions::SerializedCSVReaderOptions(CSVOption<char> single_byte_delimiter,
                                                       const CSVOption<string> &multi_byte_delimiter)
    : options(single_byte_delimiter, multi_byte_delimiter) {
}

unique_ptr<CSVFileHandle> ReadCSV::OpenCSV(const OpenFileInfo &file, const CSVReaderOptions &options,
                                           ClientContext &context) {
	return CSVFileHandle::OpenFile(context, file, options);
}

ReadCSVData::ReadCSVData() {
}

unique_ptr<FunctionData> ReadCSVData::Copy() const {
	auto result = make_uniq<ReadCSVData>();
	result->options = options;
	result->filename_col_idx = filename_col_idx;
	result->hive_partition_col_idx = hive_partition_col_idx;
	result->manually_set = manually_set;
	// The bind-time buffer manager owns mutable file position and cached buffers. A copied plan must open its own
	// reader, especially when a distributed scan task narrows the same file to a byte range.
	result->column_info = column_info;
	result->csv_schema = csv_schema;
	return std::move(result);
}

void ReadCSVData::FinalizeRead(ClientContext &context) {
	BaseCSVData::Finalize();
}

void ReadCSVTableFunction::ReadCSVAddNamedParameters(TableFunction &table_function) {
	table_function.named_parameters["sep"] = LogicalType::VARCHAR;
	table_function.named_parameters["delim"] = LogicalType::VARCHAR;
	table_function.named_parameters["quote"] = LogicalType::VARCHAR;
	table_function.named_parameters["new_line"] = LogicalType::VARCHAR;
	table_function.named_parameters["escape"] = LogicalType::VARCHAR;
	table_function.named_parameters["nullstr"] = LogicalType::ANY;
	table_function.named_parameters["columns"] = LogicalType::ANY;
	table_function.named_parameters["auto_type_candidates"] = LogicalType::ANY;
	table_function.named_parameters["header"] = LogicalType::BOOLEAN;
	table_function.named_parameters["auto_detect"] = LogicalType::BOOLEAN;
	table_function.named_parameters["sample_size"] = LogicalType::BIGINT;
	table_function.named_parameters["all_varchar"] = LogicalType::BOOLEAN;
	table_function.named_parameters["dateformat"] = LogicalType::VARCHAR;
	table_function.named_parameters["timestampformat"] = LogicalType::VARCHAR;
	table_function.named_parameters["normalize_names"] = LogicalType::BOOLEAN;
	table_function.named_parameters["compression"] = LogicalType::VARCHAR;
	table_function.named_parameters["skip"] = LogicalType::BIGINT;
	table_function.named_parameters["max_line_size"] = LogicalType::VARCHAR;
	table_function.named_parameters["maximum_line_size"] = LogicalType::VARCHAR;
	table_function.named_parameters["ignore_errors"] = LogicalType::BOOLEAN;
	table_function.named_parameters["store_rejects"] = LogicalType::BOOLEAN;
	table_function.named_parameters["rejects_table"] = LogicalType::VARCHAR;
	table_function.named_parameters["rejects_scan"] = LogicalType::VARCHAR;
	table_function.named_parameters["rejects_limit"] = LogicalType::BIGINT;
	table_function.named_parameters["force_not_null"] = LogicalType::LIST(LogicalType::VARCHAR);
	table_function.named_parameters["buffer_size"] = LogicalType::UBIGINT;
	table_function.named_parameters["decimal_separator"] = LogicalType::VARCHAR;
	table_function.named_parameters["parallel"] = LogicalType::BOOLEAN;
	table_function.named_parameters["null_padding"] = LogicalType::BOOLEAN;
	table_function.named_parameters["allow_quoted_nulls"] = LogicalType::BOOLEAN;
	table_function.named_parameters["column_types"] = LogicalType::ANY;
	table_function.named_parameters["dtypes"] = LogicalType::ANY;
	table_function.named_parameters["types"] = LogicalType::ANY;
	table_function.named_parameters["names"] = LogicalType::LIST(LogicalType::VARCHAR);
	table_function.named_parameters["column_names"] = LogicalType::LIST(LogicalType::VARCHAR);
	table_function.named_parameters["comment"] = LogicalType::VARCHAR;
	table_function.named_parameters["encoding"] = LogicalType::VARCHAR;
	table_function.named_parameters["strict_mode"] = LogicalType::BOOLEAN;
	table_function.named_parameters["thousands"] = LogicalType::VARCHAR;
	table_function.named_parameters["files_to_sniff"] = LogicalType::BIGINT;

	MultiFileReader::AddParameters(table_function);
}

static void CSVReaderSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data_p,
                               const TableFunction &function) {
	if (!bind_data_p) {
		throw InternalException("Cannot serialize read_csv without bind data");
	}
	auto &bind_data = bind_data_p->Cast<MultiFileBindData>();
	auto &csv_data = bind_data.bind_data->Cast<ReadCSVData>();

	SerializedReadCSVData serialized_data;
	for (auto &file : bind_data.file_list->GetAllFiles()) {
		serialized_data.files.emplace_back(file.path);
	}
	serialized_data.return_types = serialized_data.csv_types = bind_data.types;
	serialized_data.return_names = serialized_data.csv_names = bind_data.names;
	serialized_data.filename_col_idx = csv_data.filename_col_idx;
	serialized_data.hive_partition_col_idx = csv_data.hive_partition_col_idx;
	serialized_data.options = SerializedCSVReaderOptions(csv_data.options, bind_data.file_options);
	serialized_data.reader_bind = bind_data.reader_bind;
	serialized_data.table_columns = bind_data.table_columns;
	serialized_data.manually_set = csv_data.manually_set;

	if (!csv_data.csv_schema.Empty()) {
		serialized_data.csv_schema_names = csv_data.csv_schema.GetNames();
		serialized_data.csv_schema_types = csv_data.csv_schema.GetTypes();
		serialized_data.csv_schema_path = csv_data.csv_schema.GetPath();
		serialized_data.csv_schema_rows_read = csv_data.csv_schema.GetRowsRead();
	}

	if (bind_data.file_options.union_by_name) {
		if (csv_data.column_info.empty()) {
			throw InternalException("Cannot serialize CSV union-by-name scan without per-file reader information");
		}
		for (auto &file : serialized_data.files) {
			auto entry = std::find_if(csv_data.column_info.begin(), csv_data.column_info.end(),
			                          [&](const ColumnInfo &info) { return info.file_path == file; });
			if (entry == csv_data.column_info.end()) {
				throw InternalException("Missing serialized CSV union reader information for file \"%s\"", file);
			}
			if (entry->options.options.encoding.empty()) {
				throw InternalException("CSV union reader information has no encoding for file \"%s\"", file);
			}
			serialized_data.column_info.push_back(*entry);
		}
	}
	serializer.WriteProperty(100, "csv_data", serialized_data);
}

static unique_ptr<FunctionData> CSVReaderDeserialize(Deserializer &deserializer, TableFunction &function) {
	auto &context = deserializer.Get<ClientContext &>();
	auto serialized_data = deserializer.ReadProperty<SerializedReadCSVData>(100, "csv_data");
	if (serialized_data.files.empty()) {
		throw IOException("%s needs at least one file to read", function.name);
	}

	vector<OpenFileInfo> open_files;
	open_files.reserve(serialized_data.files.size());
	for (auto &path : serialized_data.files) {
		open_files.emplace_back(path);
	}
	auto file_list = make_shared_ptr<SimpleMultiFileList>(std::move(open_files));
	auto multi_file_reader = MultiFileReader::Create(function);
	auto interface = make_uniq<CSVMultiFileInfo>();
	interface->InitializeInterface(context, *multi_file_reader, *file_list);

	auto csv_options = make_uniq<CSVFileReaderOptions>(std::move(serialized_data.options.options));
	auto result = make_uniq<MultiFileBindData>();
	result->multi_file_reader = std::move(multi_file_reader);
	result->file_list = std::move(file_list);
	result->file_options = std::move(serialized_data.options.file_options);
	result->interface = std::move(interface);
	result->bind_data = result->interface->InitializeBindData(*result, std::move(csv_options));
	result->types = std::move(serialized_data.return_types);
	result->names = std::move(serialized_data.return_names);
	result->reader_bind = std::move(serialized_data.reader_bind);
	result->table_columns = std::move(serialized_data.table_columns);
	if (result->file_options.union_by_name) {
		if (serialized_data.column_info.size() != serialized_data.files.size()) {
			throw SerializationException("CSV union-by-name scan has %llu files but %llu per-file reader entries",
			                             serialized_data.files.size(), serialized_data.column_info.size());
		}
		for (idx_t file_idx = 0; file_idx < serialized_data.files.size(); ++file_idx) {
			if (serialized_data.column_info[file_idx].file_path != serialized_data.files[file_idx]) {
				throw SerializationException("CSV union-by-name reader entry %llu targets \"%s\", expected \"%s\"",
				                             file_idx, serialized_data.column_info[file_idx].file_path,
				                             serialized_data.files[file_idx]);
			}
		}
	}

	auto &csv_data = result->bind_data->Cast<ReadCSVData>();
	csv_data.filename_col_idx = serialized_data.filename_col_idx;
	csv_data.hive_partition_col_idx = serialized_data.hive_partition_col_idx;
	csv_data.manually_set = std::move(serialized_data.manually_set);
	if (!serialized_data.csv_schema_names.empty()) {
		csv_data.csv_schema = CSVSchema(serialized_data.csv_schema_names, serialized_data.csv_schema_types,
		                                serialized_data.csv_schema_path, serialized_data.csv_schema_rows_read);
	}
	for (auto &info : serialized_data.column_info) {
		if (info.options.options.encoding.empty()) {
			throw InternalException("Deserialized CSV union reader information has no encoding for file \"%s\"",
			                        info.file_path);
		}
		auto union_data = make_shared_ptr<CSVUnionData>(OpenFileInfo(info.file_path));
		union_data->names = info.names;
		union_data->types = info.types;
		union_data->options = info.options.options;
		result->union_readers.push_back(std::move(union_data));
	}
	csv_data.column_info = std::move(serialized_data.column_info);

	result->interface->FinalizeBindData(*result);
	result->columns = MultiFileColumnDefinition::ColumnsFromNamesAndTypes(result->names, result->types);
	virtual_column_map_t virtual_columns;
	MultiFileReader::GetVirtualColumns(context, result->reader_bind, virtual_columns);
	result->interface->GetVirtualColumns(context, *result, virtual_columns);
	result->virtual_columns = std::move(virtual_columns);
	return std::move(result);
}

TableFunction ReadCSVTableFunction::GetFunction() {
	MultiFileFunction<CSVMultiFileInfo> read_csv("read_csv");
	read_csv.serialize = CSVReaderSerialize;
	read_csv.deserialize = CSVReaderDeserialize;
	read_csv.type_pushdown = MultiFileFunction<CSVMultiFileInfo>::PushdownType;
	ReadCSVAddNamedParameters(read_csv);
	return static_cast<TableFunction>(read_csv);
}

TableFunction ReadCSVTableFunction::GetAutoFunction() {
	auto read_csv_auto = ReadCSVTableFunction::GetFunction();
	read_csv_auto.name = "read_csv_auto";
	return read_csv_auto;
}

void ReadCSVTableFunction::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(MultiFileReader::CreateFunctionSet(ReadCSVTableFunction::GetFunction()));
	set.AddFunction(MultiFileReader::CreateFunctionSet(ReadCSVTableFunction::GetAutoFunction()));
}

unique_ptr<TableRef> ReadCSVReplacement(ClientContext &context, ReplacementScanInput &input,
                                        optional_ptr<ReplacementScanData> data) {
	auto table_name = ReplacementScan::GetFullPath(input);
	auto lower_name = StringUtil::Lower(table_name);
	// remove any compression
	if (StringUtil::EndsWith(lower_name, CompressionExtensionFromType(FileCompressionType::GZIP))) {
		lower_name = lower_name.substr(0, lower_name.size() - 3);
	} else if (StringUtil::EndsWith(lower_name, CompressionExtensionFromType(FileCompressionType::ZSTD))) {
		if (!Catalog::TryAutoLoad(context, "parquet")) {
			throw MissingExtensionException("parquet extension is required for reading zst compressed file");
		}
		lower_name = lower_name.substr(0, lower_name.size() - 4);
	}
	if (!StringUtil::EndsWith(lower_name, ".csv") && !StringUtil::Contains(lower_name, ".csv?") &&
	    !StringUtil::EndsWith(lower_name, ".tsv") && !StringUtil::Contains(lower_name, ".tsv?")) {
		return nullptr;
	}
	auto table_function = make_uniq<TableFunctionRef>();
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(make_uniq<ConstantExpression>(Value(table_name)));
	table_function->function = make_uniq<FunctionExpression>("read_csv_auto", std::move(children));

	if (!FileSystem::HasGlob(table_name)) {
		auto &fs = FileSystem::GetFileSystem(context);
		table_function->alias = fs.ExtractBaseName(table_name);
	}

	return std::move(table_function);
}

void BuiltinFunctions::RegisterReadFunctions() {
	CSVCopyFunction::RegisterFunction(*this);
	ReadCSVTableFunction::RegisterFunction(*this);
	AddFunction(MultiFileReader::CreateFunctionSet(ReadDuckDBTableFunction::GetFunction()));
	auto &config = DBConfig::GetConfig(*transaction.db);
	config.replacement_scans.emplace_back(ReadCSVReplacement);
	config.replacement_scans.emplace_back(ReadDuckDBTableFunction::ReplacementScan);
}

} // namespace duckdb
