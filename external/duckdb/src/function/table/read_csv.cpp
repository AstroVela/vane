#include "duckdb/function/table/read_csv.hpp"
#include "duckdb/function/table/read_duckdb.hpp"

#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/allocator.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/set.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/multi_file/union_by_name.hpp"
#include "duckdb/execution/operator/csv_scanner/global_csv_state.hpp"
#include "duckdb/execution/operator/csv_scanner/csv_scan_range.hpp"
#include "duckdb/execution/operator/csv_scanner/scanner_boundary.hpp"
#include "duckdb/execution/operator/csv_scanner/csv_error.hpp"
#include "duckdb/execution/operator/csv_scanner/sniffer/csv_sniffer.hpp"
#include "duckdb/execution/operator/persistent/csv_rejects_table.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/function/distributed_table_function.hpp"
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
#include <algorithm>
#include "duckdb/execution/operator/csv_scanner/csv_schema.hpp"
#include "duckdb/common/multi_file/multi_file_function.hpp"
#include "duckdb/execution/operator/csv_scanner/csv_multi_file_info.hpp"

namespace duckdb {

SerializedCSVReaderOptions::SerializedCSVReaderOptions(CSVReaderOptions options_p, MultiFileOptions file_options_p)
    : options(std::move(options_p)), file_options(std::move(file_options_p)) {
}

SerializedCSVReaderOptions::SerializedCSVReaderOptions(CSVOption<char> single_byte_delimiter,
                                                       const CSVOption<string> &multi_byte_delimiter)
    : options(single_byte_delimiter, multi_byte_delimiter) {
}

CSVFileSnapshot::CSVFileSnapshot(idx_t ordinal_p, const OpenFileInfo &file) : ordinal(ordinal_p), path(file.path) {
	if (file.extended_info) {
		for (const auto &entry : file.extended_info->options) {
			options.emplace(entry.first, entry.second);
		}
	}
}

OpenFileInfo CSVFileSnapshot::ToOpenFileInfo() const {
	OpenFileInfo result(path);
	if (!options.empty()) {
		result.extended_info = make_shared_ptr<ExtendedOpenFileInfo>();
		for (const auto &entry : options) {
			result.extended_info->options.emplace(entry.first, entry.second);
		}
	}
	return result;
}

unique_ptr<CSVFileHandle> ReadCSV::OpenCSV(const OpenFileInfo &file, const CSVReaderOptions &options,
                                           ClientContext &context) {
	return CSVFileHandle::OpenFile(context, file, options);
}

ReadCSVData::ReadCSVData() {
}

unique_ptr<FunctionData> ReadCSVData::Copy() const {
	auto result = make_uniq<ReadCSVData>();
	result->column_ids = column_ids;
	result->options = options;
	result->filename_col_idx = filename_col_idx;
	result->hive_partition_col_idx = hive_partition_col_idx;
	result->manually_set = manually_set;
	// buffer_manager owns mutable buffers and a file position. Runtime readers must be created from the execution
	// context that executes this copy.
	result->column_info = column_info;
	result->csv_schema = csv_schema;
	result->distributed_worker = distributed_worker;
	result->distributed_splits_applied = distributed_splits_applied;
	result->distributed_allowed_files = distributed_allowed_files;
	result->distributed_source_multiple_files = distributed_source_multiple_files;
	result->distributed_authorization_restricted = distributed_authorization_restricted;
	result->distributed_authorized_split_ids = distributed_authorized_split_ids;
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

static bool CSVFileSnapshotMatches(const CSVFileSnapshot &snapshot, const OpenFileInfo &file) {
	CSVFileSnapshot candidate(snapshot.ordinal, CSVScanRange::Strip(file));
	return snapshot.path == candidate.path && snapshot.options == candidate.options;
}

static void ValidateCSVFileSnapshot(const CSVFileSnapshot &snapshot, const string &description) {
	if (snapshot.path.empty()) {
		throw SerializationException("%s has an empty path", description);
	}
	if (snapshot.options.find(CSVScanRange::START_OPTION) != snapshot.options.end() ||
	    snapshot.options.find(CSVScanRange::END_OPTION) != snapshot.options.end() ||
	    snapshot.options.find(CSVScanRange::ORDINAL_OPTION) != snapshot.options.end()) {
		throw SerializationException("%s contains reserved CSV byte-range options", description);
	}
}

static string DistributedCSVSplitId(idx_t ordinal, bool has_range, idx_t range_start, idx_t range_end) {
	string result = "file/" + std::to_string(ordinal);
	if (has_range) {
		result += "/bytes/" + std::to_string(range_start) + "/" + std::to_string(range_end);
	}
	return result;
}

static void ValidateCSVDistributedState(const ReadCSVData &csv_data, const vector<OpenFileInfo> &active_files) {
	set<idx_t> ordinals;
	for (const auto &snapshot : csv_data.distributed_allowed_files) {
		ValidateCSVFileSnapshot(snapshot, "Distributed CSV worker file");
		if (!ordinals.insert(snapshot.ordinal).second) {
			throw SerializationException("Distributed CSV worker file ordinal %llu appears more than once",
			                             snapshot.ordinal);
		}
	}
	if (!csv_data.distributed_worker) {
		if (csv_data.distributed_splits_applied || !csv_data.distributed_allowed_files.empty() ||
		    csv_data.distributed_source_multiple_files || csv_data.distributed_authorization_restricted ||
		    !csv_data.distributed_authorized_split_ids.empty()) {
			throw SerializationException("Local CSV bind contains distributed worker state");
		}
		for (const auto &file : active_files) {
			ValidateCSVFileSnapshot(CSVFileSnapshot(0, file), "Local CSV active file");
		}
		return;
	}
	if (!csv_data.distributed_splits_applied && !active_files.empty()) {
		throw SerializationException("Detached distributed CSV worker bind contains active files");
	}
	set<idx_t> whole_file_ordinals;
	map<idx_t, vector<pair<idx_t, idx_t>>> ranges_by_ordinal;
	vector<string> active_split_ids;
	active_split_ids.reserve(active_files.size());
	for (const auto &file : active_files) {
		idx_t active_ordinal;
		if (!CSVScanRange::TryGetOrdinal(file, active_ordinal)) {
			throw SerializationException("Distributed CSV worker active file \"%s\" has no stable ordinal", file.path);
		}
		optional_ptr<const CSVFileSnapshot> allowed;
		for (const auto &candidate : csv_data.distributed_allowed_files) {
			if (candidate.ordinal == active_ordinal && CSVFileSnapshotMatches(candidate, file)) {
				allowed = candidate;
				break;
			}
		}
		if (!allowed) {
			throw SerializationException("Distributed CSV worker active file \"%s\" is not coordinator-authorized",
			                             file.path);
		}
		CSVScanRange range {0, 0};
		const bool has_range = CSVScanRange::TryGet(file, range);
		if (has_range) {
			if (whole_file_ordinals.find(active_ordinal) != whole_file_ordinals.end()) {
				throw SerializationException(
				    "Distributed CSV worker mixes whole-file and byte-range work for file %llu", active_ordinal);
			}
			auto &ranges = ranges_by_ordinal[active_ordinal];
			for (const auto &existing : ranges) {
				if (range.start < existing.second && existing.first < range.end) {
					throw SerializationException(
					    "Distributed CSV worker contains overlapping byte ranges for file %llu", active_ordinal);
				}
			}
			ranges.emplace_back(range.start, range.end);
		} else if (!whole_file_ordinals.insert(active_ordinal).second ||
		           ranges_by_ordinal.find(active_ordinal) != ranges_by_ordinal.end()) {
			throw SerializationException("Distributed CSV worker repeats file ordinal %llu", active_ordinal);
		}
		active_split_ids.push_back(DistributedCSVSplitId(active_ordinal, has_range, range.start, range.end));
	}
	std::sort(active_split_ids.begin(), active_split_ids.end());
	if (csv_data.distributed_authorization_restricted) {
		if (!std::is_sorted(csv_data.distributed_authorized_split_ids.begin(),
		                    csv_data.distributed_authorized_split_ids.end()) ||
		    std::adjacent_find(csv_data.distributed_authorized_split_ids.begin(),
		                       csv_data.distributed_authorized_split_ids.end()) !=
		        csv_data.distributed_authorized_split_ids.end()) {
			throw SerializationException("Distributed CSV authorized split ids are not canonical and unique");
		}
	} else if (!csv_data.distributed_authorized_split_ids.empty()) {
		throw SerializationException("Distributed CSV bind has split ids without an installed authorization");
	}
	if (csv_data.distributed_splits_applied) {
		if (!csv_data.distributed_authorization_restricted) {
			throw SerializationException("Applied distributed CSV worker bind has no split authorization");
		}
		if (active_split_ids != csv_data.distributed_authorized_split_ids) {
			throw SerializationException("Distributed CSV worker active files do not match its split authorization");
		}
	}
}

static void ValidateCSVColumnInfo(const MultiFileBindData &bind_data, const ReadCSVData &csv_data,
                                  const vector<OpenFileInfo> &active_files) {
	if (bind_data.file_options.mapping != MultiFileColumnMappingMode::BY_NAME ||
	    !bind_data.file_options.custom_options.empty() ||
	    bind_data.reader_bind.mapping != MultiFileColumnMappingMode::BY_NAME || !bind_data.reader_bind.schema.empty()) {
		throw SerializationException(
		    "CSV plan serialization only supports the built-in name-based multi-file reader state");
	}
	if (!bind_data.file_options.union_by_name) {
		if (!csv_data.column_info.empty()) {
			throw SerializationException("CSV bind contains per-file union reader state without union_by_name");
		}
		return;
	}
	set<idx_t> column_info_ordinals;
	for (const auto &info : csv_data.column_info) {
		ValidateCSVFileSnapshot(info.file, "CSV union reader file");
		if (!column_info_ordinals.insert(info.file.ordinal).second) {
			throw SerializationException("CSV union reader file ordinal %llu appears more than once",
			                             info.file.ordinal);
		}
		if (info.names.size() != info.types.size()) {
			throw SerializationException("CSV union reader for file \"%s\" has %llu names but %llu types",
			                             info.file.path, info.names.size(), info.types.size());
		}
		if (info.options.options.encoding.empty()) {
			throw SerializationException("CSV union reader information has no encoding for file \"%s\"",
			                             info.file.path);
		}
	}
	const auto validate_file = [&](const OpenFileInfo &file, optional_idx expected_ordinal) {
		for (const auto &info : csv_data.column_info) {
			if (CSVFileSnapshotMatches(info.file, file) &&
			    (!expected_ordinal.IsValid() || info.file.ordinal == expected_ordinal.GetIndex())) {
				return;
			}
		}
		throw SerializationException("Missing CSV union reader information for file \"%s\"", file.path);
	};
	for (const auto &file : active_files) {
		idx_t ordinal;
		optional_idx expected_ordinal;
		if (CSVScanRange::TryGetOrdinal(file, ordinal)) {
			expected_ordinal = optional_idx(ordinal);
		}
		validate_file(CSVScanRange::Strip(file), expected_ordinal);
	}
	for (const auto &snapshot : csv_data.distributed_allowed_files) {
		validate_file(snapshot.ToOpenFileInfo(), optional_idx(snapshot.ordinal));
	}
}

static void CSVReaderSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data_p,
                               const TableFunction &function) {
	if (!bind_data_p) {
		throw InternalException("Cannot serialize %s without bind data", function.name);
	}
	auto &bind_data = bind_data_p->Cast<MultiFileBindData>();
	if (!bind_data.bind_data || !bind_data.file_list || !bind_data.multi_file_reader || !bind_data.interface) {
		throw InternalException("Cannot serialize incomplete %s bind data", function.name);
	}
	auto &csv_data = bind_data.bind_data->Cast<ReadCSVData>();
	auto active_files = bind_data.file_list->GetAllFiles();
	ValidateCSVDistributedState(csv_data, active_files);
	ValidateCSVColumnInfo(bind_data, csv_data, active_files);

	SerializedReadCSVData serialized_data;
	serialized_data.files.reserve(active_files.size());
	for (idx_t file_idx = 0; file_idx < active_files.size(); file_idx++) {
		idx_t ordinal = file_idx;
		if (csv_data.distributed_worker) {
			if (!CSVScanRange::TryGetOrdinal(active_files[file_idx], ordinal)) {
				throw InternalException("Distributed CSV active file \"%s\" has no file ordinal",
				                        active_files[file_idx].path);
			}
			optional_ptr<const CSVFileSnapshot> allowed;
			for (const auto &candidate : csv_data.distributed_allowed_files) {
				if (candidate.ordinal == ordinal && CSVFileSnapshotMatches(candidate, active_files[file_idx])) {
					allowed = candidate;
					break;
				}
			}
			if (!allowed) {
				throw InternalException("Distributed CSV active file \"%s\" has no authorization",
				                        active_files[file_idx].path);
			}
			ordinal = allowed->ordinal;
		}
		serialized_data.files.emplace_back(ordinal, active_files[file_idx]);
	}
	serialized_data.return_types = serialized_data.csv_types = bind_data.types;
	serialized_data.return_names = serialized_data.csv_names = bind_data.names;
	serialized_data.filename_col_idx = csv_data.filename_col_idx;
	serialized_data.hive_partition_col_idx = csv_data.hive_partition_col_idx;
	serialized_data.options = SerializedCSVReaderOptions(csv_data.options, bind_data.file_options);
	serialized_data.reader_bind = bind_data.reader_bind;
	serialized_data.column_info = csv_data.column_info;
	serialized_data.table_columns = bind_data.table_columns;
	serialized_data.manually_set = csv_data.manually_set;
	serialized_data.bind_column_ids = bind_data.column_ids;
	serialized_data.reader_column_ids = csv_data.column_ids;
	serialized_data.has_csv_schema = !csv_data.csv_schema.Empty();
	serialized_data.csv_schema_empty_file = csv_data.csv_schema.IsEmptyFile();
	if (serialized_data.has_csv_schema) {
		serialized_data.csv_schema_names = csv_data.csv_schema.GetNames();
		serialized_data.csv_schema_types = csv_data.csv_schema.GetTypes();
		serialized_data.csv_schema_path = csv_data.csv_schema.GetPath();
		serialized_data.csv_schema_rows_read = csv_data.csv_schema.GetRowsRead();
	}
	serialized_data.distributed_worker = csv_data.distributed_worker;
	serialized_data.distributed_splits_applied = csv_data.distributed_splits_applied;
	serialized_data.distributed_allowed_files = csv_data.distributed_allowed_files;
	serialized_data.distributed_source_multiple_files = csv_data.distributed_source_multiple_files;
	serialized_data.distributed_authorization_restricted = csv_data.distributed_authorization_restricted;
	serialized_data.distributed_authorized_split_ids = csv_data.distributed_authorized_split_ids;
	serializer.WriteProperty(100, "csv_data", serialized_data);
}

static unique_ptr<FunctionData> CSVReaderDeserialize(Deserializer &deserializer, TableFunction &function) {
	auto &context = deserializer.Get<ClientContext &>();
	auto serialized_data = deserializer.ReadProperty<SerializedReadCSVData>(100, "csv_data");
	if (serialized_data.return_types.empty()) {
		throw SerializationException("CSV bind has an empty output schema");
	}
	if (serialized_data.return_types.size() != serialized_data.return_names.size()) {
		throw SerializationException("CSV bind has %llu return types but %llu return names",
		                             serialized_data.return_types.size(), serialized_data.return_names.size());
	}
	if (serialized_data.csv_types.size() != serialized_data.csv_names.size() ||
	    serialized_data.csv_types != serialized_data.return_types ||
	    serialized_data.csv_names != serialized_data.return_names) {
		throw SerializationException("CSV bind contains inconsistent source and return schemas");
	}
	if (serialized_data.has_csv_schema) {
		if (serialized_data.csv_schema_names.empty() ||
		    serialized_data.csv_schema_names.size() != serialized_data.csv_schema_types.size()) {
			throw SerializationException("CSV schema has %llu names but %llu types",
			                             serialized_data.csv_schema_names.size(),
			                             serialized_data.csv_schema_types.size());
		}
	} else if (!serialized_data.csv_schema_names.empty() || !serialized_data.csv_schema_types.empty() ||
	           !serialized_data.csv_schema_path.empty() || serialized_data.csv_schema_rows_read != 0) {
		throw SerializationException("CSV bind contains schema state without a bound CSV schema");
	}

	vector<OpenFileInfo> active_files;
	active_files.reserve(serialized_data.files.size());
	for (idx_t file_idx = 0; file_idx < serialized_data.files.size(); file_idx++) {
		const auto &file = serialized_data.files[file_idx];
		if (file.path.empty()) {
			throw SerializationException("CSV bind contains an empty active file path");
		}
		auto active_file = file.ToOpenFileInfo();
		if (serialized_data.distributed_worker) {
			idx_t active_ordinal;
			if (!CSVScanRange::TryGetOrdinal(active_file, active_ordinal) || active_ordinal != file.ordinal) {
				throw SerializationException("Distributed CSV active file \"%s\" has inconsistent file ordinal",
				                             file.path);
			}
		} else {
			if (file.ordinal != file_idx) {
				throw SerializationException("Local CSV active file \"%s\" has non-canonical file ordinal %llu",
				                             file.path, file.ordinal);
			}
			ValidateCSVFileSnapshot(file, "Local CSV active file");
		}
		active_files.push_back(std::move(active_file));
	}
	auto file_list = make_shared_ptr<SimpleMultiFileList>(active_files);
	auto multi_file_reader = MultiFileReader::Create(function);
	auto interface = make_uniq<CSVMultiFileInfo>();
	interface->InitializeInterface(context, *multi_file_reader, *file_list);

	auto result = make_uniq<MultiFileBindData>();
	result->multi_file_reader = std::move(multi_file_reader);
	result->file_list = std::move(file_list);
	result->file_options = std::move(serialized_data.options.file_options);
	result->interface = std::move(interface);
	auto csv_data = make_uniq<ReadCSVData>();
	csv_data->options = std::move(serialized_data.options.options);
	// The active list of an already-applied worker can contain several ranges of
	// one original file. Restore the coordinator's bound option verbatim instead
	// of deriving multi_file_reader again from that execution-only list.
	csv_data->options.Verify(result->file_options);
	result->bind_data = std::move(csv_data);
	result->types = std::move(serialized_data.return_types);
	result->names = std::move(serialized_data.return_names);
	result->reader_bind = std::move(serialized_data.reader_bind);
	result->table_columns = std::move(serialized_data.table_columns);
	result->column_ids = std::move(serialized_data.bind_column_ids);

	auto &restored_csv_data = result->bind_data->Cast<ReadCSVData>();
	restored_csv_data.filename_col_idx = serialized_data.filename_col_idx;
	restored_csv_data.hive_partition_col_idx = serialized_data.hive_partition_col_idx;
	restored_csv_data.column_ids = std::move(serialized_data.reader_column_ids);
	restored_csv_data.manually_set = std::move(serialized_data.manually_set);
	if (serialized_data.has_csv_schema) {
		restored_csv_data.csv_schema = CSVSchema(serialized_data.csv_schema_names, serialized_data.csv_schema_types,
		                                         serialized_data.csv_schema_path, serialized_data.csv_schema_rows_read,
		                                         serialized_data.csv_schema_empty_file);
	} else {
		restored_csv_data.csv_schema = CSVSchema(serialized_data.csv_schema_empty_file);
	}
	restored_csv_data.column_info = std::move(serialized_data.column_info);
	restored_csv_data.distributed_worker = serialized_data.distributed_worker;
	restored_csv_data.distributed_splits_applied = serialized_data.distributed_splits_applied;
	restored_csv_data.distributed_allowed_files = std::move(serialized_data.distributed_allowed_files);
	restored_csv_data.distributed_source_multiple_files = serialized_data.distributed_source_multiple_files;
	restored_csv_data.distributed_authorization_restricted = serialized_data.distributed_authorization_restricted;
	restored_csv_data.distributed_authorized_split_ids = std::move(serialized_data.distributed_authorized_split_ids);
	ValidateCSVDistributedState(restored_csv_data, active_files);
	ValidateCSVColumnInfo(*result, restored_csv_data, active_files);

	result->interface->FinalizeBindData(*result);
	result->columns = MultiFileColumnDefinition::ColumnsFromNamesAndTypes(result->names, result->types);
	virtual_column_map_t virtual_columns;
	MultiFileReader::GetVirtualColumns(context, result->reader_bind, virtual_columns);
	result->interface->GetVirtualColumns(context, *result, virtual_columns);
	result->virtual_columns = std::move(virtual_columns);
	return std::move(result);
}

//===--------------------------------------------------------------------===//
// Explicit distributed CSV scan protocol
//===--------------------------------------------------------------------===//

static constexpr idx_t DISTRIBUTED_CSV_PROTOCOL_VERSION = 1;
static constexpr idx_t DISTRIBUTED_CSV_PAYLOAD_VERSION = 1;
static constexpr const char *DISTRIBUTED_CSV_SPLIT_CODEC = "vane.csv-file-range";
static constexpr idx_t DISTRIBUTED_CSV_SPLIT_CODEC_VERSION = 1;

struct DistributedCSVSplitPayload {
	CSVFileSnapshot file;
	bool has_range = false;
	idx_t range_start = 0;
	idx_t range_end = 0;
};

static string EncodeDistributedCSVSplitPayload(const DistributedCSVSplitPayload &payload) {
	MemoryStream stream(Allocator::DefaultAllocator());
	BinarySerializer serializer(stream);
	serializer.Begin();
	serializer.WriteProperty(1, "payload_version", DISTRIBUTED_CSV_PAYLOAD_VERSION);
	serializer.WriteProperty(2, "file_ordinal", payload.file.ordinal);
	serializer.WriteProperty(3, "path", payload.file.path);
	serializer.WriteProperty(4, "open_options", payload.file.options);
	serializer.WriteProperty(5, "has_range", payload.has_range);
	serializer.WriteProperty(6, "range_start", payload.range_start);
	serializer.WriteProperty(7, "range_end", payload.range_end);
	serializer.End();
	return string(reinterpret_cast<const char *>(stream.GetData()), stream.GetPosition());
}

static DistributedCSVSplitPayload DecodeDistributedCSVSplitPayload(const string &bytes) {
	if (bytes.empty()) {
		throw InvalidInputException("Distributed CSV split payload is empty");
	}
	auto data = reinterpret_cast<data_ptr_t>(const_cast<char *>(bytes.data()));
	MemoryStream stream(data, bytes.size());
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	const auto version = deserializer.ReadProperty<idx_t>(1, "payload_version");
	if (version != DISTRIBUTED_CSV_PAYLOAD_VERSION) {
		throw InvalidInputException("Unsupported distributed CSV payload version %llu", version);
	}
	DistributedCSVSplitPayload result;
	result.file.ordinal = deserializer.ReadProperty<idx_t>(2, "file_ordinal");
	result.file.path = deserializer.ReadProperty<string>(3, "path");
	result.file.options = deserializer.ReadProperty<map<string, Value>>(4, "open_options");
	result.has_range = deserializer.ReadProperty<bool>(5, "has_range");
	result.range_start = deserializer.ReadProperty<idx_t>(6, "range_start");
	result.range_end = deserializer.ReadProperty<idx_t>(7, "range_end");
	deserializer.End();
	if (stream.GetPosition() != bytes.size()) {
		throw InvalidInputException("Distributed CSV split payload has trailing bytes");
	}
	ValidateCSVFileSnapshot(result.file, "Distributed CSV split payload");
	if (result.has_range) {
		if (result.range_start >= result.range_end) {
			throw InvalidInputException("Distributed CSV split has invalid byte range [%llu, %llu)", result.range_start,
			                            result.range_end);
		}
	} else if (result.range_start != 0 || result.range_end != 0) {
		throw InvalidInputException("Distributed CSV whole-file split contains byte-range offsets");
	}
	return result;
}

static string DistributedCSVSplitId(const DistributedCSVSplitPayload &payload) {
	return DistributedCSVSplitId(payload.file.ordinal, payload.has_range, payload.range_start, payload.range_end);
}

static const MultiFileBindData &GetCSVDistributedBind(const TableFunctionDistributedScanInput &input) {
	auto &bind_data = input.bind_data.Cast<MultiFileBindData>();
	if (!bind_data.bind_data || !bind_data.file_list || !bind_data.multi_file_reader || !bind_data.interface) {
		throw InvalidInputException("Distributed CSV scan requires complete multi-file bind data");
	}
	bind_data.bind_data->Cast<ReadCSVData>();
	return bind_data;
}

static CSVFileSnapshot GetCSVPlanningSnapshot(const ReadCSVData &csv_data, const OpenFileInfo &file, idx_t ordinal) {
	if (!csv_data.distributed_worker) {
		CSVFileSnapshot result(ordinal, CSVScanRange::Strip(file));
		ValidateCSVFileSnapshot(result, "Distributed CSV coordinator file");
		return result;
	}
	idx_t active_ordinal;
	if (!CSVScanRange::TryGetOrdinal(file, active_ordinal)) {
		throw InvalidInputException("Distributed CSV worker file \"%s\" has no stable ordinal", file.path);
	}
	for (const auto &snapshot : csv_data.distributed_allowed_files) {
		if (snapshot.ordinal == active_ordinal && CSVFileSnapshotMatches(snapshot, file)) {
			return snapshot;
		}
	}
	throw InvalidInputException("Distributed CSV worker file \"%s\" is not coordinator-authorized", file.path);
}

struct CSVPlanningFileProperties {
	idx_t size;
	bool can_seek;
	bool is_pipe;
	FileCompressionType compression;
};

static CSVPlanningFileProperties InspectCSVPlanningFile(const TableFunctionDistributedScanPlanningInput &input,
                                                        const CSVFileSnapshot &snapshot,
                                                        const CSVReaderOptions &options) {
	auto &fs = input.file_system;
	// Opening the read end of a local FIFO can block until a writer appears. Ask
	// the filesystem first so unsupported pipe sources fail during planning
	// without performing a potentially blocking open.
	if (fs.IsPipe(snapshot.path)) {
		CSVPlanningFileProperties result;
		result.size = NumericLimits<idx_t>::Maximum();
		result.can_seek = false;
		result.is_pipe = true;
		result.compression = options.compression;
		return result;
	}
	auto handle = CSVFileHandle::OpenFileHandle(fs, Allocator::DefaultAllocator(), snapshot.ToOpenFileInfo(),
	                                            options.compression);
	if (!handle) {
		throw IOException("Could not open CSV file \"%s\" while planning distributed splits", snapshot.path);
	}
	CSVPlanningFileProperties result;
	result.can_seek = handle->CanSeek();
	result.is_pipe = handle->IsPipe();
	result.compression = handle->GetFileCompressionType();
	result.size = result.is_pipe ? NumericLimits<idx_t>::Maximum() : handle->GetFileSize();
	return result;
}

static optional_idx CSVCardinalityEstimate(idx_t total_cardinality, idx_t part_index, idx_t part_count) {
	if (total_cardinality == DConstants::INVALID_INDEX || part_count == 0) {
		return optional_idx();
	}
	const auto base = total_cardinality / part_count;
	const auto remainder = total_cardinality % part_count;
	return optional_idx(base + (part_index < remainder ? 1 : 0));
}

static DistributedScanSplit MakeDistributedCSVSplit(const DistributedCSVSplitPayload &payload,
                                                    optional_idx estimated_cardinality, optional_idx estimated_bytes) {
	DistributedScanSplit result;
	result.split_id = DistributedCSVSplitId(payload);
	result.payload = EncodeDistributedCSVSplitPayload(payload);
	result.estimated_cardinality = estimated_cardinality;
	result.estimated_bytes = estimated_bytes;
	result.Validate();
	return result;
}

static void ValidateCSVByteRangePlanning(const CSVFileSnapshot &file, const CSVReaderOptions &options,
                                         const CSVPlanningFileProperties &properties) {
	if (!options.parallel) {
		throw InvalidInputException("Distributed byte-range scanning of CSV file \"%s\" requires parallel=true",
		                            file.path);
	}
	if (options.GetSkipRows() > 0) {
		throw InvalidInputException("Distributed byte-range scanning of CSV file \"%s\" does not support skip_rows",
		                            file.path);
	}
	if (options.dialect_options.rows_until_header > 0) {
		throw InvalidInputException(
		    "Distributed byte-range scanning of CSV file \"%s\" does not support leading rows before the header",
		    file.path);
	}
	if (StringUtil::Lower(options.encoding) != "utf-8") {
		throw InvalidInputException("Distributed byte-range scanning of CSV file \"%s\" requires UTF-8 input",
		                            file.path);
	}
	if (properties.compression != FileCompressionType::UNCOMPRESSED) {
		throw InvalidInputException("Distributed byte-range scanning requires an uncompressed CSV file: \"%s\"",
		                            file.path);
	}
	if (!properties.can_seek || properties.is_pipe) {
		throw InvalidInputException("Distributed byte-range scanning requires a seekable CSV file: \"%s\"", file.path);
	}
}

static vector<DistributedScanSplit> PlanDistributedCSVSplits(const TableFunctionDistributedScanPlanningInput &input) {
	auto &bind_data = GetCSVDistributedBind(input);
	auto &csv_data = bind_data.bind_data->Cast<ReadCSVData>();
	auto files = bind_data.file_list->GetAllFiles();
	ValidateCSVDistributedState(csv_data, files);
	ValidateCSVColumnInfo(bind_data, csv_data, files);
	if (csv_data.options.store_rejects.GetValue()) {
		throw InvalidInputException("Distributed CSV scanning does not support store_rejects");
	}
	if (files.empty()) {
		return {};
	}

	vector<CSVFileSnapshot> snapshots;
	snapshots.reserve(files.size());
	for (idx_t file_idx = 0; file_idx < files.size(); file_idx++) {
		snapshots.push_back(GetCSVPlanningSnapshot(csv_data, files[file_idx], file_idx));
	}

	// An already-applied worker plan is immutable scan work. Replanning it emits the same ranges instead of
	// expanding back to the coordinator's whole-file authorization.
	if (csv_data.distributed_worker && csv_data.distributed_splits_applied) {
		vector<DistributedScanSplit> result;
		result.reserve(files.size());
		for (idx_t file_idx = 0; file_idx < files.size(); file_idx++) {
			DistributedCSVSplitPayload payload;
			payload.file = snapshots[file_idx];
			CSVScanRange range;
			payload.has_range = CSVScanRange::TryGet(files[file_idx], range);
			if (payload.has_range) {
				payload.range_start = range.start;
				payload.range_end = range.end;
			}
			optional_idx bytes;
			if (payload.has_range) {
				bytes = optional_idx(range.Size());
			}
			result.push_back(MakeDistributedCSVSplit(
			    payload, CSVCardinalityEstimate(input.estimated_cardinality, file_idx, files.size()), bytes));
		}
		return result;
	}

	vector<CSVPlanningFileProperties> properties;
	properties.reserve(files.size());
	const bool requires_byte_range_inspection = files.size() == 1 && input.target_split_count > 1;
	for (idx_t file_idx = 0; file_idx < snapshots.size(); file_idx++) {
		try {
			properties.push_back(InspectCSVPlanningFile(input, snapshots[file_idx], csv_data.options));
		} catch (const InterruptException &) {
			throw;
		} catch (const FatalException &) {
			throw;
		} catch (const InternalException &) {
			throw;
		} catch (const OutOfMemoryException &) {
			throw;
		} catch (const Exception &) {
			if (requires_byte_range_inspection) {
				throw;
			}
			// Whole-file tasks do not require a planning-time handle. Some object
			// stores can reopen an already-bound file only with execution-context
			// credentials; in that case retain an unknown byte estimate and let the
			// worker open the exact coordinator-selected OpenFileInfo.
			properties.push_back({NumericLimits<idx_t>::Maximum(), false, false, csv_data.options.compression});
		}
		if (properties.back().is_pipe) {
			throw InvalidInputException(
			    "Distributed CSV scanning requires a replayable input and does not support pipe "
			    "source \"%s\"",
			    snapshots[file_idx].path);
		}
	}

	if (files.size() == 1 && input.target_split_count > 1) {
		ValidateCSVByteRangePlanning(snapshots[0], csv_data.options, properties[0]);
	}

	if (files.size() == 1 && input.target_split_count > 1 && properties[0].size > 0) {
		const auto minimum_range_size = MaxValue<idx_t>(2, CSVIterator::BytesPerThread(csv_data.options));
		const auto safe_range_count = MaxValue<idx_t>(1, properties[0].size / minimum_range_size);
		const auto range_count = MinValue<idx_t>(input.target_split_count, safe_range_count);
		if (range_count > 1) {
			vector<DistributedScanSplit> result;
			result.reserve(range_count);
			const auto scan_unit_count = properties[0].size / minimum_range_size;
			const auto units_per_range = scan_unit_count / range_count;
			const auto extra_units = scan_unit_count % range_count;
			idx_t range_start = 0;
			for (idx_t range_idx = 0; range_idx < range_count; range_idx++) {
				const auto range_units = units_per_range + (range_idx < extra_units ? 1 : 0);
				const auto range_end =
				    range_idx + 1 == range_count ? properties[0].size : range_start + range_units * minimum_range_size;
				DistributedCSVSplitPayload payload;
				payload.file = snapshots[0];
				payload.has_range = true;
				payload.range_start = range_start;
				payload.range_end = range_end;
				result.push_back(MakeDistributedCSVSplit(
				    payload, CSVCardinalityEstimate(input.estimated_cardinality, range_idx, range_count),
				    optional_idx(range_end - range_start)));
				range_start = range_end;
			}
			return result;
		}
	}

	vector<DistributedScanSplit> result;
	result.reserve(files.size());
	for (idx_t file_idx = 0; file_idx < files.size(); file_idx++) {
		DistributedCSVSplitPayload payload;
		payload.file = snapshots[file_idx];
		optional_idx bytes;
		if (properties[file_idx].size != NumericLimits<idx_t>::Maximum()) {
			bytes = optional_idx(properties[file_idx].size);
		}
		result.push_back(MakeDistributedCSVSplit(
		    payload, CSVCardinalityEstimate(input.estimated_cardinality, file_idx, files.size()), bytes));
	}
	return result;
}

static unique_ptr<FunctionData> CreateDistributedCSVWorkerBind(const TableFunctionDistributedScanInput &input) {
	auto &source = GetCSVDistributedBind(input);
	auto &source_csv = source.bind_data->Cast<ReadCSVData>();
	auto active_files = source.file_list->GetAllFiles();
	ValidateCSVDistributedState(source_csv, active_files);
	ValidateCSVColumnInfo(source, source_csv, active_files);
	auto result = make_uniq<MultiFileBindData>();
	result->column_ids = source.column_ids;
	result->bind_data = unique_ptr_cast<FunctionData, TableFunctionData>(source_csv.Copy());
	result->file_list = make_shared_ptr<SimpleMultiFileList>(vector<OpenFileInfo> {});
	result->multi_file_reader = source.multi_file_reader->Copy();
	result->interface = source.interface->Copy();
	result->columns = source.columns;
	result->reader_bind = source.reader_bind;
	result->file_options = source.file_options;
	result->types = source.types;
	result->names = source.names;
	result->virtual_columns = source.virtual_columns;
	result->table_columns = source.table_columns;
	auto &worker = *result;
	auto &worker_csv = worker.bind_data->Cast<ReadCSVData>();

	vector<CSVFileSnapshot> allowed_files;
	set<idx_t> selected_ordinals;
	if (source_csv.distributed_worker) {
		if (source_csv.distributed_splits_applied && !active_files.empty()) {
			for (const auto &file : active_files) {
				auto allowed = GetCSVPlanningSnapshot(source_csv, file, 0);
				if (selected_ordinals.insert(allowed.ordinal).second) {
					allowed_files.push_back(std::move(allowed));
				}
			}
		} else {
			allowed_files = source_csv.distributed_allowed_files;
		}
	} else {
		allowed_files.reserve(active_files.size());
		for (idx_t file_idx = 0; file_idx < active_files.size(); file_idx++) {
			allowed_files.emplace_back(file_idx, CSVScanRange::Strip(active_files[file_idx]));
			ValidateCSVFileSnapshot(allowed_files.back(), "Distributed CSV coordinator file");
		}
	}

	if (worker.file_options.union_by_name) {
		vector<ColumnInfo> selected_column_info;
		selected_column_info.reserve(allowed_files.size());
		for (const auto &allowed : allowed_files) {
			auto entry =
			    std::find_if(source_csv.column_info.begin(), source_csv.column_info.end(), [&](const ColumnInfo &info) {
				    return CSVFileSnapshotMatches(info.file, allowed.ToOpenFileInfo());
			    });
			if (entry == source_csv.column_info.end()) {
				throw InvalidInputException("Missing CSV union reader information for file \"%s\"", allowed.path);
			}
			auto selected = *entry;
			selected.file.ordinal = allowed.ordinal;
			selected_column_info.push_back(std::move(selected));
		}
		worker_csv.column_info = std::move(selected_column_info);
	}

	worker_csv.buffer_manager.reset();
	worker_csv.distributed_worker = true;
	worker_csv.distributed_splits_applied = false;
	worker_csv.distributed_allowed_files = std::move(allowed_files);
	worker_csv.distributed_source_multiple_files =
	    source_csv.distributed_worker ? source_csv.distributed_source_multiple_files : active_files.size() > 1;
	worker_csv.distributed_authorization_restricted =
	    source_csv.distributed_worker && source_csv.distributed_authorization_restricted;
	worker_csv.distributed_authorized_split_ids = worker_csv.distributed_authorization_restricted
	                                                  ? source_csv.distributed_authorized_split_ids
	                                                  : vector<string> {};
	return std::move(result);
}

static void ApplyDistributedCSVSplits(FunctionData &worker_bind_data, const vector<DistributedScanSplit> &splits) {
	auto &worker = worker_bind_data.Cast<MultiFileBindData>();
	if (!worker.bind_data || !worker.file_list || !worker.multi_file_reader || !worker.interface) {
		throw InvalidInputException("Distributed CSV splits require complete worker bind data");
	}
	auto &csv_data = worker.bind_data->Cast<ReadCSVData>();
	if (!csv_data.distributed_worker) {
		throw InvalidInputException("Distributed CSV splits can only be applied to a detached worker bind");
	}
	ValidateCSVDistributedState(csv_data, worker.file_list->GetAllFiles());

	set<string> split_ids;
	set<idx_t> whole_file_ordinals;
	map<idx_t, vector<pair<idx_t, idx_t>>> ranges_by_ordinal;
	vector<OpenFileInfo> assigned_files;
	assigned_files.reserve(splits.size());
	for (const auto &split : splits) {
		split.Validate();
		if (!split_ids.insert(split.split_id).second) {
			throw InvalidInputException("Distributed CSV split_id '%s' appears more than once", split.split_id);
		}
		auto payload = DecodeDistributedCSVSplitPayload(split.payload);
		if (split.split_id != DistributedCSVSplitId(payload)) {
			throw InvalidInputException("Distributed CSV split_id '%s' does not match its payload", split.split_id);
		}
		optional_ptr<const CSVFileSnapshot> allowed;
		for (const auto &candidate : csv_data.distributed_allowed_files) {
			if (candidate.ordinal == payload.file.ordinal) {
				allowed = candidate;
				break;
			}
		}
		if (!allowed || allowed->path != payload.file.path || allowed->options != payload.file.options) {
			throw InvalidInputException("Distributed CSV split '%s' references a file outside its worker bind",
			                            split.split_id);
		}

		auto assigned = allowed->ToOpenFileInfo();
		if (payload.has_range) {
			if (whole_file_ordinals.find(payload.file.ordinal) != whole_file_ordinals.end()) {
				throw InvalidInputException(
				    "Distributed CSV assignment mixes whole-file and byte-range work for file %llu",
				    payload.file.ordinal);
			}
			auto &ranges = ranges_by_ordinal[payload.file.ordinal];
			for (const auto &range : ranges) {
				if (payload.range_start < range.second && range.first < payload.range_end) {
					throw InvalidInputException(
					    "Distributed CSV assignment contains overlapping byte ranges for file %llu",
					    payload.file.ordinal);
				}
			}
			ranges.emplace_back(payload.range_start, payload.range_end);
			assigned = CSVScanRange::Set(assigned, payload.range_start, payload.range_end);
		} else {
			if (!whole_file_ordinals.insert(payload.file.ordinal).second ||
			    ranges_by_ordinal.find(payload.file.ordinal) != ranges_by_ordinal.end()) {
				throw InvalidInputException("Distributed CSV assignment repeats file ordinal %llu",
				                            payload.file.ordinal);
			}
		}
		assigned_files.push_back(CSVScanRange::SetOrdinal(assigned, payload.file.ordinal));
	}

	vector<string> canonical_ids(split_ids.begin(), split_ids.end());
	if (csv_data.distributed_authorization_restricted) {
		if (canonical_ids != csv_data.distributed_authorized_split_ids) {
			throw InvalidInputException("Distributed CSV worker clone can only replay its original split assignment");
		}
	} else {
		csv_data.distributed_authorization_restricted = true;
		csv_data.distributed_authorized_split_ids = canonical_ids;
	}
	worker.file_list = make_shared_ptr<SimpleMultiFileList>(std::move(assigned_files));
	worker.initial_reader.reset();
	worker.union_readers.clear();
	csv_data.buffer_manager.reset();
	csv_data.distributed_splits_applied = true;
}

static TableFunctionDistributedScanCallbacks DistributedCSVScanCallbacks() {
	TableFunctionDistributedScanCallbacks callbacks;
	callbacks.protocol_version = DISTRIBUTED_CSV_PROTOCOL_VERSION;
	callbacks.split_codec = {DISTRIBUTED_CSV_SPLIT_CODEC, DISTRIBUTED_CSV_SPLIT_CODEC_VERSION};
	callbacks.plan_splits = PlanDistributedCSVSplits;
	callbacks.create_worker_bind = CreateDistributedCSVWorkerBind;
	callbacks.apply_splits = ApplyDistributedCSVSplits;
	return callbacks;
}

TableFunction ReadCSVTableFunction::GetFunction() {
	MultiFileFunction<CSVMultiFileInfo> read_csv("read_csv");
	read_csv.serialize = CSVReaderSerialize;
	read_csv.deserialize = CSVReaderDeserialize;
	read_csv.type_pushdown = MultiFileFunction<CSVMultiFileInfo>::PushdownType;
	ReadCSVAddNamedParameters(read_csv);
	read_csv.SetDistributedScanCallbacks(DistributedCSVScanCallbacks());
	return static_cast<TableFunction>(read_csv);
}

TableFunction ReadCSVTableFunction::GetAutoFunction() {
	auto read_csv_auto = ReadCSVTableFunction::GetFunction();
	read_csv_auto.name = "read_csv_auto";
	return read_csv_auto;
}

vector<TableFunction> ReadCSVTableFunction::GetFunctions() {
	vector<TableFunction> result;
	auto append_overloads = [&](TableFunction function) {
		auto function_set = MultiFileReader::CreateFunctionSet(std::move(function));
		for (auto &overload : function_set.functions) {
			overload.BindDistributedScanCapability("vane_core");
			result.push_back(std::move(overload));
		}
	};
	append_overloads(GetFunction());
	append_overloads(GetAutoFunction());
	return result;
}

void ReadCSVTableFunction::RegisterFunction(BuiltinFunctions &set) {
	TableFunctionSet read_csv("read_csv");
	TableFunctionSet read_csv_auto("read_csv_auto");
	for (auto &function : GetFunctions()) {
		if (function.name == "read_csv") {
			read_csv.AddFunction(std::move(function));
		} else {
			D_ASSERT(function.name == "read_csv_auto");
			read_csv_auto.AddFunction(std::move(function));
		}
	}
	set.AddFunction(std::move(read_csv));
	set.AddFunction(std::move(read_csv_auto));
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
