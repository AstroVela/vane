#include "json_scan.hpp"

#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/common/set.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "json_multi_file_info.hpp"

namespace duckdb {

JSONFileSnapshot::JSONFileSnapshot(idx_t ordinal_p, const OpenFileInfo &file) : path(file.path), ordinal(ordinal_p) {
	if (file.extended_info) {
		for (const auto &entry : file.extended_info->options) {
			if (entry.first == ORDINAL_OPTION) {
				ordinal = entry.second.GetValue<idx_t>();
				continue;
			}
			options.emplace(entry.first, entry.second);
		}
	}
}

OpenFileInfo JSONFileSnapshot::ToOpenFileInfo() const {
	OpenFileInfo result(path);
	result.extended_info = make_shared_ptr<ExtendedOpenFileInfo>();
	for (const auto &entry : options) {
		result.extended_info->options.emplace(entry.first, entry.second);
	}
	result.extended_info->options[ORDINAL_OPTION] = Value::UBIGINT(ordinal);
	return result;
}

bool JSONFileSnapshot::TryGetOrdinal(const OpenFileInfo &file, idx_t &ordinal) {
	if (!file.extended_info) {
		return false;
	}
	auto entry = file.extended_info->options.find(ORDINAL_OPTION);
	if (entry == file.extended_info->options.end()) {
		return false;
	}
	ordinal = entry->second.GetValue<idx_t>();
	return true;
}

JSONScanData::JSONScanData() {
}

void JSONScanData::InitializeFormats() {
	InitializeFormats(options.auto_detect);
}

void JSONScanData::InitializeFormats(bool auto_detect_p) {
	type_id_map_t<vector<StrpTimeFormat>> candidate_formats;
	// Initialize date_format_map if anything was specified
	if (!options.date_format.empty()) {
		DateFormatMap::AddFormat(candidate_formats, LogicalTypeId::DATE, options.date_format);
	}
	if (!options.timestamp_format.empty()) {
		DateFormatMap::AddFormat(candidate_formats, LogicalTypeId::TIMESTAMP, options.timestamp_format);
	}

	if (auto_detect_p) {
		static const type_id_map_t<vector<const char *>> FORMAT_TEMPLATES = {
		    {LogicalTypeId::DATE, {"%m-%d-%Y", "%m-%d-%y", "%d-%m-%Y", "%d-%m-%y", "%Y-%m-%d", "%y-%m-%d"}},
		    {LogicalTypeId::TIMESTAMP,
		     {"%Y-%m-%d %H:%M:%S.%f", "%m-%d-%Y %I:%M:%S %p", "%m-%d-%y %I:%M:%S %p", "%d-%m-%Y %H:%M:%S",
		      "%d-%m-%y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
		      "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"}},
		};

		// Populate possible date/timestamp formats, assume this is consistent across columns
		for (auto &kv : FORMAT_TEMPLATES) {
			const auto &logical_type = kv.first;
			if (DateFormatMap::HasFormats(candidate_formats, logical_type)) {
				continue; // Already populated
			}
			const auto &format_strings = kv.second;
			for (auto &format_string : format_strings) {
				DateFormatMap::AddFormat(candidate_formats, logical_type, format_string);
			}
		}
	}
	date_format_map = make_uniq<DateFormatMap>(std::move(candidate_formats));
}

void JSONScanData::InitializeTransformOptions() {
	transform_options.strict_cast = !options.ignore_errors;
	transform_options.error_duplicate_key = !options.ignore_errors;
	transform_options.error_missing_key = false;
	transform_options.error_unknown_key = options.auto_detect && !options.ignore_errors;
	transform_options.delay_error = true;
	transform_options.date_format_map = date_format_map.get();
	transform_options.error_message.clear();
	transform_options.object_index = DConstants::INVALID_INDEX;
	transform_options.parameters = CastParameters(false, &transform_options.error_message);
}

static vector<StrpTimeFormat> CopyJSONFormats(const optional_ptr<const DateFormatMap> format_map, LogicalTypeId type) {
	if (!format_map || !format_map->HasFormats(type)) {
		return {};
	}
	return format_map->GetFormats(type);
}

static unique_ptr<DateFormatMap> MakeJSONDateFormatMap(vector<StrpTimeFormat> date_formats,
                                                       vector<StrpTimeFormat> timestamp_formats) {
	type_id_map_t<vector<StrpTimeFormat>> candidate_formats;
	if (!date_formats.empty()) {
		candidate_formats.emplace(LogicalTypeId::DATE, std::move(date_formats));
	}
	if (!timestamp_formats.empty()) {
		candidate_formats.emplace(LogicalTypeId::TIMESTAMP, std::move(timestamp_formats));
	}
	return make_uniq<DateFormatMap>(std::move(candidate_formats));
}

unique_ptr<FunctionData> JSONScanData::Copy() const {
	auto result = make_uniq<JSONScanData>();
	result->column_ids = column_ids;
	result->options = options;
	result->key_names = key_names;
	result->date_format_map = MakeJSONDateFormatMap(CopyJSONFormats(date_format_map.get(), LogicalTypeId::DATE),
	                                                CopyJSONFormats(date_format_map.get(), LogicalTypeId::TIMESTAMP));
	result->InitializeTransformOptions();
	result->max_threads = max_threads;
	result->estimated_cardinality_per_file = estimated_cardinality_per_file;
	return std::move(result);
}

JSONScanGlobalState::JSONScanGlobalState(ClientContext &context, const MultiFileBindData &bind_data_p)
    : bind_data(bind_data_p), json_data(bind_data.bind_data->Cast<JSONScanData>()),
      transform_options(json_data.transform_options), allocator(BufferAllocator::Get(context)),
      buffer_capacity(json_data.options.maximum_object_size * 2),
      system_threads(TaskScheduler::GetScheduler(context).NumberOfThreads()),
      enable_parallel_scans(bind_data.file_list->GetTotalFileCount() < system_threads) {
}

JSONScanLocalState::JSONScanLocalState(ClientContext &context, JSONScanGlobalState &gstate)
    : scan_state(context, gstate.allocator, gstate.buffer_capacity) {
}

JSONGlobalTableFunctionState::JSONGlobalTableFunctionState(ClientContext &context, const MultiFileBindData &bind_data)
    : state(context, bind_data) {
}

JSONLocalTableFunctionState::JSONLocalTableFunctionState(ClientContext &context, JSONScanGlobalState &gstate)
    : state(context, gstate) {
}

idx_t JSONScanLocalState::Read() {
	return scan_state.current_reader->Scan(scan_state);
}

void JSONScanLocalState::ParseJSON(char *const json_start, const idx_t json_size, const idx_t remaining) {
	scan_state.current_reader->ParseJSON(scan_state, json_start, json_size, remaining);
}

bool JSONScanLocalState::TryInitializeScan(JSONScanGlobalState &gstate, JSONReader &reader) {
	// try to initialize a scan in the given reader
	// three scenarios:
	// scenario 1 - unseekable file - Read from the file and setup the buffers
	// scenario 2 - seekable file - get the position from the file to read and return
	// scenario 3 - entire file readers - if we are reading an entire file at once, do not do anything here, except for
	// setting up the basics
	auto read_type = JSONFileReadType::SCAN_PARTIAL;
	if (!gstate.enable_parallel_scans || reader.GetFormat() != JSONFormat::NEWLINE_DELIMITED) {
		read_type = JSONFileReadType::SCAN_ENTIRE_FILE;
	}
	if (read_type == JSONFileReadType::SCAN_ENTIRE_FILE) {
		if (gstate.file_is_assigned) {
			return false;
		}
		gstate.file_is_assigned = true;
	}
	return reader.InitializeScan(scan_state, read_type);
}

void JSONScanLocalState::AddTransformError(idx_t object_index, const string &error_message) {
	scan_state.current_reader->AddTransformError(scan_state, object_index, error_message);
}

static void ValidateJSONReaderOptions(const JSONReaderOptions &options, const string &function_name) {
	if (options.type != JSONScanType::READ_JSON && options.type != JSONScanType::READ_JSON_OBJECTS) {
		throw SerializationException("Cannot serialize %s with invalid JSON scan type", function_name);
	}
	switch (options.format) {
	case JSONFormat::AUTO_DETECT:
	case JSONFormat::UNSTRUCTURED:
	case JSONFormat::NEWLINE_DELIMITED:
	case JSONFormat::ARRAY:
		break;
	default:
		throw SerializationException("Cannot serialize %s with invalid JSON format", function_name);
	}
	if (options.record_type != JSONRecordType::RECORDS && options.record_type != JSONRecordType::VALUES) {
		throw SerializationException("Cannot serialize %s with unresolved JSON record type", function_name);
	}
	switch (options.compression) {
	case FileCompressionType::AUTO_DETECT:
	case FileCompressionType::UNCOMPRESSED:
	case FileCompressionType::GZIP:
	case FileCompressionType::ZSTD:
		break;
	default:
		throw SerializationException("Cannot serialize %s with invalid JSON compression type", function_name);
	}
	if (options.maximum_object_size == 0 || options.sample_size == 0 || options.maximum_sample_files == 0) {
		throw SerializationException("Cannot serialize %s with zero-sized JSON reader limits", function_name);
	}
	if (options.maximum_object_size > NumericLimits<idx_t>::Maximum() / 2) {
		throw SerializationException("Cannot serialize %s with an oversized JSON object limit", function_name);
	}
	if (!Value::DoubleIsFinite(options.field_appearance_threshold) || options.field_appearance_threshold < 0 ||
	    options.field_appearance_threshold > 1) {
		throw SerializationException("Cannot serialize %s with invalid JSON field appearance threshold", function_name);
	}
	if (options.name_list.size() != options.sql_type_list.size()) {
		throw SerializationException("Cannot serialize %s with mismatched JSON option names and types", function_name);
	}
	for (const auto &format : {options.date_format, options.timestamp_format}) {
		if (format.empty()) {
			continue;
		}
		StrpTimeFormat parsed_format;
		if (!StrTimeFormat::ParseFormatSpecifier(format, parsed_format).empty()) {
			throw SerializationException("Cannot serialize %s with an invalid JSON date format", function_name);
		}
	}
}

static void ValidateJSONFormats(const vector<StrpTimeFormat> &formats, const string &function_name) {
	for (const auto &format : formats) {
		StrpTimeFormat parsed_format;
		if (format.format_specifier.empty() ||
		    !StrTimeFormat::ParseFormatSpecifier(format.format_specifier, parsed_format).empty()) {
			throw SerializationException("Cannot serialize %s with invalid inferred JSON date formats", function_name);
		}
	}
}

static void ValidateJSONSchema(const vector<LogicalType> &types, const vector<string> &names,
                               const vector<string> &key_names, const MultiFileOptions &file_options,
                               const MultiFileReaderBindData &reader_bind, const string &function_name) {
	if (types.empty() || types.size() != names.size() || key_names.empty() || key_names.size() > names.size()) {
		throw SerializationException("Cannot serialize %s with an invalid JSON schema", function_name);
	}
	if (file_options.mapping != MultiFileColumnMappingMode::BY_NAME || !file_options.custom_options.empty() ||
	    reader_bind.mapping != MultiFileColumnMappingMode::BY_NAME || !reader_bind.schema.empty()) {
		throw SerializationException("Cannot serialize %s with non-standard multi-file column mapping state",
		                             function_name);
	}
	if (!reader_bind.filename_idx.IsValid()) {
		throw SerializationException("Cannot serialize %s without bound filename metadata", function_name);
	}
	const auto filename_idx = reader_bind.filename_idx.GetIndex();
	if (file_options.filename) {
		if (IsVirtualColumn(filename_idx) || filename_idx >= names.size() ||
		    names[filename_idx] != file_options.filename_column || types[filename_idx].id() != LogicalTypeId::VARCHAR) {
			throw SerializationException("Cannot serialize %s with an invalid filename column", function_name);
		}
	} else if (filename_idx != MultiFileReader::COLUMN_IDENTIFIER_FILENAME) {
		throw SerializationException("Cannot serialize %s with inconsistent filename metadata", function_name);
	}

	set<idx_t> hive_indexes;
	for (const auto &hive_index : reader_bind.hive_partitioning_indexes) {
		if (!file_options.hive_partitioning || hive_index.value.empty() || hive_index.index >= names.size() ||
		    !hive_indexes.insert(hive_index.index).second) {
			throw SerializationException("Cannot serialize %s with invalid hive partition columns", function_name);
		}
	}
	for (idx_t column_idx = key_names.size(); column_idx < names.size(); column_idx++) {
		if (filename_idx != column_idx && hive_indexes.find(column_idx) == hive_indexes.end()) {
			throw SerializationException("Cannot serialize %s with incomplete JSON key state", function_name);
		}
	}
}

static void ValidateJSONBindData(const MultiFileBindData &bind_data, const string &function_name) {
	if (!bind_data.bind_data || !bind_data.file_list || !bind_data.multi_file_reader || !bind_data.interface) {
		throw SerializationException("Cannot serialize incomplete %s bind data", function_name);
	}

	auto &json_data = bind_data.bind_data->Cast<JSONScanData>();
	ValidateJSONReaderOptions(json_data.options, function_name);
	ValidateJSONSchema(bind_data.types, bind_data.names, json_data.key_names, bind_data.file_options,
	                   bind_data.reader_bind, function_name);
	if (!json_data.date_format_map || json_data.transform_options.date_format_map != json_data.date_format_map.get() ||
	    json_data.transform_options.strict_cast != !json_data.options.ignore_errors ||
	    json_data.transform_options.error_duplicate_key != !json_data.options.ignore_errors ||
	    json_data.transform_options.error_missing_key ||
	    json_data.transform_options.error_unknown_key !=
	        (json_data.options.auto_detect && !json_data.options.ignore_errors) ||
	    !json_data.transform_options.delay_error ||
	    json_data.transform_options.parameters.error_message != &json_data.transform_options.error_message) {
		throw SerializationException("Cannot serialize %s with inconsistent JSON transform state", function_name);
	}
	if ((json_data.max_threads.IsValid() && json_data.max_threads.GetIndex() == 0) ||
	    (json_data.estimated_cardinality_per_file.IsValid() &&
	     json_data.estimated_cardinality_per_file.GetIndex() == 0)) {
		throw SerializationException("Cannot serialize %s with invalid JSON cardinality state", function_name);
	}
}

static void ValidateJSONFiles(const vector<JSONFileSnapshot> &files, const string &function_name,
                              const string &operation) {
	set<idx_t> ordinals;
	for (const auto &file : files) {
		if (file.path.empty()) {
			throw SerializationException("Cannot %s %s with an empty JSON file path", operation, function_name);
		}
		if (file.options.find(JSONFileSnapshot::ORDINAL_OPTION) != file.options.end()) {
			throw SerializationException("Cannot %s %s with nested JSON file ordinal metadata", operation,
			                             function_name);
		}
		if (!ordinals.insert(file.ordinal).second) {
			throw SerializationException("Cannot %s %s with duplicate JSON file ordinal %llu", operation, function_name,
			                             static_cast<unsigned long long>(file.ordinal));
		}
	}
}

void JSONScan::Serialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data_p,
                         const TableFunction &function) {
	if (!bind_data_p) {
		throw SerializationException("Cannot serialize %s without bind data", function.name);
	}
	auto &bind_data = bind_data_p->Cast<MultiFileBindData>();
	ValidateJSONBindData(bind_data, function.name);
	auto &json_data = bind_data.bind_data->Cast<JSONScanData>();

	SerializedJSONScanData serialized_data;
	auto files = bind_data.file_list->GetAllFiles();
	serialized_data.files.reserve(files.size());
	for (idx_t file_idx = 0; file_idx < files.size(); file_idx++) {
		serialized_data.files.emplace_back(file_idx, files[file_idx]);
	}
	ValidateJSONFiles(serialized_data.files, function.name, "serialize");
	serialized_data.types = bind_data.types;
	serialized_data.names = bind_data.names;
	serialized_data.file_options = bind_data.file_options;
	serialized_data.reader_bind = bind_data.reader_bind;
	serialized_data.table_columns = bind_data.table_columns;
	serialized_data.bind_column_ids = bind_data.column_ids;
	serialized_data.options = json_data.options;
	serialized_data.key_names = json_data.key_names;
	serialized_data.date_formats = CopyJSONFormats(json_data.date_format_map.get(), LogicalTypeId::DATE);
	serialized_data.timestamp_formats = CopyJSONFormats(json_data.date_format_map.get(), LogicalTypeId::TIMESTAMP);
	ValidateJSONFormats(serialized_data.date_formats, function.name);
	ValidateJSONFormats(serialized_data.timestamp_formats, function.name);
	serialized_data.max_threads = json_data.max_threads;
	serialized_data.estimated_cardinality_per_file = json_data.estimated_cardinality_per_file;
	serialized_data.reader_column_ids = json_data.column_ids;
	serializer.WriteProperty(100, "json_data", serialized_data);
}

unique_ptr<FunctionData> JSONScan::Deserialize(Deserializer &deserializer, TableFunction &function) {
	auto &context = deserializer.Get<ClientContext &>();
	auto serialized_data = deserializer.ReadProperty<SerializedJSONScanData>(100, "json_data");
	ValidateJSONReaderOptions(serialized_data.options, function.name);
	ValidateJSONFormats(serialized_data.date_formats, function.name);
	ValidateJSONFormats(serialized_data.timestamp_formats, function.name);
	ValidateJSONSchema(serialized_data.types, serialized_data.names, serialized_data.key_names,
	                   serialized_data.file_options, serialized_data.reader_bind, function.name);
	ValidateJSONFiles(serialized_data.files, function.name, "deserialize");

	vector<OpenFileInfo> files;
	files.reserve(serialized_data.files.size());
	for (const auto &file : serialized_data.files) {
		files.push_back(file.ToOpenFileInfo());
	}
	auto multi_file_reader = MultiFileReader::Create(function);
	auto file_list = make_shared_ptr<SimpleMultiFileList>(std::move(files));
	auto interface = make_uniq<JSONMultiFileInfo>();
	interface->InitializeInterface(context, *multi_file_reader, *file_list);

	auto result = make_uniq<MultiFileBindData>();
	result->file_list = std::move(file_list);
	result->multi_file_reader = std::move(multi_file_reader);
	result->interface = std::move(interface);
	result->file_options = std::move(serialized_data.file_options);
	result->reader_bind = std::move(serialized_data.reader_bind);
	result->types = std::move(serialized_data.types);
	result->names = std::move(serialized_data.names);
	result->table_columns = std::move(serialized_data.table_columns);
	result->column_ids = std::move(serialized_data.bind_column_ids);

	auto reader_options = make_uniq<JSONFileReaderOptions>();
	reader_options->options = std::move(serialized_data.options);
	result->bind_data = result->interface->InitializeBindData(*result, std::move(reader_options));
	auto &json_data = result->bind_data->Cast<JSONScanData>();
	json_data.key_names = std::move(serialized_data.key_names);
	json_data.date_format_map =
	    MakeJSONDateFormatMap(std::move(serialized_data.date_formats), std::move(serialized_data.timestamp_formats));
	json_data.InitializeTransformOptions();
	json_data.max_threads = serialized_data.max_threads;
	json_data.estimated_cardinality_per_file = serialized_data.estimated_cardinality_per_file;
	json_data.column_ids = std::move(serialized_data.reader_column_ids);

	result->columns = MultiFileColumnDefinition::ColumnsFromNamesAndTypes(result->names, result->types);
	virtual_column_map_t virtual_columns;
	MultiFileReader::GetVirtualColumns(context, result->reader_bind, virtual_columns);
	result->interface->GetVirtualColumns(context, *result, virtual_columns);
	result->virtual_columns = std::move(virtual_columns);
	result->interface->FinalizeBindData(*result);
	ValidateJSONBindData(*result, function.name);
	return std::move(result);
}

void JSONScan::TableFunctionDefaults(TableFunction &table_function) {
	table_function.named_parameters["maximum_object_size"] = LogicalType::UINTEGER;
	table_function.named_parameters["ignore_errors"] = LogicalType::BOOLEAN;
	table_function.named_parameters["format"] = LogicalType::VARCHAR;
	table_function.named_parameters["compression"] = LogicalType::VARCHAR;

	table_function.serialize = Serialize;
	table_function.deserialize = Deserialize;

	table_function.projection_pushdown = true;
	table_function.filter_pushdown = false;
	table_function.filter_prune = false;
}

} // namespace duckdb
