#include "duckdb/function/table/read_file.hpp"
#include "duckdb/function/table/direct_file_reader.hpp"
#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/common/multi_file/multi_file_list.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/function/table/range.hpp"
#include "utf8proc_wrapper.hpp"
#include "duckdb/storage/caching_file_system.hpp"

namespace duckdb {

unique_ptr<FunctionData> ReadFileBindData::Copy() const {
	auto result = make_uniq<ReadFileBindData>();
	result->column_ids = column_ids;
	if (options) {
		result->options = make_uniq<BaseFileReaderOptions>(*options);
	}
	return std::move(result);
}

namespace {

//------------------------------------------------------------------------------
// DirectMultiFileInfo
//------------------------------------------------------------------------------

template <class OP>
struct DirectMultiFileInfo : MultiFileReaderInterface {
	static unique_ptr<MultiFileReaderInterface> CreateInterface(ClientContext &context);
	unique_ptr<BaseFileReaderOptions> InitializeOptions(ClientContext &context,
	                                                    optional_ptr<TableFunctionInfo> info) override;
	bool ParseCopyOption(ClientContext &context, const string &key, const vector<Value> &values,
	                     BaseFileReaderOptions &options, vector<string> &expected_names,
	                     vector<LogicalType> &expected_types) override;
	bool ParseOption(ClientContext &context, const string &key, const Value &val, MultiFileOptions &file_options,
	                 BaseFileReaderOptions &options) override;
	unique_ptr<TableFunctionData> InitializeBindData(MultiFileBindData &multi_file_data,
	                                                 unique_ptr<BaseFileReaderOptions> options) override;
	void BindReader(ClientContext &context, vector<LogicalType> &return_types, vector<string> &names,
	                MultiFileBindData &bind_data) override;
	optional_idx MaxThreads(const MultiFileBindData &bind_data_p, const MultiFileGlobalState &global_state,
	                        FileExpandResult expand_result) override;
	unique_ptr<GlobalTableFunctionState> InitializeGlobalState(ClientContext &context, MultiFileBindData &bind_data,
	                                                           MultiFileGlobalState &global_state) override;
	unique_ptr<LocalTableFunctionState> InitializeLocalState(ExecutionContext &, GlobalTableFunctionState &) override;
	shared_ptr<BaseFileReader> CreateReader(ClientContext &context, GlobalTableFunctionState &gstate,
	                                        BaseUnionData &union_data, const MultiFileBindData &bind_data_p) override;
	shared_ptr<BaseFileReader> CreateReader(ClientContext &context, GlobalTableFunctionState &gstate,
	                                        const OpenFileInfo &file, idx_t file_idx,
	                                        const MultiFileBindData &bind_data) override;
	shared_ptr<BaseFileReader> CreateReader(ClientContext &context, const OpenFileInfo &file,
	                                        BaseFileReaderOptions &options,
	                                        const MultiFileOptions &file_options) override;
	unique_ptr<NodeStatistics> GetCardinality(const MultiFileBindData &bind_data, idx_t file_count) override;
	unique_ptr<MultiFileReaderInterface> Copy() override;
	FileGlobInput GetGlobInput() override;
};

template <class OP>
unique_ptr<MultiFileReaderInterface> DirectMultiFileInfo<OP>::CreateInterface(ClientContext &context) {
	return make_uniq<DirectMultiFileInfo>();
}

template <class OP>
unique_ptr<BaseFileReaderOptions> DirectMultiFileInfo<OP>::InitializeOptions(ClientContext &context,
                                                                             optional_ptr<TableFunctionInfo> info) {
	return make_uniq<BaseFileReaderOptions>();
}

template <class OP>
bool DirectMultiFileInfo<OP>::ParseCopyOption(ClientContext &context, const string &key, const vector<Value> &values,
                                              BaseFileReaderOptions &options, vector<string> &expected_names,
                                              vector<LogicalType> &expected_types) {
	return true;
}

template <class OP>
bool DirectMultiFileInfo<OP>::ParseOption(ClientContext &context, const string &key, const Value &val,
                                          MultiFileOptions &file_options, BaseFileReaderOptions &options) {
	return true;
}

template <class OP>
unique_ptr<TableFunctionData> DirectMultiFileInfo<OP>::InitializeBindData(MultiFileBindData &multi_file_data,
                                                                          unique_ptr<BaseFileReaderOptions> options) {
	auto result = make_uniq<ReadFileBindData>();
	result->options = std::move(options);
	return std::move(result);
}

template <class OP>
void DirectMultiFileInfo<OP>::BindReader(ClientContext &context, vector<LogicalType> &return_types,
                                         vector<string> &names, MultiFileBindData &bind_data) {
	auto &read_bind = bind_data.bind_data->Cast<ReadFileBindData>();
	bind_data.reader_bind = bind_data.multi_file_reader->BindReader(
	    context, return_types, names, *bind_data.file_list, bind_data, *read_bind.options, bind_data.file_options);
}

template <class OP>
optional_idx DirectMultiFileInfo<OP>::MaxThreads(const MultiFileBindData &bind_data_p,
                                                 const MultiFileGlobalState &global_state,
                                                 FileExpandResult expand_result) {
	return 1;
}

template <class OP>
unique_ptr<GlobalTableFunctionState>
DirectMultiFileInfo<OP>::InitializeGlobalState(ClientContext &context, MultiFileBindData &bind_data,
                                               MultiFileGlobalState &global_state) {
	auto result = make_uniq<ReadFileGlobalState>();

	result->file_list = bind_data.file_list;
	vector<idx_t> column_ids;
	for (idx_t i = 0; i < global_state.column_indexes.size(); i++) {
		// We only look at file-related columns
		if (global_state.column_indexes[i].GetPrimaryIndex() <= ReadFileBindData::FILE_LAST_MODIFIED_COLUMN) {
			column_ids.push_back(global_state.column_indexes[i].GetPrimaryIndex());
		}
	}

	for (const auto &column_id : column_ids) {
		// For everything except the 'file' name column, we need to open the file
		if (column_id != ReadFileBindData::FILE_NAME_COLUMN && column_id != COLUMN_IDENTIFIER_ROW_ID) {
			result->requires_file_open = true;
			break;
		}
	}

	result->column_ids = std::move(column_ids);
	return std::move(result);
}

template <class OP>
unique_ptr<LocalTableFunctionState> DirectMultiFileInfo<OP>::InitializeLocalState(ExecutionContext &,
                                                                                  GlobalTableFunctionState &) {
	return make_uniq<LocalTableFunctionState>();
}

template <class OP>
shared_ptr<BaseFileReader>
DirectMultiFileInfo<OP>::CreateReader(ClientContext &context, GlobalTableFunctionState &gstate,
                                      BaseUnionData &union_data, const MultiFileBindData &bind_data_p) {
	return make_shared_ptr<DirectFileReader>(union_data.file, OP::TYPE());
}

template <class OP>
shared_ptr<BaseFileReader>
DirectMultiFileInfo<OP>::CreateReader(ClientContext &context, GlobalTableFunctionState &gstate,
                                      const OpenFileInfo &file, idx_t file_idx, const MultiFileBindData &bind_data) {
	return make_shared_ptr<DirectFileReader>(file, OP::TYPE());
}

template <class OP>
shared_ptr<BaseFileReader> DirectMultiFileInfo<OP>::CreateReader(ClientContext &context, const OpenFileInfo &file,
                                                                 BaseFileReaderOptions &options_p,
                                                                 const MultiFileOptions &) {
	return make_shared_ptr<DirectFileReader>(file, OP::TYPE());
}

template <class OP>
unique_ptr<NodeStatistics> DirectMultiFileInfo<OP>::GetCardinality(const MultiFileBindData &bind_data,
                                                                   idx_t file_count) {
	auto result = make_uniq<NodeStatistics>();
	result->has_max_cardinality = true;
	result->max_cardinality = bind_data.file_list->GetTotalFileCount();
	result->has_estimated_cardinality = true;
	result->estimated_cardinality = bind_data.file_list->GetTotalFileCount();
	return result;
}

template <class OP>
unique_ptr<MultiFileReaderInterface> DirectMultiFileInfo<OP>::Copy() {
	return make_uniq<DirectMultiFileInfo>();
}

template <class OP>
FileGlobInput DirectMultiFileInfo<OP>::GetGlobInput() {
	return FileGlobOptions::ALLOW_EMPTY;
}

//------------------------------------------------------------------------------
// Operations
//------------------------------------------------------------------------------

struct ReadBlobOperation {
	static constexpr const char *NAME = "read_blob";

	static inline LogicalType TYPE() {
		return LogicalType::BLOB;
	}
};

struct ReadTextOperation {
	static constexpr const char *NAME = "read_text";

	static inline LogicalType TYPE() {
		return LogicalType::VARCHAR;
	}
};

static void ReadFileSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data_p,
                              const TableFunction &function) {
	if (!bind_data_p) {
		throw SerializationException("Cannot serialize %s without bind data", function.name);
	}
	auto &bind_data = bind_data_p->Cast<MultiFileBindData>();
	if (!bind_data.file_list) {
		throw SerializationException("Cannot serialize %s without a file list", function.name);
	}

	auto files = bind_data.file_list->GetAllFiles();
	serializer.WriteList(100, "files", files.size(), [&](Serializer::List &list, idx_t index) {
		list.WriteObject([&](Serializer &object) {
			object.WriteProperty(100, "path", files[index].path);
			unordered_map<string, Value> options;
			if (files[index].extended_info) {
				options = files[index].extended_info->options;
			}
			object.WriteProperty(101, "options", options);
		});
	});
	serializer.WriteProperty(101, "types", bind_data.types);
	serializer.WriteProperty(102, "names", bind_data.names);
	serializer.WriteProperty(103, "file_options", bind_data.file_options);
	serializer.WriteProperty(104, "reader_bind", bind_data.reader_bind);
	serializer.WriteProperty(105, "table_columns", bind_data.table_columns);
	serializer.WriteProperty(106, "column_ids", bind_data.column_ids);
}

template <class OP>
static unique_ptr<FunctionData> ReadFileDeserialize(Deserializer &deserializer, TableFunction &function) {
	auto &context = deserializer.Get<ClientContext &>();
	vector<OpenFileInfo> files;
	deserializer.ReadList(100, "files", [&](Deserializer::List &list, idx_t) {
		list.ReadObject([&](Deserializer &object) {
			OpenFileInfo file;
			file.path = object.ReadProperty<string>(100, "path");
			auto options = object.ReadProperty<unordered_map<string, Value>>(101, "options");
			if (!options.empty()) {
				auto extended_info = make_shared_ptr<ExtendedOpenFileInfo>();
				extended_info->options = std::move(options);
				file.extended_info = std::move(extended_info);
			}
			files.push_back(std::move(file));
		});
	});
	auto types = deserializer.ReadProperty<vector<LogicalType>>(101, "types");
	auto names = deserializer.ReadProperty<vector<string>>(102, "names");
	auto file_options = deserializer.ReadProperty<MultiFileOptions>(103, "file_options");
	auto reader_bind = deserializer.ReadProperty<MultiFileReaderBindData>(104, "reader_bind");
	auto table_columns = deserializer.ReadProperty<vector<string>>(105, "table_columns");
	auto column_ids = deserializer.ReadProperty<vector<idx_t>>(106, "column_ids");

	if (types.size() != names.size()) {
		throw SerializationException("Cannot deserialize %s with mismatched names and types", function.name);
	}

	auto multi_file_reader = MultiFileReader::Create(function);
	auto file_list = make_shared_ptr<SimpleMultiFileList>(std::move(files));
	auto interface = DirectMultiFileInfo<OP>::CreateInterface(context);
	interface->InitializeInterface(context, *multi_file_reader, *file_list);
	auto options = interface->InitializeOptions(context, nullptr);

	auto result = make_uniq<MultiFileBindData>();
	result->file_list = std::move(file_list);
	result->multi_file_reader = std::move(multi_file_reader);
	result->interface = std::move(interface);
	result->file_options = std::move(file_options);
	result->reader_bind = std::move(reader_bind);
	result->types = std::move(types);
	result->names = std::move(names);
	result->table_columns = std::move(table_columns);
	result->column_ids = std::move(column_ids);
	result->bind_data = result->interface->InitializeBindData(*result, std::move(options));
	result->columns = MultiFileColumnDefinition::ColumnsFromNamesAndTypes(result->names, result->types);

	virtual_column_map_t virtual_columns;
	MultiFileReader::GetVirtualColumns(context, result->reader_bind, virtual_columns);
	result->interface->GetVirtualColumns(context, *result, virtual_columns);
	result->virtual_columns = std::move(virtual_columns);
	result->interface->FinalizeBindData(*result);
	return std::move(result);
}

template <class OP>
static TableFunction GetFunction() {
	MultiFileFunction<DirectMultiFileInfo<OP>> table_function(OP::NAME);
	table_function.serialize = ReadFileSerialize;
	table_function.deserialize = ReadFileDeserialize<OP>;
	// Erase extra multi file reader options
	table_function.named_parameters.erase("filename");
	table_function.named_parameters.erase("hive_partitioning");
	table_function.named_parameters.erase("union_by_name");
	table_function.named_parameters.erase("hive_types");
	table_function.named_parameters.erase("hive_types_autocast");
	return table_function;
}

} // namespace

//------------------------------------------------------------------------------
// Register
//------------------------------------------------------------------------------

void ReadBlobFunction::RegisterFunction(BuiltinFunctions &set) {
	auto scan_fun = GetFunction<ReadBlobOperation>();
	set.AddFunction(MultiFileReader::CreateFunctionSet(scan_fun));
}

void ReadTextFunction::RegisterFunction(BuiltinFunctions &set) {
	auto scan_fun = GetFunction<ReadTextOperation>();
	set.AddFunction(MultiFileReader::CreateFunctionSet(scan_fun));
}

} // namespace duckdb
