//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/table/read_csv.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/execution/operator/csv_scanner/csv_buffer.hpp"
#include "duckdb/execution/operator/csv_scanner/csv_buffer_manager.hpp"
#include "duckdb/execution/operator/csv_scanner/csv_file_handle.hpp"
#include "duckdb/execution/operator/csv_scanner/csv_reader_options.hpp"
#include "duckdb/execution/operator/csv_scanner/csv_state_machine_cache.hpp"
#include "duckdb/function/built_in_functions.hpp"
#include "duckdb/function/scalar/strftime_format.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/execution/operator/csv_scanner/csv_file_scanner.hpp"
#include "duckdb/common/csv_writer.hpp"

namespace duckdb {
class BaseScanner;
class StringValueScanner;

class ReadCSV {
public:
	static unique_ptr<CSVFileHandle> OpenCSV(const OpenFileInfo &file, const CSVReaderOptions &options,
	                                         ClientContext &context);
};

struct BaseCSVData : public TableFunctionData {
	//! The CSV reader options
	CSVReaderOptions options;
	//! Offsets for generated columns
	idx_t filename_col_idx {};
	idx_t hive_partition_col_idx {};

	void Finalize();
};

struct SerializedCSVReaderOptions {
	SerializedCSVReaderOptions() = default;
	SerializedCSVReaderOptions(CSVReaderOptions options, MultiFileOptions file_options);
	SerializedCSVReaderOptions(CSVOption<char> single_byte_delimiter, const CSVOption<string> &multi_byte_delimiter);

	CSVReaderOptions options;
	MultiFileOptions file_options;

	void Serialize(Serializer &serializer) const;
	static SerializedCSVReaderOptions Deserialize(Deserializer &deserializer);
};

//! Owned, deterministic representation of an OpenFileInfo. The ordinal is the
//! stable file identity within one already-bound CSV scan.
struct CSVFileSnapshot {
	CSVFileSnapshot() = default;
	explicit CSVFileSnapshot(idx_t ordinal, const OpenFileInfo &file);

	idx_t ordinal {};
	string path;
	map<string, Value> options;

	OpenFileInfo ToOpenFileInfo() const;
	void Serialize(Serializer &serializer) const;
	static CSVFileSnapshot Deserialize(Deserializer &deserializer);
};

struct WriteCSVData : public BaseCSVData {
	explicit WriteCSVData(vector<string> names) {
		options.name_list = std::move(names);
		if (options.dialect_options.state_machine_options.escape == '\0') {
			options.dialect_options.state_machine_options.escape = options.dialect_options.state_machine_options.quote;
		}
	}
	//! The size of the CSV file (in bytes) that we buffer before we flush it to disk
	idx_t flush_size = 4096ULL * 8ULL;
	//! Expressions used to convert the input into strings
	vector<unique_ptr<Expression>> cast_expressions;
	//! Deserialized COPY binds reconstruct their expressions in the real worker
	//! context. Local sink states can initialize concurrently, so construction
	//! and copying of the shared expression templates must be serialized.
	mutex cast_expressions_lock;
	//! Original input types, used to reconstruct casts after plan deserialization
	vector<LogicalType> sql_types;

	unique_ptr<FunctionData> Copy() const override;
};

struct ColumnInfo {
	ColumnInfo() {
	}
	ColumnInfo(vector<std::string> names_p, vector<LogicalType> types_p, CSVFileSnapshot file_p,
	           SerializedCSVReaderOptions options_p) {
		names = std::move(names_p);
		types = std::move(types_p);
		file = std::move(file_p);
		options = std::move(options_p);
	}
	void Serialize(Serializer &serializer) const;
	static ColumnInfo Deserialize(Deserializer &deserializer);

	vector<std::string> names;
	vector<LogicalType> types;
	CSVFileSnapshot file;
	SerializedCSVReaderOptions options;
};

struct ReadCSVData : public BaseCSVData {
	ReadCSVData();
	//! If the sql types from the file were manually set
	vector<bool> manually_set;
	//! The buffer manager (if any): this is used when automatic detection is used during binding.
	//! In this case, some CSV buffers have already been read and can be reused.
	shared_ptr<CSVBufferManager> buffer_manager;
	//! Column info (used for union reader serialization)
	vector<ColumnInfo> column_info;
	//! The CSV schema, in case there is a unified schema that all files must read
	CSVSchema csv_schema;
	//! True only for worker binds created by the explicit distributed scan contract
	bool distributed_worker = false;
	//! A detached worker must remain fail-closed until apply_splits installs an assignment
	bool distributed_splits_applied = false;
	//! Base files selected by the coordinator; these are metadata, never the active worker file list
	vector<CSVFileSnapshot> distributed_allowed_files;
	//! Preserve whether the coordinator's already-pruned scan contained multiple files. Non-union auto-detect readers
	//! use this to retain ordinary per-file schema validation even when one worker receives only one whole-file task.
	bool distributed_source_multiple_files = false;
	//! Once a worker assignment has been installed, clones may only replay the same elementary splits
	bool distributed_authorization_restricted = false;
	vector<string> distributed_authorized_split_ids;

	void FinalizeRead(ClientContext &context);
	unique_ptr<FunctionData> Copy() const override;
};

struct SerializedReadCSVData {
	vector<CSVFileSnapshot> files;
	vector<LogicalType> csv_types;
	vector<string> csv_names;
	vector<LogicalType> return_types;
	vector<string> return_names;
	idx_t filename_col_idx {};
	idx_t hive_partition_col_idx {};
	SerializedCSVReaderOptions options;
	MultiFileReaderBindData reader_bind;
	vector<ColumnInfo> column_info;
	vector<string> table_columns;
	vector<bool> manually_set;
	bool has_csv_schema = false;
	vector<string> csv_schema_names;
	vector<LogicalType> csv_schema_types;
	string csv_schema_path;
	idx_t csv_schema_rows_read {};
	bool csv_schema_empty_file = false;
	vector<idx_t> bind_column_ids;
	vector<idx_t> reader_column_ids;
	bool distributed_worker = false;
	bool distributed_splits_applied = false;
	vector<CSVFileSnapshot> distributed_allowed_files;
	bool distributed_source_multiple_files = false;
	bool distributed_authorization_restricted = false;
	vector<string> distributed_authorized_split_ids;

	void Serialize(Serializer &serializer) const;
	static SerializedReadCSVData Deserialize(Deserializer &deserializer);
};

struct CSVCopyFunction {
	static void RegisterFunction(BuiltinFunctions &set);
};

struct ReadCSVTableFunction {
	static TableFunction GetFunction();
	static TableFunction GetAutoFunction();
	static vector<TableFunction> GetFunctions();
	static void ReadCSVAddNamedParameters(TableFunction &table_function);
	static void RegisterFunction(BuiltinFunctions &set);
};

} // namespace duckdb
