//===----------------------------------------------------------------------===//
//                         DuckDB
//
// json_scan.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "json_reader.hpp"
#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/pair.hpp"
#include "duckdb/common/types/type_map.hpp"
#include "duckdb/function/scalar/strftime_format.hpp"
#include "duckdb/function/table_function.hpp"
#include "json_enums.hpp"
#include "json_transform.hpp"
#include "json_reader_options.hpp"

namespace duckdb {

//! Owned, deterministic representation of an OpenFileInfo used by a bound JSON scan.
struct JSONFileSnapshot {
	JSONFileSnapshot() = default;
	JSONFileSnapshot(idx_t ordinal, const OpenFileInfo &file);

	static constexpr const char *ORDINAL_OPTION = "__vane_json_file_ordinal";
	static bool TryGetOrdinal(const OpenFileInfo &file, idx_t &ordinal);

	string path;
	map<string, Value> options;
	idx_t ordinal = 0;

	OpenFileInfo ToOpenFileInfo() const;
	void Serialize(Serializer &serializer) const;
	static JSONFileSnapshot Deserialize(Deserializer &deserializer);
};

struct JSONScanData : public TableFunctionData {
public:
	JSONScanData();

	void InitializeFormats();
	void InitializeFormats(bool auto_detect);
	void InitializeTransformOptions();
	unique_ptr<FunctionData> Copy() const override;

public:
	//! JSON reader options
	JSONReaderOptions options;

	//! The set of keys to extract (case sensitive)
	vector<string> key_names;

	//! The date format map
	unique_ptr<DateFormatMap> date_format_map;
	//! Options when transforming the JSON to columnar data
	JSONTransformOptions transform_options;

	optional_idx max_threads;
	optional_idx estimated_cardinality_per_file;
};

//! Complete owned state required to reconstruct a bound JSON multi-file scan without reopening or resampling files.
struct SerializedJSONScanData {
	vector<JSONFileSnapshot> files;
	vector<LogicalType> types;
	vector<string> names;
	MultiFileOptions file_options;
	MultiFileReaderBindData reader_bind;
	vector<string> table_columns;
	vector<idx_t> bind_column_ids;
	JSONReaderOptions options;
	vector<string> key_names;
	vector<StrpTimeFormat> date_formats;
	vector<StrpTimeFormat> timestamp_formats;
	optional_idx max_threads;
	optional_idx estimated_cardinality_per_file;
	vector<idx_t> reader_column_ids;

	void Serialize(Serializer &serializer) const;
	static SerializedJSONScanData Deserialize(Deserializer &deserializer);
};

struct JSONScanInfo : public TableFunctionInfo {
public:
	explicit JSONScanInfo(JSONScanType type_p = JSONScanType::INVALID, JSONFormat format_p = JSONFormat::AUTO_DETECT,
	                      JSONRecordType record_type_p = JSONRecordType::AUTO_DETECT, bool auto_detect_p = false)
	    : type(type_p), format(format_p), record_type(record_type_p), auto_detect(auto_detect_p) {
	}

	JSONScanType type;
	JSONFormat format;
	JSONRecordType record_type;
	bool auto_detect;
};

struct JSONScanGlobalState {
public:
	JSONScanGlobalState(ClientContext &context, const MultiFileBindData &bind_data);

public:
	//! Bound data
	const MultiFileBindData &bind_data;
	const JSONScanData &json_data;
	//! Options when transforming the JSON to columnar data
	JSONTransformOptions transform_options;

	//! Column names that we're actually reading (after projection pushdown)
	vector<string> names;
	vector<column_t> column_ids;
	vector<ColumnIndex> column_indices;

	//! Buffer manager allocator
	Allocator &allocator;
	//! The current buffer capacity
	idx_t buffer_capacity;

	//! Current number of threads active
	idx_t system_threads;
	//! Whether we enable parallel scans (only if less files than threads)
	bool enable_parallel_scans;

	bool file_is_assigned = false;
	bool initialized = false;
};

struct JSONScanLocalState {
public:
	JSONScanLocalState(ClientContext &context, JSONScanGlobalState &gstate);

public:
	idx_t Read();
	void AddTransformError(idx_t object_index, const string &error_message);

	JSONReaderScanState &GetScanState() {
		return scan_state;
	}

	const JSONReaderScanState &GetScanState() const {
		return scan_state;
	}

	bool TryInitializeScan(JSONScanGlobalState &gstate, JSONReader &reader);

public:
	//! Options when transforming the JSON to columnar data
	JSONTransformOptions transform_options;

private:
	void ParseJSON(char *const json_start, const idx_t json_size, const idx_t remaining);

private:
	//! Scan state
	JSONReaderScanState scan_state;
};

struct JSONGlobalTableFunctionState : public GlobalTableFunctionState {
public:
	JSONGlobalTableFunctionState(ClientContext &context, const MultiFileBindData &bind_data);

public:
	JSONScanGlobalState state;
};

struct JSONLocalTableFunctionState : public LocalTableFunctionState {
public:
	JSONLocalTableFunctionState(ClientContext &context, JSONScanGlobalState &gstate);

public:
	JSONScanLocalState state;
};

struct JSONScan {
public:
	static void AutoDetect(ClientContext &context, MultiFileBindData &bind_data, vector<LogicalType> &return_types,
	                       vector<string> &names);

	static void Serialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data,
	                      const TableFunction &function);
	static unique_ptr<FunctionData> Deserialize(Deserializer &deserializer, TableFunction &function);

	static void TableFunctionDefaults(TableFunction &table_function);
};

} // namespace duckdb
