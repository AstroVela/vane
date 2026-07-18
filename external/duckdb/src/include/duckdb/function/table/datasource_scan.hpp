// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/table/datasource_scan.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/function/table_function.hpp"
#include "duckdb/common/atomic.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/function/table/arrow/arrow_duck_schema.hpp"
#include "duckdb/function/built_in_functions.hpp"
#include "duckdb/function/extension_file_list_provider.hpp"

namespace duckdb {

//! C callback type: given pickled task bytes, produce an ArrowArrayStream
//! The callback must:
//!   1. Unpickle the bytes into a DataSourceTask object
//!   2. Call task.execute() to get a generator
//!   3. Wrap the generator into a RecordBatchReader
//!   4. Export via _export_to_c into the ArrowArrayStream
typedef void (*datasource_produce_stream_t)(const char *pickled_task, idx_t pickled_len, ArrowArrayStream *out_stream);

//! C callback type: given pickled DataSource bytes, produce the Arrow schema
typedef void (*datasource_get_schema_t)(const char *pickled_source, idx_t pickled_len, ArrowSchema *out_schema);

//! C callback type: set the pickled source for worker-side factory recovery
typedef void (*datasource_set_worker_source_t)(const char *pickled_source, idx_t pickled_len);

struct DataSourceScanBindData : public TableFunctionData, public ExtensionFileListProvider {
	//! Pickled DataSourceTask objects, one per task
	vector<string> pickled_tasks;
	//! Pickled DataSource object (for schema extraction on deserialize)
	string pickled_source;
	//! Callback to produce ArrowArrayStream from a pickled task
	datasource_produce_stream_t produce_stream;
	//! Arrow schema metadata
	ArrowTableSchema arrow_table;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<DataSourceScanBindData>();
		result->pickled_tasks = pickled_tasks;
		result->pickled_source = pickled_source;
		result->produce_stream = produce_stream;
		result->arrow_table = arrow_table;
		return std::move(result);
	}

	//! ExtensionFileListProvider: encode each pickled task as a fake file path
	vector<string> GetFileList() const override;

	//! ExtensionFileListProvider: decode fake file paths back to pickled tasks
	void SetFileList(const vector<string> &files) override;
};

struct DataSourceScanGlobalState : public GlobalTableFunctionState {
	//! Next task index (atomic for thread-safe work-stealing)
	atomic<idx_t> next_task_idx {0};
	//! Total tasks
	idx_t total_tasks = 0;

	idx_t MaxThreads() const override {
		return total_tasks;
	}
};

struct DataSourceScanLocalState : public LocalTableFunctionState {
	//! Per-thread arrow stream (one per task)
	unique_ptr<ArrowArrayStreamWrapper> stream;
	//! Whether this thread is done
	bool exhausted = false;
};

struct DataSourceScanFunction {
	static TableFunction GetFunction();
	static void RegisterFunction(BuiltinFunctions &set);

	//! Register a global produce_stream callback for use on distributed workers.
	//! Should be called once when the Python module loads.
	static void SetGlobalProduceStream(datasource_produce_stream_t callback);
	//! Get the global produce_stream callback (returns nullptr if not set)
	static datasource_produce_stream_t GetGlobalProduceStream();
	//! Register a global set_worker_source callback
	static void SetGlobalWorkerSourceCallback(datasource_set_worker_source_t callback);
	//! Register a global get_schema callback for worker schema restoration
	static void SetGlobalGetSchema(datasource_get_schema_t callback);
};

} // namespace duckdb
