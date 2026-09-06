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
#include "duckdb/function/built_in_functions.hpp"
#include "duckdb/function/table/arrow.hpp"

namespace duckdb {

//! C callback type: given pickled task bytes, produce an ArrowArrayStream
//! The callback must:
//!   1. Unpickle the bytes into a DataSourceTask object
//!   2. Call the task's context-aware execution hook to get a generator
//!   3. Wrap the generator into a RecordBatchReader
//!   4. Export via _export_to_c into the ArrowArrayStream
typedef void (*datasource_produce_stream_t)(const char *pickled_task, idx_t pickled_len, ArrowArrayStream *out_stream,
                                            ClientContext *execution_context);

//! C callback type: given a serialized logical source package, produce the Arrow schema
typedef void (*datasource_get_schema_t)(const char *pickled_source, idx_t pickled_len, ArrowSchema *out_schema);

//! C callback types: acquire/release the process-local factory for a logical source.
//! Acquire must complete before ProduceStream is called. An empty query_id is
//! released with the scan global state; a non-empty query_id transfers release
//! ownership to the distributed query teardown path.
typedef void (*datasource_acquire_source_t)(const char *pickled_source, idx_t pickled_len, const char *query_id,
                                            idx_t query_id_len);
typedef void (*datasource_release_source_t)(const char *pickled_source, idx_t pickled_len);

struct DataSourceScanBindData : public TableFunctionData {
	//! Pickled DataSourceTask objects, one per task
	vector<string> pickled_tasks;
	//! Serialized logical source package (for schema extraction on workers)
	string pickled_source;
	//! Distributed query owner. Empty for ordinary connection-local scans.
	string query_id;
	//! Callback to produce ArrowArrayStream from a pickled task
	datasource_produce_stream_t produce_stream;
	//! Arrow schema metadata
	ArrowTableSchema arrow_table;
	//! Optional source cardinality captured by a planner-side rewrite.
	optional_idx estimated_cardinality;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<DataSourceScanBindData>();
		result->pickled_tasks = pickled_tasks;
		result->pickled_source = pickled_source;
		result->query_id = query_id;
		result->produce_stream = produce_stream;
		result->arrow_table = arrow_table;
		result->estimated_cardinality = estimated_cardinality;
		return std::move(result);
	}
};

struct DataSourceScanGlobalState : public GlobalTableFunctionState {
	~DataSourceScanGlobalState() override;

	//! Next task index (atomic for thread-safe work-stealing)
	atomic<idx_t> next_task_idx {0};
	//! Total tasks
	idx_t total_tasks = 0;
	//! Connection-local source ownership acquired for this execution.
	datasource_release_source_t release_source = nullptr;
	string pickled_source;
	bool release_source_on_destroy = false;

	idx_t MaxThreads() const override {
		return total_tasks;
	}
};

struct DataSourceScanLocalState : public LocalTableFunctionState {
	enum class ScanState {
		NEED_TASK,
		NEED_BATCH,
		SCANNING,
		EXHAUSTED,
	};

	explicit DataSourceScanLocalState(ClientContext &context) : scan_state(make_uniq<ArrowArrayWrapper>(), context) {
	}

	//! Per-thread arrow stream (one per task)
	unique_ptr<ArrowArrayStreamWrapper> stream;
	//! Current Arrow batch and conversion offset, retained across output vectors
	ArrowScanLocalState scan_state;
	//! Explicit scan state for task, batch, and output-vector transitions
	ScanState state = ScanState::NEED_TASK;
};

struct DataSourceScanFunction {
	static TableFunction GetFunction();
	static void RegisterFunction(BuiltinFunctions &set);

	//! Register a global produce_stream callback for use on distributed workers.
	//! Should be called once when the Python module loads.
	static void SetGlobalProduceStream(datasource_produce_stream_t callback);
	//! Get the global produce_stream callback (returns nullptr if not set)
	static datasource_produce_stream_t GetGlobalProduceStream();
	//! Register global source ownership callbacks.
	static void SetGlobalAcquireSource(datasource_acquire_source_t callback);
	static void SetGlobalReleaseSource(datasource_release_source_t callback);
	//! Register a global get_schema callback for worker schema restoration
	static void SetGlobalGetSchema(datasource_get_schema_t callback);
};

} // namespace duckdb
