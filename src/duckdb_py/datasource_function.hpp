#pragma once

#include "duckdb_python/pybind11/pybind_wrapper.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/function/table/datasource_scan.hpp"

namespace py = pybind11;

namespace duckdb {

//! Factory that bridges Python DataSource tasks → ArrowArrayStream.
//! Lives in the Python binding layer (pybind11 allowed here).
//! Core datasource_scan.cpp only sees C function pointers.
struct DataSourceStreamFactory {
	//! Keep the DataSource Python object alive
	py::object datasource_obj;
	//! Pickled task bytes — one per task
	vector<string> pickled_tasks;
	//! Arrow schema (PyArrow Schema object, kept alive)
	py::object arrow_schema;

	DataSourceStreamFactory(py::object datasource, py::object schema, vector<string> tasks)
	    : datasource_obj(std::move(datasource)), pickled_tasks(std::move(tasks)), arrow_schema(std::move(schema)) {
	}

	//! C callback: unpickle task → task.execute() → RecordBatchReader → _export_to_c
	//! Called from pipeline threads — each call creates an independent stream.
	//! The factory pointer is passed indirectly: the pickled_task bytes include a
	//! prefix with the factory pointer (see ProduceStreamWithFactory).
	static void ProduceStream(const char *pickled_task, idx_t pickled_len, ArrowArrayStream *out_stream);

	//! C callback: export the cached Arrow schema to ArrowSchema
	static void GetSchema(const char *pickled_source, idx_t pickled_len, ArrowSchema *out_schema);

	//! Set the worker-side pickled source for factory recovery (C callback for datasource_set_worker_source_t)
	static void SetWorkerPickledSource(const char *pickled_source, idx_t pickled_len);
};

//! Clear all factory references to prevent segfault during Python shutdown.
//! Must be called before Python interpreter finalizes.
void ClearDataSourceFactoryRegistry();

//! Register global callbacks (ProduceStream, SetWorkerPickledSource) on DataSourceScanFunction.
//! Call once from Python module init.
void RegisterDataSourceGlobalCallbacks();

} // namespace duckdb
