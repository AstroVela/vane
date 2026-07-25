// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

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
	//! Arrow schema (PyArrow Schema object, kept alive)
	py::object arrow_schema;

	explicit DataSourceStreamFactory(py::object schema) : arrow_schema(std::move(schema)) {
	}

	//! C callback: unpickle task → task.execute() → RecordBatchReader → _export_to_c
	//! Called from pipeline threads — each call creates an independent stream.
	static void ProduceStream(const char *pickled_task, idx_t pickled_len, ArrowArrayStream *out_stream);

	//! C callback: export the serialized or cached Arrow schema to ArrowSchema
	static void GetSchema(const char *pickled_source, idx_t pickled_len, ArrowSchema *out_schema);

	//! Acquire/release one query execution owner for the logical source.
	static void AcquireSource(const char *pickled_source, idx_t pickled_len, const char *query_id, idx_t query_id_len);
	static void ReleaseSource(const char *pickled_source, idx_t pickled_len) noexcept;
};

//! Clear all factory references to prevent segfault during Python shutdown.
//! Must be called before Python interpreter finalizes.
void ClearDataSourceFactoryRegistry();

//! Release every source owned by a completed distributed query.
idx_t ReleaseDataSourceFactoriesForQuery(const string &query_id);

//! Return process-local registry diagnostics used by lifecycle regression tests.
py::dict DataSourceFactoryRegistryStateForTest();

//! Register global callbacks on DataSourceScanFunction.
//! Call once from Python module init.
void RegisterDataSourceGlobalCallbacks();

} // namespace duckdb
