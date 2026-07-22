// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/import_cache/python_import_cache.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb_python/pybind11/pybind_wrapper.hpp"
#include "duckdb.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb_python/import_cache/python_import_cache_modules.hpp"

namespace duckdb {

struct PythonImportCache {
public:
	explicit PythonImportCache() {
	}
	~PythonImportCache();

public:
	PyarrowCacheItem pyarrow;
	PandasCacheItem pandas;
	DatetimeCacheItem datetime;
	DecimalCacheItem decimal;
	IpythonCacheItem IPython;
	IpywidgetsCacheItem ipywidgets;
	NumpyCacheItem numpy;
	PathlibCacheItem pathlib;
	PolarsCacheItem polars;
	VaneCacheItem vane;
	PytzCacheItem pytz;
	TypesCacheItem types;
	TypingCacheItem typing;
	UuidCacheItem uuid;
	CollectionsCacheItem collections;

public:
	py::handle AddCache(py::object item);

private:
	vector<py::object> owned_objects;
};

} // namespace duckdb
