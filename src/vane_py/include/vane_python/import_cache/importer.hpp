// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// vane_python/import_cache/python_import_cache.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "vane_python/pybind11/pybind_wrapper.hpp"
#include "duckdb.hpp"
#include "duckdb/common/vector.hpp"
#include "vane_python/import_cache/python_import_cache_modules.hpp"
#include "duckdb/common/stack.hpp"

namespace duckdb {

struct PythonImporter {
public:
	static py::handle Import(stack<reference<PythonImportCacheItem>> &hierarchy, bool load = true);
};

} // namespace duckdb
