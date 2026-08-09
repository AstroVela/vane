// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#pragma once

#include "duckdb/common/common.hpp"
#include "vane_python/pybind11/pybind_wrapper.hpp"
#include "duckdb/main/external_dependencies.hpp"
#include "duckdb/common/types/value.hpp"

namespace duckdb {

struct DuckDBPyConnection;

struct PathLike {
	static PathLike Create(const py::object &object, DuckDBPyConnection &connection);
	// The file(s) extracted from object
	vector<string> files;
	shared_ptr<ExternalDependency> dependency;
};

} // namespace duckdb
