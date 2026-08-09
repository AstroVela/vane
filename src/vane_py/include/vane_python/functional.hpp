// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#pragma once

#include "vane_python/pybind11/pybind_wrapper.hpp"
#include "vane_python/pytype.hpp"
#include "vane_python/pyconnection/pyconnection.hpp"

namespace duckdb {

class DuckDBPyFunctional {
public:
	DuckDBPyFunctional() = delete;

public:
	static void Initialize(py::module_ &m);
};

} // namespace duckdb
