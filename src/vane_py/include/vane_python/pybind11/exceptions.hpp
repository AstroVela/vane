// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#include "vane_python/pybind11/pybind_wrapper.hpp"

namespace py = pybind11;

namespace duckdb {

void RegisterExceptions(const py::module &m);

} // namespace duckdb
