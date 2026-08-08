// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#include "duckdb_python/functional.hpp"

namespace duckdb {

void DuckDBPyFunctional::Initialize(py::module_ &parent) {
	auto m = parent.def_submodule("_func", "This module contains classes and methods related to functions and udf");

	py::enum_<duckdb::PythonUDFType>(m, "PythonUDFType", py::module_local())
	    .value("NATIVE", duckdb::PythonUDFType::NATIVE)
	    .value("ARROW", duckdb::PythonUDFType::ARROW)
	    .export_values();

	py::enum_<duckdb::FunctionNullHandling>(m, "FunctionNullHandling", py::module_local())
	    .value("DEFAULT", duckdb::FunctionNullHandling::DEFAULT_NULL_HANDLING)
	    .value("SPECIAL", duckdb::FunctionNullHandling::SPECIAL_HANDLING)
	    .export_values();
}

} // namespace duckdb
