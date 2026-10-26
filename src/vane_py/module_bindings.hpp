// SPDX-FileCopyrightText: 2018-2026 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#pragma once

#include "vane_python/pybind11/pybind_wrapper.hpp"

namespace duckdb {

// Register module-level wrappers in their original order, including overloads.
void InitializeModuleConnectionBindings(py::module_ &m);
void InitializeModuleTypesBindings(py::module_ &m);
void InitializeModuleExecutionBindings(py::module_ &m);
void InitializeModuleResultsBindings(py::module_ &m);
void InitializeModuleTransactionsBindings(py::module_ &m);
void InitializeModuleRelationsBindings(py::module_ &m);
void InitializeModuleExtensionsBindings(py::module_ &m);
void InitializeModuleRelationalOperationsBindings(py::module_ &m);
void InitializeModuleResultPropertiesBindings(py::module_ &m);

} // namespace duckdb
