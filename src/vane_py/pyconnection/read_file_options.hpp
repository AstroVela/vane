// SPDX-FileCopyrightText: 2018-2026 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#pragma once

#include "vane_python/pyconnection/pyconnection.hpp"

namespace duckdb {

// Shared CSV/JSON option conversion, implemented in read_json.cpp.
void ParseMultiFileOptions(named_parameter_map_t &options, const Optional<py::object> &filename,
                           const Optional<py::object> &hive_partitioning, const Optional<py::object> &union_by_name,
                           const Optional<py::object> &hive_types, const Optional<py::object> &hive_types_autocast);

} // namespace duckdb
