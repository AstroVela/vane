// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vane_python/pybind11/pybind_wrapper.hpp"
#include "duckdb/common/types/value.hpp"

namespace duckdb {

class Vector;

//! Python boundary conversion; IMAGE cells are NumPy arrays, not value wrappers.
struct PythonImage {
	static bool IsPIL(const py::handle &value);
	static Value FromPython(const py::handle &value, const LogicalType &type = ImageLogicalType::Create());
	static void ToVector(const py::handle &value, Vector &result, idx_t row);
	static py::object FromValue(const Value &value);
};

} // namespace duckdb
