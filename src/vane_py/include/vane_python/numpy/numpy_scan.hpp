// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#pragma once

#include "vane_python/pybind11/pybind_wrapper.hpp"
#include "duckdb/common/common.hpp"

namespace duckdb {

struct PandasColumnBindData;

struct NumpyScan {
	static void Scan(PandasColumnBindData &bind_data, idx_t count, idx_t offset, Vector &out);
	static void ScanObjectColumn(PyObject **col, idx_t stride, idx_t count, idx_t offset, Vector &out);
};

} // namespace duckdb
