// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#pragma once

#include "vane_python/pybind11/pybind_wrapper.hpp"
#include "vane_python/pybind11/python_object_container.hpp"
#include "vane_python/numpy/numpy_type.hpp"
#include "duckdb/common/helper.hpp"
#include "vane_python/pandas/pandas_column.hpp"

namespace duckdb {

class ClientContext;

struct RegisteredArray {
	explicit RegisteredArray(py::array numpy_array) : numpy_array(std::move(numpy_array)) {
	}
	py::array numpy_array;
};

struct PandasColumnBindData {
	NumpyType numpy_type;
	unique_ptr<PandasColumn> pandas_col;
	unique_ptr<RegisteredArray> mask;
	//! Only for categorical types
	string internal_categorical_type;
	//! Hold ownership of objects created during scanning
	PythonObjectContainer object_str_val;
};

struct Pandas {
	static void Bind(const ClientContext &config, py::handle df, vector<PandasColumnBindData> &out,
	                 vector<LogicalType> &return_types, vector<string> &names);
};

} // namespace duckdb
