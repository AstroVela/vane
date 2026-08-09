// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
#include "vane_python/pybind11/gil_wrapper.hpp"
//                         DuckDB
//
// vane_python/pybind11/registered_py_object.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once
#include "vane_python/pybind11/pybind_wrapper.hpp"

namespace duckdb {

class RegisteredObject {
public:
	explicit RegisteredObject(py::object obj_p) : obj(std::move(obj_p)) {
	}
	virtual ~RegisteredObject() {
		if (!obj) {
			return;
		}
		PythonGILWrapper acquire;
		obj = py::object();
	}

	py::object obj;
};

} // namespace duckdb
