//===----------------------------------------------------------------------===//
#include "duckdb_python/pybind11/gil_wrapper.hpp"
//                         DuckDB
//
// duckdb_python/pybind11/registered_py_object.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once
#include "duckdb_python/pybind11/pybind_wrapper.hpp"

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
