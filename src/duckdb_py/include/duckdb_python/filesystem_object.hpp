// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
#include "duckdb_python/pybind11/gil_wrapper.hpp"
//                         DuckDB
//
// duckdb_python/filesystem_object.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once
#include "duckdb_python/pybind11/registered_py_object.hpp"
#include "duckdb_python/pyfilesystem.hpp"

namespace duckdb {

class FileSystemObject : public RegisteredObject {
public:
	explicit FileSystemObject(py::object fs, vector<string> filenames_p)
	    : RegisteredObject(std::move(fs)), filenames(std::move(filenames_p)) {
	}
	~FileSystemObject() override {
		PythonGILWrapper acquire;
		// Assert that the 'obj' is a filesystem
		D_ASSERT(py::isinstance(obj, DuckDBPyConnection::ImportCache()->vane.filesystem.ModifiedMemoryFileSystem()));
		for (auto &file : filenames) {
			obj.attr("delete")(file);
		}
	}

	vector<string> filenames;
};

} // namespace duckdb
