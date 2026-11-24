// SPDX-FileCopyrightText: 2018-2026 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#include "vane_python/pyconnection/pyconnection.hpp"
#include "duckdb/main/database.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"
#include "vane_python/python_input_callback.hpp"
#include "vane_python/vane_fs.hpp"
#include <algorithm>

namespace duckdb {

void DuckDBPyConnection::UnregisterFilesystem(const py::str &name) {
	auto query_lock = LockForQuery();
	const string subsystem = name;
	py::gil_scoped_release release;
	unique_ptr<FileSystem> removed;
	{
		lock_guard<mutex> registration_guard(VaneFSRegistrationLock());
		removed = con.GetDatabase().GetFileSystem().ExtractSubSystem(subsystem);
	}
	// Destroy Python objects after releasing the registration lock.
}

void DuckDBPyConnection::RegisterFilesystem(AbstractFileSystem filesystem) {
	auto query_lock = LockForQuery();
	// Protocol and capability properties are user metadata callbacks too.
	PythonInputCallbackScope callback(nullptr);
	PythonGILWrapper gil_wrapper;

	if (!py::isinstance<AbstractFileSystem>(filesystem)) {
		throw InvalidInputException("Bad filesystem instance");
	}

	auto protocol = filesystem.attr("protocol");
	if (protocol.is_none() || py::str("abstract").equal(protocol)) {
		throw InvalidInputException("Must provide concrete fsspec implementation");
	}

	vector<string> protocols;
	if (py::isinstance<py::str>(protocol)) {
		protocols.push_back(py::str(protocol));
	} else {
		for (const auto &sub_protocol : protocol) {
			protocols.push_back(py::str(sub_protocol));
		}
	}

	// fsspec does not expose a general capability that distinguishes concrete
	// directories from object-store prefixes. Its LocalFileSystem hierarchy has
	// concrete directory semantics; other hierarchical implementations can opt
	// in explicitly without relying on protocol-name heuristics.
	bool directory_semantics;
	py::object directory_capability;
	try {
		directory_capability = filesystem.attr("vane_directory_semantics");
	} catch (py::error_already_set &error) {
		// hasattr() suppresses all callback errors, including reentry rejection.
		if (!error.matches(PyExc_AttributeError)) {
			throw;
		}
	}
	if (directory_capability) {
		directory_semantics = py::bool_(directory_capability);
	} else {
		auto local_filesystem = py::module::import("fsspec.implementations.local").attr("LocalFileSystem");
		directory_semantics = py::isinstance(filesystem, local_filesystem);
	}

	py::gil_scoped_release release;
	lock_guard<mutex> registration_guard(VaneFSRegistrationLock());
	auto &fs = con.GetDatabase().GetFileSystem();
	for (const auto &name : protocols) {
		if (name == "vanefs_native") {
			throw InvalidInputException("The vanefs_native filesystem name is reserved for the native VaneFS reader");
		}
		if (name == "vanefs") {
			for (const auto &registered : fs.ListSubSystems()) {
				if (registered == "vanefs_native") {
					throw InvalidInputException(
					    "Python and native VaneFS filesystems require separate Vane database instances");
				}
			}
		}
	}
	fs.RegisterSubSystem(make_uniq<PythonFilesystem>(std::move(protocols), std::move(filesystem), directory_semantics));
}

static vector<string> ReadFileSystemNames(DuckDBPyConnection &connection) {
	// Callers hold the connection lock. ExtractSubSystem moves a pointer
	// shared by registry snapshots, so copy names under the registration lock.
	py::gil_scoped_release release;
	lock_guard<mutex> registration_guard(VaneFSRegistrationLock());
	return connection.con.GetDatabase().GetFileSystem().ListSubSystems();
}

py::list DuckDBPyConnection::ListFilesystems() {
	auto query_lock = LockForQuery();
	auto subsystems = ReadFileSystemNames(*this);
	py::list names;
	for (auto &name : subsystems) {
		names.append(py::str(name));
	}
	return names;
}

bool DuckDBPyConnection::FileSystemIsRegistered(const string &name) {
	auto query_lock = LockForQuery();
	auto subsystems = ReadFileSystemNames(*this);
	return std::find(subsystems.begin(), subsystems.end(), name) != subsystems.end();
}

PathLike DuckDBPyConnection::GetPathLike(const py::object &object) {
	return PathLike::Create(object, *this);
}

ModifiedMemoryFileSystem &DuckDBPyConnection::GetObjectFileSystem() {
	if (!internal_object_filesystem) {
		D_ASSERT(!FileSystemIsRegistered("VANE_INTERNAL_OBJECTSTORE"));
		auto &import_cache_py = *ImportCache();
		auto modified_memory_fs = import_cache_py.vane.filesystem.ModifiedMemoryFileSystem();
		if (modified_memory_fs.ptr() == nullptr) {
			throw InvalidInputException(
			    "This operation could not be completed because required module 'fsspec' is not installed");
		}
		internal_object_filesystem = make_shared_ptr<ModifiedMemoryFileSystem>(modified_memory_fs());
		auto &abstract_fs = reinterpret_cast<AbstractFileSystem &>(*internal_object_filesystem);
		RegisterFilesystem(abstract_fs);
	}
	return *internal_object_filesystem;
}

} // namespace duckdb
