// SPDX-FileCopyrightText: 2018-2026 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#include "module_bindings.hpp"
#include "vane_python/pyconnection/pyconnection.hpp"
#include "vane_python/pybind11/conversions/pyconnection_default.hpp"

namespace duckdb {

void InitializeModuleConnectionBindings(py::module_ &m) {
	m.def(
	    "cursor",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Cursor();
	    },
	    "Create a duplicate of the current connection", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "register_filesystem",
	    [](AbstractFileSystem filesystem, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->RegisterFilesystem(filesystem);
	    },
	    "Register a fsspec compliant filesystem", py::arg("filesystem"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "unregister_filesystem",
	    [](const py::str &name, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->UnregisterFilesystem(name);
	    },
	    "Unregister a filesystem", py::arg("name"), py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "list_filesystems",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->ListFilesystems();
	    },
	    "List registered filesystems, including builtin ones", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "filesystem_is_registered",
	    [](const string &name, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->FileSystemIsRegistered(name);
	    },
	    "Check if a filesystem with the provided name is currently registered", py::arg("name"), py::kw_only(),
	    py::arg("connection") = py::none());
	m.def(
	    "get_profiling_information",
	    [](const py::str &format, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->GetProfilingInformation(format);
	    },
	    "Get profiling information from a query", py::kw_only(), py::arg("format") = "json",
	    py::arg("connection") = py::none());
	m.def(
	    "enable_profiling",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->EnableProfiling();
	    },
	    "Enable profiling for the current connection", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "disable_profiling",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->DisableProfiling();
	    },
	    "Disable profiling for the current connection", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "remove_function",
	    [](const string &name, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->UnregisterUDF(name);
	    },
	    "Remove a previously created function", py::arg("name"), py::kw_only(), py::arg("connection") = py::none());
}

void InitializeModuleExecutionBindings(py::module_ &m) {
	m.def(
	    "duplicate",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Cursor();
	    },
	    "Create a duplicate of the current connection", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "execute",
	    [](const py::object &query, py::object params = py::list(), shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Execute(query, params);
	    },
	    "Execute the given SQL query, optionally using prepared statements with parameters set", py::arg("query"),
	    py::arg("parameters") = py::none(), py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "executemany",
	    [](const py::object &query, py::object params = py::list(), shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->ExecuteMany(query, params);
	    },
	    "Execute the given prepared statement multiple times using the list of parameter sets in parameters",
	    py::arg("query"), py::arg("parameters") = py::none(), py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "close",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->Close();
	    },
	    "Close the connection", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "interrupt",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->Interrupt();
	    },
	    "Interrupt pending operations", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "query_progress",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->QueryProgress();
	    },
	    "Query progress of pending operation", py::kw_only(), py::arg("connection") = py::none());
}

void InitializeModuleTransactionsBindings(py::module_ &m) {
	m.def(
	    "begin",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Begin();
	    },
	    "Start a new transaction", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "commit",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Commit();
	    },
	    "Commit changes performed within a transaction", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "rollback",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Rollback();
	    },
	    "Roll back changes performed within a transaction", py::kw_only(), py::arg("connection") = py::none());
	m.def(
	    "checkpoint",
	    [](shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    return conn->Checkpoint();
	    },
	    "Synchronizes data in the write-ahead log (WAL) to the database data file (no-op for in-memory connections)",
	    py::kw_only(), py::arg("connection") = py::none());
}

void InitializeModuleExtensionsBindings(py::module_ &m) {
	m.def(
	    "install_extension",
	    [](const string &extension, bool force_install = false, const py::object &repository = py::none(),
	       const py::object &repository_url = py::none(), const py::object &version = py::none(),
	       shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->InstallExtension(extension, force_install, repository, repository_url, version);
	    },
	    "Install an extension by name, with an optional version and/or repository to get the extension from",
	    py::arg("extension"), py::kw_only(), py::arg("force_install") = false, py::arg("repository") = py::none(),
	    py::arg("repository_url") = py::none(), py::arg("version") = py::none(), py::arg("connection") = py::none());
	m.def(
	    "load_extension",
	    [](const string &extension, shared_ptr<DuckDBPyConnection> conn = nullptr) {
		    if (!conn) {
			    conn = DuckDBPyConnection::DefaultConnection();
		    }
		    conn->LoadExtension(extension);
	    },
	    "Load an installed extension", py::arg("extension"), py::kw_only(), py::arg("connection") = py::none());
}

} // namespace duckdb
