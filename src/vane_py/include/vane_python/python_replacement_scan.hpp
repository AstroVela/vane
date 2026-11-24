// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#pragma once

#include "duckdb/main/client_context_state.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/parser/tableref.hpp"
#include "duckdb/function/replacement_scan.hpp"
#include "vane_python/pybind11/pybind_wrapper.hpp"

namespace duckdb {

//! Keep the caller's lookup frame while native binding crosses Python admission.
//! The scope is thread-local and applies only to its connection. Requires the GIL.
class ScopedPythonReplacementScanFrame {
public:
	explicit ScopedPythonReplacementScanFrame(const ClientContext &context_p);
	~ScopedPythonReplacementScanFrame();
	ScopedPythonReplacementScanFrame(const ScopedPythonReplacementScanFrame &) = delete;
	ScopedPythonReplacementScanFrame &operator=(const ScopedPythonReplacementScanFrame &) = delete;

	//! An empty handle means no override; None means the caller had no Python frame.
	static py::object GetFrame(const ClientContext &context);

private:
	const ClientContext &context;
	py::object frame;
	ScopedPythonReplacementScanFrame *previous;
	static thread_local ScopedPythonReplacementScanFrame *current;
};

struct PythonReplacementScan {
public:
	static unique_ptr<TableRef> Replace(ClientContext &context, ReplacementScanInput &input,
	                                    optional_ptr<ReplacementScanData> data);
	//! Try to perform a replacement, returns NULL on error
	static unique_ptr<TableRef> TryReplacementObject(const py::object &entry, const string &name,
	                                                 ClientContext &context, bool relation = false);
	//! Perform a replacement or throw if it failed
	static unique_ptr<TableRef> ReplacementObject(const py::object &entry, const string &name, ClientContext &context,
	                                              bool relation = false);
};

} // namespace duckdb
