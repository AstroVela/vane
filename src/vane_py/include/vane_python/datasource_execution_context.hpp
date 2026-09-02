// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vane_python/pybind11/pybind_wrapper.hpp"

#include "duckdb/common/common.hpp"

namespace duckdb {

class ClientContext;

//! An internal, non-serializable handle to the ClientContext executing a
//! Python DataSource task. It lets task-owned readers use the exact worker
//! query context without constructing or consulting a process-default
//! Python connection.
class PythonDataSourceExecutionContext final {
public:
	explicit PythonDataSourceExecutionContext(shared_ptr<ClientContext> context_p);

	static void Initialize(py::module_ &m);
	void CheckInterrupted() const;
	shared_ptr<ClientContext> GetContext() const;

private:
	shared_ptr<ClientContext> context;
};

} // namespace duckdb
