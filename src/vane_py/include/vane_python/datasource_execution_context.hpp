// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vane_python/pybind11/pybind_wrapper.hpp"

#include "duckdb/common/common.hpp"

#include <atomic>
#include <mutex>

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
	void Invalidate();
	std::unique_lock<std::mutex> LockContext(shared_ptr<ClientContext> &active_context) const;

private:
	std::atomic<bool> active {true};
	mutable std::mutex context_lock;
	shared_ptr<ClientContext> context;
};

} // namespace duckdb
