// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vane_python/pyconnection/pyconnection.hpp"

namespace duckdb {
void InitializeVaneFS(py::class_<DuckDBPyConnection, shared_ptr<DuckDBPyConnection>> &connection);

// Serialize native/Python filesystem registration and introspection across connections.
// Callers must release the GIL before acquiring this lock.
mutex &VaneFSRegistrationLock();

// Public File.open owns its snapshot through the returned reader, even if an
// unrelated streaming SQL result is still active on the same connection.
class VaneFSStandaloneOpenScope {
public:
	explicit VaneFSStandaloneOpenScope(ClientContext &context);
	~VaneFSStandaloneOpenScope();
	VaneFSStandaloneOpenScope(const VaneFSStandaloneOpenScope &) = delete;
	VaneFSStandaloneOpenScope &operator=(const VaneFSStandaloneOpenScope &) = delete;

private:
	ClientContext *previous;
};
} // namespace duckdb
