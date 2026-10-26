// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vane_python/pyconnection/pyconnection.hpp"
#include "duckdb/main/client_context_state.hpp"

namespace duckdb {
void InitializeVaneFS(py::class_<DuckDBPyConnection, shared_ptr<DuckDBPyConnection>> &connection);

struct VaneFSQueryPins;
shared_ptr<VaneFSQueryPins> CreateVaneFSQueryPins(const shared_ptr<VaneFSQueryPins> &source = nullptr);
py::list CaptureVaneFSWorkspaces(ClientContext &context, const shared_ptr<VaneFSQueryPins> &pins);
void ApplyVaneFSWorkspaces(DuckDBPyConnection &connection, const py::object &workspaces,
                           const shared_ptr<VaneFSQueryPins> &pins);

// Binding can open files before a native query starts. Transfer those pins to
// the distributed plan before the binding transaction releases its own pins.
class VaneFSQueryPinScope {
public:
	explicit VaneFSQueryPinScope(ClientContext &context);
	~VaneFSQueryPinScope();
	VaneFSQueryPinScope(const VaneFSQueryPinScope &) = delete;
	VaneFSQueryPinScope &operator=(const VaneFSQueryPinScope &) = delete;
	shared_ptr<VaneFSQueryPins> Pins() const {
		return pins;
	}

private:
	shared_ptr<ClientContextState> state;
	shared_ptr<VaneFSQueryPins> pins;
	weak_ptr<VaneFSQueryPins> previous;
};

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
