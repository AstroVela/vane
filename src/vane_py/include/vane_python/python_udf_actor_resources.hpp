// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/common/shared_ptr.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/vector.hpp"
#include <pybind11/pybind11.h>

namespace duckdb {

class ClientContext;
class ClientContextState;
class PythonUDFActorResourceState;
class PreparedStatementData;

//! Read native topology through the common metadata collector without taking plan ownership.
pybind11::dict CollectNativeLocalResourceGraph(ClientContext &context, PreparedStatementData &prepared);

//! Apply the input policy during binding too, before a runtime request exists.
//! Both functions require the GIL and do not acquire the native context lock.
void EnableLocalRuntimeInputPolicy(ClientContext &context);
bool HasLocalRuntimeInputPolicy(const ClientContext &context);

class ScopedPythonUDFActorResourcePreparation {
public:
	explicit ScopedPythonUDFActorResourcePreparation(ClientContext &context,
	                                                 pybind11::object local_query = pybind11::none());
	~ScopedPythonUDFActorResourcePreparation();
	vector<string> TakeCleanupWarnings();

	ScopedPythonUDFActorResourcePreparation(const ScopedPythonUDFActorResourcePreparation &) = delete;
	ScopedPythonUDFActorResourcePreparation &operator=(const ScopedPythonUDFActorResourcePreparation &) = delete;

private:
	shared_ptr<PythonUDFActorResourceState> state;
};

} // namespace duckdb
