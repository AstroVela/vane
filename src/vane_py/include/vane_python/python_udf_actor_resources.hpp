// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/common/shared_ptr.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {

class ClientContext;
class ClientContextState;
class PythonUDFActorResourceState;

class ScopedPythonUDFActorResourcePreparation {
public:
	explicit ScopedPythonUDFActorResourcePreparation(ClientContext &context);
	~ScopedPythonUDFActorResourcePreparation();
	vector<string> TakeCleanupWarnings();

	ScopedPythonUDFActorResourcePreparation(const ScopedPythonUDFActorResourcePreparation &) = delete;
	ScopedPythonUDFActorResourcePreparation &operator=(const ScopedPythonUDFActorResourcePreparation &) = delete;

private:
	shared_ptr<PythonUDFActorResourceState> state;
};

} // namespace duckdb
