// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/common/shared_ptr.hpp"

namespace duckdb {

class ClientContext;
class ClientContextState;
class PythonUDFActorResourceState;

class ScopedPythonUDFActorResourcePreparation {
public:
	explicit ScopedPythonUDFActorResourcePreparation(ClientContext &context);
	~ScopedPythonUDFActorResourcePreparation();

	ScopedPythonUDFActorResourcePreparation(const ScopedPythonUDFActorResourcePreparation &) = delete;
	ScopedPythonUDFActorResourcePreparation &operator=(const ScopedPythonUDFActorResourcePreparation &) = delete;

private:
	shared_ptr<PythonUDFActorResourceState> state;
};

} // namespace duckdb
