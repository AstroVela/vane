// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "duckdb/function/scalar/udf_functions.hpp"

namespace duckdb {

//! Vane's connection policy uses DuckDB's native state registry. Connections
//! created directly by the engine retain native execution without Vane state.
class RunnerClientState : public ClientContextState {
public:
	explicit RunnerClientState(string runner_p) : runner(std::move(runner_p)) {
	}

	static void Initialize(ClientContext &context, const string &runner_type) {
		auto normalized = NormalizeRunnerType(runner_type);
		if (normalized.empty()) {
			normalized = "ray";
		}
		auto state = context.registered_state->GetOrCreate<RunnerClientState>("vane.runner", normalized);
		if (state->runner != normalized) {
			throw InvalidInputException("A connection's runner cannot change after creation");
		}
	}

	static string Get(ClientContext &context) {
		auto state = context.registered_state->Get<RunnerClientState>("vane.runner");
		return state ? state->runner : "local-fast";
	}

private:
	const string runner;
};

} // namespace duckdb
