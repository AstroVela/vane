// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/python_input_callback.hpp"
#include "duckdb/main/client_context.hpp"

namespace duckdb {

thread_local PythonInputCallbackScope *PythonInputCallbackScope::current = nullptr;

PythonInputCallbackScope::PythonInputCallbackScope(shared_ptr<const ClientContext> context_p)
    : context(std::move(context_p)), previous(current) {
	current = this;
}

PythonInputCallbackScope::~PythonInputCallbackScope() {
	current = previous;
}

bool PythonInputCallbackScope::IsActive() {
	return current != nullptr;
}

bool PythonInputCallbackScope::Contains(const ClientContext &context) {
	for (auto scope = current; scope; scope = scope->previous) {
		if (scope->context.get() == &context) {
			return true;
		}
	}
	return false;
}

} // namespace duckdb
