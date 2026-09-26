// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/common/shared_ptr.hpp"

namespace duckdb {

class ClientContext;

//! Identify input callbacks on both query and native worker threads, without
//! taking a connection lock or treating independent control threads as callbacks.
class PythonInputCallbackScope {
public:
	explicit PythonInputCallbackScope(shared_ptr<const ClientContext> context_p);
	~PythonInputCallbackScope();

	PythonInputCallbackScope(const PythonInputCallbackScope &) = delete;
	PythonInputCallbackScope &operator=(const PythonInputCallbackScope &) = delete;

	//! Callback lifetime is independent of whether an I/O handle already exists
	//! or an executing context is available (for example during open/metadata).
	static bool IsActive();
	static bool Contains(const ClientContext &context);

private:
	shared_ptr<const ClientContext> context;
	PythonInputCallbackScope *previous;
	static thread_local PythonInputCallbackScope *current;
};

} // namespace duckdb
