// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Python.h>

namespace duckdb {

// Detach and close the process runner only when it still wraps expected_runner.
// This is used by relation dispatch failure paths and must never replace the
// exception that triggered the invalidation.
void InvalidateVaneRunnerIfCurrent(PyObject *expected_runner) noexcept;

} // namespace duckdb
