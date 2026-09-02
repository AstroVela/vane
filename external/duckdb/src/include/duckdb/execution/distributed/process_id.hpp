// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace duckdb {
namespace distributed {

inline unsigned long long ResolveVaneProcessId() {
#if defined(_WIN32)
	return static_cast<unsigned long long>(_getpid());
#else
	return static_cast<unsigned long long>(getpid());
#endif
}

} // namespace distributed
} // namespace duckdb
