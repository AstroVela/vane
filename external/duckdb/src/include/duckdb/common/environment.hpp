// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/common.hpp"

namespace duckdb {

//! An owned snapshot of a process environment variable. Preserve the distinction
//! between an unset variable and a variable set to the empty string.
class EnvironmentVariable {
public:
	DUCKDB_API explicit EnvironmentVariable(const char *name);

	const char *GetValue() const {
		return is_set ? value.c_str() : nullptr;
	}

private:
	bool is_set = false;
	string value;
};

} // namespace duckdb
