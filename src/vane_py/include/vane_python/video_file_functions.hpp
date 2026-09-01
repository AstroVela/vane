// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/function/function_set.hpp"

namespace duckdb {

struct VideoFileFunctions {
	static ScalarFunctionSet GetFunctions();
};

} // namespace duckdb
