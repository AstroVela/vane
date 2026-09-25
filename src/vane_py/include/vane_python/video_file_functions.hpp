// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/function/function_set.hpp"

namespace duckdb {

struct CreateMacroInfo;

struct VideoFileFunctions {
	static ScalarFunctionSet GetFunctions();
	static TableFunctionSet GetReadFunctions();
	static vector<ScalarFunctionSet> GetFrameFunctions();
	static vector<unique_ptr<CreateMacroInfo>> GetFrameMacros();
};

} // namespace duckdb
