// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/function/function_set.hpp"

namespace duckdb {

struct CreateMacroInfo;

struct ImageFunctions {
	static ScalarFunctionSet GetDecodeFunctions();
	static ScalarFunctionSet GetHashFunctions();
	static vector<unique_ptr<CreateMacroInfo>> GetMacros();
	static ScalarFunctionSet GetCropFunctions();
	static ScalarFunctionSet GetEncodeFunctions();
	static ScalarFunctionSet GetResizeFunctions();
	static ScalarFunctionSet GetConvertFunctions();
};

} // namespace duckdb
