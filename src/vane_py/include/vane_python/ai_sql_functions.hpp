// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/function/function_set.hpp"

namespace duckdb {

struct CreateMacroInfo;

struct AISQLFunction {
	static ScalarFunctionSet GetPromptImplementationFunctions();
	static unique_ptr<CreateMacroInfo> GetPromptMacro();
	static ScalarFunctionSet GetEmbedImplementationFunctions();
	static unique_ptr<CreateMacroInfo> GetEmbedMacro();
};

} // namespace duckdb
