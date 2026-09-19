// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/function/function_set.hpp"

namespace duckdb {

struct CreateMacroInfo;

struct AISQLFunction {
	static ScalarFunctionSet GetPromptPackFunctions();
	static ScalarFunctionSet GetPromptImplementationFunctions();
	static unique_ptr<CreateMacroInfo> GetPromptMacro();
	static ScalarFunctionSet GetEmbedImplementationFunctions(bool image = false);
	static unique_ptr<CreateMacroInfo> GetEmbedMacro(bool image = false);
};

} // namespace duckdb
