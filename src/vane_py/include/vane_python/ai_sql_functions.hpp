// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/function/function_set.hpp"

namespace duckdb {

struct CreateMacroInfo;

enum class AIEmbeddingKind : uint8_t { TEXT, IMAGE, VIDEO, AUDIO };

struct AISQLFunction {
	static ScalarFunctionSet GetPromptPackFunctions();
	static ScalarFunctionSet GetPromptImplementationFunctions();
	static unique_ptr<CreateMacroInfo> GetPromptMacro();
	static ScalarFunctionSet GetEmbedImplementationFunctions(AIEmbeddingKind kind = AIEmbeddingKind::TEXT);
	static unique_ptr<CreateMacroInfo> GetEmbedMacro(AIEmbeddingKind kind = AIEmbeddingKind::TEXT);
	static ScalarFunctionSet GetJevImplementationFunctions();
	static unique_ptr<CreateMacroInfo> GetJevMacro();
};

} // namespace duckdb
