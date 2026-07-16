#pragma once

#include "duckdb/function/function_set.hpp"

namespace duckdb {

struct AISQLFunction {
	static ScalarFunctionSet GetPromptFunctions();
	static ScalarFunctionSet GetEmbedFunctions();
};

} // namespace duckdb
