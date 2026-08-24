// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/scalar/file_functions.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/function/scalar_function.hpp"

namespace duckdb {

struct FileFunctions {
	static vector<ScalarFunction> GetFunctions();
};

DUCKDB_API unique_ptr<FunctionData> BindFileCollectionSearch(ClientContext &context, ScalarFunction &function,
                                                             vector<unique_ptr<Expression>> &arguments);
DUCKDB_API unique_ptr<FunctionData> BindFileMapSearch(ClientContext &context, ScalarFunction &function,
                                                      vector<unique_ptr<Expression>> &arguments);

} // namespace duckdb
