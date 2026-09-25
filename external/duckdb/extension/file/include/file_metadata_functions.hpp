// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// file_metadata_functions.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/function/scalar_function.hpp"

namespace duckdb {

struct FileMetadataFunctions {
	static vector<ScalarFunction> GetFunctions();
};

} // namespace duckdb
