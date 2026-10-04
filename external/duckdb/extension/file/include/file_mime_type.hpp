// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// file_mime_type.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types.hpp"

namespace duckdb {

struct FileMimeType {
	// MIME results are bounded best-effort routing hints, not file validation.
	static bool FromPath(const string &path, string &result);
	static bool FromBytes(const_data_ptr_t data, idx_t size, string &result, bool complete_input);
	static bool IsHdf5Signature(const_data_ptr_t data, idx_t size);
};

} // namespace duckdb
