// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"

namespace duckdb {

struct ImageLayout {
	uint32_t width;
	uint32_t height;
	uint16_t channels;
	uint8_t mode;

	idx_t Size() const {
		return idx_t(width) * height * channels;
	}
};

//! Shared pixel storage access for the base engine and optional extensions.
//! Vector readers accept flattened input; writers return contiguous pixels.
struct ImageVector {
	//! Flatten metadata and pixel children for the full batch before row access.
	DUCKDB_API static void Flatten(Vector &input, idx_t count);
	DUCKDB_API static ImageLayout Layout(const Value &value);
	DUCKDB_API static ImageLayout Layout(Vector &input, idx_t row);
	DUCKDB_API static const vector<Value> &Pixels(const Value &value);
	DUCKDB_API static const_data_ptr_t Pixels(Vector &input, idx_t row);
	DUCKDB_API static data_ptr_t Allocate(Vector &output, idx_t row, uint32_t width, uint32_t height,
	                                      const string &mode);
	DUCKDB_API static void ValidateRows(Vector &input, const vector<idx_t> &rows, const string &boundary);
	DUCKDB_API static Value FromPixels(vector<Value> pixels, uint32_t width, uint32_t height, const string &mode,
	                                   const LogicalType &type);
};

} // namespace duckdb
