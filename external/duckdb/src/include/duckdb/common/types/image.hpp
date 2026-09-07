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
	//! Allocate fixed Image pixels for rows that are actually written, including NULL padding.
	DUCKDB_API static void Reserve(Vector &output, idx_t count);
	DUCKDB_API static void SetNullPixels(Vector &output, idx_t row);
	DUCKDB_API static void CopyRows(const Vector &source, Vector &target, const SelectionVector &sel,
	                                idx_t source_offset, idx_t target_offset, idx_t count);
	//! Flatten metadata and pixel children for the full batch before row access.
	DUCKDB_API static void Flatten(Vector &input, idx_t count);
	DUCKDB_API static ImageLayout Layout(const Value &value);
	DUCKDB_API static ImageLayout Layout(Vector &input, idx_t row);
	DUCKDB_API static const vector<Value> &Pixels(const Value &value);
	DUCKDB_API static const Value &PixelValues(const Value &value);
	DUCKDB_API static void CopyPixels(const Value &value, data_ptr_t target);
	DUCKDB_API static Value GetValue(const Vector &input, idx_t row);
	DUCKDB_API static const_data_ptr_t Pixels(Vector &input, idx_t row);
	DUCKDB_API static data_ptr_t Allocate(Vector &output, idx_t row, uint32_t width, uint32_t height,
	                                      const string &mode);
	DUCKDB_API static void ValidateRows(Vector &input, const vector<idx_t> &rows, const string &boundary);
	DUCKDB_API static Value FromPixels(vector<Value> pixels, uint32_t width, uint32_t height, const string &mode,
	                                   const LogicalType &type);
	DUCKDB_API static Value FromPixels(const_data_ptr_t pixels, idx_t size, uint32_t width, uint32_t height,
	                                   const string &mode, const LogicalType &type);
};

} // namespace duckdb
