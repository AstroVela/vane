// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/arrow/appender/append_data.hpp"
#include "duckdb/common/types/fixed_binary.hpp"

namespace duckdb {

struct ArrowFixedBinaryData {
	static void Initialize(ArrowAppendData &, const LogicalType &, idx_t) {
	}
	static void Append(ArrowAppendData &append, Vector &input, idx_t from, idx_t to, idx_t input_size) {
		auto width = FixedBinaryType::Size(input.GetType());
		auto count = to - from;
		if (count > NumericLimits<idx_t>::Maximum() - append.row_count ||
		    (width && append.row_count + count > NumericLimits<idx_t>::Maximum() / width)) {
			throw OutOfMemoryException("FIXEDBINARY Arrow buffer exceeds addressable storage");
		}
		UnifiedVectorFormat source;
		input.ToUnifiedFormat(input_size, source);
		auto &validity = append.GetValidityBuffer();
		ArrowAppendData::ResizeValidity(validity, append.row_count + count);
		auto &buffer = append.GetMainBuffer();
		buffer.resize((append.row_count + count) * width);
		auto values = UnifiedVectorFormat::GetData<string_t>(source);
		for (idx_t i = from; i < to; i++) {
			auto row = append.row_count + i - from;
			auto selected = source.sel->get_index(i);
			if (!source.validity.RowIsValid(selected)) {
				uint8_t bit;
				idx_t byte;
				ArrowAppendData::GetBitPosition(row, byte, bit);
				append.SetNull(validity.data(), byte, bit);
				if (width) {
					memset(buffer.data() + row * width, 0, width);
				}
			} else {
				FixedBinaryType::Validate(input.GetType(), values[selected].GetSize());
				if (width) {
					memcpy(buffer.data() + row * width, values[selected].GetData(), width);
				}
			}
		}
		append.row_count += count;
	}
	static void Finalize(ArrowAppendData &append, const LogicalType &, ArrowArray *result) {
		result->n_buffers = 2;
		result->buffers[1] = append.GetMainBuffer().data();
	}
};

} // namespace duckdb
