// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "image_operator_contract.hpp"

namespace duckdb {

//! The caller validates the source, box and output allocation. Copy at most
//! COPY_BYTES between interruption checks, including across narrow pixel rows.
template <class INTERRUPT>
void CropImagePixels(const ImagePixelView &source, const ImageCropBox &box, data_ptr_t target, idx_t size,
                     INTERRUPT check_interrupted) {
	for (idx_t offset = 0; offset < size;) {
		check_interrupted();
		auto count = MinValue(size - offset, ImageOperatorContract::COPY_BYTES);
		memset(target + offset, 0, count);
		offset += count;
	}
	// Reject disjoint boxes before adding coordinates so INT64_MIN/MAX
	// origins cannot overflow the intersection arithmetic.
	if (box.x >= source.layout.width || box.y >= source.layout.height || box.x <= -int64_t(box.width) ||
	    box.y <= -int64_t(box.height)) {
		return;
	}
	auto left = MaxValue<int64_t>(box.x, 0);
	auto top = MaxValue<int64_t>(box.y, 0);
	auto right = MinValue<int64_t>(box.x + box.width, source.layout.width);
	auto bottom = MinValue<int64_t>(box.y + box.height, source.layout.height);
	auto channels = source.layout.channels;
	auto source_stride = idx_t(source.layout.width) * channels;
	auto target_stride = idx_t(box.width) * channels;
	auto src = source.data + idx_t(top) * source_stride + idx_t(left) * channels;
	auto dst = target + idx_t(top - box.y) * target_stride + idx_t(left - box.x) * channels;
	auto row_bytes = idx_t(right - left) * channels;
	auto rows = idx_t(bottom - top);
	if (row_bytes == source_stride && row_bytes == target_stride) {
		// Full-width overlaps are contiguous even when the crop has vertical
		// padding. Coalesce them before splitting into bounded copies.
		row_bytes *= rows;
		rows = 1;
	}
	if (row_bytes > ImageOperatorContract::COPY_BYTES) {
		for (idx_t row = 0; row < rows; row++) {
			for (idx_t offset = 0; offset < row_bytes;) {
				check_interrupted();
				auto count = MinValue(row_bytes - offset, ImageOperatorContract::COPY_BYTES);
				memcpy(dst + row * target_stride + offset, src + row * source_stride + offset, count);
				offset += count;
			}
		}
		return;
	}
	auto rows_per_block = ImageOperatorContract::COPY_BYTES / row_bytes;
	for (idx_t row = 0; row < rows;) {
		check_interrupted();
		auto count = MinValue(rows - row, rows_per_block);
		for (idx_t i = 0; i < count; i++) {
			memcpy(dst + (row + i) * target_stride, src + (row + i) * source_stride, row_bytes);
		}
		row += count;
	}
}

} // namespace duckdb
