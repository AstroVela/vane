// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/common/types/data_chunk.hpp"

#include <arrow/api.h>

namespace duckdb {
namespace vane_execution {

class DirectBatch;

struct ExchangeArrowLayout {
	explicit ExchangeArrowLayout(const vector<LogicalType> &types);
	const std::shared_ptr<arrow::Schema> schema;
	// A nonempty list describes frames whose native values already match Arrow.
	const vector<idx_t> fixed_widths;
};

// Exchange schemas preserve native intermediates; result schemas retain the
// declared DECIMAL precision exposed to callers, recursively through containers.
std::shared_ptr<arrow::Schema> ArrowSchemaFor(const vector<LogicalType> &types, const vector<string> &names = {});
std::shared_ptr<arrow::Schema> ResultSchemaFor(const vector<LogicalType> &types, const vector<string> &names);
// Internal exchanges use compact decimals; public results keep decimal128.
std::shared_ptr<arrow::RecordBatch> Encode(DataChunk &chunk, const std::shared_ptr<arrow::Schema> &schema,
                                           bool compact_decimals = true);
std::shared_ptr<arrow::RecordBatch> EncodeFlightFrame(const shared_ptr<DirectBatch> &batch,
                                                      const ExchangeArrowLayout &layout);
// Decoded vectors may borrow buffers. Keep the batch alive until the chunk has
// been copied into the receiver's channel.
void Decode(const arrow::RecordBatch &batch, const vector<LogicalType> &types, const ExchangeArrowLayout &layout,
            DataChunk &chunk);

} // namespace vane_execution
} // namespace duckdb
