// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/common/types/data_chunk.hpp"

#include <arrow/api.h>

namespace duckdb {
namespace vane_execution {

std::shared_ptr<arrow::Schema> ArrowSchemaFor(const vector<LogicalType> &types, const vector<string> &names = {});
std::shared_ptr<arrow::RecordBatch> Encode(DataChunk &chunk, const std::shared_ptr<arrow::Schema> &schema);
void Decode(const arrow::RecordBatch &batch, const vector<LogicalType> &types, DataChunk &chunk);

} // namespace vane_execution
} // namespace duckdb
