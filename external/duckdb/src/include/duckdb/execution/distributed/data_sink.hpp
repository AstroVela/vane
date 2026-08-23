// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/types/value.hpp"
#include "duckdb/execution/distributed/common_types.hpp"

namespace duckdb {
namespace distributed {

static constexpr idx_t DATA_SINK_MAX_OPERATION_ID_BYTES = 256;
static constexpr idx_t DATA_SINK_MAX_METADATA_BYTES = 64 * 1024;
static constexpr idx_t DATA_SINK_MAX_WARNINGS_BYTES = 64 * 1024;
static constexpr idx_t DATA_SINK_MAX_OUTCOME_ERROR_BYTES = 4 * 1024;
static constexpr idx_t DATA_SINK_MAX_WRITE_RESULTS = 1000000;
static constexpr idx_t DATA_SINK_MAX_TOTAL_RESULT_BYTES = 64 * 1024 * 1024;

struct DataSinkWriteResult {
	string operation_id;
	string state;
	idx_t rows_received = 0;
	Value rows_affected;
	idx_t bytes_received = 0;
	string metadata_json;
	string warnings_json;
};

struct DistributedDataSinkResult {
	string operation_id;
	vector<DataSinkWriteResult> write_results;
	bool outcome_aborted = false;
	bool outcome_unknown = false;
	string outcome_error;
};

DuckDBResult<DistributedDataSinkResult> ParseDataSinkPartitions(const string &operation_id,
                                                                const vector<ResultPartitionRef> &partitions);
string BoundDataSinkOutcomeError(const string &error);

} // namespace distributed
} // namespace duckdb
