// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/distributed/common_types.hpp"

namespace duckdb {
namespace distributed {

static constexpr idx_t DATA_SINK_MAX_OPERATION_ID_BYTES = 256;
static constexpr idx_t DATA_SINK_MAX_COMMIT_TOKEN_BYTES = 64 * 1024;
static constexpr idx_t DATA_SINK_MAX_METADATA_BYTES = 64 * 1024;
static constexpr idx_t DATA_SINK_MAX_WRITE_RESULTS = 1000000;
static constexpr idx_t DATA_SINK_MAX_TOTAL_RESULT_BYTES = 64 * 1024 * 1024;

struct DataSinkWriteResult {
	std::string operation_id;
	std::string state;
	idx_t rows_received = 0;
	Value rows_affected;
	idx_t bytes_received = 0;
	Value commit_token;
	std::string metadata_json;
	std::string warnings_json;
};

struct DistributedDataSinkResult {
	std::string operation_id;
	std::vector<DataSinkWriteResult> write_results;
};

inline DuckDBResult<DistributedDataSinkResult>
ParseDataSinkPartitions(const std::string &operation_id, const std::vector<ResultPartitionRef> &partitions) {
	if (operation_id.empty() || operation_id.size() > DATA_SINK_MAX_OPERATION_ID_BYTES) {
		return DuckDBResult<DistributedDataSinkResult>::err(
		    DuckDBError::value_error("DataSink operation identity must contain 1 to 256 UTF-8 bytes"));
	}
	DistributedDataSinkResult result;
	result.operation_id = operation_id;
	idx_t total_result_bytes = 0;
	for (const auto &partition : partitions) {
		auto collection_ref = partition ? partition->to_column_data() : nullptr;
		if (!collection_ref) {
			return DuckDBResult<DistributedDataSinkResult>::err(DuckDBError("DataSink expects tabular worker results"));
		}
		auto &collection = *collection_ref;
		const auto &types = collection.Types();
		if (types.size() != 8 || types[0].id() != LogicalTypeId::VARCHAR || types[1].id() != LogicalTypeId::VARCHAR ||
		    types[2].id() != LogicalTypeId::UBIGINT || types[3].id() != LogicalTypeId::UBIGINT ||
		    types[4].id() != LogicalTypeId::UBIGINT || types[5].id() != LogicalTypeId::BLOB ||
		    types[6].id() != LogicalTypeId::VARCHAR || types[7].id() != LogicalTypeId::VARCHAR) {
			return DuckDBResult<DistributedDataSinkResult>::err(
			    DuckDBError("DataSink worker result schema must be "
			                "(VARCHAR, VARCHAR, UBIGINT, UBIGINT, UBIGINT, BLOB, VARCHAR, VARCHAR)"));
		}

		ColumnDataScanState scan_state;
		collection.InitializeScan(scan_state);
		DataChunk chunk;
		collection.InitializeScanChunk(chunk);
		while (collection.Scan(scan_state, chunk)) {
			if (chunk.ColumnCount() != 8) {
				return DuckDBResult<DistributedDataSinkResult>::err(
				    DuckDBError("DataSink worker result must contain exactly eight columns"));
			}
			for (idx_t row = 0; row < chunk.size(); row++) {
				if (result.write_results.size() >= DATA_SINK_MAX_WRITE_RESULTS) {
					return DuckDBResult<DistributedDataSinkResult>::err(
					    DuckDBError::value_error("DataSink exceeds the 1000000 write-result limit"));
				}
				auto operation_value = chunk.GetValue(0, row);
				auto state_value = chunk.GetValue(1, row);
				auto rows_received_value = chunk.GetValue(2, row);
				auto rows_affected_value = chunk.GetValue(3, row);
				auto bytes_received_value = chunk.GetValue(4, row);
				auto commit_token_value = chunk.GetValue(5, row);
				auto metadata_value = chunk.GetValue(6, row);
				auto warnings_value = chunk.GetValue(7, row);
				if (operation_value.IsNull() || state_value.IsNull() || rows_received_value.IsNull() ||
				    bytes_received_value.IsNull() || metadata_value.IsNull() || warnings_value.IsNull()) {
					return DuckDBResult<DistributedDataSinkResult>::err(
					    DuckDBError("DataSink worker result contains a NULL required field"));
				}

				DataSinkWriteResult write_result;
				write_result.operation_id = operation_value.GetValue<std::string>();
				if (write_result.operation_id != operation_id) {
					return DuckDBResult<DistributedDataSinkResult>::err(DuckDBError::invalid_state_error(
					    "DataSink worker result operation identity does not match the terminal operation"));
				}
				write_result.state = state_value.GetValue<std::string>();
				if (write_result.state != "applied" && write_result.state != "prepared") {
					return DuckDBResult<DistributedDataSinkResult>::err(
					    DuckDBError::value_error("DataSink worker result state must be applied or prepared"));
				}
				write_result.rows_received = static_cast<idx_t>(rows_received_value.GetValue<uint64_t>());
				write_result.rows_affected = std::move(rows_affected_value);
				write_result.bytes_received = static_cast<idx_t>(bytes_received_value.GetValue<uint64_t>());
				write_result.commit_token = std::move(commit_token_value);
				if (!write_result.commit_token.IsNull() &&
				    StringValue::Get(write_result.commit_token).size() > DATA_SINK_MAX_COMMIT_TOKEN_BYTES) {
					return DuckDBResult<DistributedDataSinkResult>::err(
					    DuckDBError::value_error("DataSink commit token exceeds 64 KiB"));
				}
				write_result.metadata_json = metadata_value.GetValue<std::string>();
				if (write_result.metadata_json.size() > DATA_SINK_MAX_METADATA_BYTES) {
					return DuckDBResult<DistributedDataSinkResult>::err(
					    DuckDBError::value_error("DataSink result metadata exceeds 64 KiB"));
				}
				write_result.warnings_json = warnings_value.GetValue<std::string>();
				if (write_result.warnings_json.size() > DATA_SINK_MAX_METADATA_BYTES) {
					return DuckDBResult<DistributedDataSinkResult>::err(
					    DuckDBError::value_error("DataSink result warnings exceed 64 KiB"));
				}
				auto result_bytes = write_result.operation_id.size() + write_result.state.size() +
				                    3 * sizeof(uint64_t) + write_result.metadata_json.size() +
				                    write_result.warnings_json.size();
				if (!write_result.commit_token.IsNull()) {
					result_bytes += StringValue::Get(write_result.commit_token).size();
				}
				if (result_bytes > DATA_SINK_MAX_TOTAL_RESULT_BYTES - total_result_bytes) {
					return DuckDBResult<DistributedDataSinkResult>::err(
					    DuckDBError::value_error("DataSink exceeds the 64 MiB coordinator payload limit"));
				}
				total_result_bytes += result_bytes;
				result.write_results.push_back(std::move(write_result));
			}
		}
	}
	return DuckDBResult<DistributedDataSinkResult>::ok(std::move(result));
}

} // namespace distributed
} // namespace duckdb
