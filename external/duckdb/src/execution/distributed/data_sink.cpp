// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/data_sink.hpp"

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "utf8proc_wrapper.hpp"

namespace duckdb {
namespace distributed {

string BoundDataSinkOutcomeError(const string &error) {
	string normalized = error;
	if (!Utf8Proc::IsValid(normalized.c_str(), normalized.size())) {
		normalized = Utf8Proc::RemoveInvalid(normalized.c_str(), normalized.size());
	}
	if (normalized.size() <= DATA_SINK_MAX_OUTCOME_ERROR_BYTES) {
		return normalized;
	}

	const string omission = "...";
	static_assert(DATA_SINK_MAX_OUTCOME_ERROR_BYTES > 3);
	const auto remaining = DATA_SINK_MAX_OUTCOME_ERROR_BYTES - omission.size();
	const auto prefix_bytes = remaining / 2;
	const auto suffix_bytes = remaining - prefix_bytes;
	auto prefix = Utf8Proc::RemoveInvalid(normalized.c_str(), prefix_bytes);
	auto suffix = Utf8Proc::RemoveInvalid(normalized.c_str() + normalized.size() - suffix_bytes, suffix_bytes);
	return prefix + omission + suffix;
}

DuckDBResult<DistributedDataSinkResult> ParseDataSinkPartitions(const string &operation_id,
                                                                const vector<ResultPartitionRef> &partitions) {
	if (operation_id.empty() || operation_id.size() > DATA_SINK_MAX_OPERATION_ID_BYTES) {
		return DuckDBResult<DistributedDataSinkResult>::err(
		    DuckDBError::value_error("DataSink operation identity must contain 1 to 256 UTF-8 bytes"));
	}

	DistributedDataSinkResult result;
	result.operation_id = operation_id;
	idx_t total_result_bytes = 0;
	string result_state;
	for (const auto &partition : partitions) {
		auto collection_ref = partition ? partition->to_column_data() : nullptr;
		if (!collection_ref) {
			return DuckDBResult<DistributedDataSinkResult>::err(DuckDBError("DataSink expects tabular worker results"));
		}
		auto &collection = *collection_ref;
		const auto &types = collection.Types();
		if (types.size() != 7 || types[0].id() != LogicalTypeId::VARCHAR || types[1].id() != LogicalTypeId::VARCHAR ||
		    types[2].id() != LogicalTypeId::UBIGINT || types[3].id() != LogicalTypeId::UBIGINT ||
		    types[4].id() != LogicalTypeId::UBIGINT || types[5].id() != LogicalTypeId::VARCHAR ||
		    types[6].id() != LogicalTypeId::VARCHAR) {
			return DuckDBResult<DistributedDataSinkResult>::err(
			    DuckDBError("DataSink worker result schema must be "
			                "(VARCHAR, VARCHAR, UBIGINT, UBIGINT, UBIGINT, VARCHAR, VARCHAR)"));
		}

		ColumnDataScanState scan_state;
		collection.InitializeScan(scan_state);
		DataChunk chunk;
		collection.InitializeScanChunk(chunk);
		while (collection.Scan(scan_state, chunk)) {
			if (chunk.ColumnCount() != 7) {
				return DuckDBResult<DistributedDataSinkResult>::err(
				    DuckDBError("DataSink worker result must contain exactly seven columns"));
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
				auto metadata_value = chunk.GetValue(5, row);
				auto warnings_value = chunk.GetValue(6, row);
				if (operation_value.IsNull() || state_value.IsNull() || rows_received_value.IsNull() ||
				    bytes_received_value.IsNull() || metadata_value.IsNull() || warnings_value.IsNull()) {
					return DuckDBResult<DistributedDataSinkResult>::err(
					    DuckDBError("DataSink worker result contains a NULL required field"));
				}

				DataSinkWriteResult write_result;
				write_result.operation_id = operation_value.GetValue<string>();
				if (write_result.operation_id != operation_id) {
					return DuckDBResult<DistributedDataSinkResult>::err(DuckDBError::invalid_state_error(
					    "DataSink worker result operation identity does not match the terminal operation"));
				}
				write_result.state = state_value.GetValue<string>();
				if (write_result.state != "applied" && write_result.state != "aborted") {
					return DuckDBResult<DistributedDataSinkResult>::err(
					    DuckDBError::value_error("DataSink worker result state must be applied or aborted"));
				}
				if (result_state.empty()) {
					result_state = write_result.state;
				} else if (result_state != write_result.state) {
					return DuckDBResult<DistributedDataSinkResult>::err(DuckDBError::invalid_state_error(
					    "DataSink worker results must not mix applied and aborted states"));
				}
				write_result.rows_received = static_cast<idx_t>(rows_received_value.GetValue<uint64_t>());
				write_result.rows_affected = std::move(rows_affected_value);
				write_result.bytes_received = static_cast<idx_t>(bytes_received_value.GetValue<uint64_t>());
				if (write_result.state == "aborted" &&
				    (write_result.rows_affected.IsNull() || write_result.rows_affected.GetValue<uint64_t>() != 0 ||
				     write_result.bytes_received != 0)) {
					return DuckDBResult<DistributedDataSinkResult>::err(DuckDBError::invalid_state_error(
					    "aborted DataSink worker results require zero rows affected and zero bytes received"));
				}
				write_result.metadata_json = metadata_value.GetValue<string>();
				if (write_result.metadata_json.size() > DATA_SINK_MAX_METADATA_BYTES) {
					return DuckDBResult<DistributedDataSinkResult>::err(
					    DuckDBError::value_error("DataSink result metadata exceeds 64 KiB"));
				}
				write_result.warnings_json = warnings_value.GetValue<string>();
				if (write_result.warnings_json.size() > DATA_SINK_MAX_WARNINGS_BYTES) {
					return DuckDBResult<DistributedDataSinkResult>::err(
					    DuckDBError::value_error("DataSink result warnings exceed 64 KiB"));
				}
				auto result_bytes = write_result.operation_id.size() + write_result.state.size() +
				                    3 * sizeof(uint64_t) + write_result.metadata_json.size() +
				                    write_result.warnings_json.size();
				if (result_bytes > DATA_SINK_MAX_TOTAL_RESULT_BYTES - total_result_bytes) {
					return DuckDBResult<DistributedDataSinkResult>::err(
					    DuckDBError::value_error("DataSink exceeds the 64 MiB coordinator payload limit"));
				}
				total_result_bytes += result_bytes;
				result.write_results.push_back(std::move(write_result));
			}
		}
	}
	result.outcome_aborted = result_state == "aborted";
	if (result.outcome_aborted) {
		result.outcome_error = "keyed DataSink input validation rejected the operation before writers opened";
	}
	return DuckDBResult<DistributedDataSinkResult>::ok(std::move(result));
}

} // namespace distributed
} // namespace duckdb
