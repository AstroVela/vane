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

DuckDBResult<void> ValidateDataSinkResultBudget(idx_t write_result_count, idx_t total_result_bytes,
                                                idx_t next_result_bytes) {
	if (write_result_count >= DATA_SINK_MAX_WRITE_RESULTS) {
		return DuckDBResult<void>::err(DuckDBError::value_error("DataSink exceeds the 1000000 write-result limit"));
	}
	if (total_result_bytes > DATA_SINK_MAX_TOTAL_RESULT_BYTES ||
	    next_result_bytes > DATA_SINK_MAX_TOTAL_RESULT_BYTES - total_result_bytes) {
		return DuckDBResult<void>::err(
		    DuckDBError::value_error("DataSink exceeds the 64 MiB coordinator payload limit"));
	}
	return DuckDBResult<void>::ok();
}

DataSinkResultValidationState::DataSinkResultValidationState(string operation_id)
    : operation_id_(std::move(operation_id)) {
}

DuckDBResult<void> DataSinkResultValidationState::ValidateOperationId() const {
	if (operation_id_.empty() || operation_id_.size() > DATA_SINK_MAX_OPERATION_ID_BYTES ||
	    !Utf8Proc::IsValid(operation_id_.c_str(), operation_id_.size())) {
		return DuckDBResult<void>::err(
		    DuckDBError::value_error("DataSink operation identity must contain 1 to 256 UTF-8 bytes"));
	}
	return DuckDBResult<void>::ok();
}

DuckDBResult<void> DataSinkResultValidationState::ValidateSchema(const vector<LogicalType> &types) const {
	if (types.size() != 7 || types[0].id() != LogicalTypeId::VARCHAR || types[1].id() != LogicalTypeId::VARCHAR ||
	    types[2].id() != LogicalTypeId::UBIGINT || types[3].id() != LogicalTypeId::UBIGINT ||
	    types[4].id() != LogicalTypeId::UBIGINT || types[5].id() != LogicalTypeId::VARCHAR ||
	    types[6].id() != LogicalTypeId::VARCHAR) {
		return DuckDBResult<void>::err(DuckDBError("DataSink worker result schema must be "
		                                           "(VARCHAR, VARCHAR, UBIGINT, UBIGINT, UBIGINT, VARCHAR, VARCHAR)"));
	}
	return DuckDBResult<void>::ok();
}

DuckDBResult<void> DataSinkResultValidationState::ValidateAdditionalResultCount(idx_t additional_count) const {
	if (write_result_count_ > DATA_SINK_MAX_WRITE_RESULTS ||
	    additional_count > DATA_SINK_MAX_WRITE_RESULTS - write_result_count_) {
		return DuckDBResult<void>::err(DuckDBError::value_error("DataSink exceeds the 1000000 write-result limit"));
	}
	return DuckDBResult<void>::ok();
}

DuckDBResult<void> DataSinkResultValidationState::Append(const DataChunk &chunk,
                                                         vector<DataSinkWriteResult> *retained_results) {
	auto operation_res = ValidateOperationId();
	if (operation_res.is_err()) {
		return operation_res;
	}
	auto schema_res = ValidateSchema(chunk.GetTypes());
	if (schema_res.is_err()) {
		return schema_res;
	}
	auto count_res = ValidateAdditionalResultCount(chunk.size());
	if (count_res.is_err()) {
		return count_res;
	}
	for (idx_t row = 0; row < chunk.size(); row++) {
		auto operation_value = chunk.GetValue(0, row);
		auto state_value = chunk.GetValue(1, row);
		auto rows_received_value = chunk.GetValue(2, row);
		auto rows_affected_value = chunk.GetValue(3, row);
		auto bytes_received_value = chunk.GetValue(4, row);
		auto metadata_value = chunk.GetValue(5, row);
		auto warnings_value = chunk.GetValue(6, row);
		if (operation_value.IsNull() || state_value.IsNull() || rows_received_value.IsNull() ||
		    bytes_received_value.IsNull() || metadata_value.IsNull() || warnings_value.IsNull()) {
			return DuckDBResult<void>::err(DuckDBError("DataSink worker result contains a NULL required field"));
		}

		DataSinkWriteResult write_result;
		write_result.operation_id = operation_value.GetValue<string>();
		if (write_result.operation_id != operation_id_) {
			return DuckDBResult<void>::err(DuckDBError::invalid_state_error(
			    "DataSink worker result operation identity does not match the terminal operation"));
		}
		write_result.state = state_value.GetValue<string>();
		if (write_result.state != "applied" && write_result.state != "aborted") {
			return DuckDBResult<void>::err(
			    DuckDBError::value_error("DataSink worker result state must be applied or aborted"));
		}
		if (!result_state_.empty() && result_state_ != write_result.state) {
			return DuckDBResult<void>::err(
			    DuckDBError::invalid_state_error("DataSink worker results must not mix applied and aborted states"));
		}
		write_result.rows_received = static_cast<idx_t>(rows_received_value.GetValue<uint64_t>());
		write_result.rows_affected = std::move(rows_affected_value);
		write_result.bytes_received = static_cast<idx_t>(bytes_received_value.GetValue<uint64_t>());
		if (write_result.state == "aborted" &&
		    (write_result.rows_affected.IsNull() || write_result.rows_affected.GetValue<uint64_t>() != 0 ||
		     write_result.bytes_received != 0)) {
			return DuckDBResult<void>::err(DuckDBError::invalid_state_error(
			    "aborted DataSink worker results require zero rows affected and zero bytes received"));
		}
		write_result.metadata_json = metadata_value.GetValue<string>();
		if (write_result.metadata_json.size() > DATA_SINK_MAX_METADATA_BYTES) {
			return DuckDBResult<void>::err(DuckDBError::value_error("DataSink result metadata exceeds 64 KiB"));
		}
		write_result.warnings_json = warnings_value.GetValue<string>();
		if (write_result.warnings_json.size() > DATA_SINK_MAX_WARNINGS_BYTES) {
			return DuckDBResult<void>::err(DuckDBError::value_error("DataSink result warnings exceed 64 KiB"));
		}
		const auto result_bytes = write_result.operation_id.size() + write_result.state.size() + 3 * sizeof(uint64_t) +
		                          write_result.metadata_json.size() + write_result.warnings_json.size();
		auto budget_res = ValidateDataSinkResultBudget(write_result_count_, total_result_bytes_, result_bytes);
		if (budget_res.is_err()) {
			return budget_res;
		}
		if (result_state_.empty()) {
			result_state_ = write_result.state;
		}
		write_result_count_++;
		total_result_bytes_ += result_bytes;
		if (retained_results) {
			retained_results->push_back(std::move(write_result));
		}
	}
	return DuckDBResult<void>::ok();
}

DataSinkResultCollector::DataSinkResultCollector(string operation_id) : validation_state_(operation_id) {
	result_.operation_id = std::move(operation_id);
}

DuckDBResult<void> DataSinkResultCollector::Append(const ResultPartitionRef &partition) {
	if (finalized_) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error("DataSink result collection is finalized"));
	}
	auto operation_res = validation_state_.ValidateOperationId();
	if (operation_res.is_err()) {
		return operation_res;
	}
	if (!partition) {
		return DuckDBResult<void>::err(DuckDBError("DataSink expects tabular worker results"));
	}
	auto row_count_res = partition->num_rows();
	if (row_count_res.is_err()) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error(
		    "DataSink worker result row count is unavailable: " + string(row_count_res.error().what())));
	}
	const auto partition_count = static_cast<idx_t>(row_count_res.value());
	auto count_res = validation_state_.ValidateAdditionalResultCount(partition_count);
	if (count_res.is_err()) {
		return count_res;
	}
	auto collection_ref = partition->to_column_data();
	if (!collection_ref) {
		return DuckDBResult<void>::err(DuckDBError("DataSink expects tabular worker results"));
	}
	auto &collection = *collection_ref;
	auto schema_res = validation_state_.ValidateSchema(collection.Types());
	if (schema_res.is_err()) {
		return schema_res;
	}

	ColumnDataScanState scan_state;
	collection.InitializeScan(scan_state);
	DataChunk chunk;
	collection.InitializeScanChunk(chunk);
	while (collection.Scan(scan_state, chunk)) {
		auto append_res = validation_state_.Append(chunk, &result_.write_results);
		if (append_res.is_err()) {
			return append_res;
		}
	}
	return DuckDBResult<void>::ok();
}

DuckDBResult<DistributedDataSinkResult> DataSinkResultCollector::Finalize() {
	if (finalized_) {
		return DuckDBResult<DistributedDataSinkResult>::err(
		    DuckDBError::invalid_state_error("DataSink result collection is already finalized"));
	}
	auto operation_res = validation_state_.ValidateOperationId();
	if (operation_res.is_err()) {
		return DuckDBResult<DistributedDataSinkResult>::err(operation_res.error());
	}
	finalized_ = true;
	result_.outcome_aborted = validation_state_.result_state() == "aborted";
	if (result_.outcome_aborted) {
		result_.outcome_error = "keyed DataSink input validation rejected the operation before writers opened";
	}
	return DuckDBResult<DistributedDataSinkResult>::ok(std::move(result_));
}

DuckDBResult<DistributedDataSinkResult> ParseDataSinkPartitions(const string &operation_id,
                                                                const vector<ResultPartitionRef> &partitions) {
	DataSinkResultCollector collector(operation_id);
	for (const auto &partition : partitions) {
		auto append_res = collector.Append(partition);
		if (append_res.is_err()) {
			return DuckDBResult<DistributedDataSinkResult>::err(append_res.error());
		}
	}
	return collector.Finalize();
}

} // namespace distributed
} // namespace duckdb
