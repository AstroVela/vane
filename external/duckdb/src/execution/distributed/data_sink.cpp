// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/data_sink.hpp"

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "utf8proc_wrapper.hpp"

namespace duckdb {
namespace distributed {

static string BoundDataSinkOutcomeError(const char *error, idx_t error_size) {
	if (!error) {
		return "unknown error";
	}
	if (error_size <= DATA_SINK_MAX_OUTCOME_ERROR_BYTES) {
		if (Utf8Proc::IsValid(error, error_size)) {
			return string(error, error_size);
		}
		return Utf8Proc::RemoveInvalid(error, error_size);
	}

	const string omission = "...";
	static_assert(DATA_SINK_MAX_OUTCOME_ERROR_BYTES > 3);
	const auto remaining = DATA_SINK_MAX_OUTCOME_ERROR_BYTES - omission.size();
	const auto prefix_bytes = remaining / 2;
	const auto suffix_bytes = remaining - prefix_bytes;
	// Sanitize only the bytes that can survive the bound. Copying or repairing
	// the complete provider diagnostic first would create an unbounded
	// coordinator allocation solely to discard its middle.
	auto prefix = Utf8Proc::RemoveInvalid(error, prefix_bytes);
	auto suffix = Utf8Proc::RemoveInvalid(error + error_size - suffix_bytes, suffix_bytes);
	return prefix + omission + suffix;
}

string BoundDataSinkOutcomeError(const string &error) {
	return BoundDataSinkOutcomeError(error.c_str(), error.size());
}

string BoundDataSinkOutcomeError(const char *error) {
	if (!error) {
		return "unknown error";
	}
	idx_t bounded_size = 0;
	while (bounded_size <= DATA_SINK_MAX_OUTCOME_ERROR_BYTES && error[bounded_size] != '\0') {
		bounded_size++;
	}
	if (bounded_size <= DATA_SINK_MAX_OUTCOME_ERROR_BYTES) {
		return BoundDataSinkOutcomeError(error, bounded_size);
	}

	// A raw C string does not expose its allocation length. Scan only the bytes
	// that can survive the diagnostic bound instead of traversing an arbitrarily
	// large provider message just to preserve its tail.
	const string omission = "...";
	const auto prefix_bytes = DATA_SINK_MAX_OUTCOME_ERROR_BYTES - omission.size();
	return Utf8Proc::RemoveInvalid(error, prefix_bytes) + omission;
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
	// Validate into an isolated state so one invalid row cannot leave the shared
	// parallel operator budget or state discriminator partially advanced.
	auto next_state = *this;
	auto operation_res = next_state.ValidateOperationId();
	if (operation_res.is_err()) {
		return operation_res;
	}
	auto schema_res = next_state.ValidateSchema(chunk.GetTypes());
	if (schema_res.is_err()) {
		return schema_res;
	}
	auto count_res = next_state.ValidateAdditionalResultCount(chunk.size());
	if (count_res.is_err()) {
		return count_res;
	}
	vector<DataSinkWriteResult> staged_results;
	if (retained_results) {
		staged_results.reserve(chunk.size());
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
		if (!next_state.result_state_.empty() && next_state.result_state_ != write_result.state) {
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
		auto budget_res =
		    ValidateDataSinkResultBudget(next_state.write_result_count_, next_state.total_result_bytes_, result_bytes);
		if (budget_res.is_err()) {
			return budget_res;
		}
		if (next_state.result_state_.empty()) {
			next_state.result_state_ = write_result.state;
		}
		next_state.write_result_count_++;
		next_state.total_result_bytes_ += result_bytes;
		if (retained_results) {
			staged_results.push_back(std::move(write_result));
		}
	}
	if (retained_results) {
		const auto retained_checkpoint = retained_results->size();
		try {
			retained_results->reserve(retained_checkpoint + staged_results.size());
			for (auto &write_result : staged_results) {
				retained_results->push_back(std::move(write_result));
			}
		} catch (...) {
			retained_results->resize(retained_checkpoint);
			throw;
		}
	}
	*this = std::move(next_state);
	return DuckDBResult<void>::ok();
}

DataSinkResultCollector::DataSinkResultCollector(string operation_id) : validation_state_(operation_id) {
	result_.operation_id = std::move(operation_id);
}

DuckDBResult<void> DataSinkResultCollector::Append(const ResultPartitionRef &partition) {
	return Append(std::vector<ResultPartitionRef> {partition});
}

DuckDBResult<void> DataSinkResultCollector::Append(const std::vector<ResultPartitionRef> &partitions) {
	lock_guard<mutex> guard(lock_);
	if (finalized_) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error("DataSink result collection is finalized"));
	}
	auto operation_res = validation_state_.ValidateOperationId();
	if (operation_res.is_err()) {
		return operation_res;
	}
	// A selected task may return a RefBundle containing several result
	// partitions. Preflight the complete bundle before materializing any of its
	// refs so a one-item coordinator queue still has a bounded payload owner.
	auto minimum_count_res = validation_state_.ValidateAdditionalResultCount(partitions.size());
	if (minimum_count_res.is_err()) {
		return minimum_count_res;
	}
	vector<std::pair<idx_t, idx_t>> partition_metadata;
	partition_metadata.reserve(partitions.size());
	idx_t output_row_count = 0;
	idx_t reported_output_bytes = 0;
	for (const auto &partition : partitions) {
		if (!partition) {
			return DuckDBResult<void>::err(DuckDBError("DataSink expects tabular worker results"));
		}
		auto row_count_res = partition->num_rows();
		if (row_count_res.is_err()) {
			return DuckDBResult<void>::err(
			    DuckDBError::invalid_state_error("DataSink worker result row count is unavailable: " +
			                                     BoundDataSinkOutcomeError(row_count_res.error().what())));
		}
		const auto partition_count = static_cast<idx_t>(row_count_res.value());
		if (partition_count == 0) {
			return DuckDBResult<void>::err(
			    DuckDBError::invalid_state_error("DataSink result partitions must report at least one row"));
		}
		if (output_row_count > DATA_SINK_MAX_WRITE_RESULTS ||
		    partition_count > DATA_SINK_MAX_WRITE_RESULTS - output_row_count) {
			return DuckDBResult<void>::err(DuckDBError::value_error("DataSink exceeds the 1000000 write-result limit"));
		}
		output_row_count += partition_count;
		auto count_res = validation_state_.ValidateAdditionalResultCount(output_row_count);
		if (count_res.is_err()) {
			return count_res;
		}

		auto size_res = partition->size_bytes();
		if (size_res.is_err()) {
			return DuckDBResult<void>::err(
			    DuckDBError::invalid_state_error("DataSink worker result byte size is unavailable: " +
			                                     BoundDataSinkOutcomeError(size_res.error().what())));
		}
		const auto partition_bytes = static_cast<idx_t>(size_res.value());
		if (partition_bytes == 0) {
			return DuckDBResult<void>::err(
			    DuckDBError::invalid_state_error("DataSink result partitions must report a positive byte size"));
		}
		if (reported_output_bytes > DATA_SINK_MAX_TOTAL_RESULT_BYTES ||
		    partition_bytes > DATA_SINK_MAX_TOTAL_RESULT_BYTES - reported_output_bytes) {
			return DuckDBResult<void>::err(
			    DuckDBError::value_error("DataSink result bundle exceeds the 64 MiB materialization limit"));
		}
		reported_output_bytes += partition_bytes;
		partition_metadata.emplace_back(partition_count, partition_bytes);
	}
	auto validation_checkpoint = validation_state_;
	const auto result_count_checkpoint = result_.write_results.size();
	auto rollback = [&]() {
		validation_state_ = validation_checkpoint;
		result_.write_results.resize(result_count_checkpoint);
	};
	try {
		idx_t materialized_output_bytes = 0;
		for (idx_t partition_index = 0; partition_index < partitions.size(); partition_index++) {
			const auto &metadata = partition_metadata[partition_index];
			auto append_res =
			    AppendUnlocked(partitions[partition_index], metadata.first, metadata.second, materialized_output_bytes);
			if (append_res.is_err()) {
				rollback();
				return append_res;
			}
		}
	} catch (...) {
		rollback();
		throw;
	}
	return DuckDBResult<void>::ok();
}

DuckDBResult<void> DataSinkResultCollector::AppendUnlocked(const ResultPartitionRef &partition, idx_t partition_count,
                                                           idx_t reported_partition_bytes,
                                                           idx_t &materialized_output_bytes) {
	if (materialized_output_bytes > DATA_SINK_MAX_TOTAL_RESULT_BYTES ||
	    reported_partition_bytes > DATA_SINK_MAX_TOTAL_RESULT_BYTES - materialized_output_bytes) {
		return DuckDBResult<void>::err(
		    DuckDBError::value_error("DataSink result bundle exceeds the 64 MiB materialization limit"));
	}
	auto collection_ref = partition->to_column_data();
	if (!collection_ref) {
		return DuckDBResult<void>::err(DuckDBError("DataSink expects tabular worker results"));
	}
	auto &collection = *collection_ref;
	if (collection.Count() != partition_count) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error(
		    "DataSink result partition row-count metadata does not match its materialized payload"));
	}
	const auto materialized_partition_bytes = collection.SizeInBytes();
	if (materialized_output_bytes > DATA_SINK_MAX_TOTAL_RESULT_BYTES ||
	    materialized_partition_bytes > DATA_SINK_MAX_TOTAL_RESULT_BYTES - materialized_output_bytes) {
		return DuckDBResult<void>::err(
		    DuckDBError::value_error("DataSink result bundle exceeds the 64 MiB materialization limit"));
	}
	materialized_output_bytes += materialized_partition_bytes;
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
	lock_guard<mutex> guard(lock_);
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
