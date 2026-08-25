// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/distributed/data_sink.hpp"

using namespace duckdb;
using namespace duckdb::distributed;

namespace {

ResultPartitionRef DataSinkPartition(const string &operation_id, const string &state, uint64_t rows_received,
                                     Value rows_affected, uint64_t bytes_received, const string &metadata_json = "{}",
                                     const string &warnings_json = "[]") {
	vector<LogicalType> types {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::UBIGINT, LogicalType::UBIGINT,
	                           LogicalType::UBIGINT, LogicalType::VARCHAR, LogicalType::VARCHAR};
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), types);
	chunk.SetValue(0, 0, Value(operation_id));
	chunk.SetValue(1, 0, Value(state));
	chunk.SetValue(2, 0, Value::UBIGINT(rows_received));
	chunk.SetValue(3, 0, std::move(rows_affected));
	chunk.SetValue(4, 0, Value::UBIGINT(bytes_received));
	chunk.SetValue(5, 0, Value(metadata_json));
	chunk.SetValue(6, 0, Value(warnings_json));
	chunk.SetCardinality(1);

	auto collection = std::make_shared<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
	collection->Append(chunk);
	return std::make_shared<ColumnDataResultPartition>(std::move(collection));
}

class ReportedCountDataSinkPartition final : public ResultPartition {
public:
	explicit ReportedCountDataSinkPartition(size_t rows, size_t bytes = 0,
	                                        std::shared_ptr<ColumnDataCollection> collection = nullptr)
	    : rows_(rows), bytes_(bytes), collection_(std::move(collection)) {
	}

	DuckDBResult<size_t> size_bytes() const override {
		return DuckDBResult<size_t>::ok(bytes_);
	}

	DuckDBResult<size_t> num_rows() const override {
		return DuckDBResult<size_t>::ok(rows_);
	}

	std::shared_ptr<ColumnDataCollection> to_column_data() const override {
		materialized_ = true;
		return collection_;
	}

	bool materialized() const {
		return materialized_;
	}

private:
	size_t rows_;
	size_t bytes_;
	std::shared_ptr<ColumnDataCollection> collection_;
	mutable bool materialized_ = false;
};

} // namespace

TEST_CASE("DataSink worker results preserve applied and aborted outcomes", "[distributed][datasink]") {
	vector<ResultPartitionRef> applied_partitions;
	applied_partitions.push_back(DataSinkPartition("operation-1", "applied", 3, Value::UBIGINT(3), 24));
	applied_partitions.push_back(DataSinkPartition("operation-1", "applied", 2, Value(LogicalType::UBIGINT), 16,
	                                               "{\"request_id\":\"abc\"}", "[\"notice\"]"));

	auto applied = ParseDataSinkPartitions("operation-1", applied_partitions);
	REQUIRE(applied.is_ok());
	REQUIRE(applied.value().operation_id == "operation-1");
	REQUIRE_FALSE(applied.value().outcome_aborted);
	REQUIRE_FALSE(applied.value().outcome_unknown);
	REQUIRE(applied.value().write_results.size() == 2);
	REQUIRE(applied.value().write_results[0].rows_received == 3);
	REQUIRE(applied.value().write_results[1].rows_affected.IsNull());

	vector<ResultPartitionRef> aborted_partitions;
	aborted_partitions.push_back(DataSinkPartition("operation-2", "aborted", 4, Value::UBIGINT(0), 0,
	                                               "{\"validation_error\":\"duplicate_keys\"}"));
	auto aborted = ParseDataSinkPartitions("operation-2", aborted_partitions);
	REQUIRE(aborted.is_ok());
	REQUIRE(aborted.value().outcome_aborted);
	REQUIRE_FALSE(aborted.value().outcome_error.empty());
	REQUIRE(aborted.value().write_results[0].state == "aborted");

	auto empty = ParseDataSinkPartitions("operation-empty", {});
	REQUIRE(empty.is_ok());
	REQUIRE(empty.value().write_results.empty());
	REQUIRE_FALSE(empty.value().outcome_aborted);
}

TEST_CASE("DataSink worker result protocol rejects ambiguous or invalid payloads", "[distributed][datasink]") {
	vector<ResultPartitionRef> mixed;
	mixed.push_back(DataSinkPartition("operation-1", "applied", 1, Value::UBIGINT(1), 8));
	mixed.push_back(DataSinkPartition("operation-1", "aborted", 1, Value::UBIGINT(0), 0));
	auto mixed_result = ParseDataSinkPartitions("operation-1", mixed);
	REQUIRE(mixed_result.is_err());
	REQUIRE(StringUtil::Contains(mixed_result.error().what(), "must not mix applied and aborted"));

	vector<ResultPartitionRef> mismatched;
	mismatched.push_back(DataSinkPartition("different-operation", "applied", 1, Value::UBIGINT(1), 8));
	auto mismatched_result = ParseDataSinkPartitions("operation-1", mismatched);
	REQUIRE(mismatched_result.is_err());
	REQUIRE(StringUtil::Contains(mismatched_result.error().what(), "identity does not match"));

	vector<ResultPartitionRef> invalid_aborted;
	invalid_aborted.push_back(DataSinkPartition("operation-1", "aborted", 1, Value(LogicalType::UBIGINT), 0));
	auto invalid_aborted_result = ParseDataSinkPartitions("operation-1", invalid_aborted);
	REQUIRE(invalid_aborted_result.is_err());
	REQUIRE(StringUtil::Contains(invalid_aborted_result.error().what(), "require zero rows affected"));

	auto bad_collection = std::make_shared<ColumnDataCollection>(Allocator::DefaultAllocator(),
	                                                             vector<LogicalType> {LogicalType::VARCHAR});
	DataChunk bad_chunk;
	bad_chunk.Initialize(Allocator::DefaultAllocator(), bad_collection->Types());
	bad_chunk.SetValue(0, 0, Value("invalid"));
	bad_chunk.SetCardinality(1);
	bad_collection->Append(bad_chunk);
	vector<ResultPartitionRef> invalid_schema;
	invalid_schema.push_back(std::make_shared<ColumnDataResultPartition>(std::move(bad_collection)));
	auto schema_result = ParseDataSinkPartitions("operation-1", invalid_schema);
	REQUIRE(schema_result.is_err());
	REQUIRE(StringUtil::Contains(schema_result.error().what(), "worker result schema"));

	REQUIRE(ParseDataSinkPartitions("", {}).is_err());
	REQUIRE(ParseDataSinkPartitions(string("\xFF", 1), {}).is_err());
}

TEST_CASE("DataSink outcome diagnostics are bounded before coordinator transport", "[distributed][datasink]") {
	const string oversized = "diagnostic-head:" + string(DATA_SINK_MAX_OUTCOME_ERROR_BYTES, 'x') + ":diagnostic-tail";
	const auto bounded = BoundDataSinkOutcomeError(oversized);

	REQUIRE(bounded.size() == DATA_SINK_MAX_OUTCOME_ERROR_BYTES);
	REQUIRE(StringUtil::StartsWith(bounded, "diagnostic-head:"));
	REQUIRE(StringUtil::EndsWith(bounded, ":diagnostic-tail"));
	REQUIRE(StringUtil::Contains(bounded, "..."));
	const auto bounded_c_string = BoundDataSinkOutcomeError(oversized.c_str());
	REQUIRE(bounded_c_string.size() == DATA_SINK_MAX_OUTCOME_ERROR_BYTES);
	REQUIRE(StringUtil::StartsWith(bounded_c_string, "diagnostic-head:"));
	REQUIRE(StringUtil::EndsWith(bounded_c_string, "..."));
	REQUIRE(BoundDataSinkOutcomeError("short diagnostic") == "short diagnostic");
}

TEST_CASE("DataSink result partitions are bounded while the coordinator collects them", "[distributed][datasink]") {
	DataSinkResultCollector collector("incremental-operation");
	auto partition = DataSinkPartition("incremental-operation", "applied", 1, Value::UBIGINT(1), 8);
	std::weak_ptr<ResultPartition> retained_partition = partition;
	REQUIRE(collector.Append(partition).is_ok());
	partition.reset();
	REQUIRE(retained_partition.expired());
	auto collected = collector.Finalize();
	REQUIRE(collected.is_ok());
	REQUIRE(collected.value().write_results.size() == 1);
	auto oversized_partition = std::make_shared<ReportedCountDataSinkPartition>(DATA_SINK_MAX_WRITE_RESULTS + 1);
	DataSinkResultCollector oversized_collector("oversized-operation");
	auto oversized_result = oversized_collector.Append(oversized_partition);
	REQUIRE(oversized_result.is_err());
	REQUIRE_FALSE(oversized_partition->materialized());
	auto oversized_bytes_partition =
	    std::make_shared<ReportedCountDataSinkPartition>(1, DATA_SINK_MAX_TOTAL_RESULT_BYTES + 1);
	DataSinkResultCollector oversized_bytes_collector("oversized-bytes-operation");
	auto oversized_bytes_result = oversized_bytes_collector.Append(oversized_bytes_partition);
	REQUIRE(oversized_bytes_result.is_err());
	REQUIRE(StringUtil::Contains(oversized_bytes_result.error().what(), "64 MiB materialization limit"));
	REQUIRE_FALSE(oversized_bytes_partition->materialized());
	auto first_bundle_partition =
	    std::make_shared<ReportedCountDataSinkPartition>(1, DATA_SINK_MAX_TOTAL_RESULT_BYTES / 2 + 1);
	auto second_bundle_partition =
	    std::make_shared<ReportedCountDataSinkPartition>(1, DATA_SINK_MAX_TOTAL_RESULT_BYTES / 2 + 1);
	DataSinkResultCollector oversized_bundle_collector("oversized-bundle-operation");
	auto oversized_bundle_result = oversized_bundle_collector.Append({first_bundle_partition, second_bundle_partition});
	REQUIRE(oversized_bundle_result.is_err());
	REQUIRE(StringUtil::Contains(oversized_bundle_result.error().what(), "result bundle"));
	REQUIRE_FALSE(first_bundle_partition->materialized());
	REQUIRE_FALSE(second_bundle_partition->materialized());
	auto zero_rows_partition = std::make_shared<ReportedCountDataSinkPartition>(0, 1);
	DataSinkResultCollector zero_rows_collector("zero-rows-operation");
	auto zero_rows_result = zero_rows_collector.Append(zero_rows_partition);
	REQUIRE(zero_rows_result.is_err());
	REQUIRE(StringUtil::Contains(zero_rows_result.error().what(), "must report at least one row"));
	REQUIRE_FALSE(zero_rows_partition->materialized());
	auto unknown_size_partition = std::make_shared<ReportedCountDataSinkPartition>(1, 0);
	DataSinkResultCollector unknown_size_collector("unknown-size-operation");
	auto unknown_size_result = unknown_size_collector.Append(unknown_size_partition);
	REQUIRE(unknown_size_result.is_err());
	REQUIRE(StringUtil::Contains(unknown_size_result.error().what(), "must report a positive byte size"));
	REQUIRE_FALSE(unknown_size_partition->materialized());

	auto mismatched_payload =
	    DataSinkPartition("mismatched-count-operation", "applied", 1, Value::UBIGINT(1), 8)->to_column_data();
	const auto mismatched_payload_size = mismatched_payload->SizeInBytes();
	auto mismatched_count_partition =
	    std::make_shared<ReportedCountDataSinkPartition>(2, mismatched_payload_size, std::move(mismatched_payload));
	DataSinkResultCollector mismatched_count_collector("mismatched-count-operation");
	auto mismatched_count_result = mismatched_count_collector.Append(mismatched_count_partition);
	REQUIRE(mismatched_count_result.is_err());
	REQUIRE(StringUtil::Contains(mismatched_count_result.error().what(), "row-count metadata"));
	REQUIRE(mismatched_count_partition->materialized());

	auto row_result = ValidateDataSinkResultBudget(DATA_SINK_MAX_WRITE_RESULTS, 0, 1);
	REQUIRE(row_result.is_err());
	REQUIRE(StringUtil::Contains(row_result.error().what(), "1000000 write-result limit"));

	auto byte_result = ValidateDataSinkResultBudget(0, DATA_SINK_MAX_TOTAL_RESULT_BYTES, 1);
	REQUIRE(byte_result.is_err());
	REQUIRE(StringUtil::Contains(byte_result.error().what(), "64 MiB coordinator payload limit"));

	REQUIRE(
	    ValidateDataSinkResultBudget(DATA_SINK_MAX_WRITE_RESULTS - 1, DATA_SINK_MAX_TOTAL_RESULT_BYTES - 1, 1).is_ok());
}

TEST_CASE("DataSink collector rolls back an entire unacknowledged output on validation failure",
          "[distributed][datasink]") {
	DataSinkResultCollector collector("atomic-output");
	vector<ResultPartitionRef> partitions;
	partitions.push_back(DataSinkPartition("atomic-output", "applied", 1, Value::UBIGINT(1), 8));
	partitions.push_back(DataSinkPartition("different-operation", "applied", 1, Value::UBIGINT(1), 8));

	auto append_result = collector.Append(partitions);
	REQUIRE(append_result.is_err());
	auto finalized = collector.Finalize();
	REQUIRE(finalized.is_ok());
	REQUIRE(finalized.value().write_results.empty());
}

TEST_CASE("DataSink streaming validation rolls back a partially invalid chunk", "[distributed][datasink]") {
	const string operation_id = "atomic-validation";
	vector<LogicalType> types {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::UBIGINT, LogicalType::UBIGINT,
	                           LogicalType::UBIGINT, LogicalType::VARCHAR, LogicalType::VARCHAR};
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), types);
	chunk.SetValue(0, 0, Value(operation_id));
	chunk.SetValue(1, 0, Value("applied"));
	chunk.SetValue(2, 0, Value::UBIGINT(1));
	chunk.SetValue(3, 0, Value::UBIGINT(1));
	chunk.SetValue(4, 0, Value::UBIGINT(8));
	chunk.SetValue(5, 0, Value("{}"));
	chunk.SetValue(6, 0, Value("[]"));
	chunk.SetValue(0, 1, Value("different-operation"));
	chunk.SetValue(1, 1, Value("applied"));
	chunk.SetValue(2, 1, Value::UBIGINT(1));
	chunk.SetValue(3, 1, Value::UBIGINT(1));
	chunk.SetValue(4, 1, Value::UBIGINT(8));
	chunk.SetValue(5, 1, Value("{}"));
	chunk.SetValue(6, 1, Value("[]"));
	chunk.SetCardinality(2);

	DataSinkResultValidationState validation(operation_id);
	vector<DataSinkWriteResult> retained_results;
	auto append_result = validation.Append(chunk, &retained_results);

	REQUIRE(append_result.is_err());
	REQUIRE(validation.write_result_count() == 0);
	REQUIRE(validation.total_result_bytes() == 0);
	REQUIRE(validation.result_state().empty());
	REQUIRE(retained_results.empty());
}

TEST_CASE("DataSink streaming validation rejects output before a task collector can exceed its byte budget",
          "[distributed][datasink]") {
	const string operation_id = "streaming-budget";
	vector<LogicalType> types {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::UBIGINT, LogicalType::UBIGINT,
	                           LogicalType::UBIGINT, LogicalType::VARCHAR, LogicalType::VARCHAR};
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), types);
	chunk.SetValue(0, 0, Value(operation_id));
	chunk.SetValue(1, 0, Value("applied"));
	chunk.SetValue(2, 0, Value::UBIGINT(1));
	chunk.SetValue(3, 0, Value::UBIGINT(1));
	chunk.SetValue(4, 0, Value::UBIGINT(1));
	chunk.SetValue(5, 0, Value(string(DATA_SINK_MAX_METADATA_BYTES, 'x')));
	chunk.SetValue(6, 0, Value("[]"));
	chunk.SetCardinality(1);

	DataSinkResultValidationState validation(operation_id);
	DuckDBResult<void> append_result = DuckDBResult<void>::ok();
	for (idx_t index = 0; index < 2000 && append_result.is_ok(); index++) {
		append_result = validation.Append(chunk);
	}

	REQUIRE(append_result.is_err());
	REQUIRE(StringUtil::Contains(append_result.error().what(), "64 MiB coordinator payload limit"));
	REQUIRE(validation.total_result_bytes() <= DATA_SINK_MAX_TOTAL_RESULT_BYTES);
}
