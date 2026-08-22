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
	vector<ResultPartitionRef> invalid_schema;
	invalid_schema.push_back(std::make_shared<ColumnDataResultPartition>(std::move(bad_collection)));
	auto schema_result = ParseDataSinkPartitions("operation-1", invalid_schema);
	REQUIRE(schema_result.is_err());
	REQUIRE(StringUtil::Contains(schema_result.error().what(), "worker result schema"));

	REQUIRE(ParseDataSinkPartitions("", {}).is_err());
}
