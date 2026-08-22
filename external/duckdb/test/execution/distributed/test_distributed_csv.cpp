// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"
#include "test_helpers.hpp"

#include "duckdb.hpp"
#include "duckdb/common/allocator.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/distributed/pipeline_node/translator_scan.hpp"
#include "duckdb/execution/distributed/plan/scan_split.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/physical_plan.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/main/prepared_statement_data.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <set>

using namespace duckdb;

namespace {

struct CSVTestRow {
	int64_t id;
	string payload;

	bool operator<(const CSVTestRow &other) const {
		return id < other.id;
	}
	bool operator==(const CSVTestRow &other) const {
		return id == other.id && payload == other.payload;
	}
};

struct PlannedCSVScan {
	distributed::DuckPhysicalPlanRef worker_plan;
	vector<distributed::ScanSplit> splits;
};

static distributed::ScanSplitBatch CSVSplitBatch(vector<distributed::ScanSplit> splits) {
	distributed::ScanSplitBatch result;
	result.splits = std::move(splits);
	result.Validate();
	return result;
}

static distributed::ScanSplitBatch CSVSplitBatch(const distributed::ScanSplit &split) {
	return CSVSplitBatch(vector<distributed::ScanSplit> {split});
}

static PlannedCSVScan PlanCSVScan(DuckDB &db, Connection &connection, const string &query, idx_t worker_slots) {
	auto logical_plan = connection.ExtractPlan(query);
	REQUIRE(logical_plan != nullptr);
	PhysicalPlanGenerator generator(*connection.context);
	auto physical_plan = generator.Plan(std::move(logical_plan));
	REQUIRE(physical_plan != nullptr);
	REQUIRE(physical_plan->Root().type == PhysicalOperatorType::TABLE_SCAN);
	auto &scan = physical_plan->Root().Cast<PhysicalTableScan>();
	REQUIRE(scan.function.HasDistributedScanCallbacks());

	distributed::DuckDBExecutionConfig config;
	config.set_distributed_worker_slots(worker_slots);
	PlannedCSVScan result;
	result.worker_plan = distributed::MakeTableScanPlan(scan);
	result.splits = distributed::MakeTableScanSplits(scan, config, db.instance);
	return result;
}

static vector<string> CSVScanOutputNames(const PhysicalTableScan &scan) {
	vector<idx_t> output_ids;
	if (scan.projection_ids.empty()) {
		output_ids.reserve(scan.column_ids.size());
		for (idx_t column_idx = 0; column_idx < scan.column_ids.size(); column_idx++) {
			output_ids.push_back(column_idx);
		}
	} else {
		output_ids = scan.projection_ids;
	}
	if (output_ids.size() != scan.GetTypes().size()) {
		throw InternalException("Distributed CSV test scan has %llu output columns but %llu output types",
		                        output_ids.size(), scan.GetTypes().size());
	}

	vector<string> result;
	result.reserve(output_ids.size());
	for (const auto output_id : output_ids) {
		if (output_id >= scan.column_ids.size()) {
			throw InternalException("Distributed CSV test projection index %llu is out of range", output_id);
		}
		const auto &column = scan.column_ids[output_id];
		const auto primary_id = column.GetPrimaryIndex();
		if (column.IsVirtualColumn()) {
			auto entry = scan.virtual_columns.find(primary_id);
			if (entry == scan.virtual_columns.end()) {
				throw InternalException("Distributed CSV test virtual output column %llu is not bound", primary_id);
			}
			result.push_back(entry->second.name);
		} else {
			if (primary_id >= scan.names.size()) {
				throw InternalException("Distributed CSV test output column %llu has no name", primary_id);
			}
			result.push_back(scan.names[primary_id]);
		}
	}
	return result;
}

static unique_ptr<MaterializedQueryResult> ExecuteCSVPlan(Connection &connection, unique_ptr<PhysicalPlan> plan,
                                                          PhysicalTableScan &scan) {
	auto prepared = make_shared_ptr<PreparedStatementData>(StatementType::SELECT_STATEMENT);
	prepared->names = CSVScanOutputNames(scan);
	prepared->types = scan.GetTypes();
	prepared->properties.return_type = StatementReturnType::QUERY_RESULT;
	prepared->output_type = QueryResultOutputType::FORCE_MATERIALIZED;
	prepared->memory_type = QueryResultMemoryType::IN_MEMORY;
	prepared->physical_plan = std::move(plan);
	PendingQueryParameters parameters;
	auto pending =
	    connection.context->PendingQueryPreparedStatementNoRebind("test:distributed_csv", prepared, parameters);
	REQUIRE(pending != nullptr);
	const auto pending_error = pending->HasError() ? pending->GetError() : string();
	INFO(pending_error);
	REQUIRE_FALSE(pending->HasError());
	auto result = pending->Execute();
	REQUIRE(result != nullptr);
	REQUIRE_NO_FAIL(*result);
	auto materialized = unique_ptr_cast<QueryResult, MaterializedQueryResult>(std::move(result));
	REQUIRE(materialized != nullptr);
	return materialized;
}

static vector<CSVTestRow> ExecuteCSVAssignment(DuckDB &db, Connection &connection,
                                               const distributed::DuckPhysicalPlanRef &worker_plan,
                                               const distributed::ScanSplitBatch &input_batch, idx_t scan_node_id) {
	auto batch = distributed::ScanSplitBatch::DeserializeFromBytes(input_batch.SerializeToBytes());
	auto execution_plan = make_uniq<PhysicalPlan>(Allocator::DefaultAllocator());
	PhysicalTableScan *scan = nullptr;
	{
		// Exercise the same short-lived deserialization context used by worker plan cloning.
		Connection transient_connection(db);
		scan = &distributed::ClonePhysicalPlanRootIntoPlanOrThrow(worker_plan, *execution_plan, "distributed_csv",
		                                                          transient_connection.context.get())
		            .Cast<PhysicalTableScan>();
		scan->extra_info.scan_node_id = optional_idx(scan_node_id);
		scan->extra_info.scan_group_id = optional_idx(scan_node_id);
		execution_plan->SetRoot(*scan);
		unordered_map<idx_t, distributed::ScanSplitBatch> assignments;
		assignments.emplace(scan_node_id, std::move(batch));
		string apply_error;
		REQUIRE(distributed::ApplyScanSplitBatchesToPlan(*execution_plan, assignments, &apply_error));
		REQUIRE(apply_error.empty());
		REQUIRE(distributed::ValidateDistributedScanSplitsApplied(*execution_plan));
	}

	auto result = ExecuteCSVPlan(connection, std::move(execution_plan), *scan);
	vector<CSVTestRow> rows;
	rows.reserve(result->RowCount());
	for (idx_t row_idx = 0; row_idx < result->RowCount(); row_idx++) {
		rows.push_back(
		    {result->GetValue(0, row_idx).GetValue<int64_t>(), result->GetValue(1, row_idx).GetValue<string>()});
	}
	return rows;
}

static vector<CSVTestRow> ExecuteCSVAssignment(DuckDB &db, Connection &connection,
                                               const distributed::DuckPhysicalPlanRef &worker_plan,
                                               const distributed::ScanSplit &split, idx_t scan_node_id) {
	return ExecuteCSVAssignment(db, connection, worker_plan, CSVSplitBatch(split), scan_node_id);
}

static unique_ptr<QueryResult> ExecuteDetachedCSVPlan(Connection &connection,
                                                      const distributed::DuckPhysicalPlanRef &worker_plan) {
	auto execution_plan = make_uniq<PhysicalPlan>(Allocator::DefaultAllocator());
	auto &scan = distributed::ClonePhysicalPlanRootIntoPlanOrThrow(worker_plan, *execution_plan,
	                                                               "distributed_csv_detached", connection.context.get())
	                 .Cast<PhysicalTableScan>();
	scan.extra_info.scan_node_id = optional_idx(999);
	scan.extra_info.scan_group_id = optional_idx(999);
	execution_plan->SetRoot(scan);
	auto prepared = make_shared_ptr<PreparedStatementData>(StatementType::SELECT_STATEMENT);
	prepared->names = CSVScanOutputNames(scan);
	prepared->types = scan.GetTypes();
	prepared->properties.return_type = StatementReturnType::QUERY_RESULT;
	prepared->output_type = QueryResultOutputType::FORCE_MATERIALIZED;
	prepared->memory_type = QueryResultMemoryType::IN_MEMORY;
	prepared->physical_plan = std::move(execution_plan);
	PendingQueryParameters parameters;
	auto pending = connection.context->PendingQueryPreparedStatementNoRebind("test:detached_csv", prepared, parameters);
	if (pending->HasError()) {
		return pending->Execute();
	}
	return pending->Execute();
}

static vector<CSVTestRow> WriteCSVRangeFixture(const string &path, idx_t row_count) {
	std::ofstream output(path, std::ios::out | std::ios::trunc | std::ios::binary);
	REQUIRE(output.good());
	output << "id,payload\r\n";
	vector<CSVTestRow> expected;
	expected.reserve(row_count);
	for (idx_t row_idx = 0; row_idx < row_count; row_idx++) {
		string payload;
		if (row_idx % 7 == 0) {
			payload = "quoted value \"" + std::to_string(row_idx) + "\"\ncontinuation";
			output << row_idx << ",\"quoted value \"\"" << row_idx << "\"\"\ncontinuation\"\r\n";
		} else {
			payload = "value-" + std::to_string(row_idx);
			output << row_idx << "," << payload << "\r\n";
		}
		expected.push_back({static_cast<int64_t>(row_idx), std::move(payload)});
	}
	output.close();
	REQUIRE(output.good());
	return expected;
}

static void WriteCSVBytes(const string &path, const string &contents) {
	std::ofstream output(path, std::ios::out | std::ios::trunc | std::ios::binary);
	REQUIRE(output.good());
	output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
	output.close();
	REQUIRE(output.good());
}

static vector<CSVTestRow> ExecuteAllCSVSplits(DuckDB &db, Connection &connection, const PlannedCSVScan &planned,
                                              idx_t first_node_id) {
	vector<CSVTestRow> result;
	for (idx_t split_idx = 0; split_idx < planned.splits.size(); split_idx++) {
		auto rows = ExecuteCSVAssignment(db, connection, planned.worker_plan, planned.splits[split_idx],
		                                 first_node_id + split_idx);
		result.insert(result.end(), std::make_move_iterator(rows.begin()), std::make_move_iterator(rows.end()));
	}
	std::sort(result.begin(), result.end());
	return result;
}

static vector<CSVTestRow> MaterializeCSVRows(MaterializedQueryResult &result) {
	vector<CSVTestRow> rows;
	rows.reserve(result.RowCount());
	for (idx_t row_idx = 0; row_idx < result.RowCount(); row_idx++) {
		rows.push_back(
		    {result.GetValue(0, row_idx).GetValue<int64_t>(), result.GetValue(1, row_idx).GetValue<string>()});
	}
	return rows;
}

} // namespace

TEST_CASE("Distributed CSV byte ranges round-trip and execute explicitly", "[distributed][csv]") {
	const auto path = TestCreatePath("distributed_csv_ranges.csv");
	TestDeleteFile(path);
	auto expected = WriteCSVRangeFixture(path, 1000);

	DuckDB db(nullptr);
	Connection coordinator(db);
	const auto query = StringUtil::Format("SELECT id, payload FROM read_csv('%s', header=true, auto_detect=false, "
	                                      "columns={'id':'BIGINT','payload':'VARCHAR'}, union_by_name=true, "
	                                      "buffer_size=1024, max_line_size=256)",
	                                      path);
	auto planned = PlanCSVScan(db, coordinator, query, 4);
	REQUIRE(planned.splits.size() == 4);
	set<string> split_ids;
	idx_t estimated_bytes = 0;
	for (const auto &split : planned.splits) {
		REQUIRE(split.kind == distributed::ScanSplitKind::EXTENSION);
		REQUIRE_FALSE(split.empty);
		REQUIRE(split.extension_capability.extension_name == "vane_core");
		REQUIRE(split.extension_capability.capability.name == "read_csv");
		REQUIRE(split.extension_capability.capability.protocol_version == 1);
		REQUIRE(split.split_codec.name == "vane.csv-file-range");
		REQUIRE(split.split_codec.version == 1);
		REQUIRE(split_ids.insert(split.split_id).second);
		REQUIRE(split.estimated_bytes.IsValid());
		estimated_bytes += split.estimated_bytes.GetIndex();
	}
	REQUIRE(estimated_bytes > 0);

	Connection worker(db);
	vector<CSVTestRow> actual;
	vector<vector<CSVTestRow>> rows_by_split;
	rows_by_split.reserve(planned.splits.size());
	for (idx_t split_idx = 0; split_idx < planned.splits.size(); split_idx++) {
		auto rows = ExecuteCSVAssignment(db, worker, planned.worker_plan, planned.splits[split_idx], 100 + split_idx);
		REQUIRE_FALSE(rows.empty());
		REQUIRE(rows.size() < expected.size());
		rows_by_split.push_back(rows);
		actual.insert(actual.end(), std::make_move_iterator(rows.begin()), std::make_move_iterator(rows.end()));
	}
	std::sort(actual.begin(), actual.end());
	REQUIRE(actual == expected);

	auto merged = CSVSplitBatch(planned.splits[0]);
	merged.Merge(CSVSplitBatch(planned.splits[2]));
	auto merged_rows = ExecuteCSVAssignment(db, worker, planned.worker_plan, merged, 200);
	std::sort(merged_rows.begin(), merged_rows.end());
	vector<CSVTestRow> expected_merged = rows_by_split[0];
	expected_merged.insert(expected_merged.end(), rows_by_split[2].begin(), rows_by_split[2].end());
	std::sort(expected_merged.begin(), expected_merged.end());
	REQUIRE(merged_rows == expected_merged);

	// Empty work is an explicit legal assignment and never falls back to the coordinator file.
	auto empty =
	    distributed::ScanSplit::EmptyExtension(planned.splits[0].extension_capability, planned.splits[0].split_codec);
	REQUIRE(ExecuteCSVAssignment(db, worker, planned.worker_plan, empty, 300).empty());

	// Retrying the same round-tripped descriptor is deterministic.
	auto first_rows = ExecuteCSVAssignment(db, worker, planned.worker_plan, planned.splits[0], 400);
	auto retry_rows = ExecuteCSVAssignment(db, worker, planned.worker_plan, planned.splits[0], 401);
	std::sort(first_rows.begin(), first_rows.end());
	std::sort(retry_rows.begin(), retry_rows.end());
	REQUIRE(first_rows == retry_rows);

	// An applied worker plan remains self-contained when it is cloned again. In
	// particular, several ranges of one coordinator file must not change the
	// already-bound CSV reader options during deserialization.
	auto applied_plan = std::make_shared<PhysicalPlan>(Allocator::DefaultAllocator());
	auto &applied_scan = distributed::ClonePhysicalPlanRootIntoPlanOrThrow(
	                         planned.worker_plan, *applied_plan, "distributed_csv_applied_source", worker.context.get())
	                         .Cast<PhysicalTableScan>();
	applied_scan.extra_info.scan_node_id = optional_idx(402);
	applied_scan.extra_info.scan_group_id = optional_idx(402);
	applied_plan->SetRoot(applied_scan);
	unordered_map<idx_t, distributed::ScanSplitBatch> applied_assignment;
	applied_assignment.emplace(402, merged);
	string applied_error;
	REQUIRE(distributed::ApplyScanSplitBatchesToPlan(*applied_plan, applied_assignment, &applied_error));
	REQUIRE(applied_error.empty());
	REQUIRE(distributed::ValidateDistributedScanSplitsApplied(*applied_plan));

	auto recloned_plan = make_uniq<PhysicalPlan>(Allocator::DefaultAllocator());
	PhysicalTableScan *recloned_scan = nullptr;
	{
		Connection transient_connection(db);
		recloned_scan = &distributed::ClonePhysicalPlanRootIntoPlanOrThrow(applied_plan, *recloned_plan,
		                                                                   "distributed_csv_applied_clone",
		                                                                   transient_connection.context.get())
		                     .Cast<PhysicalTableScan>();
		recloned_plan->SetRoot(*recloned_scan);
	}
	string recloned_validation_error;
	REQUIRE_FALSE(distributed::ValidateDistributedScanSplitsApplied(*recloned_plan, &recloned_validation_error));
	REQUIRE(StringUtil::Contains(recloned_validation_error, "no explicit worker split assignment"));

	// The bind-level authorization survives the clone, but the runtime fence
	// requires every attempt to inject its descriptor again. A different
	// assignment cannot broaden the already-applied worker plan.
	unordered_map<idx_t, distributed::ScanSplitBatch> wrong_recloned_assignment;
	wrong_recloned_assignment.emplace(402, CSVSplitBatch(planned.splits[1]));
	string wrong_recloned_error;
	REQUIRE_THROWS_WITH(
	    distributed::ApplyScanSplitBatchesToPlan(*recloned_plan, wrong_recloned_assignment, &wrong_recloned_error),
	    Catch::Matchers::Contains("can only replay its original split assignment"));

	unordered_map<idx_t, distributed::ScanSplitBatch> recloned_assignment;
	recloned_assignment.emplace(402, merged);
	string recloned_apply_error;
	REQUIRE(distributed::ApplyScanSplitBatchesToPlan(*recloned_plan, recloned_assignment, &recloned_apply_error));
	REQUIRE(recloned_apply_error.empty());
	REQUIRE(distributed::ValidateDistributedScanSplitsApplied(*recloned_plan));
	auto recloned_result = ExecuteCSVPlan(worker, std::move(recloned_plan), *recloned_scan);
	auto recloned_rows = MaterializeCSVRows(*recloned_result);
	std::sort(recloned_rows.begin(), recloned_rows.end());
	REQUIRE(recloned_rows == merged_rows);

	auto detached_result = ExecuteDetachedCSVPlan(worker, planned.worker_plan);
	REQUIRE(detached_result != nullptr);
	REQUIRE(detached_result->HasError());
	REQUIRE(StringUtil::Contains(detached_result->GetError(), "no explicit split assignment"));

	TestDeleteFile(path);
}

TEST_CASE("Distributed CSV rejects malformed and unauthorized splits", "[distributed][csv]") {
	const auto allowed_path = TestCreatePath("distributed_csv_allowed.csv");
	const auto unknown_path = TestCreatePath("distributed_csv_unknown.csv");
	TestDeleteFile(allowed_path);
	TestDeleteFile(unknown_path);
	WriteCSVRangeFixture(allowed_path, 256);
	WriteCSVRangeFixture(unknown_path, 256);

	DuckDB db(nullptr);
	Connection connection(db);
	auto make_query = [](const string &path) {
		return StringUtil::Format("SELECT id, payload FROM read_csv('%s', header=true, auto_detect=false, "
		                          "columns={'id':'BIGINT','payload':'VARCHAR'}, buffer_size=1024, max_line_size=256)",
		                          path);
	};
	auto allowed = PlanCSVScan(db, connection, make_query(allowed_path), 2);
	auto unknown = PlanCSVScan(db, connection, make_query(unknown_path), 2);
	REQUIRE(allowed.splits.size() == 2);
	REQUIRE(unknown.splits.size() == 2);

	auto apply_invalid = [&](distributed::ScanSplit split, idx_t node_id) {
		auto plan = make_uniq<PhysicalPlan>(Allocator::DefaultAllocator());
		auto &scan = distributed::ClonePhysicalPlanRootIntoPlanOrThrow(
		                 allowed.worker_plan, *plan, "distributed_csv_invalid", connection.context.get())
		                 .Cast<PhysicalTableScan>();
		scan.extra_info.scan_node_id = optional_idx(node_id);
		scan.extra_info.scan_group_id = optional_idx(node_id);
		plan->SetRoot(scan);
		unordered_map<idx_t, distributed::ScanSplitBatch> assignments;
		assignments.emplace(node_id, CSVSplitBatch(std::move(split)));
		string error;
		distributed::ApplyScanSplitBatchesToPlan(*plan, assignments, &error);
	};

	auto invalid_id = allowed.splits[0];
	invalid_id.split_id = "not-canonical";
	REQUIRE_THROWS_WITH(apply_invalid(std::move(invalid_id), 500), Catch::Matchers::Contains("does not match"));

	auto corrupt_payload = allowed.splits[0];
	corrupt_payload.extension_payload = "corrupt";
	REQUIRE_THROWS(apply_invalid(std::move(corrupt_payload), 501));

	auto empty_payload = allowed.splits[0];
	empty_payload.extension_payload.clear();
	REQUIRE_THROWS_WITH(apply_invalid(std::move(empty_payload), 502), Catch::Matchers::Contains("payload is empty"));

	auto unauthorized = unknown.splits[0];
	REQUIRE_THROWS_WITH(apply_invalid(std::move(unauthorized), 503),
	                    Catch::Matchers::Contains("outside its worker bind"));

	auto duplicate_batch = CSVSplitBatch(allowed.splits[0]);
	duplicate_batch.splits.push_back(allowed.splits[0]);
	REQUIRE_THROWS_WITH(duplicate_batch.Validate(), Catch::Matchers::Contains("appears more than once"));

	TestDeleteFile(allowed_path);
	TestDeleteFile(unknown_path);
}

TEST_CASE("Distributed CSV byte ranges preserve exact record boundaries", "[distributed][csv]") {
	struct BoundaryFixture {
		string suffix;
		string header;
		string row_suffix;
		string payload;
		idx_t boundary_offset;
	};
	const vector<BoundaryFixture> fixtures = {{"lf", "id,value\n", "\n", "xxxxxx", 64},
	                                          {"crlf", "id,value\r\n", "\r\n", "xxxxx", 64}};

	DuckDB db(nullptr);
	Connection coordinator(db);
	Connection worker(db);
	for (idx_t fixture_idx = 0; fixture_idx < fixtures.size(); fixture_idx++) {
		const auto &fixture = fixtures[fixture_idx];
		const auto path = TestCreatePath("distributed_csv_boundary_" + fixture.suffix + ".csv");
		TestDeleteFile(path);
		string contents = fixture.header;
		vector<CSVTestRow> expected;
		for (idx_t row_idx = 0; row_idx < 26; row_idx++) {
			contents += StringUtil::Format("%03llu,%s%s", row_idx, fixture.payload, fixture.row_suffix);
			expected.push_back({static_cast<int64_t>(row_idx), fixture.payload});
		}
		WriteCSVBytes(path, contents);
		if (fixture.suffix == "lf") {
			REQUIRE(contents.substr(fixture.boundary_offset, 4) == "005,");
		} else {
			REQUIRE(contents.substr(fixture.boundary_offset - 1, 3) == "\r\n0");
		}

		const auto query =
		    StringUtil::Format("SELECT id, value FROM read_csv('%s', header=true, auto_detect=false, "
		                       "columns={'id':'BIGINT','value':'VARCHAR'}, buffer_size=256, max_line_size=64)",
		                       path);
		auto planned = PlanCSVScan(db, coordinator, query, 4);
		REQUIRE(planned.splits.size() == 4);
		REQUIRE(ExecuteAllCSVSplits(db, worker, planned, 600 + fixture_idx * 10) == expected);
		TestDeleteFile(path);
	}
}

TEST_CASE("Distributed CSV byte ranges preserve a record immediately after the header boundary", "[distributed][csv]") {
	struct HeaderBoundaryFixture {
		string suffix;
		string newline;
		idx_t header_value_size;
	};
	const vector<HeaderBoundaryFixture> fixtures = {
	    {"lf_exact", "\n", 60}, {"lf_at_next_range", "\n", 61}, {"crlf_half", "\r\n", 60}};

	DuckDB db(nullptr);
	Connection coordinator(db);
	Connection worker(db);
	for (idx_t fixture_idx = 0; fixture_idx < fixtures.size(); fixture_idx++) {
		const auto &fixture = fixtures[fixture_idx];
		const auto path = TestCreatePath("distributed_csv_header_boundary_" + fixture.suffix + ".csv");
		TestDeleteFile(path);
		// With max_line_size=64 the first planned range ends at byte 64. The LF header ends exactly at that boundary;
		// the CRLF header places its LF at byte 64. Both cases must leave the first data row with exactly one owner.
		string contents = "id," + string(fixture.header_value_size, 'x') + fixture.newline;
		vector<CSVTestRow> expected;
		for (idx_t row_idx = 0; row_idx < 20; row_idx++) {
			contents += StringUtil::Format("%03llu,value%s", row_idx, fixture.newline);
			expected.push_back({static_cast<int64_t>(row_idx), "value"});
		}
		WriteCSVBytes(path, contents);
		if (fixture.suffix == "lf_exact") {
			REQUIRE(contents[63] == '\n');
			REQUIRE(contents.substr(64, 4) == "000,");
		} else if (fixture.suffix == "lf_at_next_range") {
			REQUIRE(contents[64] == '\n');
			REQUIRE(contents.substr(65, 4) == "000,");
		} else {
			REQUIRE(contents.substr(63, 6) == "\r\n000,");
		}

		const auto query =
		    StringUtil::Format("SELECT id, payload FROM read_csv('%s', header=true, auto_detect=false, "
		                       "columns={'id':'BIGINT','payload':'VARCHAR'}, buffer_size=256, max_line_size=64)",
		                       path);
		auto planned = PlanCSVScan(db, coordinator, query, 4);
		REQUIRE(planned.splits.size() == 4);
		REQUIRE(ExecuteAllCSVSplits(db, worker, planned, 650 + fixture_idx * 10) == expected);
		TestDeleteFile(path);
	}
}

TEST_CASE("Distributed CSV whole-file union splits preserve reader state and file indexes", "[distributed][csv]") {
	const auto first_path = TestCreatePath("distributed_csv_file_index_0.csv");
	const auto second_path = TestCreatePath("distributed_csv_file_index_1.csv");
	TestDeleteFile(first_path);
	TestDeleteFile(second_path);
	WriteCSVBytes(first_path, "id,payload\n1,first\n");
	WriteCSVBytes(second_path, "id|payload\n2|second\n");

	DuckDB db(nullptr);
	Connection coordinator(db);
	Connection worker(db);
	const auto query = StringUtil::Format(
	    "SELECT file_index, filename FROM read_csv_auto(['%s', '%s'], union_by_name=true, filename=true)", first_path,
	    second_path);
	auto planned = PlanCSVScan(db, coordinator, query, 2);
	REQUIRE(planned.splits.size() == 2);
	auto actual = ExecuteAllCSVSplits(db, worker, planned, 680);
	const vector<CSVTestRow> expected = {{0, first_path}, {1, second_path}};
	REQUIRE(actual == expected);
	auto merged = CSVSplitBatch(planned.splits[0]);
	merged.Merge(CSVSplitBatch(planned.splits[1]));
	auto merged_rows = ExecuteCSVAssignment(db, worker, planned.worker_plan, merged, 685);
	std::sort(merged_rows.begin(), merged_rows.end());
	REQUIRE(merged_rows == expected);

	const auto pruned_query = StringUtil::Format(
	    "SELECT id, payload FROM read_csv_auto(['%s', '%s'], union_by_name=true, filename=true) WHERE filename = '%s'",
	    first_path, second_path, second_path);
	auto pruned = PlanCSVScan(db, coordinator, pruned_query, 2);
	REQUIRE(pruned.splits.size() == 1);
	const vector<CSVTestRow> expected_pruned = {{2, "second"}};
	REQUIRE(ExecuteAllCSVSplits(db, worker, pruned, 690) == expected_pruned);

	const auto fully_pruned_query =
	    StringUtil::Format("SELECT id, payload FROM read_csv_auto(['%s', '%s'], union_by_name=true, filename=true) "
	                       "WHERE filename = 'not-a-bound-file.csv'",
	                       first_path, second_path);
	auto fully_pruned = PlanCSVScan(db, coordinator, fully_pruned_query, 2);
	REQUIRE(fully_pruned.splits.size() == 1);
	REQUIRE(fully_pruned.splits[0].kind == distributed::ScanSplitKind::EXTENSION);
	REQUIRE(fully_pruned.splits[0].empty);
	REQUIRE(ExecuteAllCSVSplits(db, worker, fully_pruned, 695).empty());

	TestDeleteFile(first_path);
	TestDeleteFile(second_path);
}

TEST_CASE("Distributed CSV handles empty input and rejects unsafe range modes", "[distributed][csv]") {
	const auto path = TestCreatePath("distributed_csv_planning_modes.csv");
	TestDeleteFile(path);
	DuckDB db(nullptr);
	Connection connection(db);

	WriteCSVBytes(path, "");
	const auto empty_file_query = [&](const string &options) {
		return StringUtil::Format("SELECT id, payload FROM read_csv('%s', header=false, auto_detect=false, "
		                          "columns={'id':'BIGINT','payload':'VARCHAR'}, %s)",
		                          path, options);
	};
	REQUIRE_THROWS_WITH(PlanCSVScan(db, connection, empty_file_query("parallel=false"), 2),
	                    Catch::Matchers::Contains("requires parallel=true"));

	WriteCSVBytes(path, "id,payload\n");
	const auto explicit_query = [&](const string &options) {
		return StringUtil::Format("SELECT id, payload FROM read_csv('%s', header=true, auto_detect=false, "
		                          "columns={'id':'BIGINT','payload':'VARCHAR'}, %s)",
		                          path, options);
	};
	auto empty = PlanCSVScan(db, connection, explicit_query("buffer_size=256, max_line_size=64"), 4);
	Connection worker(db);
	REQUIRE(ExecuteAllCSVSplits(db, worker, empty, 700).empty());

	WriteCSVRangeFixture(path, 128);
	REQUIRE_THROWS_WITH(PlanCSVScan(db, connection, explicit_query("parallel=false"), 2),
	                    Catch::Matchers::Contains("requires parallel=true"));
	REQUIRE_THROWS_WITH(PlanCSVScan(db, connection, explicit_query("skip=1"), 2),
	                    Catch::Matchers::Contains("does not support skip_rows"));
	REQUIRE_THROWS_WITH(PlanCSVScan(db, connection, explicit_query("store_rejects=true"), 2),
	                    Catch::Matchers::Contains("does not support store_rejects"));

	TestDeleteFile(path);
}
