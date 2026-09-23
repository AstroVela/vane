// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"
#include "test_helpers.hpp"

#include "duckdb.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/allocator.hpp"
#include "duckdb/common/local_file_system.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/distributed/pipeline_node/translator_scan.hpp"
#include "duckdb/execution/distributed/plan/scan_split.hpp"
#include "duckdb/execution/distributed/plan/fte_split_queue.hpp"
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

class NDJSONReadTrackingFileSystem : public LocalFileSystem {
public:
	explicit NDJSONReadTrackingFileSystem(string path_p) : path(std::move(path_p)) {
	}
	bool CanHandleFile(const string &candidate) override {
		return candidate == path;
	}
	string GetName() const override {
		return "NDJSONReadTrackingFileSystem";
	}
	void Read(FileHandle &handle, void *buffer, int64_t size, idx_t offset) override {
		bytes_read += size;
		LocalFileSystem::Read(handle, buffer, size, offset);
	}
	int64_t Read(FileHandle &handle, void *buffer, int64_t size) override {
		const auto result = LocalFileSystem::Read(handle, buffer, size);
		bytes_read += result;
		return result;
	}
	atomic<idx_t> bytes_read {0};

private:
	string path;
};

struct NDJSONTestRow {
	int64_t id;
	string payload;

	bool operator<(const NDJSONTestRow &other) const {
		return id < other.id;
	}
	bool operator==(const NDJSONTestRow &other) const {
		return id == other.id && payload == other.payload;
	}
};

struct PlannedNDJSONScan {
	distributed::DuckPhysicalPlanRef worker_plan;
	vector<distributed::ScanSplit> splits;
};

static distributed::ScanSplitBatch NDJSONSplitBatch(vector<distributed::ScanSplit> splits) {
	distributed::ScanSplitBatch result;
	result.splits = std::move(splits);
	result.Validate();
	return result;
}

static distributed::ScanSplitBatch NDJSONSplitBatch(const distributed::ScanSplit &split) {
	return NDJSONSplitBatch(vector<distributed::ScanSplit> {split});
}

static PlannedNDJSONScan PlanNDJSONScan(DuckDB &db, Connection &connection, const string &query, idx_t worker_slots) {
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
	PlannedNDJSONScan result;
	result.worker_plan = distributed::MakeTableScanPlan(scan);
	result.splits = distributed::MakeTableScanSplits(scan, config, db.instance);
	return result;
}

static vector<string> NDJSONScanOutputNames(const PhysicalTableScan &scan) {
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
		throw InternalException("Distributed NDJSON test scan has %llu output columns but %llu output types",
		                        output_ids.size(), scan.GetTypes().size());
	}

	vector<string> result;
	result.reserve(output_ids.size());
	for (const auto output_id : output_ids) {
		if (output_id >= scan.column_ids.size()) {
			throw InternalException("Distributed NDJSON test projection index %llu is out of range", output_id);
		}
		const auto &column = scan.column_ids[output_id];
		const auto primary_id = column.GetPrimaryIndex();
		if (column.IsVirtualColumn()) {
			auto entry = scan.virtual_columns.find(primary_id);
			if (entry == scan.virtual_columns.end()) {
				throw InternalException("Distributed NDJSON test virtual output column %llu is not bound", primary_id);
			}
			result.push_back(entry->second.name);
		} else {
			if (primary_id >= scan.names.size()) {
				throw InternalException("Distributed NDJSON test output column %llu has no name", primary_id);
			}
			result.push_back(scan.names[primary_id]);
		}
	}
	return result;
}

static unique_ptr<MaterializedQueryResult> ExecuteNDJSONPlan(Connection &connection, unique_ptr<PhysicalPlan> plan,
                                                             PhysicalTableScan &scan) {
	auto prepared = make_shared_ptr<PreparedStatementData>(StatementType::SELECT_STATEMENT);
	prepared->names = NDJSONScanOutputNames(scan);
	prepared->types = scan.GetTypes();
	prepared->properties.return_type = StatementReturnType::QUERY_RESULT;
	prepared->output_type = QueryResultOutputType::FORCE_MATERIALIZED;
	prepared->memory_type = QueryResultMemoryType::IN_MEMORY;
	prepared->physical_plan = std::move(plan);
	PendingQueryParameters parameters;
	auto pending =
	    connection.context->PendingQueryPreparedStatementNoRebind("test:distributed_ndjson", prepared, parameters);
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

static vector<NDJSONTestRow> ExecuteNDJSONAssignment(DuckDB &db, Connection &connection,
                                                     const distributed::DuckPhysicalPlanRef &worker_plan,
                                                     const distributed::ScanSplitBatch &input_batch, idx_t scan_node_id,
                                                     bool use_fte_queue = false) {
	auto batch = distributed::ScanSplitBatch::DeserializeFromBytes(input_batch.SerializeToBytes());
	auto execution_plan = make_uniq<PhysicalPlan>(Allocator::DefaultAllocator());
	PhysicalTableScan *scan = nullptr;
	{
		// Exercise the same short-lived deserialization context used by worker plan cloning.
		Connection transient_connection(db);
		scan = &distributed::ClonePhysicalPlanRootIntoPlanOrThrow(worker_plan, *execution_plan, "distributed_ndjson",
		                                                          transient_connection.context.get())
		            .Cast<PhysicalTableScan>();
		scan->extra_info.scan_node_id = optional_idx(scan_node_id);
		scan->extra_info.scan_group_id = optional_idx(scan_node_id);
		execution_plan->SetRoot(*scan);
		string apply_error;
		if (use_fte_queue) {
			auto queue = std::make_shared<distributed::FteSplitQueue>();
			for (auto &split : batch.splits) {
				queue->AddSplit(
				    distributed::TaskInput::make_scan_split_batch(NDJSONSplitBatch(split).SerializeToBytes()));
			}
			queue->NoMoreSplits();
			unordered_map<idx_t, std::shared_ptr<distributed::FteSplitQueue>> queues;
			queues.emplace(scan_node_id, std::move(queue));
			REQUIRE(distributed::ApplyFteScanSourceQueuesToPlan(*execution_plan, queues, &apply_error));
		} else {
			unordered_map<idx_t, distributed::ScanSplitBatch> assignments;
			assignments.emplace(scan_node_id, std::move(batch));
			REQUIRE(distributed::ApplyScanSplitBatchesToPlan(*execution_plan, assignments, &apply_error));
		}
		REQUIRE(apply_error.empty());
		REQUIRE(distributed::ValidateDistributedScanSplitsApplied(*execution_plan));
	}

	auto result = ExecuteNDJSONPlan(connection, std::move(execution_plan), *scan);
	vector<NDJSONTestRow> rows;
	rows.reserve(result->RowCount());
	for (idx_t row_idx = 0; row_idx < result->RowCount(); row_idx++) {
		rows.push_back(
		    {result->GetValue(0, row_idx).GetValue<int64_t>(), result->GetValue(1, row_idx).GetValue<string>()});
	}
	return rows;
}

static vector<NDJSONTestRow> ExecuteNDJSONAssignment(DuckDB &db, Connection &connection,
                                                     const distributed::DuckPhysicalPlanRef &worker_plan,
                                                     const distributed::ScanSplit &split, idx_t scan_node_id) {
	return ExecuteNDJSONAssignment(db, connection, worker_plan, NDJSONSplitBatch(split), scan_node_id);
}

} // namespace

TEST_CASE("Distributed NDJSON byte ranges preserve records and metadata", "[distributed][ndjson]") {
	const auto path = TestCreatePath("distributed_ndjson_ranges.ndjson");
	vector<NDJSONTestRow> expected;
	{
		std::ofstream output(path, std::ios::binary);
		for (idx_t i = 0; i < 45000; i++) {
			const auto payload = string(96, 'x') + "中文-" + std::to_string(i);
			output << "{\"id\":" << i << ",\"payload\":\"" << payload << "\"}";
			if (i != 44999) {
				output << (i % 2 == 0 ? "\r\n" : "\n");
			}
			if (i % 19 == 0) {
				output << "\n";
			}
			expected.push_back({static_cast<int64_t>(i), payload});
		}
	}
	DuckDB db(nullptr);
	Connection coordinator(db);
	REQUIRE_NO_FAIL(*coordinator.Query("LOAD json"));
	auto tracking = make_uniq<NDJSONReadTrackingFileSystem>(path);
	auto *tracker = tracking.get();
	db.instance->GetFileSystem().RegisterSubSystem(std::move(tracking));
	for (auto threads : {1, 4}) {
		Connection worker(db);
		REQUIRE_NO_FAIL(*worker.Query("SET threads=" + std::to_string(threads)));
		for (auto function : {"read_ndjson", "read_ndjson_auto"}) {
			const auto query =
			    StringUtil::Format("SELECT id, payload FROM %s('%s', maximum_object_size=65536)", function, path);
			auto planned = PlanNDJSONScan(db, coordinator, query, 4);
			REQUIRE(planned.splits.size() == 4);
			vector<NDJSONTestRow> rows;
			for (const auto &split : planned.splits) {
				REQUIRE(split.IsExtension());
				REQUIRE(split.extension_capability.extension_name == "json");
				tracker->bytes_read = 0;
				auto part = ExecuteNDJSONAssignment(db, worker, planned.worker_plan, split, 42);
				REQUIRE(tracker->bytes_read.load() > 0);
				REQUIRE(tracker->bytes_read.load() <= split.estimated_bytes.GetIndex() + 32768);
				auto replay = ExecuteNDJSONAssignment(db, worker, planned.worker_plan, split, 43);
				std::sort(part.begin(), part.end());
				std::sort(replay.begin(), replay.end());
				REQUIRE(part == replay);
				rows.insert(rows.end(), part.begin(), part.end());
			}
			std::sort(rows.begin(), rows.end());
			REQUIRE(rows == expected);
			auto merged =
			    ExecuteNDJSONAssignment(db, worker, planned.worker_plan, NDJSONSplitBatch(planned.splits), 44);
			std::sort(merged.begin(), merged.end());
			REQUIRE(merged == expected);
			auto fte_rows =
			    ExecuteNDJSONAssignment(db, worker, planned.worker_plan, NDJSONSplitBatch(planned.splits), 45, true);
			std::sort(fte_rows.begin(), fte_rows.end());
			REQUIRE(fte_rows == expected);
			auto empty = distributed::ScanSplit::EmptyExtension(planned.splits[0].extension_capability,
			                                                    planned.splits[0].split_codec);
			REQUIRE(ExecuteNDJSONAssignment(db, worker, planned.worker_plan, empty, 46).empty());
		}
	}
	TestDeleteFile(path);
}

TEST_CASE("Distributed JSON keeps non-NDJSON and compressed inputs whole", "[distributed][ndjson]") {
	DuckDB db(nullptr);
	Connection connection(db);
	REQUIRE_NO_FAIL(*connection.Query("LOAD json"));
	for (auto suffix : {".json", ".json.gz", ".json.zst"}) {
		const auto path = TestCreatePath(string("distributed_json_whole") + suffix);
		const auto compression = string(suffix) == ".json.gz"    ? "gzip"
		                         : string(suffix) == ".json.zst" ? "zstd"
		                                                         : "none";
		REQUIRE_NO_FAIL(
		    *connection.Query(StringUtil::Format("COPY (SELECT i AS id, repeat('x', 128) AS payload FROM range(40000) "
		                                         "t(i)) TO '%s' (FORMAT JSON, ARRAY true, COMPRESSION '%s')",
		                                         path, compression)));
		auto planned = PlanNDJSONScan(db, connection, "SELECT id, payload FROM read_json('" + path + "')", 8);
		REQUIRE(planned.splits.size() == 1);
		auto rows = ExecuteNDJSONAssignment(db, connection, planned.worker_plan, planned.splits[0], 55);
		REQUIRE(rows.size() == 40000);
		TestDeleteFile(path);
	}
}

namespace {
struct NDJSONTestFile {
	string path;
	idx_t ordinal;
	idx_t start;
	idx_t end;
	void Serialize(Serializer &serializer) const {
		map<string, Value> options;
		options["__vane_json_range_start"] = Value::UBIGINT(start);
		options["__vane_json_range_end"] = Value::UBIGINT(end);
		serializer.WritePropertyWithDefault(100, "path", path);
		serializer.WritePropertyWithDefault(101, "options", options);
		serializer.WritePropertyWithDefault(102, "ordinal", ordinal);
	}
};

static distributed::ScanSplit NDJSONTestRange(const distributed::ScanSplit &base, const string &path, idx_t start,
                                              idx_t end, idx_t ordinal = 0) {
	auto result = base;
	MemoryStream stream(Allocator::DefaultAllocator());
	BinarySerializer serializer(stream);
	serializer.Begin();
	serializer.WriteProperty<idx_t>(1, "version", 1);
	serializer.WriteProperty(2, "file", NDJSONTestFile {path, ordinal, start, end});
	serializer.End();
	result.split_id = "json:" + std::to_string(ordinal) + ":" + std::to_string(start) + ":" + std::to_string(end);
	result.extension_payload = string(reinterpret_cast<const char *>(stream.GetData()), stream.GetPosition());
	return result;
}

static bool TryNDJSONAssignment(Connection &connection, const distributed::DuckPhysicalPlanRef &worker_plan,
                                vector<distributed::ScanSplit> splits) {
	auto plan = make_uniq<PhysicalPlan>(Allocator::DefaultAllocator());
	auto &scan = distributed::ClonePhysicalPlanRootIntoPlanOrThrow(worker_plan, *plan, "ndjson-invalid",
	                                                               connection.context.get())
	                 .Cast<PhysicalTableScan>();
	scan.extra_info.scan_node_id = optional_idx(91);
	scan.extra_info.scan_group_id = optional_idx(91);
	plan->SetRoot(scan);
	unordered_map<idx_t, distributed::ScanSplitBatch> assignments;
	assignments.emplace(91, NDJSONSplitBatch(std::move(splits)));
	string error;
	try {
		return distributed::ApplyScanSplitBatchesToPlan(*plan, assignments, &error);
	} catch (const InvalidInputException &) {
		return false;
	}
}
} // namespace

TEST_CASE("NDJSON ranges align every byte boundary including empty intervals", "[distributed][ndjson]") {
	const auto path = TestCreatePath("ndjson_every_boundary.ndjson");
	const string input = "{\"id\":1,\"payload\":\"中文\\nline\"}\r\n\n{\"id\":2,\"payload\":\"last\"}";
	{
		std::ofstream file(path, std::ios::binary);
		file << input;
	}
	DuckDB db(nullptr);
	Connection connection(db);
	REQUIRE_NO_FAIL(*connection.Query("LOAD json"));
	auto planned = PlanNDJSONScan(db, connection, "SELECT id, payload FROM read_ndjson('" + path + "')", 4);
	REQUIRE(planned.splits.size() == 1);
	const vector<NDJSONTestRow> expected {{1, "中文\nline"}, {2, "last"}};
	for (auto threads : {1, 4}) {
		REQUIRE_NO_FAIL(*connection.Query("SET threads=" + std::to_string(threads)));
		for (idx_t boundary = 1; boundary < input.size(); boundary++) {
			vector<distributed::ScanSplit> splits {NDJSONTestRange(planned.splits[0], path, 0, boundary),
			                                       NDJSONTestRange(planned.splits[0], path, boundary, input.size())};
			vector<NDJSONTestRow> rows;
			for (const auto &split : splits) {
				auto part = ExecuteNDJSONAssignment(db, connection, planned.worker_plan, split, 90);
				rows.insert(rows.end(), part.begin(), part.end());
			}
			std::sort(rows.begin(), rows.end());
			REQUIRE(rows == expected);
		}
	}
	auto first = NDJSONTestRange(planned.splits[0], path, 0, 10);
	auto overlap = NDJSONTestRange(planned.splits[0], path, 9, 20);
	REQUIRE_FALSE(TryNDJSONAssignment(connection, planned.worker_plan, {first, overlap}));
	REQUIRE_FALSE(TryNDJSONAssignment(connection, planned.worker_plan, {first, planned.splits[0]}));
	REQUIRE_FALSE(
	    TryNDJSONAssignment(connection, planned.worker_plan, {NDJSONTestRange(planned.splits[0], path, 3, 3)}));
	REQUIRE_FALSE(
	    TryNDJSONAssignment(connection, planned.worker_plan, {NDJSONTestRange(planned.splits[0], path, 0, 10, 1)}));
	REQUIRE_FALSE(TryNDJSONAssignment(connection, planned.worker_plan,
	                                  {NDJSONTestRange(planned.splits[0], path + ".foreign", 0, 10)}));
	first.split_id += ":bad";
	REQUIRE_FALSE(TryNDJSONAssignment(connection, planned.worker_plan, {first}));
	TestDeleteFile(path);
}

TEST_CASE("Distributed NDJSON preserves repeated file ordinals and empty files", "[distributed][ndjson]") {
	const auto path = TestCreatePath("ndjson_repeated.ndjson");
	const auto empty_path = TestCreatePath("ndjson_empty.ndjson");
	{
		std::ofstream file(path);
		file << "{\"id\":7}\n";
		std::ofstream empty(empty_path);
	}
	DuckDB db(nullptr);
	Connection connection(db);
	REQUIRE_NO_FAIL(*connection.Query("LOAD json"));
	auto planned = PlanNDJSONScan(
	    db, connection, "SELECT id, file_index FROM read_ndjson(['" + path + "','" + empty_path + "','" + path + "'])",
	    4);
	REQUIRE(planned.splits.size() == 3);
	auto rows = ExecuteNDJSONAssignment(db, connection, planned.worker_plan, NDJSONSplitBatch(planned.splits), 92);
	REQUIRE(rows.size() == 2);
	set<string> ordinals;
	for (const auto &row : rows) {
		REQUIRE(row.id == 7);
		ordinals.insert(row.payload);
	}
	REQUIRE(ordinals == set<string> {"0", "2"});
	TestDeleteFile(path);
	TestDeleteFile(empty_path);
}

TEST_CASE("Applied NDJSON plans retain range authorization through serialization", "[distributed][ndjson]") {
	const auto path = TestCreatePath("ndjson_applied_serde.ndjson");
	const string input = "{\"id\":1,\"payload\":\"first\"}\n{\"id\":2,\"payload\":\"second\"}\n";
	{
		std::ofstream file(path);
		file << input;
	}
	DuckDB db(nullptr);
	Connection worker(db);
	REQUIRE_NO_FAIL(*worker.Query("LOAD json"));
	auto planned = PlanNDJSONScan(db, worker, "SELECT id, payload FROM read_ndjson('" + path + "')", 4);
	auto merged = NDJSONSplitBatch(vector<distributed::ScanSplit> {
	    NDJSONTestRange(planned.splits[0], path, 0, 12), NDJSONTestRange(planned.splits[0], path, 12, input.size())});
	auto applied_plan = std::make_shared<PhysicalPlan>(Allocator::DefaultAllocator());
	auto &applied_scan =
	    distributed::ClonePhysicalPlanRootIntoPlanOrThrow(planned.worker_plan, *applied_plan,
	                                                      "distributed_ndjson_applied_source", worker.context.get())
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

	distributed::DuckDBExecutionConfig config;
	config.set_distributed_worker_slots(32);
	auto replanned = distributed::MakeTableScanSplits(applied_scan, config, db.instance);
	REQUIRE(replanned.size() == merged.splits.size());
	for (idx_t i = 0; i < replanned.size(); i++) {
		REQUIRE(replanned[i].split_id == merged.splits[i].split_id);
		REQUIRE(replanned[i].extension_payload == merged.splits[i].extension_payload);
	}

	auto recloned_plan = make_uniq<PhysicalPlan>(Allocator::DefaultAllocator());
	PhysicalTableScan *recloned_scan = nullptr;
	{
		Connection transient_connection(db);
		recloned_scan = &distributed::ClonePhysicalPlanRootIntoPlanOrThrow(applied_plan, *recloned_plan,
		                                                                   "distributed_ndjson_applied_clone",
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
	wrong_recloned_assignment.emplace(402, NDJSONSplitBatch(planned.splits[0]));
	string wrong_recloned_error;
	REQUIRE_THROWS_WITH(
	    distributed::ApplyScanSplitBatchesToPlan(*recloned_plan, wrong_recloned_assignment, &wrong_recloned_error),
	    Catch::Matchers::Contains("Cannot replace an existing distributed JSON assignment"));

	unordered_map<idx_t, distributed::ScanSplitBatch> recloned_assignment;
	recloned_assignment.emplace(402, merged);
	string recloned_apply_error;
	REQUIRE(distributed::ApplyScanSplitBatchesToPlan(*recloned_plan, recloned_assignment, &recloned_apply_error));
	REQUIRE(recloned_apply_error.empty());
	REQUIRE(distributed::ValidateDistributedScanSplitsApplied(*recloned_plan));
	auto recloned_result = ExecuteNDJSONPlan(worker, std::move(recloned_plan), *recloned_scan);
	REQUIRE(recloned_result->RowCount() == 2);
	set<int64_t> ids;
	for (idx_t i = 0; i < recloned_result->RowCount(); i++) {
		ids.insert(recloned_result->GetValue(0, i).GetValue<int64_t>());
	}
	REQUIRE(ids == set<int64_t> {1, 2});

	TestDeleteFile(path);
}
