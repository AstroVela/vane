// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb.hpp"
#include "duckdb/common/allocator.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/distributed/pipeline_node/translator_scan.hpp"
#include "duckdb/execution/distributed/plan/fte_split_queue.hpp"
#include "duckdb/execution/distributed/plan/scan_split.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/physical_plan.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/distributed_table_function.hpp"
#include "duckdb/function/table/distributed_sequence.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/main/prepared_statement_data.hpp"

using namespace duckdb;

namespace {

struct PlannedPortableSource {
	distributed::DuckPhysicalPlanRef worker_plan;
	vector<distributed::ScanSplit> splits;
};

static distributed::ScanSplitBatch MakePortableSourceSplitBatch(vector<distributed::ScanSplit> splits) {
	distributed::ScanSplitBatch result;
	result.splits = std::move(splits);
	result.Validate();
	return result;
}

static distributed::ScanSplitBatch MakePortableSourceSplitBatch(const distributed::ScanSplit &split) {
	return MakePortableSourceSplitBatch(vector<distributed::ScanSplit> {split});
}

static PlannedPortableSource PlanPortableSource(DuckDB &db, Connection &connection, const string &query,
                                                idx_t worker_slots) {
	auto logical_plan = connection.ExtractPlan(query);
	REQUIRE(logical_plan != nullptr);
	PhysicalPlanGenerator generator(*connection.context);
	auto generated_plan = generator.Plan(std::move(logical_plan));
	REQUIRE(generated_plan != nullptr);
	REQUIRE(generated_plan->Root().type == PhysicalOperatorType::TABLE_SCAN);
	auto &coordinator_scan = generated_plan->Root().Cast<PhysicalTableScan>();
	REQUIRE(coordinator_scan.function.HasDistributedScanCallbacks());

	distributed::DuckDBExecutionConfig config;
	config.set_distributed_worker_slots(worker_slots);
	PlannedPortableSource result;
	result.worker_plan = distributed::MakeTableScanPlan(coordinator_scan);
	result.splits = distributed::MakeTableScanSplits(coordinator_scan, config, db.instance);
	return result;
}

static vector<vector<Value>> ExecutePortableAssignment(Connection &connection,
                                                       const distributed::DuckPhysicalPlanRef &worker_plan,
                                                       const distributed::ScanSplitBatch &batch, idx_t scan_node_id) {
	auto execution_plan = make_uniq<PhysicalPlan>(Allocator::DefaultAllocator());
	auto &scan = distributed::ClonePhysicalPlanRootIntoPlanOrThrow(
	                 worker_plan, *execution_plan, "distributed_portable_source", connection.context.get())
	                 .Cast<PhysicalTableScan>();
	scan.extra_info.scan_node_id = optional_idx(scan_node_id);
	scan.extra_info.scan_group_id = optional_idx(scan_node_id);
	execution_plan->SetRoot(scan);

	unordered_map<idx_t, distributed::ScanSplitBatch> assignments;
	assignments.emplace(scan_node_id, batch);
	string apply_error;
	REQUIRE(distributed::ApplyScanSplitBatchesToPlan(*execution_plan, assignments, &apply_error));
	REQUIRE(apply_error.empty());
	REQUIRE(distributed::ValidateDistributedScanSplitsApplied(*execution_plan));

	auto prepared = make_shared_ptr<PreparedStatementData>(StatementType::SELECT_STATEMENT);
	prepared->names = scan.names;
	prepared->types = scan.GetTypes();
	prepared->properties.return_type = StatementReturnType::QUERY_RESULT;
	prepared->output_type = QueryResultOutputType::FORCE_MATERIALIZED;
	prepared->memory_type = QueryResultMemoryType::IN_MEMORY;
	prepared->physical_plan = std::move(execution_plan);
	PendingQueryParameters parameters;
	auto pending = connection.context->PendingQueryPreparedStatementNoRebind("test:distributed_portable_source",
	                                                                         prepared, parameters);
	REQUIRE(pending != nullptr);
	REQUIRE_FALSE(pending->HasError());
	auto query_result = pending->Execute();
	REQUIRE(query_result != nullptr);
	REQUIRE_NO_FAIL(*query_result);
	auto materialized = dynamic_cast<MaterializedQueryResult *>(query_result.get());
	REQUIRE(materialized != nullptr);
	vector<vector<Value>> result;
	result.reserve(materialized->RowCount());
	for (idx_t row_index = 0; row_index < materialized->RowCount(); row_index++) {
		vector<Value> row;
		row.reserve(materialized->ColumnCount());
		for (idx_t column_index = 0; column_index < materialized->ColumnCount(); column_index++) {
			row.push_back(materialized->GetValue(column_index, row_index));
		}
		result.push_back(std::move(row));
	}
	return result;
}

static vector<vector<Value>> ExecutePortableAssignment(Connection &connection,
                                                       const distributed::DuckPhysicalPlanRef &worker_plan,
                                                       const distributed::ScanSplit &split, idx_t scan_node_id) {
	return ExecutePortableAssignment(connection, worker_plan, MakePortableSourceSplitBatch(split), scan_node_id);
}

} // namespace

TEST_CASE("Distributed repeat plans exact sequence splits", "[distributed][portable_source]") {
	DuckDB db(nullptr);
	Connection connection(db);
	auto planned = PlanPortableSource(db, connection, "SELECT * FROM repeat(NULL::INTEGER, 10)", 4);
	REQUIRE(planned.splits.size() == 4);

	idx_t row_count = 0;
	idx_t planned_cardinality = 0;
	for (idx_t split_index = 0; split_index < planned.splits.size(); split_index++) {
		const auto &split = planned.splits[split_index];
		REQUIRE(split.kind == distributed::ScanSplitKind::EXTENSION);
		REQUIRE_FALSE(split.empty);
		REQUIRE(split.split_id == std::to_string(split_index));
		REQUIRE(split.split_codec.name == DISTRIBUTED_SEQUENCE_SPLIT_CODEC);
		REQUIRE(split.split_codec.version == DISTRIBUTED_SEQUENCE_SPLIT_CODEC_VERSION);
		REQUIRE(split.estimated_cardinality.IsValid());
		REQUIRE_FALSE(split.estimated_bytes.IsValid());
		planned_cardinality += split.estimated_cardinality.GetIndex();
		auto rows = ExecutePortableAssignment(connection, planned.worker_plan, split, 100 + split_index);
		row_count += rows.size();
		for (const auto &row : rows) {
			REQUIRE(row.size() == 1);
			REQUIRE(row[0].IsNull());
		}
	}
	REQUIRE(planned_cardinality == 10);
	REQUIRE(row_count == 10);

	auto non_contiguous = MakePortableSourceSplitBatch(planned.splits[0]);
	non_contiguous.Merge(MakePortableSourceSplitBatch(planned.splits[2]));
	REQUIRE(ExecutePortableAssignment(connection, planned.worker_plan, non_contiguous, 200).size() == 5);
}

TEST_CASE("Distributed repeat_row preserves typed values across grouped splits", "[distributed][portable_source]") {
	DuckDB db(nullptr);
	Connection connection(db);
	auto planned =
	    PlanPortableSource(db, connection, "SELECT * FROM repeat_row(42, NULL::VARCHAR, [1, 2], num_rows=7)", 3);
	REQUIRE(planned.splits.size() == 3);

	auto assignment = MakePortableSourceSplitBatch(planned.splits[0]);
	assignment.Merge(MakePortableSourceSplitBatch(planned.splits[2]));
	assignment.Merge(MakePortableSourceSplitBatch(planned.splits[1]));
	auto rows = ExecutePortableAssignment(connection, planned.worker_plan, assignment, 300);
	REQUIRE(rows.size() == 7);
	for (const auto &row : rows) {
		REQUIRE(row.size() == 3);
		REQUIRE(row[0].GetValue<int32_t>() == 42);
		REQUIRE(row[1].IsNull());
		REQUIRE(row[2].ToString() == "[1, 2]");
	}
}

TEST_CASE("Distributed repeat installs an explicit empty assignment", "[distributed][portable_source]") {
	DuckDB db(nullptr);
	Connection connection(db);
	auto planned = PlanPortableSource(db, connection, "SELECT * FROM repeat('unused', 0)", 8);
	REQUIRE(planned.splits.size() == 1);
	REQUIRE(planned.splits[0].empty);
	REQUIRE(ExecutePortableAssignment(connection, planned.worker_plan, planned.splits[0], 400).empty());
}

TEST_CASE("Distributed standalone unnest uses one scheduler-owned source split", "[distributed][portable_source]") {
	DuckDB db(nullptr);
	Connection connection(db);
	auto planned = PlanPortableSource(db, connection, "SELECT * FROM unnest([1, NULL, 3])", 8);
	REQUIRE(planned.splits.size() == 1);
	const auto &split = planned.splits[0];
	REQUIRE(split.kind == distributed::ScanSplitKind::EXTENSION);
	REQUIRE_FALSE(split.empty);
	REQUIRE(split.split_id == "0");
	REQUIRE(split.extension_payload.empty());
	REQUIRE(split.split_codec.name == DISTRIBUTED_SINGLETON_SOURCE_SPLIT_CODEC);
	REQUIRE(split.split_codec.version == DISTRIBUTED_SINGLETON_SOURCE_SPLIT_CODEC_VERSION);

	auto rows = ExecutePortableAssignment(connection, planned.worker_plan, split, 500);
	REQUIRE(rows.size() == 3);
	REQUIRE(rows[0][0].GetValue<int32_t>() == 1);
	REQUIRE(rows[1][0].IsNull());
	REQUIRE(rows[2][0].GetValue<int32_t>() == 3);

	auto empty = PlanPortableSource(db, connection, "SELECT * FROM unnest([]::INTEGER[])", 8);
	REQUIRE(empty.splits.size() == 1);
	REQUIRE_FALSE(empty.splits[0].empty);
	REQUIRE(ExecutePortableAssignment(connection, empty.worker_plan, empty.splits[0], 501).empty());
}

TEST_CASE("Distributed singleton source installs an explicit empty assignment", "[distributed][portable_source]") {
	DuckDB db(nullptr);
	Connection connection(db);
	auto planned = PlanPortableSource(db, connection, "SELECT * FROM unnest([1, 2, 3])", 1);
	REQUIRE(planned.splits.size() == 1);
	auto explicit_empty =
	    distributed::ScanSplit::EmptyExtension(planned.splits[0].extension_capability, planned.splits[0].split_codec);
	REQUIRE(ExecutePortableAssignment(connection, planned.worker_plan, explicit_empty, 550).empty());
}

TEST_CASE("Distributed singleton source rejects malformed assignments", "[distributed][portable_source]") {
	DuckDB db(nullptr);
	Connection connection(db);
	auto planned = PlanPortableSource(db, connection, "SELECT * FROM unnest([1])", 1);
	auto malformed = planned.splits[0];
	malformed.split_id = "1";

	auto execution_plan = make_uniq<PhysicalPlan>(Allocator::DefaultAllocator());
	auto &scan = distributed::ClonePhysicalPlanRootIntoPlanOrThrow(
	                 planned.worker_plan, *execution_plan, "distributed_singleton_malformed", connection.context.get())
	                 .Cast<PhysicalTableScan>();
	scan.extra_info.scan_node_id = optional_idx(600);
	scan.extra_info.scan_group_id = optional_idx(600);
	execution_plan->SetRoot(scan);
	unordered_map<idx_t, distributed::ScanSplitBatch> assignments;
	assignments.emplace(600, MakePortableSourceSplitBatch(malformed));
	string apply_error;
	REQUIRE_THROWS_WITH(distributed::ApplyScanSplitBatchesToPlan(*execution_plan, assignments, &apply_error),
	                    Catch::Matchers::Contains("invalid split"));
}

TEST_CASE("Distributed optional-bind singleton accepts static and FTE assignments", "[distributed][portable_source]") {
	DuckDB db(nullptr);
	Connection connection(db);
	auto planned = PlanPortableSource(db, connection, "SELECT * FROM unnest([1])", 1);
	REQUIRE(planned.splits.size() == 1);
	auto split_batch = MakePortableSourceSplitBatch(planned.splits[0]);

	auto make_nullable_plan = [&](idx_t scan_node_id) {
		auto plan = distributed::ClonePhysicalPlanOrThrow(planned.worker_plan, "distributed_optional_bind_singleton",
		                                                  connection.context.get());
		auto &scan = plan->Root().Cast<PhysicalTableScan>();
		auto callbacks = scan.function.GetDistributedScanCallbacks();
		callbacks.bind_data_mode = TableFunctionDistributedBindDataMode::OPTIONAL;
		scan.function.SetDistributedScanCallbacks(std::move(callbacks));
		scan.bind_data.reset();
		scan.extra_info.scan_node_id = optional_idx(scan_node_id);
		scan.extra_info.scan_group_id = optional_idx(scan_node_id);
		return plan;
	};

	auto static_plan = make_nullable_plan(700);
	unordered_map<idx_t, distributed::ScanSplitBatch> static_assignments;
	static_assignments.emplace(700, split_batch);
	string static_error;
	REQUIRE(distributed::ApplyScanSplitBatchesToPlan(*static_plan, static_assignments, &static_error));
	REQUIRE(static_error.empty());
	REQUIRE(static_plan->Root().Cast<PhysicalTableScan>().distributed_scan_splits_applied);

	auto fte_plan = make_nullable_plan(701);
	auto queue = std::make_shared<distributed::FteSplitQueue>();
	queue->AddSplit(distributed::TaskInput::make_scan_split_batch(split_batch.SerializeToBytes()));
	queue->NoMoreSplits();
	unordered_map<idx_t, std::shared_ptr<distributed::FteSplitQueue>> queues;
	queues.emplace(701, std::move(queue));
	string fte_error;
	REQUIRE(distributed::ApplyFteScanSourceQueuesToPlan(*fte_plan, queues, &fte_error));
	REQUIRE(fte_error.empty());
	REQUIRE(fte_plan->Root().Cast<PhysicalTableScan>().distributed_scan_splits_applied);
}
