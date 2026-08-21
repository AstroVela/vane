// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb.hpp"
#include "duckdb/common/allocator.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/distributed/pipeline_node/translator_scan.hpp"
#include "duckdb/execution/distributed/plan/scan_task.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/physical_plan.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/table/distributed_sequence.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/main/prepared_statement_data.hpp"

#include <algorithm>

using namespace duckdb;

namespace {

struct PlannedRangeScan {
	distributed::DuckPhysicalPlanRef worker_plan;
	vector<distributed::ScanTaskDescriptor> tasks;
};

static PlannedRangeScan PlanRangeScan(DuckDB &db, Connection &connection, const string &query, idx_t worker_slots) {
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
	PlannedRangeScan result;
	result.worker_plan = distributed::MakeTableScanPlan(coordinator_scan);
	auto task_set = distributed::MakeTableScanTasks(coordinator_scan, config, db.instance);
	REQUIRE_FALSE(task_set.known_empty);
	result.tasks = std::move(task_set.tasks);
	return result;
}

static vector<Value> ExecuteRangeAssignment(Connection &connection, const distributed::DuckPhysicalPlanRef &worker_plan,
                                            const distributed::ScanTaskDescriptor &descriptor, idx_t scan_node_id) {
	auto execution_plan = make_uniq<PhysicalPlan>(Allocator::DefaultAllocator());
	auto &scan = distributed::ClonePhysicalPlanRootIntoPlanOrThrow(worker_plan, *execution_plan, "distributed_range",
	                                                               connection.context.get())
	                 .Cast<PhysicalTableScan>();
	scan.extra_info.scan_node_id = optional_idx(scan_node_id);
	scan.extra_info.scan_group_id = optional_idx(scan_node_id);
	execution_plan->SetRoot(scan);

	unordered_map<idx_t, distributed::ScanTaskDescriptor> assignments;
	assignments.emplace(scan_node_id, descriptor);
	string apply_error;
	REQUIRE(distributed::ApplyScanTasksToPlan(*execution_plan, assignments, &apply_error));
	REQUIRE(apply_error.empty());
	REQUIRE(distributed::ValidateDistributedScanTasksApplied(*execution_plan));

	auto prepared = make_shared_ptr<PreparedStatementData>(StatementType::SELECT_STATEMENT);
	prepared->names = scan.names;
	prepared->types = scan.GetTypes();
	prepared->properties.return_type = StatementReturnType::QUERY_RESULT;
	prepared->output_type = QueryResultOutputType::FORCE_MATERIALIZED;
	prepared->memory_type = QueryResultMemoryType::IN_MEMORY;
	prepared->physical_plan = std::move(execution_plan);
	PendingQueryParameters parameters;
	auto pending =
	    connection.context->PendingQueryPreparedStatementNoRebind("test:distributed_range", prepared, parameters);
	REQUIRE(pending != nullptr);
	REQUIRE_FALSE(pending->HasError());
	auto query_result = pending->Execute();
	REQUIRE(query_result != nullptr);
	REQUIRE_NO_FAIL(*query_result);
	auto materialized = dynamic_cast<MaterializedQueryResult *>(query_result.get());
	REQUIRE(materialized != nullptr);
	vector<Value> result;
	result.reserve(materialized->RowCount());
	for (idx_t row_index = 0; row_index < materialized->RowCount(); row_index++) {
		result.push_back(materialized->GetValue(0, row_index));
	}
	return result;
}

static vector<int64_t> ExecuteIntegerAssignments(Connection &connection, const PlannedRangeScan &planned) {
	vector<int64_t> result;
	for (idx_t descriptor_index = 0; descriptor_index < planned.tasks.size(); descriptor_index++) {
		auto values = ExecuteRangeAssignment(connection, planned.worker_plan, planned.tasks[descriptor_index],
		                                     100 + descriptor_index);
		for (const auto &value : values) {
			result.push_back(value.GetValue<int64_t>());
		}
	}
	std::sort(result.begin(), result.end());
	return result;
}

} // namespace

TEST_CASE("Distributed integer range plans exact index shards", "[distributed][range]") {
	DuckDB db(nullptr);
	Connection connection(db);
	auto planned = PlanRangeScan(db, connection, "SELECT * FROM range(0, 10)", 4);
	REQUIRE(planned.tasks.size() == 4);
	idx_t planned_cardinality = 0;
	for (idx_t task_index = 0; task_index < planned.tasks.size(); task_index++) {
		const auto &descriptor = planned.tasks[task_index];
		REQUIRE(descriptor.kind == distributed::ScanTaskKind::EXTENSION);
		REQUIRE(descriptor.extension_tasks.size() == 1);
		REQUIRE(descriptor.extension_tasks[0].task_id == std::to_string(task_index));
		REQUIRE(descriptor.task_codec.name == DISTRIBUTED_SEQUENCE_TASK_CODEC);
		REQUIRE(descriptor.task_codec.version == DISTRIBUTED_SEQUENCE_TASK_CODEC_VERSION);
		planned_cardinality += descriptor.estimated_cardinality;
	}
	REQUIRE(planned_cardinality == 10);
	REQUIRE(ExecuteIntegerAssignments(connection, planned) == vector<int64_t> {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

	auto non_contiguous = planned.tasks[0];
	non_contiguous.Merge(planned.tasks[2]);
	auto non_contiguous_values = ExecuteRangeAssignment(connection, planned.worker_plan, non_contiguous, 200);
	vector<int64_t> actual;
	for (const auto &value : non_contiguous_values) {
		actual.push_back(value.GetValue<int64_t>());
	}
	REQUIRE(actual == vector<int64_t> {0, 1, 2, 6, 7});
}

TEST_CASE("Distributed generate_series preserves inclusive negative steps", "[distributed][range]") {
	DuckDB db(nullptr);
	Connection connection(db);
	auto planned = PlanRangeScan(db, connection, "SELECT * FROM generate_series(5, -1, -2)", 3);
	REQUIRE(planned.tasks.size() == 3);
	REQUIRE(ExecuteIntegerAssignments(connection, planned) == vector<int64_t> {-1, 1, 3, 5});
}

TEST_CASE("Distributed range installs an explicit empty assignment", "[distributed][range]") {
	DuckDB db(nullptr);
	Connection connection(db);
	auto planned = PlanRangeScan(db, connection, "SELECT * FROM range(0)", 8);
	REQUIRE(planned.tasks.size() == 1);
	REQUIRE(planned.tasks[0].extension_tasks.empty());
	REQUIRE(ExecuteRangeAssignment(connection, planned.worker_plan, planned.tasks[0], 300).empty());
}

TEST_CASE("Distributed timestamp range splits only indexable intervals", "[distributed][range]") {
	DuckDB db(nullptr);
	Connection connection(db);
	auto fixed = PlanRangeScan(
	    db, connection, "SELECT * FROM range(TIMESTAMP '2026-01-01', TIMESTAMP '2026-01-06', INTERVAL 1 DAY)", 3);
	REQUIRE(fixed.tasks.size() == 3);
	vector<string> fixed_values;
	for (idx_t task_index = 0; task_index < fixed.tasks.size(); task_index++) {
		for (const auto &value :
		     ExecuteRangeAssignment(connection, fixed.worker_plan, fixed.tasks[task_index], 400 + task_index)) {
			fixed_values.push_back(value.ToString());
		}
	}
	std::sort(fixed_values.begin(), fixed_values.end());
	auto fixed_native =
	    connection.Query("SELECT * FROM range(TIMESTAMP '2026-01-01', TIMESTAMP '2026-01-06', INTERVAL 1 DAY)");
	REQUIRE_NO_FAIL(*fixed_native);
	vector<string> fixed_expected;
	for (idx_t row_index = 0; row_index < fixed_native->RowCount(); row_index++) {
		fixed_expected.push_back(fixed_native->GetValue(0, row_index).ToString());
	}
	std::sort(fixed_expected.begin(), fixed_expected.end());
	REQUIRE(fixed_values == fixed_expected);

	auto calendar = PlanRangeScan(
	    db, connection,
	    "SELECT * FROM generate_series(TIMESTAMP '2026-01-31', TIMESTAMP '2026-05-31', INTERVAL 1 MONTH)", 8);
	REQUIRE(calendar.tasks.size() == 1);
	REQUIRE(calendar.tasks[0].extension_tasks.size() == 1);
	REQUIRE(calendar.tasks[0].estimated_cardinality == 0);
	auto distributed_values = ExecuteRangeAssignment(connection, calendar.worker_plan, calendar.tasks[0], 500);
	auto native = connection.Query(
	    "SELECT * FROM generate_series(TIMESTAMP '2026-01-31', TIMESTAMP '2026-05-31', INTERVAL 1 MONTH)");
	REQUIRE_NO_FAIL(*native);
	auto native_materialized = dynamic_cast<MaterializedQueryResult *>(native.get());
	REQUIRE(native_materialized != nullptr);
	REQUIRE(distributed_values.size() == native_materialized->RowCount());
	for (idx_t row_index = 0; row_index < distributed_values.size(); row_index++) {
		REQUIRE(distributed_values[row_index] == native_materialized->GetValue(0, row_index));
	}
}

TEST_CASE("Distributed range rejects malformed shard payloads", "[distributed][range]") {
	DuckDB db(nullptr);
	Connection connection(db);
	auto planned = PlanRangeScan(db, connection, "SELECT * FROM range(10)", 2);
	auto malformed = planned.tasks[0];
	malformed.extension_tasks[0].payload.push_back('\0');

	auto execution_plan = make_uniq<PhysicalPlan>(Allocator::DefaultAllocator());
	auto &scan = distributed::ClonePhysicalPlanRootIntoPlanOrThrow(
	                 planned.worker_plan, *execution_plan, "distributed_range_malformed", connection.context.get())
	                 .Cast<PhysicalTableScan>();
	scan.extra_info.scan_node_id = optional_idx(600);
	scan.extra_info.scan_group_id = optional_idx(600);
	execution_plan->SetRoot(scan);
	unordered_map<idx_t, distributed::ScanTaskDescriptor> assignments;
	assignments.emplace(600, malformed);
	string apply_error;
	REQUIRE_THROWS_WITH(distributed::ApplyScanTasksToPlan(*execution_plan, assignments, &apply_error),
	                    Catch::Matchers::Contains("trailing bytes"));
}
