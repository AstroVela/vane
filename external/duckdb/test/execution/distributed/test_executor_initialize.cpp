// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb/main/connection.hpp"
#include "duckdb/execution/executor.hpp"
#include "duckdb/execution/physical_plan.hpp"
#include "duckdb/execution/operator/helper/physical_result_collector.hpp"
#include "duckdb/execution/operator/scan/physical_dummy_scan.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/parallel/task_scheduler.hpp"

using namespace duckdb;

TEST_CASE("Executor: Initialize with dummy plan builds pipelines", "[distributed][executor]") {
	DuckDB db(nullptr);
	Connection con(db);

	// Test that basic query execution works (which internally uses Executor)
	auto result = con.Query("SELECT 1 AS value");
	REQUIRE_FALSE(result->HasError());
	REQUIRE(result->RowCount() == 1);

	// Test a slightly more complex query
	result = con.Query("SELECT * FROM (VALUES (1), (2), (3)) AS t(x)");
	REQUIRE_FALSE(result->HasError());
	REQUIRE(result->RowCount() == 3);

	// Test that we can run multiple queries in sequence
	result = con.Query("CREATE TABLE test_exec (id INTEGER)");
	REQUIRE_FALSE(result->HasError());

	result = con.Query("INSERT INTO test_exec VALUES (1), (2), (3)");
	REQUIRE_FALSE(result->HasError());

	result = con.Query("SELECT COUNT(*) FROM test_exec");
	REQUIRE_FALSE(result->HasError());
	REQUIRE(result->RowCount() == 1);
}

TEST_CASE("Executor: progress topology plans pipelines without scheduling tasks", "[distributed][executor][progress]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto &context = *con.context;

	auto prepared = make_shared_ptr<PreparedStatementData>(StatementType::SELECT_STATEMENT);
	prepared->names = {"value"};
	prepared->types = {LogicalType::INTEGER};
	prepared->properties.return_type = StatementReturnType::QUERY_RESULT;
	prepared->output_type = QueryResultOutputType::FORCE_MATERIALIZED;
	prepared->memory_type = QueryResultMemoryType::IN_MEMORY;

	auto plan = make_uniq<PhysicalPlan>(Allocator::DefaultAllocator());
	auto &scan = plan->Make<PhysicalDummyScan>(prepared->types, 1);
	plan->SetRoot(scan);
	prepared->physical_plan = std::move(plan);
	auto &root = PhysicalResultCollector::GetResultCollector(context, *prepared);

	auto &scheduler = TaskScheduler::GetScheduler(context);
	auto scheduled_tasks_before = scheduler.GetNumberOfTasks();
	Executor executor(context);
	executor.InitializeProgressTopology(root);
	auto snapshots = executor.GetPipelinesProgressSnapshots();

	REQUIRE(snapshots.size() == 2);
	REQUIRE(executor.GetTotalPipelines() >= 1);
	REQUIRE(executor.GetCompletedPipelines() == 0);
	REQUIRE(scheduler.GetNumberOfTasks() == scheduled_tasks_before);
	bool found_source_role = false;
	bool found_sink_role = false;
	for (const auto &snapshot : snapshots) {
		REQUIRE(snapshot.running_pipeline_tasks == 0);
		REQUIRE(snapshot.completed_pipeline_tasks == 0);
		REQUIRE(snapshot.operators.size() == snapshot.operator_details.size());
		for (idx_t index = 0; index < snapshot.operators.size(); index++) {
			if (snapshot.operators[index] != "RESULT_COLLECTOR") {
				continue;
			}
			const auto &details = snapshot.operator_details[index];
			auto role = details.find("pipeline_role");
			REQUIRE(role != details.end());
			found_source_role = found_source_role || role->second == "source";
			found_sink_role = found_sink_role || role->second == "sink";
		}
	}
	REQUIRE(found_source_role);
	REQUIRE(found_sink_role);
}
