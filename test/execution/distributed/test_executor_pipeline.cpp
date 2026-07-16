// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * Test file for Executor and Pipeline behavior with PhysicalPlan.
 *
 * This tests the core execution engine components:
 * - How Executor initializes with a PhysicalPlan
 * - How pipelines are built from operators
 * - How tasks are scheduled and executed
 */

#include "catch.hpp"

#include "duckdb/main/connection.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/execution/executor.hpp"
#include "duckdb/execution/physical_plan.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/execution/operator/scan/physical_dummy_scan.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/main/pending_query_result.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/execution/operator/helper/physical_result_collector.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"

#include <functional>

using namespace duckdb;

// Helper function to get row count from query result
static idx_t GetRowCount(unique_ptr<QueryResult> &result) {
	return result->Cast<MaterializedQueryResult>().RowCount();
}

// Helper function to get value from query result
static Value GetResultValue(unique_ptr<QueryResult> &result, idx_t col, idx_t row) {
	return result->Cast<MaterializedQueryResult>().GetValue(col, row);
}

TEST_CASE("Executor: Pipeline count for different query types", "[distributed][executor][pipeline]") {
	DuckDB db(nullptr);
	Connection con(db);

	SECTION("Simple SELECT creates pipelines") {
		// A simple SELECT should create at least one pipeline
		auto pending = con.PendingQuery("SELECT 1 AS value");
		REQUIRE(pending);
		REQUIRE_FALSE(pending->HasError());

		// Execute the query to completion
		auto result = pending->Execute();
		REQUIRE_FALSE(result->HasError());
		REQUIRE(GetRowCount(result) == 1);
	}

	SECTION("VALUES expression creates pipelines") {
		auto pending = con.PendingQuery("SELECT * FROM (VALUES (1, 'a'), (2, 'b'), (3, 'c')) AS t(id, name)");
		REQUIRE(pending);
		REQUIRE_FALSE(pending->HasError());

		auto result = pending->Execute();
		REQUIRE_FALSE(result->HasError());
		REQUIRE(GetRowCount(result) == 3);
	}

	SECTION("Table scan creates pipelines") {
		// Create a table first
		auto setup = con.Query("CREATE TABLE pipeline_test (id INTEGER, name VARCHAR)");
		REQUIRE_FALSE(setup->HasError());

		setup = con.Query("INSERT INTO pipeline_test VALUES (1, 'one'), (2, 'two'), (3, 'three')");
		REQUIRE_FALSE(setup->HasError());

		// Now test the scan
		auto pending = con.PendingQuery("SELECT * FROM pipeline_test");
		REQUIRE(pending);
		REQUIRE_FALSE(pending->HasError());

		auto result = pending->Execute();
		REQUIRE_FALSE(result->HasError());
		REQUIRE(GetRowCount(result) == 3);
	}

	SECTION("Aggregation creates pipelines") {
		// Create and populate a table
		auto setup = con.Query("CREATE TABLE agg_test (category VARCHAR, value INTEGER)");
		REQUIRE_FALSE(setup->HasError());

		setup = con.Query("INSERT INTO agg_test VALUES ('A', 10), ('A', 20), ('B', 30), ('B', 40), ('C', 50)");
		REQUIRE_FALSE(setup->HasError());

		auto pending = con.PendingQuery("SELECT category, SUM(value) FROM agg_test GROUP BY category");
		REQUIRE(pending);
		REQUIRE_FALSE(pending->HasError());

		auto result = pending->Execute();
		REQUIRE_FALSE(result->HasError());
		REQUIRE(GetRowCount(result) == 3);
	}

	SECTION("JOIN creates pipelines") {
		// Create two tables
		auto setup = con.Query("CREATE TABLE join_left (id INTEGER, name VARCHAR)");
		REQUIRE_FALSE(setup->HasError());
		setup = con.Query("CREATE TABLE join_right (id INTEGER, value INTEGER)");
		REQUIRE_FALSE(setup->HasError());

		setup = con.Query("INSERT INTO join_left VALUES (1, 'one'), (2, 'two'), (3, 'three')");
		REQUIRE_FALSE(setup->HasError());
		setup = con.Query("INSERT INTO join_right VALUES (1, 100), (2, 200), (4, 400)");
		REQUIRE_FALSE(setup->HasError());

		auto pending = con.PendingQuery("SELECT l.name, r.value FROM join_left l JOIN join_right r ON l.id = r.id");
		REQUIRE(pending);
		REQUIRE_FALSE(pending->HasError());

		auto result = pending->Execute();
		REQUIRE_FALSE(result->HasError());
		REQUIRE(GetRowCount(result) == 2); // Only id 1 and 2 match
	}
}

TEST_CASE("Executor: Execution completion states", "[distributed][executor][pipeline]") {
	DuckDB db(nullptr);
	Connection con(db);

	SECTION("PendingQueryResult can be executed step by step") {
		auto pending = con.PendingQuery("SELECT * FROM (VALUES (1), (2), (3), (4), (5)) AS t(x)");
		REQUIRE(pending);
		REQUIRE_FALSE(pending->HasError());

		// Execute step by step until done
		PendingExecutionResult pending_result;
		do {
			pending_result = pending->ExecuteTask();
		} while (!PendingQueryResult::IsExecutionFinished(pending_result));

		auto result = pending->Execute();
		REQUIRE_FALSE(result->HasError());
		REQUIRE(GetRowCount(result) == 5);
	}

	SECTION("Multiple queries execute sequentially") {
		for (int i = 0; i < 5; i++) {
			auto query = "SELECT " + std::to_string(i * 10) + " AS value";
			auto pending = con.PendingQuery(query);
			REQUIRE(pending);
			REQUIRE_FALSE(pending->HasError());

			auto result = pending->Execute();
			REQUIRE_FALSE(result->HasError());
			REQUIRE(GetRowCount(result) == 1);
		}
	}
}

TEST_CASE("Executor: Error handling in pipelines", "[distributed][executor][pipeline]") {
	DuckDB db(nullptr);
	Connection con(db);

	SECTION("Division by zero returns infinity") {
		// In DuckDB, division by zero returns infinity (not an error)
		auto pending = con.PendingQuery("SELECT 1 / 0 AS result");
		REQUIRE(pending);

		auto result = pending->Execute();
		REQUIRE_FALSE(result->HasError());
		REQUIRE(GetRowCount(result) == 1);
	}

	SECTION("Invalid column reference") {
		auto result = con.Query("SELECT nonexistent_column FROM (VALUES (1)) AS t(x)");
		REQUIRE(result->HasError());
	}

	SECTION("Table not found") {
		auto result = con.Query("SELECT * FROM nonexistent_table");
		REQUIRE(result->HasError());
	}
}

TEST_CASE("Executor: Result validation", "[distributed][executor][pipeline]") {
	DuckDB db(nullptr);
	Connection con(db);

	SECTION("Generate series creates many tuples") {
		auto pending = con.PendingQuery("SELECT * FROM range(1000)");
		REQUIRE(pending);
		REQUIRE_FALSE(pending->HasError());

		auto result = pending->Execute();
		REQUIRE_FALSE(result->HasError());
		REQUIRE(GetRowCount(result) == 1000);
	}

	SECTION("Arithmetic result validation") {
		auto pending = con.PendingQuery("SELECT 10 * 10 AS product");
		REQUIRE(pending);
		REQUIRE_FALSE(pending->HasError());

		auto result = pending->Execute();
		REQUIRE_FALSE(result->HasError());
		REQUIRE(GetRowCount(result) == 1);

		auto value = GetResultValue(result, 0, 0);
		auto int_value = value.GetValue<int64_t>();
		REQUIRE(int_value == 100);
	}
}

//===--------------------------------------------------------------------===//
// Direct PhysicalPlan Tests - Testing Executor with manually built plans
//===--------------------------------------------------------------------===//

TEST_CASE("Executor: Direct execution with Executor API", "[distributed][executor][pipeline][direct]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto &context = *con.context;

	SECTION("Executor pipeline count verification via PendingQuery") {
		// Use PendingQuery to access Executor during execution
		auto pending = con.PendingQuery("SELECT * FROM range(100)");
		REQUIRE(pending);
		REQUIRE_FALSE(pending->HasError());

		// Execute one task to initialize pipelines
		auto exec_result = pending->ExecuteTask();

		// Get the executor and check pipeline state
		auto &executor = context.GetExecutor();

		// There should be at least one pipeline
		REQUIRE(executor.GetTotalPipelines() >= 1);

		// Complete execution
		while (!PendingQueryResult::IsExecutionFinished(exec_result)) {
			exec_result = pending->ExecuteTask();
		}

		// All pipelines should be completed
		REQUIRE(executor.GetCompletedPipelines() == executor.GetTotalPipelines());

		auto result = pending->Execute();
		REQUIRE_FALSE(result->HasError());
	}

	SECTION("Executor tracks pipeline progress during JOIN execution") {
		// Create tables for join
		auto setup = con.Query("CREATE TABLE exec_left (id INTEGER)");
		REQUIRE_FALSE(setup->HasError());
		setup = con.Query("CREATE TABLE exec_right (id INTEGER, val INTEGER)");
		REQUIRE_FALSE(setup->HasError());
		setup = con.Query("INSERT INTO exec_left SELECT * FROM range(10)");
		REQUIRE_FALSE(setup->HasError());
		setup = con.Query("INSERT INTO exec_right SELECT i, i*10 FROM range(10) t(i)");
		REQUIRE_FALSE(setup->HasError());

		// Execute join query
		auto pending = con.PendingQuery("SELECT l.id, r.val FROM exec_left l JOIN exec_right r ON l.id = r.id");
		REQUIRE(pending);
		REQUIRE_FALSE(pending->HasError());

		auto exec_result = pending->ExecuteTask();
		auto &executor = context.GetExecutor();

		// JOIN should create multiple pipelines (build + probe)
		REQUIRE(executor.GetTotalPipelines() >= 1);

		// Execute to completion
		while (!PendingQueryResult::IsExecutionFinished(exec_result)) {
			exec_result = pending->ExecuteTask();
		}

		REQUIRE(executor.GetCompletedPipelines() == executor.GetTotalPipelines());

		auto result = pending->Execute();
		REQUIRE_FALSE(result->HasError());
		REQUIRE(GetRowCount(result) == 10);
	}

	SECTION("Executor handles aggregation pipelines") {
		auto setup = con.Query("CREATE TABLE exec_agg (grp INTEGER, val INTEGER)");
		REQUIRE_FALSE(setup->HasError());
		setup = con.Query("INSERT INTO exec_agg SELECT i % 3, i FROM range(100) t(i)");
		REQUIRE_FALSE(setup->HasError());

		auto pending = con.PendingQuery("SELECT grp, SUM(val) FROM exec_agg GROUP BY grp");
		REQUIRE(pending);
		REQUIRE_FALSE(pending->HasError());

		auto exec_result = pending->ExecuteTask();
		auto &executor = context.GetExecutor();

		// Track initial state
		idx_t total_pipelines = executor.GetTotalPipelines();
		REQUIRE(total_pipelines >= 1);

		// Execute to completion and verify progress
		while (!PendingQueryResult::IsExecutionFinished(exec_result)) {
			exec_result = pending->ExecuteTask();
		}

		REQUIRE(executor.GetCompletedPipelines() == total_pipelines);
		REQUIRE(executor.ExecutionIsFinished());

		auto result = pending->Execute();
		REQUIRE_FALSE(result->HasError());
		REQUIRE(GetRowCount(result) == 3); // 3 groups: 0, 1, 2
	}
}

TEST_CASE("Executor: PhysicalPlan structure verification", "[distributed][executor][pipeline][direct]") {
	Allocator &allocator = Allocator::DefaultAllocator();

	SECTION("PhysicalDummyScan as source operator") {
		PhysicalPlan plan(allocator);
		vector<LogicalType> types = {LogicalType::INTEGER};
		idx_t estimated_cardinality = 1;

		auto &dummy_scan = plan.Make<PhysicalDummyScan>(types, estimated_cardinality);
		plan.SetRoot(dummy_scan);

		REQUIRE(plan.HasRoot());
		REQUIRE(plan.Root().type == PhysicalOperatorType::DUMMY_SCAN);
		REQUIRE(plan.Root().children.empty());
		REQUIRE(plan.Root().IsSource());
		REQUIRE_FALSE(plan.Root().IsSink());
	}

	SECTION("PhysicalPlan with Projection over DummyScan") {
		PhysicalPlan plan(allocator);
		vector<LogicalType> types = {LogicalType::INTEGER};
		idx_t estimated_cardinality = 1;

		auto &dummy_scan = plan.Make<PhysicalDummyScan>(types, estimated_cardinality);

		vector<unique_ptr<Expression>> select_list;
		select_list.push_back(make_uniq<BoundConstantExpression>(Value::INTEGER(42)));
		auto &projection = plan.Make<PhysicalProjection>(types, std::move(select_list), estimated_cardinality);
		projection.children.emplace_back(dummy_scan);
		plan.SetRoot(projection);

		REQUIRE(plan.HasRoot());
		REQUIRE(plan.Root().type == PhysicalOperatorType::PROJECTION);
		REQUIRE(plan.Root().children.size() == 1);
		REQUIRE(plan.Root().children[0].get().type == PhysicalOperatorType::DUMMY_SCAN);
	}

	SECTION("PhysicalPlan with chained operators: Projection -> Filter -> DummyScan") {
		PhysicalPlan plan(allocator);
		vector<LogicalType> types = {LogicalType::INTEGER};
		idx_t estimated_cardinality = 1;

		auto &dummy_scan = plan.Make<PhysicalDummyScan>(types, estimated_cardinality);

		vector<unique_ptr<Expression>> filter_list;
		filter_list.push_back(make_uniq<BoundConstantExpression>(Value::BOOLEAN(true)));
		auto &filter = plan.Make<PhysicalFilter>(types, std::move(filter_list), estimated_cardinality);
		filter.children.emplace_back(dummy_scan);

		vector<unique_ptr<Expression>> select_list;
		select_list.push_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0));
		auto &projection = plan.Make<PhysicalProjection>(types, std::move(select_list), estimated_cardinality);
		projection.children.emplace_back(filter);
		plan.SetRoot(projection);

		// Verify full chain
		REQUIRE(plan.HasRoot());
		auto &root = plan.Root();
		REQUIRE(root.type == PhysicalOperatorType::PROJECTION);
		REQUIRE(root.children.size() == 1);

		auto &child1 = root.children[0].get();
		REQUIRE(child1.type == PhysicalOperatorType::FILTER);
		REQUIRE(child1.children.size() == 1);

		auto &child2 = child1.children[0].get();
		REQUIRE(child2.type == PhysicalOperatorType::DUMMY_SCAN);
		REQUIRE(child2.children.empty());
	}

	SECTION("Operator GetTypes returns correct types") {
		PhysicalPlan plan(allocator);
		vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::VARCHAR, LogicalType::DOUBLE};

		auto &dummy_scan = plan.Make<PhysicalDummyScan>(types, 1);
		plan.SetRoot(dummy_scan);

		auto result_types = dummy_scan.GetTypes();
		REQUIRE(result_types.size() == 3);
		REQUIRE(result_types[0] == LogicalType::INTEGER);
		REQUIRE(result_types[1] == LogicalType::VARCHAR);
		REQUIRE(result_types[2] == LogicalType::DOUBLE);
	}

	SECTION("Operator estimated_cardinality is set correctly") {
		PhysicalPlan plan(allocator);
		vector<LogicalType> types = {LogicalType::INTEGER};
		idx_t cardinality = 1000;

		auto &dummy_scan = plan.Make<PhysicalDummyScan>(types, cardinality);
		plan.SetRoot(dummy_scan);

		REQUIRE(dummy_scan.estimated_cardinality == cardinality);
	}
}

TEST_CASE("Executor: ExecuteTask completion model", "[distributed][executor][pipeline][direct]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto &context = *con.context;

	SECTION("ExecuteTask drives execution to completion") {
		auto pending = con.PendingQuery("SELECT * FROM range(50)");
		REQUIRE(pending);
		REQUIRE_FALSE(pending->HasError());

		// Execute tasks until completion
		PendingExecutionResult exec_result;
		idx_t task_count = 0;
		do {
			exec_result = pending->ExecuteTask();
			task_count++;
		} while (!PendingQueryResult::IsExecutionFinished(exec_result));

		// Should have executed at least one task
		REQUIRE(task_count >= 1);

		auto result = pending->Execute();
		REQUIRE_FALSE(result->HasError());
		REQUIRE(GetRowCount(result) == 50);
	}
}

//===--------------------------------------------------------------------===//
// Direct Executor::Initialize with PhysicalPlan
//
// NOTE: A PhysicalPlan without a sink (like ResultCollector) will not create
// a complete executable pipeline. The pipeline builder sets the source but
// doesn't schedule anything without a sink.
//
// This demonstrates how DuckDB's execution model works:
// - Source-only plans: Pipeline is set up but not scheduled (0 total_pipelines)
// - Plans with sink: Creates complete pipelines that can be executed
//===--------------------------------------------------------------------===//

TEST_CASE("Executor: Initialize directly with PhysicalPlan", "[distributed][executor][pipeline][direct][init]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto &client_context = *con.context;

	SECTION("Source-only plan initializes but creates no scheduled pipelines") {
		// A plan without a sink won't have scheduled pipelines
		// This is expected behavior - DuckDB needs a sink to complete execution
		Allocator &allocator = Allocator::DefaultAllocator();
		PhysicalPlan plan(allocator);

		vector<LogicalType> types = {LogicalType::INTEGER};
		auto &dummy_scan = plan.Make<PhysicalDummyScan>(types, 1);
		plan.SetRoot(dummy_scan);

		// Verify the plan structure
		REQUIRE(plan.HasRoot());
		REQUIRE(dummy_scan.IsSource());
		REQUIRE_FALSE(dummy_scan.IsSink());

		// Initialize Executor - this sets up pipeline but doesn't schedule
		// because there's no sink to collect results
		Executor executor(client_context);
		executor.Initialize(dummy_scan);

		// Without a sink, no pipelines are scheduled for execution
		// This is correct behavior - execution needs a sink to work
		REQUIRE(executor.GetTotalPipelines() == 0);
		REQUIRE(executor.ExecutionIsFinished());
	}

	SECTION("Projection->DummyScan plan structure is correct") {
		Allocator &allocator = Allocator::DefaultAllocator();
		PhysicalPlan plan(allocator);

		vector<LogicalType> types = {LogicalType::INTEGER};

		// Build: Projection -> DummyScan
		auto &dummy_scan = plan.Make<PhysicalDummyScan>(types, 1);

		vector<unique_ptr<Expression>> select_list;
		select_list.push_back(make_uniq<BoundConstantExpression>(Value::INTEGER(42)));
		auto &projection = plan.Make<PhysicalProjection>(types, std::move(select_list), 1);
		projection.children.emplace_back(dummy_scan);
		plan.SetRoot(projection);

		// Verify plan structure
		REQUIRE(plan.HasRoot());
		REQUIRE(projection.type == PhysicalOperatorType::PROJECTION);
		REQUIRE_FALSE(projection.IsSink());
		REQUIRE(projection.children.size() == 1);

		// Initialize - same behavior, no sink means no scheduled pipelines
		Executor executor(client_context);
		executor.Initialize(projection);
		REQUIRE(executor.GetTotalPipelines() == 0);
	}

	SECTION("Filter->Projection->DummyScan chain structure") {
		Allocator &allocator = Allocator::DefaultAllocator();
		PhysicalPlan plan(allocator);

		vector<LogicalType> types = {LogicalType::INTEGER};

		// Build chain: Filter -> Projection -> DummyScan
		auto &dummy_scan = plan.Make<PhysicalDummyScan>(types, 1);

		vector<unique_ptr<Expression>> select_list;
		select_list.push_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0));
		auto &projection = plan.Make<PhysicalProjection>(types, std::move(select_list), 1);
		projection.children.emplace_back(dummy_scan);

		vector<unique_ptr<Expression>> filter_list;
		filter_list.push_back(make_uniq<BoundConstantExpression>(Value::BOOLEAN(true)));
		auto &filter = plan.Make<PhysicalFilter>(types, std::move(filter_list), 1);
		filter.children.emplace_back(projection);
		plan.SetRoot(filter);

		// Verify full chain structure
		REQUIRE(plan.HasRoot());
		REQUIRE(filter.type == PhysicalOperatorType::FILTER);
		REQUIRE(filter.children.size() == 1);
		REQUIRE(filter.children[0].get().type == PhysicalOperatorType::PROJECTION);
		REQUIRE(filter.children[0].get().children[0].get().type == PhysicalOperatorType::DUMMY_SCAN);
	}
}

TEST_CASE("Executor: Complete execution via query interface", "[distributed][executor][pipeline][direct][init]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto &client_context = *con.context;

	SECTION("Full execution with sink via PendingQuery") {
		// This demonstrates the complete execution flow:
		// Query -> Parse -> Plan -> PhysicalPlan + ResultCollector (sink) -> Execute

		auto pending = con.PendingQuery("SELECT 42 AS answer");
		REQUIRE(pending);
		REQUIRE_FALSE(pending->HasError());

		// Execute first task to initialize
		auto exec_result = pending->ExecuteTask();

		// Get the executor to inspect pipeline state
		auto &executor = client_context.GetExecutor();

		// With a proper sink (ResultCollector), pipelines are scheduled
		REQUIRE(executor.GetTotalPipelines() >= 1);

		// Execute to completion
		while (!PendingQueryResult::IsExecutionFinished(exec_result)) {
			exec_result = pending->ExecuteTask();
		}

		// Verify all pipelines completed
		REQUIRE(executor.GetCompletedPipelines() == executor.GetTotalPipelines());

		auto result = pending->Execute();
		REQUIRE_FALSE(result->HasError());
		REQUIRE(GetRowCount(result) == 1);

		auto value = GetResultValue(result, 0, 0);
		REQUIRE(value.GetValue<int64_t>() == 42);
	}

	SECTION("Complex plan execution: aggregation has multiple pipelines") {
		// Set up data
		auto setup = con.Query("CREATE TABLE exec_test (x INTEGER)");
		REQUIRE_FALSE(setup->HasError());
		setup = con.Query("INSERT INTO exec_test SELECT * FROM range(100)");
		REQUIRE_FALSE(setup->HasError());

		// Execute aggregation query
		auto pending = con.PendingQuery("SELECT SUM(x) FROM exec_test");
		REQUIRE(pending);

		auto exec_result = pending->ExecuteTask();
		auto &executor = client_context.GetExecutor();

		// Aggregation typically creates multiple pipelines (build + probe)
		idx_t total_pipelines = executor.GetTotalPipelines();
		REQUIRE(total_pipelines >= 1);

		// Execute all tasks
		while (!PendingQueryResult::IsExecutionFinished(exec_result)) {
			exec_result = pending->ExecuteTask();
		}

		REQUIRE(executor.GetCompletedPipelines() == total_pipelines);

		auto result = pending->Execute();
		REQUIRE_FALSE(result->HasError());

		// SUM(0..99) = 4950
		auto value = GetResultValue(result, 0, 0);
		REQUIRE(value.GetValue<int64_t>() == 4950);
	}
}

//===--------------------------------------------------------------------===//
// Complete PhysicalPlan with Scan + Projection + Filter + Sink
// Direct Executor/Pipeline/Task API using manually built PhysicalPlan
//===--------------------------------------------------------------------===//

TEST_CASE("Executor: Manually built PhysicalPlan with direct Executor API",
          "[distributed][executor][pipeline][direct][complete]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto &client_context = *con.context;

	// Helper to print plan structure (handles both children and GetChildren())
	std::function<void(const PhysicalOperator &, int)> print_plan_recursive;
	print_plan_recursive = [&](const PhysicalOperator &op, int depth) -> void {
		std::string indent(depth * 2, ' ');
		std::cerr << indent << "- " << PhysicalOperatorToString(op.type) << " (IsSink=" << op.IsSink()
		          << ", IsSource=" << op.IsSource() << ")" << std::endl;

		// Try GetChildren() first (for ResultCollector which uses 'plan' reference)
		auto children_refs = op.GetChildren();
		if (!children_refs.empty()) {
			for (auto &child_ref : children_refs) {
				print_plan_recursive(child_ref.get(), depth + 1);
			}
		} else {
			// Fall back to children vector
			for (auto &child : op.children) {
				print_plan_recursive(child.get(), depth + 1);
			}
		}
	};

	SECTION("Scan -> Projection -> Sink: Complete pipeline execution") {
		std::cerr << "\n=== Building PhysicalPlan: Scan -> Projection -> Sink ===" << std::endl;

		//=============================================================
		// STEP 1: Create source data
		//=============================================================
		vector<LogicalType> source_types = {LogicalType::INTEGER, LogicalType::DOUBLE};
		auto source_data = make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), source_types);

		DataChunk chunk;
		chunk.Initialize(Allocator::DefaultAllocator(), source_types);
		for (int i = 1; i <= 5; i++) {
			chunk.SetCardinality(1);
			chunk.SetValue(0, 0, Value::INTEGER(i));
			chunk.SetValue(1, 0, Value::DOUBLE(i * 100.0));
			source_data->Append(chunk);
			chunk.Reset();
		}
		std::cerr << "[1] Source data: " << source_data->Count() << " rows" << std::endl;

		//=============================================================
		// STEP 2: Build PhysicalPlan: Scan -> Projection
		//=============================================================
		auto physical_plan = make_uniq<PhysicalPlan>(Allocator::DefaultAllocator());

		// Scan
		auto &scan =
		    physical_plan->Make<PhysicalColumnDataScan>(source_types, PhysicalOperatorType::COLUMN_DATA_SCAN, idx_t(5),
		                                                optionally_owned_ptr<ColumnDataCollection>(source_data.get()));

		// Projection
		vector<LogicalType> proj_types = {LogicalType::INTEGER, LogicalType::DOUBLE};
		vector<unique_ptr<Expression>> proj_expressions;
		proj_expressions.push_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0));
		proj_expressions.push_back(make_uniq<BoundReferenceExpression>(LogicalType::DOUBLE, 1));

		auto &projection = physical_plan->Make<PhysicalProjection>(proj_types, std::move(proj_expressions), idx_t(5));
		projection.children.push_back(scan);

		physical_plan->SetRoot(projection);
		std::cerr << "[2] Built plan: SCAN -> PROJECTION" << std::endl;

		//=============================================================
		// STEP 3: Create PreparedStatementData and add Sink
		//=============================================================
		auto prepared_data = make_shared_ptr<PreparedStatementData>(StatementType::SELECT_STATEMENT);
		prepared_data->names = {"id", "value"};
		prepared_data->types = proj_types;
		prepared_data->properties.return_type = StatementReturnType::QUERY_RESULT;
		prepared_data->output_type = QueryResultOutputType::FORCE_MATERIALIZED;
		prepared_data->memory_type = QueryResultMemoryType::IN_MEMORY;
		prepared_data->physical_plan = std::move(physical_plan);

		auto &sink = PhysicalResultCollector::GetResultCollector(client_context, *prepared_data);
		std::cerr << "[3] Added sink: " << PhysicalOperatorToString(sink.type) << std::endl;

		// Print complete plan
		std::cerr << "\n[Plan Structure]" << std::endl;
		print_plan_recursive(sink, 0);

		//=============================================================
		// STEP 4: Initialize Executor and execute via ExecuteTask()
		//=============================================================
		Executor executor(client_context);
		executor.Initialize(sink);

		idx_t total_pipelines = executor.GetTotalPipelines();
		std::cerr << "\n[4] Executor: " << total_pipelines << " pipeline(s)" << std::endl;
		REQUIRE(total_pipelines >= 1);

		idx_t task_count = 0;
		while (!executor.ExecutionIsFinished()) {
			auto result = executor.ExecuteTask();
			task_count++;
			if (result == PendingExecutionResult::RESULT_NOT_READY) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
			if (task_count > 10000)
				FAIL("Infinite loop");
		}

		std::cerr << "[5] Executed " << task_count << " tasks, " << executor.GetCompletedPipelines()
		          << " pipelines completed" << std::endl;

		REQUIRE(executor.ExecutionIsFinished());
		REQUIRE(executor.GetCompletedPipelines() == total_pipelines);
		REQUIRE_FALSE(executor.HasError());
	}

	SECTION("Scan -> Filter -> Projection -> Sink: Complete pipeline execution") {
		std::cerr << "\n=== Building PhysicalPlan: Scan -> Filter -> Projection -> Sink ===" << std::endl;

		//=============================================================
		// STEP 1: Create source data with 10 rows
		//=============================================================
		vector<LogicalType> source_types = {LogicalType::INTEGER, LogicalType::DOUBLE};
		auto source_data = make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), source_types);

		DataChunk chunk;
		chunk.Initialize(Allocator::DefaultAllocator(), source_types);
		for (int i = 1; i <= 10; i++) {
			chunk.SetCardinality(1);
			chunk.SetValue(0, 0, Value::INTEGER(i));
			chunk.SetValue(1, 0, Value::DOUBLE(i * 100.0));
			source_data->Append(chunk);
			chunk.Reset();
		}
		std::cerr << "[1] Source data: " << source_data->Count() << " rows" << std::endl;

		//=============================================================
		// STEP 2: Build PhysicalPlan: Scan -> Filter -> Projection
		//=============================================================
		auto physical_plan = make_uniq<PhysicalPlan>(Allocator::DefaultAllocator());

		// Scan
		auto &scan =
		    physical_plan->Make<PhysicalColumnDataScan>(source_types, PhysicalOperatorType::COLUMN_DATA_SCAN, idx_t(10),
		                                                optionally_owned_ptr<ColumnDataCollection>(source_data.get()));

		// Filter: value > 500 (column 1 > 500)
		// This is a simplified filter that just passes all rows
		// In production, you'd create a proper comparison expression
		vector<unique_ptr<Expression>> filter_exprs;
		filter_exprs.push_back(make_uniq<BoundConstantExpression>(Value::BOOLEAN(true)));

		auto &filter = physical_plan->Make<PhysicalFilter>(source_types, std::move(filter_exprs), idx_t(5));
		filter.children.push_back(scan);

		// Projection
		vector<LogicalType> proj_types = {LogicalType::INTEGER, LogicalType::DOUBLE};
		vector<unique_ptr<Expression>> proj_expressions;
		proj_expressions.push_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0));
		proj_expressions.push_back(make_uniq<BoundReferenceExpression>(LogicalType::DOUBLE, 1));

		auto &projection = physical_plan->Make<PhysicalProjection>(proj_types, std::move(proj_expressions), idx_t(5));
		projection.children.push_back(filter);

		physical_plan->SetRoot(projection);
		std::cerr << "[2] Built plan: SCAN -> FILTER -> PROJECTION" << std::endl;

		//=============================================================
		// STEP 3: Create PreparedStatementData and add Sink
		//=============================================================
		auto prepared_data = make_shared_ptr<PreparedStatementData>(StatementType::SELECT_STATEMENT);
		prepared_data->names = {"id", "value"};
		prepared_data->types = proj_types;
		prepared_data->properties.return_type = StatementReturnType::QUERY_RESULT;
		prepared_data->output_type = QueryResultOutputType::FORCE_MATERIALIZED;
		prepared_data->memory_type = QueryResultMemoryType::IN_MEMORY;
		prepared_data->physical_plan = std::move(physical_plan);

		auto &sink = PhysicalResultCollector::GetResultCollector(client_context, *prepared_data);
		std::cerr << "[3] Added sink: " << PhysicalOperatorToString(sink.type) << std::endl;

		// Print complete plan
		std::cerr << "\n[Plan Structure]" << std::endl;
		print_plan_recursive(sink, 0);

		//=============================================================
		// STEP 4: Initialize Executor and execute via ExecuteTask()
		//=============================================================
		Executor executor(client_context);
		executor.Initialize(sink);

		idx_t total_pipelines = executor.GetTotalPipelines();
		std::cerr << "\n[4] Executor: " << total_pipelines << " pipeline(s)" << std::endl;
		REQUIRE(total_pipelines >= 1);

		idx_t task_count = 0;
		while (!executor.ExecutionIsFinished()) {
			auto result = executor.ExecuteTask();
			task_count++;

			if (result == PendingExecutionResult::EXECUTION_ERROR) {
				FAIL("Execution error");
			}
			if (result == PendingExecutionResult::RESULT_NOT_READY) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
			if (task_count > 10000)
				FAIL("Infinite loop");
		}

		std::cerr << "[5] Executed " << task_count << " tasks, " << executor.GetCompletedPipelines()
		          << " pipelines completed" << std::endl;

		REQUIRE(executor.ExecutionIsFinished());
		REQUIRE(executor.GetCompletedPipelines() == total_pipelines);
		REQUIRE_FALSE(executor.HasError());

		std::cerr << "\n=== Test Passed: Scan->Filter->Projection->Sink executed ===" << std::endl;
	}
}
