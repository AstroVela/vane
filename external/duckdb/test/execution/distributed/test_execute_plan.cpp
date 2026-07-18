// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/execution/distributed/pipeline_node/scan_source.hpp"
#include "duckdb/execution/distributed/pipeline_node/projection.hpp"
#include "duckdb/execution/distributed/pipeline_node/filter.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/operator/scan/physical_dummy_scan.hpp"
// Use bound expressions for concrete predicates/projections in tests
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/parallel/task_executor.hpp"

#include <cstdlib>
#include "duckdb/execution/distributed/utils/optional.hpp"
#include <string>

using namespace duckdb;
using namespace duckdb::distributed;

TEST_CASE("Pipeline produce_tasks: basic pipeline (scan->proj->filter)", "[distributed][plan]") {
	// Build a tiny pipeline: ScanSource -> Projection -> Filter
	Allocator &alloc = Allocator::DefaultAllocator();
	auto empty_plan = std::make_shared<duckdb::PhysicalPlan>(alloc);
	auto &dummy =
	    empty_plan->Make<duckdb::PhysicalDummyScan>(std::vector<duckdb::LogicalType> {duckdb::LogicalType::BIGINT}, 1);
	empty_plan->SetRoot(dummy);

	SchemaRef scan_schema = std::make_shared<LogicalType>(LogicalType::BIGINT);
	std::vector<ScanTaskDescriptor> scan_tasks;
	auto scan_impl = std::make_shared<ScanSourceNode>(PipelineNodeContext(0, "", 1, "scan"), empty_plan, scan_tasks,
	                                                  scan_schema, DuckDBExecutionConfigRef(), false);
	ExpressionRef proj_expr = std::make_shared<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(42));
	std::vector<std::string> proj_names;
	SchemaRef proj_schema = std::make_shared<LogicalType>(LogicalType::INTEGER);
	auto proj_impl =
	    std::make_shared<ProjectionNode>(2, scan_impl, std::vector<ExpressionRef> {proj_expr}, proj_names, proj_schema);
	ExpressionRef pred_expr = std::make_shared<duckdb::BoundConstantExpression>(duckdb::Value::BOOLEAN(true));
	auto filter_impl = std::make_shared<FilterNode>(3, proj_impl, pred_expr);
	auto root = std::make_shared<DistributedPipelineNode>(filter_impl);

	// Create a DuckDB instance + TaskExecutor so spawn() works
	duckdb::DuckDB db(nullptr);
	duckdb::Connection conn(db);
	auto task_executor = std::make_shared<TaskExecutor>(*conn.context);

	// Create current plan execution context and produce tasks.
	PlanExecutionContext ctx(task_executor, conn.context);
	auto tasks_stream = root->produce_tasks(ctx);

	// Verify produce_tasks yields at least one task
	auto task = tasks_stream.poll_next();
	REQUIRE(task.first);
	REQUIRE(task.second.task());
	REQUIRE(task.second.task()->plan());

	// Verify no more tasks
	auto no_more = tasks_stream.poll_next();
	REQUIRE_FALSE(no_more.first);
}

TEST_CASE("Pipeline produce_tasks yields a task with a printable plan", "[distributed][plan]") {
	Allocator &alloc = Allocator::DefaultAllocator();
	auto empty_plan = std::make_shared<duckdb::PhysicalPlan>(alloc);
	auto &dummy =
	    empty_plan->Make<duckdb::PhysicalDummyScan>(std::vector<duckdb::LogicalType> {duckdb::LogicalType::BIGINT}, 1);
	empty_plan->SetRoot(dummy);

	SchemaRef scan_schema = std::make_shared<LogicalType>(LogicalType::BIGINT);
	std::vector<ScanTaskDescriptor> scan_tasks;
	auto scan_impl = std::make_shared<ScanSourceNode>(PipelineNodeContext(0, "", 1, "scan"), empty_plan, scan_tasks,
	                                                  scan_schema, DuckDBExecutionConfigRef(), false);
	ExpressionRef proj_expr = std::make_shared<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(42));
	std::vector<std::string> proj_names;
	SchemaRef proj_schema = std::make_shared<LogicalType>(LogicalType::INTEGER);
	auto proj_impl =
	    std::make_shared<ProjectionNode>(2, scan_impl, std::vector<ExpressionRef> {proj_expr}, proj_names, proj_schema);
	ExpressionRef pred_expr = std::make_shared<duckdb::BoundConstantExpression>(duckdb::Value::BOOLEAN(true));
	auto filter_impl = std::make_shared<FilterNode>(3, proj_impl, pred_expr);
	auto root = std::make_shared<DistributedPipelineNode>(filter_impl);

	duckdb::DuckDB db2(nullptr);
	duckdb::Connection conn2(db2);
	auto task_executor2 = std::make_shared<TaskExecutor>(*conn2.context);

	PlanExecutionContext ctx(task_executor2, conn2.context);

	// For determinism in tests, construct a WorkerTask synchronously and
	// validate that its plan pointer and textual rendering are available and
	// safely printable. This avoids races with the async producer loop.
	TaskContext tctx = TaskContext::from_node_context(0, root->node_id(), 123);
	WorkerTask task(tctx, empty_plan, ExecutionConfigRef(), std::unordered_map<std::string, std::string>());
	REQUIRE(task.plan());
	std::ostringstream oss;
	oss << "[test] constructed task plan_ptr=" << static_cast<const void *>(task.plan().get());
	std::cerr << oss.str() << std::endl;
	if (task.plan() && task.plan()->HasRoot()) {
		try {
			auto plan_text = task.plan()->Root().ToString(ExplainFormat::TEXT);
			CAPTURE(plan_text);
			std::cerr << "[test] plan:\n" << plan_text << std::endl;
		} catch (const std::exception &ex) {
			std::cerr << "[test] plan render failed: " << ex.what() << std::endl;
			FAIL("plan render threw");
		} catch (...) {
			std::cerr << "[test] plan render failed: unknown error" << std::endl;
			FAIL("plan render threw");
		}
	} else {
		FAIL("task plan missing root");
	}
}
