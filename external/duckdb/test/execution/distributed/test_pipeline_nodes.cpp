// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/execution/distributed/pipeline_node/filter.hpp"
#include "duckdb/execution/distributed/pipeline_node/projection.hpp"
#include "duckdb/execution/distributed/pipeline_node/scan_source.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/execution/distributed/plan/physical_plan_helpers.hpp"

using namespace duckdb;
using namespace duckdb::distributed;

TEST_CASE("ProjectionNode: construction and display", "[distributed]") {
	// Create a dummy child scan source node with no scans
	std::vector<DuckPhysicalPlanRef> plans;
	SchemaRef schema = std::make_shared<LogicalType>(LogicalType::INTEGER);
	std::vector<ScanTaskDescriptor> scan_tasks;
	auto child = std::make_shared<ScanSourceNode>(PipelineNodeContext(0, "", 0, "scan"), DuckPhysicalPlanRef(),
	                                              scan_tasks, schema, DuckDBExecutionConfigRef(), false);

	// Build a simple projection expression: reference to column 0
	ExpressionRef expr = ExpressionRef(new BoundReferenceExpression(LogicalType::INTEGER, 0));
	std::vector<ExpressionRef> proj = {expr};

	std::vector<std::string> proj_names;
	auto node = ProjectionNode(1, child, proj, proj_names, schema);
	auto disp = node.multiline_display(false);
	REQUIRE(disp.size() >= 1);
	REQUIRE(disp[0].find("Project:") == 0);
	auto children = node.children();
	REQUIRE(children.size() == 1);
}

TEST_CASE("FilterNode: construction and display", "[distributed]") {
	std::vector<DuckPhysicalPlanRef> plans;
	SchemaRef schema = std::make_shared<LogicalType>(LogicalType::INTEGER);
	std::vector<ScanTaskDescriptor> scan_tasks;
	auto child = std::make_shared<ScanSourceNode>(PipelineNodeContext(0, "", 0, "scan"), DuckPhysicalPlanRef(),
	                                              scan_tasks, schema, DuckDBExecutionConfigRef(), false);

	ExpressionRef pred = ExpressionRef(new BoundConstantExpression(Value::INTEGER(1)));
	auto node = FilterNode(2, child, pred);
	auto disp = node.multiline_display(false);
	REQUIRE(disp.size() == 1);
	REQUIRE(disp[0].find("Filter:") == 0);
	auto children = node.children();
	REQUIRE(children.size() == 1);
}

TEST_CASE("ScanSourceNode: display", "[distributed]") {
	// Create a simple in-memory plan and attach to ScanSourceNode
	DuckPhysicalPlanRef p = duckdb::distributed::make_physical_plan_with_identity_projection({{1, 2, 3}});
	std::vector<DuckPhysicalPlanRef> plans = {p};
	SchemaRef schema = std::make_shared<LogicalType>(LogicalType::BIGINT);
	std::vector<ScanTaskDescriptor> scan_tasks = {ScanTaskDescriptor {}};
	auto node = ScanSourceNode(PipelineNodeContext(0, "", 3, "scan"), plans[0], scan_tasks, schema,
	                           DuckDBExecutionConfigRef(), false);
	auto disp = node.multiline_display(false);
	bool found = false;
	for (auto &s : disp) {
		if (s.find("Num Scan Tasks = 1") != std::string::npos) {
			found = true;
			break;
		}
	}
	REQUIRE(found);
}
