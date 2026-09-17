// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/execution/distributed/pipeline_node/filter.hpp"
#include "duckdb/execution/distributed/pipeline_node/limit.hpp"
#include "duckdb/execution/distributed/pipeline_node/projection.hpp"
#include "duckdb/execution/distributed/pipeline_node/scan_source.hpp"
#include "duckdb/execution/distributed/pipeline_node/sort.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/distributed/pipeline_node/table_inout.hpp"
#include "duckdb/execution/distributed/pipeline_node/vllm.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/execution/distributed/plan/physical_plan_helpers.hpp"

using namespace duckdb;
using namespace duckdb::distributed;

TEST_CASE("PipelineSchema preserves explicit top-level columns", "[distributed]") {
	child_list_t<LogicalType> fields;
	fields.emplace_back("value", LogicalType::INTEGER);
	fields.emplace_back("label", LogicalType::VARCHAR);
	vector<LogicalType> types = {LogicalType::STRUCT(std::move(fields))};
	vector<std::string> names = {"feature"};

	auto schema = MakeSchemaRef(types, names);
	REQUIRE(GetSchemaTypes(schema) == types);
	REQUIRE(GetSchemaNames(schema) == names);

	vector<LogicalType> two_types = {LogicalType::INTEGER, LogicalType::VARCHAR};
	REQUIRE_THROWS_AS(MakeSchemaRef(two_types, names), std::invalid_argument);
}

TEST_CASE("Schema-changing nodes preserve every output type", "[distributed]") {
	vector<LogicalType> output_types = {LogicalType::INTEGER, LogicalType::VARCHAR};

	SECTION("table in-out") {
		TableInOutNode node(1, nullptr, TableFunction {}, nullptr, vector<ColumnIndex> {}, vector<column_t> {},
		                    optional_idx(), output_types, 0);
		REQUIRE(GetSchemaTypes(node.config().schema()) == output_types);
	}

	SECTION("vLLM projection") {
		VLLMProjectNode node(1, nullptr, nullptr, "", Value(), "output", output_types);
		REQUIRE(GetSchemaTypes(node.config().schema()) == output_types);
	}
}

TEST_CASE("ProjectionNode: construction and display", "[distributed]") {
	// Create a dummy child scan source node with no scans
	std::vector<DuckPhysicalPlanRef> plans;
	SchemaRef schema = MakeSchemaRef(std::vector<LogicalType> {LogicalType::INTEGER});
	std::vector<ScanSplit> scan_splits;
	auto child = std::make_shared<ScanSourceNode>(PipelineNodeContext(0, "", 0, "scan"), DuckPhysicalPlanRef(),
	                                              scan_splits, schema, DuckDBExecutionConfigRef(), false);

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
	SchemaRef schema = MakeSchemaRef(std::vector<LogicalType> {LogicalType::INTEGER});
	std::vector<ScanSplit> scan_splits;
	auto child = std::make_shared<ScanSourceNode>(PipelineNodeContext(0, "", 0, "scan"), DuckPhysicalPlanRef(),
	                                              scan_splits, schema, DuckDBExecutionConfigRef(), false);

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
	SchemaRef schema = MakeSchemaRef(std::vector<LogicalType> {LogicalType::BIGINT});
	std::vector<ScanSplit> scan_splits = {ScanSplit::EmptyFile()};
	auto node = ScanSourceNode(PipelineNodeContext(0, "", 3, "scan"), plans[0], scan_splits, schema,
	                           DuckDBExecutionConfigRef(), false);
	auto disp = node.multiline_display(false);
	bool found = false;
	for (auto &s : disp) {
		if (s.find("Num Scan Splits = 1") != std::string::npos) {
			found = true;
			break;
		}
	}
	REQUIRE(found);
}

TEST_CASE("Global limits distinguish clustering partitions from complete task output", "[distributed][limit]") {
	auto schema = MakeSchemaRef(vector<LogicalType> {LogicalType::BIGINT});
	std::vector<ScanSplit> splits = {ScanSplit::EmptyFile(), ScanSplit::EmptyFile()};
	auto scan = std::make_shared<ScanSourceNode>(PipelineNodeContext(0, "limit-contract", 1, "scan"), nullptr, splits,
	                                             schema, nullptr, false);
	// A unary node can advertise one clustering partition while its child
	// still emits multiple independently scheduled tasks.
	auto input = std::make_shared<ProjectionNode>(2, scan, std::vector<ExpressionRef> {}, std::vector<string> {},
	                                              schema, ClusteringSpec::unknown_with_num_partitions(1));
	REQUIRE(input->config().clustering_spec()->num_partitions() == 1);
	REQUIRE_FALSE(input->has_single_task_output());

	std::vector<PipelineNodeRef> globals;
	globals.push_back(std::make_shared<LimitNode>(3, input, BoundLimitNode::ConstantValue(5), BoundLimitNode()));
	globals.push_back(
	    std::make_shared<StreamingLimitNode>(4, input, BoundLimitNode::ConstantValue(5), BoundLimitNode(), false));
	globals.push_back(
	    std::make_shared<LimitPercentNode>(5, input, BoundLimitNode::ConstantPercentage(50), BoundLimitNode()));
	globals.push_back(std::make_shared<TopNNode>(6, input, vector<BoundOrderByNode> {}, 5, 0));
	for (auto &node : globals) {
		INFO(node->name());
		REQUIRE(node->is_materialization_barrier());
		REQUIRE(node->materialized_input_node_ids() == std::vector<NodeID> {input->node_id()});
		REQUIRE(node->has_single_task_output());
		REQUIRE(node->config().clustering_spec()->num_partitions() == 1);
		REQUIRE(GetSchemaTypes(node->config().schema()) == GetSchemaTypes(schema));

		auto filtered = std::make_shared<FilterNode>(7, node, nullptr);
		auto projected = std::make_shared<ProjectionNode>(8, filtered, std::vector<ExpressionRef> {},
		                                                  std::vector<string> {}, schema);
		auto wrapped = std::make_shared<DistributedPipelineNode>(projected);
		REQUIRE(wrapped->has_single_task_output());
		LimitNode preview(9, wrapped, BoundLimitNode::ConstantValue(10000), BoundLimitNode());
		REQUIRE_FALSE(preview.is_materialization_barrier());
		REQUIRE(preview.materialized_input_node_ids().empty());
	}
}

TEST_CASE("Scan task cardinality permits only complete inputs to bypass limit gather", "[distributed][limit]") {
	auto schema = MakeSchemaRef(vector<LogicalType> {LogicalType::BIGINT});
	for (size_t split_count : {size_t(0), size_t(1), size_t(4)}) {
		INFO(split_count);
		std::vector<ScanSplit> splits(split_count, ScanSplit::EmptyFile());
		auto scan = std::make_shared<ScanSourceNode>(PipelineNodeContext(0, "limit-scans", 1, "scan"), nullptr, splits,
		                                             schema, nullptr, false);
		LimitNode limit(2, scan, BoundLimitNode::ConstantValue(5), BoundLimitNode());
		TopNNode topn(3, scan, vector<BoundOrderByNode> {}, 5, 0);
		REQUIRE(scan->has_single_task_output() == (split_count <= 1));
		REQUIRE(limit.is_materialization_barrier() == (split_count > 1));
		REQUIRE(topn.is_materialization_barrier() == (split_count > 1));
		REQUIRE(limit.config().clustering_spec()->num_partitions() == 1);
		REQUIRE(topn.config().clustering_spec()->num_partitions() == 1);
	}
}
