// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb/execution/distributed/pipeline_node/shuffles/repartition.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/execution/operator/exchange/repartition.hpp"
#include "duckdb/execution/distributed/pipeline_node/projection.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"

using namespace duckdb;
using namespace duckdb::distributed;

TEST_CASE("RepartitionNode: basic creation and properties (unittest)", "[distributed]") {
	SchemaRef schema = std::make_shared<LogicalType>(LogicalType::INTEGER);
	auto proj_impl = std::make_shared<ProjectionNode>(1, nullptr, std::vector<ExpressionRef> {},
	                                                  std::vector<std::string> {}, schema);
	auto child = std::make_shared<DistributedPipelineNode>(proj_impl);

	auto spec = ::duckdb::RepartitionSpec::create_random(4);

	PlanConfig cfg(0, "", nullptr);
	auto rnode = RepartitionNode::create(2, std::make_shared<PlanConfig>(cfg), spec, 4, nullptr, child);
	REQUIRE(rnode != nullptr);

	auto dist = rnode->into_node();
	REQUIRE(dist != nullptr);
	REQUIRE(dist->name() == "Repartition");
	REQUIRE(dist->arc_children().size() == 1);
}
