// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/execution/operator/scan/physical_dummy_scan.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/distributed/pipeline_node/translator.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"

#include <future>

using namespace duckdb;
using namespace duckdb::distributed;

TEST_CASE("PhysicalPlanToPipelineNodeTranslator: simple plan translation via visitor", "[distributed]") {
	Allocator allocator;
	LogicalType int_type = LogicalType::INTEGER;
	vector<LogicalType> types = {int_type};
	idx_t estimated_cardinality = 0;

	// Build: DummyScan -> Filter -> Projection
	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);

	auto &scan = plan_ptr->Make<PhysicalDummyScan>(types, 1);

	vector<unique_ptr<Expression>> filter_select_list;
	filter_select_list.push_back(duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(1)));
	auto &filter = plan_ptr->Make<PhysicalFilter>(types, std::move(filter_select_list), estimated_cardinality);
	filter.children.emplace_back(scan);

	vector<unique_ptr<Expression>> select_list;
	select_list.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(int_type, 0));
	auto &proj = plan_ptr->Make<PhysicalProjection>(types, std::move(select_list), estimated_cardinality);
	proj.children.emplace_back(filter);
	plan_ptr->SetRoot(proj);

	PlanConfig cfg(0, "", nullptr);
	auto fut = std::async(std::launch::async, [&]() { return physical_plan_to_pipeline_node(cfg, plan_ptr); });
	auto status = fut.wait_for(std::chrono::seconds(2));
	REQUIRE(status == std::future_status::ready);
	auto res = fut.get();
	REQUIRE(res.is_ok());
	auto root = res.value();
	REQUIRE(root != nullptr);

	// The shape should be: root (projection) -> 1 child (filter) -> 1 child (scan)
	auto children = root->arc_children();
	REQUIRE(children.size() == 1);
	REQUIRE(root->name() == "Projection");
	auto child = children[0];
	REQUIRE(child->name() == "Filter");
	REQUIRE(child->arc_children().size() == 1);
	auto leaf = child->arc_children()[0];
	REQUIRE(leaf->arc_children().empty());
	CAPTURE(leaf->name());
	REQUIRE(!leaf->name().empty());
}
