// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/execution/operator/exchange/physical_repartition.hpp"
#include "duckdb/execution/operator/exchange/repartition.hpp"

using namespace duckdb;

TEST_CASE("PhysicalRepartition: basic construction", "[execution]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);
	LogicalType int_type = LogicalType::INTEGER;
	vector<LogicalType> types = {int_type};
	idx_t estimated_cardinality = 0;

	auto spec = duckdb::RepartitionSpec::create_random(4);
	auto &base = plan.Make<PhysicalRepartition>(types, spec, estimated_cardinality);
	auto *rep_ptr = dynamic_cast<PhysicalRepartition *>(&base);
	REQUIRE(rep_ptr != nullptr);

	REQUIRE(rep_ptr->repartition_spec != nullptr);
	REQUIRE(rep_ptr->ParamsToString().contains("__repartition_spec__"));
}
