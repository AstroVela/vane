// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/operator/exchange/physical_local_exchange.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_local_exchange.hpp"

namespace duckdb {

PhysicalOperator &PhysicalPlanGenerator::CreatePlan(LogicalLocalExchange &op) {
	D_ASSERT(op.children.size() == 1);
	auto &child = CreatePlan(*op.children[0]);
	auto spec = op.repartition_spec;
	if (!spec) {
		spec = RepartitionSpec::create_random(0);
	}
	auto &exchange = Make<PhysicalLocalExchange>(op.types, spec, op.estimated_cardinality);
	exchange.children.push_back(child);
	return exchange;
}

} // namespace duckdb
