// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/operator/exchange/physical_repartition.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_repartition.hpp"

namespace duckdb {

static std::shared_ptr<RepartitionSpec> BuildPhysicalRepartitionSpec(LogicalRepartition &op) {
	auto spec = op.repartition_spec;
	if (!spec || spec->type() != RepartitionSpec::Type::Hash) {
		return spec;
	}

	auto *hash_spec = dynamic_cast<HashRepartitionSpec *>(spec.get());
	if (!hash_spec) {
		throw InternalException("Expected HashRepartitionSpec for LOGICAL_REPARTITION");
	}

	vector<ExprRef> by;
	by.reserve(op.expressions.size());
	for (auto &expr : op.expressions) {
		if (expr) {
			auto copy = expr->Copy();
			by.emplace_back(copy.release());
		}
	}
	return RepartitionSpec::create_hash(hash_spec->config()->num_partitions, std::move(by));
}

PhysicalOperator &PhysicalPlanGenerator::CreatePlan(LogicalRepartition &op) {
	D_ASSERT(op.children.size() == 1);
	auto &child = CreatePlan(*op.children[0]);
	// Create a PhysicalRepartition marker. translate.cpp converts this to RepartitionNode.
	auto &repartition = Make<PhysicalRepartition>(op.types, BuildPhysicalRepartitionSpec(op), op.estimated_cardinality);
	repartition.children.push_back(child);
	return repartition;
}

} // namespace duckdb
