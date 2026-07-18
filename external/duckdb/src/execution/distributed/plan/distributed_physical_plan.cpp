// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/plan/distributed_physical_plan.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

// Include planner / relation headers to allow building plans from Relation objects
#include "duckdb/main/relation.hpp"
#include "duckdb/parser/statement/relation_statement.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"

namespace duckdb {
namespace distributed {

DuckDBResult<std::shared_ptr<DistributedPhysicalPlan>>
DistributedPhysicalPlan::from_logical_plan_builder(const std::shared_ptr<LogicalPlan> &builder_plan,
                                                   std::string query_id, DuckDBExecutionConfigRef config) {
	// In , this function builds a LogicalPlan from a builder, assigns a query_idx and retains the config.
	// In C++ we accept an already constructed std::shared_ptr<LogicalPlan> as the "builder_plan" argument which stands
	// in for a built logical plan. We generate a new query_idx using the shared global query idx counter.
	uint16_t idx = get_query_idx_counter().fetch_add(1);
	// For now, produce an empty duckdb physical plan directly (avoid helper lookup issues).
	Allocator &alloc = Allocator::DefaultAllocator();
	auto physical_plan = std::make_shared<duckdb::PhysicalPlan>(alloc);
	auto plan = std::make_shared<DistributedPhysicalPlan>(idx, std::move(query_id), physical_plan, std::move(config));
	return DuckDBResult<std::shared_ptr<DistributedPhysicalPlan>>::ok(std::move(plan));
}

DuckDBResult<std::shared_ptr<DistributedPhysicalPlan>>
DistributedPhysicalPlan::from_duckdb_relation(const shared_ptr<duckdb::Relation> &relation, std::string query_id,
                                              DuckDBExecutionConfigRef config) {
	try {
		if (!relation) {
			return DuckDBResult<std::shared_ptr<DistributedPhysicalPlan>>::err(
			    DuckDBError("from_duckdb_relation: provided relation is null"));
		}

		// Get the ClientContext from the relation's context wrapper
		auto client_context = relation->context->GetContext();

		// Physical plan will be created inside transaction
		duckdb::shared_ptr<duckdb::PhysicalPlan> physical_plan;

		// Use RunFunctionInTransaction to ensure proper transaction context
		client_context->RunFunctionInTransaction([&]() {
			// Create a RelationStatement from the relation
			auto relation_stmt = make_uniq<RelationStatement>(relation);

			// Get a Planner instance from the client context
			Planner planner(*client_context);
			planner.CreatePlan(std::move(relation_stmt));

			if (!planner.plan) {
				throw std::runtime_error("Planner failed to create logical plan");
			}

			// Optimize the logical plan
			Optimizer optimizer(*planner.binder, *client_context);
			auto optimized_plan = optimizer.Optimize(std::move(planner.plan));

			if (!optimized_plan) {
				throw std::runtime_error("Optimizer failed to optimize plan");
			}

			// Generate physical plan from optimized logical plan
			PhysicalPlanGenerator physical_planner(*client_context);
			// Use Plan() method which creates and returns a complete PhysicalPlan
			auto plan_uptr = physical_planner.Plan(std::move(optimized_plan));

			if (!plan_uptr) {
				throw std::runtime_error("PhysicalPlanGenerator.Plan returned null");
			}

			// Convert unique_ptr to duckdb::shared_ptr
			physical_plan = duckdb::shared_ptr<duckdb::PhysicalPlan>(plan_uptr.release());
		});

		// Create DistributedPhysicalPlan with the physical plan
		// Convert duckdb::shared_ptr to std::shared_ptr
		auto std_physical_plan =
		    std::shared_ptr<duckdb::PhysicalPlan>(physical_plan.get(), [physical_plan](duckdb::PhysicalPlan *) mutable {
			    // Custom deleter that keeps the duckdb::shared_ptr alive
			    physical_plan.reset();
		    });

		uint16_t idx = get_query_idx_counter().fetch_add(1);
		if (!config) {
			config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
		}
		auto plan =
		    std::make_shared<DistributedPhysicalPlan>(idx, std::move(query_id), std_physical_plan, std::move(config));

		return DuckDBResult<std::shared_ptr<DistributedPhysicalPlan>>::ok(std::move(plan));
	} catch (const std::exception &ex) {
		return DuckDBResult<std::shared_ptr<DistributedPhysicalPlan>>::err(DuckDBError(ex.what()));
	}
}

} // namespace distributed
} // namespace duckdb
