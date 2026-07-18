// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file test_planrunner_local.cpp
 * @brief Validates that PlanRunner can be instantiated without Ray
 *        (pure C++, zero Ray dependency). This test surfaces any
 *        template-level coupling between PlanRunner and Ray-specific types.
 */

#include "catch.hpp"
#include "test_common.hpp"

#include "duckdb.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/execution/distributed/plan/distributed_physical_plan.hpp"

using namespace duckdb;
using namespace duckdb::distributed;
using namespace duckdb::distributed::testing;

TEST_CASE("PlanRunner instantiation", "[distributed][plan][local]") {
	// 1. Create mock workers and manager (pure C++, no Ray)
	auto workers = setup_workers({{make_worker_id("local-w1"), 4}});
	auto worker_mgr = std::make_shared<MockWorkerManager>(std::move(workers));

	// 2. Create DuckDB database + ClientContext (needed for TaskExecutor)
	DuckDB db;
	Connection con(db);

	// 3. Instantiate PlanRunner — this is the key decoupling test.
	//    If this compiles, PlanRunner doesn't depend on Ray-specific types.
	auto runner = std::make_shared<PlanRunner>(worker_mgr, con.context);

	REQUIRE(runner != nullptr);
}
