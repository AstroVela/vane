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
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/execution/distributed/plan/distributed_physical_plan.hpp"
#include "test_helpers.hpp"

#include <atomic>

using namespace duckdb;
using namespace duckdb::distributed;
using namespace duckdb::distributed::testing;

class FailingCopyWorkerManager final : public WorkerManager {
public:
	FailingCopyWorkerManager(int quiescence_failure_mode, FileSystem &fs, string operation_output_root)
	    : quiescence_failure_mode(quiescence_failure_mode), fs(fs),
	      operation_output_root(std::move(operation_output_root)) {
	}

	DuckDBResult<std::vector<WorkerSnapshot>> worker_snapshots() const override {
		std::vector<WorkerSnapshot> snapshots;
		snapshots.emplace_back(make_worker_id("failing-copy-w1"), 1, 0);
		return DuckDBResult<std::vector<WorkerSnapshot>>::ok(std::move(snapshots));
	}

	DuckDBResult<void> try_autoscale(const std::vector<TaskResourceRequest> &) override {
		return DuckDBResult<void>::ok();
	}

	DuckDBResult<void> shutdown() override {
		return DuckDBResult<void>::ok();
	}

	DuckDBResult<void> submit_fte_task_events(std::vector<WorkerTask>) override {
		return DuckDBResult<void>::ok();
	}

	DuckDBResult<void> task_input_stream_exhausted_for_query(const string &,
	                                                         const std::unordered_set<SourceNodeId> &) override {
		return DuckDBResult<void>::ok();
	}

	DuckDBResult<void> materialization_barrier_completed(const string &, NodeID) override {
		return DuckDBResult<void>::ok();
	}

	DuckDBResult<std::vector<MaterializedOutput>> wait_fte_query(const string &, double) override {
		return DuckDBResult<std::vector<MaterializedOutput>>::err(
		    DuckDBError::external_error("planned worker execution failure"));
	}

	DuckDBResult<void> quiesce_fte_query(const string &) override {
		quiesce_calls++;
		output_existed_during_quiescence = fs.DirectoryExists(operation_output_root);
		if (quiescence_failure_mode == 2) {
			throw std::runtime_error("planned worker quiescence exception");
		}
		if (quiescence_failure_mode == 1) {
			return DuckDBResult<void>::err(DuckDBError::external_error("planned worker quiescence failure"));
		}
		return DuckDBResult<void>::ok();
	}

	std::atomic<idx_t> quiesce_calls {0};
	std::atomic<bool> output_existed_during_quiescence {false};
	const int quiescence_failure_mode;

private:
	FileSystem &fs;
	const string operation_output_root;
};

TEST_CASE("PlanRunner instantiation", "[distributed][plan][local]") {
	// 1. Create mock workers and manager (pure C++, no Ray)
	auto workers = setup_workers({{make_worker_id("local-w1"), 4}});
	auto worker_mgr = std::make_shared<MockWorkerManager>(std::move(workers));

	// 2. Create DuckDB database + ClientContext (needed for plan control)
	DuckDB db;
	Connection con(db);

	// 3. Instantiate PlanRunner — this is the key decoupling test.
	//    If this compiles, PlanRunner doesn't depend on Ray-specific types.
	auto runner = std::make_shared<PlanRunner>(worker_mgr, con.context);

	REQUIRE(runner != nullptr);
}

TEST_CASE("PlanRunner quiesces failed COPY workers before deleting output", "[distributed][plan][copy]") {
	const int quiescence_failure_mode = GENERATE(0, 1, 2);
	DuckDB db(nullptr);
	Connection con(db);
	auto &fs = FileSystem::GetFileSystem(*con.context);
	const auto suffix = quiescence_failure_mode == 0   ? "planrunner_copy_quiescence_ok.parquet"
	                    : quiescence_failure_mode == 1 ? "planrunner_copy_quiescence_error.parquet"
	                                                   : "planrunner_copy_quiescence_exception.parquet";
	const auto output_path = TestCreatePath(suffix);
	const auto commit_root = output_path + ".duckdb_commit";
	const auto staging_root = output_path + ".duckdb_staging";
	const auto operation_output_root = DistributedCopyLocalStagingEnabled() ? staging_root : commit_root;
	RemoveDistributedCopyDirectoryTree(fs, commit_root);
	RemoveDistributedCopyDirectoryTree(fs, staging_root);

	auto sql = "COPY (SELECT 42 AS value) TO '" + StringUtil::Replace(output_path, "'", "''") +
	           "' (FORMAT PARQUET, RETURN_STATS true, USE_TMP_FILE false)";
	auto logical_plan = con.ExtractPlan(sql);
	REQUIRE(logical_plan != nullptr);
	PhysicalPlanGenerator generator(*con.context);
	auto generated_plan = generator.Plan(std::move(logical_plan));
	REQUIRE(generated_plan != nullptr);
	auto physical_plan = DuckPhysicalPlanRef(generated_plan.release());

	auto worker_manager =
	    std::make_shared<FailingCopyWorkerManager>(quiescence_failure_mode, fs, operation_output_root);
	auto runner = std::make_shared<PlanRunner>(worker_manager, con.context);
	auto execution_config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
	auto distributed_plan = std::make_shared<DistributedPhysicalPlan>(23, "planrunner-copy-worker-failure",
	                                                                  physical_plan, std::move(execution_config));

	auto result = runner->run_plan(std::move(distributed_plan));

	REQUIRE(result.is_err());
	REQUIRE(StringUtil::Contains(result.error().what(), "planned worker execution failure"));
	REQUIRE(worker_manager->quiesce_calls == 1);
	REQUIRE(worker_manager->output_existed_during_quiescence);
	if (quiescence_failure_mode == 0) {
		REQUIRE_FALSE(fs.DirectoryExists(operation_output_root));
		return;
	}

	REQUIRE(StringUtil::Contains(result.error().what(), quiescence_failure_mode == 1
	                                                        ? "planned worker quiescence failure"
	                                                        : "planned worker quiescence exception"));
	REQUIRE(StringUtil::Contains(result.error().what(), "output is retained"));
	REQUIRE(fs.DirectoryExists(operation_output_root));
	RemoveDistributedCopyDirectoryTree(fs, commit_root);
	RemoveDistributedCopyDirectoryTree(fs, staging_root);
}
