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
#include <cstdlib>
#include <thread>

using namespace duckdb;
using namespace duckdb::distributed;
using namespace duckdb::distributed::testing;

class ScopedCopyStagingMode final {
public:
	ScopedCopyStagingMode() {
		const auto *existing = std::getenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING");
		if (existing) {
			had_value = true;
			old_value = existing;
		}
#if defined(_WIN32)
		_putenv_s("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", "0");
#else
		setenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", "0", 1);
#endif
	}

	~ScopedCopyStagingMode() {
#if defined(_WIN32)
		_putenv_s("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", had_value ? old_value.c_str() : "");
#else
		if (had_value) {
			setenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING", old_value.c_str(), 1);
		} else {
			unsetenv("VANE_DISTRIBUTED_COPY_LOCAL_STAGING");
		}
#endif
	}

private:
	string old_value;
	bool had_value = false;
};

class FailingCopyWorkerManager final : public WorkerManager {
public:
	FailingCopyWorkerManager(int abort_failure_mode, FileSystem &fs, string output_base_path, string commit_root)
	    : abort_failure_mode(abort_failure_mode), fs(fs), output_base_path(std::move(output_base_path)),
	      commit_root(std::move(commit_root)) {
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
		operation_output_root = ResolveOperationOutputRoot();
		fs.CreateDirectoriesRecursive(operation_output_root);
		auto marker_path = fs.JoinPath(operation_output_root, "early-writer.marker");
		auto write_res = WriteDistributedCopyTextFileAtomically(fs, marker_path, "early");
		if (write_res.is_err()) {
			return DuckDBResult<std::vector<MaterializedOutput>>::err(write_res.error());
		}
		return DuckDBResult<std::vector<MaterializedOutput>>::err(
		    DuckDBError::external_error("planned worker execution failure"));
	}

	DuckDBResult<void> abort_and_quiesce_query(const string &) override {
		abort_calls++;
		output_existed_during_abort = fs.DirectoryExists(operation_output_root);
		std::thread late_writer([&]() {
			try {
				fs.CreateDirectoriesRecursive(operation_output_root);
				auto marker_path = fs.JoinPath(operation_output_root, "late-writer.marker");
				auto write_res = WriteDistributedCopyTextFileAtomically(fs, marker_path, "late");
				late_writer_completed = write_res.is_ok();
			} catch (...) {
				late_writer_completed = false;
			}
		});
		late_writer.join();
		if (abort_failure_mode == 2) {
			throw std::runtime_error("planned worker abort exception");
		}
		if (abort_failure_mode == 1) {
			return DuckDBResult<void>::err(DuckDBError::external_error("planned worker abort failure"));
		}
		return DuckDBResult<void>::ok();
	}

	std::atomic<idx_t> abort_calls {0};
	std::atomic<bool> output_existed_during_abort {false};
	std::atomic<bool> late_writer_completed {false};
	const int abort_failure_mode;
	string operation_output_root;

private:
	string ResolveOperationOutputRoot() {
		std::vector<string> run_ids;
		fs.ListFiles(commit_root, [&](const string &path, bool is_dir) {
			if (is_dir) {
				run_ids.push_back(StringUtil::GetFileName(path));
			}
		});
		if (run_ids.size() != 1) {
			throw std::runtime_error("expected one distributed COPY lifecycle run");
		}
		auto commit_paths = BuildDistributedCopyFinalizeCommitPaths(fs, output_base_path, run_ids.front());
		auto lifecycle = ReadDistributedCopyDirectWriteLifecycle(fs, commit_paths, output_base_path, run_ids.front());
		if (lifecycle.is_err()) {
			throw std::runtime_error(lifecycle.error().what());
		}
		const auto &worker_base_path = lifecycle.value().worker_base_path;
		return BuildCopyDirectWriteRunDirectory(worker_base_path, run_ids.front(), fs.PathSeparator(worker_base_path));
	}

	FileSystem &fs;
	const string output_base_path;
	const string commit_root;
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

TEST_CASE("PlanRunner aborts failed COPY workers before deleting output", "[distributed][plan][copy]") {
	const int abort_failure_mode = GENERATE(0, 1, 2);
	ScopedCopyStagingMode direct_write_mode;
	DuckDB db(nullptr);
	Connection con(db);
	auto &fs = FileSystem::GetFileSystem(*con.context);
	const auto suffix = abort_failure_mode == 0   ? "planrunner_copy_abort_ok.parquet"
	                    : abort_failure_mode == 1 ? "planrunner_copy_abort_error.parquet"
	                                              : "planrunner_copy_abort_exception.parquet";
	const auto output_path = TestCreatePath(suffix);
	const auto commit_root = output_path + ".duckdb_commit";
	const auto staging_root = output_path + ".duckdb_staging";
	RemoveDistributedCopyDirectoryTree(fs, output_path);
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

	auto worker_manager = std::make_shared<FailingCopyWorkerManager>(abort_failure_mode, fs, output_path, commit_root);
	auto runner = std::make_shared<PlanRunner>(worker_manager, con.context);
	auto execution_config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
	auto distributed_plan = std::make_shared<DistributedPhysicalPlan>(23, "planrunner-copy-worker-failure",
	                                                                  physical_plan, std::move(execution_config));

	auto result = runner->run_plan(std::move(distributed_plan));

	REQUIRE(result.is_err());
	REQUIRE(StringUtil::Contains(result.error().what(), "planned worker execution failure"));
	REQUIRE(worker_manager->abort_calls == 1);
	REQUIRE(worker_manager->output_existed_during_abort);
	REQUIRE(worker_manager->late_writer_completed);
	if (abort_failure_mode == 0) {
		REQUIRE_FALSE(fs.DirectoryExists(worker_manager->operation_output_root));
		RemoveDistributedCopyDirectoryTree(fs, output_path);
		return;
	}

	REQUIRE(StringUtil::Contains(result.error().what(), abort_failure_mode == 1 ? "planned worker abort failure"
	                                                                            : "planned worker abort exception"));
	REQUIRE(StringUtil::Contains(result.error().what(), "output is retained"));
	REQUIRE(fs.DirectoryExists(worker_manager->operation_output_root));
	RemoveDistributedCopyDirectoryTree(fs, output_path);
	RemoveDistributedCopyDirectoryTree(fs, commit_root);
	RemoveDistributedCopyDirectoryTree(fs, staging_root);
}
