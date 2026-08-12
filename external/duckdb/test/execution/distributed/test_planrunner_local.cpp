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
#include "duckdb/execution/operator/persistent/physical_copy_to_file.hpp"
#include "duckdb/execution/operator/scan/physical_dummy_scan.hpp"
#include "duckdb/execution/distributed/copy_finalize.hpp"
#include "duckdb/execution/distributed/extension_write_task_provider.hpp"
#include "duckdb/execution/distributed/pipeline_node/copy_finish.hpp"
#include "duckdb/execution/distributed/pipeline_node/translator.hpp"
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

namespace {

class ReplayTestExtensionWriteOperator final : public PhysicalOperator, public ExtensionWriteTaskProvider {
public:
	ReplayTestExtensionWriteOperator(PhysicalPlan &physical_plan, vector<LogicalType> types,
	                                 PhysicalOperatorType operator_type = PhysicalOperatorType::EXTENSION)
	    : PhysicalOperator(physical_plan, operator_type, std::move(types), 0) {
		plan.extension_name = "replay_test_extension";
		plan.operator_name = "write";
	}

	optional_ptr<ExtensionWriteTaskProvider> GetExtensionWriteTaskProvider() override {
		return this;
	}

	const DistributedExtensionWritePlan &WritePlan() const override {
		return plan;
	}

	void ValidateDistributedWrite(ClientContext &, const DistributedWriteOperationContext &operation) const override {
		validation_calls++;
		validation_operation_id = operation.operation_id;
		if (fail_validation) {
			throw InvalidInputException("planned extension write validation failure");
		}
	}

	idx_t FinalizeDistributedWrite(ClientContext &, const DistributedWriteOperationContext &operation,
	                               const vector<DistributedWriteTaskResult> &results) const override {
		finalize_calls++;
		finalize_operation_id = operation.operation_id;
		idx_t rows = 0;
		for (const auto &result : results) {
			rows += result.RowCount();
		}
		return rows;
	}

	void AbortDistributedWrite(ClientContext &, const DistributedWriteOperationContext &operation,
	                           const vector<DistributedWriteTaskResult> &) const override {
		abort_calls++;
		abort_operation_id = operation.operation_id;
	}

	mutable idx_t validation_calls = 0;
	mutable idx_t finalize_calls = 0;
	mutable idx_t abort_calls = 0;
	mutable string validation_operation_id;
	mutable string finalize_operation_id;
	mutable string abort_operation_id;
	bool fail_validation = false;

private:
	DistributedExtensionWritePlan plan;
};

string PlanRunnerSQLStringLiteral(const string &value) {
	return "'" + StringUtil::Replace(value, "'", "''") + "'";
}

void RegisterReplayTestExtension(DatabaseInstance &db) {
	auto &manager = DistributedExtensionManager::Get(db);
	DistributedExtensionManifest manifest;
	manifest.extension_name = "replay_test_extension";
	manifest.capabilities.push_back({DistributedExtensionCapabilityKind::WRITE_OPERATOR, "write", 1});
	DistributedWriteOperatorExtension write_operator;
	write_operator.name = "write";
	write_operator.protocol_version = 1;
	write_operator.mode = DistributedWriteMode::FILE_ARTIFACT;
	write_operator.fragment_codec = {DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC,
	                                 DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC_VERSION};
	manager.RegisterExtension(manifest,
	                          {make_shared_ptr<const DistributedWriteOperatorExtension>(std::move(write_operator))});
}

void WritePlanRunnerTestFile(FileSystem &fs, const string &path, const string &contents) {
	auto parent = StringUtil::GetFilePath(path);
	if (!parent.empty()) {
		fs.CreateDirectoriesRecursive(parent);
	}
	auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE_NEW);
	if (!contents.empty()) {
		auto data = const_cast<char *>(contents.data());
		auto written = handle->Write(data, contents.size());
		REQUIRE(written == NumericCast<int64_t>(contents.size()));
	}
	handle->Close();
}

} // namespace

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

TEST_CASE("PlanRunner rejects an unregistered extension write capability",
          "[distributed][plan][copy][extension-write]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto physical_plan = std::make_shared<PhysicalPlan>(Allocator::DefaultAllocator());
	vector<LogicalType> child_types {LogicalType::BIGINT};
	auto &child = physical_plan->Make<PhysicalDummyScan>(std::move(child_types), 1);
	vector<LogicalType> extension_types {LogicalType::BIGINT};
	auto &extension_operator = physical_plan->Make<ReplayTestExtensionWriteOperator>(std::move(extension_types));
	auto &extension = extension_operator.Cast<ReplayTestExtensionWriteOperator>();
	extension_operator.children.push_back(child);
	physical_plan->SetRoot(extension_operator);

	auto workers = setup_workers({{make_worker_id("unregistered-extension-w1"), 1}});
	auto worker_manager = std::make_shared<MockWorkerManager>(std::move(workers));
	auto runner = std::make_shared<PlanRunner>(worker_manager, con.context);
	auto execution_config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
	auto distributed_plan = std::make_shared<DistributedPhysicalPlan>(15, "planrunner-unregistered-extension",
	                                                                  physical_plan, std::move(execution_config));

	auto result = runner->run_plan(std::move(distributed_plan));
	REQUIRE(result.is_err());
	REQUIRE(StringUtil::Contains(result.error().what(), "protocol validation failed"));
	REQUIRE(StringUtil::Contains(result.error().what(), "replay_test_extension"));
	REQUIRE(extension.validation_calls == 0);
	REQUIRE(extension.finalize_calls == 0);
}

TEST_CASE("PlanRunner rejects extension writes inside an explicit DuckDB transaction",
          "[distributed][plan][copy][extension-write]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterReplayTestExtension(*db.instance);
	auto output_path = TestCreatePath("planrunner_extension_explicit_transaction");
	auto logical_plan = con.ExtractPlan("COPY (SELECT 42 AS value) TO " + PlanRunnerSQLStringLiteral(output_path) +
	                                    " (FORMAT PARQUET, RETURN_STATS true, USE_TMP_FILE false)");
	REQUIRE(logical_plan != nullptr);

	PhysicalPlanGenerator generator(*con.context);
	auto generated_plan = generator.Plan(std::move(logical_plan));
	REQUIRE(generated_plan != nullptr);
	auto &copy_root = generated_plan->Root();

	vector<LogicalType> extension_types {LogicalType::BIGINT};
	auto &extension_operator = generated_plan->Make<ReplayTestExtensionWriteOperator>(std::move(extension_types));
	auto &extension = extension_operator.Cast<ReplayTestExtensionWriteOperator>();
	extension_operator.children.push_back(copy_root);
	generated_plan->SetRoot(extension_operator);
	auto physical_plan = DuckPhysicalPlanRef(generated_plan.release());

	auto workers = setup_workers({{make_worker_id("transaction-w1"), 1}});
	auto worker_manager = std::make_shared<MockWorkerManager>(std::move(workers));
	auto runner = std::make_shared<PlanRunner>(worker_manager, con.context);
	auto execution_config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
	auto distributed_plan = std::make_shared<DistributedPhysicalPlan>(16, "planrunner-extension-explicit-transaction",
	                                                                  physical_plan, std::move(execution_config));

	con.BeginTransaction();
	auto result = runner->run_plan(std::move(distributed_plan));
	con.Rollback();

	REQUIRE(result.is_err());
	REQUIRE(StringUtil::Contains(result.error().what(), "auto-commit mode"));
	REQUIRE(extension.validation_calls == 0);
	REQUIRE(extension.finalize_calls == 0);
	REQUIRE_FALSE(FileSystem::GetFileSystem(*con.context).DirectoryExists(output_path));
}

TEST_CASE("PlanRunner aborts an extension write when coordinator validation fails",
          "[distributed][plan][copy][extension-write]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterReplayTestExtension(*db.instance);
	auto output_path = TestCreatePath("planrunner_extension_validation_failure");
	auto logical_plan = con.ExtractPlan("COPY (SELECT 42 AS value) TO " + PlanRunnerSQLStringLiteral(output_path) +
	                                    " (FORMAT PARQUET, RETURN_STATS true, USE_TMP_FILE false)");
	REQUIRE(logical_plan != nullptr);

	PhysicalPlanGenerator generator(*con.context);
	auto generated_plan = generator.Plan(std::move(logical_plan));
	REQUIRE(generated_plan != nullptr);
	auto &copy_root = generated_plan->Root();

	vector<LogicalType> extension_types {LogicalType::BIGINT};
	auto &extension_operator = generated_plan->Make<ReplayTestExtensionWriteOperator>(std::move(extension_types));
	auto &extension = extension_operator.Cast<ReplayTestExtensionWriteOperator>();
	extension.fail_validation = true;
	extension_operator.children.push_back(copy_root);
	generated_plan->SetRoot(extension_operator);
	auto physical_plan = DuckPhysicalPlanRef(generated_plan.release());

	auto workers = setup_workers({{make_worker_id("validation-w1"), 1}});
	auto worker_manager = std::make_shared<MockWorkerManager>(std::move(workers));
	auto runner = std::make_shared<PlanRunner>(worker_manager, con.context);
	auto execution_config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
	auto distributed_plan = std::make_shared<DistributedPhysicalPlan>(19, "planrunner-extension-validation-failure",
	                                                                  physical_plan, std::move(execution_config));

	auto missing_transaction = runner->run_plan(distributed_plan);
	REQUIRE(missing_transaction.is_err());
	REQUIRE(StringUtil::Contains(missing_transaction.error().what(), "active Vane-owned auto-commit transaction"));
	REQUIRE(extension.validation_calls == 0);
	REQUIRE(extension.finalize_calls == 0);
	REQUIRE(extension.abort_calls == 0);

	DuckDBResult<PlanRunner::PlanResult> result;
	con.context->RunFunctionInTransaction([&]() { result = runner->run_plan(std::move(distributed_plan)); });

	REQUIRE(result.is_err());
	REQUIRE(StringUtil::Contains(result.error().what(), "planned extension write validation failure"));
	REQUIRE(extension.validation_calls == 1);
	REQUIRE(extension.finalize_calls == 0);
	REQUIRE(extension.abort_calls == 1);
	REQUIRE(extension.validation_operation_id == "planrunner-extension-validation-failure");
	REQUIRE(extension.abort_operation_id == extension.validation_operation_id);
	REQUIRE_FALSE(FileSystem::GetFileSystem(*con.context).DirectoryExists(output_path));
}

TEST_CASE("PlanRunner requires an EXTENSION root for extension write providers",
          "[distributed][plan][copy][extension-write]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterReplayTestExtension(*db.instance);
	auto output_path = TestCreatePath("planrunner_extension_wrong_root_type");
	auto logical_plan = con.ExtractPlan("COPY (SELECT 42 AS value) TO " + PlanRunnerSQLStringLiteral(output_path) +
	                                    " (FORMAT PARQUET, RETURN_STATS true, USE_TMP_FILE false)");
	REQUIRE(logical_plan != nullptr);

	PhysicalPlanGenerator generator(*con.context);
	auto generated_plan = generator.Plan(std::move(logical_plan));
	REQUIRE(generated_plan != nullptr);
	auto &copy_root = generated_plan->Root();

	vector<LogicalType> extension_types {LogicalType::BIGINT};
	auto &invalid_root = generated_plan->Make<ReplayTestExtensionWriteOperator>(std::move(extension_types),
	                                                                            PhysicalOperatorType::PROJECTION);
	invalid_root.children.push_back(copy_root);
	generated_plan->SetRoot(invalid_root);
	auto physical_plan = DuckPhysicalPlanRef(generated_plan.release());

	auto workers = setup_workers({{make_worker_id("wrong-root-w1"), 1}});
	auto worker_manager = std::make_shared<MockWorkerManager>(std::move(workers));
	auto runner = std::make_shared<PlanRunner>(worker_manager, con.context);
	auto execution_config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
	auto distributed_plan = std::make_shared<DistributedPhysicalPlan>(18, "planrunner-extension-wrong-root",
	                                                                  physical_plan, std::move(execution_config));

	auto result = runner->run_plan(std::move(distributed_plan));

	REQUIRE(result.is_err());
	REQUIRE(StringUtil::Contains(result.error().what(), "EXTENSION physical root"));
	REQUIRE_FALSE(FileSystem::GetFileSystem(*con.context).DirectoryExists(output_path));
}

TEST_CASE("PlanRunner replays a committed extension write without coordinator or worker side effects",
          "[distributed][plan][copy][extension-write]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterReplayTestExtension(*db.instance);
	auto output_path = TestCreatePath("planrunner_committed_extension_replay");
	auto logical_plan = con.ExtractPlan("COPY (SELECT 42 AS value) TO " + PlanRunnerSQLStringLiteral(output_path) +
	                                    " (FORMAT PARQUET, RETURN_STATS true, USE_TMP_FILE false)");
	REQUIRE(logical_plan != nullptr);

	PhysicalPlanGenerator generator(*con.context);
	auto generated_plan = generator.Plan(std::move(logical_plan));
	REQUIRE(generated_plan != nullptr);
	auto &copy_root = generated_plan->Root();
	REQUIRE(copy_root.type == PhysicalOperatorType::COPY_TO_FILE);

	vector<LogicalType> extension_types {LogicalType::BIGINT};
	auto &extension_operator = generated_plan->Make<ReplayTestExtensionWriteOperator>(std::move(extension_types));
	auto &extension = extension_operator.Cast<ReplayTestExtensionWriteOperator>();
	extension_operator.children.push_back(copy_root);
	generated_plan->SetRoot(extension_operator);
	auto physical_plan = DuckPhysicalPlanRef(generated_plan.release());

	const string query_id = "planrunner-committed-extension-replay";
	auto execution_config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
	PlanConfig translation_config(17, query_id, execution_config);
	translation_config.db = db.instance;
	auto translated = physical_plan_to_pipeline_node(translation_config, physical_plan, con.context.get());
	REQUIRE(translated.is_ok());
	auto copy_finish = std::dynamic_pointer_cast<CopyFinishNode>(translated.value()->inner());
	REQUIRE(copy_finish != nullptr);
	copy_finish->copy_sink()->SetOperationIdentity(query_id);

	auto &fs = FileSystem::GetFileSystem(*con.context);
	auto base_path_res = CanonicalDistributedCopyBasePath(fs, copy_finish->spec());
	REQUIRE(base_path_res.is_ok());
	auto worker_base_res = CanonicalDistributedCopyBasePath(fs, copy_finish->spec().file_path);
	REQUIRE(worker_base_res.is_ok());
	auto base_path = std::move(base_path_res).value();
	auto worker_base_path = std::move(worker_base_res).value();
	auto run_id = copy_finish->staging_run_id();
	REQUIRE_FALSE(run_id.empty());
	REQUIRE_NOTHROW(copy_finish->copy_sink()->SetOperationIdentity(query_id));
	REQUIRE_THROWS_WITH(copy_finish->copy_sink()->SetOperationIdentity(query_id + "-different"),
	                    Catch::Matchers::Contains("identity cannot change"));
	REQUIRE(WriteDistributedCopyDirectWriteLifecycle(fs, base_path, run_id, 1, worker_base_path).is_ok());

	const string contents = "already committed extension output";
	auto data_path = BuildCopyDirectTargetFilePath(worker_base_path, run_id, "w_replay", "part.parquet",
	                                               fs.PathSeparator(worker_base_path));
	WritePlanRunnerTestFile(fs, data_path, contents);
	DistributedCopyFileInfo file_info;
	file_info.staging_path = data_path;
	file_info.row_count = 7;
	file_info.file_size_bytes = contents.size();
	vector<DistributedCopyFileInfo> files;
	files.push_back(std::move(file_info));
	REQUIRE(ProtectDistributedCopyDirectWriteCatalogCommit(fs, base_path, run_id).is_ok());

	auto workers = setup_workers({{make_worker_id("replay-w1"), 1}});
	auto worker_manager = std::make_shared<MockWorkerManager>(std::move(workers));
	auto runner = std::make_shared<PlanRunner>(worker_manager, con.context);
	auto distributed_plan =
	    std::make_shared<DistributedPhysicalPlan>(17, query_id, physical_plan, std::move(execution_config));

	auto pending_without_manifest = runner->run_plan(distributed_plan);
	REQUIRE(pending_without_manifest.is_err());
	REQUIRE(StringUtil::Contains(pending_without_manifest.error().what(), "catalog-commit-pending lifecycle"));
	REQUIRE(fs.FileExists(data_path));
	REQUIRE(extension.validation_calls == 0);
	REQUIRE(extension.finalize_calls == 0);

	auto prepare_res = FinalizeCopyFiles(copy_finish->spec(), "", std::move(files), *con.context, run_id, false);
	REQUIRE(prepare_res.is_ok());
	auto prepared = std::move(prepare_res).value();
	REQUIRE_FALSE(prepared.output_committed);
	REQUIRE(fs.FileExists(prepared.output_manifest_path));
	REQUIRE_FALSE(fs.FileExists(prepared.output_committed_marker_path));

	auto uncertain_replay = runner->run_plan(distributed_plan);

	REQUIRE(uncertain_replay.is_err());
	REQUIRE(StringUtil::Contains(uncertain_replay.error().what(), "catalog commit outcome is unknown"));
	REQUIRE(fs.FileExists(data_path));
	REQUIRE(extension.validation_calls == 0);
	REQUIRE(extension.finalize_calls == 0);

	auto commit_res = CommitPreparedDistributedCopyDirectWriteResult(std::move(prepared), *con.context);
	REQUIRE(commit_res.is_ok());
	REQUIRE(commit_res.value().output_committed);

	auto replay = runner->run_plan(std::move(distributed_plan));

	REQUIRE(replay.is_ok());
	REQUIRE(replay.value().tag == PlanRunner::PlanResult::EXTENSION_WRITE);
	REQUIRE(replay.value().extension_write_result.file_result.output_committed);
	REQUIRE(replay.value().extension_write_result.catalog_committed);
	REQUIRE(replay.value().extension_write_result.info.Name() == "write");
	REQUIRE(replay.value().extension_write_result.rows_written == 7);
	REQUIRE(replay.value().extension_write_result.selected_task_results.size() == 1);
	REQUIRE(replay.value().extension_write_result.selected_task_results[0].operation_id == query_id);
	REQUIRE(extension.validation_calls == 0);
	REQUIRE(extension.finalize_calls == 0);
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
