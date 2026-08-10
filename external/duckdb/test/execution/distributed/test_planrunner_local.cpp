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
#include "duckdb/execution/distributed/copy_finalize.hpp"
#include "duckdb/execution/distributed/extension_write_task_provider.hpp"
#include "duckdb/execution/distributed/pipeline_node/copy_finish.hpp"
#include "duckdb/execution/distributed/pipeline_node/translator.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/execution/distributed/plan/distributed_physical_plan.hpp"

using namespace duckdb;
using namespace duckdb::distributed;
using namespace duckdb::distributed::testing;

namespace {

class ReplayTestExtensionWriteOperator final : public PhysicalOperator, public ExtensionWriteTaskProvider {
public:
	ReplayTestExtensionWriteOperator(PhysicalPlan &physical_plan, vector<LogicalType> types,
	                                 PhysicalOperatorType operator_type = PhysicalOperatorType::EXTENSION)
	    : PhysicalOperator(physical_plan, operator_type, std::move(types), 0) {
	}

	optional_ptr<ExtensionWriteTaskProvider> GetExtensionWriteTaskProvider() override {
		return this;
	}

	string ExtensionWriteName() const override {
		return "replay_test_extension";
	}

	void ValidateExtensionWrite(ClientContext &) const override {
		validation_calls++;
	}

	idx_t FinalizeExtensionWrite(ClientContext &, const vector<DistributedCopyFileInfo> &files) const override {
		finalize_calls++;
		idx_t rows = 0;
		for (const auto &file : files) {
			rows += file.row_count;
		}
		return rows;
	}

	mutable idx_t validation_calls = 0;
	mutable idx_t finalize_calls = 0;
};

string PlanRunnerSQLStringLiteral(const string &value) {
	return "'" + StringUtil::Replace(value, "'", "''") + "'";
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

TEST_CASE("PlanRunner rejects extension writes inside an explicit DuckDB transaction",
          "[distributed][plan][copy][extension-write]") {
	DuckDB db(nullptr);
	Connection con(db);
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

TEST_CASE("PlanRunner requires an EXTENSION root for extension write providers",
          "[distributed][plan][copy][extension-write]") {
	DuckDB db(nullptr);
	Connection con(db);
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
	auto prepare_res = FinalizeCopyFiles(copy_finish->spec(), "", std::move(files), *con.context, run_id, false);
	REQUIRE(prepare_res.is_ok());
	auto prepared = std::move(prepare_res).value();
	REQUIRE_FALSE(prepared.output_committed);
	REQUIRE(fs.FileExists(prepared.output_manifest_path));
	REQUIRE_FALSE(fs.FileExists(prepared.output_committed_marker_path));

	auto workers = setup_workers({{make_worker_id("replay-w1"), 1}});
	auto worker_manager = std::make_shared<MockWorkerManager>(std::move(workers));
	auto runner = std::make_shared<PlanRunner>(worker_manager, con.context);
	auto distributed_plan =
	    std::make_shared<DistributedPhysicalPlan>(17, query_id, physical_plan, std::move(execution_config));

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
	REQUIRE(replay.value().copy_result.output_committed);
	REQUIRE(replay.value().copy_result.extension_write);
	REQUIRE(replay.value().copy_result.extension_catalog_committed);
	REQUIRE(replay.value().copy_result.extension_write_name == "replay_test_extension");
	REQUIRE(replay.value().copy_result.rows_copied == 7);
	REQUIRE(extension.validation_calls == 0);
	REQUIRE(extension.finalize_calls == 0);
}
