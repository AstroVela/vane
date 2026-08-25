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
#include "duckdb/execution/operator/scan/physical_empty_result.hpp"
#include "duckdb/execution/distributed/copy_finalize.hpp"
#include "duckdb/execution/distributed/extension_write_task_provider.hpp"
#include "duckdb/execution/distributed/pipeline_node/copy_finish.hpp"
#include "duckdb/execution/distributed/pipeline_node/translator.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/execution/distributed/plan/distributed_physical_plan.hpp"
#include "duckdb/execution/operator/helper/physical_data_sink.hpp"
#include "test_helpers.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
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

class PlanRunnerTestWriteGlobalState final : public DistributedWriteGlobalState {};

class PlanRunnerTestWriteLocalState final : public DistributedWriteLocalState {};

unique_ptr<DistributedWriteGlobalState> PlanRunnerTestWriteInitializeGlobal(ClientContext &,
                                                                            const DistributedExtensionWriteInfo &,
                                                                            const DistributedWriteTaskContext &) {
	return make_uniq<PlanRunnerTestWriteGlobalState>();
}

unique_ptr<DistributedWriteLocalState> PlanRunnerTestWriteInitializeLocal(ExecutionContext &,
                                                                          const DistributedExtensionWriteInfo &,
                                                                          const DistributedWriteTaskContext &,
                                                                          DistributedWriteGlobalState &) {
	return make_uniq<PlanRunnerTestWriteLocalState>();
}

void PlanRunnerTestWriteSink(ExecutionContext &, const DistributedExtensionWriteInfo &,
                             const DistributedWriteTaskContext &, DistributedWriteGlobalState &,
                             DistributedWriteLocalState &, DataChunk &) {
}

void PlanRunnerTestWriteCombine(ExecutionContext &, const DistributedExtensionWriteInfo &,
                                const DistributedWriteTaskContext &, DistributedWriteGlobalState &,
                                DistributedWriteLocalState &) {
}

vector<DistributedWriteFragment> PlanRunnerTestWriteFinalize(ClientContext &, const DistributedExtensionWriteInfo &,
                                                             const DistributedWriteTaskContext &,
                                                             DistributedWriteGlobalState &) {
	return {};
}

class CallbackResultWorkerManager final : public WorkerManager {
public:
	explicit CallbackResultWorkerManager(DistributedExtensionWriteInfo info_p) : info(std::move(info_p)) {
	}

	DuckDBResult<std::vector<WorkerSnapshot>> worker_snapshots() const override {
		std::vector<WorkerSnapshot> snapshots;
		snapshots.emplace_back(make_worker_id("callback-w1"), 1, 0);
		return DuckDBResult<std::vector<WorkerSnapshot>>::ok(std::move(snapshots));
	}

	DuckDBResult<void> try_autoscale(const std::vector<TaskResourceRequest> &) override {
		return DuckDBResult<void>::ok();
	}

	DuckDBResult<void> shutdown() override {
		return DuckDBResult<void>::ok();
	}

	DuckDBResult<void> submit_fte_task_events(std::vector<WorkerTask> tasks) override {
		for (const auto &task : tasks) {
			task_contexts.push_back(task.task_context());
		}
		return DuckDBResult<void>::ok();
	}

	DuckDBResult<void> task_input_stream_exhausted_for_query(const string &,
	                                                         const std::unordered_set<SourceNodeId> &) override {
		return DuckDBResult<void>::ok();
	}

	DuckDBResult<void> materialization_barrier_completed(const string &, NodeID) override {
		return DuckDBResult<void>::ok();
	}

	DuckDBResult<std::vector<MaterializedOutput>> wait_fte_query(const string &query_id, double) override {
		std::vector<MaterializedOutput> outputs;
		for (const auto &task_context : task_contexts) {
			if (task_context.node_ids().empty()) {
				return DuckDBResult<std::vector<MaterializedOutput>>::err(
				    DuckDBError::invalid_state_error("callback test task has no pipeline node identity"));
			}
			DistributedWriteFragment fragment;
			fragment.fragment_id = "fragment-" + std::to_string(task_context.task_id());
			fragment.row_count = 3;
			fragment.byte_count = 3 * sizeof(uint64_t);

			DistributedWriteTaskResult task_result;
			task_result.capability = info.capability;
			task_result.fragment_codec = info.fragment_codec;
			task_result.query_id = query_id;
			task_result.task_attempt_id = "attempt-" + std::to_string(task_context.task_id());
			task_result.fragments.push_back(std::move(fragment));
			auto bytes = task_result.SerializeToBytes();

			auto collection = std::make_shared<ColumnDataCollection>(Allocator::DefaultAllocator(),
			                                                         vector<LogicalType> {LogicalType::BLOB});
			DataChunk chunk;
			chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::BLOB});
			chunk.SetValue(0, 0, Value::BLOB(reinterpret_cast<const_data_ptr_t>(bytes.data()), bytes.size()));
			chunk.SetCardinality(1);
			collection->Append(chunk);
			vector<ResultPartitionRef> partitions;
			partitions.push_back(std::make_shared<ColumnDataResultPartition>(std::move(collection)));
			outputs.emplace_back(std::move(partitions), make_worker_id("callback-w1"), task_context.node_ids());
		}
		return DuckDBResult<std::vector<MaterializedOutput>>::ok(std::move(outputs));
	}

	DuckDBResult<void> abort_and_quiesce_query(const string &) override {
		abort_calls++;
		return DuckDBResult<void>::ok();
	}

	idx_t abort_calls = 0;

private:
	DistributedExtensionWriteInfo info;
	vector<TaskContext> task_contexts;
};

class InvalidDataSinkResultWorkerManager final : public WorkerManager {
public:
	DuckDBResult<std::vector<WorkerSnapshot>> worker_snapshots() const override {
		std::vector<WorkerSnapshot> snapshots;
		snapshots.emplace_back(make_worker_id("datasink-w1"), 1, 0);
		return DuckDBResult<std::vector<WorkerSnapshot>>::ok(std::move(snapshots));
	}

	DuckDBResult<void> try_autoscale(const std::vector<TaskResourceRequest> &) override {
		return DuckDBResult<void>::ok();
	}

	DuckDBResult<void> shutdown() override {
		return DuckDBResult<void>::ok();
	}

	DuckDBResult<void> submit_fte_task_events(std::vector<WorkerTask> tasks) override {
		for (const auto &task : tasks) {
			task_contexts.push_back(task.task_context());
		}
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
		    DuckDBError::invalid_state_error("DataSink must use streaming result validation"));
	}

	DuckDBResult<std::vector<MaterializedOutput>>
	wait_fte_query_streaming(const string &, double, MaterializedOutputCallback on_output) override {
		if (task_contexts.empty() || task_contexts.front().node_ids().empty()) {
			return DuckDBResult<std::vector<MaterializedOutput>>::err(
			    DuckDBError::invalid_state_error("DataSink test task has no pipeline node identity"));
		}
		vector<LogicalType> invalid_types {LogicalType::VARCHAR};
		auto collection = std::make_shared<ColumnDataCollection>(Allocator::DefaultAllocator(), invalid_types);
		DataChunk chunk;
		chunk.Initialize(Allocator::DefaultAllocator(), invalid_types);
		chunk.SetValue(0, 0, Value("invalid-result"));
		chunk.SetCardinality(1);
		collection->Append(chunk);
		vector<ResultPartitionRef> partitions;
		partitions.push_back(std::make_shared<ColumnDataResultPartition>(std::move(collection)));
		MaterializedOutput output(std::move(partitions), make_worker_id("datasink-w1"),
		                          task_contexts.front().node_ids());
		auto callback_result = on_output(output);
		if (callback_result.is_err()) {
			return DuckDBResult<std::vector<MaterializedOutput>>::err(callback_result.error());
		}
		handle_released = true;
		std::vector<MaterializedOutput> outputs;
		return DuckDBResult<std::vector<MaterializedOutput>>::ok(std::move(outputs));
	}

	DuckDBResult<void> abort_and_quiesce_query(const string &) override {
		abort_calls++;
		return DuckDBResult<void>::ok();
	}

	bool handle_released = false;
	idx_t abort_calls = 0;

private:
	vector<TaskContext> task_contexts;
};

class BackpressuredDataSinkWorkerManager final : public WorkerManager {
public:
	explicit BackpressuredDataSinkWorkerManager(string operation_id_p) : operation_id(std::move(operation_id_p)) {
	}

	DuckDBResult<std::vector<WorkerSnapshot>> worker_snapshots() const override {
		std::vector<WorkerSnapshot> snapshots;
		snapshots.emplace_back(make_worker_id("datasink-backpressure-w1"), 1, 0);
		return DuckDBResult<std::vector<WorkerSnapshot>>::ok(std::move(snapshots));
	}

	DuckDBResult<void> try_autoscale(const std::vector<TaskResourceRequest> &) override {
		return DuckDBResult<void>::ok();
	}

	DuckDBResult<void> shutdown() override {
		return DuckDBResult<void>::ok();
	}

	DuckDBResult<void> submit_fte_task_events(std::vector<WorkerTask> tasks) override {
		for (const auto &task : tasks) {
			task_contexts.push_back(task.task_context());
		}
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
		    DuckDBError::invalid_state_error("DataSink must use streaming result validation"));
	}

	DuckDBResult<std::vector<MaterializedOutput>>
	wait_fte_query_streaming(const string &, double, MaterializedOutputCallback on_output) override {
		if (task_contexts.empty() || task_contexts.front().node_ids().empty()) {
			return DuckDBResult<std::vector<MaterializedOutput>>::err(
			    DuckDBError::invalid_state_error("DataSink test task has no pipeline node identity"));
		}
		auto first_result = on_output(MakeOutput());
		if (first_result.is_err()) {
			return DuckDBResult<std::vector<MaterializedOutput>>::err(first_result.error());
		}
		{
			std::lock_guard<std::mutex> guard(lock);
			second_publish_started = true;
		}
		condition.notify_all();
		auto second_result = on_output(MakeOutput());
		{
			std::lock_guard<std::mutex> guard(lock);
			second_publish_completed = true;
		}
		condition.notify_all();
		if (second_result.is_err()) {
			return DuckDBResult<std::vector<MaterializedOutput>>::err(second_result.error());
		}
		std::vector<MaterializedOutput> outputs;
		return DuckDBResult<std::vector<MaterializedOutput>>::ok(std::move(outputs));
	}

	DuckDBResult<void> abort_and_quiesce_query(const string &) override {
		abort_calls++;
		std::unique_lock<std::mutex> guard(lock);
		abort_observed_completed =
		    condition.wait_for(guard, std::chrono::seconds(5), [&]() { return second_publish_completed; });
		if (!abort_observed_completed) {
			return DuckDBResult<void>::err(
			    DuckDBError::external_error("timed out waiting for blocked DataSink result publication"));
		}
		return DuckDBResult<void>::ok();
	}

	bool WaitForSecondPublish() {
		std::unique_lock<std::mutex> guard(lock);
		return condition.wait_for(guard, std::chrono::seconds(5), [&]() { return second_publish_started; });
	}

	idx_t abort_calls = 0;
	bool abort_observed_completed = false;
	bool second_publish_completed = false;

private:
	MaterializedOutput MakeOutput() const {
		vector<LogicalType> types {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::UBIGINT,
		                           LogicalType::UBIGINT, LogicalType::UBIGINT, LogicalType::VARCHAR,
		                           LogicalType::VARCHAR};
		auto collection = std::make_shared<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
		DataChunk chunk;
		chunk.Initialize(Allocator::DefaultAllocator(), types);
		chunk.SetValue(0, 0, Value(operation_id));
		chunk.SetValue(1, 0, Value("applied"));
		chunk.SetValue(2, 0, Value::UBIGINT(1));
		chunk.SetValue(3, 0, Value::UBIGINT(1));
		chunk.SetValue(4, 0, Value::UBIGINT(sizeof(uint64_t)));
		chunk.SetValue(5, 0, Value("{}"));
		chunk.SetValue(6, 0, Value("[]"));
		chunk.SetCardinality(1);
		collection->Append(chunk);
		vector<ResultPartitionRef> partitions;
		partitions.push_back(std::make_shared<ColumnDataResultPartition>(std::move(collection)));
		return MaterializedOutput(std::move(partitions), make_worker_id("datasink-backpressure-w1"),
		                          task_contexts.front().node_ids());
	}

	string operation_id;
	vector<TaskContext> task_contexts;
	std::mutex lock;
	std::condition_variable condition;
	bool second_publish_started = false;
};

class OversizedDataSinkErrorWorkerManager final : public WorkerManager {
public:
	DuckDBResult<std::vector<WorkerSnapshot>> worker_snapshots() const override {
		std::vector<WorkerSnapshot> snapshots;
		snapshots.emplace_back(make_worker_id("datasink-error-w1"), 1, 0);
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

	DuckDBResult<std::vector<MaterializedOutput>> wait_fte_query_streaming(const string &, double,
	                                                                       MaterializedOutputCallback) override {
		return DuckDBResult<std::vector<MaterializedOutput>>::err(
		    DuckDBError::external_error("oversized-worker-error:" + string(100000, 'x')));
	}

	DuckDBResult<void> abort_and_quiesce_query(const string &) override {
		return DuckDBResult<void>::ok();
	}
};

class PlanRunnerTestExtensionWriteOperator final : public PhysicalOperator, public ExtensionWriteTaskProvider {
public:
	PlanRunnerTestExtensionWriteOperator(PhysicalPlan &physical_plan, vector<LogicalType> types,
	                                     PhysicalOperatorType operator_type = PhysicalOperatorType::EXTENSION,
	                                     string operator_name = "write")
	    : PhysicalOperator(physical_plan, operator_type, std::move(types), 0) {
		plan.extension_name = "planrunner_test_extension";
		plan.operator_name = std::move(operator_name);
	}

	optional_ptr<ExtensionWriteTaskProvider> GetExtensionWriteTaskProvider() override {
		return this;
	}

	const DistributedExtensionWritePlan &WritePlan() const override {
		return plan;
	}

	void ValidateDistributedWrite(ClientContext &) const override {
		validation_calls++;
		if (fail_validation) {
			throw InvalidInputException("planned extension write validation failure");
		}
	}

	idx_t FinalizeDistributedWrite(ClientContext &, const vector<DistributedWriteTaskResult> &results) const override {
		finalize_calls++;
		if (fail_finalize) {
			throw IOException("planned extension coordinator finalization failure");
		}
		idx_t rows = 0;
		for (const auto &result : results) {
			rows += result.RowCount();
		}
		return mismatch_finalize_rows ? rows + 1 : rows;
	}

	void AbortDistributedWrite(ClientContext &, const vector<DistributedWriteTaskResult> &) const override {
		abort_calls++;
	}

	mutable idx_t validation_calls = 0;
	mutable idx_t finalize_calls = 0;
	mutable idx_t abort_calls = 0;
	bool fail_validation = false;
	bool fail_finalize = false;
	bool mismatch_finalize_rows = false;

private:
	DistributedExtensionWritePlan plan;
};

string PlanRunnerSQLStringLiteral(const string &value) {
	return "'" + StringUtil::Replace(value, "'", "''") + "'";
}

void RegisterPlanRunnerTestExtension(DatabaseInstance &db) {
	auto &manager = DistributedExtensionManager::Get(db);
	DistributedExtensionManifest manifest;
	manifest.extension_name = "planrunner_test_extension";
	manifest.capabilities.push_back({DistributedExtensionCapabilityKind::WRITE_OPERATOR, "write", 1});
	manifest.capabilities.push_back({DistributedExtensionCapabilityKind::WRITE_OPERATOR, "callback_write", 1});
	DistributedWriteOperatorExtension write_operator;
	write_operator.name = "write";
	write_operator.protocol_version = 1;
	write_operator.mode = DistributedWriteMode::FILE_ARTIFACT;
	write_operator.fragment_codec = {DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC,
	                                 DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC_VERSION};
	DistributedWriteOperatorExtension callback_write_operator;
	callback_write_operator.name = "callback_write";
	callback_write_operator.protocol_version = 1;
	callback_write_operator.mode = DistributedWriteMode::CALLBACK;
	callback_write_operator.fragment_codec = {"planrunner-test.fragment", 1};
	callback_write_operator.callbacks.initialize_global = PlanRunnerTestWriteInitializeGlobal;
	callback_write_operator.callbacks.initialize_local = PlanRunnerTestWriteInitializeLocal;
	callback_write_operator.callbacks.sink = PlanRunnerTestWriteSink;
	callback_write_operator.callbacks.combine = PlanRunnerTestWriteCombine;
	callback_write_operator.callbacks.finalize = PlanRunnerTestWriteFinalize;
	manager.RegisterExtension(
	    manifest, {make_shared_ptr<const DistributedWriteOperatorExtension>(std::move(write_operator)),
	               make_shared_ptr<const DistributedWriteOperatorExtension>(std::move(callback_write_operator))});
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

TEST_CASE("PlanRunner validates DataSink output before acknowledging its selected handle",
          "[distributed][plan][datasink]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto physical_plan = std::make_shared<PhysicalPlan>(Allocator::DefaultAllocator());
	vector<LogicalType> result_types {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::UBIGINT,
	                                  LogicalType::UBIGINT, LogicalType::UBIGINT, LogicalType::VARCHAR,
	                                  LogicalType::VARCHAR};
	auto &child = physical_plan->Make<PhysicalDummyScan>(result_types, 1);
	auto &sink = physical_plan->Make<PhysicalDataSink>(std::move(result_types), "validate-before-ack", 1);
	sink.children.push_back(child);
	physical_plan->SetRoot(sink);

	auto worker_manager = std::make_shared<InvalidDataSinkResultWorkerManager>();
	auto runner = std::make_shared<PlanRunner>(worker_manager, con.context);
	auto execution_config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
	auto distributed_plan = std::make_shared<DistributedPhysicalPlan>(26, "planrunner-datasink-validate-before-ack",
	                                                                  physical_plan, std::move(execution_config));

	auto result = runner->run_plan(std::move(distributed_plan));

	REQUIRE(result.is_ok());
	REQUIRE(result.value().tag == PlanRunner::PlanResult::DATA_SINK);
	REQUIRE(result.value().data_sink_result.outcome_unknown);
	REQUIRE_FALSE(worker_manager->handle_released);
	REQUIRE(worker_manager->abort_calls == 1);
}

TEST_CASE("PlanRunner disconnects DataSink backpressure before aborting after a startup callback failure",
          "[distributed][plan][datasink]") {
	DuckDB db(nullptr);
	Connection con(db);
	const string operation_id = "startup-callback-backpressure";
	auto physical_plan = std::make_shared<PhysicalPlan>(Allocator::DefaultAllocator());
	vector<LogicalType> result_types {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::UBIGINT,
	                                  LogicalType::UBIGINT, LogicalType::UBIGINT, LogicalType::VARCHAR,
	                                  LogicalType::VARCHAR};
	auto &child = physical_plan->Make<PhysicalDummyScan>(result_types, 1);
	auto &sink = physical_plan->Make<PhysicalDataSink>(std::move(result_types), operation_id, 1);
	sink.children.push_back(child);
	physical_plan->SetRoot(sink);

	auto worker_manager = std::make_shared<BackpressuredDataSinkWorkerManager>(operation_id);
	auto runner = std::make_shared<PlanRunner>(worker_manager, con.context);
	auto execution_config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
	auto distributed_plan = std::make_shared<DistributedPhysicalPlan>(27, "planrunner-datasink-startup-callback",
	                                                                  physical_plan, std::move(execution_config));

	auto result = runner->run_plan(std::move(distributed_plan), {}, [&]() {
		if (!worker_manager->WaitForSecondPublish()) {
			throw std::runtime_error("DataSink worker did not reach its backpressured publication");
		}
		throw std::runtime_error("planned DataSink startup callback failure");
	});

	REQUIRE(result.is_ok());
	REQUIRE(result.value().tag == PlanRunner::PlanResult::DATA_SINK);
	REQUIRE(result.value().data_sink_result.outcome_unknown);
	REQUIRE(StringUtil::Contains(result.value().data_sink_result.outcome_error,
	                             "planned DataSink startup callback failure"));
	REQUIRE(worker_manager->abort_calls == 1);
	REQUIRE(worker_manager->abort_observed_completed);
	REQUIRE(worker_manager->second_publish_completed);
}

TEST_CASE("PlanRunner bounds direct DataSink execution errors before retaining them", "[distributed][plan][datasink]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto physical_plan = std::make_shared<PhysicalPlan>(Allocator::DefaultAllocator());
	vector<LogicalType> result_types {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::UBIGINT,
	                                  LogicalType::UBIGINT, LogicalType::UBIGINT, LogicalType::VARCHAR,
	                                  LogicalType::VARCHAR};
	auto &child = physical_plan->Make<PhysicalDummyScan>(result_types, 1);
	auto &sink = physical_plan->Make<PhysicalDataSink>(std::move(result_types), "bounded-execution-error", 1);
	sink.children.push_back(child);
	physical_plan->SetRoot(sink);

	PlanConfig config;
	config.query_id = "planrunner-datasink-bounded-execution-error";
	config.config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
	auto translated = physical_plan_to_pipeline_node(config, physical_plan, con.context.get());
	REQUIRE(translated.is_ok());

	auto worker_manager = std::make_shared<OversizedDataSinkErrorWorkerManager>();
	auto runner = std::make_shared<PlanRunner>(worker_manager, con.context);
	auto task_executor = std::make_shared<PlanTaskExecutor>(con.context);
	auto output_channel = create_unbounded_channel<MaterializedOutput>();
	auto execution_result = runner->execute_plan(translated.value(), task_executor, std::move(output_channel.first), {},
	                                             [](const MaterializedOutput &) { return DuckDBResult<void>::ok(); });

	REQUIRE(execution_result.is_err());
	const string retained_error = execution_result.error().what();
	REQUIRE(retained_error.size() <= DATA_SINK_MAX_OUTCOME_ERROR_BYTES + 128);
	REQUIRE(StringUtil::Contains(retained_error, "oversized-worker-error:"));
}

TEST_CASE("PlanRunner finishes an empty-result plan without submitting worker tasks", "[distributed][plan][local]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto logical_plan = con.ExtractPlan("SELECT * FROM (VALUES (1), (2)) AS input(x) WHERE FALSE");
	REQUIRE(logical_plan != nullptr);

	PhysicalPlanGenerator generator(*con.context);
	auto generated_plan = generator.Plan(std::move(logical_plan));
	REQUIRE(generated_plan != nullptr);
	REQUIRE(generated_plan->Root().type == PhysicalOperatorType::EMPTY_RESULT);
	auto physical_plan = DuckPhysicalPlanRef(generated_plan.release());

	auto workers = setup_workers({{make_worker_id("empty-result-w1"), 1}});
	auto worker_manager = std::make_shared<MockWorkerManager>(std::move(workers));
	auto runner = std::make_shared<PlanRunner>(worker_manager, con.context);
	auto execution_config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
	auto distributed_plan = std::make_shared<DistributedPhysicalPlan>(17, "planrunner-empty-result", physical_plan,
	                                                                  std::move(execution_config));

	auto result = runner->run_plan(std::move(distributed_plan));
	REQUIRE(result.is_ok());
	REQUIRE(result.value().tag == PlanRunner::PlanResult::STREAMING);
	REQUIRE_FALSE(result.value().stream.next().first);
}

TEST_CASE("PlanRunner rejects an unregistered extension write capability",
          "[distributed][plan][copy][extension-write]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto physical_plan = std::make_shared<PhysicalPlan>(Allocator::DefaultAllocator());
	vector<LogicalType> child_types {LogicalType::BIGINT};
	auto &child = physical_plan->Make<PhysicalDummyScan>(std::move(child_types), 1);
	vector<LogicalType> extension_types {LogicalType::BIGINT};
	auto &extension_operator = physical_plan->Make<PlanRunnerTestExtensionWriteOperator>(std::move(extension_types));
	auto &extension = extension_operator.Cast<PlanRunnerTestExtensionWriteOperator>();
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
	REQUIRE(StringUtil::Contains(result.error().what(), "planrunner_test_extension"));
	REQUIRE(extension.validation_calls == 0);
	REQUIRE(extension.finalize_calls == 0);
}

TEST_CASE("PlanRunner rejects extension writes inside an explicit DuckDB transaction",
          "[distributed][plan][copy][extension-write]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterPlanRunnerTestExtension(*db.instance);
	auto output_path = TestCreatePath("planrunner_extension_explicit_transaction");
	auto logical_plan = con.ExtractPlan("COPY (SELECT 42 AS value) TO " + PlanRunnerSQLStringLiteral(output_path) +
	                                    " (FORMAT PARQUET, RETURN_STATS true, USE_TMP_FILE false)");
	REQUIRE(logical_plan != nullptr);

	PhysicalPlanGenerator generator(*con.context);
	auto generated_plan = generator.Plan(std::move(logical_plan));
	REQUIRE(generated_plan != nullptr);
	auto &copy_root = generated_plan->Root();

	vector<LogicalType> extension_types {LogicalType::BIGINT};
	auto &extension_operator = generated_plan->Make<PlanRunnerTestExtensionWriteOperator>(std::move(extension_types));
	auto &extension = extension_operator.Cast<PlanRunnerTestExtensionWriteOperator>();
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

TEST_CASE("PlanRunner preserves extension state when coordinator validation fails before worker output",
          "[distributed][plan][copy][extension-write]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterPlanRunnerTestExtension(*db.instance);
	auto output_path = TestCreatePath("planrunner_extension_validation_failure");
	auto logical_plan = con.ExtractPlan("COPY (SELECT 42 AS value) TO " + PlanRunnerSQLStringLiteral(output_path) +
	                                    " (FORMAT PARQUET, RETURN_STATS true, USE_TMP_FILE false)");
	REQUIRE(logical_plan != nullptr);

	PhysicalPlanGenerator generator(*con.context);
	auto generated_plan = generator.Plan(std::move(logical_plan));
	REQUIRE(generated_plan != nullptr);
	auto &copy_root = generated_plan->Root();

	vector<LogicalType> extension_types {LogicalType::BIGINT};
	auto &extension_operator = generated_plan->Make<PlanRunnerTestExtensionWriteOperator>(std::move(extension_types));
	auto &extension = extension_operator.Cast<PlanRunnerTestExtensionWriteOperator>();
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
	REQUIRE(extension.abort_calls == 0);
	REQUIRE_FALSE(FileSystem::GetFileSystem(*con.context).DirectoryExists(output_path));
}

TEST_CASE("PlanRunner validates an extension write before translating its source",
          "[distributed][plan][copy][extension-write]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterPlanRunnerTestExtension(*db.instance);
	auto physical_plan = std::make_shared<PhysicalPlan>(Allocator::DefaultAllocator());
	vector<LogicalType> extension_types {LogicalType::BIGINT};
	auto &extension_operator = physical_plan->Make<PlanRunnerTestExtensionWriteOperator>(std::move(extension_types));
	auto &extension = extension_operator.Cast<PlanRunnerTestExtensionWriteOperator>();
	extension.fail_validation = true;
	// The missing child makes translation invalid. Validation must still be the
	// first provider-visible operation and must fail before translation examines
	// or enumerates the source tree.
	physical_plan->SetRoot(extension_operator);

	auto workers = setup_workers({{make_worker_id("validation-order-w1"), 1}});
	auto worker_manager = std::make_shared<MockWorkerManager>(std::move(workers));
	auto runner = std::make_shared<PlanRunner>(worker_manager, con.context);
	auto execution_config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
	auto distributed_plan = std::make_shared<DistributedPhysicalPlan>(20, "planrunner-extension-validation-order",
	                                                                  physical_plan, std::move(execution_config));

	DuckDBResult<PlanRunner::PlanResult> result;
	con.context->RunFunctionInTransaction([&]() { result = runner->run_plan(std::move(distributed_plan)); });

	REQUIRE(result.is_err());
	REQUIRE(StringUtil::Contains(result.error().what(), "planned extension write validation failure"));
	REQUIRE_FALSE(StringUtil::Contains(result.error().what(), "requires exactly one child"));
	REQUIRE(extension.validation_calls == 1);
	REQUIRE(extension.finalize_calls == 0);
	REQUIRE(extension.abort_calls == 0);
}

TEST_CASE("PlanRunner preserves extension state when source translation fails before worker output",
          "[distributed][plan][copy][extension-write]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterPlanRunnerTestExtension(*db.instance);
	auto physical_plan = std::make_shared<PhysicalPlan>(Allocator::DefaultAllocator());
	vector<LogicalType> extension_types {LogicalType::BIGINT};
	auto &extension_operator = physical_plan->Make<PlanRunnerTestExtensionWriteOperator>(std::move(extension_types));
	auto &extension = extension_operator.Cast<PlanRunnerTestExtensionWriteOperator>();
	// Validation succeeds, but the missing child makes source translation fail.
	physical_plan->SetRoot(extension_operator);

	auto workers = setup_workers({{make_worker_id("translation-failure-w1"), 1}});
	auto worker_manager = std::make_shared<MockWorkerManager>(std::move(workers));
	auto runner = std::make_shared<PlanRunner>(worker_manager, con.context);
	auto execution_config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
	auto distributed_plan = std::make_shared<DistributedPhysicalPlan>(21, "planrunner-extension-translation-failure",
	                                                                  physical_plan, std::move(execution_config));

	DuckDBResult<PlanRunner::PlanResult> result;
	con.context->RunFunctionInTransaction([&]() { result = runner->run_plan(std::move(distributed_plan)); });

	REQUIRE(result.is_err());
	REQUIRE(StringUtil::Contains(result.error().what(), "requires exactly one child"));
	REQUIRE(extension.validation_calls == 1);
	REQUIRE(extension.finalize_calls == 0);
	REQUIRE(extension.abort_calls == 0);
}

TEST_CASE("PlanRunner cleans its lifecycle without extension abort when execution setup fails",
          "[distributed][plan][copy][extension-write]") {
	ScopedCopyStagingMode direct_write_mode;
	DuckDB db(nullptr);
	Connection con(db);
	RegisterPlanRunnerTestExtension(*db.instance);
	auto &fs = FileSystem::GetFileSystem(*con.context);
	const auto output_path = TestCreatePath("planrunner_extension_setup_failure.parquet");
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
	auto &copy_root = generated_plan->Root();
	vector<LogicalType> extension_types {LogicalType::BIGINT};
	auto &extension_operator = generated_plan->Make<PlanRunnerTestExtensionWriteOperator>(std::move(extension_types));
	auto &extension = extension_operator.Cast<PlanRunnerTestExtensionWriteOperator>();
	extension_operator.children.push_back(copy_root);
	generated_plan->SetRoot(extension_operator);
	auto physical_plan = DuckPhysicalPlanRef(generated_plan.release());

	auto workers = setup_workers({{make_worker_id("setup-failure-w1"), 1}});
	auto worker_manager = std::make_shared<MockWorkerManager>(std::move(workers));
	PlanRunner runner(worker_manager, con.context);
	auto execution_config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
	auto distributed_plan = std::make_shared<DistributedPhysicalPlan>(22, "planrunner-extension-setup-failure",
	                                                                  physical_plan, std::move(execution_config));

	DuckDBResult<PlanRunner::PlanResult> result;
	con.context->RunFunctionInTransaction([&]() { result = runner.run_plan(std::move(distributed_plan)); });

	REQUIRE(result.is_err());
	REQUIRE(StringUtil::Contains(result.error().what(), "requires shared_ptr ownership"));
	REQUIRE(extension.validation_calls == 1);
	REQUIRE(extension.finalize_calls == 0);
	REQUIRE(extension.abort_calls == 0);
	REQUIRE_FALSE(fs.DirectoryExists(output_path));
	REQUIRE_FALSE(fs.DirectoryExists(commit_root));
	REQUIRE_FALSE(fs.DirectoryExists(staging_root));
}

TEST_CASE("PlanRunner requires an EXTENSION root for extension write providers",
          "[distributed][plan][copy][extension-write]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterPlanRunnerTestExtension(*db.instance);
	auto output_path = TestCreatePath("planrunner_extension_wrong_root_type");
	auto logical_plan = con.ExtractPlan("COPY (SELECT 42 AS value) TO " + PlanRunnerSQLStringLiteral(output_path) +
	                                    " (FORMAT PARQUET, RETURN_STATS true, USE_TMP_FILE false)");
	REQUIRE(logical_plan != nullptr);

	PhysicalPlanGenerator generator(*con.context);
	auto generated_plan = generator.Plan(std::move(logical_plan));
	REQUIRE(generated_plan != nullptr);
	auto &copy_root = generated_plan->Root();

	vector<LogicalType> extension_types {LogicalType::BIGINT};
	auto &invalid_root = generated_plan->Make<PlanRunnerTestExtensionWriteOperator>(std::move(extension_types),
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

TEST_CASE("PlanRunner invokes extension coordinator finalization once and never aborts after it starts",
          "[distributed][plan][extension-write]") {
	const auto finalize_case = GENERATE(0, 1, 2);
	const bool fail_finalize = finalize_case == 1;
	const bool mismatch_finalize_rows = finalize_case == 2;
	DuckDB db(nullptr);
	Connection con(db);
	RegisterPlanRunnerTestExtension(*db.instance);

	auto physical_plan = std::make_shared<PhysicalPlan>(Allocator::DefaultAllocator());
	vector<LogicalType> child_types {LogicalType::BIGINT};
	auto &child = physical_plan->Make<PhysicalDummyScan>(std::move(child_types), 1);
	vector<LogicalType> extension_types {LogicalType::BIGINT};
	auto &extension_operator = physical_plan->Make<PlanRunnerTestExtensionWriteOperator>(
	    std::move(extension_types), PhysicalOperatorType::EXTENSION, "callback_write");
	auto &extension = extension_operator.Cast<PlanRunnerTestExtensionWriteOperator>();
	extension.fail_finalize = fail_finalize;
	extension.mismatch_finalize_rows = mismatch_finalize_rows;
	extension_operator.children.push_back(child);
	physical_plan->SetRoot(extension_operator);

	DistributedExtensionWritePlan write_plan;
	write_plan.extension_name = "planrunner_test_extension";
	write_plan.operator_name = "callback_write";
	auto write_info = ResolveDistributedExtensionWriteInfo(*con.context, write_plan);
	auto worker_manager = std::make_shared<CallbackResultWorkerManager>(std::move(write_info));
	auto runner = std::make_shared<PlanRunner>(worker_manager, con.context);
	auto execution_config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
	const string query_id = fail_finalize            ? "planrunner-callback-finalize-failure"
	                        : mismatch_finalize_rows ? "planrunner-callback-finalize-row-mismatch"
	                                                 : "planrunner-callback-finalize-success";
	auto distributed_plan =
	    std::make_shared<DistributedPhysicalPlan>(25, query_id, physical_plan, std::move(execution_config));

	DuckDBResult<PlanRunner::PlanResult> result;
	con.context->RunFunctionInTransaction([&]() { result = runner->run_plan(std::move(distributed_plan)); });

	REQUIRE(result.is_ok());
	REQUIRE(result.value().tag == PlanRunner::PlanResult::EXTENSION_WRITE);
	const auto &extension_result = result.value().extension_write_result;
	REQUIRE(extension_result.rows_written == 3);
	REQUIRE(extension_result.selected_task_results.size() == 1);
	REQUIRE(extension_result.selected_task_results[0].query_id == query_id);
	REQUIRE(extension.validation_calls == 1);
	REQUIRE(extension.finalize_calls == 1);
	REQUIRE(extension.abort_calls == 0);
	REQUIRE(worker_manager->abort_calls == 0);
	REQUIRE(extension_result.outcome_unknown == (fail_finalize || mismatch_finalize_rows));
	REQUIRE_FALSE(extension_result.catalog_committed);
	if (fail_finalize) {
		REQUIRE(
		    StringUtil::Contains(extension_result.outcome_error, "planned extension coordinator finalization failure"));
	} else if (mismatch_finalize_rows) {
		REQUIRE(
		    StringUtil::Contains(extension_result.outcome_error, "extension coordinator finalization returned 4 rows"));
	}
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

TEST_CASE("PlanRunner aborts an extension write after failed workers are quiesced",
          "[distributed][plan][copy][extension-write]") {
	ScopedCopyStagingMode direct_write_mode;
	DuckDB db(nullptr);
	Connection con(db);
	RegisterPlanRunnerTestExtension(*db.instance);
	auto &fs = FileSystem::GetFileSystem(*con.context);
	const auto output_path = TestCreatePath("planrunner_extension_worker_failure.parquet");
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
	auto &copy_root = generated_plan->Root();
	vector<LogicalType> extension_types {LogicalType::BIGINT};
	auto &extension_operator = generated_plan->Make<PlanRunnerTestExtensionWriteOperator>(std::move(extension_types));
	auto &extension = extension_operator.Cast<PlanRunnerTestExtensionWriteOperator>();
	extension_operator.children.push_back(copy_root);
	generated_plan->SetRoot(extension_operator);
	auto physical_plan = DuckPhysicalPlanRef(generated_plan.release());

	auto worker_manager = std::make_shared<FailingCopyWorkerManager>(0, fs, output_path, commit_root);
	auto runner = std::make_shared<PlanRunner>(worker_manager, con.context);
	auto execution_config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
	const string query_id = "planrunner-extension-worker-failure";
	auto distributed_plan =
	    std::make_shared<DistributedPhysicalPlan>(24, query_id, physical_plan, std::move(execution_config));

	DuckDBResult<PlanRunner::PlanResult> result;
	con.context->RunFunctionInTransaction([&]() { result = runner->run_plan(std::move(distributed_plan)); });

	REQUIRE(result.is_err());
	REQUIRE(StringUtil::Contains(result.error().what(), "planned worker execution failure"));
	REQUIRE(worker_manager->abort_calls == 1);
	REQUIRE(worker_manager->output_existed_during_abort);
	REQUIRE(worker_manager->late_writer_completed);
	REQUIRE(extension.validation_calls == 1);
	REQUIRE(extension.finalize_calls == 0);
	REQUIRE(extension.abort_calls == 1);
	REQUIRE_FALSE(fs.DirectoryExists(worker_manager->operation_output_root));
	RemoveDistributedCopyDirectoryTree(fs, output_path);
	RemoveDistributedCopyDirectoryTree(fs, commit_root);
	RemoveDistributedCopyDirectoryTree(fs, staging_root);
}
