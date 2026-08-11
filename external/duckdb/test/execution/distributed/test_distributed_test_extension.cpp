// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/distributed/pipeline_node/translator_scan.hpp"
#include "duckdb/execution/distributed/plan/scan_task.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/extension_scan_task_provider.hpp"
#include "duckdb/main/distributed_extension_manager.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

using namespace duckdb;

namespace {

static constexpr idx_t DISTRIBUTED_TEST_EXTENSION_PROTOCOL = 1;
static constexpr idx_t DISTRIBUTED_TEST_SCAN_PROTOCOL = 1;

static void DistributedTestIdentity(DataChunk &input, ExpressionState &, Vector &result) {
	result.Reference(input.data[0]);
}

struct DistributedTestScanBindData : public TableFunctionData, public ExtensionScanTaskProvider {
	idx_t requested_task_count = 0;
	vector<OpenFileInfo> tasks;

	static OpenFileInfo MakeTask(idx_t task_id) {
		OpenFileInfo task("distributed-test://" + std::to_string(task_id));
		task.extended_info = make_shared_ptr<ExtendedOpenFileInfo>();
		task.extended_info->options["task_id"] = Value::UBIGINT(task_id);
		return task;
	}

	static idx_t TaskId(const OpenFileInfo &task) {
		if (!task.extended_info) {
			throw InternalException("distributed test scan task is missing extended info");
		}
		auto entry = task.extended_info->options.find("task_id");
		if (entry == task.extended_info->options.end()) {
			throw InternalException("distributed test scan task is missing task_id");
		}
		return entry->second.GetValue<idx_t>();
	}

	DistributedExtensionCapabilityReference GetDistributedExtensionCapability() const override {
		DistributedExtensionCapabilityReference reference;
		reference.extension_name = "distributed_test";
		reference.extension_protocol_version = DISTRIBUTED_TEST_EXTENSION_PROTOCOL;
		reference.capability.kind = DistributedExtensionCapabilityKind::TABLE_FUNCTION;
		reference.capability.name = "distributed_test_scan";
		reference.capability.protocol_version = DISTRIBUTED_TEST_SCAN_PROTOCOL;
		return reference;
	}

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<DistributedTestScanBindData>();
		result->requested_task_count = requested_task_count;
		result->tasks = tasks;
		return std::move(result);
	}

	bool Equals(const FunctionData &other) const override {
		auto other_data = dynamic_cast<const DistributedTestScanBindData *>(&other);
		if (!other_data || requested_task_count != other_data->requested_task_count ||
		    tasks.size() != other_data->tasks.size()) {
			return false;
		}
		for (idx_t task_idx = 0; task_idx < tasks.size(); task_idx++) {
			if (tasks[task_idx].path != other_data->tasks[task_idx].path ||
			    TaskId(tasks[task_idx]) != TaskId(other_data->tasks[task_idx])) {
				return false;
			}
		}
		return true;
	}

	vector<OpenFileInfo> GetScanTasks() const override {
		return tasks;
	}

	void SetScanTasks(const vector<OpenFileInfo> &tasks_p) override {
		tasks = tasks_p;
	}

	optional_idx GetScanTaskEstimatedBytes(const OpenFileInfo &) const override {
		return optional_idx(sizeof(uint64_t));
	}

	optional_idx GetScanTaskEstimatedCardinality(const OpenFileInfo &) const override {
		return optional_idx(1);
	}
};

struct DistributedTestScanGlobalState : public GlobalTableFunctionState {
	idx_t task_idx = 0;
};

static unique_ptr<FunctionData> DistributedTestScanBind(ClientContext &, TableFunctionBindInput &input,
                                                        vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.size() != 1 || input.inputs[0].IsNull()) {
		throw BinderException("distributed_test_scan requires a non-null task count");
	}
	auto signed_task_count = input.inputs[0].GetValue<int64_t>();
	if (signed_task_count < 0) {
		throw BinderException("distributed_test_scan task count must not be negative");
	}
	auto task_count = NumericCast<idx_t>(signed_task_count);
	if (task_count > 1024) {
		throw BinderException("distributed_test_scan task count must not exceed 1024");
	}

	return_types.emplace_back(LogicalType::UBIGINT);
	names.emplace_back("task_id");
	auto result = make_uniq<DistributedTestScanBindData>();
	result->requested_task_count = task_count;
	result->tasks.reserve(task_count);
	for (idx_t task_id = 0; task_id < task_count; task_id++) {
		result->tasks.push_back(DistributedTestScanBindData::MakeTask(task_id));
	}
	return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> DistributedTestScanInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<DistributedTestScanGlobalState>();
}

static void DistributedTestScan(ClientContext &, TableFunctionInput &input, DataChunk &output) {
	auto &bind_data = input.bind_data->Cast<DistributedTestScanBindData>();
	auto &global_state = input.global_state->Cast<DistributedTestScanGlobalState>();
	idx_t output_idx = 0;
	while (global_state.task_idx < bind_data.tasks.size() && output_idx < STANDARD_VECTOR_SIZE) {
		output.SetValue(0, output_idx,
		                Value::UBIGINT(DistributedTestScanBindData::TaskId(bind_data.tasks[global_state.task_idx])));
		global_state.task_idx++;
		output_idx++;
	}
	output.SetCardinality(output_idx);
}

class DistributedTestExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override {
		loader.RegisterFunction(ScalarFunction("distributed_test_identity", {LogicalType::UBIGINT},
		                                       LogicalType::UBIGINT, DistributedTestIdentity));
		loader.RegisterFunction(TableFunction("distributed_test_scan", {LogicalType::BIGINT}, DistributedTestScan,
		                                      DistributedTestScanBind, DistributedTestScanInit));
		loader.RegisterDistributedExtension(DISTRIBUTED_TEST_EXTENSION_PROTOCOL);
		loader.RegisterDistributedTableFunction("distributed_test_scan", DISTRIBUTED_TEST_SCAN_PROTOCOL);
	}

	string Name() override {
		return "distributed_test";
	}

	string Version() const override {
		return "1.0.0-test";
	}
};

} // namespace

TEST_CASE("Test extension preserves native execution and distributed scan ownership",
          "[distributed][extension][extension-scan]") {
	DuckDB coordinator_db(nullptr);
	coordinator_db.LoadStaticExtension<DistributedTestExtension>();
	Connection coordinator(coordinator_db);

	auto native_result =
	    coordinator.Query("SELECT distributed_test_identity(task_id) FROM distributed_test_scan(3) ORDER BY task_id");
	REQUIRE_NO_FAIL(*native_result);
	REQUIRE(CHECK_COLUMN(native_result, 0, {0, 1, 2}));

	DistributedExtensionManifest manifest;
	auto &coordinator_manager = DistributedExtensionManager::Get(*coordinator_db.instance);
	REQUIRE(coordinator_manager.TryGetExtension("distributed_test", manifest));
	REQUIRE(manifest.CanonicalIdentity() == "distributed_test@1{table_function:distributed_test_scan@1}");

	auto logical_plan = coordinator.ExtractPlan("SELECT * FROM distributed_test_scan(3)");
	REQUIRE(logical_plan != nullptr);
	PhysicalPlanGenerator generator(*coordinator.context);
	auto generated_plan = generator.Plan(std::move(logical_plan));
	REQUIRE(generated_plan != nullptr);
	auto coordinator_plan = distributed::DuckPhysicalPlanRef(generated_plan.release());
	REQUIRE(coordinator_plan->Root().type == PhysicalOperatorType::TABLE_SCAN);

	distributed::DuckDBExecutionConfig config;
	config.set_distributed_worker_slots(3);
	auto tasks = distributed::MakeTableScanTasks(coordinator_plan->Root().Cast<PhysicalTableScan>(), config,
	                                             coordinator_db.instance);
	REQUIRE(tasks.size() == 3);
	for (idx_t task_idx = 0; task_idx < tasks.size(); task_idx++) {
		REQUIRE(tasks[task_idx].files.size() == 1);
		REQUIRE(tasks[task_idx].estimated_bytes == sizeof(uint64_t));
		REQUIRE(tasks[task_idx].estimated_cardinality == 1);
		REQUIRE(DistributedTestScanBindData::TaskId(tasks[task_idx].files[0]) == task_idx);
	}

	DuckDB worker_db(nullptr);
	worker_db.LoadStaticExtension<DistributedTestExtension>();
	Connection worker(worker_db);
	auto worker_plan =
	    distributed::ClonePhysicalPlanOrThrow(coordinator_plan, "distributed_test_extension", worker.context.get());
	auto &worker_scan = worker_plan->Root().Cast<PhysicalTableScan>();
	auto worker_provider = TryGetExtensionScanTaskProvider(*worker_scan.bind_data);
	REQUIRE(worker_provider);
	REQUIRE(worker_provider->GetScanTasks().size() == 3);
	REQUIRE_NOTHROW(DistributedExtensionManager::Get(*worker_db.instance)
	                    .RequireCapability(worker_provider->GetDistributedExtensionCapability()));

	worker_scan.extra_info.scan_node_id = optional_idx(7);
	unordered_map<idx_t, distributed::ScanTaskDescriptor> assigned_tasks;
	assigned_tasks.emplace(7, tasks[1]);
	string apply_error;
	REQUIRE(distributed::ApplyScanTasksToPlan(*worker_plan, assigned_tasks, &apply_error));
	auto assigned = worker_provider->GetScanTasks();
	REQUIRE(assigned.size() == 1);
	REQUIRE(DistributedTestScanBindData::TaskId(assigned[0]) == 1);
}
