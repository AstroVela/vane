// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb.hpp"
#include "duckdb/common/atomic.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/distributed/pipeline_node/translator_scan.hpp"
#include "duckdb/execution/distributed/extension_write_task_provider.hpp"
#include "duckdb/execution/distributed/plan/fte_split_queue.hpp"
#include "duckdb/execution/distributed/plan/scan_split.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/execution/operator/persistent/physical_distributed_extension_write.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/execution/operator/scan/physical_dummy_scan.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/operator/set/physical_union.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/distributed_table_function.hpp"
#include "duckdb/main/distributed_extension_manager.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/parallel/thread_context.hpp"

using namespace duckdb;

namespace {

static constexpr idx_t DISTRIBUTED_TEST_SCAN_PROTOCOL = 1;
static constexpr idx_t DISTRIBUTED_TEST_SPLIT_CODEC_VERSION = 1;
static constexpr idx_t DISTRIBUTED_TEST_WRITE_PROTOCOL = 1;
static constexpr idx_t DISTRIBUTED_TEST_WRITE_CODEC_VERSION = 1;
static const string DISTRIBUTED_TEST_SCAN_SPLIT_CODEC = "distributed-test.scan-split";
static const string DISTRIBUTED_TEST_WRITE_CODEC = "distributed-test.opaque-write-fragment";
static const string DISTRIBUTED_TEST_WRITE_NAME = "distributed_test_opaque_write";
static atomic<idx_t> distributed_test_create_worker_bind_calls {0};
static atomic<idx_t> distributed_test_last_target_split_count {0};

struct DistributedTestSourceUnit {
	idx_t unit_id = 0;
	string resource;
	string artifact;
};

struct DistributedTestScanBindData : public TableFunctionData {
	idx_t requested_unit_count = 0;
	vector<DistributedTestSourceUnit> units;

	bool Equals(const FunctionData &other) const override {
		auto other_data = dynamic_cast<const DistributedTestScanBindData *>(&other);
		if (!other_data || requested_unit_count != other_data->requested_unit_count ||
		    units.size() != other_data->units.size()) {
			return false;
		}
		for (idx_t unit_index = 0; unit_index < units.size(); unit_index++) {
			const auto &left = units[unit_index];
			const auto &right = other_data->units[unit_index];
			if (left.unit_id != right.unit_id || left.resource != right.resource || left.artifact != right.artifact) {
				return false;
			}
		}
		return true;
	}
};

struct DistributedTestScanGlobalState : public GlobalTableFunctionState {
	idx_t unit_index = 0;
};

static atomic<idx_t> distributed_test_bind_calls {0};
static atomic<idx_t> distributed_test_deserialize_calls {0};

static string FileResource(idx_t unit_id) {
	return "synthetic-file://partition-" + std::to_string(unit_id) + ".parquet";
}

static string FragmentPayload(idx_t unit_id) {
	string result;
	result.push_back('\0');
	result += "fragment:" + std::to_string(unit_id);
	return result;
}

static string FragmentArtifact(idx_t unit_id) {
	string result;
	result.push_back(static_cast<char>(0x7f));
	result.push_back('\0');
	result += "delete-vector:" + std::to_string(unit_id);
	return result;
}

static string FragmentSplitPayload(idx_t unit_id) {
	MemoryStream stream(Allocator::DefaultAllocator());
	BinarySerializer serializer(stream);
	serializer.Begin();
	serializer.WriteProperty(1, "fragment", FragmentPayload(unit_id));
	serializer.WriteProperty(2, "delete_vector", FragmentArtifact(unit_id));
	serializer.End();
	return string(reinterpret_cast<const char *>(stream.GetData()), stream.GetPosition());
}

static pair<string, string> DecodeFragmentSplitPayload(const string &payload) {
	if (payload.empty()) {
		throw InvalidInputException("distributed test fragment envelope is empty");
	}
	auto *data = reinterpret_cast<data_ptr_t>(const_cast<char *>(payload.data()));
	MemoryStream stream(data, payload.size());
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto fragment = deserializer.ReadProperty<string>(1, "fragment");
	auto delete_vector = deserializer.ReadProperty<string>(2, "delete_vector");
	deserializer.End();
	return {std::move(fragment), std::move(delete_vector)};
}

static idx_t ParseSplitIndex(const string &split_id, const string &prefix) {
	if (!StringUtil::StartsWith(split_id, prefix)) {
		throw InvalidInputException("distributed test split '%s' does not start with '%s'", split_id, prefix);
	}
	auto suffix = split_id.substr(prefix.size());
	if (suffix.empty()) {
		throw InvalidInputException("distributed test split '%s' has no numeric identity", split_id);
	}
	if (suffix.size() > 1 && suffix[0] == '0') {
		throw InvalidInputException("distributed test split '%s' has a non-canonical numeric identity", split_id);
	}
	idx_t result = 0;
	for (auto character : suffix) {
		if (character < '0' || character > '9') {
			throw InvalidInputException("distributed test split '%s' has an invalid numeric identity", split_id);
		}
		const auto digit = NumericCast<idx_t>(character - '0');
		if (result > (NumericLimits<idx_t>::Maximum() - digit) / 10) {
			throw InvalidInputException("distributed test split '%s' numeric identity overflows idx_t", split_id);
		}
		result = result * 10 + digit;
	}
	return result;
}

static DistributedExtensionCapabilityReference DistributedTestCapability() {
	DistributedExtensionCapabilityReference reference;
	reference.extension_name = "distributed_test";
	reference.capability.kind = DistributedExtensionCapabilityKind::TABLE_FUNCTION;
	reference.capability.name = "distributed_test_scan";
	reference.capability.protocol_version = DISTRIBUTED_TEST_SCAN_PROTOCOL;
	reference.capability.function_signature =
	    GetDistributedTableFunctionSignature("distributed_test_scan", {LogicalType::BIGINT}, LogicalType::INVALID);
	return reference;
}

static bool SyntheticExtensionHasContractIdentity(DistributedExtensionManager &manager, const string &identity) {
	for (const auto &registered : manager.GetContractIdentities()) {
		if (registered == identity) {
			return true;
		}
	}
	return false;
}

static string DistributedTestWriteBindData() {
	string result;
	result.push_back('\0');
	result += "distributed-test-write-bind";
	return result;
}

class DistributedTestWriteGlobalState final : public DistributedWriteGlobalState {
public:
	idx_t row_count = 0;
	idx_t value_sum = 0;
};

class DistributedTestWriteLocalState final : public DistributedWriteLocalState {
public:
	idx_t row_count = 0;
	idx_t value_sum = 0;
};

static unique_ptr<DistributedWriteGlobalState>
DistributedTestWriteInitializeGlobal(ClientContext &, const DistributedExtensionWriteInfo &info,
                                     const DistributedWriteTaskContext &task) {
	if (info.worker_bind_data != DistributedTestWriteBindData()) {
		throw InvalidInputException("distributed test write received invalid worker state");
	}
	task.Validate();
	return make_uniq<DistributedTestWriteGlobalState>();
}

static unique_ptr<DistributedWriteLocalState> DistributedTestWriteInitializeLocal(ExecutionContext &,
                                                                                  const DistributedExtensionWriteInfo &,
                                                                                  const DistributedWriteTaskContext &,
                                                                                  DistributedWriteGlobalState &) {
	return make_uniq<DistributedTestWriteLocalState>();
}

static void DistributedTestWriteSink(ExecutionContext &, const DistributedExtensionWriteInfo &,
                                     const DistributedWriteTaskContext &, DistributedWriteGlobalState &,
                                     DistributedWriteLocalState &local_state_p, DataChunk &input) {
	auto &local_state = local_state_p.Cast<DistributedTestWriteLocalState>();
	for (idx_t row = 0; row < input.size(); row++) {
		auto value = input.GetValue(0, row);
		if (!value.IsNull()) {
			local_state.value_sum += value.GetValue<uint64_t>();
		}
		local_state.row_count++;
	}
}

static void DistributedTestWriteCombine(ExecutionContext &, const DistributedExtensionWriteInfo &,
                                        const DistributedWriteTaskContext &,
                                        DistributedWriteGlobalState &global_state_p,
                                        DistributedWriteLocalState &local_state_p) {
	auto &global_state = global_state_p.Cast<DistributedTestWriteGlobalState>();
	auto &local_state = local_state_p.Cast<DistributedTestWriteLocalState>();
	global_state.row_count += local_state.row_count;
	global_state.value_sum += local_state.value_sum;
}

static vector<DistributedWriteFragment> DistributedTestWriteFinalize(ClientContext &,
                                                                     const DistributedExtensionWriteInfo &,
                                                                     const DistributedWriteTaskContext &task,
                                                                     DistributedWriteGlobalState &global_state_p) {
	auto &global_state = global_state_p.Cast<DistributedTestWriteGlobalState>();
	DistributedWriteArtifact artifact;
	artifact.artifact_id = "manifest";
	artifact.uri = "synthetic-write://" + task.query_id + "/" + task.task_attempt_id;
	artifact.codec = {"distributed-test.manifest", 1};
	artifact.payload.push_back('\0');
	artifact.payload += "rows=" + std::to_string(global_state.row_count);

	DistributedWriteFragment fragment;
	fragment.fragment_id = task.query_id + "/" + task.task_attempt_id + "/fragment";
	fragment.payload.push_back('\0');
	fragment.payload += "sum=" + std::to_string(global_state.value_sum);
	fragment.artifacts.push_back(std::move(artifact));
	fragment.row_count = global_state.row_count;
	fragment.byte_count = global_state.row_count * sizeof(uint64_t);
	return {std::move(fragment)};
}

static DistributedExtensionWriteCallbacks DistributedTestWriteCallbacks() {
	DistributedExtensionWriteCallbacks callbacks;
	callbacks.initialize_global = DistributedTestWriteInitializeGlobal;
	callbacks.initialize_local = DistributedTestWriteInitializeLocal;
	callbacks.sink = DistributedTestWriteSink;
	callbacks.combine = DistributedTestWriteCombine;
	callbacks.finalize = DistributedTestWriteFinalize;
	return callbacks;
}

static DistributedWriteOperatorExtension DistributedTestWriteOperator() {
	DistributedWriteOperatorExtension result;
	result.name = DISTRIBUTED_TEST_WRITE_NAME;
	result.protocol_version = DISTRIBUTED_TEST_WRITE_PROTOCOL;
	result.mode = DistributedWriteMode::CALLBACK;
	result.fragment_codec = {DISTRIBUTED_TEST_WRITE_CODEC, DISTRIBUTED_TEST_WRITE_CODEC_VERSION};
	result.callbacks = DistributedTestWriteCallbacks();
	return result;
}

static distributed::DistributedExtensionWritePlan DistributedTestWritePlan() {
	distributed::DistributedExtensionWritePlan result;
	result.extension_name = "distributed_test";
	result.operator_name = DISTRIBUTED_TEST_WRITE_NAME;
	result.worker_bind_data = DistributedTestWriteBindData();
	return result;
}

static DistributedExtensionWriteInfo ResolveDistributedTestWriteInfo(ClientContext &context) {
	return distributed::ResolveDistributedExtensionWriteInfo(context, DistributedTestWritePlan());
}

static unique_ptr<FunctionData> DistributedTestScanBind(ClientContext &, TableFunctionBindInput &input,
                                                        vector<LogicalType> &return_types, vector<string> &names) {
	distributed_test_bind_calls++;
	if (input.inputs.size() != 1 || input.inputs[0].IsNull()) {
		throw BinderException("distributed test scan requires a non-null source unit count");
	}
	auto signed_unit_count = input.inputs[0].GetValue<int64_t>();
	if (signed_unit_count < 0 || signed_unit_count > 1024) {
		throw BinderException("distributed test scan source unit count must be between 0 and 1024");
	}

	return_types.emplace_back(LogicalType::UBIGINT);
	names.emplace_back("unit_id");
	auto result = make_uniq<DistributedTestScanBindData>();
	result->requested_unit_count = NumericCast<idx_t>(signed_unit_count);
	result->units.reserve(result->requested_unit_count);
	for (idx_t unit_id = 0; unit_id < result->requested_unit_count; unit_id++) {
		DistributedTestSourceUnit unit;
		unit.unit_id = unit_id;
		if (unit_id % 2 == 0) {
			unit.resource = FileResource(unit_id);
		} else {
			unit.resource = FragmentPayload(unit_id);
			unit.artifact = FragmentArtifact(unit_id);
		}
		result->units.push_back(std::move(unit));
	}
	return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> DistributedTestScanInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<DistributedTestScanGlobalState>();
}

static void DistributedTestScan(ClientContext &, TableFunctionInput &input, DataChunk &output) {
	auto &bind_data = input.bind_data->Cast<DistributedTestScanBindData>();
	auto &global_state = input.global_state->Cast<DistributedTestScanGlobalState>();
	idx_t output_index = 0;
	while (global_state.unit_index < bind_data.units.size() && output_index < STANDARD_VECTOR_SIZE) {
		output.SetValue(0, output_index, Value::UBIGINT(bind_data.units[global_state.unit_index].unit_id));
		global_state.unit_index++;
		output_index++;
	}
	output.SetCardinality(output_index);
}

static void DistributedTestScanSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data_p,
                                         const TableFunction &) {
	auto &bind_data = bind_data_p->Cast<DistributedTestScanBindData>();
	serializer.WriteProperty(100, "requested_unit_count", bind_data.requested_unit_count);
	serializer.WriteList(101, "units", bind_data.units.size(), [&](Serializer::List &units, idx_t unit_index) {
		const auto &unit = bind_data.units[unit_index];
		units.WriteObject([&](Serializer &unit_serializer) {
			unit_serializer.WriteProperty(1, "unit_id", unit.unit_id);
			unit_serializer.WriteProperty(2, "resource", unit.resource);
			unit_serializer.WriteProperty(3, "artifact", unit.artifact);
		});
	});
}

static unique_ptr<FunctionData> DistributedTestScanDeserialize(Deserializer &deserializer, TableFunction &) {
	distributed_test_deserialize_calls++;
	auto result = make_uniq<DistributedTestScanBindData>();
	result->requested_unit_count = deserializer.ReadProperty<idx_t>(100, "requested_unit_count");
	deserializer.ReadList(101, "units", [&](Deserializer::List &units, idx_t) {
		DistributedTestSourceUnit unit;
		units.ReadObject([&](Deserializer &unit_deserializer) {
			unit.unit_id = unit_deserializer.ReadProperty<idx_t>(1, "unit_id");
			unit.resource = unit_deserializer.ReadProperty<string>(2, "resource");
			unit.artifact = unit_deserializer.ReadProperty<string>(3, "artifact");
		});
		result->units.push_back(std::move(unit));
	});
	return std::move(result);
}

static unique_ptr<FunctionData> DistributedTestCreateWorkerBind(const TableFunctionDistributedScanInput &input) {
	distributed_test_create_worker_bind_calls++;
	if (!input.bind_data) {
		throw InvalidInputException("distributed test scan requires bind data");
	}
	auto &source_bind = input.bind_data->Cast<DistributedTestScanBindData>();
	auto worker_bind = make_uniq<DistributedTestScanBindData>();
	worker_bind->requested_unit_count = source_bind.requested_unit_count;
	return std::move(worker_bind);
}

static vector<DistributedScanSplit> DistributedTestPlanSplits(const TableFunctionDistributedScanPlanningInput &input) {
	distributed_test_last_target_split_count = input.target_split_count;
	if (!input.bind_data) {
		throw InvalidInputException("distributed test scan requires bind data");
	}
	auto &bind_data = input.bind_data->Cast<DistributedTestScanBindData>();
	vector<DistributedScanSplit> result;
	result.reserve(bind_data.units.size());
	for (const auto &unit : bind_data.units) {
		DistributedScanSplit split;
		split.estimated_cardinality = optional_idx(1);
		split.payload = unit.resource;
		if (unit.artifact.empty()) {
			split.split_id = "file-" + std::to_string(unit.unit_id);
			split.estimated_bytes = optional_idx(1024 + unit.unit_id);
		} else {
			split.split_id = "fragment-" + std::to_string(unit.unit_id);
			split.estimated_bytes = optional_idx(2048 + unit.unit_id);
			split.payload = FragmentSplitPayload(unit.unit_id);
		}
		result.push_back(std::move(split));
	}
	return result;
}

static void DistributedTestApplySplits(optional_ptr<FunctionData> worker_bind_data,
                                       const vector<DistributedScanSplit> &splits) {
	if (!worker_bind_data) {
		throw InvalidInputException("distributed test scan requires worker bind data");
	}
	auto &bind_data = worker_bind_data->Cast<DistributedTestScanBindData>();
	vector<DistributedTestSourceUnit> assigned_units;
	assigned_units.reserve(splits.size());
	for (const auto &split : splits) {
		if (StringUtil::StartsWith(split.split_id, "file-")) {
			auto unit_id = ParseSplitIndex(split.split_id, "file-");
			if (split.payload != FileResource(unit_id)) {
				throw InvalidInputException("invalid distributed test file split '%s'", split.split_id);
			}
			assigned_units.push_back({unit_id, split.payload, string()});
		} else {
			auto unit_id = ParseSplitIndex(split.split_id, "fragment-");
			auto fragment = DecodeFragmentSplitPayload(split.payload);
			if (fragment.first != FragmentPayload(unit_id) || fragment.second != FragmentArtifact(unit_id)) {
				throw InvalidInputException("invalid distributed test opaque fragment split '%s'", split.split_id);
			}
			assigned_units.push_back({unit_id, std::move(fragment.first), std::move(fragment.second)});
		}
	}
	bind_data.units = std::move(assigned_units);
}

static TableFunction DistributedTestScanFunction() {
	TableFunction function("distributed_test_scan", {LogicalType::BIGINT}, DistributedTestScan, DistributedTestScanBind,
	                       DistributedTestScanInit);
	function.serialize = DistributedTestScanSerialize;
	function.deserialize = DistributedTestScanDeserialize;
	TableFunctionDistributedScanCallbacks callbacks;
	callbacks.protocol_version = DISTRIBUTED_TEST_SCAN_PROTOCOL;
	callbacks.split_codec = {DISTRIBUTED_TEST_SCAN_SPLIT_CODEC, DISTRIBUTED_TEST_SPLIT_CODEC_VERSION};
	callbacks.plan_splits = DistributedTestPlanSplits;
	callbacks.create_worker_bind = DistributedTestCreateWorkerBind;
	callbacks.apply_splits = DistributedTestApplySplits;
	function.SetDistributedScanCallbacks(std::move(callbacks));
	return function;
}

class DistributedTestExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override {
		loader.RegisterFunction(DistributedTestScanFunction());
		DistributedWriteOperatorExtension::Register(loader, DistributedTestWriteOperator());
	}

	string Name() override {
		return "distributed_test";
	}

	string Version() const override {
		return "1.0.0-test";
	}
};

struct PlannedDistributedTestScan {
	distributed::DuckPhysicalPlanRef worker_plan;
	vector<distributed::ScanSplit> splits;
};

static distributed::ScanSplitBatch MakeScanSplitBatch(vector<distributed::ScanSplit> splits) {
	distributed::ScanSplitBatch batch;
	batch.splits = std::move(splits);
	batch.Validate();
	return batch;
}

static distributed::ScanSplitBatch MakeScanSplitBatch(const distributed::ScanSplit &split) {
	return MakeScanSplitBatch(vector<distributed::ScanSplit> {split});
}

static PlannedDistributedTestScan PlanDistributedTestScan(DuckDB &db, Connection &connection, const string &query,
                                                          idx_t worker_slots) {
	auto logical_plan = connection.ExtractPlan(query);
	if (!logical_plan) {
		throw InternalException("distributed test failed to extract logical plan");
	}
	PhysicalPlanGenerator generator(*connection.context);
	auto generated_plan = generator.Plan(std::move(logical_plan));
	if (!generated_plan || generated_plan->Root().type != PhysicalOperatorType::TABLE_SCAN) {
		throw InternalException("distributed test did not produce a table scan physical plan");
	}
	auto coordinator_plan = distributed::DuckPhysicalPlanRef(generated_plan.release());
	auto &coordinator_scan = coordinator_plan->Root().Cast<PhysicalTableScan>();
	auto coordinator_unit_count = coordinator_scan.bind_data->Cast<DistributedTestScanBindData>().units.size();
	distributed::DuckDBExecutionConfig config;
	config.set_distributed_worker_slots(worker_slots);
	PlannedDistributedTestScan result;
	auto create_calls_before_plan = distributed_test_create_worker_bind_calls.load();
	auto deserialize_calls_before_plan = distributed_test_deserialize_calls.load();
	result.worker_plan = distributed::MakeTableScanPlan(coordinator_scan);
	REQUIRE(distributed_test_create_worker_bind_calls.load() == create_calls_before_plan + 1);
	REQUIRE(distributed_test_deserialize_calls.load() == deserialize_calls_before_plan);
	REQUIRE(coordinator_scan.bind_data->Cast<DistributedTestScanBindData>().units.size() == coordinator_unit_count);
	result.splits = distributed::MakeTableScanSplits(coordinator_scan, config, db.instance);
	return result;
}

static distributed::DuckPhysicalPlanRef CloneAndApply(Connection &worker,
                                                      const distributed::DuckPhysicalPlanRef &worker_plan,
                                                      const distributed::ScanSplitBatch &batch, idx_t scan_node_id) {
	auto bind_calls_before_clone = distributed_test_bind_calls.load();
	auto cloned =
	    distributed::ClonePhysicalPlanOrThrow(worker_plan, "distributed_test_extension", worker.context.get());
	REQUIRE(distributed_test_bind_calls.load() == bind_calls_before_clone);
	auto &worker_scan = cloned->Root().Cast<PhysicalTableScan>();
	REQUIRE(worker_scan.bind_data->Cast<DistributedTestScanBindData>().units.empty());
	worker_scan.extra_info.scan_node_id = optional_idx(scan_node_id);
	unordered_map<idx_t, distributed::ScanSplitBatch> assigned_batches;
	assigned_batches.emplace(scan_node_id, batch);
	string apply_error;
	REQUIRE(distributed::ApplyScanSplitBatchesToPlan(*cloned, assigned_batches, &apply_error));
	return cloned;
}

} // namespace

TEST_CASE("Distributed synthetic extension transports file splits with an explicit worker bind",
          "[distributed][extension][extension-scan]") {
	REQUIRE_THROWS_WITH(distributed::ScanSplitBatch::DeserializeFromBytes(""),
	                    Catch::Matchers::Contains("empty scan split batch"));
	REQUIRE_THROWS_WITH(distributed::ScanSplitBatch::DeserializeFromBase64(""),
	                    Catch::Matchers::Contains("empty base64 scan split batch"));

	auto invalid_function = DistributedTestScanFunction();
	auto invalid_callbacks = invalid_function.GetDistributedScanCallbacks();
	invalid_callbacks.protocol_version = 0;
	REQUIRE_THROWS_WITH(invalid_function.SetDistributedScanCallbacks(std::move(invalid_callbacks)),
	                    Catch::Matchers::Contains("greater than zero"));
	auto incomplete_function = DistributedTestScanFunction();
	auto incomplete_callbacks = incomplete_function.GetDistributedScanCallbacks();
	incomplete_callbacks.create_worker_bind = nullptr;
	REQUIRE_THROWS_WITH(incomplete_function.SetDistributedScanCallbacks(std::move(incomplete_callbacks)),
	                    Catch::Matchers::Contains("create_worker_bind"));
	auto invalid_bind_mode_function = DistributedTestScanFunction();
	auto invalid_bind_mode_callbacks = invalid_bind_mode_function.GetDistributedScanCallbacks();
	invalid_bind_mode_callbacks.bind_data_mode = static_cast<TableFunctionDistributedBindDataMode>(255);
	REQUIRE_THROWS_WITH(invalid_bind_mode_function.SetDistributedScanCallbacks(std::move(invalid_bind_mode_callbacks)),
	                    Catch::Matchers::Contains("invalid bind-data mode"));

	DuckDB coordinator_db(nullptr);
	coordinator_db.LoadStaticExtension<DistributedTestExtension>();
	Connection coordinator(coordinator_db);

	auto native_result = coordinator.Query("SELECT * FROM distributed_test_scan(3) ORDER BY unit_id");
	REQUIRE_NO_FAIL(*native_result);
	REQUIRE(CHECK_COLUMN(native_result, 0, {0, 1, 2}));

	auto duplicate_logical_plan = coordinator.ExtractPlan("SELECT * FROM distributed_test_scan(2)");
	REQUIRE(duplicate_logical_plan != nullptr);
	PhysicalPlanGenerator duplicate_generator(*coordinator.context);
	auto duplicate_physical_plan = duplicate_generator.Plan(std::move(duplicate_logical_plan));
	REQUIRE(duplicate_physical_plan != nullptr);
	auto &duplicate_scan = duplicate_physical_plan->Root().Cast<PhysicalTableScan>();
	auto &duplicate_bind = duplicate_scan.bind_data->Cast<DistributedTestScanBindData>();
	duplicate_bind.units[1] = duplicate_bind.units[0];
	distributed::DuckDBExecutionConfig duplicate_config;
	REQUIRE_THROWS_WITH(distributed::MakeTableScanSplits(duplicate_scan, duplicate_config, coordinator_db.instance),
	                    Catch::Matchers::Contains("planned duplicate split_id 'file-0'"));

	auto &coordinator_manager = DistributedExtensionManager::Get(*coordinator_db.instance);
	REQUIRE(SyntheticExtensionHasContractIdentity(coordinator_manager,
	                                              "distributed_test{table_function:distributed_test_scan(BIGINT)@1,"
	                                              "write_operator:distributed_test_opaque_write@1}"));

	auto planned = PlanDistributedTestScan(coordinator_db, coordinator, "SELECT * FROM distributed_test_scan(1)", 3);
	REQUIRE(planned.splits.size() == 1);
	REQUIRE(distributed_test_last_target_split_count.load() == 3);
	auto &detached_bind =
	    planned.worker_plan->Root().Cast<PhysicalTableScan>().bind_data->Cast<DistributedTestScanBindData>();
	REQUIRE(detached_bind.units.empty());
	const auto &split = planned.splits[0];
	REQUIRE(split.kind == distributed::ScanSplitKind::EXTENSION);
	REQUIRE(split.file.path.empty());
	REQUIRE(split.extension_capability == DistributedTestCapability());
	REQUIRE(split.split_codec.name == DISTRIBUTED_TEST_SCAN_SPLIT_CODEC);
	REQUIRE(split.split_codec.version == DISTRIBUTED_TEST_SPLIT_CODEC_VERSION);
	REQUIRE(split.split_id == "file-0");
	REQUIRE(split.extension_payload == FileResource(0));
	REQUIRE(split.estimated_cardinality == 1);
	REQUIRE(split.estimated_bytes == 1024);

	auto roundtrip = distributed::ScanSplitBatch::DeserializeFromBytes(MakeScanSplitBatch(split).SerializeToBytes());
	REQUIRE(roundtrip.splits[0].extension_payload == FileResource(0));

	DuckDB worker_db(nullptr);
	worker_db.LoadStaticExtension<DistributedTestExtension>();
	Connection worker(worker_db);
	auto missing_assignment_plan = distributed::ClonePhysicalPlanOrThrow(
	    planned.worker_plan, "distributed_test_extension_missing_assignment", worker.context.get());
	string missing_assignment_error;
	REQUIRE_FALSE(
	    distributed::ValidateDistributedScanSplitsApplied(*missing_assignment_plan, &missing_assignment_error));
	REQUIRE(StringUtil::Contains(missing_assignment_error, "no runtime scan node identity"));
	auto &missing_assignment_scan = missing_assignment_plan->Root().Cast<PhysicalTableScan>();
	missing_assignment_scan.extra_info.scan_node_id = optional_idx(6);
	REQUIRE_FALSE(distributed::ValidateScanSplitAssignments(*missing_assignment_plan, set<idx_t> {}));
	REQUIRE(distributed::ValidateScanSplitAssignments(*missing_assignment_plan, set<idx_t> {6}));
	REQUIRE_FALSE(distributed::ValidateScanSplitAssignments(*missing_assignment_plan, set<idx_t> {6, 999}));
	missing_assignment_error.clear();
	REQUIRE_FALSE(
	    distributed::ValidateDistributedScanSplitsApplied(*missing_assignment_plan, &missing_assignment_error));
	REQUIRE(StringUtil::Contains(missing_assignment_error, "no explicit worker split assignment"));
	auto worker_plan = CloneAndApply(worker, planned.worker_plan, roundtrip, 7);
	string assigned_error;
	REQUIRE(distributed::ValidateDistributedScanSplitsApplied(*worker_plan, &assigned_error));
	auto &assigned_bind = worker_plan->Root().Cast<PhysicalTableScan>().bind_data->Cast<DistributedTestScanBindData>();
	REQUIRE(assigned_bind.units.size() == 1);
	REQUIRE(assigned_bind.units[0].unit_id == 0);
	REQUIRE(assigned_bind.units[0].resource == FileResource(0));

	auto empty_planned =
	    PlanDistributedTestScan(coordinator_db, coordinator, "SELECT * FROM distributed_test_scan(0)", 3);
	REQUIRE(empty_planned.splits.size() == 1);
	REQUIRE(empty_planned.splits[0].kind == distributed::ScanSplitKind::EXTENSION);
	REQUIRE(empty_planned.splits[0].empty);
	REQUIRE(empty_planned.splits[0].split_id == "empty");
	auto empty_roundtrip = distributed::ScanSplitBatch::DeserializeFromBytes(
	    MakeScanSplitBatch(empty_planned.splits[0]).SerializeToBytes());
	auto empty_worker_plan = CloneAndApply(worker, empty_planned.worker_plan, empty_roundtrip, 8);
	string empty_assignment_error;
	REQUIRE(distributed::ValidateDistributedScanSplitsApplied(*empty_worker_plan, &empty_assignment_error));
	auto &empty_worker_scan = empty_worker_plan->Root().Cast<PhysicalTableScan>();
	REQUIRE(empty_worker_scan.distributed_scan_empty);
	auto &empty_bind = empty_worker_scan.bind_data->Cast<DistributedTestScanBindData>();
	REQUIRE(empty_bind.units.empty());

	auto missing_fte_batch_plan = distributed::ClonePhysicalPlanOrThrow(
	    empty_planned.worker_plan, "distributed_test_extension_missing_fte_batch", worker.context.get());
	auto &missing_fte_batch_scan = missing_fte_batch_plan->Root().Cast<PhysicalTableScan>();
	missing_fte_batch_scan.extra_info.scan_node_id = optional_idx(9);
	auto empty_queue = std::make_shared<distributed::FteSplitQueue>();
	empty_queue->NoMoreSplits();
	unordered_map<idx_t, std::shared_ptr<distributed::FteSplitQueue>> missing_fte_batch_queues;
	missing_fte_batch_queues.emplace(9, std::move(empty_queue));
	string missing_fte_batch_error;
	REQUIRE_FALSE(distributed::ApplyFteScanSourceQueuesToPlan(*missing_fte_batch_plan, missing_fte_batch_queues,
	                                                          &missing_fte_batch_error));
	REQUIRE(StringUtil::Contains(missing_fte_batch_error, "without an explicit split batch"));
	REQUIRE_FALSE(missing_fte_batch_scan.distributed_scan_splits_applied);

	auto empty_fte_plan = distributed::ClonePhysicalPlanOrThrow(
	    empty_planned.worker_plan, "distributed_test_extension_empty_fte_batch", worker.context.get());
	auto &empty_fte_scan = empty_fte_plan->Root().Cast<PhysicalTableScan>();
	empty_fte_scan.extra_info.scan_node_id = optional_idx(10);
	auto empty_batch_queue = std::make_shared<distributed::FteSplitQueue>();
	empty_batch_queue->AddSplit(distributed::TaskInput::make_scan_split_batch(empty_roundtrip.SerializeToBytes()));
	empty_batch_queue->NoMoreSplits();
	unordered_map<idx_t, std::shared_ptr<distributed::FteSplitQueue>> empty_fte_queues;
	empty_fte_queues.emplace(10, std::move(empty_batch_queue));
	string empty_fte_error;
	REQUIRE(distributed::ApplyFteScanSourceQueuesToPlan(*empty_fte_plan, empty_fte_queues, &empty_fte_error));
	REQUIRE(empty_fte_scan.distributed_scan_splits_applied);
	REQUIRE(empty_fte_scan.distributed_scan_empty);
	REQUIRE(empty_fte_scan.bind_data->Cast<DistributedTestScanBindData>().units.empty());

	auto duplicate_fte_plan = make_uniq<PhysicalPlan>(Allocator::DefaultAllocator());
	auto &left_duplicate_scan = distributed::ClonePhysicalPlanRootIntoPlanOrThrow(
	                                empty_planned.worker_plan, *duplicate_fte_plan,
	                                "distributed_test_extension_duplicate_fte_left", worker.context.get())
	                                .Cast<PhysicalTableScan>();
	auto &right_duplicate_scan = distributed::ClonePhysicalPlanRootIntoPlanOrThrow(
	                                 empty_planned.worker_plan, *duplicate_fte_plan,
	                                 "distributed_test_extension_duplicate_fte_right", worker.context.get())
	                                 .Cast<PhysicalTableScan>();
	auto &third_duplicate_scan = distributed::ClonePhysicalPlanRootIntoPlanOrThrow(
	                                 empty_planned.worker_plan, *duplicate_fte_plan,
	                                 "distributed_test_extension_duplicate_fte_third", worker.context.get())
	                                 .Cast<PhysicalTableScan>();
	left_duplicate_scan.extra_info.scan_node_id = optional_idx(11);
	left_duplicate_scan.extra_info.scan_group_id = optional_idx(11);
	right_duplicate_scan.extra_info.scan_node_id = optional_idx(12);
	right_duplicate_scan.extra_info.scan_group_id = optional_idx(11);
	third_duplicate_scan.extra_info.scan_node_id = optional_idx(12);
	third_duplicate_scan.extra_info.scan_group_id = optional_idx(11);
	ArenaLinkedList<reference<PhysicalOperator>> duplicate_children(duplicate_fte_plan->ArenaRef());
	duplicate_children.push_back(left_duplicate_scan);
	duplicate_children.push_back(right_duplicate_scan);
	duplicate_children.push_back(third_duplicate_scan);
	auto &duplicate_union =
	    duplicate_fte_plan->Make<PhysicalUnion>(left_duplicate_scan.GetTypes(), duplicate_children, 0, false);
	duplicate_fte_plan->SetRoot(duplicate_union);
	auto left_batch = MakeScanSplitBatch(planned.splits[0]);
	auto right_split = planned.splits[0];
	right_split.split_id = "file-1";
	right_split.extension_payload = FileResource(1);
	right_split.Validate();
	auto right_batch = MakeScanSplitBatch(std::move(right_split));
	auto left_queue = std::make_shared<distributed::FteSplitQueue>();
	left_queue->AddSplit(distributed::TaskInput::make_scan_split_batch(left_batch.SerializeToBytes()));
	left_queue->NoMoreSplits();
	auto right_queue = std::make_shared<distributed::FteSplitQueue>();
	right_queue->AddSplit(distributed::TaskInput::make_scan_split_batch(right_batch.SerializeToBytes()));
	right_queue->NoMoreSplits();
	unordered_map<idx_t, std::shared_ptr<distributed::FteSplitQueue>> duplicate_fte_queues;
	duplicate_fte_queues.emplace(11, std::move(left_queue));
	duplicate_fte_queues.emplace(12, std::move(right_queue));
	string duplicate_fte_error;
	REQUIRE(
	    distributed::ApplyFteScanSourceQueuesToPlan(*duplicate_fte_plan, duplicate_fte_queues, &duplicate_fte_error));
	REQUIRE(left_duplicate_scan.distributed_scan_splits_applied);
	REQUIRE(right_duplicate_scan.distributed_scan_splits_applied);
	REQUIRE(third_duplicate_scan.distributed_scan_splits_applied);
	REQUIRE(left_duplicate_scan.extra_info.scan_node_id != right_duplicate_scan.extra_info.scan_node_id);
	REQUIRE(left_duplicate_scan.extra_info.scan_node_id != third_duplicate_scan.extra_info.scan_node_id);
	REQUIRE(right_duplicate_scan.extra_info.scan_node_id != third_duplicate_scan.extra_info.scan_node_id);
	REQUIRE(left_duplicate_scan.bind_data->Cast<DistributedTestScanBindData>().units.size() == 1);
	REQUIRE(right_duplicate_scan.bind_data->Cast<DistributedTestScanBindData>().units.size() == 1);
	REQUIRE(third_duplicate_scan.bind_data->Cast<DistributedTestScanBindData>().units.size() == 1);
	REQUIRE(left_duplicate_scan.bind_data->Cast<DistributedTestScanBindData>().units[0].resource == FileResource(0));
	REQUIRE(right_duplicate_scan.bind_data->Cast<DistributedTestScanBindData>().units[0].resource == FileResource(1));
	REQUIRE(third_duplicate_scan.bind_data->Cast<DistributedTestScanBindData>().units[0].resource == FileResource(1));
	REQUIRE(distributed::ValidateDistributedScanSplitsApplied(*duplicate_fte_plan));

	auto no_serde_plan = planned.worker_plan;
	no_serde_plan->Root().Cast<PhysicalTableScan>().function.serialize = nullptr;
	REQUIRE_THROWS_WITH(distributed::ClonePhysicalPlanOrThrow(no_serde_plan, "distributed_test_missing_scan_serde",
	                                                          worker.context.get()),
	                    Catch::Matchers::Contains("worker rebind is not supported"));
}

TEST_CASE("Distributed extension scan constructs its worker bind without FunctionData::Copy",
          "[distributed][extension][extension-scan]") {
	DuckDB coordinator_db(nullptr);
	coordinator_db.LoadStaticExtension<DistributedTestExtension>();
	Connection coordinator(coordinator_db);
	auto logical_plan = coordinator.ExtractPlan("SELECT * FROM distributed_test_scan(1)");
	REQUIRE(logical_plan != nullptr);
	PhysicalPlanGenerator generator(*coordinator.context);
	auto generated_plan = generator.Plan(std::move(logical_plan));
	REQUIRE(generated_plan != nullptr);
	REQUIRE(generated_plan->Root().type == PhysicalOperatorType::TABLE_SCAN);
	auto &coordinator_scan = generated_plan->Root().Cast<PhysicalTableScan>();
	auto &coordinator_bind = coordinator_scan.bind_data->Cast<DistributedTestScanBindData>();
	REQUIRE(coordinator_bind.units.size() == 1);
	REQUIRE_THROWS_WITH(coordinator_bind.Copy(), Catch::Matchers::Contains("Copy not supported"));

	auto worker_plan = distributed::MakeTableScanPlan(coordinator_scan);
	auto &worker_bind = worker_plan->Root().Cast<PhysicalTableScan>().bind_data->Cast<DistributedTestScanBindData>();
	REQUIRE(worker_bind.requested_unit_count == coordinator_bind.requested_unit_count);
	REQUIRE(worker_bind.units.empty());
	REQUIRE(coordinator_bind.units.size() == 1);
}

TEST_CASE("Distributed synthetic extension mixes file splits with opaque fat fragment payloads",
          "[distributed][extension][extension-scan]") {
	DuckDB coordinator_db(nullptr);
	coordinator_db.LoadStaticExtension<DistributedTestExtension>();
	Connection coordinator(coordinator_db);

	auto native_result = coordinator.Query("SELECT * FROM distributed_test_scan(3) ORDER BY unit_id");
	REQUIRE_NO_FAIL(*native_result);
	REQUIRE(CHECK_COLUMN(native_result, 0, {0, 1, 2}));

	auto planned = PlanDistributedTestScan(coordinator_db, coordinator, "SELECT * FROM distributed_test_scan(3)", 3);
	REQUIRE(planned.splits.size() == 3);
	for (idx_t split_index = 0; split_index < planned.splits.size(); split_index++) {
		const auto &split = planned.splits[split_index];
		REQUIRE(split.kind == distributed::ScanSplitKind::EXTENSION);
		REQUIRE(split.file.path.empty());
		REQUIRE(split.split_codec.name == DISTRIBUTED_TEST_SCAN_SPLIT_CODEC);
		REQUIRE(split.split_codec.version == DISTRIBUTED_TEST_SPLIT_CODEC_VERSION);
		if (split_index % 2 == 0) {
			REQUIRE(split.split_id == "file-" + std::to_string(split_index));
			REQUIRE(split.extension_payload == FileResource(split_index));
		} else {
			REQUIRE(split.split_id == "fragment-" + std::to_string(split_index));
			auto fragment = DecodeFragmentSplitPayload(split.extension_payload);
			REQUIRE(fragment.first == FragmentPayload(split_index));
			REQUIRE(fragment.first[0] == '\0');
			REQUIRE(fragment.second == FragmentArtifact(split_index));
			REQUIRE(fragment.second[1] == '\0');
		}
	}

	auto merged = MakeScanSplitBatch(planned.splits[0]);
	merged.Merge(MakeScanSplitBatch(planned.splits[1]));
	REQUIRE(merged.splits.size() == 2);
	auto roundtrip = distributed::ScanSplitBatch::DeserializeFromBytes(merged.SerializeToBytes());
	REQUIRE(roundtrip.splits.size() == 2);
	REQUIRE(DecodeFragmentSplitPayload(roundtrip.splits[1].extension_payload).second == FragmentArtifact(1));

	DuckDB worker_db(nullptr);
	worker_db.LoadStaticExtension<DistributedTestExtension>();
	Connection worker(worker_db);
	auto worker_plan = CloneAndApply(worker, planned.worker_plan, roundtrip, 11);
	auto &assigned_bind = worker_plan->Root().Cast<PhysicalTableScan>().bind_data->Cast<DistributedTestScanBindData>();
	REQUIRE(assigned_bind.units.size() == 2);
	REQUIRE(assigned_bind.units[0].unit_id == 0);
	REQUIRE(assigned_bind.units[0].resource == FileResource(0));
	REQUIRE(assigned_bind.units[0].artifact.empty());
	REQUIRE(assigned_bind.units[1].unit_id == 1);
	REQUIRE(assigned_bind.units[1].artifact == FragmentArtifact(1));

	auto mismatched = MakeScanSplitBatch(planned.splits[2]);
	mismatched.splits[0].split_codec = {"distributed-test.invalid-split", 1};
	REQUIRE_THROWS_WITH(roundtrip.Merge(std::move(mismatched)),
	                    Catch::Matchers::Contains("different protocol identities"));
	auto duplicate_merge_target = MakeScanSplitBatch(planned.splits[0]);
	auto duplicate_merge_source = MakeScanSplitBatch(planned.splits[0]);
	REQUIRE_THROWS_WITH(duplicate_merge_target.Merge(std::move(duplicate_merge_source)),
	                    Catch::Matchers::Contains("duplicate split_id"));
	REQUIRE(duplicate_merge_target.splits.size() == 1);
	REQUIRE(duplicate_merge_target.splits[0].split_id == "file-0");

	auto mismatched_worker_plan = distributed::ClonePhysicalPlanOrThrow(
	    planned.worker_plan, "distributed_test_extension_codec_mismatch", worker.context.get());
	auto &mismatched_worker_scan = mismatched_worker_plan->Root().Cast<PhysicalTableScan>();
	mismatched_worker_scan.extra_info.scan_node_id = optional_idx(12);
	auto mismatched_assignment = MakeScanSplitBatch(planned.splits[1]);
	mismatched_assignment.splits[0].split_codec = {"distributed-test.invalid-split", 1};
	unordered_map<idx_t, distributed::ScanSplitBatch> mismatched_batches;
	mismatched_batches.emplace(12, std::move(mismatched_assignment));
	string mismatch_error;
	REQUIRE_FALSE(
	    distributed::ApplyScanSplitBatchesToPlan(*mismatched_worker_plan, mismatched_batches, &mismatch_error));
	REQUIRE(StringUtil::Contains(mismatch_error, "split codec mismatch"));

	auto unknown_node_plan = distributed::ClonePhysicalPlanOrThrow(
	    planned.worker_plan, "distributed_test_extension_unknown_node", worker.context.get());
	auto &unknown_node_scan = unknown_node_plan->Root().Cast<PhysicalTableScan>();
	unknown_node_scan.extra_info.scan_node_id = optional_idx(13);
	unordered_map<idx_t, distributed::ScanSplitBatch> unknown_node_batches;
	unknown_node_batches.emplace(13, MakeScanSplitBatch(planned.splits[0]));
	unknown_node_batches.emplace(999, MakeScanSplitBatch(planned.splits[1]));
	string unknown_node_error;
	REQUIRE_FALSE(
	    distributed::ApplyScanSplitBatchesToPlan(*unknown_node_plan, unknown_node_batches, &unknown_node_error));
	REQUIRE(StringUtil::Contains(unknown_node_error, "node_id=999 is not present in the worker plan"));
	REQUIRE_FALSE(unknown_node_scan.distributed_scan_splits_applied);
	REQUIRE(unknown_node_scan.bind_data->Cast<DistributedTestScanBindData>().units.empty());

	auto invalid_opaque_plan = distributed::ClonePhysicalPlanOrThrow(
	    planned.worker_plan, "distributed_test_extension_invalid_opaque_split", worker.context.get());
	auto &invalid_opaque_scan = invalid_opaque_plan->Root().Cast<PhysicalTableScan>();
	invalid_opaque_scan.extra_info.scan_node_id = optional_idx(14);
	auto invalid_opaque_batch = MakeScanSplitBatch(planned.splits[0]);
	invalid_opaque_batch.Merge(MakeScanSplitBatch(planned.splits[1]));
	invalid_opaque_batch.splits[1].split_id += "junk";
	unordered_map<idx_t, distributed::ScanSplitBatch> invalid_opaque_batches;
	invalid_opaque_batches.emplace(14, std::move(invalid_opaque_batch));
	REQUIRE_THROWS_WITH(distributed::ApplyScanSplitBatchesToPlan(*invalid_opaque_plan, invalid_opaque_batches),
	                    Catch::Matchers::Contains("invalid numeric identity"));
	REQUIRE_FALSE(invalid_opaque_scan.distributed_scan_splits_applied);
	REQUIRE(invalid_opaque_scan.bind_data->Cast<DistributedTestScanBindData>().units.empty());

	auto aliased_split_plan = distributed::ClonePhysicalPlanOrThrow(
	    planned.worker_plan, "distributed_test_extension_aliased_split", worker.context.get());
	auto &aliased_split_scan = aliased_split_plan->Root().Cast<PhysicalTableScan>();
	aliased_split_scan.extra_info.scan_node_id = optional_idx(15);
	auto aliased_split_batch = MakeScanSplitBatch(planned.splits[1]);
	aliased_split_batch.splits[0].split_id = "fragment-01";
	unordered_map<idx_t, distributed::ScanSplitBatch> aliased_batches;
	aliased_batches.emplace(15, std::move(aliased_split_batch));
	REQUIRE_THROWS_WITH(distributed::ApplyScanSplitBatchesToPlan(*aliased_split_plan, aliased_batches),
	                    Catch::Matchers::Contains("non-canonical numeric identity"));
	REQUIRE_FALSE(aliased_split_scan.distributed_scan_splits_applied);
	REQUIRE(aliased_split_scan.bind_data->Cast<DistributedTestScanBindData>().units.empty());
}

TEST_CASE("Distributed extension file writes use the fixed artifact adapter",
          "[distributed][extension][extension-write]") {
	const string query_id = "distributed-test-file-write";

	DistributedExtensionWriteInfo info;
	info.capability.extension_name = "distributed_test";
	info.capability.capability = {DistributedExtensionCapabilityKind::WRITE_OPERATOR, "distributed_test_file_write", 1};
	info.mode = DistributedWriteMode::FILE_ARTIFACT;
	info.fragment_codec = {distributed::DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC,
	                       distributed::DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC_VERSION};
	auto invalid_info = info;
	invalid_info.capability.capability.kind = DistributedExtensionCapabilityKind::TABLE_FUNCTION;
	invalid_info.capability.capability.function_signature =
	    GetDistributedTableFunctionSignature("distributed_test_file_write", {});
	REQUIRE_THROWS_WITH(invalid_info.Validate(), Catch::Matchers::Contains("must be a write operator"));

	distributed::DistributedCopyFileInfo file;
	file.staging_path = "synthetic-stage://part-0.parquet";
	file.final_path = "synthetic-final://part-0.parquet";
	file.row_count = 3;
	file.file_size_bytes = 128;
	file.footer_size_bytes = Value::UBIGINT(16);
	file.column_statistics = Value("binary-safe-statistics");
	file.partition_keys = Value("partition=0");

	REQUIRE_THROWS_WITH(distributed::EncodeDistributedFileWriteResults(info, "", {file}),
	                    Catch::Matchers::Contains("non-empty query identity"));
	auto encoded = distributed::EncodeDistributedFileWriteResults(info, query_id, {file});
	REQUIRE(encoded.size() == 1);
	REQUIRE(encoded[0].query_id == query_id);
	REQUIRE_THROWS_WITH(distributed::EncodeDistributedFileWriteResults(info, query_id, {file, file}),
	                    Catch::Matchers::Contains("selected file"));
	REQUIRE(encoded[0].RowCount() == 3);
	REQUIRE(encoded[0].ByteCount() == 128);
	REQUIRE(encoded[0].fragments.size() == 1);
	REQUIRE(encoded[0].fragments[0].artifacts.size() == 1);
	REQUIRE(encoded[0].fragments[0].artifacts[0].uri == file.final_path);

	auto roundtrip = DistributedWriteTaskResult::DeserializeFromBytes(encoded[0].SerializeToBytes());
	auto invalid_result = roundtrip;
	invalid_result.capability.capability.kind = DistributedExtensionCapabilityKind::TABLE_FUNCTION;
	invalid_result.capability.capability.function_signature =
	    GetDistributedTableFunctionSignature("distributed_test_file_write", {});
	REQUIRE_THROWS_WITH(invalid_result.Validate(), Catch::Matchers::Contains("not a write operator"));
	auto decoded = distributed::DecodeDistributedFileWriteResults(info, {roundtrip});
	REQUIRE(decoded.size() == 1);
	REQUIRE(decoded[0].staging_path == file.staging_path);
	REQUIRE(decoded[0].final_path == file.final_path);
	REQUIRE(decoded[0].row_count == file.row_count);
	REQUIRE(decoded[0].file_size_bytes == file.file_size_bytes);
	REQUIRE(decoded[0].footer_size_bytes == file.footer_size_bytes);
	REQUIRE(decoded[0].column_statistics == file.column_statistics);
	REQUIRE(decoded[0].partition_keys == file.partition_keys);

	roundtrip.fragments[0].artifacts[0].codec = {"distributed-test.invalid-file", 1};
	REQUIRE_THROWS_WITH(distributed::DecodeDistributedFileWriteResults(info, {roundtrip}),
	                    Catch::Matchers::Contains("invalid file artifact"));
	roundtrip = DistributedWriteTaskResult::DeserializeFromBytes(encoded[0].SerializeToBytes());
	roundtrip.fragments[0].artifacts[0].payload = "unexpected";
	REQUIRE_THROWS_WITH(distributed::DecodeDistributedFileWriteResults(info, {roundtrip}),
	                    Catch::Matchers::Contains("invalid file artifact"));
	roundtrip = DistributedWriteTaskResult::DeserializeFromBytes(encoded[0].SerializeToBytes());
	auto different_query = roundtrip;
	different_query.query_id = "different-query";
	REQUIRE_THROWS_WITH(distributed::DecodeDistributedFileWriteResults(info, {roundtrip, different_query}),
	                    Catch::Matchers::Contains("multiple query identities"));
	REQUIRE(distributed::DecodeDistributedFileWriteResults(info, {}).empty());
}

TEST_CASE("Distributed synthetic extension produces opaque write fragments through registered callbacks",
          "[distributed][extension][extension-write]") {
	DuckDB worker_db(nullptr);
	worker_db.LoadStaticExtension<DistributedTestExtension>();
	Connection worker(worker_db);
	auto info = ResolveDistributedTestWriteInfo(*worker.context);
	auto write_operator = DistributedExtensionManager::Get(*worker_db.instance).GetWriteOperator(info.capability);
	const auto &callbacks = write_operator->callbacks;
	callbacks.Validate(info.capability.CanonicalIdentity());
	REQUIRE(write_operator->fragment_codec.name == DISTRIBUTED_TEST_WRITE_CODEC);
	REQUIRE(write_operator->fragment_codec.version == DISTRIBUTED_TEST_WRITE_CODEC_VERSION);

	DistributedWriteTaskContext missing_task_context;
	REQUIRE_THROWS_WITH(missing_task_context.Validate(), Catch::Matchers::Contains("query identity"));
	missing_task_context.query_id = "query-1";
	REQUIRE_THROWS_WITH(missing_task_context.Validate(), Catch::Matchers::Contains("task-attempt identity"));
	const DistributedWriteTaskContext task_context {"query-1", "fragment-2.partition-3.attempt-4"};
	REQUIRE_NOTHROW(task_context.Validate());
	auto global_state = callbacks.initialize_global(*worker.context, info, task_context);
	ThreadContext thread_context(*worker.context);
	ExecutionContext execution_context(*worker.context, thread_context, nullptr);
	auto local_state = callbacks.initialize_local(execution_context, info, task_context, *global_state);

	DataChunk input;
	input.Initialize(Allocator::DefaultAllocator(), {LogicalType::UBIGINT});
	input.SetValue(0, 0, Value::UBIGINT(4));
	input.SetValue(0, 1, Value::UBIGINT(5));
	input.SetValue(0, 2, Value::UBIGINT(6));
	input.SetCardinality(3);
	callbacks.sink(execution_context, info, task_context, *global_state, *local_state, input);
	callbacks.combine(execution_context, info, task_context, *global_state, *local_state);
	auto fragments = callbacks.finalize(*worker.context, info, task_context, *global_state);
	REQUIRE(fragments.size() == 1);
	REQUIRE(fragments[0].row_count == 3);
	REQUIRE(fragments[0].byte_count == 3 * sizeof(uint64_t));
	REQUIRE(fragments[0].payload[0] == '\0');
	REQUIRE(fragments[0].payload.substr(1) == "sum=15");
	REQUIRE(fragments[0].artifacts.size() == 1);
	REQUIRE(fragments[0].artifacts[0].payload[0] == '\0');

	DistributedWriteTaskResult task_result;
	task_result.capability = info.capability;
	task_result.fragment_codec = info.fragment_codec;
	task_result.query_id = task_context.query_id;
	task_result.task_attempt_id = task_context.task_attempt_id;
	task_result.fragments = std::move(fragments);
	auto envelope = task_result.SerializeToBytes();
	auto roundtrip = DistributedWriteTaskResult::DeserializeFromBytes(envelope);
	REQUIRE(roundtrip.query_id == task_context.query_id);
	REQUIRE(roundtrip.task_attempt_id == task_context.task_attempt_id);
	REQUIRE(roundtrip.fragments[0].payload[0] == '\0');
	REQUIRE(roundtrip.fragments[0].artifacts[0].payload[0] == '\0');

	DataChunk envelope_chunk;
	envelope_chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::BLOB});
	envelope_chunk.SetValue(0, 0, Value::BLOB(reinterpret_cast<const_data_ptr_t>(envelope.data()), envelope.size()));
	envelope_chunk.SetCardinality(1);
	auto collection =
	    std::make_shared<ColumnDataCollection>(Allocator::DefaultAllocator(), vector<LogicalType> {LogicalType::BLOB});
	collection->Append(envelope_chunk);
	vector<distributed::ResultPartitionRef> partitions;
	partitions.push_back(std::make_shared<distributed::ColumnDataResultPartition>(collection));
	auto parsed = distributed::ParseDistributedWriteTaskResults(info, task_context.query_id, partitions);
	REQUIRE(parsed.size() == 1);
	REQUIRE(parsed[0].RowCount() == 3);
	REQUIRE(parsed[0].ByteCount() == 3 * sizeof(uint64_t));

	DataChunk extra_envelope_chunk;
	extra_envelope_chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::BLOB});
	extra_envelope_chunk.SetValue(0, 0,
	                              Value::BLOB(reinterpret_cast<const_data_ptr_t>(envelope.data()), envelope.size()));
	extra_envelope_chunk.SetValue(0, 1,
	                              Value::BLOB(reinterpret_cast<const_data_ptr_t>(envelope.data()), envelope.size()));
	extra_envelope_chunk.SetCardinality(2);
	auto extra_envelope_collection =
	    std::make_shared<ColumnDataCollection>(Allocator::DefaultAllocator(), vector<LogicalType> {LogicalType::BLOB});
	extra_envelope_collection->Append(extra_envelope_chunk);
	vector<distributed::ResultPartitionRef> extra_envelope_partitions;
	extra_envelope_partitions.push_back(
	    std::make_shared<distributed::ColumnDataResultPartition>(extra_envelope_collection));
	REQUIRE_THROWS_WITH(
	    distributed::ParseDistributedWriteTaskResults(info, task_context.query_id, extra_envelope_partitions),
	    Catch::Matchers::Contains("exactly one task envelope"));

	partitions.push_back(partitions[0]);
	REQUIRE_THROWS_WITH(distributed::ParseDistributedWriteTaskResults(info, task_context.query_id, partitions),
	                    Catch::Matchers::Contains("more than once"));
	partitions.pop_back();
	auto duplicate_fragment = roundtrip;
	duplicate_fragment.task_attempt_id += "-different";
	auto duplicate_fragment_envelope = duplicate_fragment.SerializeToBytes();
	DataChunk duplicate_fragment_chunk;
	duplicate_fragment_chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::BLOB});
	duplicate_fragment_chunk.SetValue(
	    0, 0,
	    Value::BLOB(reinterpret_cast<const_data_ptr_t>(duplicate_fragment_envelope.data()),
	                duplicate_fragment_envelope.size()));
	duplicate_fragment_chunk.SetCardinality(1);
	auto duplicate_fragment_collection =
	    std::make_shared<ColumnDataCollection>(Allocator::DefaultAllocator(), vector<LogicalType> {LogicalType::BLOB});
	duplicate_fragment_collection->Append(duplicate_fragment_chunk);
	partitions.push_back(std::make_shared<distributed::ColumnDataResultPartition>(duplicate_fragment_collection));
	REQUIRE_THROWS_WITH(distributed::ParseDistributedWriteTaskResults(info, task_context.query_id, partitions),
	                    Catch::Matchers::Contains("selected fragment"));
	partitions.pop_back();
	REQUIRE_THROWS_WITH(distributed::ParseDistributedWriteTaskResults(info, "different-query", partitions),
	                    Catch::Matchers::Contains("mismatched task result protocol"));

	auto plan = std::make_shared<PhysicalPlan>(Allocator::DefaultAllocator());
	vector<LogicalType> input_types {LogicalType::UBIGINT};
	auto &scan = plan->Make<PhysicalDummyScan>(input_types, 1);
	auto &write = plan->Make<PhysicalDistributedExtensionWrite>(info, 1).Cast<PhysicalDistributedExtensionWrite>();
	write.children.push_back(scan);
	plan->SetRoot(write);
	auto cloned = distributed::ClonePhysicalPlanOrThrow(plan, "distributed_test_opaque_write", worker.context.get());
	auto &cloned_write = cloned->Root().Cast<PhysicalDistributedExtensionWrite>();
	REQUIRE(cloned_write.task_context.query_id.empty());
	REQUIRE(cloned_write.task_context.task_attempt_id.empty());
	REQUIRE(cloned_write.info.worker_bind_data == DistributedTestWriteBindData());
	REQUIRE(ValidateDistributedWriteTaskContextAssignment(*cloned, task_context) == 1);
	REQUIRE(cloned_write.task_context.query_id.empty());
	REQUIRE(cloned_write.task_context.task_attempt_id.empty());
	REQUIRE(ApplyDistributedWriteTaskContext(*cloned, task_context) == 1);
	REQUIRE(cloned_write.task_context.query_id == task_context.query_id);
	REQUIRE(cloned_write.task_context.task_attempt_id == task_context.task_attempt_id);
	REQUIRE_NOTHROW(ApplyDistributedWriteTaskContext(*cloned, task_context));
	auto changed_task_context = task_context;
	changed_task_context.task_attempt_id += "-different";
	REQUIRE_THROWS_WITH(ValidateDistributedWriteTaskContextAssignment(*cloned, changed_task_context),
	                    Catch::Matchers::Contains("cannot change"));
	REQUIRE_THROWS_WITH(ApplyDistributedWriteTaskContext(*cloned, changed_task_context),
	                    Catch::Matchers::Contains("cannot change"));
	changed_task_context = task_context;
	changed_task_context.query_id += "-different";
	REQUIRE_THROWS_WITH(ApplyDistributedWriteTaskContext(*cloned, changed_task_context),
	                    Catch::Matchers::Contains("cannot change"));
	auto ambiguous_plan = std::make_shared<PhysicalPlan>(Allocator::DefaultAllocator());
	vector<LogicalType> ambiguous_input_types {LogicalType::UBIGINT};
	auto &ambiguous_scan = ambiguous_plan->Make<PhysicalDummyScan>(ambiguous_input_types, 1);
	auto &inner_write =
	    ambiguous_plan->Make<PhysicalDistributedExtensionWrite>(info, 1).Cast<PhysicalDistributedExtensionWrite>();
	inner_write.children.push_back(ambiguous_scan);
	auto &outer_write =
	    ambiguous_plan->Make<PhysicalDistributedExtensionWrite>(info, 1).Cast<PhysicalDistributedExtensionWrite>();
	outer_write.children.push_back(inner_write);
	ambiguous_plan->SetRoot(outer_write);
	REQUIRE_THROWS_WITH(ValidateDistributedWriteTaskContextAssignment(*ambiguous_plan, task_context),
	                    Catch::Matchers::Contains("more than one distributed extension write operator"));
	REQUIRE_THROWS_WITH(ApplyDistributedWriteTaskContext(*ambiguous_plan, task_context),
	                    Catch::Matchers::Contains("more than one distributed extension write operator"));
	REQUIRE(inner_write.task_context.query_id.empty());
	REQUIRE(outer_write.task_context.query_id.empty());

	auto recloned =
	    distributed::ClonePhysicalPlanOrThrow(cloned, "distributed_test_opaque_write_reclone", worker.context.get());
	REQUIRE(recloned->Root().Cast<PhysicalDistributedExtensionWrite>().task_context.query_id.empty());

	MemoryStream injected_identity_stream(Allocator::DefaultAllocator());
	BinarySerializer injected_identity_serializer(injected_identity_stream);
	injected_identity_serializer.Begin();
	injected_identity_serializer.WriteProperty(100, "type", PhysicalOperatorType::DISTRIBUTED_EXTENSION_WRITE);
	injected_identity_serializer.WriteProperty(101, "types", vector<LogicalType> {LogicalType::BLOB});
	injected_identity_serializer.WriteProperty<idx_t>(102, "estimated_cardinality", 1);
	injected_identity_serializer.WriteObject(103, "distributed_write_info",
	                                         [&](Serializer &object) { info.Serialize(object); });
	injected_identity_serializer.WriteProperty(104, "query_id", task_context.query_id);
	injected_identity_serializer.WriteProperty(105, "task_attempt_id", task_context.task_attempt_id);
	injected_identity_serializer.WriteList(198, "children", 0, [](Serializer::List &, idx_t) {});
	injected_identity_serializer.End();
	injected_identity_stream.Rewind();
	BinaryDeserializer injected_identity_deserializer(injected_identity_stream);
	injected_identity_deserializer.Begin();
	PhysicalPlan injected_identity_plan(Allocator::DefaultAllocator());
	REQUIRE_THROWS_WITH(PhysicalOperator::Deserialize(injected_identity_deserializer, injected_identity_plan),
	                    Catch::Matchers::Contains("must not transport a runtime task context"));

	auto execution_plan = make_uniq<PhysicalPlan>(Allocator::DefaultAllocator());
	vector<LogicalType> execution_types {LogicalType::UBIGINT};
	auto execution_collection = make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), execution_types);
	DataChunk values;
	values.Initialize(Allocator::DefaultAllocator(), execution_types);
	values.SetValue(0, 0, Value::UBIGINT(7));
	values.SetValue(0, 1, Value::UBIGINT(8));
	values.SetCardinality(2);
	execution_collection->Append(values);
	auto &execution_scan = execution_plan->Make<PhysicalColumnDataScan>(
	    execution_types, PhysicalOperatorType::COLUMN_DATA_SCAN, 2, std::move(execution_collection));
	auto &execution_write =
	    execution_plan->Make<PhysicalDistributedExtensionWrite>(info, 2).Cast<PhysicalDistributedExtensionWrite>();
	execution_write.children.push_back(execution_scan);
	execution_plan->SetRoot(execution_write);
	const DistributedWriteTaskContext execution_task_context {"query-2", "fragment-1.partition-0.attempt-0"};
	REQUIRE(ApplyDistributedWriteTaskContext(*execution_plan, execution_task_context) == 1);

	auto prepared = make_shared_ptr<PreparedStatementData>(StatementType::SELECT_STATEMENT);
	prepared->names = {"write_result"};
	prepared->types = {LogicalType::BLOB};
	prepared->properties.return_type = StatementReturnType::QUERY_RESULT;
	prepared->output_type = QueryResultOutputType::FORCE_MATERIALIZED;
	prepared->memory_type = QueryResultMemoryType::IN_MEMORY;
	prepared->physical_plan = std::move(execution_plan);
	PendingQueryParameters parameters;
	auto pending = worker.context->PendingQueryPreparedStatementNoRebind("test:distributed_test_opaque_write", prepared,
	                                                                     parameters);
	REQUIRE(pending != nullptr);
	REQUIRE_FALSE(pending->HasError());
	auto query_result = pending->Execute();
	REQUIRE(query_result != nullptr);
	REQUIRE_NO_FAIL(*query_result);
	auto *materialized = dynamic_cast<MaterializedQueryResult *>(query_result.get());
	REQUIRE(materialized != nullptr);
	REQUIRE(materialized->RowCount() == 1);
	auto executed = DistributedWriteTaskResult::DeserializeFromBytes(StringValue::Get(materialized->GetValue(0, 0)));
	REQUIRE(executed.query_id == execution_task_context.query_id);
	REQUIRE(executed.task_attempt_id == execution_task_context.task_attempt_id);
	REQUIRE(executed.RowCount() == 2);
	REQUIRE(executed.ByteCount() == 2 * sizeof(uint64_t));
	REQUIRE(executed.fragments[0].payload.substr(1) == "sum=15");
}

TEST_CASE("Distributed synthetic extension composes opaque scan and write callbacks",
          "[distributed][extension][extension-scan][extension-write]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<DistributedTestExtension>();
	Connection connection(db);
	auto planned = PlanDistributedTestScan(db, connection, "SELECT * FROM distributed_test_scan(3)", 1);
	REQUIRE(planned.splits.size() == 3);
	REQUIRE(distributed_test_last_target_split_count.load() == 1);

	auto execution_plan = make_uniq<PhysicalPlan>(Allocator::DefaultAllocator());
	auto &scan = distributed::ClonePhysicalPlanRootIntoPlanOrThrow(
	                 planned.worker_plan, *execution_plan, "distributed_test_scan_write", connection.context.get())
	                 .Cast<PhysicalTableScan>();
	scan.extra_info.scan_node_id = optional_idx(21);
	scan.extra_info.scan_group_id = optional_idx(21);
	auto info = ResolveDistributedTestWriteInfo(*connection.context);
	auto &write =
	    execution_plan->Make<PhysicalDistributedExtensionWrite>(info, 3).Cast<PhysicalDistributedExtensionWrite>();
	write.children.push_back(scan);
	execution_plan->SetRoot(write);

	set<idx_t> assigned_scan_node_ids {21};
	REQUIRE(distributed::ValidateScanSplitAssignments(*execution_plan, assigned_scan_node_ids));
	unordered_map<idx_t, distributed::ScanSplitBatch> assigned_batches;
	assigned_batches.emplace(21, MakeScanSplitBatch(planned.splits));
	string apply_error;
	REQUIRE(distributed::ApplyScanSplitBatchesToPlan(*execution_plan, assigned_batches, &apply_error));
	REQUIRE(apply_error.empty());
	REQUIRE(distributed::ValidateDistributedScanSplitsApplied(*execution_plan));

	const DistributedWriteTaskContext task_context {"query-scan-write", "fragment-0.partition-0.attempt-0"};
	REQUIRE(ValidateDistributedWriteTaskContextAssignment(*execution_plan, task_context) == 1);
	REQUIRE(ApplyDistributedWriteTaskContext(*execution_plan, task_context) == 1);

	auto prepared = make_shared_ptr<PreparedStatementData>(StatementType::SELECT_STATEMENT);
	prepared->names = {"write_result"};
	prepared->types = {LogicalType::BLOB};
	prepared->properties.return_type = StatementReturnType::QUERY_RESULT;
	prepared->output_type = QueryResultOutputType::FORCE_MATERIALIZED;
	prepared->memory_type = QueryResultMemoryType::IN_MEMORY;
	prepared->physical_plan = std::move(execution_plan);
	PendingQueryParameters parameters;
	auto pending = connection.context->PendingQueryPreparedStatementNoRebind("test:distributed_test_scan_write",
	                                                                         prepared, parameters);
	REQUIRE(pending != nullptr);
	REQUIRE_FALSE(pending->HasError());
	auto query_result = pending->Execute();
	REQUIRE(query_result != nullptr);
	REQUIRE_NO_FAIL(*query_result);
	auto *materialized = dynamic_cast<MaterializedQueryResult *>(query_result.get());
	REQUIRE(materialized != nullptr);
	REQUIRE(materialized->RowCount() == 1);
	auto result = DistributedWriteTaskResult::DeserializeFromBytes(StringValue::Get(materialized->GetValue(0, 0)));
	REQUIRE(result.query_id == task_context.query_id);
	REQUIRE(result.task_attempt_id == task_context.task_attempt_id);
	REQUIRE(result.RowCount() == 3);
	REQUIRE(result.ByteCount() == 3 * sizeof(uint64_t));
	REQUIRE(result.fragments[0].payload.substr(1) == "sum=3");
}
