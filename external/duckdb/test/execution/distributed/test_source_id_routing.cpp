// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB Distributed Execution Tests
//
// test/distributed/test_source_id_routing.cpp
//
// Unit tests for SourceId-based pset routing
// Tests cover: TaskInput types, source_node_id on PhysicalColumnDataScan,
// translator preservation, and WorkerTask inputs.
//===----------------------------------------------------------------------===//

#include "catch.hpp"

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/execution/distributed/exchange/exchange_manager.hpp"
#include "duckdb/execution/distributed/scheduling/task.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/execution/distributed/pipeline_node/translator.hpp"
#include "duckdb/execution/distributed/pipeline_node/scan_source.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/join/join_filter_pushdown.hpp"
#include "duckdb/planner/joinside.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"

#define private public
#include "duckdb/execution/distributed/pipeline_node/join/hash_join.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/broadcast_join.hpp"
#undef private

#include <memory>

using namespace duckdb;
using namespace duckdb::distributed;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

static unique_ptr<ColumnDataCollection> MakeCollection(const vector<LogicalType> &types, idx_t num_rows = 1) {
	auto collection = make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), types);
	for (idx_t row = 0; row < num_rows; row++) {
		for (idx_t col = 0; col < types.size(); col++) {
			chunk.SetValue(col, row, Value::BIGINT(row * 10 + col));
		}
	}
	chunk.SetCardinality(num_rows);
	collection->Append(chunk);
	return collection;
}

static DuckPhysicalPlanRef MakeScanPlanWithRoot() {
	Allocator &alloc = Allocator::DefaultAllocator();
	auto plan = std::make_shared<PhysicalPlan>(alloc);
	vector<LogicalType> types = {LogicalType::BIGINT};
	auto collection = MakeCollection(types);
	auto &scan =
	    plan->Make<PhysicalColumnDataScan>(types, PhysicalOperatorType::COLUMN_DATA_SCAN, 0, std::move(collection));
	plan->SetRoot(scan);
	return plan;
}

static WorkerTask MakeWorkerTaskWithInput(NodeID node_id, const std::string &node_name, SourceNodeId source_node_id,
                                          const std::string &input_bytes) {
	WorkerTask task(TaskContext::from_node_context(1, node_id, static_cast<TaskID>(node_id)), MakeScanPlanWithRoot(),
	                DuckDBExecutionConfigRef(), PipelineNodeContext(1, "join-query", node_id, node_name).to_hashmap());
	task.mutable_inputs()[source_node_id] = TaskInput::make_scan_task(input_bytes);
	return task;
}

static WorkerTask MakeWorkerTaskWithoutRoot(NodeID node_id, const std::string &node_name) {
	Allocator &alloc = Allocator::DefaultAllocator();
	auto plan = std::make_shared<PhysicalPlan>(alloc);
	return WorkerTask(TaskContext::from_node_context(1, node_id, static_cast<TaskID>(node_id)), plan,
	                  DuckDBExecutionConfigRef(),
	                  PipelineNodeContext(1, "join-query", node_id, node_name).to_hashmap());
}

//===----------------------------------------------------------------------===//
// TaskInput type tests
//===----------------------------------------------------------------------===//

TEST_CASE("TaskInput: make_scan_task factory", "[distributed][source_id]") {
	auto input = TaskInput::make_scan_task("dGVzdA=="); // base64 for "test"

	REQUIRE(input.kind == TaskInput::Kind::ScanTask);
	REQUIRE(input.scan_task_bytes == "dGVzdA==");
}

TEST_CASE("TaskInputs: map insertion and lookup", "[distributed][source_id]") {
	TaskInputs inputs;

	inputs[1] = TaskInput::make_scan_task("scan_data_1");
	inputs[5] = TaskInput::make_scan_task("scan_data_5");

	REQUIRE(inputs.size() == 2);
	REQUIRE(inputs.count(1) == 1);
	REQUIRE(inputs.count(5) == 1);
	REQUIRE(inputs.count(99) == 0);

	REQUIRE(inputs[1].kind == TaskInput::Kind::ScanTask);
	REQUIRE(inputs[1].scan_task_bytes == "scan_data_1");
}

//===----------------------------------------------------------------------===//
// PhysicalColumnDataScan source_node_id tests
//===----------------------------------------------------------------------===//

TEST_CASE("PhysicalColumnDataScan: source_node_id defaults to invalid", "[distributed][source_id]") {
	Allocator &alloc = Allocator::DefaultAllocator();
	auto plan = std::make_shared<PhysicalPlan>(alloc);
	vector<LogicalType> types = {LogicalType::BIGINT};
	auto collection = MakeCollection(types);

	auto &scan_op =
	    plan->Make<PhysicalColumnDataScan>(types, PhysicalOperatorType::COLUMN_DATA_SCAN, 1, std::move(collection));
	auto &scan = scan_op.Cast<PhysicalColumnDataScan>();

	// source_node_id should be invalid by default
	REQUIRE_FALSE(scan.source_node_id.IsValid());
}

TEST_CASE("PhysicalColumnDataScan: source_node_id can be set and read", "[distributed][source_id]") {
	Allocator &alloc = Allocator::DefaultAllocator();
	auto plan = std::make_shared<PhysicalPlan>(alloc);
	vector<LogicalType> types = {LogicalType::BIGINT};
	auto collection = MakeCollection(types);

	auto &scan_op =
	    plan->Make<PhysicalColumnDataScan>(types, PhysicalOperatorType::COLUMN_DATA_SCAN, 1, std::move(collection));
	auto &scan = scan_op.Cast<PhysicalColumnDataScan>();

	scan.source_node_id = optional_idx(static_cast<idx_t>(42));

	REQUIRE(scan.source_node_id.IsValid());
	REQUIRE(scan.source_node_id.GetIndex() == 42);
}

//===----------------------------------------------------------------------===//
// Translator source_node_id preservation tests
//===----------------------------------------------------------------------===//

TEST_CASE("PhysicalPlanTranslator: preserves source_node_id on column data scan", "[distributed][source_id]") {
	Allocator allocator;
	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> types = {LogicalType::BIGINT};

	auto collection = MakeCollection(types);
	auto &scan_op =
	    plan_ptr->Make<PhysicalColumnDataScan>(types, PhysicalOperatorType::COLUMN_DATA_SCAN, 1, std::move(collection));
	auto &scan = scan_op.Cast<PhysicalColumnDataScan>();

	// Set a specific source_node_id
	scan.source_node_id = optional_idx(static_cast<idx_t>(77));
	plan_ptr->SetRoot(scan_op);

	auto res = physical_plan_to_pipeline_node(PlanConfig {}, plan_ptr);
	REQUIRE(res.ok);
	REQUIRE(res.value() != nullptr);

	auto inner = res.value()->inner();
	auto scan_node = std::dynamic_pointer_cast<ScanSourceNode>(inner);
	REQUIRE(scan_node != nullptr);

	// The ScanSourceNode should have preserved the source_node_id as its node_id
	REQUIRE(scan_node->node_id() == 77);
	REQUIRE(scan_node->scan_pset_key() == "77");
}

TEST_CASE("PhysicalPlanTranslator: assigns fresh id when source_node_id is not set", "[distributed][source_id]") {
	Allocator allocator;
	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> types = {LogicalType::BIGINT};

	auto collection = MakeCollection(types);
	auto &scan =
	    plan_ptr->Make<PhysicalColumnDataScan>(types, PhysicalOperatorType::COLUMN_DATA_SCAN, 1, std::move(collection));

	// Do NOT set source_node_id — should fall back to get_next_pipeline_node_id()
	plan_ptr->SetRoot(scan);

	auto res = physical_plan_to_pipeline_node(PlanConfig {}, plan_ptr);
	REQUIRE(res.ok);
	REQUIRE(res.value() != nullptr);

	auto inner = res.value()->inner();
	auto scan_node = std::dynamic_pointer_cast<ScanSourceNode>(inner);
	REQUIRE(scan_node != nullptr);

	// Should have SOME node_id (auto-assigned)
	REQUIRE(scan_node->node_id() >= 0);
}

//===----------------------------------------------------------------------===//
// WorkerTask inputs_ tests
//===----------------------------------------------------------------------===//

TEST_CASE("WorkerTask: inputs default to empty", "[distributed][source_id]") {
	Allocator &alloc = Allocator::DefaultAllocator();
	auto plan = std::make_shared<PhysicalPlan>(alloc);
	vector<LogicalType> types = {LogicalType::BIGINT};
	auto collection = MakeCollection(types);
	auto &scan =
	    plan->Make<PhysicalColumnDataScan>(types, PhysicalOperatorType::COLUMN_DATA_SCAN, 0, std::move(collection));
	plan->SetRoot(scan);

	TaskContext tctx(0, 0, 1, {});
	WorkerTask task(tctx, plan, ExecutionConfigRef(), {});

	REQUIRE(task.inputs().empty());
}

TEST_CASE("WorkerTask: inputs can be populated via mutable_inputs", "[distributed][source_id]") {
	Allocator &alloc = Allocator::DefaultAllocator();
	auto plan = std::make_shared<PhysicalPlan>(alloc);
	vector<LogicalType> types = {LogicalType::BIGINT};
	auto collection = MakeCollection(types);
	auto &scan =
	    plan->Make<PhysicalColumnDataScan>(types, PhysicalOperatorType::COLUMN_DATA_SCAN, 0, std::move(collection));
	plan->SetRoot(scan);

	TaskContext tctx(0, 0, 1, {});
	WorkerTask task(tctx, plan, ExecutionConfigRef(), {});

	// Populate via mutable_inputs()
	task.mutable_inputs()[5] = TaskInput::make_scan_task("test_base64");

	REQUIRE(task.inputs().size() == 1);
	REQUIRE(task.inputs().at(5).kind == TaskInput::Kind::ScanTask);
	REQUIRE(task.inputs().at(5).scan_task_bytes == "test_base64");
}

TEST_CASE("WorkerTask: inputs passed via constructor", "[distributed][source_id]") {
	Allocator &alloc = Allocator::DefaultAllocator();
	auto plan = std::make_shared<PhysicalPlan>(alloc);
	vector<LogicalType> types = {LogicalType::BIGINT};
	auto collection = MakeCollection(types);
	auto &scan =
	    plan->Make<PhysicalColumnDataScan>(types, PhysicalOperatorType::COLUMN_DATA_SCAN, 0, std::move(collection));
	plan->SetRoot(scan);

	// Create inputs before constructing task
	TaskInputs inputs;
	inputs[7] = TaskInput::make_scan_task("from_ctor");

	TaskContext tctx(0, 0, 1, {});
	WorkerTask task(tctx, plan, ExecutionConfigRef(), {}, "TestTask", std::move(inputs));

	REQUIRE(task.inputs().size() == 1);
	REQUIRE(task.inputs().at(7).scan_task_bytes == "from_ctor");
}

TEST_CASE("WorkerTask: clone preserves inputs", "[distributed][source_id]") {
	Allocator &alloc = Allocator::DefaultAllocator();
	auto plan = std::make_shared<PhysicalPlan>(alloc);
	vector<LogicalType> types = {LogicalType::BIGINT};
	auto collection = MakeCollection(types);
	auto &scan =
	    plan->Make<PhysicalColumnDataScan>(types, PhysicalOperatorType::COLUMN_DATA_SCAN, 0, std::move(collection));
	plan->SetRoot(scan);

	TaskInputs inputs;
	inputs[42] = TaskInput::make_scan_task("clone_me");

	TaskContext tctx(0, 0, 1, {});
	WorkerTask task(tctx, plan, ExecutionConfigRef(), {}, "TestTask", std::move(inputs));

	auto cloned = task.clone();
	REQUIRE(cloned->inputs().size() == 1);
	REQUIRE(cloned->inputs().at(42).kind == TaskInput::Kind::ScanTask);
	REQUIRE(cloned->inputs().at(42).scan_task_bytes == "clone_me");
}

TEST_CASE("HashJoinNode: replacement task preserves both side inputs", "[distributed][source_id][join]") {
	PlanConfig plan_cfg(1, "join-query", std::make_shared<DuckDBExecutionConfig>());
	vector<LogicalType> output_types = {LogicalType::BIGINT, LogicalType::BIGINT};
	auto schema = MakeSchemaRef(std::vector<LogicalType> {LogicalType::BIGINT, LogicalType::BIGINT});

	HashJoinNode node(300, plan_cfg, {}, JoinType::INNER, output_types, {}, {},
	                  PhysicalHashJoin::JoinProjectionColumns(), PhysicalHashJoin::JoinProjectionColumns(),
	                  PhysicalHashJoin::JoinProjectionColumns(), {}, nullptr, 1, nullptr, nullptr, schema);

	auto left_task = SubmittableTask<WorkerTask>(MakeWorkerTaskWithInput(10, "left", 10, "left_scan"));
	auto right_task = SubmittableTask<WorkerTask>(MakeWorkerTaskWithInput(20, "right", 20, "right_scan"));
	TaskIDCounter task_id_counter;

	auto joined_task = node.BuildHashJoinTask(std::move(left_task), std::move(right_task), task_id_counter, nullptr);

	REQUIRE(joined_task.task()->inputs().size() == 2);
	REQUIRE(joined_task.task()->inputs().at(10).kind == TaskInput::Kind::ScanTask);
	REQUIRE(joined_task.task()->inputs().at(10).scan_task_bytes == "left_scan");
	REQUIRE(joined_task.task()->inputs().at(20).kind == TaskInput::Kind::ScanTask);
	REQUIRE(joined_task.task()->inputs().at(20).scan_task_bytes == "right_scan");

	// BuildHashJoinTask has returned and the temporary right-side plan has been
	// destroyed. Re-cloning forces a full tree serialization and verifies that
	// the joined plan still owns both child trees.
	auto joined_plan_clone =
	    ClonePhysicalPlanOrThrow(joined_task.task()->plan(), "hash_join_owned_children_test", nullptr);
	REQUIRE(joined_plan_clone->HasRoot());
	REQUIRE(joined_plan_clone->Root().children.size() == 2);
}

TEST_CASE("BroadcastJoinNode: replacement task preserves receiver inputs", "[distributed][source_id][join]") {
	PlanConfig plan_cfg(1, "join-query", std::make_shared<DuckDBExecutionConfig>());
	vector<LogicalType> output_types = {LogicalType::BIGINT, LogicalType::BIGINT};
	auto schema = MakeSchemaRef(std::vector<LogicalType> {LogicalType::BIGINT, LogicalType::BIGINT});

	BroadcastJoinNode node(301, plan_cfg, {}, JoinType::INNER, output_types, {}, {},
	                       PhysicalHashJoin::JoinProjectionColumns(), PhysicalHashJoin::JoinProjectionColumns(),
	                       PhysicalHashJoin::JoinProjectionColumns(), {}, nullptr, 1, false, nullptr, nullptr, schema);

	auto receiver_task = SubmittableTask<WorkerTask>(MakeWorkerTaskWithInput(30, "receiver", 30, "receiver_scan"));
	auto broadcast_plan = MakeScanPlanWithRoot();

	auto joined_task = node.BuildBroadcastHashJoinTask(std::move(receiver_task), broadcast_plan, nullptr);

	REQUIRE(joined_task.task()->inputs().size() == 1);
	REQUIRE(joined_task.task()->inputs().at(30).kind == TaskInput::Kind::ScanTask);
	REQUIRE(joined_task.task()->inputs().at(30).scan_task_bytes == "receiver_scan");
}

TEST_CASE("HashJoinNode: invalid child plan throws instead of passing through", "[distributed][source_id][join]") {
	PlanConfig plan_cfg(1, "join-query", std::make_shared<DuckDBExecutionConfig>());
	vector<LogicalType> output_types = {LogicalType::BIGINT, LogicalType::BIGINT};
	auto schema = MakeSchemaRef(std::vector<LogicalType> {LogicalType::BIGINT, LogicalType::BIGINT});

	HashJoinNode node(302, plan_cfg, {}, JoinType::INNER, output_types, {}, {},
	                  PhysicalHashJoin::JoinProjectionColumns(), PhysicalHashJoin::JoinProjectionColumns(),
	                  PhysicalHashJoin::JoinProjectionColumns(), {}, nullptr, 1, nullptr, nullptr, schema);

	auto left_task = SubmittableTask<WorkerTask>(MakeWorkerTaskWithoutRoot(10, "left"));
	auto right_task = SubmittableTask<WorkerTask>(MakeWorkerTaskWithInput(20, "right", 20, "right_scan"));
	TaskIDCounter task_id_counter;

	bool saw_error = false;
	try {
		node.BuildHashJoinTask(std::move(left_task), std::move(right_task), task_id_counter, nullptr);
	} catch (const std::exception &ex) {
		saw_error = true;
		REQUIRE(std::string(ex.what()).find("HashJoinNode cannot build join task") != std::string::npos);
	}
	REQUIRE(saw_error);
}

TEST_CASE("BroadcastJoinNode: invalid receiver plan throws instead of passing through",
          "[distributed][source_id][join]") {
	PlanConfig plan_cfg(1, "join-query", std::make_shared<DuckDBExecutionConfig>());
	vector<LogicalType> output_types = {LogicalType::BIGINT, LogicalType::BIGINT};
	auto schema = MakeSchemaRef(std::vector<LogicalType> {LogicalType::BIGINT, LogicalType::BIGINT});

	BroadcastJoinNode node(303, plan_cfg, {}, JoinType::INNER, output_types, {}, {},
	                       PhysicalHashJoin::JoinProjectionColumns(), PhysicalHashJoin::JoinProjectionColumns(),
	                       PhysicalHashJoin::JoinProjectionColumns(), {}, nullptr, 1, false, nullptr, nullptr, schema);

	auto receiver_task = SubmittableTask<WorkerTask>(MakeWorkerTaskWithoutRoot(30, "receiver"));
	auto broadcast_plan = MakeScanPlanWithRoot();

	bool saw_error = false;
	try {
		node.BuildBroadcastHashJoinTask(std::move(receiver_task), broadcast_plan, nullptr);
	} catch (const std::exception &ex) {
		saw_error = true;
		REQUIRE(std::string(ex.what()).find("BroadcastJoinNode cannot build join task") != std::string::npos);
	}
	REQUIRE(saw_error);
}
