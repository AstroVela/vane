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

#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/execution/distributed/exchange/exchange_manager.hpp"
#include "duckdb/execution/distributed/scheduling/task.hpp"
#include "duckdb/execution/distributed/plan/exchange_source_task.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/execution/distributed/pipeline_node/translator.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/join_output_types.hpp"
#include "duckdb/execution/distributed/pipeline_node/scan_source.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/join/physical_asof_join.hpp"
#include "duckdb/execution/operator/join/physical_blockwise_nl_join.hpp"
#include "duckdb/execution/operator/join/physical_cross_product.hpp"
#include "duckdb/execution/operator/join/physical_positional_join.hpp"
#include "duckdb/execution/distributed/pipeline_node/shuffles/repartition.hpp"
#include "duckdb/execution/operator/join/physical_iejoin.hpp"
#include "duckdb/execution/operator/join/join_filter_pushdown.hpp"
#include "duckdb/execution/operator/join/physical_piecewise_merge_join.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/parallel/task_executor.hpp"
#include "duckdb/planner/operator/logical_any_join.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/joinside.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"

#define private public
#include "duckdb/execution/distributed/pipeline_node/join/asof_join.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/cross_product.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/hash_join.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/broadcast_join.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/nested_loop_join.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/positional_join.hpp"
#undef private

#include <algorithm>
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

static DuckPhysicalPlanRef MakeEmptyScanPlanWithRoot(const vector<LogicalType> &types) {
	Allocator &alloc = Allocator::DefaultAllocator();
	auto plan = std::make_shared<PhysicalPlan>(alloc);
	auto collection = make_uniq<ColumnDataCollection>(alloc, types);
	auto &scan =
	    plan->Make<PhysicalColumnDataScan>(types, PhysicalOperatorType::COLUMN_DATA_SCAN, 0, std::move(collection));
	plan->SetRoot(scan);
	return plan;
}

static JoinCondition MakeJoinCondition(ExpressionType comparison, idx_t left_index = 0, idx_t right_index = 0) {
	JoinCondition condition;
	condition.left = make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, left_index);
	condition.right = make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, right_index);
	condition.comparison = comparison;
	return condition;
}

static vector<JoinCondition> MakeNonEqualityJoinConditions() {
	vector<JoinCondition> conditions;
	conditions.push_back(MakeJoinCondition(ExpressionType::COMPARE_GREATERTHAN));
	return conditions;
}

static PhysicalHashJoin::JoinProjectionColumns MakeProjectionColumns(const vector<LogicalType> &types) {
	PhysicalHashJoin::JoinProjectionColumns result;
	result.col_idxs.reserve(types.size());
	result.col_types.reserve(types.size());
	for (idx_t index = 0; index < types.size(); index++) {
		result.col_idxs.push_back(index);
		result.col_types.push_back(types[index]);
	}
	return result;
}

static WorkerTask MakeWorkerTaskWithInput(NodeID node_id, const std::string &node_name, SourceNodeId source_node_id,
                                          const std::string &input_bytes) {
	WorkerTask task(TaskContext::from_node_context(1, node_id, static_cast<TaskID>(node_id)), MakeScanPlanWithRoot(),
	                DuckDBExecutionConfigRef(), PipelineNodeContext(1, "join-query", node_id, node_name).to_hashmap());
	task.mutable_inputs()[source_node_id] = TaskInput::make_scan_split_batch(input_bytes);
	return task;
}

static WorkerTask MakeWorkerTaskWithTypesAndInput(NodeID node_id, const std::string &node_name,
                                                  SourceNodeId source_node_id, const std::string &input_bytes,
                                                  const vector<LogicalType> &types) {
	WorkerTask task(TaskContext::from_node_context(1, node_id, static_cast<TaskID>(node_id)),
	                MakeEmptyScanPlanWithRoot(types), DuckDBExecutionConfigRef(),
	                PipelineNodeContext(1, "join-query", node_id, node_name).to_hashmap());
	task.mutable_inputs()[source_node_id] = TaskInput::make_scan_split_batch(input_bytes);
	return task;
}

static WorkerTask MakeWorkerTaskWithoutRoot(NodeID node_id, const std::string &node_name) {
	Allocator &alloc = Allocator::DefaultAllocator();
	auto plan = std::make_shared<PhysicalPlan>(alloc);
	return WorkerTask(TaskContext::from_node_context(1, node_id, static_cast<TaskID>(node_id)), plan,
	                  DuckDBExecutionConfigRef(),
	                  PipelineNodeContext(1, "join-query", node_id, node_name).to_hashmap());
}

class StaticTaskNode : public PipelineNodeImpl {
public:
	StaticTaskNode(NodeID node_id, SourceNodeId source_node_id, const std::string &node_name,
	               const std::vector<std::string> &input_bytes)
	    : config_(MakeSchemaRef(std::vector<LogicalType> {LogicalType::BIGINT}),
	              std::make_shared<DuckDBExecutionConfig>(), ClusteringSpec::unknown_with_num_partitions(1)),
	      context_(1, "join-query", node_id, node_name) {
		for (const auto &input : input_bytes) {
			tasks_.push_back(MakeWorkerTaskWithInput(node_id, node_name, source_node_id, input));
		}
	}

	const PipelineNodeContext &context() const override {
		return context_;
	}

	const PipelineNodeConfig &config() const override {
		return config_;
	}

	std::vector<PipelineNodeRef> children() const override {
		return {};
	}

	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext & /*plan_context*/) override {
		struct VectorTaskStream {
			explicit VectorTaskStream(std::vector<WorkerTask> tasks_p) : tasks(std::move(tasks_p)) {
			}

			std::pair<bool, SubmittableTask<WorkerTask>> poll_next() {
				if (index >= tasks.size()) {
					return std::make_pair(false, SubmittableTask<WorkerTask>());
				}
				return std::make_pair(true, SubmittableTask<WorkerTask>(std::move(tasks[index++])));
			}

			std::pair<bool, SubmittableTask<WorkerTask>> try_poll_next() {
				return poll_next();
			}

			bool is_exhausted() const {
				return index >= tasks.size();
			}

			std::vector<WorkerTask> tasks;
			idx_t index = 0;
		};

		auto stream = VectorTaskStream(std::move(tasks_));
		return SubmittableTaskStream<WorkerTask>(boxed<SubmittableTask<WorkerTask>>(std::move(stream)));
	}

	std::vector<std::string> multiline_display(bool /*verbose*/) const override {
		return {context_.node_name()};
	}

private:
	PipelineNodeConfig config_;
	PipelineNodeContext context_;
	std::vector<WorkerTask> tasks_;
};

static std::shared_ptr<HashJoinNode> MakeHashJoinNode(const std::vector<std::string> &left_inputs,
                                                      const std::vector<std::string> &right_inputs) {
	PlanConfig plan_cfg(1, "join-query", std::make_shared<DuckDBExecutionConfig>());
	auto left_impl = std::make_shared<StaticTaskNode>(10, 10, "left", left_inputs);
	auto right_impl = std::make_shared<StaticTaskNode>(20, 20, "right", right_inputs);
	auto left = std::make_shared<DistributedPipelineNode>(left_impl);
	auto right = std::make_shared<DistributedPipelineNode>(right_impl);
	auto schema = MakeSchemaRef(std::vector<LogicalType> {LogicalType::BIGINT, LogicalType::BIGINT});

	return std::make_shared<HashJoinNode>(
	    300, plan_cfg, vector<JoinCondition> {}, JoinType::INNER,
	    vector<LogicalType> {LogicalType::BIGINT, LogicalType::BIGINT}, vector<LogicalType> {}, vector<LogicalType> {},
	    PhysicalHashJoin::JoinProjectionColumns(), PhysicalHashJoin::JoinProjectionColumns(),
	    PhysicalHashJoin::JoinProjectionColumns(), vector<unique_ptr<BaseStatistics>> {}, nullptr, 1, std::move(left),
	    std::move(right), std::move(schema));
}

static std::vector<std::string> MakeInputNames(const std::string &prefix, idx_t count) {
	std::vector<std::string> result;
	for (idx_t index = 0; index < count; index++) {
		result.push_back(prefix + std::to_string(index));
	}
	return result;
}

static void RequireHashJoinFanInPreservesInputs(idx_t left_count, idx_t right_count) {
	auto left_inputs = MakeInputNames("left-", left_count);
	auto right_inputs = MakeInputNames("right-", right_count);
	auto node = MakeHashJoinNode(left_inputs, right_inputs);

	DuckDB db(nullptr);
	Connection connection(db);
	auto task_executor = std::make_shared<PlanTaskExecutor>(connection.context);
	PlanExecutionContext plan_context(task_executor, connection.context);
	auto stream = node->produce_tasks(plan_context);

	std::vector<std::string> actual_left_inputs;
	std::vector<std::string> actual_right_inputs;
	std::string fragment_node_id;
	while (true) {
		auto next = stream.poll_next();
		if (!next.first) {
			break;
		}
		const auto &context = next.second.task()->context();
		auto node_id_entry = context.find("node_id");
		REQUIRE(node_id_entry != context.end());
		REQUIRE_FALSE(node_id_entry->second.empty());
		if (fragment_node_id.empty()) {
			fragment_node_id = node_id_entry->second;
		} else {
			REQUIRE(node_id_entry->second == fragment_node_id);
		}

		const auto &inputs = next.second.task()->inputs();
		REQUIRE_FALSE(inputs.empty());
		REQUIRE(inputs.size() <= 2);
		auto left_entry = inputs.find(10);
		if (left_entry != inputs.end()) {
			actual_left_inputs.push_back(left_entry->second.scan_split_batch_bytes);
		}
		auto right_entry = inputs.find(20);
		if (right_entry != inputs.end()) {
			actual_right_inputs.push_back(right_entry->second.scan_split_batch_bytes);
		}
	}
	REQUIRE(actual_left_inputs == left_inputs);
	REQUIRE(actual_right_inputs == right_inputs);
}

//===----------------------------------------------------------------------===//
// TaskInput type tests
//===----------------------------------------------------------------------===//

TEST_CASE("TaskInput: make_scan_split_batch factory", "[distributed][source_id]") {
	auto input = TaskInput::make_scan_split_batch("dGVzdA=="); // base64 for "test"

	REQUIRE(input.kind == TaskInput::Kind::ScanSplitBatch);
	REQUIRE(input.scan_split_batch_bytes == "dGVzdA==");
}

TEST_CASE("TaskInputs: map insertion and lookup", "[distributed][source_id]") {
	TaskInputs inputs;

	inputs[1] = TaskInput::make_scan_split_batch("scan_data_1");
	inputs[5] = TaskInput::make_scan_split_batch("scan_data_5");

	REQUIRE(inputs.size() == 2);
	REQUIRE(inputs.count(1) == 1);
	REQUIRE(inputs.count(5) == 1);
	REQUIRE(inputs.count(99) == 0);

	REQUIRE(inputs[1].kind == TaskInput::Kind::ScanSplitBatch);
	REQUIRE(inputs[1].scan_split_batch_bytes == "scan_data_1");
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
	REQUIRE(res.is_ok());
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
	REQUIRE(res.is_ok());
	REQUIRE(res.value() != nullptr);

	auto inner = res.value()->inner();
	auto scan_node = std::dynamic_pointer_cast<ScanSourceNode>(inner);
	REQUIRE(scan_node != nullptr);

	// Should have SOME node_id (auto-assigned)
	REQUIRE(scan_node->node_id() >= 0);
}

TEST_CASE("PhysicalPlanTranslator: translates cross product with two inputs", "[distributed][cross_product]") {
	Allocator allocator;
	auto plan = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> input_types = {LogicalType::BIGINT};
	vector<LogicalType> output_types = {LogicalType::BIGINT, LogicalType::BIGINT};

	auto left_collection = MakeCollection(input_types, 2);
	auto &left = plan->Make<PhysicalColumnDataScan>(input_types, PhysicalOperatorType::COLUMN_DATA_SCAN, 2,
	                                                std::move(left_collection));
	auto right_collection = MakeCollection(input_types, 3);
	auto &right = plan->Make<PhysicalColumnDataScan>(input_types, PhysicalOperatorType::COLUMN_DATA_SCAN, 3,
	                                                 std::move(right_collection));
	auto &cross = plan->Make<PhysicalCrossProduct>(output_types, left, right, 6);
	plan->SetRoot(cross);

	auto result = physical_plan_to_pipeline_node(PlanConfig {}, plan);
	REQUIRE(result.is_ok());
	REQUIRE(result.value() != nullptr);
	auto cross_node = std::dynamic_pointer_cast<CrossProductNode>(result.value()->inner());
	REQUIRE(cross_node != nullptr);
	REQUIRE(cross_node->children().size() == 2);
	REQUIRE(cross_node->config().clustering_spec()->num_partitions() == 1);
}

TEST_CASE("PhysicalPlanTranslator: translates positional join through ordered gathers",
          "[distributed][positional_join]") {
	Allocator allocator;
	auto plan = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> input_types = {LogicalType::BIGINT};
	vector<LogicalType> output_types = {LogicalType::BIGINT, LogicalType::BIGINT};

	auto &left = plan->Make<PhysicalColumnDataScan>(input_types, PhysicalOperatorType::COLUMN_DATA_SCAN, 2,
	                                                MakeCollection(input_types, 2));
	auto &right = plan->Make<PhysicalColumnDataScan>(input_types, PhysicalOperatorType::COLUMN_DATA_SCAN, 3,
	                                                 MakeCollection(input_types, 3));
	auto &join = plan->Make<PhysicalPositionalJoin>(output_types, left, right, 3);
	plan->SetRoot(join);

	auto result = physical_plan_to_pipeline_node(PlanConfig {}, plan);
	REQUIRE(result.is_ok());
	REQUIRE(result.value() != nullptr);
	auto positional_node = std::dynamic_pointer_cast<PositionalJoinNode>(result.value()->inner());
	REQUIRE(positional_node != nullptr);
	REQUIRE(positional_node->children().size() == 2);
	REQUIRE(positional_node->config().clustering_spec()->num_partitions() == 1);
	for (const auto &child : positional_node->children()) {
		auto ordered_gather = std::dynamic_pointer_cast<RepartitionNode>(child);
		REQUIRE(ordered_gather != nullptr);
		REQUIRE(ordered_gather->PreservesOrder());
		REQUIRE(ordered_gather->is_materialization_barrier());
		auto materialized_inputs = ordered_gather->materialized_input_node_ids();
		REQUIRE(materialized_inputs.size() == 1);
		REQUIRE(materialized_inputs[0] == ordered_gather->children()[0]->node_id());
	}
}

TEST_CASE("PhysicalPlanTranslator: translates ASOF joins to a gathered native worker plan",
          "[distributed][join][asof_join]") {
	Allocator allocator;
	auto plan = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> left_types = {LogicalType::BIGINT, LogicalType::BIGINT};
	vector<LogicalType> right_types = {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT};
	vector<LogicalType> output_types = {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT};

	auto &left = plan->Make<PhysicalColumnDataScan>(left_types, PhysicalOperatorType::COLUMN_DATA_SCAN, 2,
	                                                MakeCollection(left_types, 2));
	auto &right = plan->Make<PhysicalColumnDataScan>(right_types, PhysicalOperatorType::COLUMN_DATA_SCAN, 3,
	                                                 MakeCollection(right_types, 3));
	LogicalComparisonJoin logical_join(JoinType::LEFT);
	logical_join.types = output_types;
	logical_join.estimated_cardinality = 6;
	logical_join.right_projection_map = {2};
	logical_join.conditions.push_back(MakeJoinCondition(ExpressionType::COMPARE_EQUAL, 0, 0));
	logical_join.conditions.push_back(MakeJoinCondition(ExpressionType::COMPARE_GREATERTHANOREQUALTO, 1, 1));
	auto &join = plan->Make<PhysicalAsOfJoin>(logical_join, left, right);
	plan->SetRoot(join);

	auto result = physical_plan_to_pipeline_node(PlanConfig {}, plan);
	REQUIRE(result.is_ok());
	REQUIRE(result.value() != nullptr);
	auto join_node = std::dynamic_pointer_cast<AsOfJoinNode>(result.value()->inner());
	REQUIRE(join_node != nullptr);
	REQUIRE(join_node->children().size() == 2);
	REQUIRE(join_node->config().clustering_spec()->num_partitions() == 1);
	REQUIRE(join_node->conditions_.size() == 2);
	REQUIRE(join_node->right_projection_map_ == vector<column_t> {2});

	auto left_task =
	    SubmittableTask<WorkerTask>(MakeWorkerTaskWithTypesAndInput(10, "left", 10, "left_scan", left_types));
	auto right_task =
	    SubmittableTask<WorkerTask>(MakeWorkerTaskWithTypesAndInput(20, "right", 20, "right_scan", right_types));
	TaskIDCounter task_id_counter;
	auto join_task =
	    join_node->BuildAsOfJoinTask(std::move(left_task), std::move(right_task), task_id_counter, nullptr);

	REQUIRE(join_task.task()->inputs().size() == 2);
	REQUIRE(join_task.task()->inputs().at(10).scan_split_batch_bytes == "left_scan");
	REQUIRE(join_task.task()->inputs().at(20).scan_split_batch_bytes == "right_scan");
	REQUIRE(join_task.task()->plan()->Root().type == PhysicalOperatorType::ASOF_JOIN);
	REQUIRE(join_task.task()->plan()->Root().GetTypes() == output_types);
	auto &worker_join = join_task.task()->plan()->Root().Cast<PhysicalAsOfJoin>();
	REQUIRE(worker_join.children.size() == 2);
	REQUIRE(worker_join.conditions.size() == 2);
	REQUIRE(worker_join.right_projection_map == vector<column_t> {2});

	auto cloned = ClonePhysicalPlanOrThrow(join_task.task()->plan(), "asof_join_owned_children_test", nullptr);
	REQUIRE(cloned->Root().type == PhysicalOperatorType::ASOF_JOIN);
	REQUIRE(cloned->Root().GetTypes() == output_types);
	REQUIRE(cloned->Root().children.size() == 2);
	REQUIRE(cloned->Root().Cast<PhysicalAsOfJoin>().right_projection_map == vector<column_t> {2});
}

TEST_CASE("PhysicalPlanTranslator: normalizes range joins to a gathered nested-loop node",
          "[distributed][join][nested_loop_join]") {
	for (auto source_type : {PhysicalOperatorType::PIECEWISE_MERGE_JOIN, PhysicalOperatorType::IE_JOIN}) {
		CAPTURE(static_cast<int>(source_type));
		Allocator allocator;
		auto plan = std::make_shared<PhysicalPlan>(allocator);
		vector<LogicalType> input_types = {LogicalType::BIGINT};
		vector<LogicalType> output_types = {LogicalType::BIGINT, LogicalType::BIGINT};
		auto &left = plan->Make<PhysicalColumnDataScan>(input_types, PhysicalOperatorType::COLUMN_DATA_SCAN, 2,
		                                                MakeCollection(input_types, 2));
		auto &right = plan->Make<PhysicalColumnDataScan>(input_types, PhysicalOperatorType::COLUMN_DATA_SCAN, 3,
		                                                 MakeCollection(input_types, 3));
		LogicalComparisonJoin logical_join(JoinType::INNER);
		logical_join.types = output_types;
		vector<JoinCondition> conditions;
		conditions.push_back(MakeJoinCondition(ExpressionType::COMPARE_LESSTHAN));

		if (source_type == PhysicalOperatorType::PIECEWISE_MERGE_JOIN) {
			auto &join = plan->Make<PhysicalPiecewiseMergeJoin>(logical_join, left, right, std::move(conditions),
			                                                    JoinType::INNER, 6, nullptr);
			plan->SetRoot(join);
		} else {
			conditions.push_back(MakeJoinCondition(ExpressionType::COMPARE_GREATERTHANOREQUALTO));
			auto &join =
			    plan->Make<PhysicalIEJoin>(logical_join, left, right, std::move(conditions), JoinType::INNER, 6);
			plan->SetRoot(join);
		}

		auto result = physical_plan_to_pipeline_node(PlanConfig {}, plan);
		REQUIRE(result.is_ok());
		REQUIRE(result.value() != nullptr);
		auto join_node = std::dynamic_pointer_cast<NestedLoopJoinNode>(result.value()->inner());
		REQUIRE(join_node != nullptr);
		REQUIRE(join_node->children().size() == 2);
		REQUIRE(join_node->config().clustering_spec()->num_partitions() == 1);
		auto display = join_node->multiline_display(false);
		REQUIRE(std::find(display.begin(), display.end(), "Source operator: " + EnumUtil::ToString(source_type)) !=
		        display.end());
	}
}

TEST_CASE("PhysicalPlanTranslator: preserves range join projection maps", "[distributed][join][nested_loop_join]") {
	Allocator allocator;
	auto plan = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> left_types = {LogicalType::BIGINT, LogicalType::BIGINT};
	vector<LogicalType> right_types = {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT};
	vector<LogicalType> output_types = {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT,
	                                    LogicalType::BIGINT};
	auto &left = plan->Make<PhysicalColumnDataScan>(left_types, PhysicalOperatorType::COLUMN_DATA_SCAN, 2,
	                                                MakeCollection(left_types, 2));
	auto &right = plan->Make<PhysicalColumnDataScan>(right_types, PhysicalOperatorType::COLUMN_DATA_SCAN, 3,
	                                                 MakeCollection(right_types, 3));
	LogicalComparisonJoin logical_join(JoinType::LEFT);
	logical_join.types = output_types;
	logical_join.left_projection_map = {0, 1};
	logical_join.right_projection_map = {0, 1};
	vector<JoinCondition> conditions;
	conditions.push_back(MakeJoinCondition(ExpressionType::COMPARE_GREATERTHANOREQUALTO, 0, 0));
	conditions.push_back(MakeJoinCondition(ExpressionType::COMPARE_LESSTHAN, 0, 2));
	auto &join = plan->Make<PhysicalIEJoin>(logical_join, left, right, std::move(conditions), JoinType::LEFT, 2);
	plan->SetRoot(join);

	auto result = physical_plan_to_pipeline_node(PlanConfig {}, plan);
	REQUIRE(result.is_ok());
	auto join_node = std::dynamic_pointer_cast<NestedLoopJoinNode>(result.value()->inner());
	REQUIRE(join_node != nullptr);
	REQUIRE(join_node->left_projection_map_ == vector<idx_t> {0, 1});
	REQUIRE(join_node->right_projection_map_ == vector<idx_t> {0, 1});

	auto left_task =
	    SubmittableTask<WorkerTask>(MakeWorkerTaskWithTypesAndInput(10, "left", 10, "left_scan", left_types));
	auto right_task =
	    SubmittableTask<WorkerTask>(MakeWorkerTaskWithTypesAndInput(20, "right", 20, "right_scan", right_types));
	TaskIDCounter task_id_counter;
	auto join_task =
	    join_node->BuildNestedLoopJoinTask(std::move(left_task), std::move(right_task), task_id_counter, nullptr);

	REQUIRE(join_task.task()->plan()->Root().type == PhysicalOperatorType::PROJECTION);
	auto &project = join_task.task()->plan()->Root().Cast<PhysicalProjection>();
	REQUIRE(project.GetTypes() == output_types);
	REQUIRE(project.children.size() == 1);
	REQUIRE(project.children[0].get().type == PhysicalOperatorType::NESTED_LOOP_JOIN);
	REQUIRE(project.children[0].get().GetTypes().size() == 5);
	REQUIRE(project.select_list.size() == 4);
	for (idx_t index = 0; index < project.select_list.size(); index++) {
		REQUIRE(project.select_list[index]->Cast<BoundReferenceExpression>().index == index);
	}

	auto cloned = ClonePhysicalPlanOrThrow(join_task.task()->plan(), "range_join_projection_test", nullptr);
	REQUIRE(cloned->Root().type == PhysicalOperatorType::PROJECTION);
	REQUIRE(cloned->Root().GetTypes() == output_types);
	REQUIRE(cloned->Root().children[0].get().type == PhysicalOperatorType::NESTED_LOOP_JOIN);
}

TEST_CASE("PhysicalPlanTranslator: translates blockwise nested-loop joins", "[distributed][join][nested_loop_join]") {
	Allocator allocator;
	auto plan = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> input_types = {LogicalType::BIGINT};
	vector<LogicalType> output_types = {LogicalType::BIGINT, LogicalType::BIGINT};
	auto &left = plan->Make<PhysicalColumnDataScan>(input_types, PhysicalOperatorType::COLUMN_DATA_SCAN, 2,
	                                                MakeCollection(input_types, 2));
	auto &right = plan->Make<PhysicalColumnDataScan>(input_types, PhysicalOperatorType::COLUMN_DATA_SCAN, 3,
	                                                 MakeCollection(input_types, 3));
	auto condition = make_uniq<BoundComparisonExpression>(ExpressionType::COMPARE_LESSTHAN,
	                                                      make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, 0),
	                                                      make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, 1));
	LogicalAnyJoin logical_join(JoinType::INNER);
	logical_join.types = output_types;
	auto &join =
	    plan->Make<PhysicalBlockwiseNLJoin>(logical_join, left, right, std::move(condition), JoinType::INNER, 6);
	plan->SetRoot(join);

	auto result = physical_plan_to_pipeline_node(PlanConfig {}, plan);
	REQUIRE(result.is_ok());
	REQUIRE(result.value() != nullptr);
	auto join_node = std::dynamic_pointer_cast<NestedLoopJoinNode>(result.value()->inner());
	REQUIRE(join_node != nullptr);
	REQUIRE(join_node->children().size() == 2);
	REQUIRE(join_node->config().clustering_spec()->num_partitions() == 1);
	auto display = join_node->multiline_display(false);
	REQUIRE(std::find(display.begin(), display.end(), "Source operator: BLOCKWISE_NL_JOIN") != display.end());
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
	task.mutable_inputs()[5] = TaskInput::make_scan_split_batch("test_base64");

	REQUIRE(task.inputs().size() == 1);
	REQUIRE(task.inputs().at(5).kind == TaskInput::Kind::ScanSplitBatch);
	REQUIRE(task.inputs().at(5).scan_split_batch_bytes == "test_base64");
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
	inputs[7] = TaskInput::make_scan_split_batch("from_ctor");

	TaskContext tctx(0, 0, 1, {});
	WorkerTask task(tctx, plan, ExecutionConfigRef(), {}, "TestTask", std::move(inputs));

	REQUIRE(task.inputs().size() == 1);
	REQUIRE(task.inputs().at(7).scan_split_batch_bytes == "from_ctor");
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
	inputs[42] = TaskInput::make_scan_split_batch("clone_me");

	TaskContext tctx(0, 0, 1, {});
	WorkerTask task(tctx, plan, ExecutionConfigRef(), {}, "TestTask", std::move(inputs));

	auto cloned = task.clone();
	REQUIRE(cloned->inputs().size() == 1);
	REQUIRE(cloned->inputs().at(42).kind == TaskInput::Kind::ScanSplitBatch);
	REQUIRE(cloned->inputs().at(42).scan_split_batch_bytes == "clone_me");
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
	REQUIRE(joined_task.task()->inputs().at(10).kind == TaskInput::Kind::ScanSplitBatch);
	REQUIRE(joined_task.task()->inputs().at(10).scan_split_batch_bytes == "left_scan");
	REQUIRE(joined_task.task()->inputs().at(20).kind == TaskInput::Kind::ScanSplitBatch);
	REQUIRE(joined_task.task()->inputs().at(20).scan_split_batch_bytes == "right_scan");

	// BuildHashJoinTask has returned and the temporary right-side plan has been
	// destroyed. Re-cloning forces a full tree serialization and verifies that
	// the joined plan still owns both child trees.
	auto joined_plan_clone =
	    ClonePhysicalPlanOrThrow(joined_task.task()->plan(), "hash_join_owned_children_test", nullptr);
	REQUIRE(joined_plan_clone->HasRoot());
	REQUIRE(joined_plan_clone->Root().children.size() == 2);
}

TEST_CASE("CrossProductNode: replacement task owns both inputs", "[distributed][source_id][cross_product]") {
	PlanConfig plan_cfg(1, "cross-product-query", std::make_shared<DuckDBExecutionConfig>());
	vector<LogicalType> output_types = {LogicalType::BIGINT, LogicalType::BIGINT};
	auto schema = MakeSchemaRef(output_types);
	CrossProductNode node(302, plan_cfg, output_types, 1, nullptr, nullptr, schema);

	auto left_task = SubmittableTask<WorkerTask>(MakeWorkerTaskWithInput(10, "left", 10, "left_scan"));
	auto right_task = SubmittableTask<WorkerTask>(MakeWorkerTaskWithInput(20, "right", 20, "right_scan"));
	TaskIDCounter task_id_counter;
	auto cross_task = node.BuildCrossProductTask(std::move(left_task), std::move(right_task), task_id_counter, nullptr);

	REQUIRE(cross_task.task()->inputs().size() == 2);
	REQUIRE(cross_task.task()->inputs().at(10).scan_split_batch_bytes == "left_scan");
	REQUIRE(cross_task.task()->inputs().at(20).scan_split_batch_bytes == "right_scan");
	REQUIRE(cross_task.task()->plan()->Root().type == PhysicalOperatorType::CROSS_PRODUCT);
	REQUIRE(cross_task.task()->plan()->Root().children.size() == 2);
	auto &cross_root = cross_task.task()->plan()->Root();
	auto &left_scan = cross_root.children[0].get().Cast<PhysicalColumnDataScan>();
	auto &right_scan = cross_root.children[1].get().Cast<PhysicalColumnDataScan>();
	REQUIRE(left_scan.collection);
	REQUIRE(right_scan.collection);
	REQUIRE(left_scan.collection->Count() == 1);
	REQUIRE(right_scan.collection->Count() == 1);

	auto cloned = ClonePhysicalPlanOrThrow(cross_task.task()->plan(), "cross_product_owned_children_test", nullptr);
	REQUIRE(cloned->HasRoot());
	REQUIRE(cloned->Root().type == PhysicalOperatorType::CROSS_PRODUCT);
	REQUIRE(cloned->Root().children.size() == 2);
	REQUIRE(cloned->Root().GetTypes() == output_types);
	REQUIRE(cloned->Root().children[0].get().Cast<PhysicalColumnDataScan>().collection->Count() == 1);
	REQUIRE(cloned->Root().children[1].get().Cast<PhysicalColumnDataScan>().collection->Count() == 1);
}

TEST_CASE("PositionalJoinNode: replacement task owns both ordered inputs",
          "[distributed][source_id][positional_join]") {
	PlanConfig plan_cfg(1, "positional-join-query", std::make_shared<DuckDBExecutionConfig>());
	vector<LogicalType> output_types = {LogicalType::BIGINT, LogicalType::BIGINT};
	auto schema = MakeSchemaRef(output_types);
	PositionalJoinNode node(304, plan_cfg, output_types, 3, nullptr, nullptr, schema);

	auto left_task = SubmittableTask<WorkerTask>(MakeWorkerTaskWithInput(10, "left", 10, "left_scan"));
	auto right_task = SubmittableTask<WorkerTask>(MakeWorkerTaskWithInput(20, "right", 20, "right_scan"));
	TaskIDCounter task_id_counter;
	auto join_task =
	    node.BuildPositionalJoinTask(std::move(left_task), std::move(right_task), task_id_counter, nullptr);

	REQUIRE(join_task.task()->inputs().size() == 2);
	REQUIRE(join_task.task()->inputs().at(10).scan_split_batch_bytes == "left_scan");
	REQUIRE(join_task.task()->inputs().at(20).scan_split_batch_bytes == "right_scan");
	REQUIRE(join_task.task()->plan()->Root().type == PhysicalOperatorType::POSITIONAL_JOIN);
	REQUIRE(join_task.task()->plan()->Root().children.size() == 2);
	REQUIRE(join_task.task()->plan()->Root().GetTypes() == output_types);

	auto cloned = ClonePhysicalPlanOrThrow(join_task.task()->plan(), "positional_join_owned_children_test", nullptr);
	REQUIRE(cloned->HasRoot());
	REQUIRE(cloned->Root().type == PhysicalOperatorType::POSITIONAL_JOIN);
	REQUIRE(cloned->Root().children.size() == 2);
	REQUIRE(cloned->Root().GetTypes() == output_types);
}

TEST_CASE("NestedLoopJoinNode: replacement tasks own both inputs", "[distributed][source_id][nested_loop_join]") {
	PlanConfig plan_cfg(1, "nested-loop-query", std::make_shared<DuckDBExecutionConfig>());
	vector<LogicalType> output_types = {LogicalType::BIGINT, LogicalType::BIGINT};
	auto schema = MakeSchemaRef(output_types);

	SECTION("comparison join") {
		NestedLoopJoinNode node(303, plan_cfg, PhysicalOperatorType::PIECEWISE_MERGE_JOIN,
		                        MakeNonEqualityJoinConditions(), nullptr, JoinType::INNER, output_types, 1, nullptr,
		                        nullptr, schema);
		auto left_task = SubmittableTask<WorkerTask>(MakeWorkerTaskWithInput(10, "left", 10, "left_scan"));
		auto right_task = SubmittableTask<WorkerTask>(MakeWorkerTaskWithInput(20, "right", 20, "right_scan"));
		TaskIDCounter task_id_counter;
		auto join_task =
		    node.BuildNestedLoopJoinTask(std::move(left_task), std::move(right_task), task_id_counter, nullptr);

		REQUIRE(join_task.task()->inputs().size() == 2);
		REQUIRE(join_task.task()->inputs().at(10).scan_split_batch_bytes == "left_scan");
		REQUIRE(join_task.task()->inputs().at(20).scan_split_batch_bytes == "right_scan");
		REQUIRE(join_task.task()->plan()->Root().type == PhysicalOperatorType::NESTED_LOOP_JOIN);
		REQUIRE(join_task.task()->plan()->Root().children.size() == 2);
		auto cloned = ClonePhysicalPlanOrThrow(join_task.task()->plan(), "nested_loop_owned_children_test", nullptr);
		REQUIRE(cloned->Root().type == PhysicalOperatorType::NESTED_LOOP_JOIN);
		REQUIRE(cloned->Root().GetTypes() == output_types);
	}

	SECTION("blockwise join") {
		auto condition = make_uniq<BoundComparisonExpression>(
		    ExpressionType::COMPARE_LESSTHAN, make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, 0),
		    make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, 1));
		NestedLoopJoinNode node(304, plan_cfg, std::move(condition), JoinType::INNER, output_types, 1, nullptr, nullptr,
		                        schema);
		auto left_task = SubmittableTask<WorkerTask>(MakeWorkerTaskWithInput(10, "left", 10, "left_scan"));
		auto right_task = SubmittableTask<WorkerTask>(MakeWorkerTaskWithInput(20, "right", 20, "right_scan"));
		TaskIDCounter task_id_counter;
		auto join_task =
		    node.BuildNestedLoopJoinTask(std::move(left_task), std::move(right_task), task_id_counter, nullptr);

		REQUIRE(join_task.task()->inputs().size() == 2);
		REQUIRE(join_task.task()->plan()->Root().type == PhysicalOperatorType::BLOCKWISE_NL_JOIN);
		REQUIRE(join_task.task()->plan()->Root().children.size() == 2);
		auto cloned = ClonePhysicalPlanOrThrow(join_task.task()->plan(), "blockwise_owned_children_test", nullptr);
		REQUIRE(cloned->Root().type == PhysicalOperatorType::BLOCKWISE_NL_JOIN);
		REQUIRE(cloned->Root().GetTypes() == output_types);
	}

	SECTION("nested comparison types fall back to blockwise execution") {
		auto list_type = LogicalType::LIST(LogicalType::BIGINT);
		vector<LogicalType> list_output_types = {list_type, list_type};
		vector<JoinCondition> conditions;
		JoinCondition condition;
		condition.left = make_uniq<BoundReferenceExpression>(list_type, 0);
		condition.right = make_uniq<BoundReferenceExpression>(list_type, 0);
		condition.comparison = ExpressionType::COMPARE_LESSTHAN;
		conditions.push_back(std::move(condition));
		NestedLoopJoinNode node(305, plan_cfg, PhysicalOperatorType::PIECEWISE_MERGE_JOIN, std::move(conditions),
		                        nullptr, JoinType::INNER, list_output_types, 1, nullptr, nullptr,
		                        MakeSchemaRef(list_output_types));
		auto left_task =
		    SubmittableTask<WorkerTask>(MakeWorkerTaskWithTypesAndInput(10, "left", 10, "left_scan", {list_type}));
		auto right_task =
		    SubmittableTask<WorkerTask>(MakeWorkerTaskWithTypesAndInput(20, "right", 20, "right_scan", {list_type}));
		TaskIDCounter task_id_counter;
		auto join_task =
		    node.BuildNestedLoopJoinTask(std::move(left_task), std::move(right_task), task_id_counter, nullptr);

		REQUIRE(join_task.task()->plan()->Root().type == PhysicalOperatorType::BLOCKWISE_NL_JOIN);
		auto &join = join_task.task()->plan()->Root().Cast<PhysicalBlockwiseNLJoin>();
		auto &comparison = join.condition->Cast<BoundComparisonExpression>();
		REQUIRE(comparison.left->Cast<BoundReferenceExpression>().index == 0);
		REQUIRE(comparison.right->Cast<BoundReferenceExpression>().index == 1);
		auto cloned = ClonePhysicalPlanOrThrow(join_task.task()->plan(), "nested_type_blockwise_test", nullptr);
		REQUIRE(cloned->Root().type == PhysicalOperatorType::BLOCKWISE_NL_JOIN);
		REQUIRE(cloned->Root().GetTypes() == list_output_types);
	}
}

TEST_CASE("Join output types follow join semantics", "[distributed][join]") {
	const vector<LogicalType> left_types = {LogicalType::INTEGER, LogicalType::VARCHAR};
	const vector<LogicalType> right_types = {LogicalType::BIGINT, LogicalType::DOUBLE};
	const vector<LogicalType> both_types = {LogicalType::INTEGER, LogicalType::VARCHAR, LogicalType::BIGINT,
	                                        LogicalType::DOUBLE};
	const vector<LogicalType> mark_types = {LogicalType::INTEGER, LogicalType::VARCHAR, LogicalType::BOOLEAN};

	for (auto join_type : {JoinType::INNER, JoinType::LEFT, JoinType::RIGHT, JoinType::OUTER, JoinType::SINGLE}) {
		CAPTURE(static_cast<int>(join_type));
		REQUIRE(BuildJoinOutputTypes(join_type, left_types, right_types) == both_types);
	}
	for (auto join_type : {JoinType::SEMI, JoinType::ANTI}) {
		CAPTURE(static_cast<int>(join_type));
		REQUIRE(BuildJoinOutputTypes(join_type, left_types, right_types) == left_types);
	}
	REQUIRE(BuildJoinOutputTypes(JoinType::MARK, left_types, right_types) == mark_types);
	for (auto join_type : {JoinType::RIGHT_SEMI, JoinType::RIGHT_ANTI}) {
		CAPTURE(static_cast<int>(join_type));
		REQUIRE(BuildJoinOutputTypes(join_type, left_types, right_types) == right_types);
	}
}

TEST_CASE("HashJoinNode: non-equality simple joins preserve their output schema", "[distributed][source_id][join]") {
	struct TestCase {
		JoinType join_type;
		vector<LogicalType> output_types;
	};
	const vector<TestCase> test_cases = {
	    {JoinType::SEMI, {LogicalType::BIGINT}},
	    {JoinType::ANTI, {LogicalType::BIGINT}},
	    {JoinType::MARK, {LogicalType::BIGINT, LogicalType::BOOLEAN}},
	};

	for (const auto &test_case : test_cases) {
		CAPTURE(static_cast<int>(test_case.join_type));
		PlanConfig plan_cfg(1, "join-query", std::make_shared<DuckDBExecutionConfig>());
		auto schema = MakeSchemaRef(test_case.output_types);
		auto lhs_output_columns = MakeProjectionColumns({LogicalType::BIGINT});
		HashJoinNode node(300, plan_cfg, MakeNonEqualityJoinConditions(), test_case.join_type, test_case.output_types,
		                  {}, {LogicalType::BIGINT}, PhysicalHashJoin::JoinProjectionColumns(),
		                  std::move(lhs_output_columns), PhysicalHashJoin::JoinProjectionColumns(), {}, nullptr, 1,
		                  nullptr, nullptr, std::move(schema));

		auto left_task = SubmittableTask<WorkerTask>(MakeWorkerTaskWithInput(10, "left", 10, "left_scan"));
		auto right_task = SubmittableTask<WorkerTask>(MakeWorkerTaskWithInput(20, "right", 20, "right_scan"));
		TaskIDCounter task_id_counter;
		auto joined_task =
		    node.BuildHashJoinTask(std::move(left_task), std::move(right_task), task_id_counter, nullptr);

		REQUIRE(joined_task.task()->plan()->Root().type == PhysicalOperatorType::NESTED_LOOP_JOIN);
		REQUIRE(joined_task.task()->plan()->Root().GetTypes() == test_case.output_types);
		auto cloned_plan = ClonePhysicalPlanOrThrow(joined_task.task()->plan(), "nlj_output_schema_test", nullptr);
		REQUIRE(cloned_plan->Root().GetTypes() == test_case.output_types);
	}
}

TEST_CASE("HashJoinNode: non-equality joins reject a stale output schema", "[distributed][source_id][join]") {
	PlanConfig plan_cfg(1, "join-query", std::make_shared<DuckDBExecutionConfig>());
	vector<LogicalType> stale_output_types = {LogicalType::INTEGER};
	auto schema = MakeSchemaRef(stale_output_types);
	auto lhs_output_columns = MakeProjectionColumns({LogicalType::BIGINT});
	HashJoinNode node(300, plan_cfg, MakeNonEqualityJoinConditions(), JoinType::SEMI, stale_output_types, {},
	                  {LogicalType::BIGINT}, PhysicalHashJoin::JoinProjectionColumns(), std::move(lhs_output_columns),
	                  PhysicalHashJoin::JoinProjectionColumns(), {}, nullptr, 1, nullptr, nullptr, std::move(schema));

	auto left_task = SubmittableTask<WorkerTask>(MakeWorkerTaskWithInput(10, "left", 10, "left_scan"));
	auto right_task = SubmittableTask<WorkerTask>(MakeWorkerTaskWithInput(20, "right", 20, "right_scan"));
	TaskIDCounter task_id_counter;

	REQUIRE_THROWS_WITH(node.BuildHashJoinTask(std::move(left_task), std::move(right_task), task_id_counter, nullptr),
	                    Catch::Matchers::Contains("output schema that does not match its children"));
}

TEST_CASE("HashJoinNode: MARK join embeds the global build summary from its right source",
          "[distributed][source_id][join]") {
	PlanConfig plan_cfg(1, "mark-join-query", std::make_shared<DuckDBExecutionConfig>());
	JoinCondition condition;
	condition.left = make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, 0);
	condition.right = make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, 0);
	condition.comparison = ExpressionType::COMPARE_EQUAL;
	vector<JoinCondition> conditions;
	conditions.push_back(std::move(condition));
	auto schema = MakeSchemaRef(std::vector<LogicalType> {LogicalType::BIGINT, LogicalType::BOOLEAN});

	HashJoinNode node(301, plan_cfg, std::move(conditions), JoinType::MARK,
	                  vector<LogicalType> {LogicalType::BIGINT, LogicalType::BOOLEAN}, {},
	                  vector<LogicalType> {LogicalType::BIGINT}, PhysicalHashJoin::JoinProjectionColumns(),
	                  PhysicalHashJoin::JoinProjectionColumns(), PhysicalHashJoin::JoinProjectionColumns(), {}, nullptr,
	                  1, nullptr, nullptr, schema, optional_idx(20));

	auto left_task = SubmittableTask<WorkerTask>(MakeWorkerTaskWithInput(10, "left", 10, "left_scan"));
	auto right_worker = MakeWorkerTaskWithInput(20, "right", 21, "right_scan");
	ExchangeSourceTaskDescriptor right_source;
	right_source.mark_join_build_summary = MarkJoinBuildSummary::Create(true, true);
	right_worker.mutable_inputs()[20] = TaskInput::make_exchange_source_task(right_source.SerializeToBytes());
	auto right_task = SubmittableTask<WorkerTask>(std::move(right_worker));
	TaskIDCounter task_id_counter;

	auto joined_task = node.BuildHashJoinTask(std::move(left_task), std::move(right_task), task_id_counter, nullptr);
	auto *join = dynamic_cast<PhysicalHashJoin *>(&joined_task.task()->plan()->Root());
	REQUIRE(join != nullptr);
	REQUIRE(join->mark_join_build_summary.valid);
	REQUIRE(join->mark_join_build_summary.has_rows);
	REQUIRE(join->mark_join_build_summary.has_null);
}

TEST_CASE("HashJoinNode: fan-in preserves child task streams", "[distributed][source_id][join]") {
	SECTION("streams have equal length") {
		RequireHashJoinFanInPreservesInputs(3, 3);
	}
	SECTION("left stream is longer") {
		RequireHashJoinFanInPreservesInputs(3, 1);
	}
	SECTION("right stream is longer") {
		RequireHashJoinFanInPreservesInputs(1, 3);
	}
}

TEST_CASE("HashJoinNode: asymmetric empty child stream fails loudly", "[distributed][source_id][join]") {
	SECTION("right stream is empty") {
		auto node = MakeHashJoinNode(MakeInputNames("left-", 1), {});
		DuckDB db(nullptr);
		Connection connection(db);
		auto task_executor = std::make_shared<PlanTaskExecutor>(connection.context);
		PlanExecutionContext plan_context(task_executor, connection.context);
		auto stream = node->produce_tasks(plan_context);

		REQUIRE_THROWS_WITH(stream.poll_next(), Catch::Matchers::Contains("empty right task stream"));
	}
	SECTION("left stream is empty") {
		auto node = MakeHashJoinNode({}, MakeInputNames("right-", 1));
		DuckDB db(nullptr);
		Connection connection(db);
		auto task_executor = std::make_shared<PlanTaskExecutor>(connection.context);
		PlanExecutionContext plan_context(task_executor, connection.context);
		auto stream = node->produce_tasks(plan_context);

		REQUIRE_THROWS_WITH(stream.poll_next(), Catch::Matchers::Contains("empty left task stream"));
	}
}

TEST_CASE("HashJoinNode: two empty child streams produce no task", "[distributed][source_id][join]") {
	auto node = MakeHashJoinNode({}, {});
	DuckDB db(nullptr);
	Connection connection(db);
	auto task_executor = std::make_shared<PlanTaskExecutor>(connection.context);
	PlanExecutionContext plan_context(task_executor, connection.context);
	auto stream = node->produce_tasks(plan_context);

	REQUIRE_FALSE(stream.poll_next().first);
}

TEST_CASE("BroadcastJoinNode: replacement task preserves receiver inputs", "[distributed][source_id][join]") {
	PlanConfig plan_cfg(1, "join-query", std::make_shared<DuckDBExecutionConfig>());
	vector<LogicalType> output_types = {LogicalType::BIGINT, LogicalType::BIGINT};
	auto schema = MakeSchemaRef(std::vector<LogicalType> {LogicalType::BIGINT, LogicalType::BIGINT});

	BroadcastJoinNode node(301, plan_cfg, {}, JoinType::INNER, output_types, {}, {},
	                       PhysicalHashJoin::JoinProjectionColumns(), PhysicalHashJoin::JoinProjectionColumns(),
	                       PhysicalHashJoin::JoinProjectionColumns(), {}, nullptr, 1, BroadcastJoinSide::LEFT, nullptr,
	                       nullptr, schema);

	auto receiver_task = SubmittableTask<WorkerTask>(MakeWorkerTaskWithInput(30, "receiver", 30, "receiver_scan"));
	auto broadcast_plan = MakeScanPlanWithRoot();

	auto joined_task = node.BuildBroadcastHashJoinTask(std::move(receiver_task), broadcast_plan, nullptr);

	REQUIRE(joined_task.task()->inputs().size() == 1);
	REQUIRE(joined_task.task()->inputs().at(30).kind == TaskInput::Kind::ScanSplitBatch);
	REQUIRE(joined_task.task()->inputs().at(30).scan_split_batch_bytes == "receiver_scan");
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
	                       PhysicalHashJoin::JoinProjectionColumns(), {}, nullptr, 1, BroadcastJoinSide::LEFT, nullptr,
	                       nullptr, schema);

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

TEST_CASE("BroadcastJoinNode: broadcast side must not be semantically preserved", "[distributed][source_id][join]") {
	struct SafetyCase {
		JoinType join_type;
		bool left_safe;
		bool right_safe;
	};
	const std::vector<SafetyCase> cases = {
	    {JoinType::INNER, true, true},       {JoinType::LEFT, false, true},     {JoinType::RIGHT, true, false},
	    {JoinType::OUTER, false, false},     {JoinType::SEMI, false, true},     {JoinType::ANTI, false, true},
	    {JoinType::MARK, false, true},       {JoinType::SINGLE, false, true},   {JoinType::RIGHT_SEMI, true, false},
	    {JoinType::RIGHT_ANTI, true, false}, {JoinType::INVALID, false, false},
	};

	for (const auto &entry : cases) {
		INFO("join type: " << static_cast<int>(entry.join_type));
		REQUIRE(IsBroadcastJoinSideSemanticallySafe(entry.join_type, BroadcastJoinSide::LEFT) == entry.left_safe);
		REQUIRE(IsBroadcastJoinSideSemanticallySafe(entry.join_type, BroadcastJoinSide::RIGHT) == entry.right_safe);
	}

	PlanConfig plan_cfg(1, "join-query", std::make_shared<DuckDBExecutionConfig>());
	vector<LogicalType> output_types = {LogicalType::BIGINT, LogicalType::BIGINT};
	auto schema = MakeSchemaRef(std::vector<LogicalType> {LogicalType::BIGINT, LogicalType::BIGINT});

	REQUIRE_THROWS_WITH(BroadcastJoinNode(304, plan_cfg, {}, JoinType::LEFT, output_types, {}, {},
	                                      PhysicalHashJoin::JoinProjectionColumns(),
	                                      PhysicalHashJoin::JoinProjectionColumns(),
	                                      PhysicalHashJoin::JoinProjectionColumns(), {}, nullptr, 1,
	                                      BroadcastJoinSide::LEFT, nullptr, nullptr, schema),
	                    Catch::Matchers::Contains("semantically preserved"));
}
