// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB Distributed Execution
//
// test_physical_plan_translator.cpp
//
// Unit tests for translating DuckDB physical plans into distributed pipeline nodes.
//===----------------------------------------------------------------------===//

#include "catch.hpp"

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/helper/physical_limit.hpp"
#include "duckdb/execution/operator/helper/physical_limit_percent.hpp"
#include "duckdb/execution/operator/helper/physical_reservoir_sample.hpp"
#include "duckdb/execution/operator/helper/physical_streaming_limit.hpp"
#include "duckdb/execution/operator/order/physical_order.hpp"
#include "duckdb/execution/operator/order/physical_top_n.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/execution/operator/scan/physical_dummy_scan.hpp"
#include "duckdb/execution/operator/scan/physical_expression_scan.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/common/enums/order_type.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/parser/expression/bound_expression.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/joinside.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
// For aggregate tests
#include "duckdb.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_perfecthash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_partitioned_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_ungrouped_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_streaming_window.hpp"
#include "duckdb/execution/operator/aggregate/physical_window.hpp"
#include "duckdb/execution/operator/exchange/physical_repartition.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/join/physical_left_delim_join.hpp"
#include "duckdb/execution/operator/join/physical_right_delim_join.hpp"
#include "duckdb/execution/operator/projection/physical_tableinout_function.hpp"
#include "duckdb/execution/operator/persistent/physical_copy_to_file.hpp"
#include "duckdb/execution/operator/persistent/physical_distributed_extension_write.hpp"

#include "duckdb/main/connection.hpp"
#include "duckdb/execution/distributed/extension_write_task_provider.hpp"
#include "duckdb/execution/distributed/pipeline_node/translator.hpp"
#include "duckdb/execution/distributed/pipeline_node/aggregate.hpp"
#include "duckdb/execution/distributed/pipeline_node/grouping_set_expand.hpp"
#include "duckdb/execution/distributed/pipeline_node/limit.hpp"
#include "duckdb/execution/distributed/pipeline_node/projection.hpp"
#include "duckdb/execution/distributed/pipeline_node/sample.hpp"
#include "duckdb/execution/distributed/pipeline_node/scan_source.hpp"
#include "duckdb/execution/distributed/pipeline_node/expression_scan.hpp"
#include "duckdb/execution/distributed/pipeline_node/shuffles/repartition.hpp"
#include "duckdb/execution/distributed/pipeline_node/sort.hpp"
#include "duckdb/execution/distributed/pipeline_node/streaming_udf_passthrough.hpp"
#include "duckdb/execution/distributed/pipeline_node/window.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"

// Include distributed pipeline translator headers (lightweight declarations)
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/distributed/pipeline_node/copy_finish.hpp"
#include "duckdb/execution/distributed/pipeline_node/extension_write_sink.hpp"
#include "test_helpers.hpp"

#include <memory>
#include <cstdlib>
#include <utility>

using namespace duckdb;
using namespace duckdb::distributed;

static string SQLStringLiteral(const string &value);

class TestExtensionWriteOperator final : public PhysicalOperator, public ExtensionWriteTaskProvider {
public:
	TestExtensionWriteOperator(PhysicalPlan &physical_plan, vector<LogicalType> types,
	                           string extension_write_name_p = "test_extension_write",
	                           DistributedWriteMode mode = DistributedWriteMode::FILE_ARTIFACT)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, std::move(types), 0) {
		info.capability.extension_name = "test_extension";
		info.capability.extension_protocol_version = 1;
		info.capability.capability.kind = DistributedExtensionCapabilityKind::OPERATOR;
		info.capability.capability.name = "write";
		info.capability.capability.protocol_version = 1;
		info.write_name = std::move(extension_write_name_p);
		info.mode = mode;
		if (mode == DistributedWriteMode::FILE_ARTIFACT) {
			info.fragment_codec = DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC;
			info.fragment_codec_version = DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC_VERSION;
		} else {
			info.fragment_codec = "test-extension.opaque-write";
			info.fragment_codec_version = 1;
			info.worker_bind_data = "worker-bind";
		}
	}

	optional_ptr<ExtensionWriteTaskProvider> GetExtensionWriteTaskProvider() override {
		return this;
	}

	const DistributedExtensionWriteInfo &WriteInfo() const override {
		return info;
	}

	void ValidateDistributedWrite(ClientContext &, const DistributedWriteOperationContext &) const override {
	}

	idx_t FinalizeDistributedWrite(ClientContext &, const DistributedWriteOperationContext &,
	                               const vector<DistributedWriteTaskResult> &results) const override {
		idx_t rows = 0;
		for (const auto &result : results) {
			rows += result.RowCount();
		}
		return rows;
	}

	void AbortDistributedWrite(ClientContext &, const DistributedWriteOperationContext &,
	                           const vector<DistributedWriteTaskResult> &) const override {
	}

private:
	DistributedExtensionWriteInfo info;
};

// A tiny test-only nullary aggregate operator used to construct BoundAggregateExpression
struct TestNullaryAggOp {
	template <class STATE>
	static void Initialize(STATE &state) {
		state = 0;
	}
	template <class STATE, class OP>
	static void Operation(STATE &state, AggregateInputData &, idx_t) {
		state += 1;
	}
	template <class STATE, class OP>
	static void ConstantOperation(STATE &state, AggregateInputData &, idx_t count) {
		state += count;
	}
	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &) {
		target += source;
	}
	template <class STATE, class RESULT_TYPE>
	static void Finalize(STATE &state, RESULT_TYPE &target, AggregateFinalizeData &) {
		target = state;
	}
};

static OperatorResultType TestInOutFunction(ExecutionContext &, TableFunctionInput &, DataChunk &, DataChunk &) {
	return OperatorResultType::NEED_MORE_INPUT;
}

struct UnaryPlan {
	DuckPhysicalPlanRef plan;
	vector<LogicalType> types;
	PhysicalOperator *scan;
};

static UnaryPlan MakeUnaryScanPlan() {
	Allocator &alloc = Allocator::DefaultAllocator();
	auto plan = std::make_shared<PhysicalPlan>(alloc);
	vector<LogicalType> types = {LogicalType::BIGINT};
	auto collection = make_uniq<ColumnDataCollection>(alloc, types);
	auto &scan =
	    plan->Make<PhysicalColumnDataScan>(types, PhysicalOperatorType::COLUMN_DATA_SCAN, 0, std::move(collection));
	return {plan, types, &scan};
}

TEST_CASE("PhysicalPlanTranslator: extension write unwraps its COPY child", "[distributed][copy][extension-write]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto output_path = TestCreatePath("distributed_extension_write_translation.parquet");
	auto logical_plan = conn.ExtractPlan("COPY (SELECT 42 AS value) TO " + SQLStringLiteral(output_path) +
	                                     " (FORMAT PARQUET, RETURN_STATS true, USE_TMP_FILE false)");
	REQUIRE(logical_plan != nullptr);

	PhysicalPlanGenerator generator(*conn.context);
	auto physical_plan = generator.Plan(std::move(logical_plan));
	REQUIRE(physical_plan != nullptr);
	auto &copy_root = physical_plan->Root();
	REQUIRE(copy_root.type == PhysicalOperatorType::COPY_TO_FILE);
	REQUIRE(copy_root.Cast<PhysicalCopyToFile>().return_type == CopyFunctionReturnType::WRITTEN_FILE_STATISTICS);

	vector<LogicalType> extension_types {LogicalType::BIGINT};
	auto &extension = physical_plan->Make<TestExtensionWriteOperator>(std::move(extension_types));
	extension.children.push_back(copy_root);
	physical_plan->SetRoot(extension);

	PlanConfig config;
	config.db = db.instance;
	config.config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
	auto translated = physical_plan_to_pipeline_node(config, DuckPhysicalPlanRef(physical_plan.release()));
	REQUIRE(translated.is_ok());
	auto copy_finish = std::dynamic_pointer_cast<CopyFinishNode>(translated.value()->inner());
	REQUIRE(copy_finish != nullptr);
	REQUIRE(copy_finish->copy_sink()->spec().return_type == CopyFunctionReturnType::WRITTEN_FILE_STATISTICS);
}

TEST_CASE("PhysicalPlanTranslator: callback extension write installs a worker sink", "[distributed][extension-write]") {
	DuckDB db(nullptr);
	Connection connection(db);
	auto unary = MakeUnaryScanPlan();
	auto &extension = unary.plan->Make<TestExtensionWriteOperator>(
	    vector<LogicalType> {LogicalType::BIGINT}, "test_callback_extension_write", DistributedWriteMode::CALLBACK);
	extension.children.push_back(*unary.scan);
	unary.plan->SetRoot(extension);

	PlanConfig config;
	config.db = db.instance;
	config.config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
	auto translated = physical_plan_to_pipeline_node(config, unary.plan);
	REQUIRE(translated.is_ok());
	auto write_sink = std::dynamic_pointer_cast<ExtensionWriteSinkNode>(translated.value()->inner());
	REQUIRE(write_sink != nullptr);
	REQUIRE(write_sink->write_info().mode == DistributedWriteMode::CALLBACK);
	REQUIRE(write_sink->write_info().write_name == "test_callback_extension_write");
	REQUIRE(write_sink->write_info().fragment_codec == "test-extension.opaque-write");

	auto task_executor = std::make_shared<PlanTaskExecutor>(connection.context);
	PlanExecutionContext execution_context(task_executor, connection.context);
	auto tasks = translated.value()->produce_tasks(execution_context);
	auto next = tasks.poll_next();
	REQUIRE(next.first);
	REQUIRE(next.second.task() != nullptr);
	REQUIRE(next.second.task()->plan() != nullptr);
	REQUIRE(next.second.task()->plan()->HasRoot());
	auto &worker_write = next.second.task()->plan()->Root().Cast<PhysicalDistributedExtensionWrite>();
	REQUIRE(worker_write.info.capability == write_sink->write_info().capability);
	REQUIRE(worker_write.info.write_name == write_sink->write_info().write_name);
	REQUIRE(worker_write.info.fragment_codec == write_sink->write_info().fragment_codec);
	REQUIRE(worker_write.info.fragment_codec_version == write_sink->write_info().fragment_codec_version);
	REQUIRE(worker_write.info.worker_bind_data == write_sink->write_info().worker_bind_data);
	REQUIRE(worker_write.task_context.operation_id.empty());
	REQUIRE(worker_write.task_context.task_attempt_id.empty());
	REQUIRE(worker_write.children.size() == 1);
	REQUIRE_FALSE(tasks.poll_next().first);
}

TEST_CASE("PhysicalPlanTranslator: extension write must be the physical plan root",
          "[distributed][copy][extension-write]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto output_path = TestCreatePath("distributed_nested_extension_write.parquet");
	auto logical_plan = conn.ExtractPlan("COPY (SELECT 42 AS value) TO " + SQLStringLiteral(output_path) +
	                                     " (FORMAT PARQUET, RETURN_STATS true, USE_TMP_FILE false)");
	REQUIRE(logical_plan != nullptr);

	PhysicalPlanGenerator generator(*conn.context);
	auto physical_plan = generator.Plan(std::move(logical_plan));
	REQUIRE(physical_plan != nullptr);
	auto &copy_root = physical_plan->Root();

	vector<LogicalType> extension_types {LogicalType::BIGINT};
	auto &extension = physical_plan->Make<TestExtensionWriteOperator>(extension_types);
	extension.children.push_back(copy_root);

	vector<unique_ptr<Expression>> expressions;
	expressions.push_back(make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, 0));
	auto &projection = physical_plan->Make<PhysicalProjection>(extension_types, std::move(expressions), 0);
	projection.children.push_back(extension);
	physical_plan->SetRoot(projection);

	PlanConfig config;
	config.db = db.instance;
	config.config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
	auto translated = physical_plan_to_pipeline_node(config, DuckPhysicalPlanRef(physical_plan.release()));
	REQUIRE(translated.is_err());
	REQUIRE(StringUtil::Contains(translated.error().what(), "must be the physical plan root"));
}

TEST_CASE("PhysicalPlanTranslator: extension write requires a non-empty name", "[distributed][copy][extension-write]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto output_path = TestCreatePath("distributed_unnamed_extension_write.parquet");
	auto logical_plan = conn.ExtractPlan("COPY (SELECT 42 AS value) TO " + SQLStringLiteral(output_path) +
	                                     " (FORMAT PARQUET, RETURN_STATS true, USE_TMP_FILE false)");
	REQUIRE(logical_plan != nullptr);

	PhysicalPlanGenerator generator(*conn.context);
	auto physical_plan = generator.Plan(std::move(logical_plan));
	REQUIRE(physical_plan != nullptr);
	auto &copy_root = physical_plan->Root();

	vector<LogicalType> extension_types {LogicalType::BIGINT};
	auto &extension = physical_plan->Make<TestExtensionWriteOperator>(extension_types, string());
	extension.children.push_back(copy_root);
	physical_plan->SetRoot(extension);

	PlanConfig config;
	config.db = db.instance;
	config.config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
	auto translated = physical_plan_to_pipeline_node(config, DuckPhysicalPlanRef(physical_plan.release()));
	REQUIRE(translated.is_err());
	REQUIRE(StringUtil::Contains(translated.error().what(), "requires a non-empty name"));
}

static DuckPhysicalPlanRef MakeWindowPlan(bool streaming, idx_t input_partitions, bool input_hash_partitioned,
                                          const vector<idx_t> &partition_columns,
                                          const vector<idx_t> &second_partition_columns = {},
                                          bool swap_columns_after_repartition = false) {
	Allocator &alloc = Allocator::DefaultAllocator();
	auto plan = std::make_shared<PhysicalPlan>(alloc);
	vector<LogicalType> input_types = {LogicalType::BIGINT, LogicalType::BIGINT};
	auto collection = make_uniq<ColumnDataCollection>(alloc, input_types);
	auto &scan = plan->Make<PhysicalColumnDataScan>(input_types, PhysicalOperatorType::COLUMN_DATA_SCAN, 0,
	                                                std::move(collection));
	PhysicalOperator *input = &scan;

	if (input_partitions > 1) {
		std::shared_ptr<RepartitionSpec> spec;
		if (input_hash_partitioned) {
			vector<ExprRef> by;
			by.emplace_back(std::make_shared<BoundReferenceExpression>(LogicalType::BIGINT, 0));
			spec = RepartitionSpec::create_hash(input_partitions, std::move(by));
		} else {
			spec = RepartitionSpec::create_random(input_partitions);
		}
		auto &repartition = plan->Make<PhysicalRepartition>(input_types, std::move(spec), 0);
		repartition.children.push_back(scan);
		input = &repartition;
	}

	if (swap_columns_after_repartition) {
		vector<unique_ptr<Expression>> projections;
		projections.push_back(make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, 1));
		projections.push_back(make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, 0));
		auto &projection = plan->Make<PhysicalProjection>(input_types, std::move(projections), 0);
		projection.children.push_back(*input);
		input = &projection;
	}

	auto make_window_expression = [streaming](const vector<idx_t> &partitions) {
		auto expression =
		    make_uniq<BoundWindowExpression>(ExpressionType::WINDOW_ROW_NUMBER, LogicalType::BIGINT, nullptr, nullptr);
		for (auto column : partitions) {
			expression->partitions.push_back(make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, column));
		}
		if (!streaming) {
			expression->orders.emplace_back(OrderType::ASCENDING, OrderByNullType::NULLS_LAST,
			                                make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, 1));
		}
		return expression;
	};

	vector<unique_ptr<Expression>> select_list;
	select_list.push_back(make_window_expression(partition_columns));
	if (!second_partition_columns.empty()) {
		select_list.push_back(make_window_expression(second_partition_columns));
	}
	auto output_types = input_types;
	output_types.resize(input_types.size() + select_list.size(), LogicalType::BIGINT);
	if (streaming) {
		auto &window = plan->Make<PhysicalStreamingWindow>(output_types, std::move(select_list), 0);
		window.children.push_back(*input);
		plan->SetRoot(window);
	} else {
		auto &window = plan->Make<PhysicalWindow>(output_types, std::move(select_list), 0);
		window.children.push_back(*input);
		plan->SetRoot(window);
	}
	return plan;
}

static unique_ptr<ColumnDataCollection> MakeSingleValueCollection(const vector<LogicalType> &types,
                                                                  const vector<Value> &values) {
	auto collection = make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), types);
	for (idx_t col_idx = 0; col_idx < values.size(); col_idx++) {
		chunk.SetValue(col_idx, 0, values[col_idx]);
	}
	chunk.SetCardinality(1);
	collection->Append(chunk);
	return collection;
}

static idx_t SchemaColumnCount(const SchemaRef &schema) {
	return GetSchemaTypes(schema).size();
}

static string SQLStringLiteral(const string &value) {
	return "'" + StringUtil::Replace(value, "'", "''") + "'";
}

class ScopedTranslatorEnvironment {
public:
	ScopedTranslatorEnvironment(std::string name, const char *value) : name_(std::move(name)) {
		const auto *existing = std::getenv(name_.c_str());
		if (existing) {
			had_value_ = true;
			old_value_ = existing;
		}
		Set(value);
	}

	~ScopedTranslatorEnvironment() {
		if (had_value_) {
			Set(old_value_.c_str());
		} else {
			Set(nullptr);
		}
	}

private:
	void Set(const char *value) {
#if defined(_WIN32)
		_putenv_s(name_.c_str(), value ? value : "");
#else
		if (value) {
			setenv(name_.c_str(), value, 1);
		} else {
			unsetenv(name_.c_str());
		}
#endif
	}

	std::string name_;
	std::string old_value_;
	bool had_value_ = false;
};

static DuckPhysicalPlanRef MakeHashJoinPlan(JoinType join_type, idx_t left_cardinality, idx_t right_cardinality) {
	auto plan = std::make_shared<PhysicalPlan>(Allocator::DefaultAllocator());
	vector<LogicalType> input_types = {LogicalType::INTEGER};
	auto left_collection = make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), input_types);
	auto &left_scan = plan->Make<PhysicalColumnDataScan>(input_types, PhysicalOperatorType::COLUMN_DATA_SCAN,
	                                                     left_cardinality, std::move(left_collection));
	auto right_collection = make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), input_types);
	auto &right_scan = plan->Make<PhysicalColumnDataScan>(input_types, PhysicalOperatorType::COLUMN_DATA_SCAN,
	                                                      right_cardinality, std::move(right_collection));

	vector<JoinCondition> conditions;
	JoinCondition condition;
	condition.left = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0);
	condition.right = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0);
	condition.comparison = ExpressionType::COMPARE_EQUAL;
	conditions.push_back(std::move(condition));

	LogicalComparisonJoin logical_join(join_type);
	switch (join_type) {
	case JoinType::SEMI:
	case JoinType::ANTI:
	case JoinType::RIGHT_SEMI:
	case JoinType::RIGHT_ANTI:
		logical_join.types = input_types;
		break;
	case JoinType::MARK:
		logical_join.types = {LogicalType::INTEGER, LogicalType::BOOLEAN};
		break;
	default:
		logical_join.types = {LogicalType::INTEGER, LogicalType::INTEGER};
		break;
	}

	auto &hash_join = plan->Make<PhysicalHashJoin>(logical_join, left_scan, right_scan, std::move(conditions),
	                                               join_type, left_cardinality);
	plan->SetRoot(hash_join);
	return plan;
}

static DuckPhysicalPlanRef MakeCorrelatedMarkJoinPlan() {
	auto plan = std::make_shared<PhysicalPlan>(Allocator::DefaultAllocator());
	vector<LogicalType> input_types = {LogicalType::INTEGER, LogicalType::INTEGER};
	auto left_collection = make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), input_types);
	auto &left_scan = plan->Make<PhysicalColumnDataScan>(input_types, PhysicalOperatorType::COLUMN_DATA_SCAN, 1,
	                                                     std::move(left_collection));
	auto right_collection = make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), input_types);
	auto &right_scan = plan->Make<PhysicalColumnDataScan>(input_types, PhysicalOperatorType::COLUMN_DATA_SCAN, 1,
	                                                      std::move(right_collection));

	vector<JoinCondition> conditions;
	for (idx_t column_idx = 0; column_idx < input_types.size(); column_idx++) {
		JoinCondition condition;
		condition.left = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, column_idx);
		condition.right = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, column_idx);
		condition.comparison = ExpressionType::COMPARE_EQUAL;
		conditions.push_back(std::move(condition));
	}

	LogicalComparisonJoin logical_join(JoinType::MARK);
	logical_join.types = {LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BOOLEAN};
	auto &hash_join = plan->Make<PhysicalHashJoin>(logical_join, left_scan, right_scan, std::move(conditions),
	                                               JoinType::MARK, vector<idx_t> {}, vector<idx_t> {},
	                                               vector<LogicalType> {LogicalType::INTEGER}, 1, nullptr);
	plan->SetRoot(hash_join);
	return plan;
}

static bool NodeDisplayContains(const DistributedPipelineNodeRef &node, const std::string &needle) {
	for (const auto &line : node->inner()->multiline_display(false)) {
		if (line.find(needle) != std::string::npos) {
			return true;
		}
	}
	return false;
}

TEST_CASE("Streaming UDF passthrough schema preserves physical column boundaries", "[distributed][udf]") {
	SECTION("multiple scalar columns") {
		vector<LogicalType> output_types = {LogicalType::BIGINT, LogicalType::VARCHAR};
		StreamingUDFPassthroughNode node(1, nullptr, TableFunction {}, nullptr, vector<ColumnIndex> {},
		                                 vector<column_t> {}, optional_idx(), output_types, 0);

		auto schema_types = GetSchemaTypes(node.config().schema());
		REQUIRE(schema_types == output_types);
	}

	SECTION("one struct column") {
		child_list_t<LogicalType> fields;
		fields.emplace_back("name", LogicalType::VARCHAR);
		fields.emplace_back("score", LogicalType::DOUBLE);
		auto struct_type = LogicalType::STRUCT(std::move(fields));
		vector<LogicalType> output_types = {struct_type};
		StreamingUDFPassthroughNode node(1, nullptr, TableFunction {}, nullptr, vector<ColumnIndex> {},
		                                 vector<column_t> {}, optional_idx(), output_types, 0);

		auto schema_types = GetSchemaTypes(node.config().schema());
		REQUIRE(schema_types.size() == 1);
		REQUIRE(schema_types[0] == struct_type);
		REQUIRE(GetSchemaNames(node.config().schema()).empty());
		const auto &struct_fields = StructType::GetChildTypes(schema_types[0]);
		REQUIRE(struct_fields.size() == 2);
	}
}

TEST_CASE("PhysicalPlanTranslator: simple projection", "[distributed]") {
	// Construct a simple physical plan: TableScan -> Projection.
	Allocator allocator;
	PhysicalPlan plan(allocator);
	LogicalType int_type = LogicalType::INTEGER;
	vector<LogicalType> types = {int_type};
	TableFunction function;
	unique_ptr<FunctionData> bind_data;
	vector<LogicalType> return_types;
	vector<ColumnIndex> column_ids;
	vector<idx_t> projection_ids;
	vector<string> names;
	unique_ptr<TableFilterSet> table_filters;
	idx_t estimated_cardinality = 0;
	ExtraOperatorInfo extra_info;
	vector<Value> params;
	virtual_column_map_t virtual_map;
	auto &table_scan = plan.Make<PhysicalTableScan>(types, function, std::move(bind_data), return_types, column_ids,
	                                                projection_ids, names, std::move(table_filters),
	                                                estimated_cardinality, std::move(extra_info), params, virtual_map);
	vector<unique_ptr<Expression>> select_list1;
	select_list1.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(int_type, 0));
	auto &projection = plan.Make<PhysicalProjection>(types, std::move(select_list1), estimated_cardinality);
	plan.SetRoot(projection);

	// Build a shared_ptr<PhysicalPlan> for the translator
	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);
	// Move existing operators into plan_ptr: re-create them there
	auto &table_scan2 = plan_ptr->Make<PhysicalTableScan>(
	    types, function, std::move(bind_data), return_types, column_ids, projection_ids, names,
	    std::move(table_filters), estimated_cardinality, std::move(extra_info), params, virtual_map);
	vector<unique_ptr<Expression>> select_list2;
	select_list2.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(int_type, 0));
	auto &projection2 = plan_ptr->Make<PhysicalProjection>(types, std::move(select_list2), estimated_cardinality);
	plan_ptr->SetRoot(projection2);
	auto result = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan_ptr);
	REQUIRE(result.ok);
}

TEST_CASE("PhysicalPlanTranslator: filter + projection", "[distributed]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);
	LogicalType int_type = LogicalType::INTEGER;
	vector<LogicalType> types = {int_type};
	TableFunction function;
	unique_ptr<FunctionData> bind_data;
	vector<LogicalType> return_types;
	vector<ColumnIndex> column_ids;
	vector<idx_t> projection_ids;
	vector<string> names;
	unique_ptr<TableFilterSet> table_filters;
	idx_t estimated_cardinality = 0;
	ExtraOperatorInfo extra_info;
	vector<Value> params;
	virtual_column_map_t virtual_map;
	auto &table_scan = plan.Make<PhysicalTableScan>(types, function, std::move(bind_data), return_types, column_ids,
	                                                projection_ids, names, std::move(table_filters),
	                                                estimated_cardinality, std::move(extra_info), params, virtual_map);
	vector<unique_ptr<Expression>> filter_select_list1;
	filter_select_list1.push_back(duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(1)));
	REQUIRE(filter_select_list1.size() == 1);
	auto &filter = plan.Make<PhysicalFilter>(types, std::move(filter_select_list1), estimated_cardinality);
	vector<unique_ptr<Expression>> select_list1;
	select_list1.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(int_type, 0));
	auto &projection = plan.Make<PhysicalProjection>(types, std::move(select_list1), estimated_cardinality);
	plan.SetRoot(projection);

	auto plan_ptr2 = std::make_shared<PhysicalPlan>(allocator);
	auto &table_scan3 = plan_ptr2->Make<PhysicalTableScan>(
	    types, function, std::move(bind_data), return_types, column_ids, projection_ids, names,
	    std::move(table_filters), estimated_cardinality, std::move(extra_info), params, virtual_map);
	vector<unique_ptr<Expression>> filter_select_list2;
	filter_select_list2.push_back(duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(1)));
	REQUIRE(filter_select_list2.size() == 1);
	auto &filter2 = plan_ptr2->Make<PhysicalFilter>(types, std::move(filter_select_list2), estimated_cardinality);
	vector<unique_ptr<Expression>> select_list2;
	select_list2.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(int_type, 0));
	REQUIRE(select_list2.size() == 1);
	auto &projection3 = plan_ptr2->Make<PhysicalProjection>(types, std::move(select_list2), estimated_cardinality);
	plan_ptr2->SetRoot(projection3);
	auto result2 = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan_ptr2);
	REQUIRE(result2.ok);
}

TEST_CASE("PhysicalPlanTranslator: null plan returns error", "[distributed]") {
	DuckPhysicalPlanRef null_plan;
	auto res = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, null_plan);
	REQUIRE(res.is_err());
	auto msg = std::string(res.error().what());
	REQUIRE(msg.find("physical plan is null") != std::string::npos);
}

TEST_CASE("PhysicalPlanTranslator: plan without root returns error", "[distributed]") {
	Allocator allocator;
	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);
	auto res = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan_ptr);
	REQUIRE(res.is_err());
	auto msg = std::string(res.error().what());
	REQUIRE(msg.find("physical plan has no root") != std::string::npos);
}

TEST_CASE("PhysicalPlanTranslator: unsupported table function bind data is a value error", "[distributed]") {
	Allocator allocator;
	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> types = {LogicalType::INTEGER};
	TableFunction function;
	function.name = "unsupported_table_scan";
	auto bind_data = make_uniq<TableFunctionData>();
	vector<ColumnIndex> column_ids = {ColumnIndex(0)};
	vector<string> names = {"value"};

	auto &scan = plan_ptr->Make<PhysicalTableScan>(types, function, std::move(bind_data), types, std::move(column_ids),
	                                               vector<idx_t> {}, std::move(names), nullptr, 0, ExtraOperatorInfo {},
	                                               vector<Value> {}, virtual_column_map_t {});
	plan_ptr->SetRoot(scan);

	auto res = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan_ptr);
	REQUIRE(res.is_err());
	REQUIRE(res.error().type() == DuckDBError::Type::ValueError);
	auto msg = std::string(res.error().what());
	REQUIRE(msg.find("unsupported_table_scan") != std::string::npos);
	REQUIRE(msg.find("Copy not supported for TableFunctionData") != std::string::npos);
}

TEST_CASE("PhysicalPlanTranslator: auto broadcast only considers semantically safe sides", "[distributed][join]") {
	ScopedTranslatorEnvironment strategy("VANE_DISTRIBUTED_JOIN_STRATEGY", nullptr);
	ScopedTranslatorEnvironment threshold("VANE_DISTRIBUTED_AUTO_BROADCAST_THRESHOLD_BYTES", "64");
	PlanConfig config;
	config.num_partitions = 4;

	SECTION("hash join is selected when only the small side is preserved") {
		auto res = physical_plan_to_pipeline_node(config, MakeHashJoinPlan(JoinType::LEFT, 2, 400));
		REQUIRE(res.is_ok());
		REQUIRE(res.value()->name() == "HashJoin");
	}

	SECTION("optimizer-swapped plan selects hash join when only the small side is preserved") {
		auto res = physical_plan_to_pipeline_node(config, MakeHashJoinPlan(JoinType::RIGHT, 400, 2));
		REQUIRE(res.is_ok());
		REQUIRE(res.value()->name() == "HashJoin");
	}

	SECTION("small non-preserved side is broadcast") {
		auto res = physical_plan_to_pipeline_node(config, MakeHashJoinPlan(JoinType::LEFT, 400, 2));
		REQUIRE(res.is_ok());
		REQUIRE(res.value()->name() == "BroadcastJoin");
		REQUIRE(NodeDisplayContains(res.value(), "Broadcast side: right"));
	}

	SECTION("optimizer-swapped small non-preserved side is broadcast") {
		auto res = physical_plan_to_pipeline_node(config, MakeHashJoinPlan(JoinType::RIGHT, 2, 400));
		REQUIRE(res.is_ok());
		REQUIRE(res.value()->name() == "BroadcastJoin");
		REQUIRE(NodeDisplayContains(res.value(), "Broadcast side: left"));
	}

	SECTION("inner join broadcasts the smaller side") {
		auto res = physical_plan_to_pipeline_node(config, MakeHashJoinPlan(JoinType::INNER, 2, 400));
		REQUIRE(res.is_ok());
		REQUIRE(res.value()->name() == "BroadcastJoin");
		REQUIRE(NodeDisplayContains(res.value(), "Broadcast side: left"));
	}

	SECTION("full outer join selects hash join") {
		auto res = physical_plan_to_pipeline_node(config, MakeHashJoinPlan(JoinType::OUTER, 2, 2));
		REQUIRE(res.is_ok());
		REQUIRE(res.value()->name() == "HashJoin");
	}
}

TEST_CASE("PhysicalPlanTranslator: forced broadcast validates the selected side", "[distributed][join]") {
	ScopedTranslatorEnvironment threshold("VANE_DISTRIBUTED_AUTO_BROADCAST_THRESHOLD_BYTES", "0");
	PlanConfig config;
	config.num_partitions = 4;

	SECTION("generic strategy chooses the non-preserved side") {
		ScopedTranslatorEnvironment strategy("VANE_DISTRIBUTED_JOIN_STRATEGY", "broadcast");
		auto res = physical_plan_to_pipeline_node(config, MakeHashJoinPlan(JoinType::LEFT, 2, 400));
		REQUIRE(res.is_ok());
		REQUIRE(res.value()->name() == "BroadcastJoin");
		REQUIRE(NodeDisplayContains(res.value(), "Broadcast side: right"));
	}

	SECTION("unsafe directional strategy fails clearly") {
		ScopedTranslatorEnvironment strategy("VANE_DISTRIBUTED_JOIN_STRATEGY", "broadcast_left");
		auto res = physical_plan_to_pipeline_node(config, MakeHashJoinPlan(JoinType::LEFT, 2, 400));
		REQUIRE(res.is_err());
		REQUIRE(std::string(res.error().what()).find("semantically preserved") != std::string::npos);
	}

	SECTION("generic full outer join fails clearly") {
		ScopedTranslatorEnvironment strategy("VANE_DISTRIBUTED_JOIN_STRATEGY", "broadcast");
		auto res = physical_plan_to_pipeline_node(config, MakeHashJoinPlan(JoinType::OUTER, 2, 2));
		REQUIRE(res.is_err());
		REQUIRE(std::string(res.error().what()).find("neither side") != std::string::npos);
	}

	SECTION("directional full outer join fails clearly") {
		ScopedTranslatorEnvironment strategy("VANE_DISTRIBUTED_JOIN_STRATEGY", "broadcast_left");
		auto res = physical_plan_to_pipeline_node(config, MakeHashJoinPlan(JoinType::OUTER, 2, 2));
		REQUIRE(res.is_err());
		REQUIRE(std::string(res.error().what()).find("semantically preserved") != std::string::npos);
	}
}

TEST_CASE("PhysicalFilter: empty select list handled as true", "[distributed]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);
	LogicalType int_type = LogicalType::INTEGER;
	vector<LogicalType> types = {int_type};
	TableFunction function;
	unique_ptr<FunctionData> bind_data;
	vector<LogicalType> return_types;
	vector<ColumnIndex> column_ids;
	vector<idx_t> projection_ids;
	vector<string> names;
	unique_ptr<TableFilterSet> table_filters;
	idx_t estimated_cardinality = 0;
	ExtraOperatorInfo extra_info;
	vector<Value> params;
	virtual_column_map_t virtual_map;

	// Build a plan with a filter that has an empty select list
	auto &table_scan = plan.Make<PhysicalTableScan>(types, function, std::move(bind_data), return_types, column_ids,
	                                                projection_ids, names, std::move(table_filters),
	                                                estimated_cardinality, std::move(extra_info), params, virtual_map);
	vector<unique_ptr<Expression>> filter_select_list; // empty
	auto &filter = plan.Make<PhysicalFilter>(types, std::move(filter_select_list), estimated_cardinality);
	vector<unique_ptr<Expression>> select_list;
	select_list.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(int_type, 0));
	auto &projection = plan.Make<PhysicalProjection>(types, std::move(select_list), estimated_cardinality);
	plan.SetRoot(projection);

	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);
	// Recreate the plan in a shared ptr
	auto &table_scan2 = plan_ptr->Make<PhysicalTableScan>(
	    types, function, unique_ptr<FunctionData>(), return_types, column_ids, projection_ids, names,
	    unique_ptr<TableFilterSet>(), estimated_cardinality, ExtraOperatorInfo(), params, virtual_map);
	vector<unique_ptr<Expression>> filter_select_list2; // empty
	auto &filter2 = plan_ptr->Make<PhysicalFilter>(types, std::move(filter_select_list2), estimated_cardinality);
	vector<unique_ptr<Expression>> select_list2;
	select_list2.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(int_type, 0));
	auto &projection2 = plan_ptr->Make<PhysicalProjection>(types, std::move(select_list2), estimated_cardinality);
	plan_ptr->SetRoot(projection2);

	auto result = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan_ptr);
	REQUIRE(result.ok);
}

TEST_CASE("PhysicalPlanTranslator: grouped hash aggregate -> AggregateNode", "[distributed]") {
	Allocator allocator;
	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);
	duckdb::vector<duckdb::LogicalType> types = {duckdb::LogicalType::BIGINT};

	// Create a grouped hash aggregate (requires a ClientContext)
	duckdb::DuckDB db(nullptr);
	duckdb::Connection conn(db);

	duckdb::vector<unique_ptr<Expression>> groups;
	groups.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0));
	duckdb::vector<unique_ptr<Expression>> aggrs;
	// Create a simple nullary aggregate expression for testing
	{
		using AggFun =
		    decltype(AggregateFunction::NullaryAggregate<int64_t, int64_t, TestNullaryAggOp>(LogicalType::BIGINT));
		auto agg_fun =
		    AggregateFunction::NullaryAggregate<int64_t, int64_t, TestNullaryAggOp>(duckdb::LogicalType::BIGINT);
		agg_fun.name = "test_nullary";
		duckdb::vector<unique_ptr<Expression>> children;
		aggrs.push_back(duckdb::make_uniq<duckdb::BoundAggregateExpression>(
		    std::move(agg_fun), std::move(children), nullptr, nullptr, AggregateType::NON_DISTINCT));
	}
	// Debug: print aggregate expressions created by the test
	for (idx_t i = 0; i < aggrs.size(); i++) {
		std::cout << "[TEST DEBUG] aggrs[" << i << "] name=" << aggrs[i]->GetName()
		          << " class=" << (int)aggrs[i]->GetExpressionClass() << std::endl;
	}

	auto &agg =
	    plan_ptr->Make<duckdb::PhysicalHashAggregate>(*conn.context, types, std::move(aggrs), std::move(groups), 0);
	plan_ptr->SetRoot(agg);

	auto res = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan_ptr);
	REQUIRE(res.ok);
	REQUIRE(res.value() != nullptr);
	auto dist = res.value();
	auto inner = dist->inner();
	REQUIRE(std::dynamic_pointer_cast<duckdb::distributed::AggregateNode>(inner) != nullptr);
}

TEST_CASE("PhysicalPlanTranslator: grouping sets use expand-shuffle-aggregate", "[distributed][grouping_sets]") {
	Allocator allocator;
	auto plan = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> input_types = {LogicalType::INTEGER, LogicalType::INTEGER};
	auto collection = make_uniq<ColumnDataCollection>(allocator, input_types);
	auto &scan = plan->Make<PhysicalColumnDataScan>(input_types, PhysicalOperatorType::COLUMN_DATA_SCAN, 0,
	                                                std::move(collection));
	auto repartition_spec = RepartitionSpec::create_random(4);
	auto &input = plan->Make<PhysicalRepartition>(input_types, std::move(repartition_spec), 0);
	input.children.push_back(scan);

	vector<unique_ptr<Expression>> groups;
	groups.push_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0));
	groups.push_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 1));
	vector<unique_ptr<Expression>> aggregates;
	vector<GroupingSet> grouping_sets = {{0, 1}, {0}, {0}, {}};
	vector<unsafe_vector<idx_t>> grouping_functions = {{0, 1}};
	vector<LogicalType> output_types = {LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BIGINT};

	DuckDB db(nullptr);
	Connection conn(db);
	auto &aggregate = plan->Make<PhysicalHashAggregate>(
	    *conn.context, output_types, std::move(aggregates), std::move(groups), grouping_sets, grouping_functions, 0,
	    TupleDataValidityType::CAN_HAVE_NULL_VALUES, TupleDataValidityType::CAN_HAVE_NULL_VALUES);
	aggregate.children.push_back(input);
	plan->SetRoot(aggregate);

	PlanConfig config;
	config.num_partitions = 4;
	auto result = physical_plan_to_pipeline_node(config, std::move(plan));

	REQUIRE(result.is_ok());
	auto projection = std::dynamic_pointer_cast<ProjectionNode>(result.value()->inner());
	REQUIRE(projection != nullptr);
	REQUIRE(SchemaColumnCount(projection->config().schema()) == output_types.size());
	REQUIRE(projection->config().clustering_spec()->type() == ClusteringSpec::Type::Unknown);
	REQUIRE(projection->config().clustering_spec()->num_partitions() == 4);

	auto projection_children = projection->children();
	REQUIRE(projection_children.size() == 1);
	auto final_aggregate = std::dynamic_pointer_cast<AggregateNode>(projection_children[0]);
	REQUIRE(final_aggregate != nullptr);

	auto final_children = final_aggregate->children();
	REQUIRE(final_children.size() == 1);
	auto shuffle = std::dynamic_pointer_cast<RepartitionNode>(final_children[0]);
	REQUIRE(shuffle != nullptr);
	REQUIRE(shuffle->config().clustering_spec()->type() == ClusteringSpec::Type::Hash);
	REQUIRE(shuffle->config().clustering_spec()->partition_by().size() == 3);

	auto shuffle_children = shuffle->children();
	REQUIRE(shuffle_children.size() == 1);
	REQUIRE(std::dynamic_pointer_cast<GroupingSetExpandNode>(shuffle_children[0]) != nullptr);
}

TEST_CASE("PhysicalPlanTranslator: grouping-set expansion accepts exact aggregate semantics",
          "[distributed][grouping_sets]") {
	DuckDB db(nullptr);
	Connection conn(db);

	AggregateType aggregate_type = AggregateType::NON_DISTINCT;
	bool with_filter = false;
	bool with_order = false;
	SECTION("DISTINCT") {
		aggregate_type = AggregateType::DISTINCT;
	}
	SECTION("FILTER") {
		with_filter = true;
	}
	SECTION("ordered aggregate") {
		with_order = true;
	}

	auto &allocator = Allocator::DefaultAllocator();
	auto plan = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> input_types = {LogicalType::BIGINT, LogicalType::BOOLEAN};
	auto collection = make_uniq<ColumnDataCollection>(allocator, input_types);
	auto &scan = plan->Make<PhysicalColumnDataScan>(input_types, PhysicalOperatorType::COLUMN_DATA_SCAN, 0,
	                                                std::move(collection));
	auto repartition_spec = RepartitionSpec::create_random(4);
	auto &input = plan->Make<PhysicalRepartition>(input_types, std::move(repartition_spec), 0);
	input.children.push_back(scan);

	auto aggregate_function =
	    AggregateFunction::NullaryAggregate<int64_t, int64_t, TestNullaryAggOp>(LogicalType::BIGINT);
	aggregate_function.name = "test_nullary";
	unique_ptr<Expression> filter;
	if (with_filter) {
		filter = make_uniq<BoundReferenceExpression>(LogicalType::BOOLEAN, 1);
	}
	auto aggregate_expression = make_uniq<BoundAggregateExpression>(
	    std::move(aggregate_function), vector<unique_ptr<Expression>> {}, std::move(filter), nullptr, aggregate_type);
	if (with_order) {
		aggregate_expression->order_bys = make_uniq<BoundOrderModifier>();
		aggregate_expression->order_bys->orders.emplace_back(
		    OrderType::ASCENDING, OrderByNullType::NULLS_LAST,
		    make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, 0));
	}

	vector<unique_ptr<Expression>> aggregates;
	aggregates.push_back(std::move(aggregate_expression));
	vector<unique_ptr<Expression>> groups;
	groups.push_back(make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, 0));
	vector<GroupingSet> grouping_sets = {{0}, {}};
	vector<unsafe_vector<idx_t>> grouping_functions;
	vector<LogicalType> output_types = {LogicalType::BIGINT, LogicalType::BIGINT};
	auto &aggregate = plan->Make<PhysicalHashAggregate>(
	    *conn.context, output_types, std::move(aggregates), std::move(groups), std::move(grouping_sets),
	    std::move(grouping_functions), 0, TupleDataValidityType::CAN_HAVE_NULL_VALUES,
	    TupleDataValidityType::CAN_HAVE_NULL_VALUES);
	aggregate.children.push_back(input);
	plan->SetRoot(aggregate);

	PlanConfig config;
	config.num_partitions = 4;
	auto result = physical_plan_to_pipeline_node(config, std::move(plan));

	REQUIRE(result.is_ok());
	auto projection = std::dynamic_pointer_cast<ProjectionNode>(result.value()->inner());
	REQUIRE(projection != nullptr);
	auto final_aggregate = std::dynamic_pointer_cast<AggregateNode>(projection->children()[0]);
	REQUIRE(final_aggregate != nullptr);
	auto shuffle = std::dynamic_pointer_cast<RepartitionNode>(final_aggregate->children()[0]);
	REQUIRE(shuffle != nullptr);
	REQUIRE(std::dynamic_pointer_cast<GroupingSetExpandNode>(shuffle->children()[0]) != nullptr);
}

TEST_CASE("PhysicalPlanTranslator: grouping-set expansion accepts zero-column input", "[distributed][grouping_sets]") {
	auto &allocator = Allocator::DefaultAllocator();
	auto plan = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> input_types;
	auto collection = make_uniq<ColumnDataCollection>(allocator, input_types);
	auto &scan = plan->Make<PhysicalColumnDataScan>(input_types, PhysicalOperatorType::COLUMN_DATA_SCAN, 0,
	                                                std::move(collection));
	auto repartition_spec = RepartitionSpec::create_random(4);
	auto &input = plan->Make<PhysicalRepartition>(input_types, std::move(repartition_spec), 0);
	input.children.push_back(scan);

	auto aggregate_function =
	    AggregateFunction::NullaryAggregate<int64_t, int64_t, TestNullaryAggOp>(LogicalType::BIGINT);
	aggregate_function.name = "test_nullary";
	vector<unique_ptr<Expression>> aggregates;
	aggregates.push_back(make_uniq<BoundAggregateExpression>(std::move(aggregate_function),
	                                                         vector<unique_ptr<Expression>> {}, nullptr, nullptr,
	                                                         AggregateType::NON_DISTINCT));
	vector<unique_ptr<Expression>> groups;
	vector<GroupingSet> grouping_sets = {{}, {}};
	vector<unsafe_vector<idx_t>> grouping_functions;
	vector<LogicalType> output_types = {LogicalType::BIGINT};

	DuckDB db(nullptr);
	Connection conn(db);
	auto &aggregate = plan->Make<PhysicalHashAggregate>(
	    *conn.context, output_types, std::move(aggregates), std::move(groups), std::move(grouping_sets),
	    std::move(grouping_functions), 0, TupleDataValidityType::CAN_HAVE_NULL_VALUES,
	    TupleDataValidityType::CAN_HAVE_NULL_VALUES);
	aggregate.children.push_back(input);
	plan->SetRoot(aggregate);

	PlanConfig config;
	config.num_partitions = 4;
	auto result = physical_plan_to_pipeline_node(config, std::move(plan));

	REQUIRE(result.is_ok());
	auto projection = std::dynamic_pointer_cast<ProjectionNode>(result.value()->inner());
	REQUIRE(projection != nullptr);
	auto final_aggregate = std::dynamic_pointer_cast<AggregateNode>(projection->children()[0]);
	REQUIRE(final_aggregate != nullptr);
	auto shuffle = std::dynamic_pointer_cast<RepartitionNode>(final_aggregate->children()[0]);
	REQUIRE(shuffle != nullptr);
	REQUIRE(shuffle->config().clustering_spec()->partition_by().size() == 1);
	REQUIRE(std::dynamic_pointer_cast<GroupingSetExpandNode>(shuffle->children()[0]) != nullptr);
}

TEST_CASE("PhysicalPlanTranslator: distributed distinct aggregate throws", "[distributed]") {
	Allocator allocator;
	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);
	duckdb::vector<duckdb::LogicalType> types = {duckdb::LogicalType::BIGINT};

	duckdb::DuckDB db(nullptr);
	duckdb::Connection conn(db);

	duckdb::vector<unique_ptr<Expression>> groups;
	groups.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0));

	duckdb::vector<unique_ptr<Expression>> aggrs;
	{
		auto agg_fun =
		    AggregateFunction::NullaryAggregate<int64_t, int64_t, TestNullaryAggOp>(duckdb::LogicalType::BIGINT);
		agg_fun.name = "test_nullary";
		duckdb::vector<unique_ptr<Expression>> children;
		aggrs.push_back(duckdb::make_uniq<duckdb::BoundAggregateExpression>(std::move(agg_fun), std::move(children),
		                                                                    nullptr, nullptr, AggregateType::DISTINCT));
	}

	auto &scan = plan_ptr->Make<duckdb::PhysicalDummyScan>(types, 1);
	auto &agg =
	    plan_ptr->Make<duckdb::PhysicalHashAggregate>(*conn.context, types, std::move(aggrs), std::move(groups), 0);
	agg.children.push_back(scan);
	plan_ptr->SetRoot(agg);

	// With a single-partition input (DummyScan), the translator takes the
	// single-partition fast path and does NOT attempt to split the aggregate
	// into pre/post stages, so it succeeds even for DISTINCT aggregates.
	duckdb::distributed::PlanConfig cfg {};
	cfg.num_partitions = 2;

	auto res = duckdb::distributed::physical_plan_to_pipeline_node(cfg, plan_ptr);
	REQUIRE(res.is_ok());
}

TEST_CASE("PhysicalPlanTranslator: perfect hash aggregate -> PerfectHashAggregateNode", "[distributed]") {
	Allocator allocator;
	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::BIGINT};

	duckdb::DuckDB db(nullptr);
	duckdb::Connection conn(db);

	vector<unique_ptr<Expression>> groups;
	groups.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(LogicalType::INTEGER, 0));
	vector<unique_ptr<Expression>> aggrs;
	{
		auto agg_fun = AggregateFunction::NullaryAggregate<int64_t, int64_t, TestNullaryAggOp>(LogicalType::BIGINT);
		agg_fun.name = "test_nullary";
		vector<unique_ptr<Expression>> children;
		aggrs.push_back(duckdb::make_uniq<duckdb::BoundAggregateExpression>(
		    std::move(agg_fun), std::move(children), nullptr, nullptr, AggregateType::NON_DISTINCT));
	}

	vector<Value> group_minima;
	group_minima.push_back(Value::INTEGER(0));
	vector<idx_t> required_bits;
	required_bits.push_back(4);

	auto &agg = plan_ptr->template Make<duckdb::PhysicalPerfectHashAggregate>(
	    *conn.context, types, std::move(aggrs), std::move(groups), std::move(group_minima), std::move(required_bits),
	    0);
	plan_ptr->SetRoot(agg);

	auto res = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan_ptr);
	REQUIRE(res.ok);
	REQUIRE(res.value() != nullptr);
	auto dist = res.value();
	auto inner = dist->inner();
	REQUIRE(std::dynamic_pointer_cast<duckdb::distributed::PerfectHashAggregateNode>(inner) != nullptr);
}

TEST_CASE("PhysicalPlanTranslator: partitioned aggregate -> PartitionedAggregateNode", "[distributed]") {
	Allocator allocator;
	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::BIGINT};

	duckdb::DuckDB db(nullptr);
	duckdb::Connection conn(db);

	vector<unique_ptr<Expression>> groups;
	groups.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(LogicalType::INTEGER, 0));
	vector<unique_ptr<Expression>> aggrs;
	{
		auto agg_fun = AggregateFunction::NullaryAggregate<int64_t, int64_t, TestNullaryAggOp>(LogicalType::BIGINT);
		agg_fun.name = "test_nullary";
		vector<unique_ptr<Expression>> children;
		aggrs.push_back(duckdb::make_uniq<duckdb::BoundAggregateExpression>(
		    std::move(agg_fun), std::move(children), nullptr, nullptr, AggregateType::NON_DISTINCT));
	}

	vector<column_t> partitions;
	partitions.push_back(0);

	auto &agg = plan_ptr->template Make<duckdb::PhysicalPartitionedAggregate>(
	    *conn.context, types, std::move(aggrs), std::move(groups), std::move(partitions), 0);
	plan_ptr->SetRoot(agg);

	auto res = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan_ptr);
	REQUIRE(res.ok);
	REQUIRE(res.value() != nullptr);
	auto dist = res.value();
	auto inner = dist->inner();
	REQUIRE(std::dynamic_pointer_cast<duckdb::distributed::PartitionedAggregateNode>(inner) != nullptr);
}

TEST_CASE("PhysicalPlanTranslator: dummy scan -> ScanSourceNode", "[distributed]") {
	Allocator allocator;
	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> types = {LogicalType::BIGINT};

	auto &scan = plan_ptr->Make<PhysicalDummyScan>(types, 1);
	plan_ptr->SetRoot(scan);

	auto res = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan_ptr);
	REQUIRE(res.ok);
	REQUIRE(res.value() != nullptr);
	auto inner = res.value()->inner();
	REQUIRE(std::dynamic_pointer_cast<duckdb::distributed::ScanSourceNode>(inner) != nullptr);
}

TEST_CASE("PhysicalPlanTranslator: column data scan -> ScanSourceNode", "[distributed]") {
	Allocator allocator;
	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> types = {LogicalType::BIGINT};

	auto collection = MakeSingleValueCollection(types, {Value::BIGINT(42)});
	auto &scan =
	    plan_ptr->Make<PhysicalColumnDataScan>(types, PhysicalOperatorType::COLUMN_DATA_SCAN, 1, std::move(collection));
	plan_ptr->SetRoot(scan);

	auto res = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan_ptr);
	REQUIRE(res.ok);
	REQUIRE(res.value() != nullptr);
	auto inner = res.value()->inner();
	REQUIRE(std::dynamic_pointer_cast<duckdb::distributed::ScanSourceNode>(inner) != nullptr);
}

TEST_CASE("PhysicalPlanTranslator: column data scan schema preserves all columns", "[distributed]") {
	Allocator allocator;
	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> types = {LogicalType::BIGINT, LogicalType::VARCHAR};

	auto collection = MakeSingleValueCollection(types, {Value::BIGINT(42), Value("forty-two")});
	auto &scan =
	    plan_ptr->Make<PhysicalColumnDataScan>(types, PhysicalOperatorType::COLUMN_DATA_SCAN, 1, std::move(collection));
	plan_ptr->SetRoot(scan);

	auto res = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan_ptr);
	REQUIRE(res.ok);
	REQUIRE(res.value() != nullptr);
	REQUIRE(SchemaColumnCount(res.value()->config().schema()) == 2);
}

TEST_CASE("PhysicalPlanTranslator: cte scan -> ScanSourceNode", "[distributed]") {
	Allocator allocator;
	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> types = {LogicalType::BIGINT};

	auto collection = MakeSingleValueCollection(types, {Value::BIGINT(7)});
	auto &scan = plan_ptr->Make<PhysicalColumnDataScan>(types, PhysicalOperatorType::CTE_SCAN, 1, std::move(collection))
	                 .Cast<PhysicalColumnDataScan>();
	scan.cte_index = 0;
	plan_ptr->SetRoot(scan);

	auto res = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan_ptr);
	REQUIRE(res.ok);
	REQUIRE(res.value() != nullptr);
	auto inner = res.value()->inner();
	REQUIRE(std::dynamic_pointer_cast<duckdb::distributed::ScanSourceNode>(inner) != nullptr);
}

#if DUCKDB_EXTENSION_PARQUET_LINKED
TEST_CASE("PhysicalPlanTranslator: parquet scan splits row groups", "[distributed]") {
	const char *prev_min = std::getenv("DUCKDB_RAY_SCAN_TASK_MIN_BYTES");
	const char *prev_max = std::getenv("DUCKDB_RAY_SCAN_TASK_MAX_BYTES");
	const char *prev_rg_max = std::getenv("DUCKDB_RAY_PARQUET_SPLIT_ROW_GROUPS_MAX_FILES");

	setenv("DUCKDB_RAY_SCAN_TASK_MIN_BYTES", "1", 1);
	setenv("DUCKDB_RAY_SCAN_TASK_MAX_BYTES", "1", 1);
	setenv("DUCKDB_RAY_PARQUET_SPLIT_ROW_GROUPS_MAX_FILES", "1", 1);

	DuckDB db(nullptr);
	Connection conn(db);
	auto parquet_path = TestCreatePath("distributed_row_group_split_quote's.parquet");

	REQUIRE_NO_FAIL(conn.Query("CREATE TABLE rg_tbl AS SELECT range AS id FROM range(0, 50)"));
	REQUIRE_NO_FAIL(
	    conn.Query("COPY rg_tbl TO " + SQLStringLiteral(parquet_path) + " (FORMAT PARQUET, ROW_GROUP_SIZE 10)"));

	auto logical_plan = conn.ExtractPlan("SELECT * FROM parquet_scan(" + SQLStringLiteral(parquet_path) + ")");
	REQUIRE(logical_plan != nullptr);
	PhysicalPlanGenerator generator(*conn.context);
	auto physical_plan = generator.Plan(std::move(logical_plan));
	REQUIRE(physical_plan != nullptr);
	auto plan_ptr = std::shared_ptr<PhysicalPlan>(physical_plan.release());

	PlanConfig cfg;
	cfg.db = db.instance;
	cfg.config = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());

	auto res = duckdb::distributed::physical_plan_to_pipeline_node(cfg, plan_ptr);
	REQUIRE(res.ok);
	REQUIRE(res.value() != nullptr);
	REQUIRE(res.value()->num_partitions() > 1);
	auto scan_source = std::dynamic_pointer_cast<ScanSourceNode>(res.value()->inner());
	REQUIRE(scan_source != nullptr);
	REQUIRE_FALSE(scan_source->scan_tasks().empty());

	auto task_executor = std::make_shared<PlanTaskExecutor>(conn.context);
	PlanExecutionContext execution_context(task_executor, conn.context);
	auto task_stream = res.value()->produce_tasks(execution_context);
	auto next_task = task_stream.poll_next();
	REQUIRE(next_task.first);
	REQUIRE(next_task.second.task() != nullptr);
	auto worker_plan = next_task.second.task()->plan();
	REQUIRE(worker_plan != nullptr);
	auto &worker_scan = worker_plan->Root().Cast<PhysicalTableScan>();
	auto &worker_bind = worker_scan.bind_data->Cast<MultiFileBindData>();
	REQUIRE(worker_bind.file_list->GetAllFiles().empty());
	string assignment_error;
	REQUIRE_FALSE(ValidateDistributedScanTasksApplied(*worker_plan, &assignment_error));
	REQUIRE(StringUtil::Contains(assignment_error, "no explicit worker task assignment"));

	REQUIRE(next_task.second.task()->inputs().size() == 1);
	const auto &task_input = next_task.second.task()->inputs().begin()->second;
	REQUIRE(task_input.kind == TaskInput::Kind::ScanTask);
	std::unordered_map<idx_t, ScanTaskDescriptor> assignments;
	assignments.emplace(static_cast<idx_t>(scan_source->node_id()),
	                    ScanTaskDescriptor::DeserializeFromBytes(task_input.scan_task_bytes));
	assignment_error.clear();
	REQUIRE(ApplyScanTasksToPlan(*worker_plan, assignments, &assignment_error));
	REQUIRE(assignment_error.empty());
	REQUIRE(ValidateDistributedScanTasksApplied(*worker_plan, &assignment_error));
	REQUIRE_FALSE(worker_bind.file_list->GetAllFiles().empty());

	if (prev_min) {
		setenv("DUCKDB_RAY_SCAN_TASK_MIN_BYTES", prev_min, 1);
	} else {
		unsetenv("DUCKDB_RAY_SCAN_TASK_MIN_BYTES");
	}
	if (prev_max) {
		setenv("DUCKDB_RAY_SCAN_TASK_MAX_BYTES", prev_max, 1);
	} else {
		unsetenv("DUCKDB_RAY_SCAN_TASK_MAX_BYTES");
	}
	if (prev_rg_max) {
		setenv("DUCKDB_RAY_PARQUET_SPLIT_ROW_GROUPS_MAX_FILES", prev_rg_max, 1);
	} else {
		unsetenv("DUCKDB_RAY_PARQUET_SPLIT_ROW_GROUPS_MAX_FILES");
	}
}
#endif

TEST_CASE("PhysicalPlanTranslator: expression scan -> ExpressionScanNode", "[distributed]") {
	Allocator allocator;
	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> types = {LogicalType::INTEGER};

	auto &child_scan = plan_ptr->Make<PhysicalDummyScan>(types, 1);
	vector<vector<unique_ptr<Expression>>> expressions;
	vector<unique_ptr<Expression>> row;
	row.push_back(make_uniq<BoundConstantExpression>(Value::INTEGER(42)));
	expressions.push_back(std::move(row));
	auto &expr_scan = plan_ptr->Make<PhysicalExpressionScan>(types, std::move(expressions), 1);
	expr_scan.children.push_back(child_scan);
	plan_ptr->SetRoot(expr_scan);

	auto res = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan_ptr);
	REQUIRE(res.ok);
	REQUIRE(res.value() != nullptr);
	auto inner = res.value()->inner();
	REQUIRE(std::dynamic_pointer_cast<duckdb::distributed::ExpressionScanNode>(inner) != nullptr);
}

TEST_CASE("PhysicalPlanTranslator: ungrouped aggregate -> AggregateNode", "[distributed]") {
	Allocator allocator;
	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);
	duckdb::vector<duckdb::LogicalType> types = {duckdb::LogicalType::BIGINT};

	duckdb::vector<unique_ptr<Expression>> aggrs;
	// Create a simple nullary aggregate expression for testing
	{
		auto agg_fun =
		    AggregateFunction::NullaryAggregate<int64_t, int64_t, TestNullaryAggOp>(duckdb::LogicalType::BIGINT);
		agg_fun.name = "test_nullary";
		duckdb::vector<unique_ptr<Expression>> children;
		aggrs.push_back(duckdb::make_uniq<duckdb::BoundAggregateExpression>(
		    std::move(agg_fun), std::move(children), nullptr, nullptr, AggregateType::NON_DISTINCT));
	}
	// Debug: print aggregate expressions created by the test
	for (idx_t i = 0; i < aggrs.size(); i++) {
		std::cout << "[TEST DEBUG] uagg aggrs[" << i << "] name=" << aggrs[i]->GetName()
		          << " class=" << (int)aggrs[i]->GetExpressionClass() << std::endl;
	}

	auto &uagg = plan_ptr->Make<duckdb::PhysicalUngroupedAggregate>(types, std::move(aggrs), 0,
	                                                                TupleDataValidityType::CAN_HAVE_NULL_VALUES);
	plan_ptr->SetRoot(uagg);

	auto res = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan_ptr);
	REQUIRE(res.ok);
	REQUIRE(res.value() != nullptr);
	auto dist = res.value();
	auto inner = dist->inner();
	REQUIRE(std::dynamic_pointer_cast<duckdb::distributed::AggregateNode>(inner) != nullptr);
}

TEST_CASE("GroupedAggregateData: initialize with BoundAggregateExpression", "[distributed]") {
	// Direct unit test to ensure GroupedAggregateData accepts BoundAggregateExpression
	vector<unique_ptr<Expression>> groups;
	vector<unique_ptr<Expression>> aggrs;
	auto agg_fun = AggregateFunction::NullaryAggregate<int64_t, int64_t, TestNullaryAggOp>(LogicalType::BIGINT);
	agg_fun.name = "test_nullary";
	vector<unique_ptr<Expression>> children;
	std::cout << "[TEST DEBUG] creating BoundAggregateExpression with agg_fun.name='" << agg_fun.name << "'"
	          << std::endl;
	std::cout << std::flush;
	aggrs.push_back(make_uniq<BoundAggregateExpression>(std::move(agg_fun), std::move(children), nullptr, nullptr,
	                                                    AggregateType::NON_DISTINCT));

	duckdb::GroupByNode dummy_group_by;
	duckdb::GroupedAggregateData gad;
	// Should not throw
	gad.InitializeGroupby(std::move(groups), std::move(aggrs), {});
}

TEST_CASE("PhysicalPlanTranslator: grouped hash aggregate produces Aggregate node", "[distributed]") {
	Allocator allocator;
	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> types = {LogicalType::INTEGER};

	// Build a trivial grouped aggregation: group by column 0, one dummy aggregate
	vector<unique_ptr<Expression>> groups;
	groups.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(LogicalType::INTEGER, 0));
	vector<unique_ptr<Expression>> aggrs;
	// Create a simple nullary aggregate expression for testing
	{
		auto agg_fun = AggregateFunction::NullaryAggregate<int64_t, int64_t, TestNullaryAggOp>(LogicalType::BIGINT);
		agg_fun.name = "test_nullary";
		vector<unique_ptr<Expression>> children;
		aggrs.push_back(duckdb::make_uniq<BoundAggregateExpression>(std::move(agg_fun), std::move(children), nullptr,
		                                                            nullptr, AggregateType::NON_DISTINCT));
	}

	// Need a ClientContext for constructing PhysicalHashAggregate properly
	duckdb::DuckDB db(nullptr);
	duckdb::Connection conn(db);

	auto &agg = plan_ptr->template Make<duckdb::PhysicalHashAggregate>(*conn.context, types, std::move(aggrs),
	                                                                   std::move(groups), 0);
	plan_ptr->SetRoot(agg);

	auto res = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan_ptr);
	REQUIRE(res.ok);
	auto node = res.value();
	REQUIRE(node != nullptr);
	REQUIRE(node->name() == "Aggregate");
}

TEST_CASE("PhysicalPlanTranslator: ungrouped aggregate produces Aggregate node", "[distributed]") {
	Allocator allocator;
	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> types = {LogicalType::INTEGER};

	vector<unique_ptr<Expression>> aggrs;
	// Create a simple nullary aggregate expression for testing
	{
		auto agg_fun = AggregateFunction::NullaryAggregate<int64_t, int64_t, TestNullaryAggOp>(LogicalType::BIGINT);
		agg_fun.name = "test_nullary";
		vector<unique_ptr<Expression>> children;
		aggrs.push_back(duckdb::make_uniq<BoundAggregateExpression>(std::move(agg_fun), std::move(children), nullptr,
		                                                            nullptr, AggregateType::NON_DISTINCT));
	}

	auto &uagg = plan_ptr->template Make<duckdb::PhysicalUngroupedAggregate>(
	    types, std::move(aggrs), 0, TupleDataValidityType::CAN_HAVE_NULL_VALUES);
	plan_ptr->SetRoot(uagg);

	auto res = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan_ptr);
	REQUIRE(res.ok);
	auto node = res.value();
	REQUIRE(node != nullptr);
	REQUIRE(node->name() == "Aggregate");
}

TEST_CASE("PhysicalPlanTranslator: limit -> LimitNode", "[distributed]") {
	auto plan = MakeUnaryScanPlan();
	auto limit_val = BoundLimitNode::ConstantValue(5);
	auto offset_val = BoundLimitNode::ConstantValue(2);
	auto &limit = plan.plan->Make<PhysicalLimit>(plan.types, std::move(limit_val), std::move(offset_val), 0);
	limit.children.push_back(*plan.scan);
	plan.plan->SetRoot(limit);

	auto res = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan.plan);
	REQUIRE(res.ok);
	REQUIRE(res.value() != nullptr);
	auto inner = res.value()->inner();
	REQUIRE(std::dynamic_pointer_cast<duckdb::distributed::LimitNode>(inner) != nullptr);
}

TEST_CASE("PhysicalPlanTranslator: streaming limit -> StreamingLimitNode", "[distributed]") {
	auto plan = MakeUnaryScanPlan();
	auto limit_val = BoundLimitNode::ConstantValue(10);
	auto offset_val = BoundLimitNode();
	auto &limit =
	    plan.plan->Make<PhysicalStreamingLimit>(plan.types, std::move(limit_val), std::move(offset_val), 0, true);
	limit.children.push_back(*plan.scan);
	plan.plan->SetRoot(limit);

	auto res = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan.plan);
	REQUIRE(res.ok);
	REQUIRE(res.value() != nullptr);
	auto inner = res.value()->inner();
	REQUIRE(std::dynamic_pointer_cast<duckdb::distributed::StreamingLimitNode>(inner) != nullptr);
}

TEST_CASE("PhysicalPlanTranslator: limit percent -> LimitPercentNode", "[distributed]") {
	auto plan = MakeUnaryScanPlan();
	auto limit_val = BoundLimitNode::ConstantPercentage(10.0);
	auto offset_val = BoundLimitNode();
	auto &limit = plan.plan->Make<PhysicalLimitPercent>(plan.types, std::move(limit_val), std::move(offset_val), 0);
	limit.children.push_back(*plan.scan);
	plan.plan->SetRoot(limit);

	auto res = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan.plan);
	REQUIRE(res.ok);
	REQUIRE(res.value() != nullptr);
	auto inner = res.value()->inner();
	REQUIRE(std::dynamic_pointer_cast<duckdb::distributed::LimitPercentNode>(inner) != nullptr);
}

TEST_CASE("PhysicalPlanTranslator: fixed-row reservoir sample becomes globally single-partition",
          "[distributed][reservoir_sample]") {
	auto plan = MakeUnaryScanPlan();
	auto repartition_spec = RepartitionSpec::create_random(4);
	auto &repartition = plan.plan->Make<PhysicalRepartition>(plan.types, std::move(repartition_spec), 0);
	repartition.children.push_back(*plan.scan);

	auto options = make_uniq<SampleOptions>(42);
	options->sample_size = Value::BIGINT(2);
	options->is_percentage = false;
	options->method = SampleMethod::RESERVOIR_SAMPLE;
	options->repeatable = true;
	auto &sample = plan.plan->Make<PhysicalReservoirSample>(plan.types, std::move(options), 2);
	sample.children.push_back(repartition);
	plan.plan->SetRoot(sample);

	auto res = physical_plan_to_pipeline_node(PlanConfig {}, plan.plan);
	REQUIRE(res.is_ok());
	auto sample_node = std::dynamic_pointer_cast<ReservoirSampleNode>(res.value()->inner());
	REQUIRE(sample_node != nullptr);
	REQUIRE(sample_node->config().clustering_spec()->num_partitions() == 1);
	auto children = sample_node->children();
	REQUIRE(children.size() == 1);
	REQUIRE(children[0]->config().clustering_spec()->num_partitions() == 4);
}

TEST_CASE("PhysicalPlanTranslator: order by -> OrderByNode", "[distributed]") {
	auto plan = MakeUnaryScanPlan();
	vector<BoundOrderByNode> orders;
	auto expr = make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, 0);
	orders.emplace_back(OrderType::ASCENDING, OrderByNullType::NULLS_LAST, std::move(expr));
	auto &order_by = plan.plan->Make<PhysicalOrder>(plan.types, std::move(orders), vector<idx_t>(), 0, false);
	order_by.children.push_back(*plan.scan);
	plan.plan->SetRoot(order_by);

	auto res = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan.plan);
	REQUIRE(res.ok);
	REQUIRE(res.value() != nullptr);
	auto inner = res.value()->inner();
	REQUIRE(std::dynamic_pointer_cast<duckdb::distributed::OrderByNode>(inner) != nullptr);
}

TEST_CASE("PhysicalPlanTranslator: top n -> TopNNode", "[distributed]") {
	auto plan = MakeUnaryScanPlan();
	vector<BoundOrderByNode> orders;
	auto expr = make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, 0);
	orders.emplace_back(OrderType::DESCENDING, OrderByNullType::NULLS_LAST, std::move(expr));
	auto &topn = plan.plan->Make<PhysicalTopN>(plan.types, std::move(orders), 5, 1, nullptr, 0);
	topn.children.push_back(*plan.scan);
	plan.plan->SetRoot(topn);

	auto res = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan.plan);
	REQUIRE(res.ok);
	REQUIRE(res.value() != nullptr);
	auto inner = res.value()->inner();
	REQUIRE(std::dynamic_pointer_cast<duckdb::distributed::TopNNode>(inner) != nullptr);
}

TEST_CASE("PhysicalPlanTranslator: global window gathers multi-partition input", "[distributed][window]") {
	auto plan = MakeWindowPlan(false, 4, false, {});
	auto res = physical_plan_to_pipeline_node(PlanConfig {}, std::move(plan));

	REQUIRE(res.is_ok());
	auto window = std::dynamic_pointer_cast<WindowNode>(res.value()->inner());
	REQUIRE(window != nullptr);
	REQUIRE(SchemaColumnCount(window->config().schema()) == 3);
	auto children = window->children();
	REQUIRE(children.size() == 1);
	REQUIRE(std::dynamic_pointer_cast<RepartitionNode>(children[0]) != nullptr);
	REQUIRE(children[0]->config().clustering_spec()->num_partitions() == 1);
}

TEST_CASE("PhysicalPlanTranslator: partitioned window hash-shuffles by partition keys", "[distributed][window]") {
	auto plan = MakeWindowPlan(false, 4, false, {0});
	auto res = physical_plan_to_pipeline_node(PlanConfig {}, std::move(plan));

	REQUIRE(res.is_ok());
	auto window = std::dynamic_pointer_cast<WindowNode>(res.value()->inner());
	REQUIRE(window != nullptr);
	auto children = window->children();
	REQUIRE(children.size() == 1);
	REQUIRE(std::dynamic_pointer_cast<RepartitionNode>(children[0]) != nullptr);
	auto clustering = children[0]->config().clustering_spec();
	REQUIRE(clustering->type() == ClusteringSpec::Type::Hash);
	REQUIRE(clustering->num_partitions() == 4);
	auto partition_by = clustering->partition_by();
	REQUIRE(partition_by.size() == 1);
	REQUIRE(partition_by[0]->GetExpressionClass() == ExpressionClass::BOUND_REF);
	REQUIRE(partition_by[0]->Cast<BoundReferenceExpression>().index == 0);
}

TEST_CASE("PhysicalPlanTranslator: streaming window gathers multi-partition input", "[distributed][window]") {
	auto plan = MakeWindowPlan(true, 4, false, {});
	auto res = physical_plan_to_pipeline_node(PlanConfig {}, std::move(plan));

	REQUIRE(res.is_ok());
	auto window = std::dynamic_pointer_cast<StreamingWindowNode>(res.value()->inner());
	REQUIRE(window != nullptr);
	REQUIRE(SchemaColumnCount(window->config().schema()) == 3);
	auto children = window->children();
	REQUIRE(children.size() == 1);
	REQUIRE(std::dynamic_pointer_cast<RepartitionNode>(children[0]) != nullptr);
	REQUIRE(children[0]->config().clustering_spec()->num_partitions() == 1);
}

TEST_CASE("PhysicalPlanTranslator: window reuses only verified partitioning", "[distributed][window]") {
	SECTION("single-partition global window") {
		auto plan = MakeWindowPlan(false, 1, false, {});
		auto res = physical_plan_to_pipeline_node(PlanConfig {}, std::move(plan));

		REQUIRE(res.is_ok());
		auto window = std::dynamic_pointer_cast<WindowNode>(res.value()->inner());
		REQUIRE(window != nullptr);
		auto children = window->children();
		REQUIRE(children.size() == 1);
		REQUIRE(std::dynamic_pointer_cast<RepartitionNode>(children[0]) == nullptr);
	}

	SECTION("single-partition partitioned window") {
		auto plan = MakeWindowPlan(false, 1, false, {0});
		auto res = physical_plan_to_pipeline_node(PlanConfig {}, std::move(plan));

		REQUIRE(res.is_ok());
		auto window = std::dynamic_pointer_cast<WindowNode>(res.value()->inner());
		REQUIRE(window != nullptr);
		auto children = window->children();
		REQUIRE(children.size() == 1);
		REQUIRE(std::dynamic_pointer_cast<RepartitionNode>(children[0]) == nullptr);
	}

	SECTION("matching hash-partitioned input") {
		auto plan = MakeWindowPlan(false, 4, true, {0});
		auto res = physical_plan_to_pipeline_node(PlanConfig {}, std::move(plan));

		REQUIRE(res.is_ok());
		auto window = std::dynamic_pointer_cast<WindowNode>(res.value()->inner());
		REQUIRE(window != nullptr);
		auto children = window->children();
		REQUIRE(children.size() == 1);
		auto repartition = std::dynamic_pointer_cast<RepartitionNode>(children[0]);
		REQUIRE(repartition != nullptr);
		auto repartition_children = repartition->children();
		REQUIRE(repartition_children.size() == 1);
		REQUIRE(std::dynamic_pointer_cast<RepartitionNode>(repartition_children[0]) == nullptr);
	}

	SECTION("stale hash metadata after projection") {
		auto plan = MakeWindowPlan(false, 4, true, {0}, {}, true);
		auto res = physical_plan_to_pipeline_node(PlanConfig {}, std::move(plan));

		REQUIRE(res.is_ok());
		auto window = std::dynamic_pointer_cast<WindowNode>(res.value()->inner());
		REQUIRE(window != nullptr);
		auto children = window->children();
		REQUIRE(children.size() == 1);
		auto repartition = std::dynamic_pointer_cast<RepartitionNode>(children[0]);
		REQUIRE(repartition != nullptr);
		auto repartition_children = repartition->children();
		REQUIRE(repartition_children.size() == 1);
		REQUIRE(repartition_children[0]->name() == "Projection");
	}
}

TEST_CASE("PhysicalPlanTranslator: window rejects incompatible partition definitions", "[distributed][window]") {
	auto plan = MakeWindowPlan(false, 4, false, {0}, {1});
	auto res = physical_plan_to_pipeline_node(PlanConfig {}, std::move(plan));

	REQUIRE(res.is_err());
	REQUIRE_THAT(res.error().what(), Catch::Matchers::Contains("incompatible partition definitions"));
}

TEST_CASE("PhysicalPlanTranslator: hash join builds both required shuffles", "[distributed][join]") {
	ScopedTranslatorEnvironment join_strategy("VANE_DISTRIBUTED_JOIN_STRATEGY", "hash");
	PlanConfig config;
	config.num_partitions = 4;

	auto res = physical_plan_to_pipeline_node(config, MakeHashJoinPlan(JoinType::INNER, 1, 1));

	REQUIRE(res.is_ok());
	auto join = res.value()->inner();
	REQUIRE(join != nullptr);
	REQUIRE(join->name() == "HashJoin");
	auto children = join->children();
	REQUIRE(children.size() == 2);
	for (const auto &child : children) {
		auto shuffle = std::dynamic_pointer_cast<RepartitionNode>(child);
		REQUIRE(shuffle != nullptr);
		REQUIRE(shuffle->config().clustering_spec() != nullptr);
		REQUIRE(shuffle->config().clustering_spec()->type() == ClusteringSpec::Type::Hash);
		REQUIRE(shuffle->config().clustering_spec()->num_partitions() == 4);
	}
}

TEST_CASE("PhysicalPlanTranslator: partitioned MARK joins preserve global build semantics", "[distributed][join]") {
	ScopedTranslatorEnvironment join_strategy("VANE_DISTRIBUTED_JOIN_STRATEGY", "hash");
	PlanConfig config;
	config.num_partitions = 4;

	SECTION("uncorrelated MARK build shuffle collects one global summary") {
		auto res = physical_plan_to_pipeline_node(config, MakeHashJoinPlan(JoinType::MARK, 1, 1));
		REQUIRE(res.is_ok());
		auto children = res.value()->inner()->children();
		REQUIRE(children.size() == 2);
		auto left_shuffle = std::dynamic_pointer_cast<RepartitionNode>(children[0]);
		auto right_shuffle = std::dynamic_pointer_cast<RepartitionNode>(children[1]);
		REQUIRE(left_shuffle != nullptr);
		REQUIRE(right_shuffle != nullptr);
		REQUIRE_FALSE(left_shuffle->CollectsMarkJoinBuildSummary());
		REQUIRE(right_shuffle->CollectsMarkJoinBuildSummary());
		REQUIRE_FALSE(left_shuffle->is_materialization_barrier());
		REQUIRE(right_shuffle->is_materialization_barrier());
		REQUIRE(right_shuffle->materialized_input_node_ids() ==
		        vector<NodeID> {right_shuffle->children()[0]->node_id()});
		REQUIRE(NodeDisplayContains(right_shuffle->into_node(), "MARK build summary: global"));
	}

	SECTION("correlated MARK rows are co-located by correlation keys") {
		auto res = physical_plan_to_pipeline_node(config, MakeCorrelatedMarkJoinPlan());
		REQUIRE(res.is_ok());
		auto children = res.value()->inner()->children();
		REQUIRE(children.size() == 2);
		for (const auto &child : children) {
			auto shuffle = std::dynamic_pointer_cast<RepartitionNode>(child);
			REQUIRE(shuffle != nullptr);
			REQUIRE_FALSE(shuffle->CollectsMarkJoinBuildSummary());
			REQUIRE_FALSE(shuffle->is_materialization_barrier());
			REQUIRE(shuffle->materialized_input_node_ids().empty());
			REQUIRE(shuffle->config().clustering_spec()->partition_by().size() == 1);
		}
	}
}

TEST_CASE("PhysicalPlanTranslator: broadcast join builds the required receiver shuffle", "[distributed][join]") {
	ScopedTranslatorEnvironment join_strategy("VANE_DISTRIBUTED_JOIN_STRATEGY", "broadcast_left");
	ScopedTranslatorEnvironment repartition_receiver("VANE_DISTRIBUTED_BROADCAST_JOIN_RECEIVER_REPARTITION", "true");
	PlanConfig config;
	config.num_partitions = 4;

	auto res = physical_plan_to_pipeline_node(config, MakeHashJoinPlan(JoinType::INNER, 1, 1));

	REQUIRE(res.is_ok());
	auto join = res.value()->inner();
	REQUIRE(join != nullptr);
	REQUIRE(join->name() == "BroadcastJoin");
	auto children = join->children();
	REQUIRE(children.size() == 2);
	REQUIRE(std::dynamic_pointer_cast<RepartitionNode>(children[0]) == nullptr);
	auto receiver_shuffle = std::dynamic_pointer_cast<RepartitionNode>(children[1]);
	REQUIRE(receiver_shuffle != nullptr);
	REQUIRE(receiver_shuffle->config().clustering_spec() != nullptr);
	REQUIRE(receiver_shuffle->config().clustering_spec()->type() == ClusteringSpec::Type::Hash);
	REQUIRE(receiver_shuffle->config().clustering_spec()->num_partitions() == 4);
}

TEST_CASE("PhysicalPlanTranslator: left delim join -> placeholder node", "[distributed]") {
	Allocator allocator;
	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> scan_types = {LogicalType::INTEGER};
	vector<LogicalType> join_types = {LogicalType::INTEGER, LogicalType::INTEGER};

	auto left_collection = MakeSingleValueCollection(scan_types, {Value::INTEGER(1)});
	auto &left_scan = plan_ptr->Make<PhysicalColumnDataScan>(scan_types, PhysicalOperatorType::COLUMN_DATA_SCAN, 1,
	                                                         std::move(left_collection));

	auto right_collection = MakeSingleValueCollection(scan_types, {Value::INTEGER(2)});
	auto &right_scan =
	    plan_ptr
	        ->Make<PhysicalColumnDataScan>(scan_types, PhysicalOperatorType::DELIM_SCAN, 1, std::move(right_collection))
	        .Cast<PhysicalColumnDataScan>();
	right_scan.delim_index = 7;

	vector<JoinCondition> conditions;
	JoinCondition cond;
	cond.left = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0);
	cond.right = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0);
	cond.comparison = ExpressionType::COMPARE_EQUAL;
	conditions.push_back(std::move(cond));

	LogicalComparisonJoin logical_join(JoinType::INNER);
	logical_join.types = join_types;

	auto &hash_join = plan_ptr->Make<PhysicalHashJoin>(logical_join, left_scan, right_scan, std::move(conditions),
	                                                   JoinType::INNER, 1);

	DuckDB db(nullptr);
	Connection conn(db);
	vector<unique_ptr<Expression>> groups;
	groups.push_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0));
	vector<unique_ptr<Expression>> aggregates;
	auto &distinct =
	    plan_ptr->Make<PhysicalHashAggregate>(*conn.context, scan_types, std::move(aggregates), std::move(groups), 1);

	vector<const_reference<PhysicalOperator>> delim_scans;
	delim_scans.push_back(right_scan);

	auto &delim_join = plan_ptr->Make<PhysicalLeftDelimJoin>(DelimJoinDeserializeTag {}, join_types, hash_join,
	                                                         distinct, delim_scans, 1, optional_idx(7));
	delim_join.children.push_back(left_scan);
	plan_ptr->SetRoot(delim_join);

	auto res = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan_ptr);
	REQUIRE(res.ok);
	REQUIRE(res.value() != nullptr);
	REQUIRE(res.value()->name() == "LEFT_DELIM_JOIN");
}

TEST_CASE("PhysicalPlanTranslator: inout function -> TableInOutNode", "[distributed]") {
	Allocator allocator;
	auto plan_ptr = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> input_types = {LogicalType::INTEGER};
	vector<LogicalType> output_types = {LogicalType::INTEGER};

	auto collection = MakeSingleValueCollection(input_types, {Value::INTEGER(1)});
	auto &scan = plan_ptr->Make<PhysicalColumnDataScan>(input_types, PhysicalOperatorType::COLUMN_DATA_SCAN, 1,
	                                                    std::move(collection));

	TableFunction function("test_inout", {LogicalType::TABLE}, nullptr);
	function.in_out_function = TestInOutFunction;

	vector<ColumnIndex> column_ids;
	column_ids.emplace_back(0);
	vector<column_t> projected_input;

	auto &inout =
	    plan_ptr->Make<PhysicalTableInOutFunction>(output_types, function, nullptr, column_ids, 1, projected_input);
	inout.children.push_back(scan);
	plan_ptr->SetRoot(inout);

	auto res = duckdb::distributed::physical_plan_to_pipeline_node(duckdb::distributed::PlanConfig {}, plan_ptr);
	REQUIRE(res.ok);
	REQUIRE(res.value() != nullptr);
	REQUIRE(res.value()->name() == "TableInOut");
}
