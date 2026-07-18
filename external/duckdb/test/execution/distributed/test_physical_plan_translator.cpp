// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB Distributed Execution
//
// test_physical_plan_translator.cpp
//
// 单元测试：DuckDB 物理计划到分布式流水线节点的转换
//===----------------------------------------------------------------------===//

#include "catch.hpp"

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/helper/physical_limit.hpp"
#include "duckdb/execution/operator/helper/physical_limit_percent.hpp"
#include "duckdb/execution/operator/helper/physical_streaming_limit.hpp"
#include "duckdb/execution/operator/order/physical_order.hpp"
#include "duckdb/execution/operator/order/physical_top_n.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/execution/operator/scan/physical_dummy_scan.hpp"
#include "duckdb/execution/operator/scan/physical_expression_scan.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
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
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/join/physical_left_delim_join.hpp"
#include "duckdb/execution/operator/join/physical_right_delim_join.hpp"
#include "duckdb/execution/operator/projection/physical_tableinout_function.hpp"

#include "duckdb/main/connection.hpp"
#include "duckdb/execution/distributed/pipeline_node/translator.hpp"
#include "duckdb/execution/distributed/pipeline_node/aggregate.hpp"
#include "duckdb/execution/distributed/pipeline_node/limit.hpp"
#include "duckdb/execution/distributed/pipeline_node/scan_source.hpp"
#include "duckdb/execution/distributed/pipeline_node/expression_scan.hpp"
#include "duckdb/execution/distributed/pipeline_node/sort.hpp"

// Include distributed pipeline translator headers (lightweight declarations)
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "test_helpers.hpp"

#include <memory>
#include <utility>
#include <cstdlib>

using namespace duckdb;
using namespace duckdb::distributed;

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
	if (!schema) {
		return 0;
	}
	if (schema->id() == LogicalTypeId::STRUCT) {
		return StructType::GetChildTypes(*schema).size();
	}
	return 1;
}

static std::string SQLStringLiteral(const std::string &value) {
	return "'" + StringUtil::Replace(value, "'", "''") + "'";
}

TEST_CASE("PhysicalPlanTranslator: simple projection", "[distributed]") {
	// 构造一个简单的物理计划: TableScan -> Projection
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
	// ...检查类型和警告...
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
	// ...检查类型和警告...
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

// 此测试已被上面两个更详细的测试覆盖，可移除或重写为更复杂的计划测试

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
