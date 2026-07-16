// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB Distributed Execution
//
// test_physical_plan_serialization.cpp
//
// Unit tests: PhysicalOperator and PhysicalPlan serialization/deserialization
//===----------------------------------------------------------------------===//

#include "catch.hpp"

#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/execution/physical_plan.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/helper/physical_limit.hpp"
#include "duckdb/execution/operator/helper/physical_streaming_limit.hpp"
#include "duckdb/execution/operator/helper/physical_limit_percent.hpp"
#include "duckdb/execution/operator/order/physical_order.hpp"
#include "duckdb/execution/operator/order/physical_top_n.hpp"
#include "duckdb/execution/operator/exchange/physical_remote_exchange_sink.hpp"
#include "duckdb/execution/operator/exchange/physical_remote_exchange_source.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_ungrouped_aggregate.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/projection/physical_tableinout_function.hpp"
#include "duckdb/execution/distributed/exchange/flight_exchange_manager.hpp"
#include "duckdb/execution/distributed/plan/exchange_source_task.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/common/constants.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/common/enums/order_type.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/allocator.hpp"

#include <memory>
#include <iostream>

using namespace duckdb;

namespace {

unique_ptr<BoundAggregateExpression> MakeCountAggregate(ClientContext &context, idx_t column_index,
                                                        const LogicalType &input_type) {
	auto &func_entry =
	    Catalog::GetEntry<AggregateFunctionCatalogEntry>(context, SYSTEM_CATALOG, DEFAULT_SCHEMA, "count");
	auto bound_function = func_entry.functions.GetFunctionByArguments(context, {input_type});

	vector<unique_ptr<Expression>> children;
	children.push_back(make_uniq<BoundReferenceExpression>(input_type, column_index));

	FunctionBinder binder(context);
	return binder.BindAggregateFunction(std::move(bound_function), std::move(children), nullptr,
	                                    AggregateType::NON_DISTINCT);
}

PhysicalColumnDataScan &MakeColumnDataScan(PhysicalPlan &plan, const vector<LogicalType> &types) {
	auto &op =
	    plan.Make<PhysicalColumnDataScan>(types, PhysicalOperatorType::COLUMN_DATA_SCAN, 0, DConstants::INVALID_INDEX);
	return op.Cast<PhysicalColumnDataScan>();
}

struct TestInOutBindData : public TableFunctionData {
	idx_t marker = 0;

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<TestInOutBindData>();
		copy->marker = marker;
		return std::move(copy);
	}

	bool Equals(const FunctionData &other) const override {
		return marker == other.Cast<TestInOutBindData>().marker;
	}
};

unique_ptr<FunctionData> TestInOutBind(ClientContext &, TableFunctionBindInput &, vector<LogicalType> &return_types,
                                       vector<string> &names) {
	return_types.emplace_back(LogicalType::INTEGER);
	names.emplace_back("value");
	auto data = make_uniq<TestInOutBindData>();
	data->marker = 7;
	return std::move(data);
}

OperatorResultType TestInOutFunction(ExecutionContext &, TableFunctionInput &, DataChunk &, DataChunk &output) {
	output.SetCardinality(0);
	return OperatorResultType::NEED_MORE_INPUT;
}

void TestInOutSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data, const TableFunction &) {
	idx_t marker = 0;
	if (bind_data) {
		marker = bind_data->Cast<TestInOutBindData>().marker;
	}
	serializer.WriteProperty(100, "marker", marker);
}

unique_ptr<FunctionData> TestInOutDeserialize(Deserializer &deserializer, TableFunction &) {
	auto marker = deserializer.ReadProperty<idx_t>(100, "marker");
	auto data = make_uniq<TestInOutBindData>();
	data->marker = marker;
	return std::move(data);
}

TableFunction MakeTestInOutFunction() {
	TableFunction func("test_inout_serialization", {LogicalType::TABLE}, nullptr, TestInOutBind);
	func.in_out_function = TestInOutFunction;
	func.serialize = TestInOutSerialize;
	func.deserialize = TestInOutDeserialize;
	return func;
}

} // namespace

TEST_CASE("PhysicalProjection serialization roundtrip", "[serialization][physical_plan]") {
	// Create an allocator for the physical plan
	Allocator allocator;
	PhysicalPlan plan(allocator);

	// Create projection expressions: just reference column 0
	vector<unique_ptr<Expression>> select_list;
	select_list.push_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0));
	select_list.push_back(make_uniq<BoundConstantExpression>(Value::INTEGER(42)));

	// Return types
	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::INTEGER};
	idx_t estimated_cardinality = 1000;

	// Create the projection operator
	auto &projection = plan.Make<PhysicalProjection>(types, std::move(select_list), estimated_cardinality);

	// Verify the projection was created correctly
	REQUIRE(projection.type == PhysicalOperatorType::PROJECTION);
	REQUIRE(projection.types.size() == 2);
	REQUIRE(projection.estimated_cardinality == 1000);

	// Serialize the projection
	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);

	serializer.Begin();
	projection.Serialize(serializer);
	serializer.End();

	auto serialized_size = stream.GetPosition();
	std::cerr << "[test] Serialized PhysicalProjection size: " << serialized_size << " bytes" << std::endl;
	REQUIRE(serialized_size > 0);

	// Rewind and deserialize using base class dispatcher
	stream.Rewind();
	BinaryDeserializer deserializer(stream);

	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);

	// Verify the deserialized projection
	auto *proj_ptr = dynamic_cast<PhysicalProjection *>(deserialized_op.get());
	REQUIRE(proj_ptr != nullptr);
	REQUIRE(proj_ptr->type == PhysicalOperatorType::PROJECTION);
	REQUIRE(proj_ptr->types.size() == 2);
	REQUIRE(proj_ptr->types[0] == LogicalType::INTEGER);
	REQUIRE(proj_ptr->types[1] == LogicalType::INTEGER);
	REQUIRE(proj_ptr->estimated_cardinality == 1000);
	REQUIRE(proj_ptr->select_list.size() == 2);

	std::cerr << "[test] PhysicalProjection serialization roundtrip PASSED" << std::endl;
}

TEST_CASE("PhysicalFilter serialization roundtrip", "[serialization][physical_plan]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	// Create a filter expression: column 0 > 10
	auto col_ref = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0);
	auto constant = make_uniq<BoundConstantExpression>(Value::INTEGER(10));
	auto filter_expr = make_uniq<BoundComparisonExpression>(ExpressionType::COMPARE_GREATERTHAN, std::move(col_ref),
	                                                        std::move(constant));

	// Wrap in vector for PhysicalFilter constructor
	vector<unique_ptr<Expression>> filter_list;
	filter_list.push_back(std::move(filter_expr));

	// Return types
	vector<LogicalType> types = {LogicalType::INTEGER};
	idx_t estimated_cardinality = 500;

	// Create the filter operator
	auto &filter = plan.Make<PhysicalFilter>(types, std::move(filter_list), estimated_cardinality);

	REQUIRE(filter.type == PhysicalOperatorType::FILTER);
	REQUIRE(filter.types.size() == 1);

	// Serialize
	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);

	serializer.Begin();
	filter.Serialize(serializer);
	serializer.End();

	auto serialized_size = stream.GetPosition();
	std::cerr << "[test] Serialized PhysicalFilter size: " << serialized_size << " bytes" << std::endl;
	REQUIRE(serialized_size > 0);

	// Deserialize using base class dispatcher
	stream.Rewind();
	BinaryDeserializer deserializer(stream);

	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);

	auto *filter_ptr = dynamic_cast<PhysicalFilter *>(deserialized_op.get());
	REQUIRE(filter_ptr != nullptr);
	REQUIRE(filter_ptr->type == PhysicalOperatorType::FILTER);
	REQUIRE(filter_ptr->types.size() == 1);
	REQUIRE(filter_ptr->estimated_cardinality == 500);

	std::cerr << "[test] PhysicalFilter serialization roundtrip PASSED" << std::endl;
}

TEST_CASE("PhysicalLimit serialization roundtrip", "[serialization][physical_plan]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<LogicalType> types = {LogicalType::INTEGER};
	auto limit_val = BoundLimitNode::ConstantValue(5);
	auto offset_val = BoundLimitNode::ConstantValue(2);

	auto &limit = plan.Make<PhysicalLimit>(types, std::move(limit_val), std::move(offset_val), 100);

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);

	serializer.Begin();
	limit.Serialize(serializer);
	serializer.End();

	auto serialized_size = stream.GetPosition();
	std::cerr << "[test] Serialized PhysicalLimit size: " << serialized_size << " bytes" << std::endl;
	REQUIRE(serialized_size > 0);

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);
	auto *limit_ptr = dynamic_cast<PhysicalLimit *>(deserialized_op.get());
	REQUIRE(limit_ptr != nullptr);
	REQUIRE(limit_ptr->type == PhysicalOperatorType::LIMIT);
	REQUIRE(limit_ptr->types.size() == 1);
	REQUIRE(limit_ptr->limit_val.Type() == LimitNodeType::CONSTANT_VALUE);
	REQUIRE(limit_ptr->limit_val.GetConstantValue() == 5);
	REQUIRE(limit_ptr->offset_val.Type() == LimitNodeType::CONSTANT_VALUE);
	REQUIRE(limit_ptr->offset_val.GetConstantValue() == 2);

	std::cerr << "[test] PhysicalLimit serialization roundtrip PASSED" << std::endl;
}

TEST_CASE("PhysicalLimit serialization roundtrip (expression)", "[serialization][physical_plan]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<LogicalType> types = {LogicalType::INTEGER};
	auto expr = make_uniq<BoundConstantExpression>(Value::INTEGER(3));
	auto limit_val = BoundLimitNode::ExpressionValue(std::move(expr));
	auto offset_val = BoundLimitNode::ConstantValue(1);

	auto &limit = plan.Make<PhysicalLimit>(types, std::move(limit_val), std::move(offset_val), 100);

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);

	serializer.Begin();
	limit.Serialize(serializer);
	serializer.End();

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);
	auto *limit_ptr = dynamic_cast<PhysicalLimit *>(deserialized_op.get());
	REQUIRE(limit_ptr != nullptr);
	REQUIRE(limit_ptr->limit_val.Type() == LimitNodeType::EXPRESSION_VALUE);
	REQUIRE(limit_ptr->limit_val.GetValueExpression().GetExpressionClass() == ExpressionClass::BOUND_CONSTANT);
	REQUIRE(limit_ptr->offset_val.Type() == LimitNodeType::CONSTANT_VALUE);
	REQUIRE(limit_ptr->offset_val.GetConstantValue() == 1);

	std::cerr << "[test] PhysicalLimit serialization roundtrip (expression) PASSED" << std::endl;
}

TEST_CASE("PhysicalStreamingLimit serialization roundtrip", "[serialization][physical_plan]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<LogicalType> types = {LogicalType::INTEGER};
	auto limit_val = BoundLimitNode::ConstantValue(7);
	auto offset_val = BoundLimitNode::ConstantValue(1);

	auto &limit = plan.Make<PhysicalStreamingLimit>(types, std::move(limit_val), std::move(offset_val), 50, true);

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);

	serializer.Begin();
	limit.Serialize(serializer);
	serializer.End();

	auto serialized_size = stream.GetPosition();
	std::cerr << "[test] Serialized PhysicalStreamingLimit size: " << serialized_size << " bytes" << std::endl;
	REQUIRE(serialized_size > 0);

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);
	auto *limit_ptr = dynamic_cast<PhysicalStreamingLimit *>(deserialized_op.get());
	REQUIRE(limit_ptr != nullptr);
	REQUIRE(limit_ptr->type == PhysicalOperatorType::STREAMING_LIMIT);
	REQUIRE(limit_ptr->types.size() == 1);
	REQUIRE(limit_ptr->limit_val.Type() == LimitNodeType::CONSTANT_VALUE);
	REQUIRE(limit_ptr->limit_val.GetConstantValue() == 7);
	REQUIRE(limit_ptr->offset_val.Type() == LimitNodeType::CONSTANT_VALUE);
	REQUIRE(limit_ptr->offset_val.GetConstantValue() == 1);
	REQUIRE(limit_ptr->parallel);

	std::cerr << "[test] PhysicalStreamingLimit serialization roundtrip PASSED" << std::endl;
}

TEST_CASE("PhysicalLimitPercent serialization roundtrip", "[serialization][physical_plan]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<LogicalType> types = {LogicalType::INTEGER};
	auto limit_val = BoundLimitNode::ConstantPercentage(12.5);
	auto offset_val = BoundLimitNode::ConstantValue(0);

	auto &limit = plan.Make<PhysicalLimitPercent>(types, std::move(limit_val), std::move(offset_val), 40);

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);

	serializer.Begin();
	limit.Serialize(serializer);
	serializer.End();

	auto serialized_size = stream.GetPosition();
	std::cerr << "[test] Serialized PhysicalLimitPercent size: " << serialized_size << " bytes" << std::endl;
	REQUIRE(serialized_size > 0);

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);
	auto *limit_ptr = dynamic_cast<PhysicalLimitPercent *>(deserialized_op.get());
	REQUIRE(limit_ptr != nullptr);
	REQUIRE(limit_ptr->type == PhysicalOperatorType::LIMIT_PERCENT);
	REQUIRE(limit_ptr->types.size() == 1);
	REQUIRE(limit_ptr->limit_val.Type() == LimitNodeType::CONSTANT_PERCENTAGE);
	REQUIRE(limit_ptr->limit_val.GetConstantPercentage() == Approx(12.5));
	REQUIRE(limit_ptr->offset_val.Type() == LimitNodeType::CONSTANT_VALUE);
	REQUIRE(limit_ptr->offset_val.GetConstantValue() == 0);

	std::cerr << "[test] PhysicalLimitPercent serialization roundtrip PASSED" << std::endl;
}

TEST_CASE("PhysicalLimitPercent serialization roundtrip (expression)", "[serialization][physical_plan]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<LogicalType> types = {LogicalType::INTEGER};
	auto expr = make_uniq<BoundConstantExpression>(Value::DOUBLE(12.5));
	auto limit_val = BoundLimitNode::ExpressionPercentage(std::move(expr));
	auto offset_val = BoundLimitNode();

	auto &limit = plan.Make<PhysicalLimitPercent>(types, std::move(limit_val), std::move(offset_val), 40);

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);

	serializer.Begin();
	limit.Serialize(serializer);
	serializer.End();

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);
	auto *limit_ptr = dynamic_cast<PhysicalLimitPercent *>(deserialized_op.get());
	REQUIRE(limit_ptr != nullptr);
	REQUIRE(limit_ptr->limit_val.Type() == LimitNodeType::EXPRESSION_PERCENTAGE);
	REQUIRE(limit_ptr->limit_val.GetPercentageExpression().GetExpressionClass() == ExpressionClass::BOUND_CONSTANT);

	std::cerr << "[test] PhysicalLimitPercent serialization roundtrip (expression) PASSED" << std::endl;
}

TEST_CASE("PhysicalOrder serialization roundtrip", "[serialization][physical_plan]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<LogicalType> types = {LogicalType::INTEGER};
	vector<BoundOrderByNode> orders;
	auto expr = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0);
	orders.emplace_back(OrderType::ASCENDING, OrderByNullType::NULLS_LAST, std::move(expr));

	auto &order_by = plan.Make<PhysicalOrder>(types, std::move(orders), vector<idx_t>(), 10, false);

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);

	serializer.Begin();
	order_by.Serialize(serializer);
	serializer.End();

	auto serialized_size = stream.GetPosition();
	std::cerr << "[test] Serialized PhysicalOrder size: " << serialized_size << " bytes" << std::endl;
	REQUIRE(serialized_size > 0);

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);
	auto *order_ptr = dynamic_cast<PhysicalOrder *>(deserialized_op.get());
	REQUIRE(order_ptr != nullptr);
	REQUIRE(order_ptr->type == PhysicalOperatorType::ORDER_BY);
	REQUIRE(order_ptr->orders.size() == 1);
	REQUIRE(order_ptr->orders[0].type == OrderType::ASCENDING);
	REQUIRE(order_ptr->orders[0].null_order == OrderByNullType::NULLS_LAST);
	REQUIRE(order_ptr->orders[0].expression != nullptr);
	REQUIRE(order_ptr->orders[0].expression->GetExpressionClass() == ExpressionClass::BOUND_REF);

	std::cerr << "[test] PhysicalOrder serialization roundtrip PASSED" << std::endl;
}

TEST_CASE("PhysicalOrder serialization roundtrip (expression)", "[serialization][physical_plan]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<LogicalType> types = {LogicalType::INTEGER};
	vector<BoundOrderByNode> orders;
	auto left = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0);
	auto right = make_uniq<BoundConstantExpression>(Value::INTEGER(10));
	auto cmp =
	    make_uniq<BoundComparisonExpression>(ExpressionType::COMPARE_GREATERTHAN, std::move(left), std::move(right));
	orders.emplace_back(OrderType::ASCENDING, OrderByNullType::NULLS_LAST, std::move(cmp));

	auto &order_by = plan.Make<PhysicalOrder>(types, std::move(orders), vector<idx_t>(), 10, false);

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);

	serializer.Begin();
	order_by.Serialize(serializer);
	serializer.End();

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);
	auto *order_ptr = dynamic_cast<PhysicalOrder *>(deserialized_op.get());
	REQUIRE(order_ptr != nullptr);
	REQUIRE(order_ptr->orders.size() == 1);
	REQUIRE(order_ptr->orders[0].expression != nullptr);
	REQUIRE(order_ptr->orders[0].expression->GetExpressionClass() == ExpressionClass::BOUND_COMPARISON);

	std::cerr << "[test] PhysicalOrder serialization roundtrip (expression) PASSED" << std::endl;
}

TEST_CASE("PhysicalTopN serialization roundtrip", "[serialization][physical_plan]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<LogicalType> types = {LogicalType::INTEGER};
	vector<BoundOrderByNode> orders;
	auto expr = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0);
	orders.emplace_back(OrderType::DESCENDING, OrderByNullType::NULLS_LAST, std::move(expr));

	auto &top_n = plan.Make<PhysicalTopN>(types, std::move(orders), 5, 2, nullptr, 20);

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);

	serializer.Begin();
	top_n.Serialize(serializer);
	serializer.End();

	auto serialized_size = stream.GetPosition();
	std::cerr << "[test] Serialized PhysicalTopN size: " << serialized_size << " bytes" << std::endl;
	REQUIRE(serialized_size > 0);

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);
	auto *top_ptr = dynamic_cast<PhysicalTopN *>(deserialized_op.get());
	REQUIRE(top_ptr != nullptr);
	REQUIRE(top_ptr->type == PhysicalOperatorType::TOP_N);
	REQUIRE(top_ptr->orders.size() == 1);
	REQUIRE(top_ptr->orders[0].type == OrderType::DESCENDING);
	REQUIRE(top_ptr->orders[0].null_order == OrderByNullType::NULLS_LAST);
	REQUIRE(top_ptr->limit == 5);
	REQUIRE(top_ptr->offset == 2);

	std::cerr << "[test] PhysicalTopN serialization roundtrip PASSED" << std::endl;
}

TEST_CASE("PhysicalPlan tree: Limit -> OrderBy -> ColumnDataScan", "[serialization][physical_plan]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<LogicalType> types = {LogicalType::INTEGER};
	auto &scan = MakeColumnDataScan(plan, types);

	vector<BoundOrderByNode> orders;
	auto expr = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0);
	orders.emplace_back(OrderType::ASCENDING, OrderByNullType::NULLS_LAST, std::move(expr));
	auto &order_by = plan.Make<PhysicalOrder>(types, std::move(orders), vector<idx_t>(), 10, false);
	order_by.children.push_back(scan);

	auto limit_val = BoundLimitNode::ConstantValue(5);
	auto offset_val = BoundLimitNode::ConstantValue(0);
	auto &limit = plan.Make<PhysicalLimit>(types, std::move(limit_val), std::move(offset_val), 10);
	limit.children.push_back(order_by);
	plan.SetRoot(limit);

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);
	serializer.Begin();
	plan.Serialize(serializer);
	serializer.End();

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	PhysicalPlan deserialized_plan(allocator);
	deserializer.Begin();
	auto root_op = deserialized_plan.Deserialize(deserializer);
	deserializer.End();

	REQUIRE(root_op != nullptr);
	REQUIRE(root_op->type == PhysicalOperatorType::LIMIT);
	REQUIRE(root_op->children.size() == 1);
	REQUIRE(root_op->children[0].get().type == PhysicalOperatorType::ORDER_BY);
	REQUIRE(root_op->children[0].get().children.size() == 1);
	REQUIRE(root_op->children[0].get().children[0].get().type == PhysicalOperatorType::COLUMN_DATA_SCAN);

	std::cerr << "[test] PhysicalPlan tree Limit->OrderBy roundtrip PASSED" << std::endl;
}

TEST_CASE("PhysicalPlan tree: StreamingLimit -> ColumnDataScan", "[serialization][physical_plan]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<LogicalType> types = {LogicalType::INTEGER};
	auto &scan = MakeColumnDataScan(plan, types);

	auto limit_val = BoundLimitNode::ConstantValue(7);
	auto offset_val = BoundLimitNode::ConstantValue(1);
	auto &limit = plan.Make<PhysicalStreamingLimit>(types, std::move(limit_val), std::move(offset_val), 10, false);
	limit.children.push_back(scan);
	plan.SetRoot(limit);

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);
	serializer.Begin();
	plan.Serialize(serializer);
	serializer.End();

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	PhysicalPlan deserialized_plan(allocator);
	deserializer.Begin();
	auto root_op = deserialized_plan.Deserialize(deserializer);
	deserializer.End();

	REQUIRE(root_op != nullptr);
	REQUIRE(root_op->type == PhysicalOperatorType::STREAMING_LIMIT);
	REQUIRE(root_op->children.size() == 1);
	REQUIRE(root_op->children[0].get().type == PhysicalOperatorType::COLUMN_DATA_SCAN);

	std::cerr << "[test] PhysicalPlan tree StreamingLimit roundtrip PASSED" << std::endl;
}

TEST_CASE("PhysicalPlan tree: LimitPercent -> ColumnDataScan", "[serialization][physical_plan]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<LogicalType> types = {LogicalType::INTEGER};
	auto &scan = MakeColumnDataScan(plan, types);

	auto limit_val = BoundLimitNode::ConstantPercentage(5.0);
	auto offset_val = BoundLimitNode();
	auto &limit = plan.Make<PhysicalLimitPercent>(types, std::move(limit_val), std::move(offset_val), 10);
	limit.children.push_back(scan);
	plan.SetRoot(limit);

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);
	serializer.Begin();
	plan.Serialize(serializer);
	serializer.End();

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	PhysicalPlan deserialized_plan(allocator);
	deserializer.Begin();
	auto root_op = deserialized_plan.Deserialize(deserializer);
	deserializer.End();

	REQUIRE(root_op != nullptr);
	REQUIRE(root_op->type == PhysicalOperatorType::LIMIT_PERCENT);
	REQUIRE(root_op->children.size() == 1);
	REQUIRE(root_op->children[0].get().type == PhysicalOperatorType::COLUMN_DATA_SCAN);

	std::cerr << "[test] PhysicalPlan tree LimitPercent roundtrip PASSED" << std::endl;
}

TEST_CASE("PhysicalPlan tree: TopN -> ColumnDataScan", "[serialization][physical_plan]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<LogicalType> types = {LogicalType::INTEGER};
	auto &scan = MakeColumnDataScan(plan, types);

	vector<BoundOrderByNode> orders;
	auto expr = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0);
	orders.emplace_back(OrderType::DESCENDING, OrderByNullType::NULLS_LAST, std::move(expr));
	auto &top_n = plan.Make<PhysicalTopN>(types, std::move(orders), 5, 2, nullptr, 20);
	top_n.children.push_back(scan);
	plan.SetRoot(top_n);

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);
	serializer.Begin();
	plan.Serialize(serializer);
	serializer.End();

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	PhysicalPlan deserialized_plan(allocator);
	deserializer.Begin();
	auto root_op = deserialized_plan.Deserialize(deserializer);
	deserializer.End();

	REQUIRE(root_op != nullptr);
	REQUIRE(root_op->type == PhysicalOperatorType::TOP_N);
	REQUIRE(root_op->children.size() == 1);
	REQUIRE(root_op->children[0].get().type == PhysicalOperatorType::COLUMN_DATA_SCAN);

	std::cerr << "[test] PhysicalPlan tree TopN roundtrip PASSED" << std::endl;
}

TEST_CASE("PhysicalHashJoin serialization roundtrip", "[serialization][physical_plan]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<LogicalType> left_types = {LogicalType::INTEGER};
	vector<LogicalType> right_types = {LogicalType::INTEGER};

	auto &left_scan = MakeColumnDataScan(plan, left_types);
	auto &right_scan = MakeColumnDataScan(plan, right_types);

	vector<JoinCondition> conditions;
	JoinCondition cond;
	cond.left = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0);
	cond.right = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0);
	cond.comparison = ExpressionType::COMPARE_EQUAL;
	conditions.push_back(std::move(cond));

	LogicalComparisonJoin logical_join(JoinType::INNER);
	logical_join.types = {LogicalType::INTEGER, LogicalType::INTEGER};

	vector<idx_t> left_projection_map;
	vector<idx_t> right_projection_map;
	vector<LogicalType> delim_types;
	auto pushdown_info = make_uniq<JoinFilterPushdownInfo>();
	pushdown_info->join_condition.push_back(0);
	pushdown_info->min_max_aggregates.push_back(make_uniq<BoundConstantExpression>(Value::INTEGER(1)));
	JoinFilterPushdownFilter pushdown_filter;
	pushdown_filter.dynamic_filters = make_shared_ptr<DynamicTableFilterSet>();
	JoinFilterPushdownColumn pushdown_column;
	pushdown_column.probe_column_index = ColumnBinding(0, 0);
	pushdown_filter.columns.push_back(pushdown_column);
	pushdown_info->probe_info.push_back(std::move(pushdown_filter));

	idx_t estimated_cardinality = 100;
	auto &hash_join = plan.Make<PhysicalHashJoin>(
	    logical_join, left_scan, right_scan, std::move(conditions), JoinType::INNER, left_projection_map,
	    right_projection_map, std::move(delim_types), estimated_cardinality, std::move(pushdown_info));

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);

	serializer.Begin();
	hash_join.Serialize(serializer);
	serializer.End();

	auto serialized_size = stream.GetPosition();
	std::cerr << "[test] Serialized PhysicalHashJoin size: " << serialized_size << " bytes" << std::endl;
	REQUIRE(serialized_size > 0);

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);
	auto *join_ptr = dynamic_cast<PhysicalHashJoin *>(deserialized_op.get());
	REQUIRE(join_ptr != nullptr);
	REQUIRE(join_ptr->join_type == JoinType::INNER);
	REQUIRE(join_ptr->conditions.size() == 1);
	REQUIRE(join_ptr->condition_types.size() == 1);
	REQUIRE(join_ptr->condition_types[0] == LogicalType::INTEGER);
	REQUIRE(join_ptr->lhs_output_columns.col_idxs.size() == 1);
	REQUIRE(join_ptr->lhs_output_columns.col_types.size() == 1);
	REQUIRE(join_ptr->rhs_output_columns.col_idxs.size() == 1);
	REQUIRE(join_ptr->rhs_output_columns.col_types.size() == 1);
	REQUIRE(join_ptr->payload_columns.col_types.empty());
	REQUIRE(join_ptr->filter_pushdown != nullptr);
	REQUIRE(join_ptr->filter_pushdown->join_condition.size() == 1);
	REQUIRE(join_ptr->filter_pushdown->probe_info.size() == 1);
	REQUIRE(join_ptr->filter_pushdown->probe_info[0].columns.size() == 1);
	REQUIRE(join_ptr->filter_pushdown->probe_info[0].dynamic_filters != nullptr);
	REQUIRE(join_ptr->filter_pushdown->min_max_aggregates.size() == 1);

	std::cerr << "[test] PhysicalHashJoin serialization roundtrip PASSED" << std::endl;
}

TEST_CASE("PhysicalUngroupedAggregate serialization roundtrip", "[serialization][physical_plan]") {
	DuckDB db(nullptr);
	Connection conn(db);
	conn.BeginTransaction();
	auto &context = *conn.context;

	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<unique_ptr<Expression>> aggregates;
	aggregates.push_back(MakeCountAggregate(context, 0, LogicalType::INTEGER));

	vector<LogicalType> types;
	types.push_back(aggregates[0]->return_type);

	idx_t estimated_cardinality = 10;
	auto &uagg = plan.Make<PhysicalUngroupedAggregate>(types, std::move(aggregates), estimated_cardinality,
	                                                   TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);

	serializer.Begin();
	uagg.Serialize(serializer);
	serializer.End();

	auto serialized_size = stream.GetPosition();
	std::cerr << "[test] Serialized PhysicalUngroupedAggregate size: " << serialized_size << " bytes" << std::endl;
	REQUIRE(serialized_size > 0);

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Set<ClientContext &>(context);

	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);
	auto *uagg_ptr = dynamic_cast<PhysicalUngroupedAggregate *>(deserialized_op.get());
	REQUIRE(uagg_ptr != nullptr);
	REQUIRE(uagg_ptr->type == PhysicalOperatorType::UNGROUPED_AGGREGATE);
	REQUIRE(uagg_ptr->types.size() == 1);
	REQUIRE(uagg_ptr->types[0] == types[0]);
	REQUIRE(uagg_ptr->estimated_cardinality == estimated_cardinality);
	REQUIRE(uagg_ptr->aggregates.size() == 1);
	REQUIRE(uagg_ptr->aggregates[0]->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE);
	REQUIRE(uagg_ptr->distinct_validity == TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);

	std::cerr << "[test] PhysicalUngroupedAggregate serialization roundtrip PASSED" << std::endl;
	conn.Rollback();
}

TEST_CASE("PhysicalHashAggregate serialization roundtrip", "[serialization][physical_plan]") {
	DuckDB db(nullptr);
	Connection conn(db);
	conn.BeginTransaction();
	auto &context = *conn.context;

	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<unique_ptr<Expression>> groups;
	groups.push_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0));
	LogicalType group_type = groups[0]->return_type;

	vector<unique_ptr<Expression>> aggregates;
	aggregates.push_back(MakeCountAggregate(context, 1, LogicalType::INTEGER));
	LogicalType aggregate_type = aggregates[0]->return_type;

	vector<LogicalType> types = {group_type, aggregate_type};
	idx_t estimated_cardinality = 42;

	auto &hash_agg = plan.Make<PhysicalHashAggregate>(context, types, std::move(aggregates), std::move(groups),
	                                                  estimated_cardinality);

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);

	serializer.Begin();
	hash_agg.Serialize(serializer);
	serializer.End();

	auto serialized_size = stream.GetPosition();
	std::cerr << "[test] Serialized PhysicalHashAggregate size: " << serialized_size << " bytes" << std::endl;
	REQUIRE(serialized_size > 0);

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Set<ClientContext &>(context);

	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);
	auto *hash_ptr = dynamic_cast<PhysicalHashAggregate *>(deserialized_op.get());
	REQUIRE(hash_ptr != nullptr);
	REQUIRE(hash_ptr->type == PhysicalOperatorType::HASH_GROUP_BY);
	REQUIRE(hash_ptr->types.size() == 2);
	REQUIRE(hash_ptr->types[0] == group_type);
	REQUIRE(hash_ptr->types[1] == aggregate_type);
	REQUIRE(hash_ptr->estimated_cardinality == estimated_cardinality);
	REQUIRE(hash_ptr->grouped_aggregate_data.groups.size() == 1);
	REQUIRE(hash_ptr->grouped_aggregate_data.aggregates.size() == 1);
	REQUIRE(hash_ptr->grouped_aggregate_data.groups[0]->GetExpressionClass() == ExpressionClass::BOUND_REF);
	REQUIRE(hash_ptr->grouped_aggregate_data.aggregates[0]->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE);

	std::cerr << "[test] PhysicalHashAggregate serialization roundtrip PASSED" << std::endl;
	conn.Rollback();
}

TEST_CASE("PhysicalPlan with chain: Projection -> Filter", "[serialization][physical_plan]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	// First create the filter (this will be the child)
	auto col_ref = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0);
	auto constant = make_uniq<BoundConstantExpression>(Value::INTEGER(10));
	auto filter_expr = make_uniq<BoundComparisonExpression>(ExpressionType::COMPARE_GREATERTHAN, std::move(col_ref),
	                                                        std::move(constant));

	// Wrap in vector for PhysicalFilter constructor
	vector<unique_ptr<Expression>> filter_list;
	filter_list.push_back(std::move(filter_expr));

	vector<LogicalType> filter_types = {LogicalType::INTEGER, LogicalType::VARCHAR};
	auto &filter = plan.Make<PhysicalFilter>(filter_types, std::move(filter_list), 500);

	// Create projection (parent of filter)
	vector<unique_ptr<Expression>> select_list;
	select_list.push_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0));

	vector<LogicalType> proj_types = {LogicalType::INTEGER};
	auto &projection = plan.Make<PhysicalProjection>(proj_types, std::move(select_list), 500);

	// Link: Projection -> Filter
	projection.children.push_back(filter);

	// Set root
	plan.SetRoot(projection);

	REQUIRE(plan.HasRoot());
	REQUIRE(plan.Root().type == PhysicalOperatorType::PROJECTION);
	REQUIRE(plan.Root().children.size() == 1);
	REQUIRE(plan.Root().children[0].get().type == PhysicalOperatorType::FILTER);

	std::cerr << "[test] PhysicalPlan chain created successfully" << std::endl;

	// =========================================================================
	// Full plan serialization - base Serialize handles tree traversal automatically
	// =========================================================================
	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);

	serializer.Begin();
	// Serialize the entire tree in one call - delegate to plan Serialize helper
	plan.Serialize(serializer);
	serializer.End();

	auto total_serialized_size = stream.GetPosition();
	std::cerr << "[test] Serialized full plan tree size: " << total_serialized_size << " bytes" << std::endl;
	REQUIRE(total_serialized_size > 0);

	// =========================================================================
	// Full plan deserialization - base Deserialize handles tree traversal automatically
	// =========================================================================
	stream.Rewind();
	BinaryDeserializer deserializer(stream);

	PhysicalPlan deserialized_plan(allocator);

	deserializer.Begin();
	// Deserialize the entire tree in one call - delegate to plan Deserialize helper
	auto root_op = deserialized_plan.Deserialize(deserializer);
	deserializer.End();

	// Note: root_op is a unique_ptr, we need to transfer ownership properly
	// For now, we'll verify the structure directly on root_op
	REQUIRE(root_op != nullptr);
	REQUIRE(root_op->type == PhysicalOperatorType::PROJECTION);
	REQUIRE(root_op->children.size() == 1);
	REQUIRE(root_op->children[0].get().type == PhysicalOperatorType::FILTER);

	// Verify projection details
	auto &deser_proj = root_op->Cast<PhysicalProjection>();
	REQUIRE(deser_proj.types.size() == 1);
	REQUIRE(deser_proj.types[0] == LogicalType::INTEGER);
	REQUIRE(deser_proj.estimated_cardinality == 500);
	REQUIRE(deser_proj.select_list.size() == 1);

	// Verify filter details
	auto &deser_filter = root_op->children[0].get().Cast<PhysicalFilter>();
	REQUIRE(deser_filter.types.size() == 2);
	REQUIRE(deser_filter.types[0] == LogicalType::INTEGER);
	REQUIRE(deser_filter.types[1] == LogicalType::VARCHAR);
	REQUIRE(deser_filter.estimated_cardinality == 500);
	REQUIRE(deser_filter.expression != nullptr);

	std::cerr << "[test] Full PhysicalPlan tree serialization roundtrip PASSED" << std::endl;
}

TEST_CASE("PhysicalColumnDataScan serialization roundtrip", "[serialization][physical_plan]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	// Create ColumnDataScan with no collection (CTE index = INVALID)
	vector<LogicalType> types = {LogicalType::INTEGER};
	idx_t estimated_cardinality = 0;

	auto &scan = plan.Make<PhysicalColumnDataScan>(types, PhysicalOperatorType::COLUMN_DATA_SCAN, estimated_cardinality,
	                                               DConstants::INVALID_INDEX);

	REQUIRE(scan.type == PhysicalOperatorType::COLUMN_DATA_SCAN);
	REQUIRE(scan.types.size() == 1);
	// The Make() helper returns a reference to PhysicalOperator; cast to the concrete type to access specific fields
	auto *orig_scan_ptr = dynamic_cast<PhysicalColumnDataScan *>(&scan);
	REQUIRE(orig_scan_ptr != nullptr);
	REQUIRE(orig_scan_ptr->cte_index == DConstants::INVALID_INDEX);

	// Serialize
	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);
	serializer.Begin();
	scan.Serialize(serializer);
	serializer.End();

	auto serialized_size = stream.GetPosition();
	std::cerr << "[test] Serialized PhysicalColumnDataScan size: " << serialized_size << " bytes" << std::endl;
	REQUIRE(serialized_size > 0);

	// Deserialize
	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);
	auto *scan_ptr = dynamic_cast<PhysicalColumnDataScan *>(deserialized_op.get());
	REQUIRE(scan_ptr != nullptr);
	REQUIRE(scan_ptr->type == PhysicalOperatorType::COLUMN_DATA_SCAN);
	REQUIRE(scan_ptr->types.size() == 1);
	REQUIRE(scan_ptr->cte_index == DConstants::INVALID_INDEX);
	REQUIRE(!scan_ptr->collection);

	std::cerr << "[test] PhysicalColumnDataScan serialization roundtrip PASSED" << std::endl;
}

TEST_CASE("PhysicalTableInOutFunction serialization roundtrip", "[serialization][physical_plan]") {
	DuckDB db(nullptr);
	Connection conn(db);
	conn.BeginTransaction();
	auto &context = *conn.context;

	auto &catalog = Catalog::GetSystemCatalog(context);
	auto func = MakeTestInOutFunction();
	CreateTableFunctionInfo info(func);
	catalog.CreateTableFunction(context, info);

	auto &entry = Catalog::GetEntry<TableFunctionCatalogEntry>(context, SYSTEM_CATALOG, DEFAULT_SCHEMA,
	                                                           "test_inout_serialization");
	auto table_func = entry.functions.GetFunctionByArguments(context, {LogicalType::TABLE});

	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<LogicalType> types = {LogicalType::INTEGER};
	vector<ColumnIndex> column_ids;
	column_ids.emplace_back(0);

	auto bind_data = make_uniq<TestInOutBindData>();
	bind_data->marker = 123;
	vector<column_t> projected_input;
	idx_t estimated_cardinality = 11;

	auto &inout = plan.Make<PhysicalTableInOutFunction>(types, table_func, std::move(bind_data), column_ids,
	                                                    estimated_cardinality, projected_input);
	auto &inout_ref = inout.Cast<PhysicalTableInOutFunction>();
	inout_ref.ordinality_idx = optional_idx(0);

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);
	serializer.Begin();
	inout.Serialize(serializer);
	serializer.End();

	auto serialized_size = stream.GetPosition();
	std::cerr << "[test] Serialized PhysicalTableInOutFunction size: " << serialized_size << " bytes" << std::endl;
	REQUIRE(serialized_size > 0);

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Set<ClientContext &>(context);

	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);
	auto *inout_ptr = dynamic_cast<PhysicalTableInOutFunction *>(deserialized_op.get());
	REQUIRE(inout_ptr != nullptr);
	REQUIRE(inout_ptr->type == PhysicalOperatorType::INOUT_FUNCTION);
	REQUIRE(inout_ptr->types.size() == 1);
	REQUIRE(inout_ptr->types[0] == LogicalType::INTEGER);
	REQUIRE(inout_ptr->estimated_cardinality == estimated_cardinality);
	REQUIRE(inout_ptr->ordinality_idx.IsValid());
	REQUIRE(inout_ptr->ordinality_idx.GetIndex() == 0);
	auto params = inout_ptr->ParamsToString();
	auto name_it = params.find("Name");
	REQUIRE(name_it != params.end());
	REQUIRE(name_it->second == "test_inout_serialization");

	std::cerr << "[test] PhysicalTableInOutFunction serialization roundtrip PASSED" << std::endl;
	conn.Rollback();
}

TEST_CASE("PhysicalRemoteExchangeSink serialization preserves sink instance metadata",
          "[serialization][physical_plan][exchange]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<LogicalType> types = {LogicalType::INTEGER};
	distributed::ExchangeSinkInstanceHandle sink_handle;
	sink_handle.sink_handle.task_partition_id = 7;
	sink_handle.attempt_id = 2;
	sink_handle.output_location = "shuffle_stage__sink_7__attempt_2";
	sink_handle.output_partition_count = 4;

	distributed::FlightExchangeConfig flight_config;
	flight_config.node_id = "node-1";
	auto exchange_mgr = std::make_shared<distributed::FlightExchangeManager>(std::move(flight_config));

	vector<unique_ptr<Expression>> partition_by;
	auto &sink = plan.Make<PhysicalRemoteExchangeSink>(types, 123, "shuffle_stage", 4, RepartitionSpec::Type::Random,
	                                                   std::move(partition_by), sink_handle, exchange_mgr);

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);
	serializer.Begin();
	sink.Serialize(serializer);
	serializer.End();

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);
	auto *sink_ptr = dynamic_cast<PhysicalRemoteExchangeSink *>(deserialized_op.get());
	REQUIRE(sink_ptr != nullptr);
	REQUIRE(sink_ptr->ExchangeId() == "shuffle_stage");
	REQUIRE(sink_ptr->NumPartitions() == 4);
	REQUIRE(sink_ptr->SinkHandle().sink_handle.task_partition_id == 7);
	REQUIRE(sink_ptr->SinkHandle().attempt_id == 2);
	REQUIRE(sink_ptr->SinkHandle().output_location == "shuffle_stage__sink_7__attempt_2");
	REQUIRE(sink_ptr->SinkHandle().output_partition_count == 4);
}

TEST_CASE("PhysicalRemoteExchangeSource serialization preserves explicit source handles",
          "[serialization][physical_plan][exchange]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<LogicalType> types = {LogicalType::INTEGER};
	vector<idx_t> partition_indices = {0, 1};
	vector<string> source_nodes = {"node-1", "node-2"};
	std::vector<distributed::ExchangeSourceHandle> source_handles;

	distributed::ExchangeSourceHandle handle0;
	handle0.partition_id = 0;
	handle0.attempt_id = 3;
	handle0.node_id = "node-1";
	handle0.files.push_back(ExchangeSourceFile("shuffle_stage__sink_0__attempt_0", 0));
	source_handles.push_back(handle0);

	distributed::ExchangeSourceHandle handle1;
	handle1.partition_id = 0;
	handle1.attempt_id = 4;
	handle1.node_id = "node-2";
	handle1.files.push_back(ExchangeSourceFile("shuffle_stage__sink_1__attempt_0", 0));
	source_handles.push_back(handle1);

	distributed::ExchangeSourceHandle handle2;
	handle2.partition_id = 1;
	handle2.attempt_id = 3;
	handle2.node_id = "node-1";
	handle2.files.push_back(ExchangeSourceFile("shuffle_stage__sink_0__attempt_0", 0));
	source_handles.push_back(handle2);

	distributed::FlightExchangeConfig flight_config;
	flight_config.node_id = "node-1";
	auto exchange_mgr = std::make_shared<distributed::FlightExchangeManager>(std::move(flight_config));

	auto &source = plan.Make<PhysicalRemoteExchangeSource>(types, 456, "shuffle_stage", partition_indices,
	                                                       source_handles, exchange_mgr, source_nodes);

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);
	serializer.Begin();
	source.Serialize(serializer);
	serializer.End();

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);
	auto *source_ptr = dynamic_cast<PhysicalRemoteExchangeSource *>(deserialized_op.get());
	REQUIRE(source_ptr != nullptr);
	REQUIRE(source_ptr->ExchangeId() == "shuffle_stage");
	REQUIRE(source_ptr->PartitionIndices() == partition_indices);
	REQUIRE(source_ptr->SourceNodes() == source_nodes);
	REQUIRE(source_ptr->SourceHandles().size() == source_handles.size());
	REQUIRE(source_ptr->SourceHandles()[0].partition_id == 0);
	REQUIRE(source_ptr->SourceHandles()[0].attempt_id == 3);
	REQUIRE(source_ptr->SourceHandles()[0].node_id == "node-1");
	REQUIRE(source_ptr->SourceHandles()[0].files.size() == 1);
	REQUIRE(source_ptr->SourceHandles()[0].files[0].path == "shuffle_stage__sink_0__attempt_0");
	REQUIRE(source_ptr->SourceHandles()[1].partition_id == 0);
	REQUIRE(source_ptr->SourceHandles()[1].attempt_id == 4);
	REQUIRE(source_ptr->SourceHandles()[1].node_id == "node-2");
	REQUIRE(source_ptr->SourceHandles()[1].files.size() == 1);
	REQUIRE(source_ptr->SourceHandles()[1].files[0].path == "shuffle_stage__sink_1__attempt_0");
	REQUIRE(source_ptr->SourceHandles()[2].partition_id == 1);
	REQUIRE(source_ptr->SourceHandles()[2].attempt_id == 3);
	REQUIRE(source_ptr->SourceHandles()[2].node_id == "node-1");
	REQUIRE(source_ptr->SourceHandles()[2].files.size() == 1);
	REQUIRE(source_ptr->SourceHandles()[2].files[0].path == "shuffle_stage__sink_0__attempt_0");
}

TEST_CASE("PhysicalRemoteExchangeSource serialization preserves runtime source binding node id",
          "[serialization][physical_plan][exchange]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<LogicalType> types = {LogicalType::INTEGER};
	vector<string> source_nodes = {"node-1", "node-2"};

	distributed::FlightExchangeConfig flight_config;
	flight_config.node_id = "node-1";
	auto exchange_mgr = std::make_shared<distributed::FlightExchangeManager>(std::move(flight_config));

	auto &source_op = plan.Make<PhysicalRemoteExchangeSource>(types, 456, "shuffle_stage", vector<idx_t>(),
	                                                          std::vector<distributed::ExchangeSourceHandle>(),
	                                                          exchange_mgr, source_nodes, optional_idx(42));
	auto &source = dynamic_cast<PhysicalRemoteExchangeSource &>(source_op);

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);
	serializer.Begin();
	source.Serialize(serializer);
	serializer.End();

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);
	auto *source_ptr = dynamic_cast<PhysicalRemoteExchangeSource *>(deserialized_op.get());
	REQUIRE(source_ptr != nullptr);
	REQUIRE(source_ptr->ExchangeId() == "shuffle_stage");
	REQUIRE(source_ptr->PartitionIndices().empty());
	REQUIRE(source_ptr->SourceHandles().empty());
	REQUIRE(source_ptr->RuntimeSourceNodeId().IsValid());
	REQUIRE(source_ptr->RuntimeSourceNodeId().GetIndex() == 42);
}

TEST_CASE("ExchangeSourceTaskDescriptor serialization preserves source handle attempt ids",
          "[serialization][physical_plan][exchange]") {
	distributed::ExchangeSourceTaskDescriptor descriptor;
	descriptor.partition_indices = {0, 1};
	descriptor.source_partition_count = 2;
	descriptor.source_task_count = 2;

	distributed::ExchangeSourceHandle handle0;
	handle0.partition_id = 0;
	handle0.attempt_id = 7;
	handle0.node_id = "node-1";
	handle0.flight_port = 5010;
	handle0.files.push_back(ExchangeSourceFile("shuffle_stage__sink_0__attempt_7", 11));
	descriptor.source_handles.push_back(handle0);

	distributed::ExchangeSourceHandle handle1;
	handle1.partition_id = 1;
	handle1.attempt_id = 2;
	handle1.node_id = "node-2";
	handle1.flight_port = 5011;
	handle1.files.push_back(ExchangeSourceFile("shuffle_stage__sink_1__attempt_2", 17));
	descriptor.source_handles.push_back(handle1);

	auto roundtrip = distributed::ExchangeSourceTaskDescriptor::DeserializeFromBytes(descriptor.SerializeToBytes());

	REQUIRE(roundtrip.partition_indices == descriptor.partition_indices);
	REQUIRE(roundtrip.source_partition_count == 2);
	REQUIRE(roundtrip.source_task_count == 2);
	REQUIRE(roundtrip.source_handles.size() == 2);
	REQUIRE(roundtrip.source_handles[0].partition_id == 0);
	REQUIRE(roundtrip.source_handles[0].attempt_id == 7);
	REQUIRE(roundtrip.source_handles[0].node_id == "node-1");
	REQUIRE(roundtrip.source_handles[0].flight_port == 5010);
	REQUIRE(roundtrip.source_handles[0].files.size() == 1);
	REQUIRE(roundtrip.source_handles[0].files[0].path == "shuffle_stage__sink_0__attempt_7");
	REQUIRE(roundtrip.source_handles[0].files[0].file_size == 11);
	REQUIRE(roundtrip.source_handles[1].partition_id == 1);
	REQUIRE(roundtrip.source_handles[1].attempt_id == 2);
	REQUIRE(roundtrip.source_handles[1].node_id == "node-2");
	REQUIRE(roundtrip.source_handles[1].flight_port == 5011);
	REQUIRE(roundtrip.source_handles[1].files.size() == 1);
	REQUIRE(roundtrip.source_handles[1].files[0].path == "shuffle_stage__sink_1__attempt_2");
	REQUIRE(roundtrip.source_handles[1].files[0].file_size == 17);
}

TEST_CASE("ApplyExchangeSourceTasksToPlan patches runtime-bound exchange source",
          "[serialization][physical_plan][exchange]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<LogicalType> types = {LogicalType::INTEGER};
	vector<string> source_nodes = {"node-1", "node-2"};

	distributed::FlightExchangeConfig flight_config;
	flight_config.node_id = "node-1";
	auto exchange_mgr = std::make_shared<distributed::FlightExchangeManager>(std::move(flight_config));

	auto &source_op = plan.Make<PhysicalRemoteExchangeSource>(types, 456, "shuffle_stage", vector<idx_t>(),
	                                                          std::vector<distributed::ExchangeSourceHandle>(),
	                                                          exchange_mgr, source_nodes, optional_idx(42));
	auto &source = dynamic_cast<PhysicalRemoteExchangeSource &>(source_op);
	plan.SetRoot(source);

	distributed::ExchangeSourceTaskDescriptor descriptor;
	descriptor.partition_indices = {0, 1};
	distributed::ExchangeSourceHandle handle0;
	handle0.partition_id = 0;
	handle0.attempt_id = 5;
	handle0.node_id = "node-1";
	handle0.files.push_back(ExchangeSourceFile("shuffle_stage__sink_0__attempt_0", 11));
	descriptor.source_handles.push_back(handle0);
	distributed::ExchangeSourceHandle handle1;
	handle1.partition_id = 1;
	handle1.attempt_id = 6;
	handle1.node_id = "node-2";
	handle1.files.push_back(ExchangeSourceFile("shuffle_stage__sink_1__attempt_0", 17));
	descriptor.source_handles.push_back(handle1);

	std::unordered_map<idx_t, distributed::ExchangeSourceTaskDescriptor> tasks;
	tasks.emplace(42, descriptor);

	string error;
	REQUIRE(distributed::ApplyExchangeSourceTasksToPlan(plan, tasks, &error));
	REQUIRE(error.empty());
	REQUIRE(source.PartitionIndices() == descriptor.partition_indices);
	REQUIRE(source.SourceHandles().size() == descriptor.source_handles.size());
	REQUIRE(source.SourceHandles()[0].partition_id == 0);
	REQUIRE(source.SourceHandles()[0].attempt_id == 5);
	REQUIRE(source.SourceHandles()[0].node_id == "node-1");
	REQUIRE(source.SourceHandles()[0].files.size() == 1);
	REQUIRE(source.SourceHandles()[0].files[0].path == "shuffle_stage__sink_0__attempt_0");
	REQUIRE(source.SourceHandles()[0].files[0].file_size == 11);
	REQUIRE(source.SourceHandles()[1].partition_id == 1);
	REQUIRE(source.SourceHandles()[1].attempt_id == 6);
	REQUIRE(source.SourceHandles()[1].node_id == "node-2");
	REQUIRE(source.SourceHandles()[1].files.size() == 1);
	REQUIRE(source.SourceHandles()[1].files[0].path == "shuffle_stage__sink_1__attempt_0");
	REQUIRE(source.SourceHandles()[1].files[0].file_size == 17);
}

TEST_CASE("Empty PhysicalPlan", "[serialization][physical_plan]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	// Empty plan should not have a root
	REQUIRE_FALSE(plan.HasRoot());

	std::cerr << "[test] Empty PhysicalPlan test PASSED" << std::endl;
}

TEST_CASE("PhysicalPlan SetRoot and get Root", "[serialization][physical_plan]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	// Create a simple projection
	vector<unique_ptr<Expression>> select_list;
	select_list.push_back(make_uniq<BoundConstantExpression>(Value::INTEGER(1)));

	vector<LogicalType> types = {LogicalType::INTEGER};
	auto &projection = plan.Make<PhysicalProjection>(types, std::move(select_list), 100);

	// Before setting root
	REQUIRE_FALSE(plan.HasRoot());

	// Set root
	plan.SetRoot(projection);

	// After setting root
	REQUIRE(plan.HasRoot());
	REQUIRE(&plan.Root() == &projection);
	REQUIRE(plan.Root().type == PhysicalOperatorType::PROJECTION);

	std::cerr << "[test] PhysicalPlan SetRoot test PASSED" << std::endl;
}
