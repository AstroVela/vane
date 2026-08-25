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
#include "duckdb/execution/operator/projection/physical_grouping_set_expand.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/helper/physical_limit.hpp"
#include "duckdb/execution/operator/helper/physical_streaming_limit.hpp"
#include "duckdb/execution/operator/helper/physical_limit_percent.hpp"
#include "duckdb/execution/operator/helper/physical_distributed_reservoir_sample.hpp"
#include "duckdb/execution/operator/order/physical_order.hpp"
#include "duckdb/execution/operator/order/physical_top_n.hpp"
#include "duckdb/execution/operator/exchange/physical_remote_exchange_sink.hpp"
#include "duckdb/execution/operator/exchange/physical_remote_exchange_source.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/execution/operator/scan/physical_empty_result.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_ungrouped_aggregate.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/projection/physical_tableinout_function.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/distributed/exchange/flight_exchange_manager.hpp"
#include "duckdb/execution/distributed/plan/exchange_sink_instance_task.hpp"
#include "duckdb/execution/distributed/plan/exchange_source_task.hpp"
#include "duckdb/execution/distributed/plan/fte_split_queue.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/common/constants.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/common/enums/order_type.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/allocator.hpp"
#include "duckdb/execution/reservoir_sample.hpp"

#include <algorithm>
#include <memory>
#include <iostream>
#include <unordered_set>

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

unique_ptr<FunctionData> TestContextSettingDeserialize(Deserializer &deserializer, TableFunction &) {
	auto marker = deserializer.ReadProperty<idx_t>(100, "marker");
	auto &context = deserializer.Get<ClientContext &>();
	Value threads;
	if (!context.TryGetCurrentSetting("threads", threads) || threads.GetValue<idx_t>() != 3) {
		marker = 0;
	}
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

TableFunction MakeContextSettingTestInOutFunction() {
	TableFunction func("test_inout_context_setting", {LogicalType::TABLE}, nullptr, TestInOutBind);
	func.in_out_function = TestInOutFunction;
	func.serialize = TestInOutSerialize;
	func.deserialize = TestContextSettingDeserialize;
	return func;
}

void SerializePreStrictRemoteExchangeSink(Serializer &serializer) {
	vector<LogicalType> types = {LogicalType::INTEGER};
	vector<unique_ptr<Expression>> partition_by;
	vector<string> local_dirs = {"/legacy/local"};
	vector<string> range_boundaries;
	vector<string> range_order_modifiers;
	serializer.WriteProperty(100, "type", PhysicalOperatorType::EXCHANGE_SINK);
	serializer.WriteProperty(101, "types", types);
	serializer.WriteProperty<idx_t>(102, "estimated_cardinality", 0);
	serializer.WriteProperty(103, "exchange_id", string("legacy-sink"));
	serializer.WriteProperty(104, "node_id", string("legacy-node"));
	serializer.WriteProperty<idx_t>(105, "num_partitions", 1);
	serializer.WriteProperty<uint8_t>(106, "repartition_type", static_cast<uint8_t>(RepartitionSpec::Type::Random));
	serializer.WriteProperty(107, "partition_by", partition_by);
	serializer.WriteProperty(108, "local_dirs", local_dirs);
	serializer.WriteProperty(109, "flight_bind_host", string("0.0.0.0"));
	serializer.WriteProperty<int>(110, "flight_port", 31337);
	serializer.WriteProperty<idx_t>(111, "sink_task_partition_id", 0);
	serializer.WriteProperty<idx_t>(112, "sink_attempt_id", 0);
	serializer.WriteProperty(113, "sink_output_location", string("legacy-sink-attempt"));
	serializer.WriteProperty(114, "range_boundaries", range_boundaries);
	serializer.WriteProperty(115, "range_order_modifiers", range_order_modifiers);
	serializer.WriteProperty(116, "flight_server_epoch", string());
	serializer.WriteProperty(117, "query_id", string("legacy-query"));
	serializer.WriteProperty(118, "allow_insecure_flight", true);
	serializer.WriteList(198, "children", 0, [](Serializer::List &, idx_t) {});
}

void SerializePreStrictRemoteExchangeSource(Serializer &serializer) {
	vector<LogicalType> types = {LogicalType::INTEGER};
	vector<idx_t> partition_indices = {0};
	vector<string> source_nodes = {"legacy-node"};
	vector<idx_t> handle_partition_ids = {0};
	vector<string> handle_node_ids = {"legacy-node"};
	vector<string> handle_paths = {"legacy-source-attempt"};
	vector<int> handle_flight_ports = {31337};
	vector<idx_t> handle_attempt_ids = {1};
	vector<string> local_dirs = {"/legacy/local"};
	vector<string> handle_server_epochs = {"legacy-epoch"};
	serializer.WriteProperty(100, "type", PhysicalOperatorType::EXCHANGE_SOURCE);
	serializer.WriteProperty(101, "types", types);
	serializer.WriteProperty<idx_t>(102, "estimated_cardinality", 0);
	serializer.WriteProperty(103, "exchange_id", string("legacy-source"));
	serializer.WriteProperty(104, "partition_indices", partition_indices);
	serializer.WriteProperty(105, "source_nodes", source_nodes);
	serializer.WriteProperty(106, "flight_location_template", string("grpc://{node}:31337"));
	serializer.WriteProperty<double>(107, "flight_timeout_seconds", 0);
	serializer.WriteProperty(108, "source_handle_partition_ids", handle_partition_ids);
	serializer.WriteProperty(109, "source_handle_node_ids", handle_node_ids);
	serializer.WriteProperty(110, "source_handle_paths", handle_paths);
	serializer.WriteProperty(111, "source_handle_flight_ports", handle_flight_ports);
	serializer.WritePropertyWithDefault(112, "runtime_source_node_id", optional_idx(), optional_idx());
	serializer.WriteProperty(113, "source_handle_attempt_ids", handle_attempt_ids);
	serializer.WriteProperty(114, "local_dirs", local_dirs);
	serializer.WriteProperty(115, "source_handle_flight_server_epochs", handle_server_epochs);
	serializer.WriteProperty(116, "source_catalog_handles_explicit", true);
	serializer.WriteProperty(117, "allow_insecure_flight", true);
	serializer.WriteList(198, "children", 0, [](Serializer::List &, idx_t) {});
}

string SerializeSinkDescriptorWithoutFlightHost() {
	MemoryStream stream(Allocator::DefaultAllocator());
	BinarySerializer serializer(stream);
	serializer.Begin();
	serializer.WriteProperty<idx_t>(1, "task_partition_id", 0);
	serializer.WriteProperty<idx_t>(2, "attempt_id", 0);
	serializer.WriteProperty(3, "output_location", string("pre-strict-sink"));
	serializer.WriteProperty<idx_t>(4, "output_partition_count", 1);
	serializer.WriteProperty(5, "flight_server_epoch", string("pre-strict-epoch"));
	serializer.WriteProperty(6, "query_id", string("pre-strict-query"));
	serializer.End();
	return string(reinterpret_cast<const char *>(stream.GetData()), stream.GetPosition());
}

string SerializeSourceDescriptorWithoutFlightHost() {
	MemoryStream stream(Allocator::DefaultAllocator());
	BinarySerializer serializer(stream);
	serializer.Begin();
	serializer.WriteProperty(1, "partition_indices", vector<idx_t> {0});
	serializer.WriteList(2, "source_handles", 1, [&](Serializer::List &list, idx_t) {
		list.WriteObject([&](Serializer &obj) {
			obj.WriteProperty<idx_t>(1, "partition_id", 0);
			obj.WriteProperty(2, "node_id", string("pre-strict-node"));
			obj.WriteProperty<int>(3, "flight_port", 5010);
			obj.WriteList(4, "files", 1, [&](Serializer::List &files, idx_t) {
				files.WriteObject([&](Serializer &file_obj) {
					file_obj.WriteProperty(1, "path", string("pre-strict-source"));
					file_obj.WriteProperty<uint64_t>(2, "file_size", 0);
					file_obj.WriteProperty<idx_t>(3, "rows", 0);
				});
			});
			obj.WriteProperty<idx_t>(5, "attempt_id", 0);
			obj.WriteProperty(6, "flight_server_epoch", string("pre-strict-epoch"));
		});
	});
	serializer.WriteProperty<idx_t>(3, "source_partition_count", 1);
	serializer.WriteProperty<idx_t>(4, "source_task_count", 1);
	serializer.WriteProperty<bool>(5, "replicated", false);
	serializer.End();
	return string(reinterpret_cast<const char *>(stream.GetData()), stream.GetPosition());
}

void AddReservoirRows(ReservoirSample &sample, Allocator &allocator, idx_t begin, idx_t end) {
	DataChunk chunk;
	chunk.Initialize(allocator, {LogicalType::BIGINT, LogicalType::VARCHAR});
	for (idx_t offset = begin; offset < end;) {
		chunk.Reset();
		const auto count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, end - offset);
		for (idx_t row_idx = 0; row_idx < count; row_idx++) {
			const auto value = offset + row_idx;
			chunk.SetValue(0, row_idx, Value::BIGINT(NumericCast<int64_t>(value)));
			chunk.SetValue(1, row_idx, Value("value-" + std::to_string(value)));
		}
		chunk.SetCardinality(count);
		sample.AddToReservoir(chunk);
		offset += count;
	}
}

unique_ptr<BlockingSample> RoundTripReservoirState(ReservoirSample &sample) {
	sample.PrepareForMerge();
	MemoryStream stream(Allocator::DefaultAllocator());
	BinarySerializer serializer(stream);
	serializer.Begin();
	sample.Serialize(serializer);
	serializer.End();

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto result = BlockingSample::Deserialize(deserializer);
	deserializer.End();
	return result;
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

TEST_CASE("PhysicalEmptyResult serialization roundtrip", "[serialization][physical_plan]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);
	vector<LogicalType> types = {LogicalType::BIGINT, LogicalType::VARCHAR};
	PhysicalEmptyResult empty_result(plan, types, 0);

	MemoryStream stream(allocator);
	BinarySerializer serializer(stream);
	serializer.Begin();
	empty_result.Serialize(serializer);
	serializer.End();

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);
	REQUIRE(deserialized_op->type == PhysicalOperatorType::EMPTY_RESULT);
	REQUIRE(deserialized_op->GetTypes() == types);
	REQUIRE(deserialized_op->estimated_cardinality == 0);
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

TEST_CASE("PhysicalDistributedReservoirSample serialization roundtrip",
          "[serialization][physical_plan][reservoir_sample]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	auto options = make_uniq<SampleOptions>(42);
	options->sample_size = Value::BIGINT(17);
	options->is_percentage = false;
	options->method = SampleMethod::RESERVOIR_SAMPLE;
	options->repeatable = true;
	vector<LogicalType> types = {LogicalType::UBIGINT, LogicalType::BLOB};
	auto &sample = plan.Make<PhysicalDistributedReservoirSample>(types, std::move(options),
	                                                             DistributedReservoirSampleStage::LOCAL, 3, 1);

	MemoryStream stream(allocator);
	BinarySerializer serializer(stream);
	serializer.Begin();
	sample.Serialize(serializer);
	serializer.End();

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);
	auto *sample_ptr = dynamic_cast<PhysicalDistributedReservoirSample *>(deserialized_op.get());
	REQUIRE(sample_ptr != nullptr);
	REQUIRE(sample_ptr->stage == DistributedReservoirSampleStage::LOCAL);
	REQUIRE(sample_ptr->task_index == 3);
	REQUIRE(sample_ptr->options->sample_size.GetValue<int64_t>() == 17);
	REQUIRE(sample_ptr->options->GetSeed() == 42);
	REQUIRE(sample_ptr->options->repeatable);
}

TEST_CASE("ReservoirSample states merge arbitrary row counts and preserve strings",
          "[serialization][physical_plan][reservoir_sample]") {
	Allocator allocator;
	constexpr idx_t sample_count = 2500;
	const vector<idx_t> partition_boundaries {0, 13, 3213, 10213};
	vector<unique_ptr<BlockingSample>> states;
	vector<pair<double, int64_t>> weighted_candidates;

	for (idx_t partition_idx = 0; partition_idx + 1 < partition_boundaries.size(); partition_idx++) {
		ReservoirSample local(allocator, sample_count, NumericCast<int64_t>(101 + partition_idx));
		AddReservoirRows(local, allocator, partition_boundaries[partition_idx],
		                 partition_boundaries[partition_idx + 1]);
		auto state = RoundTripReservoirState(local);
		auto &reservoir = state->Cast<ReservoirSample>();
		auto weights = reservoir.base_reservoir_sample->reservoir_weights;
		while (!weights.empty()) {
			const auto entry = weights.top();
			weights.pop();
			const auto value = reservoir.Chunk().GetValue(0, entry.second).GetValue<int64_t>();
			weighted_candidates.emplace_back(-entry.first, value);
		}
		states.push_back(std::move(state));
	}

	std::sort(weighted_candidates.begin(), weighted_candidates.end(),
	          [](const pair<double, int64_t> &left, const pair<double, int64_t> &right) {
		          if (left.first != right.first) {
			          return left.first > right.first;
		          }
		          return left.second < right.second;
	          });
	std::unordered_set<int64_t> expected;
	for (idx_t candidate_idx = 0; candidate_idx < sample_count; candidate_idx++) {
		expected.insert(weighted_candidates[candidate_idx].second);
	}
	REQUIRE(expected.size() == sample_count);

	vector<unique_ptr<BlockingSample>> reverse_states;
	reverse_states.reserve(states.size());
	for (const auto &state : states) {
		reverse_states.push_back(state->Copy());
	}

	ReservoirSample merged(allocator, sample_count, 999);
	for (auto &state : states) {
		merged.Merge(std::move(state));
	}
	REQUIRE(merged.GetTuplesSeen() == partition_boundaries.back());

	std::unordered_set<int64_t> selected;
	idx_t output_count = 0;
	while (true) {
		auto chunk = merged.GetChunk();
		if (!chunk) {
			break;
		}
		for (idx_t row_idx = 0; row_idx < chunk->size(); row_idx++) {
			const auto value = chunk->GetValue(0, row_idx).GetValue<int64_t>();
			REQUIRE(StringValue::Get(chunk->GetValue(1, row_idx)) == "value-" + std::to_string(value));
			REQUIRE(selected.insert(value).second);
			output_count++;
		}
	}
	REQUIRE(output_count == sample_count);
	REQUIRE(selected == expected);

	ReservoirSample reverse_merged(allocator, sample_count, 999);
	for (auto state = reverse_states.rbegin(); state != reverse_states.rend(); state++) {
		reverse_merged.Merge(std::move(*state));
	}
	REQUIRE(reverse_merged.GetTuplesSeen() == partition_boundaries.back());

	std::unordered_set<int64_t> reverse_selected;
	while (true) {
		auto chunk = reverse_merged.GetChunk();
		if (!chunk) {
			break;
		}
		for (idx_t row_idx = 0; row_idx < chunk->size(); row_idx++) {
			const auto value = chunk->GetValue(0, row_idx).GetValue<int64_t>();
			REQUIRE(StringValue::Get(chunk->GetValue(1, row_idx)) == "value-" + std::to_string(value));
			REQUIRE(reverse_selected.insert(value).second);
		}
	}
	REQUIRE(reverse_selected == expected);
}

TEST_CASE("ReservoirSample state merge handles every skewed arrival order",
          "[serialization][physical_plan][reservoir_sample]") {
	Allocator allocator;
	constexpr idx_t sample_count = 7;
	const vector<idx_t> partition_boundaries {0, 2, 7, 27, 80};
	vector<unique_ptr<BlockingSample>> states;
	vector<pair<double, int64_t>> weighted_candidates;

	for (idx_t partition_idx = 0; partition_idx + 1 < partition_boundaries.size(); partition_idx++) {
		ReservoirSample local(allocator, sample_count, NumericCast<int64_t>(201 + partition_idx));
		AddReservoirRows(local, allocator, partition_boundaries[partition_idx],
		                 partition_boundaries[partition_idx + 1]);
		auto state = RoundTripReservoirState(local);
		auto &reservoir = state->Cast<ReservoirSample>();
		auto weights = reservoir.base_reservoir_sample->reservoir_weights;
		while (!weights.empty()) {
			const auto entry = weights.top();
			weights.pop();
			const auto value = reservoir.Chunk().GetValue(0, entry.second).GetValue<int64_t>();
			weighted_candidates.emplace_back(-entry.first, value);
		}
		states.push_back(std::move(state));
	}

	std::sort(weighted_candidates.begin(), weighted_candidates.end(),
	          [](const pair<double, int64_t> &left, const pair<double, int64_t> &right) {
		          if (left.first != right.first) {
			          return left.first > right.first;
		          }
		          return left.second < right.second;
	          });
	std::unordered_set<int64_t> expected;
	for (idx_t candidate_idx = 0; candidate_idx < sample_count; candidate_idx++) {
		expected.insert(weighted_candidates[candidate_idx].second);
	}

	vector<idx_t> arrival_order {0, 1, 2, 3};
	do {
		vector<unique_ptr<DataChunk>> output_chunks;
		{
			ReservoirSample merged(allocator, sample_count, 999);
			for (const auto partition_idx : arrival_order) {
				merged.Merge(states[partition_idx]->Copy());
			}
			REQUIRE(merged.GetTuplesSeen() == partition_boundaries.back());
			while (auto chunk = merged.GetChunk()) {
				output_chunks.push_back(std::move(chunk));
			}
		}

		std::unordered_set<int64_t> selected;
		for (const auto &chunk : output_chunks) {
			for (idx_t row_idx = 0; row_idx < chunk->size(); row_idx++) {
				const auto value = chunk->GetValue(0, row_idx).GetValue<int64_t>();
				REQUIRE(StringValue::Get(chunk->GetValue(1, row_idx)) == "value-" + std::to_string(value));
				REQUIRE(selected.insert(value).second);
			}
		}
		REQUIRE(selected == expected);
	} while (std::next_permutation(arrival_order.begin(), arrival_order.end()));
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

TEST_CASE("PhysicalHashJoin serialization preserves a global MARK build summary",
          "[serialization][physical_plan][join]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);
	vector<LogicalType> input_types = {LogicalType::INTEGER};
	auto &left_scan = MakeColumnDataScan(plan, input_types);
	auto &right_scan = MakeColumnDataScan(plan, input_types);

	JoinCondition condition;
	condition.left = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0);
	condition.right = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0);
	condition.comparison = ExpressionType::COMPARE_EQUAL;
	vector<JoinCondition> conditions;
	conditions.push_back(std::move(condition));

	LogicalComparisonJoin logical_join(JoinType::MARK);
	logical_join.types = {LogicalType::INTEGER, LogicalType::BOOLEAN};
	auto &hash_join =
	    plan.Make<PhysicalHashJoin>(logical_join, left_scan, right_scan, std::move(conditions), JoinType::MARK, 100)
	        .Cast<PhysicalHashJoin>();
	hash_join.mark_join_build_summary = MarkJoinBuildSummary::Create(true, true);

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);
	serializer.Begin();
	hash_join.Serialize(serializer);
	serializer.End();

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	auto *join_ptr = dynamic_cast<PhysicalHashJoin *>(deserialized_op.get());
	REQUIRE(join_ptr != nullptr);
	REQUIRE(join_ptr->join_type == JoinType::MARK);
	REQUIRE(join_ptr->mark_join_build_summary.valid);
	REQUIRE(join_ptr->mark_join_build_summary.has_rows);
	REQUIRE(join_ptr->mark_join_build_summary.has_null);
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

TEST_CASE("PhysicalHashAggregate grouping sets serialization roundtrip",
          "[serialization][physical_plan][grouping_sets]") {
	DuckDB db(nullptr);
	Connection conn(db);
	conn.BeginTransaction();
	auto &context = *conn.context;

	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<unique_ptr<Expression>> groups;
	groups.push_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0));
	groups.push_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 1));

	vector<unique_ptr<Expression>> aggregates;
	auto count = MakeCountAggregate(context, 2, LogicalType::INTEGER);
	count->Cast<BoundAggregateExpression>().filter = make_uniq<BoundReferenceExpression>(LogicalType::BOOLEAN, 3);
	aggregates.push_back(std::move(count));

	vector<GroupingSet> grouping_sets;
	grouping_sets.push_back({0, 1});
	grouping_sets.push_back({0});
	grouping_sets.push_back({0}); // Duplicate grouping sets have duplicate-row semantics.
	grouping_sets.push_back({});

	vector<unsafe_vector<idx_t>> grouping_functions;
	grouping_functions.push_back({0, 1});

	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BIGINT, LogicalType::BIGINT};
	auto &hash_agg = plan.Make<PhysicalHashAggregate>(
	    context, types, std::move(aggregates), std::move(groups), grouping_sets, grouping_functions, 42,
	    TupleDataValidityType::CAN_HAVE_NULL_VALUES, TupleDataValidityType::CAN_HAVE_NULL_VALUES);

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);
	serializer.Begin();
	hash_agg.Serialize(serializer);
	serializer.End();

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Set<ClientContext &>(context);
	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);
	auto *hash_ptr = dynamic_cast<PhysicalHashAggregate *>(deserialized_op.get());
	REQUIRE(hash_ptr != nullptr);
	REQUIRE(hash_ptr->grouping_sets == grouping_sets);
	REQUIRE(hash_ptr->grouped_aggregate_data.grouping_functions.size() == grouping_functions.size());
	REQUIRE(hash_ptr->grouped_aggregate_data.grouping_functions[0] == grouping_functions[0]);
	auto &aggregate = hash_ptr->grouped_aggregate_data.aggregates[0]->Cast<BoundAggregateExpression>();
	auto &filter = aggregate.filter->Cast<BoundReferenceExpression>();
	REQUIRE(filter.index == 1);
	auto filter_index = hash_ptr->filter_indexes.find(aggregate.filter.get());
	REQUIRE(filter_index != hash_ptr->filter_indexes.end());
	REQUIRE(filter_index->second == 3);

	conn.Rollback();
}

TEST_CASE("PhysicalHashAggregate sorted aggregate serialization roundtrip",
          "[serialization][physical_plan][grouping_sets]") {
	DuckDB db(nullptr);
	Connection conn(db);
	conn.BeginTransaction();
	auto &context = *conn.context;

	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<unique_ptr<Expression>> groups;
	groups.push_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0));

	auto aggregate = MakeCountAggregate(context, 1, LogicalType::INTEGER);
	aggregate->order_bys = make_uniq<BoundOrderModifier>();
	aggregate->order_bys->orders.emplace_back(OrderType::DESCENDING, OrderByNullType::NULLS_FIRST,
	                                          make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 2));
	FunctionBinder::BindSortedAggregate(context, *aggregate, groups, nullptr);
	REQUIRE(aggregate->order_bys == nullptr);

	vector<unique_ptr<Expression>> aggregates;
	aggregates.push_back(std::move(aggregate));
	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::BIGINT};
	auto &hash_agg = plan.Make<PhysicalHashAggregate>(context, types, std::move(aggregates), std::move(groups), 42);

	// Reproduce distributed stage construction: the query transaction that
	// created the sorted aggregate has ended before the task plan is serialized.
	conn.Rollback();

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);
	serializer.Begin();
	hash_agg.Serialize(serializer);
	serializer.End();

	conn.BeginTransaction();
	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Set<ClientContext &>(context);
	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);
	auto *hash_ptr = dynamic_cast<PhysicalHashAggregate *>(deserialized_op.get());
	REQUIRE(hash_ptr != nullptr);
	auto &roundtrip = hash_ptr->grouped_aggregate_data.aggregates[0]->Cast<BoundAggregateExpression>();
	REQUIRE(roundtrip.order_bys == nullptr);
	REQUIRE(roundtrip.children.size() == 2);

	auto portable = FunctionBinder::UnbindSortedAggregate(roundtrip);
	REQUIRE(portable->children.size() == 1);
	REQUIRE(portable->order_bys != nullptr);
	REQUIRE(portable->order_bys->orders.size() == 1);
	auto &order = portable->order_bys->orders[0];
	REQUIRE(order.type == OrderType::DESCENDING);
	REQUIRE(order.null_order == OrderByNullType::NULLS_FIRST);
	REQUIRE(order.expression->Cast<BoundReferenceExpression>().index == 2);

	conn.Rollback();
}

TEST_CASE("PhysicalGroupingSetExpand serialization roundtrip", "[serialization][physical_plan][grouping_sets]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<unique_ptr<Expression>> groups;
	groups.push_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0));
	groups.push_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 1));
	vector<GroupingSet> grouping_sets = {{0, 1}, {0}, {0}, {}};
	vector<vector<idx_t>> grouping_functions = {{0, 1}};
	vector<idx_t> filter_indexes = {DConstants::INVALID_INDEX, 2};
	vector<LogicalType> types = {LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BOOLEAN,
	                             LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::BIGINT,
	                             LogicalType::UBIGINT, LogicalType::BOOLEAN, LogicalType::BOOLEAN};
	auto &expand = plan.Make<PhysicalGroupingSetExpand>(types, std::move(groups), grouping_sets, grouping_functions,
	                                                    filter_indexes, 3, true, 42);

	MemoryStream stream(allocator);
	SerializationOptions options;
	BinarySerializer serializer(stream, options);
	serializer.Begin();
	expand.Serialize(serializer);
	serializer.End();

	stream.Rewind();
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
	deserializer.End();

	REQUIRE(deserialized_op != nullptr);
	auto *expand_ptr = dynamic_cast<PhysicalGroupingSetExpand *>(deserialized_op.get());
	REQUIRE(expand_ptr != nullptr);
	REQUIRE(expand_ptr->grouping_sets == grouping_sets);
	REQUIRE(expand_ptr->grouping_functions == grouping_functions);
	REQUIRE(expand_ptr->filter_indexes == filter_indexes);
	REQUIRE(expand_ptr->input_column_count == 3);
	REQUIRE(expand_ptr->emit_empty_grouping_sets);
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

TEST_CASE("Distributed physical plan clone exposes client settings to native bind deserialization",
          "[serialization][physical_plan][distributed]") {
	DuckDB db(nullptr);
	Connection conn(db);
	auto set_result = conn.Query("SET threads=3");
	REQUIRE(!set_result->HasError());

	conn.BeginTransaction();
	auto &context = *conn.context;
	auto &catalog = Catalog::GetSystemCatalog(context);
	auto func = MakeContextSettingTestInOutFunction();
	CreateTableFunctionInfo info(func);
	catalog.CreateTableFunction(context, info);
	conn.Commit();

	conn.BeginTransaction();
	auto &entry = Catalog::GetEntry<TableFunctionCatalogEntry>(context, SYSTEM_CATALOG, DEFAULT_SCHEMA,
	                                                           "test_inout_context_setting");
	auto table_func = entry.functions.GetFunctionByArguments(context, {LogicalType::TABLE});
	auto plan = std::make_shared<PhysicalPlan>(Allocator::DefaultAllocator());
	vector<LogicalType> types = {LogicalType::INTEGER};
	vector<ColumnIndex> column_ids;
	column_ids.emplace_back(0);
	auto bind_data = make_uniq<TestInOutBindData>();
	bind_data->marker = 123;
	vector<column_t> projected_input;
	auto &inout =
	    plan->Make<PhysicalTableInOutFunction>(types, table_func, std::move(bind_data), column_ids, 1, projected_input);
	plan->SetRoot(inout);

	auto cloned = distributed::ClonePhysicalPlanOrThrow(plan, "client_settings_test", conn.context.get());
	auto &cloned_inout = cloned->Root().Cast<PhysicalTableInOutFunction>();
	auto cloned_bind_data = cloned_inout.GetBindData();
	REQUIRE(cloned_bind_data);
	REQUIRE(cloned_bind_data->Cast<TestInOutBindData>().marker == 123);
	conn.Rollback();
}

TEST_CASE("PhysicalRemoteExchangeSink serialization preserves static sink configuration",
          "[serialization][physical_plan][exchange]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	vector<LogicalType> types = {LogicalType::INTEGER};
	distributed::FlightExchangeConfig flight_config;
	flight_config.node_id = "node-1";
	flight_config.local_dirs = {"/session-a/shuffle-0", "/session-a/shuffle-1"};
	auto exchange_mgr = std::make_shared<distributed::FlightExchangeManager>(std::move(flight_config));

	vector<unique_ptr<Expression>> partition_by;
	auto &sink = plan.Make<PhysicalRemoteExchangeSink>(types, 123, "exchange", 4, RepartitionSpec::Type::Random,
	                                                   std::move(partition_by), "query-session-a", "exchange-instance",
	                                                   exchange_mgr);
	vector<unique_ptr<Expression>> mark_join_build_expressions;
	mark_join_build_expressions.push_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0));
	sink.Cast<PhysicalRemoteExchangeSink>().EnableMarkJoinBuildSummary(std::move(mark_join_build_expressions));

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
	REQUIRE(sink_ptr->ExchangeId() == "exchange");
	REQUIRE(sink_ptr->NumPartitions() == 4);
	REQUIRE(sink_ptr->SinkQueryId() == "query-session-a");
	REQUIRE(sink_ptr->SinkOutputLocationPrefix() == "exchange-instance");
	REQUIRE_FALSE(sink_ptr->HasBoundSinkHandle());
	REQUIRE_THROWS(sink_ptr->SinkHandle());
	REQUIRE(sink_ptr->CollectsMarkJoinBuildSummary());
	REQUIRE(sink_ptr->MarkJoinBuildExpressions().size() == 1);
	REQUIRE(sink_ptr->MarkJoinBuildExpressions()[0]->return_type == LogicalType::INTEGER);
	auto roundtrip_manager =
	    std::dynamic_pointer_cast<distributed::FlightExchangeManager>(sink_ptr->GetExchangeManager());
	const std::vector<std::string> expected_local_dirs = {"/session-a/shuffle-0", "/session-a/shuffle-1"};
	REQUIRE(roundtrip_manager != nullptr);
	REQUIRE(roundtrip_manager->config().node_id == "node-1");
	REQUIRE(roundtrip_manager->config().local_dirs == expected_local_dirs);
}

TEST_CASE("ApplyExchangeSinkInstanceToPlan validates runtime sink ownership",
          "[serialization][physical_plan][exchange]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	distributed::FlightExchangeConfig flight_config;
	flight_config.node_id = "node-1";
	auto exchange_mgr = std::make_shared<distributed::FlightExchangeManager>(std::move(flight_config));
	auto &sink_op = plan.Make<PhysicalRemoteExchangeSink>(
	    vector<LogicalType> {LogicalType::INTEGER}, 123, "diagnostic-exchange", 4, RepartitionSpec::Type::Random,
	    vector<unique_ptr<Expression>> {}, "query-runtime-sink", "opaque-exchange", exchange_mgr);
	auto &sink = sink_op.Cast<PhysicalRemoteExchangeSink>();
	auto sample_options = make_uniq<SampleOptions>(42);
	sample_options->sample_size = Value::BIGINT(17);
	sample_options->is_percentage = false;
	sample_options->method = SampleMethod::RESERVOIR_SAMPLE;
	auto &local_sample =
	    plan.Make<PhysicalDistributedReservoirSample>(vector<LogicalType> {LogicalType::UBIGINT, LogicalType::BLOB},
	                                                  std::move(sample_options), DistributedReservoirSampleStage::LOCAL,
	                                                  DConstants::INVALID_INDEX, 1)
	        .Cast<PhysicalDistributedReservoirSample>();
	sink.children.push_back(local_sample);
	plan.SetRoot(sink);

	distributed::ExchangeSinkInstanceTaskDescriptor descriptor;
	descriptor.sink_instance.sink_handle.task_partition_id = 7;
	descriptor.sink_instance.attempt_id = 2;
	descriptor.sink_instance.query_id = "query-runtime-sink";
	descriptor.sink_instance.output_location = "opaque-exchange__sink_7__attempt_2";
	descriptor.sink_instance.output_partition_count = 4;

	string error;
	auto invalid = descriptor;
	invalid.sink_instance.sink_handle.task_partition_id = DConstants::INVALID_INDEX;
	REQUIRE_FALSE(distributed::ApplyExchangeSinkInstanceToPlan(plan, invalid, &error));
	REQUIRE(error.find("invalid task identity") != string::npos);
	REQUIRE_FALSE(sink.HasBoundSinkHandle());
	REQUIRE(local_sample.task_index == DConstants::INVALID_INDEX);

	error.clear();
	invalid = descriptor;
	invalid.sink_instance.query_id = "other-query";
	REQUIRE_FALSE(distributed::ApplyExchangeSinkInstanceToPlan(plan, invalid, &error));
	REQUIRE(error.find("query") != string::npos);
	REQUIRE_FALSE(sink.HasBoundSinkHandle());
	REQUIRE(local_sample.task_index == DConstants::INVALID_INDEX);
	REQUIRE_THROWS_AS(local_sample.GetEffectiveSeed(), InternalException);

	error.clear();
	invalid = descriptor;
	invalid.sink_instance.sink_handle.task_partition_id = 8;
	REQUIRE_FALSE(distributed::ApplyExchangeSinkInstanceToPlan(plan, invalid, &error));
	REQUIRE(error.find("output location") != string::npos);
	REQUIRE_FALSE(sink.HasBoundSinkHandle());

	error.clear();
	invalid = descriptor;
	invalid.sink_instance.sink_handle.task_partition_id = 8;
	invalid.sink_instance.output_location = "opaque-exchange__sink_8__attempt_2";
	REQUIRE(distributed::ApplyExchangeSinkInstanceToPlan(plan, invalid, &error));
	REQUIRE(error.empty());
	REQUIRE(sink.SinkHandle().sink_handle.task_partition_id == 8);
	REQUIRE(sink.SinkHandle().output_location == "opaque-exchange__sink_8__attempt_2");
	REQUIRE(local_sample.task_index == 8);
	REQUIRE(local_sample.options->GetSeed() == 42);
	REQUIRE_NOTHROW(local_sample.GetEffectiveSeed());

	error.clear();
	invalid = descriptor;
	invalid.sink_instance.sink_handle.task_partition_id = 7;
	invalid.sink_instance.output_location = "opaque-exchange__sink_7__attempt_2";
	REQUIRE(distributed::ApplyExchangeSinkInstanceToPlan(plan, invalid, &error));
	REQUIRE(error.empty());

	error.clear();
	invalid = descriptor;
	invalid.sink_instance.output_partition_count = 0;
	REQUIRE_FALSE(distributed::ApplyExchangeSinkInstanceToPlan(plan, invalid, &error));
	REQUIRE(error.find("partition count") != string::npos);
	REQUIRE(sink.SinkHandle().attempt_id == 2);

	error.clear();
	invalid = descriptor;
	invalid.sink_instance.output_location = "other-exchange__sink_7__attempt_2";
	REQUIRE_FALSE(distributed::ApplyExchangeSinkInstanceToPlan(plan, invalid, &error));
	REQUIRE(error.find("output location") != string::npos);
	REQUIRE(sink.SinkHandle().attempt_id == 2);

	error.clear();
	REQUIRE(distributed::ApplyExchangeSinkInstanceToPlan(plan, descriptor, &error));
	REQUIRE(error.empty());
	REQUIRE(sink.SinkHandle().query_id == "query-runtime-sink");
	REQUIRE(sink.SinkHandle().sink_handle.task_partition_id == 7);
	REQUIRE(sink.SinkHandle().attempt_id == 2);
	REQUIRE(sink.SinkHandle().output_location == "opaque-exchange__sink_7__attempt_2");
	REQUIRE(sink.SinkHandle().output_partition_count == 4);
}

TEST_CASE("Exchange sink task binding rejects worker plans with multiple sinks",
          "[serialization][physical_plan][exchange]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);
	distributed::FlightExchangeConfig flight_config;
	flight_config.node_id = "node-1";
	auto exchange_mgr = std::make_shared<distributed::FlightExchangeManager>(std::move(flight_config));
	vector<LogicalType> types {LogicalType::INTEGER};
	auto &inner = plan.Make<PhysicalRemoteExchangeSink>(types, 123, "inner-exchange", 4, RepartitionSpec::Type::Random,
	                                                    vector<unique_ptr<Expression>> {}, "query-multiple-sinks",
	                                                    "inner-output", exchange_mgr)
	                  .Cast<PhysicalRemoteExchangeSink>();
	auto &outer = plan.Make<PhysicalRemoteExchangeSink>(types, 123, "outer-exchange", 4, RepartitionSpec::Type::Random,
	                                                    vector<unique_ptr<Expression>> {}, "query-multiple-sinks",
	                                                    "outer-output", exchange_mgr)
	                  .Cast<PhysicalRemoteExchangeSink>();
	outer.children.push_back(inner);
	plan.SetRoot(outer);

	const PhysicalRemoteExchangeSink *found_sink = nullptr;
	string error;
	REQUIRE_FALSE(distributed::TryGetUniqueRemoteExchangeSink(plan.Root(), found_sink, &error));
	REQUIRE(error.find("at most one") != string::npos);
	REQUIRE(found_sink == nullptr);

	distributed::ExchangeSinkInstanceTaskDescriptor descriptor;
	descriptor.sink_instance.sink_handle.task_partition_id = 7;
	descriptor.sink_instance.attempt_id = 2;
	descriptor.sink_instance.query_id = "query-multiple-sinks";
	descriptor.sink_instance.output_location = "outer-output__sink_7__attempt_2";
	descriptor.sink_instance.output_partition_count = 4;
	error.clear();
	REQUIRE_FALSE(distributed::ApplyExchangeSinkInstanceToPlan(plan, descriptor, &error));
	REQUIRE(error.find("at most one") != string::npos);
	REQUIRE_FALSE(outer.HasBoundSinkHandle());
	REQUIRE_FALSE(inner.HasBoundSinkHandle());
}

TEST_CASE("ExchangeSinkInstanceTaskDescriptor serialization preserves the worker endpoint",
          "[serialization][physical_plan][exchange]") {
	distributed::ExchangeSinkInstanceTaskDescriptor descriptor;
	descriptor.sink_instance.sink_handle.task_partition_id = 3;
	descriptor.sink_instance.attempt_id = 2;
	descriptor.sink_instance.query_id = "endpoint-query";
	descriptor.sink_instance.output_location = "endpoint-exchange__sink_3__attempt_2";
	descriptor.sink_instance.output_partition_count = 4;
	descriptor.sink_instance.flight_host = "flight-worker.internal";
	descriptor.sink_instance.flight_server_epoch = "endpoint-epoch";
	descriptor.sink_instance.mark_join_build_summary = MarkJoinBuildSummary::Create(true, true);

	auto roundtrip =
	    distributed::ExchangeSinkInstanceTaskDescriptor::DeserializeFromBytes(descriptor.SerializeToBytes());

	REQUIRE(roundtrip.sink_instance.sink_handle.task_partition_id == 3);
	REQUIRE(roundtrip.sink_instance.attempt_id == 2);
	REQUIRE(roundtrip.sink_instance.query_id == "endpoint-query");
	REQUIRE(roundtrip.sink_instance.output_location == "endpoint-exchange__sink_3__attempt_2");
	REQUIRE(roundtrip.sink_instance.output_partition_count == 4);
	REQUIRE(roundtrip.sink_instance.flight_host == "flight-worker.internal");
	REQUIRE(roundtrip.sink_instance.flight_server_epoch == "endpoint-epoch");
	REQUIRE(roundtrip.sink_instance.mark_join_build_summary.valid);
	REQUIRE(roundtrip.sink_instance.mark_join_build_summary.has_rows);
	REQUIRE(roundtrip.sink_instance.mark_join_build_summary.has_null);
}

TEST_CASE("Exchange task descriptors reject MARK summary payload without validity",
          "[serialization][physical_plan][exchange]") {
	SECTION("sink task") {
		distributed::ExchangeSinkInstanceTaskDescriptor descriptor;
		descriptor.sink_instance.mark_join_build_summary.has_rows = true;
		REQUIRE_THROWS(descriptor.SerializeToBytes());
	}

	SECTION("source task") {
		distributed::ExchangeSourceTaskDescriptor descriptor;
		descriptor.mark_join_build_summary.has_rows = true;
		REQUIRE_THROWS(descriptor.SerializeToBytes());
	}
}

TEST_CASE("Exchange task descriptors reject missing advertised hosts", "[serialization][physical_plan][exchange]") {
	REQUIRE_THROWS(distributed::ExchangeSinkInstanceTaskDescriptor::DeserializeFromBytes(
	    SerializeSinkDescriptorWithoutFlightHost()));
	REQUIRE_THROWS(
	    distributed::ExchangeSourceTaskDescriptor::DeserializeFromBytes(SerializeSourceDescriptorWithoutFlightHost()));
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
	handle0.source_task_partition_id = 10;
	handle0.attempt_id = 3;
	handle0.node_id = "node-1";
	handle0.flight_host = "flight-node-1.internal";
	handle0.flight_port = 6123;
	handle0.flight_server_epoch = "epoch-1";
	handle0.files.push_back(ExchangeSourceFile("exchange__sink_0__attempt_0", 0));
	source_handles.push_back(handle0);

	distributed::ExchangeSourceHandle handle1;
	handle1.partition_id = 0;
	handle1.source_task_partition_id = 11;
	handle1.attempt_id = 4;
	handle1.node_id = "node-2";
	handle1.flight_host = "flight-node-2.internal";
	handle1.flight_port = 6124;
	handle1.flight_server_epoch = "epoch-2";
	handle1.files.push_back(ExchangeSourceFile("exchange__sink_1__attempt_0", 0));
	source_handles.push_back(handle1);

	distributed::ExchangeSourceHandle handle2;
	handle2.partition_id = 1;
	handle2.source_task_partition_id = 10;
	handle2.attempt_id = 3;
	handle2.node_id = "node-1";
	handle2.flight_host = "flight-node-1.internal";
	handle2.flight_port = 6123;
	handle2.flight_server_epoch = "epoch-1";
	handle2.files.push_back(ExchangeSourceFile("exchange__sink_0__attempt_0", 0));
	source_handles.push_back(handle2);

	distributed::FlightExchangeConfig flight_config;
	flight_config.node_id = "node-1";
	flight_config.flight_timeout_seconds = 7.5;
	auto exchange_mgr = std::make_shared<distributed::FlightExchangeManager>(std::move(flight_config));

	auto &source = plan.Make<PhysicalRemoteExchangeSource>(types, 456, "exchange", partition_indices, source_handles,
	                                                       exchange_mgr, source_nodes);
	plan.SetRoot(source);
	string assignment_error;
	REQUIRE(distributed::ValidateExchangeSourceAssignments(plan, set<idx_t> {}, &assignment_error));
	REQUIRE(assignment_error.empty());

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
	REQUIRE(source_ptr->ExchangeId() == "exchange");
	REQUIRE(source_ptr->PartitionIndices() == partition_indices);
	REQUIRE(source_ptr->SourceNodes() == source_nodes);
	REQUIRE(source_ptr->SourceHandles().size() == source_handles.size());
	REQUIRE(source_ptr->SourceHandles()[0].partition_id == 0);
	REQUIRE(source_ptr->SourceHandles()[0].source_task_partition_id == 10);
	REQUIRE(source_ptr->SourceHandles()[0].attempt_id == 3);
	REQUIRE(source_ptr->SourceHandles()[0].node_id == "node-1");
	REQUIRE(source_ptr->SourceHandles()[0].flight_host == "flight-node-1.internal");
	REQUIRE(source_ptr->SourceHandles()[0].flight_port == 6123);
	REQUIRE(source_ptr->SourceHandles()[0].flight_server_epoch == "epoch-1");
	REQUIRE(source_ptr->SourceHandles()[0].files.size() == 1);
	REQUIRE(source_ptr->SourceHandles()[0].files[0].path == "exchange__sink_0__attempt_0");
	REQUIRE(source_ptr->SourceHandles()[1].partition_id == 0);
	REQUIRE(source_ptr->SourceHandles()[1].source_task_partition_id == 11);
	REQUIRE(source_ptr->SourceHandles()[1].attempt_id == 4);
	REQUIRE(source_ptr->SourceHandles()[1].node_id == "node-2");
	REQUIRE(source_ptr->SourceHandles()[1].flight_host == "flight-node-2.internal");
	REQUIRE(source_ptr->SourceHandles()[1].flight_port == 6124);
	REQUIRE(source_ptr->SourceHandles()[1].flight_server_epoch == "epoch-2");
	REQUIRE(source_ptr->SourceHandles()[1].files.size() == 1);
	REQUIRE(source_ptr->SourceHandles()[1].files[0].path == "exchange__sink_1__attempt_0");
	REQUIRE(source_ptr->SourceHandles()[2].partition_id == 1);
	REQUIRE(source_ptr->SourceHandles()[2].source_task_partition_id == 10);
	REQUIRE(source_ptr->SourceHandles()[2].attempt_id == 3);
	REQUIRE(source_ptr->SourceHandles()[2].node_id == "node-1");
	REQUIRE(source_ptr->SourceHandles()[2].flight_host == "flight-node-1.internal");
	REQUIRE(source_ptr->SourceHandles()[2].flight_port == 6123);
	REQUIRE(source_ptr->SourceHandles()[2].flight_server_epoch == "epoch-1");
	REQUIRE(source_ptr->SourceHandles()[2].files.size() == 1);
	REQUIRE(source_ptr->SourceHandles()[2].files[0].path == "exchange__sink_0__attempt_0");
	auto roundtrip_manager =
	    std::dynamic_pointer_cast<distributed::FlightExchangeManager>(source_ptr->GetExchangeManager());
	REQUIRE(roundtrip_manager != nullptr);
	REQUIRE(roundtrip_manager->config().flight_timeout_seconds == 7.5);
}

TEST_CASE("Remote exchange plans reject pre-strict endpoint payloads", "[serialization][physical_plan][exchange]") {
	{
		Allocator allocator;
		PhysicalPlan plan(allocator);
		MemoryStream stream(allocator);
		SerializationOptions options;
		BinarySerializer serializer(stream, options);
		serializer.Begin();
		SerializePreStrictRemoteExchangeSink(serializer);
		serializer.End();

		stream.Rewind();
		BinaryDeserializer deserializer(stream);
		deserializer.Begin();
		REQUIRE_THROWS(PhysicalOperator::Deserialize(deserializer, plan));
	}

	{
		Allocator allocator;
		PhysicalPlan plan(allocator);
		MemoryStream stream(allocator);
		SerializationOptions options;
		BinarySerializer serializer(stream, options);
		serializer.Begin();
		SerializePreStrictRemoteExchangeSource(serializer);
		serializer.End();

		stream.Rewind();
		BinaryDeserializer deserializer(stream);
		deserializer.Begin();
		REQUIRE_THROWS(PhysicalOperator::Deserialize(deserializer, plan));
	}
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

	auto &source_op = plan.Make<PhysicalRemoteExchangeSource>(types, 456, "exchange", vector<idx_t>(),
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
	REQUIRE(source_ptr->ExchangeId() == "exchange");
	REQUIRE(source_ptr->PartitionIndices().empty());
	REQUIRE(source_ptr->SourceHandles().empty());
	REQUIRE(source_ptr->RuntimeSourceNodeId().IsValid());
	REQUIRE(source_ptr->RuntimeSourceNodeId().GetIndex() == 42);
	auto roundtrip_manager =
	    std::dynamic_pointer_cast<distributed::FlightExchangeManager>(source_ptr->GetExchangeManager());
	REQUIRE(roundtrip_manager != nullptr);
}

TEST_CASE("PhysicalRemoteExchangeSource serialization preserves an explicit empty catalog",
          "[serialization][physical_plan][exchange]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);

	distributed::FlightExchangeConfig flight_config;
	flight_config.node_id = "node-1";
	auto exchange_mgr = std::make_shared<distributed::FlightExchangeManager>(std::move(flight_config));
	auto &source_op = plan.Make<PhysicalRemoteExchangeSource>(
	    vector<LogicalType> {LogicalType::INTEGER}, 0, "empty-exchange", vector<idx_t> {0},
	    std::vector<distributed::ExchangeSourceHandle> {}, exchange_mgr, vector<string> {}, optional_idx());
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

	auto *source_ptr = dynamic_cast<PhysicalRemoteExchangeSource *>(deserialized_op.get());
	REQUIRE(source_ptr != nullptr);
	REQUIRE(source_ptr->ExchangeId() == "empty-exchange");
	REQUIRE(source_ptr->PartitionIndices() == vector<idx_t> {0});
	REQUIRE(source_ptr->SourceHandles().empty());
	REQUIRE_FALSE(source_ptr->RuntimeSourceNodeId().IsValid());
}

TEST_CASE("ExchangeSourceTaskDescriptor serialization preserves source handle attempt ids",
          "[serialization][physical_plan][exchange]") {
	REQUIRE_THROWS_WITH(distributed::ExchangeSourceTaskDescriptor::DeserializeFromBytes(""),
	                    Catch::Matchers::Contains("empty exchange source task descriptor"));
	distributed::ExchangeSourceTaskDescriptor descriptor;
	descriptor.partition_indices = {0, 1};
	descriptor.source_partition_count = 2;
	descriptor.source_task_count = 2;
	descriptor.mark_join_build_summary = MarkJoinBuildSummary::Create(true, true);

	distributed::ExchangeSourceHandle handle0;
	handle0.partition_id = 0;
	handle0.source_task_partition_id = 21;
	handle0.attempt_id = 7;
	handle0.node_id = "node-1";
	handle0.flight_host = "flight-node-1.internal";
	handle0.flight_port = 5010;
	handle0.flight_server_epoch = "epoch-1";
	handle0.files.push_back(ExchangeSourceFile("exchange__sink_0__attempt_7", 0, (uint64_t(1) << 40) + 123));
	descriptor.source_handles.push_back(handle0);

	distributed::ExchangeSourceHandle handle1;
	handle1.partition_id = 1;
	handle1.source_task_partition_id = 22;
	handle1.attempt_id = 2;
	handle1.node_id = "node-2";
	handle1.flight_host = "flight-node-2.internal";
	handle1.flight_port = 5011;
	handle1.flight_server_epoch = "epoch-2";
	handle1.files.push_back(ExchangeSourceFile("exchange__sink_1__attempt_2", 0, 17));
	descriptor.source_handles.push_back(handle1);

	auto roundtrip = distributed::ExchangeSourceTaskDescriptor::DeserializeFromBytes(descriptor.SerializeToBytes());

	REQUIRE(roundtrip.partition_indices == descriptor.partition_indices);
	REQUIRE(roundtrip.source_partition_count == 2);
	REQUIRE(roundtrip.source_task_count == 2);
	REQUIRE(roundtrip.mark_join_build_summary.valid);
	REQUIRE(roundtrip.mark_join_build_summary.has_rows);
	REQUIRE(roundtrip.mark_join_build_summary.has_null);
	REQUIRE(roundtrip.source_handles.size() == 2);
	REQUIRE(roundtrip.source_handles[0].partition_id == 0);
	REQUIRE(roundtrip.source_handles[0].source_task_partition_id == 21);
	REQUIRE(roundtrip.source_handles[0].attempt_id == 7);
	REQUIRE(roundtrip.source_handles[0].node_id == "node-1");
	REQUIRE(roundtrip.source_handles[0].flight_host == "flight-node-1.internal");
	REQUIRE(roundtrip.source_handles[0].flight_port == 5010);
	REQUIRE(roundtrip.source_handles[0].flight_server_epoch == "epoch-1");
	REQUIRE(roundtrip.source_handles[0].files.size() == 1);
	REQUIRE(roundtrip.source_handles[0].files[0].path == "exchange__sink_0__attempt_7");
	REQUIRE(roundtrip.source_handles[0].files[0].file_size == (uint64_t(1) << 40) + 123);
	REQUIRE(roundtrip.source_handles[1].partition_id == 1);
	REQUIRE(roundtrip.source_handles[1].source_task_partition_id == 22);
	REQUIRE(roundtrip.source_handles[1].attempt_id == 2);
	REQUIRE(roundtrip.source_handles[1].node_id == "node-2");
	REQUIRE(roundtrip.source_handles[1].flight_host == "flight-node-2.internal");
	REQUIRE(roundtrip.source_handles[1].flight_port == 5011);
	REQUIRE(roundtrip.source_handles[1].flight_server_epoch == "epoch-2");
	REQUIRE(roundtrip.source_handles[1].files.size() == 1);
	REQUIRE(roundtrip.source_handles[1].files[0].path == "exchange__sink_1__attempt_2");
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

	auto &source_op = plan.Make<PhysicalRemoteExchangeSource>(types, 456, "exchange", vector<idx_t>(),
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
	handle0.files.push_back(ExchangeSourceFile("exchange__sink_0__attempt_0", 0, 11));
	descriptor.source_handles.push_back(handle0);
	distributed::ExchangeSourceHandle handle1;
	handle1.partition_id = 1;
	handle1.attempt_id = 6;
	handle1.node_id = "node-2";
	handle1.files.push_back(ExchangeSourceFile("exchange__sink_1__attempt_0", 0, 17));
	descriptor.source_handles.push_back(handle1);

	std::unordered_map<idx_t, distributed::ExchangeSourceTaskDescriptor> tasks;
	tasks.emplace(42, descriptor);
	set<idx_t> assigned_node_ids {42};
	REQUIRE(distributed::ValidateExchangeSourceAssignments(plan, assigned_node_ids));
	REQUIRE_FALSE(distributed::ValidateExchangeSourceAssignments(plan, set<idx_t> {}));
	REQUIRE_FALSE(distributed::ValidateExchangeSourceAssignments(plan, set<idx_t> {42, 999}));

	string error;
	auto tasks_with_unknown_node = tasks;
	tasks_with_unknown_node.emplace(999, descriptor);
	REQUIRE_FALSE(distributed::ApplyExchangeSourceTasksToPlan(plan, tasks_with_unknown_node, &error));
	REQUIRE(StringUtil::Contains(error, "node_id=999 is not present in the worker plan"));
	REQUIRE(source.PartitionIndices().empty());
	REQUIRE(source.SourceHandles().empty());
	error.clear();
	REQUIRE(distributed::ApplyExchangeSourceTasksToPlan(plan, tasks, &error));
	REQUIRE(error.empty());
	REQUIRE(source.PartitionIndices() == descriptor.partition_indices);
	REQUIRE(source.SourceHandles().size() == descriptor.source_handles.size());
	REQUIRE(source.SourceHandles()[0].partition_id == 0);
	REQUIRE(source.SourceHandles()[0].attempt_id == 5);
	REQUIRE(source.SourceHandles()[0].node_id == "node-1");
	REQUIRE(source.SourceHandles()[0].files.size() == 1);
	REQUIRE(source.SourceHandles()[0].files[0].path == "exchange__sink_0__attempt_0");
	REQUIRE(source.SourceHandles()[0].files[0].file_size == 11);
	REQUIRE(source.SourceHandles()[1].partition_id == 1);
	REQUIRE(source.SourceHandles()[1].attempt_id == 6);
	REQUIRE(source.SourceHandles()[1].node_id == "node-2");
	REQUIRE(source.SourceHandles()[1].files.size() == 1);
	REQUIRE(source.SourceHandles()[1].files[0].path == "exchange__sink_1__attempt_0");
	REQUIRE(source.SourceHandles()[1].files[0].file_size == 17);
}

TEST_CASE("Static and FTE exchange assignments apply as validated disjoint subsets",
          "[serialization][physical_plan][exchange]") {
	Allocator allocator;
	PhysicalPlan plan(allocator);
	vector<LogicalType> types = {LogicalType::INTEGER};
	vector<string> source_nodes = {"node-1"};

	distributed::FlightExchangeConfig flight_config;
	flight_config.node_id = "node-1";
	auto exchange_mgr = std::make_shared<distributed::FlightExchangeManager>(std::move(flight_config));
	distributed::ExchangeSourceHandle embedded_handle;
	embedded_handle.partition_id = 0;
	embedded_handle.node_id = "node-1";
	auto &embedded_source =
	    plan.Make<PhysicalRemoteExchangeSource>(types, 1, "embedded-exchange", vector<idx_t> {0},
	                                            vector<distributed::ExchangeSourceHandle> {embedded_handle},
	                                            exchange_mgr, source_nodes)
	        .Cast<PhysicalRemoteExchangeSource>();
	auto &static_source = plan.Make<PhysicalRemoteExchangeSource>(types, 1, "static-exchange", vector<idx_t>(),
	                                                              vector<distributed::ExchangeSourceHandle>(),
	                                                              exchange_mgr, source_nodes, optional_idx(42))
	                          .Cast<PhysicalRemoteExchangeSource>();
	auto &fte_source = plan.Make<PhysicalRemoteExchangeSource>(types, 1, "fte-exchange", vector<idx_t>(),
	                                                           vector<distributed::ExchangeSourceHandle>(),
	                                                           exchange_mgr, source_nodes, optional_idx(43))
	                       .Cast<PhysicalRemoteExchangeSource>();
	fte_source.children.push_back(embedded_source);
	static_source.children.push_back(fte_source);
	plan.SetRoot(static_source);

	distributed::ExchangeSourceTaskDescriptor descriptor;
	descriptor.partition_indices = {0};
	distributed::ExchangeSourceHandle handle;
	handle.partition_id = 0;
	handle.node_id = "node-1";
	descriptor.source_handles.push_back(handle);
	unordered_map<idx_t, distributed::ExchangeSourceTaskDescriptor> static_tasks {{42, descriptor}};
	auto queue = std::make_shared<distributed::FteSplitQueue>();
	unordered_map<idx_t, std::shared_ptr<distributed::FteSplitQueue>> fte_queues {{43, queue}};

	string error;
	REQUIRE(distributed::ValidateExchangeSourceAssignments(plan, set<idx_t> {42, 43}, &error));
	REQUIRE(error.empty());
	REQUIRE(distributed::ApplyExchangeSourceTasksToPlan(plan, static_tasks, &error));
	REQUIRE(error.empty());
	REQUIRE(distributed::ApplyFteExchangeSourceQueuesToPlan(plan, fte_queues, &error));
	REQUIRE(error.empty());
	REQUIRE(static_source.SourceHandles().size() == 1);
	REQUIRE(fte_source.SourceHandles().empty());
	REQUIRE(embedded_source.SourceHandles().size() == 1);
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
