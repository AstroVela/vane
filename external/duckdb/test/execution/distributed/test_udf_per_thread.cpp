// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/execution/executor.hpp"
#include "duckdb/execution/operator/helper/physical_result_collector.hpp"
#include "duckdb/execution/operator/projection/physical_tableinout_function.hpp"
#include "duckdb/execution/operator/projection/physical_udf_inout.hpp"
#include "duckdb/execution/operator/scan/physical_dummy_scan.hpp"
#include "duckdb/execution/physical_plan.hpp"
#include "duckdb/function/scalar/udf_functions.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/prepared_statement_data.hpp"

using namespace duckdb;

namespace {

Value MakeIntegerOutputSchema() {
	child_list_t<Value> entry_children;
	entry_children.emplace_back("name", Value("out"));
	entry_children.emplace_back("kind", Value("duckdb_type"));
	entry_children.emplace_back("type", Value("INTEGER"));
	entry_children.emplace_back("dtype", Value(LogicalType::VARCHAR));
	entry_children.emplace_back("shape", Value(LogicalType::LIST(LogicalType::BIGINT)));
	vector<Value> entries;
	entries.emplace_back(Value::STRUCT(std::move(entry_children)));
	child_list_t<LogicalType> schema_children;
	schema_children.emplace_back("name", LogicalType::VARCHAR);
	schema_children.emplace_back("kind", LogicalType::VARCHAR);
	schema_children.emplace_back("type", LogicalType::VARCHAR);
	schema_children.emplace_back("dtype", LogicalType::VARCHAR);
	schema_children.emplace_back("shape", LogicalType::LIST(LogicalType::BIGINT));
	return Value::LIST(LogicalType::STRUCT(std::move(schema_children)), std::move(entries));
}

Value MakeStreamingPayload() {
	child_list_t<Value> children;
	children.emplace_back("payload_version", Value::BIGINT(1));
	children.emplace_back("udf_name", Value("test_udf"));
	children.emplace_back("call_mode", Value("map_batches"));
	children.emplace_back("execution_backend", Value("ray_task"));
	children.emplace_back("output_schema", MakeIntegerOutputSchema());
	children.emplace_back("ref_output_types", Value::LIST(LogicalType::VARCHAR, {Value("INTEGER")}));
	children.emplace_back("streaming_output_mode", Value("ray_block_stream"));
	children.emplace_back("produce_ray_block_stream", Value::BOOLEAN(true));
	children.emplace_back("produce_ref_bundle_output", Value::BOOLEAN(false));
	children.emplace_back("prebatched_input", Value::BOOLEAN(false));
	children.emplace_back("udf_task_input_max_bytes", Value::BIGINT(128 * 1024 * 1024));
	children.emplace_back("udf_output_target_max_bytes", Value::BIGINT(128 * 1024 * 1024));
	return Value::STRUCT(std::move(children));
}

idx_t GetStreamingUDFSourceMaxThreads(Connection &con, const Value &payload) {
	auto physical_plan = make_uniq<PhysicalPlan>(Allocator::DefaultAllocator());
	vector<LogicalType> return_types = {LogicalType::INTEGER};
	vector<string> return_names = {"out"};
	auto table_function = MakeUDFTableFunction(payload, return_types, return_names);
	auto bind_data = make_uniq<UDFFunctionData>(payload, return_types[0]);
	vector<ColumnIndex> column_ids;
	column_ids.emplace_back(0);

	auto &streaming_op =
	    physical_plan->Make<PhysicalStreamingUDF>(return_types, std::move(table_function), std::move(bind_data),
	                                              std::move(column_ids), idx_t(1), vector<column_t>());
	auto gstate = streaming_op.GetGlobalSourceState(*con.context);
	return gstate->MaxThreads();
}

void ExecutePullBasedUDFPlan(Connection &con, const Value &payload) {
	auto &context = *con.context;
	auto physical_plan = make_uniq<PhysicalPlan>(Allocator::DefaultAllocator());
	vector<LogicalType> types = {LogicalType::INTEGER};
	auto &scan = physical_plan->Make<PhysicalDummyScan>(types, idx_t(1));

	vector<string> names = {"out"};
	auto table_function = MakeUDFTableFunction(payload, types, names);
	auto bind_data = make_uniq<UDFFunctionData>(payload, types[0]);
	vector<ColumnIndex> column_ids;
	column_ids.emplace_back(0);
	auto &inout = physical_plan->Make<PhysicalTableInOutFunction>(
	    types, std::move(table_function), std::move(bind_data), std::move(column_ids), idx_t(1), vector<column_t>());
	inout.children.emplace_back(scan);
	physical_plan->SetRoot(inout);

	auto prepared = make_shared_ptr<PreparedStatementData>(StatementType::SELECT_STATEMENT);
	prepared->names = names;
	prepared->types = types;
	prepared->properties.return_type = StatementReturnType::QUERY_RESULT;
	prepared->output_type = QueryResultOutputType::FORCE_MATERIALIZED;
	prepared->memory_type = QueryResultMemoryType::IN_MEMORY;
	prepared->physical_plan = std::move(physical_plan);

	auto sink = PhysicalResultCollector::GetResultCollector(context, *prepared);
	Executor executor(context);
	executor.Initialize(std::move(sink));
	while (!executor.ExecutionIsFinished()) {
		auto result = executor.ExecuteTask();
		if (result == PendingExecutionResult::BLOCKED || result == PendingExecutionResult::NO_TASKS_AVAILABLE) {
			executor.WaitForTask();
		}
		if (executor.HasError()) {
			executor.ThrowException();
		}
	}
}

} // namespace

TEST_CASE("Streaming UDF has one queue-draining source task", "[execution][udf][streaming]") {
	DuckDB db(nullptr);
	Connection con(db);

	REQUIRE(GetStreamingUDFSourceMaxThreads(con, MakeStreamingPayload()) == 1);
}

TEST_CASE("Pull-based UDF physical plans fail instead of falling back", "[execution][udf][streaming]") {
	DuckDB db(nullptr);
	Connection con(db);

	REQUIRE_THROWS_WITH(ExecutePullBasedUDFPlan(con, MakeStreamingPayload()),
	                    Catch::Matchers::Contains("pull-based UDF plans are unsupported"));
}
