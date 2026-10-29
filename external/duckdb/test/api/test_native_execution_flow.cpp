// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"
#include "duckdb/common/atomic.hpp"
#include "test_helpers.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/execution/media_buffer.hpp"
#include "duckdb/execution/operator/projection/physical_media_project.hpp"
#include "duckdb/execution/operator/scan/physical_dummy_scan.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/prepared_statement.hpp"
#include "duckdb/main/stream_query_result.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

using namespace duckdb;

namespace {
struct OpaqueProbe {
	atomic<idx_t> calls {0};
	idx_t output_length = 0;
	bool fail = false;
};

ScalarFunction OpaqueText(shared_ptr<OpaqueProbe> probe) {
	ScalarFunction function("opaque_native_text", {LogicalType::BIGINT}, LogicalType::VARCHAR,
	                        [probe](DataChunk &args, ExpressionState &, Vector &output) {
		                        probe->calls++;
		                        if (probe->fail) {
			                        throw InvalidInputException("injected opaque program failure");
		                        }
		                        auto value = args.GetValue(0, 0);
		                        if (value.IsNull()) {
			                        output.SetValue(0, Value(LogicalType::VARCHAR));
		                        } else {
			                        auto text = probe->output_length ? string(probe->output_length, 'x')
			                                                         : "row-" + value.ToString();
			                        output.SetValue(0, Value(text));
		                        }
	                        });
	function.SetStability(FunctionStability::VOLATILE);
	function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	return function; // No function info, resource hint or replacement callback.
}

ScalarFunction OpaqueNumber(shared_ptr<OpaqueProbe> probe) {
	ScalarFunction function("opaque_native_number", {LogicalType::BIGINT}, LogicalType::BIGINT,
	                        [probe](DataChunk &args, ExpressionState &, Vector &output) {
		                        probe->calls++;
		                        output.SetValue(0, Value::BIGINT(args.GetValue(0, 0).GetValue<int64_t>() * 7));
	                        });
	function.SetStability(FunctionStability::VOLATILE);
	return function;
}

struct FlowFixture {
	explicit FlowFixture(idx_t capacity = 256)
	    : db(nullptr), con(db), thread(*con.context), context(*con.context, thread, nullptr),
	      plan(Allocator::DefaultAllocator()), probe(make_shared_ptr<OpaqueProbe>()) {
		ClientConfig::GetConfig(*con.context).streaming_buffer_size = capacity;
		input_types = {LogicalType::BIGINT, LogicalType::VARCHAR};
		auto function = OpaqueText(probe);
		REQUIRE_FALSE(function.function_info);
		vector<unique_ptr<Expression>> args;
		args.push_back(make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, 0));
		vector<unique_ptr<Expression>> expressions;
		expressions.push_back(make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, 0));
		expressions.push_back(
		    make_uniq<BoundFunctionExpression>(LogicalType::VARCHAR, function, std::move(args), nullptr));
		expressions.push_back(make_uniq<BoundReferenceExpression>(LogicalType::VARCHAR, 1));
		op = make_uniq<PhysicalMediaProject>(
		    plan, vector<LogicalType> {LogicalType::BIGINT, LogicalType::VARCHAR, LogicalType::VARCHAR},
		    std::move(expressions), 5, false);
		op->children.push_back(plan.Make<PhysicalDummyScan>(input_types, 5));
		sink_global = op->GetGlobalSinkState(*con.context);
		sink_local = op->GetLocalSinkState(context);
		source_global = op->GetGlobalSourceState(*con.context);
		source_local = op->GetLocalSourceState(context, *source_global);
		signal = make_shared_ptr<InterruptDoneSignalState>();
	}
	~FlowFixture() {
		op->ResetExecutionState();
	}
	unique_ptr<DataChunk> Input(idx_t rows = 5, idx_t start = 0) {
		auto input = make_uniq<DataChunk>();
		input->Initialize(Allocator::DefaultAllocator(), input_types, rows);
		const idx_t widths[] = {13, 100, 2, 31, 65};
		for (idx_t row = 0; row < rows; row++) {
			input->SetValue(0, row, Value::BIGINT(start + row));
			input->SetValue(1, row, Value(string(widths[row % 5], char('a' + row))));
		}
		input->SetCardinality(rows);
		return input;
	}
	SinkResultType Accept(DataChunk &chunk, bool batch = false) {
		InterruptState interrupt {weak_ptr<InterruptDoneSignalState>(signal)};
		OperatorSinkInput input {*sink_global, *sink_local, interrupt};
		if (batch) {
			ExecutionBatch envelope;
			envelope.materialized = make_uniq<DataChunk>();
			envelope.materialized->InitializeEmpty(input_types);
			envelope.materialized->Reference(chunk);
			return op->SinkBatch(context, envelope, input);
		}
		return op->Sink(context, chunk, input);
	}
	SourceResultType Next(DataChunk &chunk, bool batch = false) {
		chunk.Destroy();
		InterruptState interrupt {weak_ptr<InterruptDoneSignalState>(signal)};
		OperatorSourceInput input {*source_global, *source_local, interrupt};
		if (batch) {
			ExecutionBatch envelope;
			auto result = op->GetDataBatch(context, envelope, input);
			chunk.InitializeEmpty(op->GetTypes());
			chunk.Reference(*envelope.materialized);
			return result;
		}
		return op->GetDataInternal(context, chunk, input);
	}
	DuckDB db;
	Connection con;
	ThreadContext thread;
	ExecutionContext context;
	PhysicalPlan plan;
	shared_ptr<OpaqueProbe> probe;
	vector<LogicalType> input_types;
	unique_ptr<PhysicalMediaProject> op;
	unique_ptr<GlobalSinkState> sink_global;
	unique_ptr<LocalSinkState> sink_local;
	unique_ptr<GlobalSourceState> source_global;
	unique_ptr<LocalSourceState> source_local;
	shared_ptr<InterruptDoneSignalState> signal;
};
} // namespace

TEST_CASE("Opaque native programs wait for output credit before invocation", "[native-execution-flow]") {
	if (!GetNativeExecutionFlow()) {
		return;
	}
	for (auto batch : {false, true}) {
		FlowFixture fixture;
		auto input = fixture.Input();
		MediaFlowWindow upstream(1);
		input->AddMediaFlow(upstream.TakeGrant(*upstream.Request(1)));
		REQUIRE(fixture.Accept(*input, batch) == SinkResultType::NEED_MORE_INPUT);
		CHECK(upstream.Snapshot().reserved.bytes == 0);
		input.reset();
		DataChunk output, retained;
		REQUIRE(fixture.Next(output, batch) == SourceResultType::HAVE_MORE_OUTPUT);
		retained.InitializeEmpty(output.GetTypes());
		retained.Reference(output);
		CHECK(retained.HasMediaFlow());
		for (idx_t retry = 0; retry < 3; retry++) {
			CHECK(fixture.Next(output, batch) == SourceResultType::BLOCKED);
			CHECK(fixture.probe->calls.load() == 1);
		}
		retained.AcknowledgeMediaFlow();
		retained.AcknowledgeMediaFlow();
		for (idx_t row = 1; row < 5; row++) {
			REQUIRE(fixture.Next(output, batch) == SourceResultType::HAVE_MORE_OUTPUT);
			CHECK(output.GetValue(0, 0) == Value::BIGINT(row));
			CHECK(output.GetValue(1, 0) == Value("row-" + std::to_string(row)));
			output.AcknowledgeMediaFlow();
		}
		CHECK(fixture.probe->calls.load() == 5);
		CHECK(retained.GetValue(1, 0) == Value("row-0"));
		CHECK(retained.GetValue(2, 0) == Value(string(13, 'a')));
		CHECK(fixture.Next(output, batch) == SourceResultType::BLOCKED); // Await input, not another function run.
		CHECK(fixture.probe->calls.load() == 5);
	}
}

TEST_CASE("Native flow rejects busy input without acknowledgement and wakes on consumer close",
          "[native-execution-flow]") {
	if (!GetNativeExecutionFlow()) {
		return;
	}
	FlowFixture fixture;
	auto first = fixture.Input();
	REQUIRE(fixture.Accept(*first) == SinkResultType::NEED_MORE_INPUT);
	auto second = fixture.Input(1, 5);
	MediaFlowWindow upstream(1);
	second->AddMediaFlow(upstream.TakeGrant(*upstream.Request(1)));
	CHECK(fixture.Accept(*second) == SinkResultType::BLOCKED);
	CHECK(upstream.Snapshot().reserved.bytes == 1);
	DataChunk output, retained;
	REQUIRE(fixture.Next(output) == SourceResultType::HAVE_MORE_OUTPUT);
	retained.InitializeEmpty(output.GetTypes());
	retained.Reference(output);
	CHECK(fixture.Next(output) == SourceResultType::BLOCKED);
	fixture.source_global->OnConsumerFinished();
	CHECK(fixture.Next(output) == SourceResultType::FINISHED);
	CHECK(fixture.Accept(*second) == SinkResultType::FINISHED);
	CHECK(upstream.Snapshot().reserved.bytes == 1);
	CHECK(fixture.probe->calls.load() == 1);
	CHECK(retained.GetValue(1, 0) == Value("row-0"));
}

TEST_CASE("Native flow copy failure and input oversize never acknowledge input", "[native-execution-flow]") {
	if (!GetNativeExecutionFlow()) {
		return;
	}
	FlowFixture fixture;
	auto input = fixture.Input();
	MediaFlowWindow upstream(1);
	input->AddMediaFlow(upstream.TakeGrant(*upstream.Request(1)));
	REQUIRE_NO_FAIL(fixture.con.Query("SET memory_limit='1B'"));
	REQUIRE_THROWS_AS(fixture.Accept(*input), OutOfMemoryException);
	CHECK(upstream.Snapshot().reserved.bytes == 1);
	CHECK(fixture.probe->calls.load() == 0);
	REQUIRE_NO_FAIL(fixture.con.Query("SET memory_limit='10MiB'"));
	input->SetValue(1, 2, Value(string(300, 'z')));
	REQUIRE_THROWS_WITH(fixture.Accept(*input), Catch::Matchers::Contains("input row exceeds"));
	CHECK(upstream.Snapshot().reserved.bytes == 1);
	CHECK(fixture.probe->calls.load() == 0);
	input->SetValue(1, 2, Value("cc"));
	REQUIRE(fixture.Accept(*input) == SinkResultType::NEED_MORE_INPUT);
	CHECK(upstream.Snapshot().reserved.bytes == 0);
}

TEST_CASE("Native flow output oversize and kernel failure do not publish partial output", "[native-execution-flow]") {
	if (!GetNativeExecutionFlow()) {
		return;
	}
	for (auto fail : {false, true}) {
		FlowFixture fixture;
		fixture.probe->fail = fail;
		fixture.probe->output_length = 300;
		auto input = fixture.Input();
		REQUIRE(fixture.Accept(*input) == SinkResultType::NEED_MORE_INPUT);
		DataChunk output;
		if (fail) {
			REQUIRE_THROWS_WITH(fixture.Next(output), Catch::Matchers::Contains("opaque program failure"));
		} else {
			REQUIRE_THROWS_WITH(fixture.Next(output), Catch::Matchers::Contains("output row exceeds"));
		}
		CHECK_FALSE(output.HasMediaFlow());
		CHECK(fixture.probe->calls.load() == 1);
		fixture.source_global->OnConsumerFinished();
		CHECK(fixture.Next(output) == SourceResultType::FINISHED);
	}
}

TEST_CASE("Opaque native flow preserves CASE nested programs lists and prepared reuse", "[native-execution-flow]") {
	if (!GetNativeExecutionFlow()) {
		return;
	}
	DuckDB db;
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET threads=1"));
	REQUIRE_NO_FAIL(con.Query("SET streaming_buffer_size='256B'"));
	auto text = make_shared_ptr<OpaqueProbe>();
	auto numbers = make_shared_ptr<OpaqueProbe>();
	ExtensionLoader loader(*db.instance, "native_execution_probe");
	loader.RegisterFunction(OpaqueText(text));
	loader.RegisterFunction(OpaqueNumber(numbers));
	auto prepared = con.Prepare("SELECT i,CASE WHEN i%2=0 THEN upper(opaque_native_text(i)) ELSE 'skip' END AS s,"
	                            "opaque_native_number(i) AS n,[i,i+1] AS a FROM range(6)t(i)");
	REQUIRE_FALSE(prepared->HasError());
	for (idx_t iteration = 0; iteration < 2; iteration++) {
		auto result = prepared->Execute();
		REQUIRE_NO_FAIL(*result);
		idx_t count = 0;
		while (auto chunk = result->Fetch()) {
			for (idx_t row = 0; row < chunk->size(); row++) {
				CHECK(chunk->GetValue(0, row) == Value::BIGINT(count));
				CHECK(chunk->GetValue(1, row) == Value(count % 2 ? "skip" : "ROW-" + std::to_string(count)));
				CHECK(chunk->GetValue(2, row) == Value::BIGINT(count * 7));
				auto list = ListValue::GetChildren(chunk->GetValue(3, row));
				REQUIRE(list.size() == 2);
				CHECK(list[0] == Value::BIGINT(count));
				CHECK(list[1] == Value::BIGINT(count + 1));
				count++;
			}
		}
		CHECK(count == 6);
		CHECK(text->calls.load() == (iteration + 1) * 3);
		CHECK(numbers->calls.load() == (iteration + 1) * 6);
	}
	for (idx_t iteration = 0; iteration < 2; iteration++) {
		auto limited = con.Query("SELECT opaque_native_text(i) FROM range(100)t(i) LIMIT 1");
		REQUIRE_NO_FAIL(*limited);
		CHECK(limited->RowCount() == 1);
	}
	CHECK(text->calls.load() == 8);
	auto empty = con.Query("SELECT opaque_native_text(i) FROM range(0)t(i)");
	REQUIRE_NO_FAIL(*empty);
	CHECK(empty->RowCount() == 0);
	CHECK(text->calls.load() == 8);
}

TEST_CASE("Bounded native results become ready before upstream flow pipelines finish", "[native-execution-flow]") {
	if (!GetNativeExecutionFlow()) {
		return;
	}
	DuckDB db;
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET threads=4"));
	REQUIRE_NO_FAIL(con.Query("SET streaming_buffer_size='32B'"));
	auto probe = make_shared_ptr<OpaqueProbe>();
	ExtensionLoader loader(*db.instance, "native_execution_stream");
	loader.RegisterFunction(OpaqueNumber(probe));
	const auto sql = "SELECT opaque_native_number(j) FROM "
	                 "(SELECT opaque_native_number(i) AS j FROM range(128)t(i) LIMIT 64)t";
	auto result = con.SendQuery(sql);
	REQUIRE_NO_FAIL(*result);
	REQUIRE(result->type == QueryResultType::STREAM_RESULT);
	auto first = result->Fetch();
	REQUIRE(first);
	CHECK(first->GetValue(0, 0) == Value::BIGINT(0));
	CHECK(probe->calls.load() < 128);
	result->Cast<StreamQueryResult>().Close();
	CHECK(first->GetValue(0, 0) == Value::BIGINT(0));
	REQUIRE_NO_FAIL(con.Query("SELECT 7"));
	auto streamed = con.SendQuery(sql);
	REQUIRE_NO_FAIL(*streamed);
	idx_t rows = 0;
	while (auto chunk = streamed->Fetch()) {
		for (idx_t row = 0; row < chunk->size(); row++) {
			CHECK(chunk->GetValue(0, row) == Value::BIGINT(rows++ * 49));
		}
	}
	REQUIRE_NO_FAIL(*streamed);
	CHECK(rows == 64);
	auto repeated = con.Query(sql);
	REQUIRE_NO_FAIL(*repeated);
	CHECK(repeated->RowCount() == 64);
	CHECK(repeated->GetValue(0, 63) == Value::BIGINT(63 * 49));
}

TEST_CASE("Opaque native flow aliases outlive their connection", "[native-execution-flow]") {
	if (!GetNativeExecutionFlow()) {
		return;
	}
	unique_ptr<DataChunk> retained;
	{
		FlowFixture fixture;
		auto input = fixture.Input();
		REQUIRE(fixture.Accept(*input) == SinkResultType::NEED_MORE_INPUT);
		retained = make_uniq<DataChunk>();
		REQUIRE(fixture.Next(*retained) == SourceResultType::HAVE_MORE_OUTPUT);
	}
	CHECK(retained->GetValue(1, 0) == Value("row-0"));
	CHECK(retained->GetValue(2, 0) == Value(string(13, 'a')));
	retained->AcknowledgeMediaFlow();
	retained.reset();
}

TEST_CASE("Native flow plans serialize policy independently of function hints", "[native-execution-flow]") {
	if (!GetNativeExecutionFlow()) {
		return;
	}
	FlowFixture fixture;
	ExtensionLoader loader(*fixture.db.instance, "native_execution_serde");
	loader.RegisterFunction(OpaqueText(fixture.probe));
	MemoryStream stream;
	BinarySerializer::Serialize(*fixture.op, stream);
	fixture.con.context->RunFunctionInTransaction([&]() {
		stream.Rewind();
		BinaryDeserializer deserializer(stream);
		deserializer.Set<ClientContext &>(*fixture.con.context);
		PhysicalPlan decoded(Allocator::DefaultAllocator());
		deserializer.Begin();
		auto root = decoded.Deserialize(deserializer);
		deserializer.End();
		REQUIRE(root->type == PhysicalOperatorType::MEDIA_PROJECT);
		CHECK_FALSE(root->Cast<PhysicalMediaProject>().resource_admission);
		CHECK(root->GetName() == "NATIVE_PROJECT");
		CHECK(fixture.probe->calls.load() == 0);
	});
}
