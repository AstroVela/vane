// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"
#include "test_helpers.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/types/image.hpp"
#include "duckdb/execution/managed_media_scalar.hpp"
#include "duckdb/execution/media_buffer.hpp"
#include "duckdb/execution/operator/projection/physical_media_project.hpp"
#include "duckdb/execution/operator/scan/physical_dummy_scan.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/pending_query_result.hpp"
#include "duckdb/main/prepared_statement.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "image_resize_admission.hpp"

using namespace duckdb;

namespace {
bool ScalarAdmissionEnabled() {
	// CI runs this tag explicitly with the opt-in flag and a fresh host process.
	// The ordinary native suite must continue to exercise the disabled default.
	return GetNativeMediaScalarAdmission();
}

struct ScalarProbe {
	idx_t executed = 0;
	bool fail = false;
};

ScalarFunction ManagedFunction(shared_ptr<ScalarProbe> probe, idx_t scratch = 17) {
	auto input = ImageLogicalType::Create("RGB", 2, 2);
	auto output = ImageLogicalType::Create("RGB", 1, 1);
	ScalarFunction function("test_managed_resize", {input, LogicalType::BIGINT, LogicalType::BIGINT}, output,
	                        [](DataChunk &, ExpressionState &, Vector &) {
		                        throw InternalException("managed test called the ordinary scalar callback");
	                        });
	function.SetStability(FunctionStability::VOLATILE);
	function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	function.SetExtraFunctionInfo(make_shared_ptr<ManagedMediaScalarFunctionInfo>(
	    [scratch](ClientContext &, const BoundFunctionExpression &) {
		    return ManagedMediaWorkingSet {3, scratch};
	    },
	    [probe](ClientContext &, DataChunk &args, Vector &output, MediaResourceLease lease, shared_ptr<void> lifetime) {
		    AllocateMediaArray(output, 1, std::move(lease), std::move(lifetime));
		    if (probe->fail) {
			    throw InvalidInputException("injected managed scalar failure");
		    }
		    probe->executed++;
		    if (FlatVector::IsNull(args.data[0], 0)) {
			    FlatVector::SetNull(output, 0, true);
			    return;
		    }
		    auto input = FlatVector::GetData<uint8_t>(ArrayVector::GetEntry(args.data[0]));
		    auto target = FlatVector::GetData<uint8_t>(ArrayVector::GetEntry(output));
		    for (idx_t i = 0; i < 3; i++) {
			    target[i] = input[i];
		    }
	    }));
	return function;
}

struct ScalarFixture {
	explicit ScalarFixture(idx_t copy_limit = 40, bool real_resize = false, bool antialias = false, idx_t scratch = 17)
	    : db(nullptr), con(db), thread(*con.context), context(*con.context, thread, nullptr),
	      plan(Allocator::DefaultAllocator()), probe(make_shared_ptr<ScalarProbe>()) {
		ClientConfig::GetConfig(*con.context).streaming_buffer_size = copy_limit;
		input_types = {LogicalType::BIGINT, ImageLogicalType::Create("RGB", 2, 2)};
		auto function = ManagedFunction(probe, scratch);
		if (real_resize) {
			function.SetExtraFunctionInfo(make_shared_ptr<ManagedMediaScalarFunctionInfo>(
			    ManagedImageResize::WorkingSet, ManagedImageResize::Execute));
		}
		vector<unique_ptr<Expression>> args;
		args.push_back(make_uniq<BoundReferenceExpression>(input_types[1], 1));
		args.push_back(make_uniq<BoundConstantExpression>(Value::BIGINT(1)));
		args.push_back(make_uniq<BoundConstantExpression>(Value::BIGINT(1)));
		if (antialias) {
			function.arguments.push_back(LogicalType::BOOLEAN);
			args.push_back(make_uniq<BoundConstantExpression>(Value::BOOLEAN(true)));
		}
		vector<unique_ptr<Expression>> expressions;
		expressions.push_back(make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, 0));
		expressions.push_back(
		    make_uniq<BoundFunctionExpression>(function.return_type, function, std::move(args), nullptr));
		op = make_uniq<PhysicalMediaProject>(plan, vector<LogicalType> {LogicalType::BIGINT, function.return_type},
		                                     std::move(expressions), 5);
		op->children.push_back(plan.Make<PhysicalDummyScan>(input_types, 5));
		sink_global = op->GetGlobalSinkState(*con.context);
		sink_local = op->GetLocalSinkState(context);
		source_global = op->GetGlobalSourceState(*con.context);
		source_local = op->GetLocalSourceState(context, *source_global);
		signal = make_shared_ptr<InterruptDoneSignalState>();
	}
	~ScalarFixture() {
		op->ResetExecutionState();
	}
	unique_ptr<DataChunk> Input(idx_t rows = 5, idx_t start = 0, bool null_row = false) {
		auto input = make_uniq<DataChunk>();
		input->Initialize(Allocator::DefaultAllocator(), input_types, rows);
		for (idx_t row = 0; row < rows; row++) {
			input->SetValue(0, row, Value::BIGINT(start + row));
			uint8_t pixels[12];
			memset(pixels, uint8_t(start + row), sizeof(pixels));
			ImageVector::WritePixels(input->data[1], row, 2, 2, "RGB", pixels);
		}
		if (null_row) {
			FlatVector::SetNull(input->data[1], 1, true);
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
		InterruptState interrupt {weak_ptr<InterruptDoneSignalState>(signal)};
		OperatorSourceInput input {*source_global, *source_local, interrupt};
		if (batch) {
			chunk.Destroy();
			ExecutionBatch envelope;
			auto result = op->GetDataBatch(context, envelope, input);
			chunk.Destroy();
			chunk.InitializeEmpty(op->GetTypes());
			chunk.Reference(*envelope.materialized);
			return result;
		}
		return op->GetDataInternal(context, chunk, input);
	}
	shared_ptr<MediaResourcePool> Pool() {
		return db.instance->GetMediaOutputPool();
	}
	DuckDB db;
	Connection con;
	ThreadContext thread;
	ExecutionContext context;
	PhysicalPlan plan;
	shared_ptr<ScalarProbe> probe;
	vector<LogicalType> input_types;
	unique_ptr<PhysicalMediaProject> op;
	unique_ptr<GlobalSinkState> sink_global;
	unique_ptr<LocalSinkState> sink_local;
	unique_ptr<GlobalSourceState> source_global;
	unique_ptr<LocalSourceState> source_local;
	shared_ptr<InterruptDoneSignalState> signal;
};

unique_ptr<QueryResult> CompleteScalar(PendingQueryResult &pending) {
	for (idx_t task = 0; task < 10000; task++) {
		auto status = pending.ExecuteTask();
		if (status == PendingExecutionResult::EXECUTION_ERROR) {
			return make_uniq<MaterializedQueryResult>(pending.GetErrorObject());
		}
		if (PendingQueryResult::IsResultReady(status)) {
			return pending.Execute();
		}
	}
	FAIL("Managed scalar failed to progress within its deterministic task budget");
	return nullptr;
}
} // namespace

TEST_CASE("Managed scalar receives before waiting for a complete working set", "[media-scalar]") {
	if (!ScalarAdmissionEnabled()) {
		return;
	}
	ScalarFixture fixture;
	auto scope = fixture.Pool()->CreateScope();
	auto upstream = scope->Request(fixture.Pool()->Snapshot().capacity)->TakeGrant();
	auto input = fixture.Input();
	MediaFlowWindow window(1);
	input->AddMediaFlow(window.TakeGrant(*window.Request(1)));
	REQUIRE(fixture.Accept(*input) == SinkResultType::NEED_MORE_INPUT);
	CHECK(window.Snapshot().reserved.bytes == 0);
	input.reset(); // The sealed copy does not borrow the original input.
	DataChunk output;
	CHECK(fixture.Next(output) == SourceResultType::BLOCKED);
	CHECK(fixture.probe->executed == 0);
	upstream.Release();
	REQUIRE(fixture.Next(output) == SourceResultType::HAVE_MORE_OUTPUT);
	CHECK(output.GetValue(0, 0) == Value::BIGINT(0));
	CHECK(fixture.probe->executed == 1);
	// 40-byte cache + 20-byte owned row + 3-byte output + 17-byte scratch.
	CHECK(fixture.Pool()->Snapshot().reserved.bytes == 80);
}

TEST_CASE("Managed scalar resumes exact rows across prefix and output backpressure", "[media-scalar]") {
	if (!ScalarAdmissionEnabled()) {
		return;
	}
	for (auto batch : {false, true}) {
		for (auto retain_input : {false, true}) {
			ScalarFixture fixture;
			auto input = fixture.Input(5, 0, true);
			REQUIRE(fixture.Accept(*input, batch) == SinkResultType::NEED_MORE_INPUT);
			DataChunk output;
			REQUIRE(fixture.Next(output, batch) == SourceResultType::HAVE_MORE_OUTPUT);
			auto retained = make_uniq<Vector>(output.data[retain_input ? 0 : 1]);
			output.Destroy();
			CHECK(fixture.Next(output, batch) == SourceResultType::BLOCKED);
			CHECK(fixture.probe->executed == 1);
			retained.reset();
			for (idx_t row = 1; row < 5; row++) {
				REQUIRE(fixture.Next(output, batch) == SourceResultType::HAVE_MORE_OUTPUT);
				CHECK(output.GetValue(0, 0) == Value::BIGINT(row));
				CHECK(FlatVector::IsNull(output.data[1], 0) == (row == 1));
				if (row != 1) {
					CHECK(FlatVector::GetData<uint8_t>(ArrayVector::GetEntry(output.data[1]))[0] == row);
				}
			}
			output.Destroy();
			CHECK(fixture.Next(output, batch) == SourceResultType::BLOCKED); // Await another input, not EOF.
			CHECK(fixture.probe->executed == 5);
			CHECK(fixture.Pool()->Snapshot().reserved.bytes == 0);
		}
	}
}

TEST_CASE("Managed scalar rejected input is not acknowledged and early finish cancels admission", "[media-scalar]") {
	if (!ScalarAdmissionEnabled()) {
		return;
	}
	ScalarFixture fixture;
	auto scope = fixture.Pool()->CreateScope();
	auto upstream = scope->Request(fixture.Pool()->Snapshot().capacity)->TakeGrant();
	auto first = fixture.Input();
	REQUIRE(fixture.Accept(*first) == SinkResultType::NEED_MORE_INPUT);
	auto second = fixture.Input(1, 5);
	MediaFlowWindow window(1);
	second->AddMediaFlow(window.TakeGrant(*window.Request(1)));
	CHECK(fixture.Accept(*second) == SinkResultType::BLOCKED);
	CHECK(window.Snapshot().reserved.bytes == 1);
	DataChunk output;
	CHECK(fixture.Next(output) == SourceResultType::BLOCKED);
	fixture.source_global->OnConsumerFinished();
	CHECK(fixture.Pool()->Snapshot().waiting_requests == 0);
	CHECK(fixture.Accept(*second) == SinkResultType::FINISHED);
	CHECK(window.Snapshot().reserved.bytes == 1);
	CHECK(fixture.probe->executed == 0);
}

TEST_CASE("Managed scalar detached pixels retain ancestor reservation after teardown", "[media-scalar]") {
	if (!ScalarAdmissionEnabled()) {
		return;
	}
	unique_ptr<Vector> retained;
	shared_ptr<MediaResourcePool> pool;
	{
		ScalarFixture fixture;
		pool = fixture.Pool();
		auto input = fixture.Input();
		REQUIRE(fixture.Accept(*input) == SinkResultType::NEED_MORE_INPUT);
		DataChunk output;
		REQUIRE(fixture.Next(output) == SourceResultType::HAVE_MORE_OUTPUT);
		retained = make_uniq<Vector>(output.data[1]);
	}
	CHECK(pool->Snapshot().reserved.bytes == 80);
	CHECK(FlatVector::GetData<uint8_t>(ArrayVector::GetEntry(*retained))[0] == 0);
	retained.reset();
	CHECK(pool->Snapshot().reserved.bytes == 0);
}

TEST_CASE("Managed scalar failures return all reservations", "[media-scalar]") {
	if (!ScalarAdmissionEnabled()) {
		return;
	}
	shared_ptr<MediaResourcePool> pool;
	{
		ScalarFixture fixture;
		pool = fixture.Pool();
		fixture.probe->fail = true;
		auto input = fixture.Input();
		REQUIRE(fixture.Accept(*input) == SinkResultType::NEED_MORE_INPUT);
		DataChunk output;
		REQUIRE_THROWS_WITH(fixture.Next(output), Catch::Matchers::Contains("injected managed scalar failure"));
	}
	CHECK(pool->Snapshot().reserved.bytes == 0);
	CHECK(pool->Snapshot().waiting_requests == 0);
}

TEST_CASE("Managed scalar SQL plans stream with one thread and reject unsupported contexts", "[media-scalar]") {
	if (!ScalarAdmissionEnabled()) {
		return;
	}
	DuckDB db;
	Connection con(db);
	auto probe = make_shared_ptr<ScalarProbe>();
	ExtensionLoader loader(*db.instance, "media_scalar_test");
	loader.RegisterFunction(ManagedFunction(probe));
	REQUIRE_NO_FAIL(con.Query("SET threads=1"));
	REQUIRE_NO_FAIL(con.Query("SET streaming_buffer_size='128B'"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE scalar_input AS SELECT i, "
	                          "image(repeat(chr(65 + i::INTEGER), 12)::BLOB, 2, 2, 3, 'RGB')::IMAGE('RGB',2,2) AS data "
	                          "FROM range(9) t(i)"));
	auto pending = con.PendingQuery("SELECT i, test_managed_resize(data, 1, 1) AS data FROM scalar_input");
	REQUIRE_FALSE(pending->HasError());
	auto result = CompleteScalar(*pending);
	REQUIRE_NO_FAIL(*result);
	idx_t row_count = 0;
	while (auto chunk = result->Fetch()) {
		for (idx_t row = 0; row < chunk->size(); row++) {
			CHECK(chunk->GetValue(0, row) == Value::BIGINT(row_count));
			CHECK(FlatVector::GetData<uint8_t>(ArrayVector::GetEntry(chunk->data[1]))[row * 3] == 65 + row_count);
			row_count++;
		}
	}
	CHECK(row_count == 9);
	CHECK(probe->executed == 9);
	REQUIRE_NO_FAIL(con.Query("SET preserve_insertion_order=false"));
	auto ordered = con.Query("SELECT i, test_managed_resize(data,1,1) FROM "
	                         "(SELECT * FROM scalar_input ORDER BY i DESC) t");
	REQUIRE_NO_FAIL(*ordered);
	row_count = 0;
	while (auto chunk = ordered->Fetch()) {
		for (idx_t row = 0; row < chunk->size(); row++) {
			CHECK(chunk->GetValue(0, row) == Value::BIGINT(8 - row_count));
			row_count++;
		}
	}
	CHECK(row_count == 9);
	auto prepared = con.Prepare("SELECT i,test_managed_resize(data,1,1) FROM scalar_input LIMIT 2");
	REQUIRE_FALSE(prepared->HasError());
	for (idx_t iteration = 0; iteration < 2; iteration++) {
		auto limited = prepared->Execute();
		REQUIRE_NO_FAIL(*limited);
		idx_t rows = 0;
		while (auto chunk = limited->Fetch()) {
			rows += chunk->size();
		}
		CHECK(rows == 2);
		CHECK(db.instance->GetMediaOutputPool()->Snapshot().reserved.bytes == 0);
	}
	CHECK(probe->executed == 22);
	REQUIRE_NO_FAIL(con.Query("SET disabled_optimizers='constant_folding'"));
	auto constant_dimensions = con.Query("SELECT i,test_managed_resize(data,1+0,1) FROM scalar_input LIMIT 1");
	REQUIRE_NO_FAIL(*constant_dimensions);
	idx_t constant_rows = 0;
	while (auto chunk = constant_dimensions->Fetch()) {
		constant_rows += chunk->size();
	}
	CHECK(constant_rows == 1);
	CHECK(probe->executed == 23);
	REQUIRE_NO_FAIL(con.Query("SET disabled_optimizers=''"));
	for (auto sql : {"SELECT CASE WHEN i=0 THEN test_managed_resize(data,1,1) ELSE NULL END FROM scalar_input",
	                 "SELECT i FROM scalar_input WHERE test_managed_resize(data,1,1) IS NOT NULL",
	                 "SELECT test_managed_resize(data,1,1),test_managed_resize(data,2,1) FROM scalar_input"}) {
		auto rejected = con.Query(sql);
		REQUIRE(rejected->HasError());
		CHECK(rejected->GetError().find("Managed media") != string::npos);
	}
	CHECK(db.instance->GetMediaOutputPool()->Snapshot().reserved.bytes == 0);
}

TEST_CASE("Managed scalar rejects a receiving capacity smaller than one row", "[media-scalar]") {
	if (!ScalarAdmissionEnabled()) {
		return;
	}
	REQUIRE_THROWS_WITH(ScalarFixture(19), Catch::Matchers::Contains("streaming_buffer_size receiving capacity"));
	REQUIRE_THROWS_WITH(ScalarFixture(40, false, false, GetNativeMediaOutputPool()->Snapshot().capacity.bytes),
	                    Catch::Matchers::Contains("minimum working set"));
	CHECK(GetNativeMediaOutputPool()->Snapshot().reserved.bytes == 0);
	ScalarFixture fixture;
	auto input = fixture.Input(1);
	REQUIRE(fixture.Accept(*input) == SinkResultType::NEED_MORE_INPUT);
	DataChunk output;
	REQUIRE(fixture.Next(output) == SourceResultType::HAVE_MORE_OUTPUT);
	// A short input reserves its actual 20-byte prefix, not the 40-byte maximum.
	CHECK(fixture.Pool()->Snapshot().reserved.bytes == 60);
}

TEST_CASE("Managed scalar copy failure leaves input unacknowledged and retryable", "[media-scalar]") {
	if (!ScalarAdmissionEnabled()) {
		return;
	}
	ScalarFixture fixture;
	auto input = fixture.Input();
	MediaFlowWindow window(1);
	input->AddMediaFlow(window.TakeGrant(*window.Request(1)));
	REQUIRE_NO_FAIL(fixture.con.Query("SET memory_limit='1B'"));
	REQUIRE_THROWS_AS(fixture.Accept(*input), OutOfMemoryException);
	CHECK(window.Snapshot().reserved.bytes == 1);
	CHECK(fixture.Pool()->Snapshot().reserved.bytes == 0);
	CHECK(fixture.Pool()->Snapshot().waiting_requests == 0);
	REQUIRE_NO_FAIL(fixture.con.Query("SET memory_limit='10MiB'"));
	REQUIRE(fixture.Accept(*input) == SinkResultType::NEED_MORE_INPUT);
	CHECK(window.Snapshot().reserved.bytes == 0);
	DataChunk output;
	REQUIRE(fixture.Next(output) == SourceResultType::HAVE_MORE_OUTPUT);
	CHECK(output.GetValue(0, 0) == Value::BIGINT(0));
}

TEST_CASE("Managed native resize kernel reserves antialias scratch and preserves NULLs", "[media-scalar]") {
	if (!ScalarAdmissionEnabled()) {
		return;
	}
	for (auto antialias : {false, true}) {
		ScalarFixture fixture(40, true, antialias);
		auto input = fixture.Input(3, 0, true);
		uint8_t pixels[] = {0, 10, 20, 40, 50, 60, 80, 90, 100, 120, 130, 140};
		ImageVector::WritePixels(input->data[1], 0, 2, 2, "RGB", pixels);
		REQUIRE(fixture.Accept(*input) == SinkResultType::NEED_MORE_INPUT);
		DataChunk output;
		REQUIRE(fixture.Next(output) == SourceResultType::HAVE_MORE_OUTPUT);
		auto values = FlatVector::GetData<uint8_t>(ArrayVector::GetEntry(output.data[1]));
		CHECK(values[0] == 60);
		CHECK(values[1] == 70);
		CHECK(values[2] == 80);
		CHECK(fixture.Pool()->Snapshot().reserved.bytes == (antialias ? 111 : 63));
		REQUIRE(fixture.Next(output) == SourceResultType::HAVE_MORE_OUTPUT);
		CHECK(FlatVector::IsNull(output.data[1], 0));
	}
}

TEST_CASE("Managed scalar serialization recovers catalog capability without runtime leases", "[media-scalar]") {
	if (!ScalarAdmissionEnabled()) {
		return;
	}
	ScalarFixture fixture;
	ExtensionLoader loader(*fixture.db.instance, "media_scalar_serde_test");
	loader.RegisterFunction(ManagedFunction(fixture.probe));
	MemoryStream before;
	BinarySerializer::Serialize(*fixture.op, before);
	auto input = fixture.Input();
	REQUIRE(fixture.Accept(*input) == SinkResultType::NEED_MORE_INPUT);
	MemoryStream during;
	BinarySerializer::Serialize(*fixture.op, during);
	REQUIRE(before.GetPosition() == during.GetPosition());
	CHECK(memcmp(before.GetData(), during.GetData(), before.GetPosition()) == 0);
	fixture.op->ResetExecutionState();
	fixture.source_local.reset();
	fixture.source_global.reset();
	fixture.sink_local.reset();
	fixture.sink_global.reset();
	CHECK(fixture.Pool()->Snapshot().reserved.bytes == 0);
	PhysicalPlan decoded(Allocator::DefaultAllocator());
	fixture.con.context->RunFunctionInTransaction([&]() {
		before.Rewind();
		BinaryDeserializer deserializer(before);
		deserializer.Set<ClientContext &>(*fixture.con.context);
		deserializer.Begin();
		auto root = decoded.Deserialize(deserializer);
		deserializer.End();
		REQUIRE(root->type == PhysicalOperatorType::MEDIA_PROJECT);
		auto &projection = root->Cast<PhysicalMediaProject>();
		REQUIRE(projection.select_list.size() == 2);
		CHECK(GetManagedMediaScalar(*projection.select_list[1]) != nullptr);
		CHECK(projection.children.size() == 1);
		CHECK(projection.children[0].get().GetTypes() == fixture.input_types);
		CHECK(fixture.Pool()->Snapshot().reserved.bytes == 0);
	});
}
