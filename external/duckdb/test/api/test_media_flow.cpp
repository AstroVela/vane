// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"
#include "test_helpers.hpp"
#include "duckdb/common/types/image.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/execution/media_buffer.hpp"
#include "duckdb/execution/operator/projection/physical_udf_inout.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/execution/udf_executor.hpp"
#include "duckdb/function/scalar/udf_functions.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/pending_query_result.hpp"
#include "duckdb/parallel/thread_context.hpp"

#include <thread>
#include <cstdlib>
#include <type_traits>

using namespace duckdb;

static_assert(!std::is_copy_constructible<MediaFlowWindow>::value,
              "Copying a scoped flow window could close credits still owned by its producer");

namespace {
shared_ptr<MediaFlowTicket> TakeFlow(MediaFlowWindow &window, idx_t bytes) {
	auto request = window.Request(bytes);
	REQUIRE(request->PollOrArm([] {}) == MediaAdmissionStatus::GRANTED);
	return window.TakeGrant(*request);
}

// Default to refusing task admission. The real UDF ingress must still accept
// only its reserved byte window, and must distinguish accepted BLOCKED from
// rejected BLOCKED without duplicating the input on retry.
struct SubmitProbe {
	idx_t accepts_remaining = 0;
	idx_t accepted = 0;
	vector<idx_t> proposed_bytes;
	vector<idx_t> actual_bytes;
	vector<vector<idx_t>> chunk_bytes;
};

class RefusingExecutor : public UDFExecutor {
public:
	explicit RefusingExecutor(shared_ptr<SubmitProbe> probe_p) : probe(std::move(probe_p)) {
	}
	idx_t Submit(DataChunk &, DataChunk &, ClientContext &) override {
		throw InternalException("unexpected blocking submit");
	}
	bool TrySubmit(DataChunk &, DataChunk &, ClientContext &, idx_t &) override {
		return false;
	}
	bool TrySubmitWithRetainedBytes(DataChunk &, DataChunk &, ClientContext &, idx_t, idx_t &) override {
		return false;
	}
	bool TrySubmitEnvelope(vector<unique_ptr<DataChunk>> &, DataChunk &, ClientContext &, idx_t &) override {
		return false;
	}
	bool TrySubmitEnvelopeWithRetainedBytes(vector<unique_ptr<DataChunk>> &chunks, DataChunk &, ClientContext &,
	                                        idx_t bytes, idx_t &submit_id) override {
		idx_t actual_bytes = 0;
		vector<idx_t> chunk_bytes;
		for (auto &chunk : chunks) {
			auto payload_bytes = EstimateMediaFlowInputBytes(*chunk);
			actual_bytes += payload_bytes;
			chunk_bytes.push_back(payload_bytes);
		}
		probe->proposed_bytes.push_back(bytes);
		probe->actual_bytes.push_back(actual_bytes);
		probe->chunk_bytes.push_back(std::move(chunk_bytes));
		if (!probe->accepts_remaining) {
			return false;
		}
		probe->accepts_remaining--;
		submit_id = ++probe->accepted;
		return true;
	}
	bool SupportsRefBundleInput() override {
		return false;
	}
	idx_t SubmitRefBundle(LazyRefDataChunk &, DataChunk &, ClientContext &) override {
		throw InternalException("unexpected ref submit");
	}
	bool TrySubmitRefBundle(LazyRefDataChunk &, DataChunk &, ClientContext &, idx_t &) override {
		return false;
	}
	bool TrySubmitRefBundleWithRetainedBytes(LazyRefDataChunk &, DataChunk &, ClientContext &, idx_t,
	                                         idx_t &) override {
		return false;
	}
	void FinishedSubmitting(ClientContext &) override {
	}
	void Cancel() override {
	}
	bool AllTasksFinished(ClientContext &) override {
		return false;
	}
	bool SupportsAsyncWakeup() override {
		return true;
	}
	UDFWakeupRegistrationResult RegisterWakeup(InterruptState &) override {
		return UDFWakeupRegistrationResult::ARMED;
	}
	void RegisterWakeupCallback(std::function<void()>) override {
	}
	void RegisterOutputConsumer(UDFOutputConsumer) override {
	}
	void NotifyOutputConsumerSpaceAvailable() override {
	}
	idx_t DebugSlotId() override {
		return 0;
	}
	InsertionOrderPreservingMap<string> Stats() override {
		return {};
	}

private:
	shared_ptr<SubmitProbe> probe;
};

unique_ptr<UDFExecutor> RefuseTasks(ClientContext &, const Value &, UDFConfig &, shared_ptr<void> handles) {
	auto probe = handles ? shared_ptr_cast<void, SubmitProbe>(handles) : make_shared_ptr<SubmitProbe>();
	return make_uniq<RefusingExecutor>(std::move(probe));
}

struct CopyProbe {
	bool fail = false;
};
struct CopyAllocatorData : public PrivateAllocatorData {
	explicit CopyAllocatorData(shared_ptr<CopyProbe> probe_p) : probe(std::move(probe_p)) {
	}
	shared_ptr<CopyProbe> probe;
};
DBConfig *ConfigureCopyAllocator(DBConfig &config, shared_ptr<CopyProbe> probe) {
	config.allocator = make_uniq<Allocator>(
	    [](PrivateAllocatorData *data, idx_t bytes) -> data_ptr_t {
		    if (data->Cast<CopyAllocatorData>().probe->fail) {
			    throw OutOfMemoryException("injected ingress allocation failure");
		    }
		    return static_cast<data_ptr_t>(std::malloc(bytes));
	    },
	    [](PrivateAllocatorData *, data_ptr_t pointer, idx_t) { std::free(pointer); },
	    [](PrivateAllocatorData *, data_ptr_t pointer, idx_t, idx_t bytes) -> data_ptr_t {
		    return static_cast<data_ptr_t>(std::realloc(pointer, bytes));
	    },
	    make_uniq<CopyAllocatorData>(std::move(probe)));
	return &config;
}

struct SinkFixture {
	SinkFixture(idx_t input_bytes, idx_t batch_rows = 4, bool preserve_work_units = true)
	    : probe(make_shared_ptr<CopyProbe>()), submits(make_shared_ptr<SubmitProbe>()),
	      db(nullptr, ConfigureCopyAllocator(config, probe)), con(db), thread(*con.context),
	      context(*con.context, thread, nullptr), plan(Allocator::DefaultAllocator()) {
		SetUDFExecutorFactory(*db.instance, RefuseTasks);
		auto payload = Value::STRUCT({{"execution_backend", Value("subprocess_task")},
		                              {"call_mode", Value("map")},
		                              {"batch_size", Value::BIGINT(batch_rows)},
		                              {"min_task_batch_size", Value::BIGINT(preserve_work_units ? batch_rows : 0)},
		                              {"udf_task_input_max_bytes", Value::BIGINT(input_bytes)},
		                              {"udf_output_target_max_bytes", Value::BIGINT(4096)}});
		sink = make_uniq<PhysicalStreamingUDF>(plan, vector<LogicalType> {LogicalType::BIGINT},
		                                       MakeUDFTableFunction(payload, {LogicalType::BIGINT}, {"result"}),
		                                       make_uniq<UDFFunctionData>(payload, LogicalType::BIGINT, submits),
		                                       vector<ColumnIndex> {ColumnIndex(0)}, 0, vector<column_t> {});
		global = sink->GetGlobalSinkState(*con.context);
		global->PipelineMaxThreadsResolved(1);
		local = sink->GetLocalSinkState(context);
		other_local = sink->GetLocalSinkState(context);
		signal = make_shared_ptr<InterruptDoneSignalState>();
	}
	SinkResultType Accept(DataChunk &chunk, bool other = false, bool batch = false) {
		InterruptState interrupt {weak_ptr<InterruptDoneSignalState>(signal)};
		OperatorSinkInput input {*global, other ? *other_local : *local, interrupt};
		if (batch) {
			ExecutionBatch envelope;
			envelope.materialized = make_uniq<DataChunk>();
			envelope.materialized->InitializeEmpty(chunk.GetTypes());
			envelope.materialized->Reference(chunk);
			return sink->SinkBatch(context, envelope, input);
		}
		return sink->Sink(context, chunk, input);
	}
	shared_ptr<CopyProbe> probe;
	shared_ptr<SubmitProbe> submits;
	DBConfig config;
	DuckDB db;
	Connection con;
	ThreadContext thread;
	ExecutionContext context;
	PhysicalPlan plan;
	unique_ptr<PhysicalStreamingUDF> sink;
	unique_ptr<GlobalSinkState> global;
	unique_ptr<LocalSinkState> local;
	unique_ptr<LocalSinkState> other_local;
	shared_ptr<InterruptDoneSignalState> signal;
};

struct FlowProbe {
	shared_ptr<MediaFlowWindow> window;
	shared_ptr<MediaFlowTicket> initial_view;
	bool hold_initial_view = false;
	idx_t produced = 0;
};
struct FlowInfo : public TableFunctionInfo {
	explicit FlowInfo(shared_ptr<FlowProbe> probe_p) : probe(std::move(probe_p)) {
	}
	shared_ptr<FlowProbe> probe;
};
struct FlowBind : public TableFunctionData {
	explicit FlowBind(shared_ptr<FlowProbe> probe_p) : probe(std::move(probe_p)) {
	}
	shared_ptr<FlowProbe> probe;
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<FlowBind>(probe);
	}
	bool Equals(const FunctionData &other) const override {
		return probe == other.Cast<FlowBind>().probe;
	}
};
struct FlowGlobal : public GlobalTableFunctionState {
	explicit FlowGlobal(shared_ptr<FlowProbe> probe_p)
	    : probe(std::move(probe_p)), window(make_shared_ptr<MediaFlowWindow>(8)) {
		probe->window = window;
		probe->produced = 0;
		if (probe->hold_initial_view) {
			auto request = window->Request(8);
			probe->initial_view = window->TakeGrant(*request);
		}
	}
	~FlowGlobal() override {
		window->Close();
	}
	shared_ptr<FlowProbe> probe;
	shared_ptr<MediaFlowWindow> window;
	unique_ptr<MediaAdmissionRequest> request;
};
unique_ptr<FunctionData> BindFlow(ClientContext &, TableFunctionBindInput &input, vector<LogicalType> &types,
                                  vector<string> &names) {
	types.push_back(LogicalType::BIGINT);
	names.push_back("i");
	return make_uniq<FlowBind>(input.info->Cast<FlowInfo>().probe);
}
unique_ptr<GlobalTableFunctionState> InitFlow(ClientContext &, TableFunctionInitInput &input) {
	return make_uniq<FlowGlobal>(input.bind_data->Cast<FlowBind>().probe);
}
SourceResultType PollFlow(ClientContext &, TableFunctionInput &input, DataChunk &output,
                          const InterruptState &interrupt) {
	auto &state = input.global_state->Cast<FlowGlobal>();
	if (state.probe->produced == 24) {
		return SourceResultType::FINISHED;
	}
	if (!state.request) {
		state.request = state.window->Request(8);
	}
	if (state.request->PollOrArm(interrupt) == MediaAdmissionStatus::WAITING) {
		return SourceResultType::BLOCKED;
	}
	output.AddMediaFlow(state.window->TakeGrant(*state.request));
	state.request.reset();
	output.SetValue(0, 0, Value::BIGINT(state.probe->produced++));
	output.SetCardinality(1);
	return SourceResultType::HAVE_MORE_OUTPUT;
}
shared_ptr<FlowProbe> RegisterFlow(DuckDB &db) {
	auto probe = make_shared_ptr<FlowProbe>();
	ExtensionLoader loader(*db.instance, "media_flow_test");
	TableFunction source("test_media_flow", {}, nullptr, BindFlow, InitFlow);
	source.poll_function = PollFlow;
	source.media_flow_source = true;
	source.function_info = make_shared_ptr<FlowInfo>(probe);
	loader.RegisterFunction(std::move(source));
	return probe;
}
} // namespace

TEST_CASE("Media flow receipt does not revoke retained physical storage", "[media-resource][media-flow]") {
	MediaFlowWindow window(12);
	MediaResourcePool storage({0, 12});
	auto storage_scope = storage.CreateScope();
	auto bytes = storage_scope->Request({0, 12});
	REQUIRE(bytes->PollOrArm([] {}) == MediaAdmissionStatus::GRANTED);
	auto ticket = TakeFlow(window, 12);
	auto pixels = AllocateMediaVector(LogicalType::UTINYINT, 12, bytes->TakeGrant(),
	                                  Allocator::DefaultAllocatorReference(), ticket);
	FlatVector::GetData<uint8_t>(*pixels)[0] = 42;
	Vector view(*pixels, 0, 1);
	DataChunk source, copy;
	source.InitializeEmpty({LogicalType::UTINYINT});
	source.data[0].Reference(*pixels);
	source.SetCardinality(1);
	source.AddMediaFlow(ticket);
	copy.Initialize(Allocator::DefaultAllocator(), source.GetTypes(), 1);
	source.Copy(copy);
	DuckDB db(nullptr);
	Connection con(db);
	ArrowArray arrow;
	ArrowConverter::ToArrowArray(source, &arrow, con.context->GetClientProperties(), {});
	// Copying is not yet acceptance by a bounded receiver.
	CHECK(window.Snapshot().reserved.bytes == 12);
	auto waiting = window.Request(12);
	idx_t wakes = 0;
	REQUIRE(waiting->PollOrArm([&] {
		CHECK(window.Snapshot().reserved.bytes == 12); // callback can reenter
		wakes++;
	}) == MediaAdmissionStatus::WAITING);
	copy.AcknowledgeMediaFlow();
	copy.AcknowledgeMediaFlow();
	CHECK(wakes == 1);
	CHECK(storage.Snapshot().reserved.bytes == 12);
	CHECK(FlatVector::GetData<uint8_t>(view)[0] == 42);
	CHECK_FALSE(ticket->IsActive());
	waiting.reset();
	source.Destroy();
	pixels.reset();
	CHECK(storage.Snapshot().reserved.bytes == 12); // standalone slice remains
	Vector empty(LogicalType::UTINYINT);
	view.Reference(empty);
	CHECK(storage.Snapshot().reserved.bytes == 0);
	CHECK(window.Snapshot().reserved.bytes == 0);
	CHECK(static_cast<const uint8_t *>(arrow.children[0]->buffers[1])[0] == 42);
	arrow.release(&arrow);
}

TEST_CASE("Media flow last view and query close return production credits", "[media-resource][media-flow]") {
	MediaFlowWindow window(8);
	auto pixels =
	    AllocateMediaVector(LogicalType::UTINYINT, 8, {}, Allocator::DefaultAllocatorReference(), TakeFlow(window, 8));
	Vector view(*pixels, 0, 1);
	pixels.reset();
	CHECK(window.Snapshot().reserved.bytes == 8);
	SECTION("last view releases normally") {
		Vector empty(LogicalType::UTINYINT);
		view.Reference(empty);
		CHECK(window.Snapshot().reserved.bytes == 0);
	}
	SECTION("close cancels waiters without waiting for retained views") {
		auto waiting = window.Request(8);
		idx_t wakes = 0;
		REQUIRE(waiting->PollOrArm([&] { wakes++; }) == MediaAdmissionStatus::WAITING);
		window.Close();
		window.Close();
		CHECK(wakes == 1);
		CHECK(waiting->PollOrArm([] {}) == MediaAdmissionStatus::CANCELLED);
		CHECK(window.Snapshot().reserved.bytes == 0);
		CHECK(window.Request(8)->PollOrArm([] {}) == MediaAdmissionStatus::CANCELLED);
	}
}

TEST_CASE("Chunk ownership transformations retain unreceived flow until the last view",
          "[media-resource][media-flow]") {
	MediaFlowWindow window(8);
	DataChunk source, columns, selected, moved, appended, right;
	const vector<LogicalType> types {LogicalType::BIGINT, LogicalType::BIGINT};
	source.Initialize(Allocator::DefaultAllocator(), types, 1);
	source.SetCardinality(1);
	source.AddMediaFlow(TakeFlow(window, 8));
	columns.InitializeEmpty({LogicalType::BIGINT});
	columns.ReferenceColumns(source, {1});
	selected.InitializeEmpty(types);
	SelectionVector selection(1);
	selection.set_index(0, 0);
	selected.Slice(source, selection, 1);
	moved.Move(selected);
	appended.Initialize(Allocator::DefaultAllocator(), types, 1);
	appended.Append(source);
	appended.Split(right, 1);
	appended.Fuse(right);
	source.Destroy();
	columns.Destroy();
	appended.Destroy();
	CHECK(window.Snapshot().reserved.bytes == 8);
	CHECK(moved.HasMediaFlow());
	moved.Destroy();
	CHECK(window.Snapshot().reserved.bytes == 0);
	CHECK_FALSE(selected.HasMediaFlow());
}

TEST_CASE("One-frame flow progresses through projection and cached filters into a blocked UDF",
          "[media-resource][media-flow]") {
	DuckDB db(nullptr);
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET threads=1"));
	auto probe = RegisterFlow(db);
	SetUDFExecutorFactory(*db.instance, RefuseTasks);
	auto pending = con.PendingQuery(R"(
        SELECT udf(i * 2 + 1, {'payload_version': 1, 'udf_name': 'flow-fixture', 'call_mode': 'map',
                              'execution_backend': 'subprocess_task', 'method_return_type': 'BIGINT',
                              'function_pickle': 'fixture'::BLOB, 'function_pickle_size_bytes': 7,
                              'batch_size': 4, 'min_task_batch_size': 4,
                              'udf_task_input_max_bytes': 32, 'udf_output_target_max_bytes': 4096})
        FROM test_media_flow() WHERE i % 2 = 0
    )");
	REQUIRE_FALSE(pending->HasError());
	for (idx_t step = 0; step < 1000 && probe->produced < 3 && !pending->HasError(); step++) {
		pending->ExecuteTask();
	}
	INFO((pending->HasError() ? pending->GetError() : ""));
	REQUIRE_FALSE(pending->HasError());
	// Each accepted scalar-map row contains its argument and passthrough i
	// (16 bytes). 0 and 2 fill the receiver; 1 is filtered without waiting
	// for the caching operator's preferred vector size.
	CHECK(probe->produced == 3);
	con.Interrupt();
	for (idx_t step = 0; step < 1000; step++) {
		if (pending->ExecuteTask() == PendingExecutionResult::EXECUTION_ERROR) {
			break;
		}
	}
	pending.reset();
	CHECK(probe->window->Snapshot().reserved.bytes == 0);
	CHECK(probe->window->Snapshot().waiting_requests == 0);
	REQUIRE_NO_FAIL(con.Query("SELECT 42"));
}

TEST_CASE("Enabled native flow rejects unsupported materialization before producing rows",
          "[media-resource][media-flow]") {
	DuckDB db(nullptr);
	if (!db.instance->GetMediaFlowCapacity()) {
		return;
	}
	Connection con(db);
	auto probe = RegisterFlow(db);
	for (auto query : {"SELECT i FROM test_media_flow()", "SELECT i FROM test_media_flow() ORDER BY i",
	                   "SELECT sum(i) FROM test_media_flow()"}) {
		auto result = con.Query(query);
		REQUIRE(result->HasError());
		CHECK_THAT(result->GetError(), Catch::Matchers::Contains("materialization budget handoff"));
		CHECK(probe->produced == 0);
		CHECK(probe->window->Snapshot().reserved.bytes == 0);
	}
	REQUIRE_NO_FAIL(con.Query("SELECT 42"));
}

TEST_CASE("Blocked flow source wakes on a receipt and cancels without releasing external views",
          "[media-resource][media-flow]") {
	const auto cancel_while_waiting = GENERATE(false, true);
	DuckDB db(nullptr);
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET threads=1"));
	auto probe = RegisterFlow(db);
	probe->hold_initial_view = true;
	SetUDFExecutorFactory(*db.instance, RefuseTasks);
	auto pending = con.PendingQuery(R"(
        SELECT udf(i, {'payload_version': 1, 'udf_name': 'flow-wakeup', 'call_mode': 'map',
                       'execution_backend': 'subprocess_task', 'method_return_type': 'BIGINT',
                       'function_pickle': 'fixture'::BLOB, 'function_pickle_size_bytes': 7,
                       'batch_size': 4, 'min_task_batch_size': 4,
                       'udf_task_input_max_bytes': 32, 'udf_output_target_max_bytes': 4096})
        FROM test_media_flow()
    )");
	REQUIRE_FALSE(pending->HasError());
	for (idx_t step = 0; step < 1000; step++) {
		pending->ExecuteTask();
		if (probe->window && probe->window->Snapshot().waiting_requests) {
			break;
		}
	}
	REQUIRE(probe->window);
	REQUIRE(probe->window->Snapshot().waiting_requests == 1);
	CHECK(probe->produced == 0);
	if (!cancel_while_waiting) {
		probe->initial_view->Acknowledge();
		for (idx_t step = 0; step < 1000 && probe->produced < 2; step++) {
			pending->ExecuteTask();
		}
		CHECK(probe->produced == 2);
	}
	con.Interrupt();
	bool cancelled = false;
	for (idx_t step = 0; step < 1000; step++) {
		if (pending->ExecuteTask() == PendingExecutionResult::EXECUTION_ERROR) {
			cancelled = true;
			break;
		}
	}
	CHECK(cancelled);
	pending.reset();
	CHECK_FALSE(probe->initial_view->IsActive());
	CHECK(probe->window->Snapshot().reserved.bytes == 0);
	CHECK(probe->window->Snapshot().waiting_requests == 0);
	REQUIRE_NO_FAIL(con.Query("SELECT 42"));
}

TEST_CASE("Concurrent media flow close and receipts return each credit once", "[media-resource][media-flow]") {
	for (idx_t iteration = 0; iteration < 100; iteration++) {
		MediaFlowWindow window(16);
		auto first = TakeFlow(window, 8);
		auto second_request = window.Request(8);
		REQUIRE(second_request->PollOrArm([] {}) == MediaAdmissionStatus::GRANTED);
		shared_ptr<MediaFlowTicket> second;
		std::thread close([&] { window.Close(); });
		std::thread receipt([&] { first->Acknowledge(); });
		second = window.TakeGrant(*second_request);
		close.join();
		receipt.join();
		if (second) {
			CHECK_FALSE(second->IsActive());
		}
		CHECK(window.Snapshot().reserved.bytes == 0);
		CHECK(window.Snapshot().active_leases == 0);
	}
}

TEST_CASE("UDF ingress receipts follow reservation and commit for both sink paths", "[media-resource][media-flow]") {
	auto batch = GENERATE(false, true);
	SinkFixture receiver(16);
	MediaFlowWindow window(8);
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::BIGINT}, 1);
	chunk.SetValue(0, 0, Value::BIGINT(7));
	chunk.SetCardinality(1);
	chunk.AddMediaFlow(TakeFlow(window, 8));
	auto waiting = window.Request(8);
	idx_t receipts = 0;
	REQUIRE(waiting->PollOrArm([&] {
		// Reading sink statistics here takes the receiver lock. This detects
		// receipts invoked while holding that lock and receipts before commit.
		CHECK(receiver.sink->ParamsToString()["udf_accepted_input_rows"] == "1");
		receipts++;
	}) == MediaAdmissionStatus::WAITING);
	CHECK(receiver.Accept(chunk, false, batch) == SinkResultType::NEED_MORE_INPUT);
	CHECK(receipts == 1);
	chunk.Reset();
	chunk.SetValue(0, 0, Value::BIGINT(8));
	chunk.SetCardinality(1);
	chunk.AddMediaFlow(window.TakeGrant(*waiting));
	// A short preferred batch must be submitted under input-byte pressure.
	// The fake task domain refuses it, so it remains in the bounded ingress.
	CHECK(receiver.Accept(chunk, false, batch) == SinkResultType::BLOCKED);
	CHECK(window.Snapshot().reserved.bytes == 0); // accepted BLOCKED has a receipt
	CHECK(receiver.Accept(chunk, false, batch) == SinkResultType::BLOCKED);
	CHECK(receiver.sink->ParamsToString()["udf_accepted_input_rows"] == "2");
	CHECK(receiver.sink->ParamsToString()["udf_pending_input_bytes"] == "16");
	chunk.Reset();
	chunk.SetValue(0, 0, Value::BIGINT(9));
	chunk.SetCardinality(1);
	auto rejected = TakeFlow(window, 8);
	chunk.AddMediaFlow(rejected);
	CHECK(receiver.Accept(chunk, true, batch) == SinkResultType::BLOCKED);
	CHECK(rejected->IsActive()); // rejected BLOCKED retains its producer
	CHECK(receiver.sink->ParamsToString()["udf_accepted_input_rows"] == "2");
	receiver.sink->ResetStreamingState(false);
	window.Close();
	CHECK_FALSE(rejected->IsActive());
}

TEST_CASE("Oversized UDF flow input cannot use the legacy one-chunk escape", "[media-resource][media-flow]") {
	SinkFixture receiver(7);
	MediaFlowWindow window(8);
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::BIGINT}, 1);
	chunk.SetCardinality(1);
	auto ticket = TakeFlow(window, 8);
	chunk.AddMediaFlow(ticket);
	REQUIRE_THROWS_AS(receiver.Accept(chunk), OutOfMemoryException);
	CHECK(ticket->IsActive());
	CHECK(receiver.sink->ParamsToString()["udf_accepted_input_rows"] == "0");
	CHECK(receiver.sink->ParamsToString()["udf_pending_input_bytes"] == "0");
}

TEST_CASE("Splitting skewed media input preserves pending bytes before another receipt",
          "[media-resource][media-flow][media-flow-split]") {
	const auto batch = GENERATE(false, true);
	const auto accept_prefix = GENERATE(false, true);
	const auto nested = GENERATE(false, true);
	const auto large_first = GENERATE(false, true);
	SinkFixture receiver(1000, 2, false);
	receiver.submits->accepts_remaining = accept_prefix ? 1 : 0;
	MediaFlowWindow window(1000);
	auto type = nested ? LogicalType::STRUCT({{"payload", LogicalType::VARCHAR}}) : LogicalType::VARCHAR;
	auto value = [&](idx_t bytes) {
		auto text = Value(string(bytes, 'x'));
		return nested ? Value::STRUCT({{"payload", text}}) : text;
	};
	DataChunk first, next;
	first.Initialize(Allocator::DefaultAllocator(), {type}, 3);
	for (idx_t row = 0; row < 3; row++) {
		first.SetValue(0, row, value(row == (large_first ? 0 : 2) ? 900 : 0));
	}
	first.SetCardinality(3);
	REQUIRE(EstimateMediaFlowInputBytes(first) == 948);
	first.AddMediaFlow(TakeFlow(window, 948));
	CHECK(receiver.Accept(first, false, batch) == SinkResultType::NEED_MORE_INPUT);
	const idx_t prefix_bytes = large_first ? 932 : 32;
	const idx_t remaining_bytes = 948 - prefix_bytes;
	const idx_t pending_bytes = accept_prefix ? remaining_bytes : 948;
	REQUIRE_FALSE(receiver.submits->proposed_bytes.empty());
	CHECK(receiver.submits->proposed_bytes.front() == prefix_bytes);
	CHECK(receiver.submits->proposed_bytes == receiver.submits->actual_bytes);
	CHECK(receiver.sink->ParamsToString()["udf_pending_input_bytes"] == std::to_string(pending_bytes));
	CHECK(receiver.sink->ParamsToString()["udf_accepted_input_rows"] == "3");
	CHECK(window.Snapshot().reserved.bytes == 0);

	next.Initialize(Allocator::DefaultAllocator(), {type}, 1);
	next.SetValue(0, 0, value(584));
	next.SetCardinality(1);
	REQUIRE(EstimateMediaFlowInputBytes(next) == 600);
	auto ticket = TakeFlow(window, 600);
	next.AddMediaFlow(ticket);
	const auto would_fit = pending_bytes + 600 <= 1000;
	// With the long row last and the task domain refusing a 32-byte prefix,
	// all 948 bytes still belong to ingress. The old proportional deduction
	// reported 348 and incorrectly acknowledged these additional 600 bytes.
	for (idx_t retry = 0; retry < (would_fit ? 1 : 3); retry++) {
		CHECK(receiver.Accept(next, true, batch) ==
		      (would_fit ? SinkResultType::NEED_MORE_INPUT : SinkResultType::BLOCKED));
		CHECK(ticket->IsActive() == !would_fit);
		CHECK(receiver.sink->ParamsToString()["udf_pending_input_bytes"] ==
		      std::to_string(pending_bytes + (would_fit ? 600 : 0)));
		CHECK(receiver.sink->ParamsToString()["udf_accepted_input_rows"] == (would_fit ? "4" : "3"));
		CHECK(receiver.sink->ParamsToString()["udf_accepted_input_bytes"] == (would_fit ? "1548" : "948"));
		CHECK(receiver.submits->proposed_bytes == receiver.submits->actual_bytes);
	}
	window.Close();
	CHECK_FALSE(ticket->IsActive());
	CHECK(window.Snapshot().reserved.bytes == 0);
}

TEST_CASE("Repeated media splits measure payloads at the remaining row offset",
          "[media-resource][media-flow][media-flow-split]") {
	const auto batch = GENERATE(false, true);
	const idx_t rows = 3 * STANDARD_VECTOR_SIZE + 1;
	const idx_t total_bytes = rows * sizeof(string_t) + 900;
	SinkFixture receiver(200000, 2, false);
	MediaFlowWindow window(200000);
	DataChunk first, next;
	first.Initialize(Allocator::DefaultAllocator(), {LogicalType::VARCHAR}, rows);
	for (idx_t row = 0; row < rows; row++) {
		first.SetValue(0, row, Value(string(row == STANDARD_VECTOR_SIZE + 1 ? 900 : 0, 'x')));
	}
	first.SetCardinality(rows);
	REQUIRE(EstimateMediaFlowInputBytes(first) == total_bytes);
	first.AddMediaFlow(TakeFlow(window, total_bytes));
	CHECK(receiver.Accept(first, false, batch) == SinkResultType::NEED_MORE_INPUT);
	REQUIRE(receiver.submits->proposed_bytes.size() == 1);
	CHECK(receiver.submits->proposed_bytes[0] == total_bytes - sizeof(string_t));
	CHECK(receiver.submits->proposed_bytes == receiver.submits->actual_bytes);
	CHECK(receiver.submits->chunk_bytes[0] == vector<idx_t> {STANDARD_VECTOR_SIZE * sizeof(string_t),
	                                                         STANDARD_VECTOR_SIZE * sizeof(string_t) + 900,
	                                                         STANDARD_VECTOR_SIZE * sizeof(string_t)});
	CHECK(receiver.sink->ParamsToString()["udf_pending_input_bytes"] == std::to_string(total_bytes));
	next.Initialize(Allocator::DefaultAllocator(), {LogicalType::VARCHAR}, 1);
	next.SetValue(0, 0, Value(string(120000 - sizeof(string_t), 'x')));
	next.SetCardinality(1);
	auto ticket = TakeFlow(window, 120000);
	next.AddMediaFlow(ticket);
	for (idx_t retry = 0; retry < 3; retry++) {
		CHECK(receiver.Accept(next, true, batch) == SinkResultType::BLOCKED);
		CHECK(ticket->IsActive());
		CHECK(receiver.sink->ParamsToString()["udf_pending_input_bytes"] == std::to_string(total_bytes));
		CHECK(receiver.sink->ParamsToString()["udf_accepted_input_rows"] == std::to_string(rows));
	}
	window.Close();
}

TEST_CASE("Failed UDF ingress copy retains the producer and rolls back its reservation",
          "[media-resource][media-flow]") {
	SinkFixture receiver(16);
	MediaFlowWindow window(8);
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), {LogicalType::BIGINT}, 1);
	chunk.SetValue(0, 0, Value::BIGINT(42));
	chunk.SetCardinality(1);
	auto ticket = TakeFlow(window, 8);
	chunk.AddMediaFlow(ticket);
	receiver.probe->fail = true;
	CHECK_THROWS_AS(receiver.Accept(chunk), OutOfMemoryException);
	receiver.probe->fail = false;
	CHECK(ticket->IsActive());
	CHECK(window.Snapshot().reserved.bytes == 8);
	CHECK(receiver.sink->ParamsToString()["udf_accepted_input_rows"] == "0");
	CHECK(receiver.sink->ParamsToString()["udf_pending_input_bytes"] == "0");
	CHECK(receiver.Accept(chunk) == SinkResultType::NEED_MORE_INPUT);
	CHECK_FALSE(ticket->IsActive());
	CHECK(receiver.sink->ParamsToString()["udf_accepted_input_rows"] == "1");
}

TEST_CASE("Flow ingress copy cost follows selected values and fixed image payloads", "[media-resource][media-flow]") {
	const auto nested = GENERATE(false, true);
	auto image = ImageLogicalType::Create("RGB", 2, 2);
	auto path_type = LogicalType::STRUCT({{"path", LogicalType::VARCHAR}});
	auto metadata = nested ? LogicalType::STRUCT({{"details", path_type}}) : path_type;
	DataChunk source, selected, copied;
	source.Initialize(Allocator::DefaultAllocator(), {image, metadata}, 2);
	for (idx_t row = 0; row < 2; row++) {
		source.SetValue(0, row, ImageVector::FromPixels(vector<Value>(12, Value::UTINYINT(42)), 2, 2, "RGB", image));
		auto path = Value::STRUCT({{"path", Value(string(30 + row, 'x'))}});
		source.SetValue(1, row, nested ? Value::STRUCT({{"details", path}}) : path);
	}
	source.SetCardinality(2);
	SelectionVector sel(1);
	sel.set_index(0, 1);
	selected.InitializeEmpty(source.GetTypes());
	selected.Slice(source, sel, 1);
	CHECK(EstimateMediaFlowInputBytes(selected) == 12 + sizeof(string_t) + 31);
	copied.Initialize(Allocator::DefaultAllocator(), source.GetTypes(), 1);
	selected.Copy(copied);
	CHECK(EstimateMediaFlowInputBytes(copied) == EstimateMediaFlowInputBytes(selected));
	CHECK(copied.GetValue(0, 0) == selected.GetValue(0, 0));
	DataChunk repeated, repeated_copy;
	SelectionVector repeat(2);
	repeat.set_index(0, 1);
	repeat.set_index(1, 1);
	repeated.InitializeEmpty(source.GetTypes());
	repeated.Slice(source, repeat, 2);
	repeated_copy.Initialize(Allocator::DefaultAllocator(), source.GetTypes(), 2);
	repeated.Copy(repeated_copy);
	CHECK(EstimateMediaFlowInputBytes(repeated) == 2 * (12 + sizeof(string_t) + 31));
	CHECK(EstimateMediaFlowInputBytes(repeated_copy) == EstimateMediaFlowInputBytes(repeated));
	DataChunk dynamic;
	dynamic.Initialize(Allocator::DefaultAllocator(), {LogicalType::LIST(LogicalType::UTINYINT)}, 1);
	dynamic.SetCardinality(1);
	REQUIRE_THROWS_AS(EstimateMediaFlowInputBytes(dynamic), NotImplementedException);
}
