// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"
#include "test_helpers.hpp"
#include "duckdb/common/types/image.hpp"
#include "duckdb/common/types/column/column_data_allocator.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/physical_arrow_collector.hpp"
#include "duckdb/common/arrow/result_arrow_wrapper.hpp"
#include "duckdb/execution/media_buffer.hpp"
#include "duckdb/execution/operator/projection/physical_udf_inout.hpp"
#include "duckdb/execution/operator/helper/physical_buffered_collector.hpp"
#include "duckdb/execution/operator/helper/physical_media_materialized_collector.hpp"
#include "duckdb/execution/operator/order/physical_order.hpp"
#include "duckdb/execution/operator/scan/physical_dummy_scan.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/execution/udf_executor.hpp"
#include "duckdb/function/scalar/udf_functions.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/pending_query_result.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/main/prepared_statement.hpp"
#include "duckdb/main/stream_query_result.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/parallel/event.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/storage/buffer_manager.hpp"

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
	bool allocator_destroyed = false;
	std::function<void()> before_allocate;
};
struct CopyAllocatorData : public PrivateAllocatorData {
	explicit CopyAllocatorData(shared_ptr<CopyProbe> probe_p) : probe(std::move(probe_p)) {
	}
	~CopyAllocatorData() override {
		probe->allocator_destroyed = true;
	}
	shared_ptr<CopyProbe> probe;
};
DBConfig *ConfigureCopyAllocator(DBConfig &config, shared_ptr<CopyProbe> probe) {
	config.allocator = make_uniq<Allocator>(
	    [](PrivateAllocatorData *data, idx_t bytes) -> data_ptr_t {
		    auto probe = data->Cast<CopyAllocatorData>().probe;
		    auto before = std::move(probe->before_allocate);
		    if (before) {
			    before();
		    }
		    if (probe->fail) {
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
	bool images = false;
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
	if (input.info->Cast<FlowInfo>().probe->images) {
		types.push_back(ImageLogicalType::Create("RGB", 128, 128));
		names.push_back("data");
	}
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
	const auto row = state.probe->produced++;
	output.SetValue(0, 0, Value::BIGINT(row));
	if (state.probe->images) {
		vector<uint8_t> pixels(128 * 128 * 3, NumericCast<uint8_t>(row));
		ImageVector::WritePixels(output.data[1], 0, 128, 128, "RGB", pixels.data());
	}
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

struct ResultQueueFixture {
	explicit ResultQueueFixture(idx_t bytes)
	    : copy(make_shared_ptr<CopyProbe>()), db(nullptr, ConfigureCopyAllocator(config, copy)), con(db) {
		ClientConfig::GetConfig(*con.context).streaming_buffer_size = bytes;
		queue = make_shared_ptr<SimpleBufferedData>(*con.context, true, Allocator::Get(*con.context));
	}
	shared_ptr<CopyProbe> copy;
	DBConfig config;
	DuckDB db;
	Connection con;
	shared_ptr<SimpleBufferedData> queue;
};

class ResultWakeTask : public Task {
public:
	TaskExecutionResult Execute(TaskExecutionMode) override {
		return TaskExecutionResult::TASK_FINISHED;
	}
	uint64_t CurrentInterruptEpoch() const override {
		return 1;
	}
	void Reschedule(uint64_t epoch) override {
		CHECK(epoch == 1);
		wakes++;
		if (on_wake) {
			on_wake();
		}
	}
	idx_t wakes = 0;
	std::function<void()> on_wake;
};

class BatchMediaResultCollector : public PhysicalBufferedCollector {
public:
	BatchMediaResultCollector(PhysicalPlan &plan, PreparedStatementData &data)
	    : PhysicalBufferedCollector(plan, data, false, true) {
	}
	ExecutionBatchRequirement GetExecutionBatchRequirement(PipelineOperatorRole role) const override {
		return role == PipelineOperatorRole::SINK ? ExecutionBatchRequirement::BATCH_REQUIRED
		                                          : ExecutionBatchRequirement::BATCH_OPTIONAL;
	}
};

unique_ptr<PhysicalOperator> CollectFlowBatches(ClientContext &context, PreparedStatementData &data) {
	if (data.output_type == QueryResultOutputType::ALLOW_STREAMING &&
	    PhysicalPlanGenerator::HasNativeMediaFlow(context, data.physical_plan->Root())) {
		return make_uniq<BatchMediaResultCollector>(*data.physical_plan, data);
	}
	return PhysicalResultCollector::GetResultCollector(context, data);
}

struct MaterializedMediaFixture {
	MaterializedMediaFixture(const vector<LogicalType> &types, idx_t copy_limit)
	    : copy(make_shared_ptr<CopyProbe>()), db(nullptr, ConfigureCopyAllocator(config, copy)), con(db),
	      thread(*con.context), context(*con.context, thread, nullptr) {
		ClientConfig::GetConfig(*con.context).streaming_buffer_size = copy_limit;
		prepared = con.Prepare("SELECT 1");
		REQUIRE_FALSE(prepared->HasError());
		auto &data = *prepared->data;
		data.types = types;
		data.names = vector<string>(types.size(), "value");
		sink = make_uniq<PhysicalMediaMaterializedCollector>(*data.physical_plan, data);
		global = sink->GetGlobalSinkState(*con.context);
		local = sink->GetLocalSinkState(context);
	}
	SinkResultType Accept(DataChunk &chunk, bool batch = false) {
		InterruptState interrupt;
		OperatorSinkInput input {*global, *local, interrupt};
		if (batch) {
			ExecutionBatch envelope;
			envelope.materialized = make_uniq<DataChunk>();
			envelope.materialized->InitializeEmpty(chunk.GetTypes());
			envelope.materialized->Reference(chunk);
			return sink->SinkBatch(context, envelope, input);
		}
		return sink->Sink(context, chunk, input);
	}
	unique_ptr<QueryResult> Finish() {
		return sink->GetResult(*global);
	}
	shared_ptr<CopyProbe> copy;
	DBConfig config;
	DuckDB db;
	Connection con;
	ThreadContext thread;
	ExecutionContext context;
	unique_ptr<PreparedStatement> prepared;
	unique_ptr<PhysicalMediaMaterializedCollector> sink;
	unique_ptr<GlobalSinkState> global;
	unique_ptr<LocalSinkState> local;
};

class BatchMediaMaterializedCollector : public PhysicalMediaMaterializedCollector {
public:
	BatchMediaMaterializedCollector(PhysicalPlan &plan, PreparedStatementData &data)
	    : PhysicalMediaMaterializedCollector(plan, data) {
	}
	ExecutionBatchRequirement GetExecutionBatchRequirement(PipelineOperatorRole role) const override {
		return role == PipelineOperatorRole::SINK ? ExecutionBatchRequirement::BATCH_REQUIRED
		                                          : ExecutionBatchRequirement::BATCH_OPTIONAL;
	}
};

unique_ptr<PhysicalOperator> MaterializeFlowBatches(ClientContext &context, PreparedStatementData &data) {
	if (PhysicalPlanGenerator::HasNativeMediaFlow(context, data.physical_plan->Root())) {
		return make_uniq<BatchMediaMaterializedCollector>(*data.physical_plan, data);
	}
	return PhysicalResultCollector::GetResultCollector(context, data);
}

class MediaSortTestEvent : public Event {
public:
	explicit MediaSortTestEvent(Executor &executor) : Event(executor) {
	}
	void Schedule() override {
	}
};

struct MediaSortFixture {
	MediaSortFixture(vector<LogicalType> types, idx_t copy_limit)
	    : db(nullptr), con(db), thread(*con.context), context(*con.context, thread, nullptr),
	      plan(Allocator::DefaultAllocator()), executor(*con.context), pipeline(executor), event(executor) {
		REQUIRE_NO_FAIL(con.Query("BEGIN TRANSACTION"));
		ClientConfig::GetConfig(*con.context).streaming_buffer_size = copy_limit;
		vector<BoundOrderByNode> orders;
		orders.emplace_back(OrderType::ASCENDING, OrderByNullType::NULLS_LAST,
		                    make_uniq<BoundReferenceExpression>(types[0], 0));
		vector<idx_t> projections;
		for (idx_t i = 0; i < types.size(); i++) {
			projections.push_back(i);
		}
		sink = make_uniq<PhysicalOrder>(plan, types, std::move(orders), std::move(projections), 0, false, true);
		sink->children.push_back(plan.Make<PhysicalDummyScan>(types, 0));
		sink->sink_state = sink->GetGlobalSinkState(*con.context);
		local = sink->GetLocalSinkState(context);
	}
	SinkResultType Accept(DataChunk &chunk, bool batch = false) {
		InterruptState interrupt;
		OperatorSinkInput input {*sink->sink_state, *local, interrupt};
		if (batch) {
			ExecutionBatch envelope;
			envelope.materialized = make_uniq<DataChunk>();
			envelope.materialized->InitializeEmpty(chunk.GetTypes());
			envelope.materialized->Reference(chunk);
			return sink->SinkBatch(context, envelope, input);
		}
		return sink->Sink(context, chunk, input);
	}
	SinkFinalizeType Finish() {
		InterruptState interrupt;
		OperatorSinkFinalizeInput input {*sink->sink_state, interrupt};
		auto result = sink->Finalize(pipeline, event, *con.context, input);
		source = sink->GetGlobalSourceState(*con.context);
		source_local = sink->GetLocalSourceState(context, *source);
		return result;
	}
	unique_ptr<DataChunk> Fetch() {
		auto chunk = make_uniq<DataChunk>();
		chunk->Initialize(Allocator::DefaultAllocator(), sink->types);
		InterruptState interrupt;
		OperatorSourceInput input {*source, *source_local, interrupt};
		sink->GetDataInternal(context, *chunk, input);
		return chunk->size() ? std::move(chunk) : nullptr;
	}
	DuckDB db;
	Connection con;
	ThreadContext thread;
	ExecutionContext context;
	PhysicalPlan plan;
	Executor executor;
	Pipeline pipeline;
	MediaSortTestEvent event;
	unique_ptr<PhysicalOrder> sink;
	unique_ptr<LocalSinkState> local;
	unique_ptr<GlobalSourceState> source;
	unique_ptr<LocalSourceState> source_local;
};
} // namespace

TEST_CASE("Media sort commits selected skewed payload and key storage before a receipt",
          "[media-resource][media-sort]") {
	const auto batch = GENERATE(false, true);
	const auto nested = GENERATE(false, true);
	const auto payload_type = nested ? LogicalType::STRUCT({{"s", LogicalType::VARCHAR}}) : LogicalType::VARCHAR;
	MediaSortFixture receiver({LogicalType::BIGINT, payload_type}, 1000);
	DataChunk original, input;
	original.Initialize(Allocator::DefaultAllocator(), {LogicalType::BIGINT, payload_type}, 4);
	for (idx_t i = 0; i < 4; i++) {
		original.SetValue(0, i, Value::BIGINT(4 - i));
		auto text = Value(string(i == 0 ? 900 : (i == 3 ? 2000 : 0), 'a' + i));
		original.SetValue(1, i, nested ? Value::STRUCT({{"s", text}}) : text);
	}
	original.SetCardinality(4);
	SelectionVector selected(3);
	selected.set_index(0, 2);
	selected.set_index(1, 0);
	selected.set_index(2, 1);
	input.InitializeEmpty(original.GetTypes());
	input.Slice(original, selected, 3);
	MediaFlowWindow window(1000);
	auto ticket = TakeFlow(window, 1000);
	input.AddMediaFlow(ticket);
	auto waiting = window.Request(1000);
	idx_t receipts = 0;
	REQUIRE(waiting->PollOrArm([&] {
		// The callback can finalize and scan both domains immediately.
		CHECK(receiver.Finish() == SinkFinalizeType::READY);
		receipts++;
	}) == MediaAdmissionStatus::WAITING);
	REQUIRE(receiver.Accept(input, batch) == SinkResultType::NEED_MORE_INPUT);
	CHECK(receipts == 1);
	CHECK_FALSE(ticket->IsActive());
	idx_t row = 0;
	while (auto output = receiver.Fetch()) {
		REQUIRE(output->size() == 1);
		const auto original_row = 2 - row;
		CHECK(output->GetValue(0, 0) == original.GetValue(0, original_row));
		CHECK(output->GetValue(1, 0) == original.GetValue(1, original_row));
		row++;
	}
	CHECK(row == 3);
	CHECK(input.data[1].GetVectorType() == VectorType::DICTIONARY_VECTOR);
}

TEST_CASE("Native media sort spills images and reaches EOF with one frame of production credit",
          "[media-resource][media-sort]") {
	const auto batch = GENERATE(false, true);
	const auto filtered = GENERATE(false, true);
	DuckDB db(nullptr);
	if (!db.instance->GetMediaFlowCapacity()) {
		return;
	}
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET threads=4"));
	REQUIRE_NO_FAIL(con.Query("SET streaming_buffer_size='50KiB'"));
	REQUIRE_NO_FAIL(con.Query("SET memory_limit='2MiB'"));
	if (batch) {
		ClientConfig::GetConfig(*con.context).get_result_collector = MaterializeFlowBatches;
	}
	auto probe = RegisterFlow(db);
	probe->images = true;
	auto result = con.Query(filtered ? "SELECT i, data FROM test_media_flow() WHERE i % 2 = 0 ORDER BY i DESC"
	                                 : "SELECT i, data FROM test_media_flow() ORDER BY i DESC");
	INFO((result->HasError() ? result->GetError() : ""));
	REQUIRE_FALSE(result->HasError());
	CHECK(probe->produced == 24);
	CHECK(probe->window->Snapshot().reserved.bytes == 0);
	auto &manager = BufferManager::GetBufferManager(*con.context);
	CHECK(manager.GetUsedSwap() > 0);
	CHECK(manager.GetUsedMemory() <= manager.GetMaxMemory());
	idx_t count = 0;
	while (auto chunk = result->Fetch()) {
		for (idx_t row = 0; row < chunk->size(); row++, count++) {
			const auto expected = filtered ? 22 - 2 * count : 23 - count;
			CHECK(chunk->GetValue(0, row) == Value::BIGINT(expected));
			auto pixels = ImageVector::Pixels(chunk->data[1], row);
			for (idx_t pixel = 0; pixel < 128 * 128 * 3; pixel++) {
				if (pixels[pixel] != expected) {
					FAIL("Sorting changed the correspondence between keys and image pixels");
				}
			}
		}
	}
	CHECK(count == (filtered ? 12 : 24));
	result.reset();
	CHECK(manager.GetUsedSwap() == 0);
	REQUIRE_NO_FAIL(con.Query("SELECT 42"));
}

TEST_CASE("Media sort merges across native key chunks without retaining pixels in the merge",
          "[media-resource][media-sort]") {
	const auto batch = GENERATE(false, true);
	MediaSortFixture receiver({LogicalType::BIGINT, LogicalType::BIGINT}, 1024 * 1024);
	REQUIRE_NO_FAIL(receiver.con.Query("SET memory_limit='2MiB'"));
	const idx_t total = 2 * STANDARD_VECTOR_SIZE + 3;
	for (idx_t offset = 0; offset < total;) {
		const auto rows = MinValue<idx_t>(STANDARD_VECTOR_SIZE, total - offset);
		DataChunk input;
		input.Initialize(Allocator::DefaultAllocator(), {LogicalType::BIGINT, LogicalType::BIGINT}, rows);
		for (idx_t row = 0; row < rows; row++) {
			input.SetValue(0, row, Value::BIGINT((total - offset - row) * 17));
			input.SetValue(1, row, Value::BIGINT(offset + row));
		}
		input.SetCardinality(rows);
		MediaFlowWindow window(rows * 16);
		auto ticket = TakeFlow(window, rows * 16);
		input.AddMediaFlow(ticket);
		REQUIRE(receiver.Accept(input, batch) == SinkResultType::NEED_MORE_INPUT);
		CHECK_FALSE(ticket->IsActive());
		offset += rows;
	}
	REQUIRE(receiver.Finish() == SinkFinalizeType::READY);
	idx_t row = 0;
	while (auto output = receiver.Fetch()) {
		CHECK(output->GetValue(0, 0) == Value::BIGINT((row + 1) * 17));
		CHECK(output->GetValue(1, 0) == Value::BIGINT(total - row - 1));
		row++;
	}
	CHECK(row == total);
}

TEST_CASE("Media sort failed inputs retain receipts and release every uncommitted storage domain",
          "[media-resource][media-sort]") {
	const auto batch = GENERATE(false, true);
	const auto failure = GENERATE(0, 1, 2);
	MediaSortFixture receiver({LogicalType::BIGINT, LogicalType::VARCHAR}, 648);
	REQUIRE_NO_FAIL(receiver.con.Query("SET memory_limit='2MiB'"));
	if (failure == 1) {
		REQUIRE_NO_FAIL(receiver.con.Query("SET memory_limit='512KiB'"));
	}
	if (failure == 2) {
		REQUIRE_NO_FAIL(receiver.con.Query("SET max_temp_directory_size='0B'"));
	}
	DataChunk input;
	input.Initialize(Allocator::DefaultAllocator(), {LogicalType::BIGINT, LogicalType::VARCHAR}, 8);
	for (idx_t row = 0; row < 8; row++) {
		input.SetValue(0, row, Value::BIGINT(row));
		input.SetValue(1, row, Value(string(failure == 0 && row == 1 ? 700 : 600, 'x')));
	}
	input.SetCardinality(8);
	MediaFlowWindow window(6000);
	auto ticket = TakeFlow(window, 6000);
	input.AddMediaFlow(ticket);
	auto &manager = BufferManager::GetBufferManager(*receiver.con.context);
	const auto baseline = manager.GetUsedMemory();
	REQUIRE_THROWS_AS(receiver.Accept(input, batch), OutOfMemoryException);
	CHECK(ticket->IsActive());
	CHECK(manager.GetUsedMemory() == baseline);
	CHECK(manager.GetUsedSwap() == 0);
	CHECK(receiver.Finish() == SinkFinalizeType::NO_OUTPUT_POSSIBLE);
	CHECK_FALSE(receiver.Fetch());
}

TEST_CASE("Media sort shrinks pixel prefixes and retires key pins before reading large stored images",
          "[media-resource][media-sort]") {
	const auto batch = GENERATE(false, true);
	auto image_type = ImageLogicalType::Create("RGB", 128, 128);
	MediaSortFixture receiver({LogicalType::BIGINT, image_type}, 3 * 1024 * 1024);
	REQUIRE_NO_FAIL(receiver.con.Query("SET memory_limit='2MiB'"));
	const idx_t rows = 60;
	DataChunk input;
	input.Initialize(Allocator::DefaultAllocator(), {LogicalType::BIGINT, image_type}, rows);
	for (idx_t row = 0; row < rows; row++) {
		input.SetValue(0, row, Value::BIGINT(rows - row - 1));
		vector<uint8_t> pixels(128 * 128 * 3, NumericCast<uint8_t>(rows - row - 1));
		ImageVector::WritePixels(input.data[1], row, 128, 128, "RGB", pixels.data());
	}
	input.SetCardinality(rows);
	const auto bytes = EstimateMediaFlowInputBytes(input);
	REQUIRE(bytes > 2 * 1024 * 1024);
	MediaFlowWindow window(bytes);
	auto ticket = TakeFlow(window, bytes);
	input.AddMediaFlow(ticket);
	REQUIRE(receiver.Accept(input, batch) == SinkResultType::NEED_MORE_INPUT);
	CHECK_FALSE(ticket->IsActive());
	REQUIRE(receiver.Finish() == SinkFinalizeType::READY);
	auto &manager = BufferManager::GetBufferManager(*receiver.con.context);
	CHECK(manager.GetUsedSwap() > 0);
	CHECK(manager.GetUsedMemory() <= manager.GetMaxMemory());
	idx_t row = 0;
	while (auto output = receiver.Fetch()) {
		CHECK(output->GetValue(0, 0) == Value::BIGINT(row));
		// Physical sources may return dictionary vectors. ImageVector pixel
		// access requires flat input, as QueryResult::Fetch supplies publicly.
		output->Flatten();
		CHECK(ImageVector::Pixels(output->data[1], 0)[0] == row);
		row++;
	}
	CHECK(row == rows);
}

TEST_CASE("Native media sorting rejects unsupported key types before producing any rows",
          "[media-resource][media-sort]") {
	DuckDB db(nullptr);
	if (!db.instance->GetMediaFlowCapacity()) {
		return;
	}
	Connection con(db);
	auto probe = RegisterFlow(db);
	for (auto query : {"SELECT i::VARCHAR AS k FROM test_media_flow() ORDER BY k",
	                   "SELECT [i] AS k FROM test_media_flow() ORDER BY k"}) {
		auto result = con.Query(query);
		REQUIRE(result->HasError());
		CHECK_THAT(result->GetError(), Catch::Matchers::Contains("fixed-width primitive sort keys"));
		CHECK(probe->produced == 0);
	}
	REQUIRE_NO_FAIL(con.Query("SELECT 42"));
}

TEST_CASE("Native media sort preserves multiple keys NULL ordering and projected images",
          "[media-resource][media-sort]") {
	const auto nulls_first = GENERATE(false, true);
	DuckDB db(nullptr);
	if (!db.instance->GetMediaFlowCapacity()) {
		return;
	}
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET threads=4"));
	REQUIRE_NO_FAIL(con.Query("SET memory_limit='2MiB'"));
	REQUIRE_NO_FAIL(con.Query("SET streaming_buffer_size='50KiB'"));
	auto probe = RegisterFlow(db);
	probe->images = true;
	auto result = con.Query("SELECT i, data, CASE WHEN i % 5 = 0 THEN NULL ELSE i % 3 END AS k "
	                        "FROM test_media_flow() ORDER BY k DESC NULLS " +
	                        string(nulls_first ? "FIRST" : "LAST") + ", i ASC");
	INFO((result->HasError() ? result->GetError() : ""));
	REQUIRE_FALSE(result->HasError());
	vector<idx_t> expected;
	for (idx_t i = 0; i < 24; i++) {
		expected.push_back(i);
	}
	std::sort(expected.begin(), expected.end(), [&](idx_t a, idx_t b) {
		const bool a_null = a % 5 == 0;
		const bool b_null = b % 5 == 0;
		if (a_null != b_null) {
			return nulls_first ? a_null : !a_null;
		}
		if (!a_null && a % 3 != b % 3) {
			return a % 3 > b % 3;
		}
		return a < b;
	});
	idx_t row = 0;
	while (auto chunk = result->Fetch()) {
		for (idx_t i = 0; i < chunk->size(); i++, row++) {
			CHECK(chunk->GetValue(0, i) == Value::BIGINT(expected[row]));
			CHECK(ImageVector::Pixels(chunk->data[1], i)[0] == expected[row]);
			CHECK(chunk->GetValue(2, i).IsNull() == (expected[row] % 5 == 0));
		}
	}
	CHECK(row == 24);
	result.reset();
	auto images_only = con.Query("SELECT data FROM test_media_flow() ORDER BY i DESC");
	REQUIRE_FALSE(images_only->HasError());
	row = 0;
	while (auto chunk = images_only->Fetch()) {
		for (idx_t i = 0; i < chunk->size(); i++, row++) {
			CHECK(ImageVector::Pixels(chunk->data[0], i)[0] == 23 - row);
		}
	}
	CHECK(row == 24);
}

TEST_CASE("Media sort receipts preserve independently retained physical pixel leases", "[media-resource][media-sort]") {
	MediaSortFixture receiver({LogicalType::UTINYINT}, 64);
	MediaFlowWindow window(12);
	MediaResourcePool pool({0, 12});
	auto scope = pool.CreateScope();
	auto request = scope->Request({0, 12});
	REQUIRE(request->PollOrArm([] {}) == MediaAdmissionStatus::GRANTED);
	auto ticket = TakeFlow(window, 12);
	auto pixels = AllocateMediaVector(LogicalType::UTINYINT, 12, request->TakeGrant(),
	                                  Allocator::DefaultAllocatorReference(), ticket);
	for (idx_t row = 0; row < 12; row++) {
		FlatVector::GetData<uint8_t>(*pixels)[row] = NumericCast<uint8_t>(11 - row);
	}
	Vector retained(*pixels, 0, 1);
	DataChunk input;
	input.InitializeEmpty({LogicalType::UTINYINT});
	input.data[0].Reference(*pixels);
	input.SetCardinality(12);
	input.AddMediaFlow(ticket);
	REQUIRE(receiver.Accept(input) == SinkResultType::NEED_MORE_INPUT);
	CHECK(window.Snapshot().reserved.bytes == 0);
	CHECK(pool.Snapshot().reserved.bytes == 12);
	input.Destroy();
	pixels.reset();
	CHECK(pool.Snapshot().reserved.bytes == 12);
	retained.Reference(Value::UTINYINT(0));
	CHECK(pool.Snapshot().reserved.bytes == 0);
	REQUIRE(receiver.Finish() == SinkFinalizeType::READY);
	idx_t row = 0;
	while (auto chunk = receiver.Fetch()) {
		CHECK(chunk->GetValue(0, 0) == Value::UTINYINT(row++));
	}
	CHECK(row == 12);
}

TEST_CASE("Native media sorting cleans storage and credits on spill failure expression failure and cancellation",
          "[media-resource][media-sort]") {
	const auto failure = GENERATE(0, 1, 2);
	DuckDB db(nullptr);
	if (!db.instance->GetMediaFlowCapacity()) {
		return;
	}
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET memory_limit='2MiB'"));
	REQUIRE_NO_FAIL(con.Query("SET streaming_buffer_size='50KiB'"));
	if (failure == 0) {
		REQUIRE_NO_FAIL(con.Query("SET max_temp_directory_size='0B'"));
	}
	auto probe = RegisterFlow(db);
	probe->images = true;
	ExtensionLoader loader(*db.instance, "media_sort_cancel_test");
	loader.RegisterFunction(ScalarFunction("interrupt_media_sort", {LogicalType::BIGINT}, LogicalType::BIGINT,
	                                       [&](DataChunk &input, ExpressionState &, Vector &output) {
		                                       output.Reference(input.data[0]);
		                                       if (input.GetValue(0, 0) == Value::BIGINT(3)) {
			                                       con.Interrupt();
		                                       }
	                                       }));
	auto &manager = BufferManager::GetBufferManager(*con.context);
	const auto baseline = manager.GetUsedMemory();
	auto query = failure == 1 ? "SELECT CASE WHEN i=3 THEN CAST('bad' || i::VARCHAR AS BIGINT) ELSE i END AS k, data "
	                            "FROM test_media_flow() ORDER BY k"
	             : failure == 2 ? "SELECT interrupt_media_sort(i) AS k, data FROM test_media_flow() ORDER BY k"
	                            : "SELECT i, data FROM test_media_flow() ORDER BY i";
	auto result = con.Query(query);
	REQUIRE(result->HasError());
	CHECK(probe->produced > 0);
	CHECK(probe->window->Snapshot().reserved.bytes == 0);
	CHECK(probe->window->Snapshot().waiting_requests == 0);
	CHECK(manager.GetUsedMemory() == baseline);
	CHECK(manager.GetUsedSwap() == 0);
	REQUIRE_NO_FAIL(con.Query("SET max_temp_directory_size='10MiB'"));
	REQUIRE_NO_FAIL(con.Query("SELECT i FROM test_media_flow() ORDER BY i DESC"));
	REQUIRE_NO_FAIL(con.Query("SELECT 42"));
}

TEST_CASE("Sorted media result copies remain valid after stream close and database destruction",
          "[media-resource][media-sort]") {
	unique_ptr<DataChunk> retained;
	ArrowArray arrow;
	{
		DuckDB db(nullptr);
		if (!db.instance->GetMediaFlowCapacity()) {
			return;
		}
		Connection con(db);
		REQUIRE_NO_FAIL(con.Query("SET memory_limit='2MiB'"));
		REQUIRE_NO_FAIL(con.Query("SET streaming_buffer_size='50KiB'"));
		auto probe = RegisterFlow(db);
		probe->images = true;
		auto result = con.SendQuery("SELECT i, data FROM test_media_flow() ORDER BY i DESC");
		REQUIRE_FALSE(result->HasError());
		REQUIRE(result->type == QueryResultType::STREAM_RESULT);
		retained = result->Fetch();
		REQUIRE(retained);
		CHECK(retained->size() == 1);
		CHECK(retained->GetValue(0, 0) == Value::BIGINT(23));
		ArrowConverter::ToArrowArray(*retained, &arrow, con.context->GetClientProperties(), {});
		result->Cast<StreamQueryResult>().Close();
		CHECK(probe->produced == 24);
		CHECK(probe->window->Snapshot().reserved.bytes == 0);
		REQUIRE_NO_FAIL(con.Query("SELECT 42"));
	}
	CHECK(retained->GetValue(0, 0) == Value::BIGINT(23));
	CHECK(ImageVector::Pixels(retained->data[1], 0)[0] == 23);
	CHECK(static_cast<const int64_t *>(arrow.children[0]->buffers[1])[0] == 23);
	retained.reset();
	CHECK(static_cast<const int64_t *>(arrow.children[0]->buffers[1])[0] == 23);
	arrow.release(&arrow);
}

TEST_CASE("Arrow collector selection preserves native media receiving and exported lifetime",
          "[media-resource][media-sort][media-arrow]") {
	const auto sorted = GENERATE(false, true);
	const auto streaming = GENERATE(false, true);
	CAPTURE(sorted, streaming);
	unique_ptr<DataChunk> retained;
	ArrowArray arrow {};
	const auto first = sorted ? 23 : 0;
	{
		DuckDB db(nullptr);
		if (!db.instance->GetMediaFlowCapacity()) {
			return;
		}
		Connection con(db);
		REQUIRE_NO_FAIL(con.Query("SET threads=4"));
		REQUIRE_NO_FAIL(con.Query("SET memory_limit='2MiB'"));
		REQUIRE_NO_FAIL(con.Query("SET streaming_buffer_size='50KiB'"));
		auto probe = RegisterFlow(db);
		probe->images = true;
		auto &config = ClientConfig::GetConfig(*con.context);
		// Match relation Arrow table/capsule export, which installs its own factory.
		config.get_result_collector = [](ClientContext &context, PreparedStatementData &data) {
			return PhysicalArrowCollector::Create(context, data, 1000000);
		};
		{
			auto ordinary = con.context->Query("SELECT 42", false);
			REQUIRE_FALSE(ordinary->HasError());
			CHECK(ordinary->type == QueryResultType::ARROW_RESULT);
		}
		auto result = con.context->Query(sorted ? "SELECT i, data FROM test_media_flow() ORDER BY i DESC"
		                                        : "SELECT i, data FROM test_media_flow()",
		                                 streaming);
		INFO((result->HasError() ? result->GetError() : ""));
		REQUIRE_FALSE(result->HasError());
		CHECK(result->type == (streaming ? QueryResultType::STREAM_RESULT : QueryResultType::MATERIALIZED_RESULT));
		if (sorted || !streaming) {
			CHECK(BufferManager::GetBufferManager(*con.context).GetUsedSwap() > 0);
		}
		idx_t count = 0;
		while (auto chunk = result->Fetch()) {
			for (idx_t row = 0; row < chunk->size(); row++, count++) {
				const auto expected = sorted ? 23 - count : count;
				CHECK(chunk->GetValue(0, row) == Value::BIGINT(expected));
				auto pixels = ImageVector::Pixels(chunk->data[1], row);
				for (idx_t pixel = 0; pixel < 128 * 128 * 3; pixel++) {
					if (pixels[pixel] != expected) {
						FAIL("Arrow receiving changed image pixels or sort association");
					}
				}
			}
			if (!retained) {
				retained = std::move(chunk);
				ArrowConverter::ToArrowArray(*retained, &arrow, con.context->GetClientProperties(), {});
			}
		}
		CHECK(count == 24);
		CHECK(probe->produced == 24);
		CHECK(probe->window->Snapshot().reserved.bytes == 0);
		result.reset();
		config.get_result_collector = nullptr;
		REQUIRE_NO_FAIL(con.Query("SELECT 42"));
	}
	CHECK(retained->GetValue(0, 0) == Value::BIGINT(first));
	CHECK(ImageVector::Pixels(retained->data[1], 0)[0] == first);
	retained.reset();
	CHECK(static_cast<const int64_t *>(arrow.children[0]->buffers[1])[0] == first);
	arrow.release(&arrow);
}

TEST_CASE("Materialized media commits every selected prefix before acknowledging the input",
          "[media-resource][media-materialization]") {
	const auto nested = GENERATE(false, true);
	const auto batch = GENERATE(false, true);
	const auto type = nested ? LogicalType::STRUCT({{"s", LogicalType::VARCHAR}}) : LogicalType::VARCHAR;
	MaterializedMediaFixture receiver({type}, 916);
	DataChunk source, input;
	source.Initialize(Allocator::DefaultAllocator(), {type}, 4);
	for (idx_t row = 0; row < 4; row++) {
		auto value = Value(string(row == 0 ? 900 : (row == 3 ? 2000 : 0), 'x'));
		source.SetValue(0, row, nested ? Value::STRUCT({{"s", value}}) : value);
	}
	source.SetCardinality(4);
	SelectionVector selected(3);
	selected.set_index(0, 2);
	selected.set_index(1, 0);
	selected.set_index(2, 1);
	input.InitializeEmpty({type});
	input.Slice(source, selected, 3);
	CHECK(EstimateMediaFlowInputBytes(input) == 948);
	MediaFlowWindow window(948);
	auto ticket = TakeFlow(window, 948);
	input.AddMediaFlow(ticket);
	auto waiting = window.Request(948);
	unique_ptr<QueryResult> result;
	REQUIRE(waiting->PollOrArm([&] {
		// A receipt can re-enter the collector. Its entire input must already
		// be visible, with every prefix unpinned and owned by BufferManager.
		result = receiver.Finish();
		CHECK(result->Cast<MaterializedQueryResult>().RowCount() == 3);
		CHECK(BufferManager::GetBufferManager(*receiver.con.context).GetUsedMemory() > 0);
	}) == MediaAdmissionStatus::WAITING);
	REQUIRE(receiver.Accept(input, batch) == SinkResultType::NEED_MORE_INPUT);
	REQUIRE(result);
	CHECK_FALSE(ticket->IsActive());
	CHECK(input.data[0].GetVectorType() == VectorType::DICTIONARY_VECTOR);
	idx_t row = 0;
	while (auto chunk = result->Fetch()) {
		for (idx_t i = 0; i < chunk->size(); i++, row++) {
			CHECK(chunk->GetValue(0, i) == source.GetValue(0, selected.get_index(row)));
		}
	}
	CHECK(row == 3);
}

TEST_CASE("Materialized media failures keep the receipt and roll back uncommitted prefixes",
          "[media-resource][media-materialization]") {
	const auto batch = GENERATE(false, true);
	const auto failure = GENERATE(0, 1, 2);
	MaterializedMediaFixture receiver({LogicalType::VARCHAR}, 616);
	REQUIRE_NO_FAIL(receiver.con.Query("SET memory_limit='768KiB'"));
	REQUIRE_NO_FAIL(receiver.con.Query("SET max_temp_directory_size='0B'"));
	if (failure == 1) {
		REQUIRE_NO_FAIL(receiver.con.Query("SET memory_limit='1B'"));
	}
	DataChunk input;
	input.Initialize(Allocator::DefaultAllocator(), {LogicalType::VARCHAR}, 8);
	for (idx_t row = 0; row < 8; row++) {
		// Oversized row follows a legal prefix. Neither may be published.
		input.SetValue(0, row, Value(string(failure == 0 && row == 1 ? 700 : 600, 'x')));
	}
	input.SetCardinality(8);
	MediaFlowWindow window(6000);
	auto ticket = TakeFlow(window, 6000);
	input.AddMediaFlow(ticket);
	auto &manager = BufferManager::GetBufferManager(*receiver.con.context);
	const auto baseline = manager.GetUsedMemory();
	REQUIRE_THROWS_AS(receiver.Accept(input, batch), OutOfMemoryException);
	CHECK(ticket->IsActive());
	CHECK(manager.GetUsedMemory() == baseline);
	CHECK(receiver.Finish()->Cast<MaterializedQueryResult>().RowCount() == 0);
	REQUIRE_NO_FAIL(receiver.con.Query("SET memory_limit='10MiB'"));
	REQUIRE_NO_FAIL(receiver.con.Query("SELECT 42"));
}

TEST_CASE("Materialized media shrinks a legal payload prefix to fit its pinned storage working set",
          "[media-resource][media-materialization]") {
	const auto batch = GENERATE(false, true);
	auto image_type = ImageLogicalType::Create("RGB", 128, 128);
	MaterializedMediaFixture receiver({LogicalType::BIGINT, image_type}, 1024 * 1024);
	REQUIRE_NO_FAIL(receiver.con.Query("SET memory_limit='512KiB'"));
	DataChunk input;
	input.Initialize(Allocator::DefaultAllocator(), {LogicalType::BIGINT, image_type}, 12);
	for (idx_t row = 0; row < 12; row++) {
		input.SetValue(0, row, Value::BIGINT(row));
		vector<uint8_t> pixels(128 * 128 * 3, NumericCast<uint8_t>(row));
		ImageVector::WritePixels(input.data[1], row, 128, 128, "RGB", pixels.data());
	}
	input.SetCardinality(12);
	const auto bytes = EstimateMediaFlowInputBytes(input);
	REQUIRE(bytes < 1024 * 1024);
	REQUIRE(bytes > 512 * 1024);
	MediaFlowWindow window(bytes);
	auto ticket = TakeFlow(window, bytes);
	input.AddMediaFlow(ticket);
	REQUIRE(receiver.Accept(input, batch) == SinkResultType::NEED_MORE_INPUT);
	CHECK_FALSE(ticket->IsActive());
	auto &manager = BufferManager::GetBufferManager(*receiver.con.context);
	CHECK(manager.GetUsedSwap() > 0);
	CHECK(manager.GetUsedMemory() <= manager.GetMaxMemory());
	auto result = receiver.Finish();
	idx_t row = 0;
	while (auto chunk = result->Fetch()) {
		for (idx_t i = 0; i < chunk->size(); i++, row++) {
			CHECK(chunk->GetValue(0, i) == Value::BIGINT(row));
			auto pixels = ImageVector::Pixels(chunk->data[1], i);
			for (idx_t pixel = 0; pixel < 128 * 128 * 3; pixel++) {
				if (pixels[pixel] != row) {
					FAIL("Shrinking the receiving prefix changed the image pixels");
				}
			}
		}
	}
	CHECK(row == 12);
}

TEST_CASE("Materialized media releases production credit independently of retained pixels",
          "[media-resource][media-materialization]") {
	MaterializedMediaFixture receiver({LogicalType::UTINYINT}, 8);
	MediaFlowWindow window(12);
	MediaResourcePool pool({0, 12});
	auto scope = pool.CreateScope();
	auto request = scope->Request({0, 12});
	REQUIRE(request->PollOrArm([] {}) == MediaAdmissionStatus::GRANTED);
	auto ticket = TakeFlow(window, 12);
	auto pixels = AllocateMediaVector(LogicalType::UTINYINT, 12, request->TakeGrant(),
	                                  Allocator::DefaultAllocatorReference(), ticket);
	for (idx_t row = 0; row < 12; row++) {
		FlatVector::GetData<uint8_t>(*pixels)[row] = NumericCast<uint8_t>(row);
	}
	Vector retained(*pixels, 0, 1);
	DataChunk input;
	input.InitializeEmpty({LogicalType::UTINYINT});
	input.data[0].Reference(*pixels);
	input.SetCardinality(12);
	input.AddMediaFlow(ticket);
	REQUIRE(receiver.Accept(input) == SinkResultType::NEED_MORE_INPUT);
	CHECK(window.Snapshot().reserved.bytes == 0);
	CHECK(pool.Snapshot().reserved.bytes == 12);
	input.Destroy();
	pixels.reset();
	CHECK(pool.Snapshot().reserved.bytes == 12);
	retained.Reference(Value::UTINYINT(0));
	CHECK(pool.Snapshot().reserved.bytes == 0);
	auto result = receiver.Finish();
	CHECK(result->Cast<MaterializedQueryResult>().GetValue(0, 11) == Value::UTINYINT(11));
}

TEST_CASE("Native materialization reaches EOF and spills sealed images within a small memory limit",
          "[media-resource][media-materialization]") {
	const auto batch = GENERATE(false, true);
	const auto filtered = GENERATE(false, true);
	DuckDB db(nullptr);
	if (!db.instance->GetMediaFlowCapacity()) {
		return;
	}
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET threads=4"));
	REQUIRE_NO_FAIL(con.Query("SET streaming_buffer_size='50KiB'"));
	REQUIRE_NO_FAIL(con.Query("SET memory_limit='768KiB'"));
	if (batch) {
		ClientConfig::GetConfig(*con.context).get_result_collector = MaterializeFlowBatches;
	}
	auto probe = RegisterFlow(db);
	probe->images = true;
	auto result = con.Query(filtered ? "SELECT i * 2 AS n, data FROM test_media_flow() WHERE i % 2 = 0"
	                                 : "SELECT i AS n, data FROM test_media_flow()");
	INFO((result->HasError() ? result->GetError() : ""));
	REQUIRE_FALSE(result->HasError());
	CHECK(probe->produced == 24);
	CHECK(probe->window->Snapshot().reserved.bytes == 0);
	auto &manager = BufferManager::GetBufferManager(*con.context);
	CHECK(manager.GetUsedSwap() > 0);
	CHECK(manager.GetUsedMemory() <= manager.GetMaxMemory());
	CHECK(result->Collection().GetAllocatorType() == ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR);
	idx_t row = 0;
	while (auto chunk = result->Fetch()) {
		for (idx_t i = 0; i < chunk->size(); i++, row++) {
			const auto original = row * (filtered ? 2 : 1);
			CHECK(chunk->GetValue(0, i) == Value::BIGINT(original * (filtered ? 2 : 1)));
			auto pixels = ImageVector::Pixels(chunk->data[1], i);
			for (idx_t pixel = 0; pixel < 128 * 128 * 3; pixel++) {
				if (pixels[pixel] != original) {
					FAIL("Spilled image pixels were not preserved");
				}
			}
		}
	}
	CHECK(row == (filtered ? 12 : 24));
	result.reset();
	CHECK(manager.GetUsedSwap() == 0);
	auto limited = con.Query("SELECT i, data FROM test_media_flow() WHERE i % 2 = 0 LIMIT 2");
	REQUIRE_FALSE(limited->HasError());
	CHECK(limited->RowCount() == 2);
	CHECK(probe->produced < 24);
	CHECK(probe->window->Snapshot().reserved.bytes == 0);
	REQUIRE_NO_FAIL(con.Query("SELECT 42"));
}

TEST_CASE("Unread native materialization and fetched copies survive connection and database destruction",
          "[media-resource][media-materialization]") {
	unique_ptr<QueryResult> result;
	shared_ptr<CopyProbe> allocator_probe;
	{
		MaterializedMediaFixture receiver({LogicalType::VARCHAR}, 1200);
		allocator_probe = receiver.copy;
		DataChunk input;
		input.Initialize(Allocator::DefaultAllocator(), {LogicalType::VARCHAR}, 4);
		for (idx_t row = 0; row < 4; row++) {
			input.SetValue(0, row, Value(string(1000, 'a' + row)));
		}
		input.SetCardinality(4);
		REQUIRE(receiver.Accept(input) == SinkResultType::NEED_MORE_INPUT);
		result = receiver.Finish();
	}
	CHECK_FALSE(allocator_probe->allocator_destroyed);
	// GetValue caches an additional scan state in the materialized base class.
	// Its pins must also be released before the retained database is destroyed.
	CHECK(result->Cast<MaterializedQueryResult>().GetValue(0, 3) == Value(string(1000, 'd')));
	auto first = result->Fetch();
	REQUIRE(first);
	ArrowArray arrow;
	ArrowConverter::ToArrowArray(*first, &arrow, ClientProperties(), {});
	idx_t rows = first->size();
	while (auto chunk = result->Fetch()) {
		rows += chunk->size();
	}
	CHECK(rows == 4);
	result.reset();
	CHECK(allocator_probe->allocator_destroyed);
	CHECK(first->GetValue(0, 0) == Value(string(1000, 'a')));
	CHECK(static_cast<const char *>(arrow.children[0]->buffers[2])[0] == 'a');
	first.reset();
	arrow.release(&arrow);
}

TEST_CASE("Native materialization cleans received storage and flow after failure or cancellation",
          "[media-resource][media-materialization]") {
	const auto failure = GENERATE(0, 1, 2, 3);
	DuckDB db(nullptr);
	if (!db.instance->GetMediaFlowCapacity()) {
		return;
	}
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET memory_limit='768KiB'"));
	REQUIRE_NO_FAIL(con.Query("SET streaming_buffer_size='50KiB'"));
	if (failure == 0) {
		REQUIRE_NO_FAIL(con.Query("SET max_temp_directory_size='0B'"));
	}
	if (failure == 3) {
		REQUIRE_NO_FAIL(con.Query("SET streaming_buffer_size='49159B'"));
	}
	auto probe = RegisterFlow(db);
	probe->images = true;
	ExtensionLoader loader(*db.instance, "media_materialization_cancel_test");
	loader.RegisterFunction(ScalarFunction("interrupt_media_materialization", {LogicalType::BIGINT},
	                                       LogicalType::BIGINT,
	                                       [&](DataChunk &input, ExpressionState &, Vector &output) {
		                                       output.Reference(input.data[0]);
		                                       if (input.GetValue(0, 0) == Value::BIGINT(3)) {
			                                       con.Interrupt();
		                                       }
	                                       }));
	auto &manager = BufferManager::GetBufferManager(*con.context);
	const auto baseline = manager.GetUsedMemory();
	auto query = failure == 1 ? "SELECT CASE WHEN i=3 THEN CAST('bad' || i::VARCHAR AS BIGINT) ELSE i END, data "
	                            "FROM test_media_flow()"
	             : failure == 2 ? "SELECT interrupt_media_materialization(i), data FROM test_media_flow()"
	                            : "SELECT i, data FROM test_media_flow()";
	auto result = con.Query(query);
	REQUIRE(result->HasError());
	CHECK(probe->produced > 0);
	CHECK(probe->produced < 24);
	CHECK(probe->window->Snapshot().reserved.bytes == 0);
	CHECK(probe->window->Snapshot().waiting_requests == 0);
	CHECK(manager.GetUsedMemory() == baseline);
	CHECK(manager.GetUsedSwap() == 0);
	REQUIRE_NO_FAIL(con.Query("SET max_temp_directory_size='10MiB'"));
	REQUIRE_NO_FAIL(con.Query("SET streaming_buffer_size='50KiB'"));
	REQUIRE_NO_FAIL(con.Query("SELECT i, data FROM test_media_flow() LIMIT 1"));
	REQUIRE_NO_FAIL(con.Query("SELECT 42"));
}

TEST_CASE("A reset materialization allocator preserves its managed database lifetime",
          "[media-resource][media-materialization]") {
	unique_ptr<ColumnDataAllocator> replacement;
	{
		DuckDB db(nullptr);
		ColumnDataAllocator original(BufferManager::GetBufferManager(*db.instance),
		                             ColumnDataCollectionLifetime::THROW_ERROR_AFTER_DATABASE_CLOSES);
		// Combining/resetting sealed collections creates a fresh allocator. It
		// must inherit the lifetime guard and unregister itself on destruction.
		replacement = make_uniq<ColumnDataAllocator>(original);
		CHECK(replacement->GetDatabase() == db.instance);
	}
	CHECK_THROWS_AS(replacement->GetDatabase(), ConnectionException);
	replacement.reset();
}

TEST_CASE("Streaming media results split skewed payloads without acknowledging a partial receipt",
          "[media-resource][media-result-flow]") {
	const auto nested = GENERATE(false, true);
	const auto large_first = GENERATE(false, true);
	ResultQueueFixture receiver(1000);
	const auto type = nested ? LogicalType::STRUCT({{"payload", LogicalType::VARCHAR}}) : LogicalType::VARCHAR;
	auto value = [&](idx_t size) {
		auto text = Value(string(size, 'x'));
		return nested ? Value::STRUCT({{"payload", text}}) : text;
	};
	MediaFlowWindow initial_window(600), window(948);
	DataChunk initial, input;
	initial.Initialize(Allocator::DefaultAllocator(), {type}, 1);
	initial.SetValue(0, 0, value(584));
	initial.SetCardinality(1);
	initial.AddMediaFlow(TakeFlow(initial_window, 600));
	idx_t initial_offset = 0;
	REQUIRE(receiver.queue->AppendMediaFlow(initial, initial_offset, InterruptState()) ==
	        SinkResultType::NEED_MORE_INPUT);
	input.Initialize(Allocator::DefaultAllocator(), {type}, 3);
	for (idx_t row = 0; row < 3; row++) {
		input.SetValue(0, row, value(row == (large_first ? 0 : 2) ? 900 : 0));
	}
	input.SetCardinality(3);
	auto ticket = TakeFlow(window, 948);
	input.AddMediaFlow(ticket);
	idx_t offset = 0;
	auto task = make_shared_ptr<ResultWakeTask>();
	InterruptState interrupt {weak_ptr<Task>(task)};
	for (idx_t retry = 0; retry < 3; retry++) {
		REQUIRE(receiver.queue->AppendMediaFlow(input, offset, interrupt) == SinkResultType::BLOCKED);
		CHECK(offset == (large_first ? 0 : 2));
		CHECK(receiver.queue->SnapshotMediaFlow().queued_bytes == (large_first ? 600 : 632));
		CHECK(receiver.queue->SnapshotMediaFlow().reserved_bytes == 0);
		CHECK(receiver.queue->SnapshotMediaFlow().waiting_sinks == 1);
		CHECK(ticket->IsActive());
	}
	task->on_wake = [&] {
		CHECK(receiver.queue->SnapshotMediaFlow().queued_bytes == (large_first ? 0 : 32));
	};
	auto retained = receiver.queue->Scan();
	REQUIRE(retained);
	CHECK(retained->GetValue(0, 0) == value(584));
	CHECK(task->wakes == 1);
	REQUIRE(receiver.queue->AppendMediaFlow(input, offset, interrupt) == SinkResultType::NEED_MORE_INPUT);
	CHECK(offset == 0);
	CHECK_FALSE(ticket->IsActive());
	CHECK(receiver.queue->SnapshotMediaFlow().queued_bytes == 948);
	CHECK(receiver.queue->SnapshotMediaFlow().peak_bytes <= 1000);
	idx_t row = 0;
	while (auto chunk = receiver.queue->Scan()) {
		for (idx_t i = 0; i < chunk->size(); i++) {
			CHECK(chunk->GetValue(0, i) == input.GetValue(0, row++));
		}
	}
	CHECK(row == 3);
	receiver.queue->Close();
	CHECK(retained->GetValue(0, 0) == value(584));
}

TEST_CASE("Streaming result copy failure and close roll back reserved capacity without a receipt",
          "[media-resource][media-result-flow]") {
	const auto close_during_copy = GENERATE(false, true);
	ResultQueueFixture receiver(8);
	MediaFlowWindow window(8);
	DataChunk input;
	input.Initialize(Allocator::DefaultAllocator(), {LogicalType::BIGINT}, 1);
	input.SetValue(0, 0, Value::BIGINT(42));
	input.SetCardinality(1);
	auto ticket = TakeFlow(window, 8);
	input.AddMediaFlow(ticket);
	idx_t offset = 0;
	receiver.copy->before_allocate = [&] {
		auto snapshot = receiver.queue->SnapshotMediaFlow();
		CHECK(snapshot.reserved_bytes == 8);
		CHECK(snapshot.queued_bytes == 0);
		if (close_during_copy) {
			receiver.queue->Close();
			CHECK(receiver.queue->SnapshotMediaFlow().reserved_bytes == 8);
		}
	};
	receiver.copy->fail = !close_during_copy;
	CHECK_THROWS(receiver.queue->AppendMediaFlow(input, offset, InterruptState()));
	CHECK(offset == 0);
	CHECK(ticket->IsActive());
	CHECK(receiver.queue->SnapshotMediaFlow().queued_bytes == 0);
	CHECK(receiver.queue->SnapshotMediaFlow().reserved_bytes == 0);
	if (!close_during_copy) {
		receiver.copy->fail = false;
		REQUIRE(receiver.queue->AppendMediaFlow(input, offset, InterruptState()) == SinkResultType::NEED_MORE_INPUT);
		CHECK_FALSE(ticket->IsActive());
		receiver.queue->Close();
	}
	CHECK_THROWS(receiver.queue->AppendMediaFlow(input, offset, InterruptState()));
	window.Close();
	CHECK(window.Snapshot().reserved.bytes == 0);
}

TEST_CASE("Overlapping result copies cannot spend another producer's reservation",
          "[media-resource][media-result-flow]") {
	ResultQueueFixture receiver(16);
	MediaFlowWindow first_window(8), second_window(16);
	DataChunk first, second;
	first.Initialize(Allocator::DefaultAllocator(), {LogicalType::BIGINT}, 1);
	first.SetValue(0, 0, Value::BIGINT(10));
	first.SetCardinality(1);
	first.AddMediaFlow(TakeFlow(first_window, 8));
	second.Initialize(Allocator::DefaultAllocator(), {LogicalType::BIGINT}, 2);
	second.SetValue(0, 0, Value::BIGINT(20));
	second.SetValue(0, 1, Value::BIGINT(30));
	second.SetCardinality(2);
	auto ticket = TakeFlow(second_window, 16);
	second.AddMediaFlow(ticket);
	idx_t first_offset = 0, second_offset = 0;
	auto task = make_shared_ptr<ResultWakeTask>();
	InterruptState interrupt {weak_ptr<Task>(task)};
	receiver.copy->before_allocate = [&] {
		CHECK(receiver.queue->SnapshotMediaFlow().reserved_bytes == 8);
		REQUIRE(receiver.queue->AppendMediaFlow(second, second_offset, interrupt) == SinkResultType::BLOCKED);
		CHECK(second_offset == 1);
		auto snapshot = receiver.queue->SnapshotMediaFlow();
		CHECK(snapshot.queued_bytes == 8);
		CHECK(snapshot.reserved_bytes == 8);
		CHECK(snapshot.peak_bytes == 16);
		CHECK(ticket->IsActive());
	};
	REQUIRE(receiver.queue->AppendMediaFlow(first, first_offset, interrupt) == SinkResultType::NEED_MORE_INPUT);
	CHECK(receiver.queue->SnapshotMediaFlow().queued_bytes == 16);
	task->on_wake = [&] {
		CHECK(receiver.queue->SnapshotMediaFlow().queued_bytes == 8);
	};
	auto next = receiver.queue->Scan();
	REQUIRE(next);
	CHECK(next->GetValue(0, 0) == Value::BIGINT(20));
	CHECK(task->wakes == 1);
	REQUIRE(receiver.queue->AppendMediaFlow(second, second_offset, interrupt) == SinkResultType::NEED_MORE_INPUT);
	CHECK_FALSE(ticket->IsActive());
	for (auto value : {10, 30}) {
		next = receiver.queue->Scan();
		REQUIRE(next);
		CHECK(next->GetValue(0, 0) == Value::BIGINT(value));
	}
	CHECK(receiver.queue->SnapshotMediaFlow().queued_bytes == 0);
	CHECK(receiver.queue->SnapshotMediaFlow().peak_bytes == 16);
}

TEST_CASE("A streaming result row larger than its receiving capacity fails before copying",
          "[media-resource][media-result-flow]") {
	ResultQueueFixture receiver(8);
	MediaFlowWindow window(32);
	DataChunk input;
	input.Initialize(Allocator::DefaultAllocator(), {LogicalType::VARCHAR}, 1);
	input.SetValue(0, 0, Value(string(16, 'x')));
	input.SetCardinality(1);
	auto ticket = TakeFlow(window, 32);
	input.AddMediaFlow(ticket);
	idx_t offset = 0;
	CHECK_THROWS_WITH(receiver.queue->AppendMediaFlow(input, offset, InterruptState()),
	                  Catch::Matchers::Contains("streaming_buffer_size receiving capacity"));
	CHECK(ticket->IsActive());
	CHECK(receiver.queue->SnapshotMediaFlow().peak_bytes == 0);
	CHECK(receiver.queue->SnapshotMediaFlow().waiting_sinks == 0);
}

TEST_CASE("A received result copy does not revoke the original pixels or borrow their allocation lease",
          "[media-resource][media-result-flow]") {
	ResultQueueFixture receiver(12);
	MediaFlowWindow window(12);
	MediaResourcePool pixels_pool({0, 12});
	auto scope = pixels_pool.CreateScope();
	auto request = scope->Request({0, 12});
	REQUIRE(request->PollOrArm([] {}) == MediaAdmissionStatus::GRANTED);
	auto ticket = TakeFlow(window, 12);
	auto pixels = AllocateMediaVector(LogicalType::UTINYINT, 12, request->TakeGrant(),
	                                  Allocator::DefaultAllocatorReference(), ticket);
	for (idx_t row = 0; row < 12; row++) {
		FlatVector::GetData<uint8_t>(*pixels)[row] = 42;
	}
	Vector view(*pixels, 0, 1);
	DataChunk input;
	input.InitializeEmpty({LogicalType::UTINYINT});
	input.data[0].Reference(*pixels);
	input.SetCardinality(12);
	input.AddMediaFlow(ticket);
	idx_t offset = 0;
	REQUIRE(receiver.queue->AppendMediaFlow(input, offset, InterruptState()) == SinkResultType::NEED_MORE_INPUT);
	CHECK(window.Snapshot().reserved.bytes == 0);
	CHECK(pixels_pool.Snapshot().reserved.bytes == 12);
	auto copy = receiver.queue->Scan();
	REQUIRE(copy);
	ArrowArray arrow;
	ArrowConverter::ToArrowArray(*copy, &arrow, receiver.con.context->GetClientProperties(), {});
	input.Destroy();
	pixels.reset();
	CHECK(pixels_pool.Snapshot().reserved.bytes == 12);
	Vector empty(LogicalType::UTINYINT);
	view.Reference(empty);
	CHECK(pixels_pool.Snapshot().reserved.bytes == 0);
	receiver.queue->Close();
	copy.reset();
	CHECK(static_cast<const uint8_t *>(arrow.children[0]->buffers[1])[0] == 42);
	arrow.release(&arrow);
}

TEST_CASE("Returned media result copies outlive their database allocator", "[media-resource][media-result-flow]") {
	auto probe = make_shared_ptr<CopyProbe>();
	unique_ptr<DataChunk> retained;
	unique_ptr<Vector> view;
	ArrowArray arrow;
	{
		DBConfig config;
		DuckDB db(nullptr, ConfigureCopyAllocator(config, probe));
		Connection con(db);
		ClientConfig::GetConfig(*con.context).streaming_buffer_size = 256;
		auto queue = make_shared_ptr<SimpleBufferedData>(*con.context, true);
		DataChunk input;
		input.Initialize(
		    Allocator::DefaultAllocator(),
		    {LogicalType::BIGINT, LogicalType::VARCHAR, LogicalType::STRUCT({{"s", LogicalType::VARCHAR}})}, 1);
		input.SetValue(0, 0, Value::BIGINT(42));
		input.SetValue(1, 0, Value(string(32, 'x')));
		input.SetValue(2, 0, Value::STRUCT({{"s", Value(string(32, 'y'))}}));
		input.SetCardinality(1);
		idx_t offset = 0;
		REQUIRE(queue->AppendMediaFlow(input, offset, InterruptState()) == SinkResultType::NEED_MORE_INPUT);
		retained = queue->Scan();
		REQUIRE(retained);
		CHECK(retained->data[0].GetBuffer()->GetAllocator().get() == &Allocator::DefaultAllocator());
		CHECK(retained->data[1].GetBuffer()->GetAllocator().get() == &Allocator::DefaultAllocator());
		view = make_uniq<Vector>(retained->data[0], 0, 1);
		ArrowConverter::ToArrowArray(*retained, &arrow, con.context->GetClientProperties(), {});
		queue->Close();
	}
	CHECK(probe->allocator_destroyed);
	CHECK(retained->GetValue(0, 0) == Value::BIGINT(42));
	CHECK(retained->GetValue(1, 0) == Value(string(32, 'x')));
	CHECK(retained->GetValue(2, 0) == Value::STRUCT({{"s", Value(string(32, 'y'))}}));
	retained.reset();
	CHECK(view->GetValue(0) == Value::BIGINT(42));
	view.reset();
	CHECK(static_cast<const int64_t *>(arrow.children[0]->buffers[1])[0] == 42);
	arrow.release(&arrow);
}

TEST_CASE("A one-frame native scan streams short result chunks through filters and limits",
          "[media-resource][media-result-flow]") {
	const auto batches = GENERATE(false, true);
	const auto threads = GENERATE(1, 4);
	const auto query = GENERATE(string("SELECT i * 2 + 1 FROM test_media_flow()"),
	                            string("SELECT i * 2 + 1 FROM test_media_flow() WHERE i % 2 = 0"),
	                            string("SELECT i * 2 + 1 FROM test_media_flow() WHERE i % 2 = 0 LIMIT 3"));
	DuckDB db(nullptr);
	if (!db.instance->GetMediaFlowCapacity()) {
		return;
	}
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET threads=" + to_string(threads)));
	REQUIRE_NO_FAIL(con.Query("SET streaming_buffer_size='12B'"));
	if (batches) {
		ClientConfig::GetConfig(*con.context).get_result_collector = CollectFlowBatches;
	}
	auto probe = RegisterFlow(db);
	auto result = con.SendQuery(query);
	REQUIRE_FALSE(result->HasError());
	REQUIRE(result->type == QueryResultType::STREAM_RESULT);
	auto &stream = result->Cast<StreamQueryResult>();
	CHECK(stream.IsMediaFlowResult());
	auto retained = result->Fetch();
	REQUIRE(retained);
	CHECK(retained->GetValue(0, 0) == Value::BIGINT(1));
	ArrowArray arrow;
	ArrowConverter::ToArrowArray(*retained, &arrow, con.context->GetClientProperties(), {});
	for (idx_t step = 0; step < 100; step++) {
		if (StreamQueryResult::IsChunkReady(stream.ExecuteTask())) {
			break;
		}
	}
	const auto produced = probe->produced;
	CHECK(produced < 24);
	for (idx_t retry = 0; retry < 3; retry++) {
		stream.ExecuteTask();
		CHECK(probe->produced == produced);
	}
	vector<int64_t> values {1};
	while (auto chunk = result->Fetch()) {
		for (idx_t row = 0; row < chunk->size(); row++) {
			values.push_back(chunk->GetValue(0, row).GetValue<int64_t>());
		}
	}
	REQUIRE_FALSE(result->HasError());
	vector<int64_t> expected;
	const auto filtered = query.find("WHERE") != string::npos;
	const auto limited = query.find("LIMIT") != string::npos;
	for (idx_t row = 0; row < (limited ? 6 : 24); row += filtered ? 2 : 1) {
		expected.push_back(NumericCast<int64_t>(2 * row + 1));
	}
	CHECK(values == expected);
	CHECK(probe->window->Snapshot().reserved.bytes == 0);
	CHECK(probe->window->Snapshot().waiting_requests == 0);
	CHECK(retained->GetValue(0, 0) == Value::BIGINT(1));
	CHECK(static_cast<const int64_t *>(arrow.children[0]->buffers[1])[0] == 1);
	arrow.release(&arrow);
	REQUIRE_NO_FAIL(con.Query("SELECT 42"));
}

TEST_CASE("Closing a buffered media result retires its blocked scan while a returned chunk survives",
          "[media-resource][media-result-flow]") {
	const auto explicit_close = GENERATE(false, true);
	DuckDB db(nullptr);
	if (!db.instance->GetMediaFlowCapacity()) {
		return;
	}
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET streaming_buffer_size='8B'"));
	auto probe = RegisterFlow(db);
	auto result = con.SendQuery("SELECT i FROM test_media_flow()");
	REQUIRE_FALSE(result->HasError());
	REQUIRE(result->type == QueryResultType::STREAM_RESULT);
	auto retained = result->Fetch();
	REQUIRE(retained);
	result->Cast<StreamQueryResult>().ExecuteTask();
	if (explicit_close) {
		result->Cast<StreamQueryResult>().Close();
	}
	result.reset();
	CHECK(probe->window->Snapshot().reserved.bytes == 0);
	CHECK(probe->window->Snapshot().waiting_requests == 0);
	CHECK(retained->GetValue(0, 0) == Value::BIGINT(0));
	REQUIRE_NO_FAIL(con.Query("SELECT 42"));
}

TEST_CASE("Media Arrow export bounds prefetch and does not retire a newer query on release",
          "[media-resource][media-result-flow]") {
	const auto replace_query = GENERATE(false, true);
	DuckDB db(nullptr);
	if (!db.instance->GetMediaFlowCapacity()) {
		return;
	}
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET streaming_buffer_size='12B'"));
	auto probe = RegisterFlow(db);
	auto result = con.SendQuery("SELECT i FROM test_media_flow()");
	REQUIRE_FALSE(result->HasError());
	auto wrapper = new ResultArrowArrayStreamWrapper(std::move(result), 1000000);
	CHECK(wrapper->batch_size == 1);
	auto stream = wrapper->stream;
	ArrowArray retained;
	REQUIRE(stream.get_next(&stream, &retained) == 0);
	REQUIRE(retained.release);
	CHECK(retained.length == 1);
	CHECK(probe->produced < 24);
	unique_ptr<QueryResult> next;
	if (replace_query) {
		next = con.SendQuery("SELECT 42");
		REQUIRE_FALSE(next->HasError());
	}
	stream.release(&stream);
	CHECK(probe->window->Snapshot().reserved.bytes == 0);
	CHECK(probe->window->Snapshot().waiting_requests == 0);
	CHECK(static_cast<const int64_t *>(retained.children[0]->buffers[1])[0] == 0);
	if (next) {
		auto chunk = next->Fetch();
		REQUIRE(chunk);
		CHECK(chunk->GetValue(0, 0) == Value::BIGINT(42));
	}
	retained.release(&retained);
	REQUIRE_NO_FAIL(con.Query("SELECT 43"));
}

TEST_CASE("A newer query callback can release its retired media Arrow stream", "[media-resource][media-result-flow]") {
	DuckDB db(nullptr);
	if (!db.instance->GetMediaFlowCapacity()) {
		return;
	}
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET streaming_buffer_size='8B'"));
	auto probe = RegisterFlow(db);
	ArrowArrayStream stream {};
	bool released = false;
	ExtensionLoader loader(*db.instance, "media_result_release_test");
	loader.RegisterFunction(ScalarFunction("release_old_media_stream", {LogicalType::BIGINT}, LogicalType::BIGINT,
	                                       [&](DataChunk &, ExpressionState &, Vector &output) {
		                                       stream.release(&stream);
		                                       released = true;
		                                       output.Reference(Value::BIGINT(42));
	                                       }));
	auto result = con.SendQuery("SELECT i FROM test_media_flow()");
	REQUIRE_FALSE(result->HasError());
	auto wrapper = new ResultArrowArrayStreamWrapper(std::move(result), 1000000);
	stream = wrapper->stream;
	ArrowArray retained;
	REQUIRE(stream.get_next(&stream, &retained) == 0);
	REQUIRE(retained.release);
	auto next = con.Query("SELECT release_old_media_stream(i) FROM range(1) t(i)");
	REQUIRE_FALSE(next->HasError());
	CHECK(released);
	CHECK(next->GetValue(0, 0) == Value::BIGINT(42));
	CHECK(probe->window->Snapshot().reserved.bytes == 0);
	CHECK(probe->window->Snapshot().waiting_requests == 0);
	CHECK(static_cast<const int64_t *>(retained.children[0]->buffers[1])[0] == 0);
	retained.release(&retained);
	REQUIRE_NO_FAIL(con.Query("SELECT 43"));
}

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
	for (auto query : {"SELECT i FROM test_media_flow() ORDER BY i LIMIT 3", "SELECT sum(i) FROM test_media_flow()"}) {
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
	dynamic.SetValue(0, 0, Value::LIST(LogicalType::UTINYINT, {}));
	dynamic.SetCardinality(1);
	CHECK(EstimateMediaFlowInputBytes(dynamic) == sizeof(list_entry_t));
}
