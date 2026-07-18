// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/atomic.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/execution/udf_executor.hpp"
#include "duckdb/execution/executor.hpp"
#include "duckdb/execution/operator/helper/physical_result_collector.hpp"
#include "duckdb/execution/operator/projection/physical_udf_inout.hpp"
#include "duckdb/execution/operator/projection/physical_tableinout_function.hpp"
#include "duckdb/execution/operator/scan/physical_dummy_scan.hpp"
#include "duckdb/execution/physical_plan.hpp"
#include "duckdb/function/scalar/udf_functions.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/parallel/interrupt.hpp"

#include <chrono>
#include <cstdlib>
#include "duckdb/execution/distributed/utils/optional.hpp"
#include <condition_variable>
#include <stdexcept>
#include <set>
#include <thread>

using namespace duckdb;

namespace {

// Track which thread IDs create executors — verifies 1:1 thread-to-executor mapping.
struct PerThreadTracker {
	mutex lock;
	std::set<std::thread::id> executor_thread_ids;
	atomic<idx_t> executor_count {0};

	void RecordCreation() {
		lock_guard<mutex> guard(lock);
		executor_thread_ids.insert(std::this_thread::get_id());
		executor_count++;
	}
};

static PerThreadTracker *g_tracker = nullptr;
static idx_t ExecuteSingleRowUDFPlan(Connection &con, const Value &payload);

class RetainedBytesTestUDFExecutor : public UDFExecutor {
public:
	bool TrySubmit(DataChunk &args, DataChunk &rows, ClientContext &context, idx_t &submit_id) override {
		submit_id = Submit(args, rows, context);
		return true;
	}

	bool TrySubmitEnvelope(vector<unique_ptr<DataChunk>> &args, DataChunk &rows, ClientContext &context,
	                       idx_t &submit_id) override {
		if (args.size() != 1 || !args[0]) {
			throw InvalidInputException("test UDF executor requires exactly one materialized input chunk");
		}
		return TrySubmit(*args[0], rows, context, submit_id);
	}

	bool SupportsRefBundleInput() override {
		return false;
	}

	idx_t SubmitRefBundle(LazyRefDataChunk &, DataChunk &, ClientContext &) override {
		throw InvalidInputException("test UDF executor does not accept ref bundle input");
	}

	bool TrySubmitRefBundle(LazyRefDataChunk &bundle, DataChunk &rows, ClientContext &context,
	                        idx_t &submit_id) override {
		submit_id = SubmitRefBundle(bundle, rows, context);
		return true;
	}

	bool TrySubmitWithRetainedBytes(DataChunk &args, DataChunk &rows, ClientContext &context, idx_t,
	                                idx_t &submit_id) override {
		return TrySubmit(args, rows, context, submit_id);
	}

	bool TrySubmitEnvelopeWithRetainedBytes(vector<unique_ptr<DataChunk>> &args, DataChunk &rows,
	                                        ClientContext &context, idx_t, idx_t &submit_id) override {
		return TrySubmitEnvelope(args, rows, context, submit_id);
	}

	bool TrySubmitRefBundleWithRetainedBytes(LazyRefDataChunk &bundle, DataChunk &rows, ClientContext &context, idx_t,
	                                         idx_t &submit_id) override {
		return TrySubmitRefBundle(bundle, rows, context, submit_id);
	}

	bool SupportsAsyncWakeup() override {
		return false;
	}

	UDFWakeupRegistrationResult RegisterWakeup(InterruptState &) override {
		return UDFWakeupRegistrationResult::UNSUPPORTED;
	}

	void RegisterWakeupCallback(std::function<void()>) override {
		throw InvalidInputException("test UDF executor does not accept wakeup callbacks");
	}

	void EnqueueDeferredWakeup(std::function<void()>) override {
		throw InvalidInputException("test UDF executor does not accept deferred wakeups");
	}

	bool SupportsOutputConsumer() override {
		return false;
	}

	void RegisterOutputConsumer(UDFOutputConsumer) override {
		throw InvalidInputException("test UDF executor does not accept output consumers");
	}

	void NotifyOutputConsumerSpaceAvailable() override {
		throw InvalidInputException("test UDF executor does not accept output consumer notifications");
	}

	idx_t DebugSlotId() override {
		return 0;
	}

	InsertionOrderPreservingMap<string> Stats() override {
		return {};
	}
};

class TrackingUDFExecutor : public RetainedBytesTestUDFExecutor {
public:
	TrackingUDFExecutor() {
		if (g_tracker) {
			g_tracker->RecordCreation();
		}
	}

	idx_t Submit(DataChunk &, DataChunk &rows, ClientContext &context) override {
		lock_guard<mutex> guard(result_lock);
		if (!rows_copy) {
			rows_copy = make_uniq<DataChunk>();
			rows_copy->Initialize(context, rows.GetTypes(), rows.size());
			rows_copy->Append(rows, true);
			has_result = true;
		}
		return 1;
	}

	std::pair<bool, UDFResult> TakeReadyResult(ClientContext &context) override {
		lock_guard<mutex> guard(result_lock);
		if (!has_result) {
			return std::pair<bool, UDFResult>(false, UDFResult());
		}
		has_result = false;
		finished = true;

		auto outputs = make_uniq<DataChunk>();
		outputs->Initialize(context, {LogicalType::INTEGER}, rows_copy->size());
		outputs->SetCardinality(rows_copy->size());
		outputs->data[0].Reference(Value::INTEGER(42));

		UDFResult result;
		result.outputs = std::move(outputs);
		result.rows = std::move(rows_copy);
		return std::make_pair(true, std::move(result));
	}

	void FinishedSubmitting(ClientContext &) override {
	}

	bool AllTasksFinished(ClientContext &) override {
		lock_guard<mutex> guard(result_lock);
		return finished;
	}

private:
	mutex result_lock;
	bool has_result = false;
	bool finished = false;
	unique_ptr<DataChunk> rows_copy;
};

unique_ptr<UDFExecutor> CreateTrackingExecutor(ClientContext &, const Value &, UDFConfig &, shared_ptr<void>) {
	return make_uniq<TrackingUDFExecutor>();
}

struct ScopedFactory {
	udf_executor_factory_t previous;

	explicit ScopedFactory(udf_executor_factory_t next) {
		previous = GetUDFExecutorFactory();
		SetUDFExecutorFactory(next);
	}

	~ScopedFactory() {
		SetUDFExecutorFactory(previous);
	}
};

void AddPayloadIdentityFields(child_list_t<Value> &children, const string &execution_backend, const string &call_mode) {
	children.emplace_back("payload_version", Value::BIGINT(1));
	children.emplace_back("udf_name", Value("test_udf"));
	children.emplace_back("call_mode", Value(call_mode));
	children.emplace_back("execution_backend", Value(execution_backend));
}

void AddScalarPayloadFields(child_list_t<Value> &children, const string &execution_backend) {
	AddPayloadIdentityFields(children, execution_backend, "map");
	children.emplace_back("method_return_type", Value("INTEGER"));
	children.emplace_back("ref_output_types", Value::LIST(LogicalType::VARCHAR, {Value("INTEGER")}));
}

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

Value MakePerThreadPayload() {
	child_list_t<Value> children;
	AddScalarPayloadFields(children, "subprocess_task");
	return Value::STRUCT(std::move(children));
}

Value MakePayloadWithActorCount(idx_t actor_count, const char *parallel_mode = nullptr) {
	child_list_t<Value> children;
	AddScalarPayloadFields(children, "ray_actor");
	children.emplace_back("actor_number", Value::BIGINT(static_cast<int64_t>(actor_count)));
	if (parallel_mode) {
		children.emplace_back("parallel_mode", Value(parallel_mode));
	}
	return Value::STRUCT(std::move(children));
}

Value MakeStreamingPayload() {
	child_list_t<Value> children;
	AddPayloadIdentityFields(children, "ray_task", "map_batches");
	children.emplace_back("output_schema", MakeIntegerOutputSchema());
	children.emplace_back("ref_output_types", Value::LIST(LogicalType::VARCHAR, {Value("INTEGER")}));
	children.emplace_back("streaming_breaker", Value::BOOLEAN(true));
	children.emplace_back("udf_task_input_max_bytes", Value::BIGINT(128 * 1024 * 1024));
	children.emplace_back("udf_output_target_max_bytes", Value::BIGINT(128 * 1024 * 1024));
	return Value::STRUCT(std::move(children));
}

struct AsyncWakeupTracker {
	atomic<idx_t> register_count {0};
	atomic<idx_t> callback_count {0};
	atomic<idx_t> ready_count {0};
	atomic<idx_t> wait_count {0};
};

static AsyncWakeupTracker *g_async_tracker = nullptr;

class AsyncWakeupUDFExecutor : public RetainedBytesTestUDFExecutor {
public:
	~AsyncWakeupUDFExecutor() override {
		if (worker.joinable()) {
			worker.join();
		}
	}

	idx_t Submit(DataChunk &, DataChunk &rows, ClientContext &context) override {
		{
			lock_guard<mutex> guard(lock);
			if (submitted) {
				throw std::runtime_error("AsyncWakeupUDFExecutor received duplicate submit");
			}
			rows_copy = make_uniq<DataChunk>();
			rows_copy->Initialize(context, rows.GetTypes(), rows.size());
			rows.Copy(*rows_copy, 0);
			submitted = true;
		}
		worker = std::thread([this]() {
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			InterruptState callback_state;
			bool should_callback = false;
			{
				lock_guard<mutex> guard(lock);
				result_ready = true;
				if (has_interrupt_state) {
					callback_state = interrupt_state;
					should_callback = true;
				}
			}
			cv.notify_all();
			if (should_callback) {
				if (g_async_tracker) {
					g_async_tracker->callback_count++;
				}
				callback_state.Callback();
			}
		});
		return 1;
	}

	std::pair<bool, UDFResult> TakeReadyResult(ClientContext &context) override {
		lock_guard<mutex> guard(lock);
		if (!result_ready) {
			return std::make_pair(false, UDFResult());
		}
		result_ready = false;
		finished = true;

		auto outputs = make_uniq<DataChunk>();
		outputs->Initialize(context, {LogicalType::INTEGER}, rows_copy->size());
		outputs->SetCardinality(rows_copy->size());
		outputs->data[0].Reference(Value::INTEGER(7));

		UDFResult result;
		result.outputs = std::move(outputs);
		result.rows = std::move(rows_copy);
		cv.notify_all();
		return std::make_pair(true, std::move(result));
	}

	void FinishedSubmitting(ClientContext &) override {
	}

	bool AllTasksFinished(ClientContext &) override {
		lock_guard<mutex> guard(lock);
		return submitted && finished && !result_ready;
	}

	bool SupportsAsyncWakeup() override {
		return true;
	}

	UDFWakeupRegistrationResult RegisterWakeup(InterruptState &interrupt_state_p) override {
		{
			lock_guard<mutex> guard(lock);
			if (result_ready) {
				if (g_async_tracker) {
					g_async_tracker->ready_count++;
				}
				return UDFWakeupRegistrationResult::READY;
			}
			interrupt_state = interrupt_state_p;
			has_interrupt_state = true;
		}
		if (g_async_tracker) {
			g_async_tracker->register_count++;
		}
		return UDFWakeupRegistrationResult::ARMED;
	}

private:
	mutex lock;
	std::condition_variable cv;
	bool submitted = false;
	bool result_ready = false;
	bool finished = false;
	bool has_interrupt_state = false;
	InterruptState interrupt_state;
	unique_ptr<DataChunk> rows_copy;
	std::thread worker;
};

unique_ptr<UDFExecutor> CreateAsyncWakeupExecutor(ClientContext &, const Value &, UDFConfig &, shared_ptr<void>) {
	return make_uniq<AsyncWakeupUDFExecutor>();
}

class ReadyDuringWakeupRegistrationExecutor : public RetainedBytesTestUDFExecutor {
public:
	idx_t Submit(DataChunk &, DataChunk &rows, ClientContext &context) override {
		lock_guard<mutex> guard(lock);
		if (submitted) {
			throw std::runtime_error("ReadyDuringWakeupRegistrationExecutor received duplicate submit");
		}
		rows_copy = make_uniq<DataChunk>();
		rows_copy->Initialize(context, rows.GetTypes(), rows.size());
		rows.Copy(*rows_copy, 0);
		submitted = true;
		return 1;
	}

	std::pair<bool, UDFResult> TakeReadyResult(ClientContext &context) override {
		lock_guard<mutex> guard(lock);
		if (!submitted) {
			return std::make_pair(false, UDFResult());
		}
		if (finished) {
			return std::make_pair(false, UDFResult());
		}
		if (!result_ready) {
			result_ready = true;
			cv.notify_all();
			return std::make_pair(false, UDFResult());
		}
		result_ready = false;
		finished = true;

		auto outputs = make_uniq<DataChunk>();
		outputs->Initialize(context, {LogicalType::INTEGER}, rows_copy->size());
		outputs->SetCardinality(rows_copy->size());
		outputs->data[0].Reference(Value::INTEGER(11));

		UDFResult result;
		result.outputs = std::move(outputs);
		result.rows = std::move(rows_copy);
		cv.notify_all();
		return std::make_pair(true, std::move(result));
	}

	void FinishedSubmitting(ClientContext &) override {
	}

	bool AllTasksFinished(ClientContext &) override {
		lock_guard<mutex> guard(lock);
		return submitted && finished && !result_ready;
	}

	bool SupportsAsyncWakeup() override {
		return true;
	}

	UDFWakeupRegistrationResult RegisterWakeup(InterruptState &interrupt_state_p) override {
		lock_guard<mutex> guard(lock);
		if (g_async_tracker) {
			g_async_tracker->register_count++;
		}
		if (result_ready) {
			if (g_async_tracker) {
				g_async_tracker->ready_count++;
			}
			return UDFWakeupRegistrationResult::READY;
		}
		interrupt_state = interrupt_state_p;
		has_interrupt_state = true;
		return UDFWakeupRegistrationResult::ARMED;
	}

private:
	mutex lock;
	std::condition_variable cv;
	bool submitted = false;
	bool result_ready = false;
	bool finished = false;
	bool has_interrupt_state = false;
	InterruptState interrupt_state;
	unique_ptr<DataChunk> rows_copy;
};

unique_ptr<UDFExecutor> CreateReadyDuringWakeupRegistrationExecutor(ClientContext &, const Value &, UDFConfig &,
                                                                    shared_ptr<void>) {
	return make_uniq<ReadyDuringWakeupRegistrationExecutor>();
}

static idx_t ExecuteSingleRowUDFPlan(Connection &con, const Value &payload) {
	auto &context = *con.context;

	auto physical_plan = make_uniq<PhysicalPlan>(Allocator::DefaultAllocator());
	vector<LogicalType> input_types = {LogicalType::INTEGER};
	auto &scan = physical_plan->Make<PhysicalDummyScan>(input_types, idx_t(1));

	vector<LogicalType> return_types = {LogicalType::INTEGER};
	vector<string> return_names = {"out"};
	auto table_function = MakeUDFTableFunction(payload, return_types, return_names);
	auto bind_data = make_uniq<UDFFunctionData>(payload, return_types[0]);
	vector<ColumnIndex> column_ids;
	column_ids.emplace_back(0);

	vector<LogicalType> output_types = {LogicalType::INTEGER, LogicalType::INTEGER};
	auto &inout_op =
	    physical_plan->Make<PhysicalTableInOutFunction>(output_types, std::move(table_function), std::move(bind_data),
	                                                    std::move(column_ids), idx_t(1), vector<column_t>());
	inout_op.children.emplace_back(scan);
	physical_plan->SetRoot(inout_op);

	auto prepared = make_shared_ptr<PreparedStatementData>(StatementType::SELECT_STATEMENT);
	prepared->names = {"input", "out"};
	prepared->types = output_types;
	prepared->properties.return_type = StatementReturnType::QUERY_RESULT;
	prepared->output_type = QueryResultOutputType::FORCE_MATERIALIZED;
	prepared->memory_type = QueryResultMemoryType::IN_MEMORY;
	prepared->physical_plan = std::move(physical_plan);

	auto &sink = PhysicalResultCollector::GetResultCollector(context, *prepared);

	Executor executor(context);
	executor.Initialize(sink);

	idx_t blocked_count = 0;
	while (!executor.ExecutionIsFinished()) {
		auto result = executor.ExecuteTask();
		if (result == PendingExecutionResult::BLOCKED) {
			blocked_count++;
			executor.WaitForTask();
		} else if (result == PendingExecutionResult::NO_TASKS_AVAILABLE) {
			executor.WaitForTask();
		}
		if (executor.HasError()) {
			executor.ThrowException();
		}
	}
	return blocked_count;
}

static idx_t GetUDFOperatorMaxThreads(Connection &con, const Value &payload, idx_t source_max_threads) {
	auto physical_plan = make_uniq<PhysicalPlan>(Allocator::DefaultAllocator());
	vector<LogicalType> input_types = {LogicalType::INTEGER};
	auto &scan = physical_plan->Make<PhysicalDummyScan>(input_types, idx_t(1));

	vector<LogicalType> return_types = {LogicalType::INTEGER};
	vector<string> return_names = {"out"};
	auto table_function = MakeUDFTableFunction(payload, return_types, return_names);
	auto bind_data = make_uniq<UDFFunctionData>(payload, return_types[0]);
	vector<ColumnIndex> column_ids;
	column_ids.emplace_back(0);

	vector<LogicalType> output_types = {LogicalType::INTEGER, LogicalType::INTEGER};
	auto &inout_op =
	    physical_plan->Make<PhysicalTableInOutFunction>(output_types, std::move(table_function), std::move(bind_data),
	                                                    std::move(column_ids), idx_t(1), vector<column_t>());
	inout_op.children.emplace_back(scan);

	auto gstate = inout_op.GetGlobalOperatorState(*con.context);
	return gstate->MaxThreads(source_max_threads);
}

static idx_t GetStreamingUDFSourceMaxThreads(Connection &con, const Value &payload) {
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

} // namespace

TEST_CASE("Per-thread executor creation", "[execution][udf][per_thread]") {
	SECTION("each execution creates its own executor via factory") {
		PerThreadTracker tracker;
		g_tracker = &tracker;

		DuckDB db(nullptr);
		Connection con(db);

		ScopedFactory factory(CreateTrackingExecutor);
		auto payload = MakePerThreadPayload();
		ExecuteSingleRowUDFPlan(con, payload);

		// The factory should have been called at least once (one executor per slot)
		REQUIRE(tracker.executor_count > 0);

		g_tracker = nullptr;
	}

	SECTION("async executor waits for result without task blocking") {
		AsyncWakeupTracker tracker;
		g_async_tracker = &tracker;

		DuckDB db(nullptr);
		Connection con(db);

		ScopedFactory factory(CreateAsyncWakeupExecutor);
		auto blocked_count = ExecuteSingleRowUDFPlan(con, MakePerThreadPayload());

		REQUIRE(blocked_count > 0);
		REQUIRE(tracker.register_count > 0);
		REQUIRE(tracker.callback_count > 0);
		REQUIRE(tracker.wait_count == 0);

		g_async_tracker = nullptr;
	}

	SECTION("ready result during blocking wait does not block") {
		AsyncWakeupTracker tracker;
		g_async_tracker = &tracker;

		DuckDB db(nullptr);
		Connection con(db);

		ScopedFactory factory(CreateReadyDuringWakeupRegistrationExecutor);
		auto blocked_count = ExecuteSingleRowUDFPlan(con, MakePerThreadPayload());

		REQUIRE(blocked_count == 0);
		REQUIRE(tracker.register_count > 0);
		REQUIRE(tracker.ready_count > 0);
		REQUIRE(tracker.callback_count == 0);
		REQUIRE(tracker.wait_count == 0);

		g_async_tracker = nullptr;
	}

	SECTION("operator max threads stays with source width") {
		DuckDB db(nullptr);
		Connection con(db);

		REQUIRE(GetUDFOperatorMaxThreads(con, MakePerThreadPayload(), 36) == 36);
		REQUIRE(GetUDFOperatorMaxThreads(con, MakePayloadWithActorCount(2), 36) == 36);
		REQUIRE(GetUDFOperatorMaxThreads(con, MakePayloadWithActorCount(4), 36) == 36);
		REQUIRE(GetUDFOperatorMaxThreads(con, MakePayloadWithActorCount(8), 6) == 6);
		REQUIRE(GetUDFOperatorMaxThreads(con, MakePayloadWithActorCount(8, "off"), 36) == 36);
	}

	SECTION("streaming breaker has one queue-draining source task") {
		DuckDB db(nullptr);
		Connection con(db);

		REQUIRE(GetStreamingUDFSourceMaxThreads(con, MakeStreamingPayload()) == 1);
	}
}
