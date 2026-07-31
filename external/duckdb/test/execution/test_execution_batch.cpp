// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb/execution/executor.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/parallel/interrupt.hpp"
#include "duckdb/parallel/pipeline_executor.hpp"
#include "duckdb/parallel/thread_context.hpp"

using namespace duckdb;

namespace {

class RetryBlockingSink : public PhysicalOperator {
public:
	explicit RetryBlockingSink(PhysicalPlan &physical_plan)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::RESULT_COLLECTOR, {}, 3) {
	}

	SinkResultType Sink(ExecutionContext &, DataChunk &chunk, OperatorSinkInput &) const override {
		sink_calls++;
		if (sink_calls <= 2) {
			return SinkResultType::BLOCKED;
		}

		accepted_calls++;
		accepted_rows = chunk.size();
		accepted_values.clear();
		for (idx_t row = 0; row < chunk.size(); row++) {
			accepted_values.push_back(chunk.GetValue(0, row).GetValue<int64_t>());
		}
		return SinkResultType::NEED_MORE_INPUT;
	}

	mutable idx_t sink_calls = 0;
	mutable idx_t accepted_calls = 0;
	mutable idx_t accepted_rows = 0;
	mutable vector<int64_t> accepted_values;
};

class MultiOutputOperator : public PhysicalOperator {
public:
	explicit MultiOutputOperator(PhysicalPlan &physical_plan)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::PROJECTION, {LogicalType::BIGINT}, 3) {
	}

	OperatorResultType Execute(ExecutionContext &, DataChunk &input, DataChunk &output, GlobalOperatorState &,
	                           OperatorState &) const override {
		execute_calls++;
		observed_payloads.push_back(&input);
		observed_values.emplace_back();
		for (idx_t row = 0; row < input.size(); row++) {
			observed_values.back().push_back(input.GetValue(0, row).GetValue<int64_t>());
		}
		output.Reference(input);

		if (execute_calls <= 2) {
			return OperatorResultType::HAVE_MORE_OUTPUT;
		}
		consumed_inputs++;
		return OperatorResultType::NEED_MORE_INPUT;
	}

	mutable idx_t execute_calls = 0;
	mutable idx_t consumed_inputs = 0;
	mutable vector<const DataChunk *> observed_payloads;
	mutable vector<vector<int64_t>> observed_values;
};

class MaterializedCountingSource : public PhysicalOperator {
public:
	explicit MaterializedCountingSource(PhysicalPlan &physical_plan)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::TABLE_SCAN, {LogicalType::BIGINT}, 3) {
	}

	bool IsSource() const override {
		return true;
	}

	SourceResultType GetDataBatch(ExecutionContext &context, ExecutionBatch &batch,
	                              OperatorSourceInput &input) const override {
		batch_calls++;
		return PhysicalOperator::GetDataBatch(context, batch, input);
	}

	mutable idx_t data_calls = 0;
	mutable idx_t batch_calls = 0;
	mutable vector<const DataChunk *> output_addresses;

protected:
	SourceResultType GetDataInternal(ExecutionContext &, DataChunk &chunk, OperatorSourceInput &) const override {
		data_calls++;
		output_addresses.push_back(&chunk);
		if (next_value >= 3) {
			return SourceResultType::FINISHED;
		}
		chunk.SetCardinality(1);
		chunk.SetValue(0, 0, Value::BIGINT(NumericCast<int64_t>(next_value + 1)));
		next_value++;
		return SourceResultType::HAVE_MORE_OUTPUT;
	}

private:
	mutable idx_t next_value = 0;
};

class MaterializedCountingOperator : public PhysicalOperator {
public:
	explicit MaterializedCountingOperator(PhysicalPlan &physical_plan, bool requires_batch_p = false)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::PROJECTION, {LogicalType::BIGINT}, 3),
	      requires_batch(requires_batch_p) {
	}

	OperatorResultType Execute(ExecutionContext &, DataChunk &input, DataChunk &output, GlobalOperatorState &,
	                           OperatorState &) const override {
		execute_calls++;
		output_addresses.push_back(&output);
		output.Reference(input);
		return OperatorResultType::NEED_MORE_INPUT;
	}

	OperatorResultType ExecuteBatch(ExecutionContext &context, ExecutionBatch &input, ExecutionBatch &output,
	                                GlobalOperatorState &gstate, OperatorState &state) const override {
		batch_calls++;
		return PhysicalOperator::ExecuteBatch(context, input, output, gstate, state);
	}

	ExecutionBatchRequirement GetExecutionBatchRequirement(PipelineOperatorRole role) const override {
		return requires_batch && role == PipelineOperatorRole::INTERMEDIATE ? ExecutionBatchRequirement::REQUIRED
		                                                                    : ExecutionBatchRequirement::OPTIONAL;
	}

	mutable idx_t execute_calls = 0;
	mutable idx_t batch_calls = 0;
	mutable vector<const DataChunk *> output_addresses;

private:
	bool requires_batch;
};

class MaterializedCountingSink : public PhysicalOperator {
public:
	explicit MaterializedCountingSink(PhysicalPlan &physical_plan, bool requires_batch_p = false)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::RESULT_COLLECTOR, {}, 3),
	      requires_batch(requires_batch_p) {
	}

	bool IsSink() const override {
		return true;
	}

	SinkResultType Sink(ExecutionContext &, DataChunk &chunk, OperatorSinkInput &) const override {
		sink_calls++;
		for (idx_t row = 0; row < chunk.size(); row++) {
			values.push_back(chunk.GetValue(0, row).GetValue<int64_t>());
		}
		return SinkResultType::NEED_MORE_INPUT;
	}

	SinkResultType SinkBatch(ExecutionContext &context, ExecutionBatch &batch,
	                         OperatorSinkInput &input) const override {
		batch_calls++;
		return PhysicalOperator::SinkBatch(context, batch, input);
	}

	ExecutionBatchRequirement GetExecutionBatchRequirement(PipelineOperatorRole role) const override {
		return requires_batch && role == PipelineOperatorRole::SINK ? ExecutionBatchRequirement::REQUIRED
		                                                            : ExecutionBatchRequirement::OPTIONAL;
	}

	mutable idx_t sink_calls = 0;
	mutable idx_t batch_calls = 0;
	mutable vector<int64_t> values;

private:
	bool requires_batch;
};

class LazyBatchSource : public PhysicalOperator {
public:
	explicit LazyBatchSource(PhysicalPlan &physical_plan)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::TABLE_SCAN, {LogicalType::BIGINT}, 3) {
	}

	bool IsSource() const override {
		return true;
	}

	ExecutionBatchRequirement GetExecutionBatchRequirement(PipelineOperatorRole role) const override {
		return role == PipelineOperatorRole::SOURCE ? ExecutionBatchRequirement::REQUIRED
		                                            : ExecutionBatchRequirement::OPTIONAL;
	}

	SourceResultType GetDataBatch(ExecutionContext &, ExecutionBatch &batch, OperatorSourceInput &) const override {
		batch_calls++;
		batch = ExecutionBatch();
		if (emitted) {
			return SourceResultType::FINISHED;
		}

		ExternalBlockDescriptor block;
		block.metadata.num_rows = 3;
		block.metadata.size_bytes = 24;
		auto lazy = make_uniq<LazyDataChunk>();
		lazy->logical_types = types;
		lazy->names = {"value"};
		lazy->blocks.push_back(std::move(block));
		lazy->RecomputeCardinality();

		batch.kind = ExecutionBatchKind::LAZY_DATA_CHUNK;
		batch.rows = lazy->cardinality;
		batch.estimated_bytes = lazy->EstimatedBytes();
		batch.lazy = std::move(lazy);
		emitted = true;
		return SourceResultType::HAVE_MORE_OUTPUT;
	}

	mutable idx_t data_calls = 0;
	mutable idx_t batch_calls = 0;

protected:
	SourceResultType GetDataInternal(ExecutionContext &, DataChunk &, OperatorSourceInput &) const override {
		data_calls++;
		throw InternalException("LazyBatchSource requires ExecutionBatch execution");
	}

private:
	mutable bool emitted = false;
};

class LazyBatchSink : public PhysicalOperator {
public:
	explicit LazyBatchSink(PhysicalPlan &physical_plan)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::RESULT_COLLECTOR, {}, 3) {
	}

	bool IsSink() const override {
		return true;
	}

	SinkResultType Sink(ExecutionContext &, DataChunk &, OperatorSinkInput &) const override {
		data_calls++;
		throw InternalException("LazyBatchSink received materialized input");
	}

	SinkResultType SinkBatch(ExecutionContext &, ExecutionBatch &batch, OperatorSinkInput &) const override {
		batch_calls++;
		REQUIRE(batch.kind == ExecutionBatchKind::LAZY_DATA_CHUNK);
		REQUIRE(batch.lazy);
		accepted_rows += batch.lazy->cardinality;
		return SinkResultType::NEED_MORE_INPUT;
	}

	mutable idx_t data_calls = 0;
	mutable idx_t batch_calls = 0;
	mutable idx_t accepted_rows = 0;
};

static void RequireRetryablePayload(const ExecutionBatch &batch, const DataChunk *expected_payload) {
	REQUIRE(batch.kind == ExecutionBatchKind::MATERIALIZED_CHUNK);
	REQUIRE(batch.rows == 3);
	REQUIRE(batch.materialized);
	REQUIRE(batch.materialized.get() == expected_payload);
	REQUIRE(batch.materialized->size() == 3);
	REQUIRE(batch.materialized->GetValue(0, 0).GetValue<int64_t>() == 11);
	REQUIRE(batch.materialized->GetValue(0, 1).GetValue<int64_t>() == 22);
	REQUIRE(batch.materialized->GetValue(0, 2).GetValue<int64_t>() == 33);
}

static void VerifyStreamingBackpressure(idx_t threads) {
	DuckDB db(nullptr);
	Connection con(db);

	auto setting_result = con.Query("SET streaming_buffer_size='32KB'");
	REQUIRE_FALSE(setting_result->HasError());
	setting_result = con.Query("SET threads=" + to_string(threads));
	REQUIRE_FALSE(setting_result->HasError());

	auto result = con.SendQuery(R"(
		SELECT
			i,
			repeat(chr(65 + (i % 26)::INTEGER), 256) || ':' || i::VARCHAR AS payload,
			CASE WHEN i % 17 = 0 THEN NULL ELSE i * 3 END AS nullable
		FROM range(20000) t(i)
	)");
	REQUIRE_FALSE(result->HasError());
	REQUIRE(result->type == QueryResultType::STREAM_RESULT);

	idx_t expected_row = 0;
	while (auto chunk = result->Fetch()) {
		REQUIRE(chunk->size() > 0);
		for (idx_t row = 0; row < chunk->size(); row++) {
			REQUIRE(chunk->GetValue(0, row).GetValue<int64_t>() == NumericCast<int64_t>(expected_row));

			string expected_payload(256, static_cast<char>('A' + expected_row % 26));
			expected_payload += ":" + to_string(expected_row);
			REQUIRE(chunk->GetValue(1, row).GetValue<string>() == expected_payload);

			auto nullable = chunk->GetValue(2, row);
			if (expected_row % 17 == 0) {
				REQUIRE(nullable.IsNull());
			} else {
				REQUIRE(nullable.GetValue<int64_t>() == NumericCast<int64_t>(expected_row * 3));
			}
			expected_row++;
		}
	}

	REQUIRE_FALSE(result->HasError());
	REQUIRE(expected_row == 20000);
}

} // namespace

TEST_CASE("Materialized pipelines use and reuse DataChunk callbacks", "[execution_batch][pipeline]") {
	DuckDB db(nullptr);
	Connection con(db);
	Executor executor(*con.context);
	Pipeline pipeline(executor);
	PipelineBuildState build_state;
	PhysicalPlan physical_plan(Allocator::DefaultAllocator());
	MaterializedCountingSource source(physical_plan);
	MaterializedCountingOperator op(physical_plan);
	MaterializedCountingSink sink(physical_plan);

	build_state.SetPipelineSource(pipeline, source);
	build_state.AddPipelineOperator(pipeline, op);
	build_state.SetPipelineSink(pipeline, sink, 0);
	pipeline.Ready();
	REQUIRE(pipeline.GetExecutionMode() == PipelineExecutionMode::DATA_CHUNK);

	pipeline.Reset();
	PipelineExecutor pipeline_executor(*con.context, pipeline);
	REQUIRE(pipeline_executor.Execute() == PipelineExecuteResult::FINISHED);

	REQUIRE(source.data_calls == 4);
	REQUIRE(source.batch_calls == 0);
	REQUIRE(op.execute_calls == 3);
	REQUIRE(op.batch_calls == 0);
	REQUIRE(sink.sink_calls == 3);
	REQUIRE(sink.batch_calls == 0);
	REQUIRE(sink.values == vector<int64_t> {1, 2, 3});

	REQUIRE_FALSE(source.output_addresses.empty());
	for (auto address : source.output_addresses) {
		REQUIRE(address == source.output_addresses.front());
	}
	REQUIRE_FALSE(op.output_addresses.empty());
	for (auto address : op.output_addresses) {
		REQUIRE(address == op.output_addresses.front());
	}
}

TEST_CASE("Batch-required sources preserve lazy payloads through the pipeline", "[execution_batch][pipeline]") {
	DuckDB db(nullptr);
	Connection con(db);
	Executor executor(*con.context);
	Pipeline pipeline(executor);
	PipelineBuildState build_state;
	PhysicalPlan physical_plan(Allocator::DefaultAllocator());
	LazyBatchSource source(physical_plan);
	LazyBatchSink sink(physical_plan);

	build_state.SetPipelineSource(pipeline, source);
	build_state.SetPipelineSink(pipeline, sink, 0);
	pipeline.Ready();
	REQUIRE(pipeline.GetExecutionMode() == PipelineExecutionMode::EXECUTION_BATCH);

	pipeline.Reset();
	PipelineExecutor pipeline_executor(*con.context, pipeline);
	REQUIRE(pipeline_executor.Execute() == PipelineExecuteResult::FINISHED);

	REQUIRE(source.data_calls == 0);
	REQUIRE(source.batch_calls == 2);
	REQUIRE(sink.data_calls == 0);
	REQUIRE(sink.batch_calls == 1);
	REQUIRE(sink.accepted_rows == 3);
}

TEST_CASE("Intermediate and sink requirements select ExecutionBatch", "[execution_batch][pipeline]") {
	DuckDB db(nullptr);
	Connection con(db);
	PhysicalPlan physical_plan(Allocator::DefaultAllocator());

	SECTION("intermediate operator") {
		Executor executor(*con.context);
		Pipeline pipeline(executor);
		PipelineBuildState build_state;
		MaterializedCountingSource source(physical_plan);
		MaterializedCountingOperator op(physical_plan, true);
		MaterializedCountingSink sink(physical_plan);

		build_state.SetPipelineSource(pipeline, source);
		build_state.AddPipelineOperator(pipeline, op);
		build_state.SetPipelineSink(pipeline, sink, 0);
		pipeline.Ready();
		REQUIRE(pipeline.GetExecutionMode() == PipelineExecutionMode::EXECUTION_BATCH);
	}

	SECTION("sink") {
		Executor executor(*con.context);
		Pipeline pipeline(executor);
		PipelineBuildState build_state;
		MaterializedCountingSource source(physical_plan);
		MaterializedCountingSink sink(physical_plan, true);

		build_state.SetPipelineSource(pipeline, source);
		build_state.SetPipelineSink(pipeline, sink, 0);
		pipeline.Ready();
		REQUIRE(pipeline.GetExecutionMode() == PipelineExecutionMode::EXECUTION_BATCH);
	}
}

TEST_CASE("ExecutionBatch payload remains retryable after a sink blocks", "[execution_batch][sink]") {
	DuckDB db(nullptr);
	Connection con(db);
	ThreadContext thread(*con.context);
	ExecutionContext context(*con.context, thread, nullptr);
	PhysicalPlan physical_plan(Allocator::DefaultAllocator());
	RetryBlockingSink sink(physical_plan);

	auto payload = make_uniq<DataChunk>();
	payload->Initialize(Allocator::DefaultAllocator(), {LogicalType::BIGINT});
	payload->SetCardinality(3);
	payload->SetValue(0, 0, Value::BIGINT(11));
	payload->SetValue(0, 1, Value::BIGINT(22));
	payload->SetValue(0, 2, Value::BIGINT(33));
	auto payload_ptr = payload.get();

	ExecutionBatch batch;
	batch.kind = ExecutionBatchKind::MATERIALIZED_CHUNK;
	batch.rows = payload->size();
	batch.estimated_bytes = payload->GetAllocationSize();
	batch.materialized = std::move(payload);

	GlobalSinkState global_state;
	LocalSinkState local_state;
	InterruptState interrupt_state;
	OperatorSinkInput input {global_state, local_state, interrupt_state};

	REQUIRE(sink.SinkBatch(context, batch, input) == SinkResultType::BLOCKED);
	RequireRetryablePayload(batch, payload_ptr);
	REQUIRE(sink.accepted_calls == 0);

	REQUIRE(sink.SinkBatch(context, batch, input) == SinkResultType::BLOCKED);
	RequireRetryablePayload(batch, payload_ptr);
	REQUIRE(sink.accepted_calls == 0);

	REQUIRE(sink.SinkBatch(context, batch, input) == SinkResultType::NEED_MORE_INPUT);
	RequireRetryablePayload(batch, payload_ptr);
	REQUIRE(sink.sink_calls == 3);
	REQUIRE(sink.accepted_calls == 1);
	REQUIRE(sink.accepted_rows == 3);
	REQUIRE(sink.accepted_values == vector<int64_t> {11, 22, 33});
}

TEST_CASE("ExecutionBatch payload remains stable while an operator has more output", "[execution_batch][operator]") {
	DuckDB db(nullptr);
	Connection con(db);
	ThreadContext thread(*con.context);
	ExecutionContext context(*con.context, thread, nullptr);
	PhysicalPlan physical_plan(Allocator::DefaultAllocator());
	MultiOutputOperator multi_output(physical_plan);

	auto payload = make_uniq<DataChunk>();
	payload->Initialize(Allocator::DefaultAllocator(), {LogicalType::BIGINT});
	payload->SetCardinality(3);
	payload->SetValue(0, 0, Value::BIGINT(11));
	payload->SetValue(0, 1, Value::BIGINT(22));
	payload->SetValue(0, 2, Value::BIGINT(33));
	auto payload_ptr = payload.get();

	ExecutionBatch input;
	input.kind = ExecutionBatchKind::MATERIALIZED_CHUNK;
	input.rows = payload->size();
	input.estimated_bytes = payload->GetAllocationSize();
	input.materialized = std::move(payload);

	ExecutionBatch output;
	GlobalOperatorState global_state;
	OperatorState operator_state;
	for (idx_t call = 0; call < 3; call++) {
		auto result = multi_output.ExecuteBatch(context, input, output, global_state, operator_state);
		if (call < 2) {
			REQUIRE(result == OperatorResultType::HAVE_MORE_OUTPUT);
		} else {
			REQUIRE(result == OperatorResultType::NEED_MORE_INPUT);
		}

		RequireRetryablePayload(input, payload_ptr);
		REQUIRE(output.kind == ExecutionBatchKind::MATERIALIZED_CHUNK);
		REQUIRE(output.rows == 3);
		REQUIRE(output.materialized);
		REQUIRE(output.materialized->GetValue(0, 0).GetValue<int64_t>() == 11);
		REQUIRE(output.materialized->GetValue(0, 1).GetValue<int64_t>() == 22);
		REQUIRE(output.materialized->GetValue(0, 2).GetValue<int64_t>() == 33);
	}

	REQUIRE(multi_output.execute_calls == 3);
	REQUIRE(multi_output.consumed_inputs == 1);
	REQUIRE(multi_output.observed_payloads.size() == 3);
	REQUIRE(multi_output.observed_values.size() == 3);
	for (idx_t call = 0; call < 3; call++) {
		REQUIRE(multi_output.observed_payloads[call] == payload_ptr);
		REQUIRE(multi_output.observed_values[call] == vector<int64_t> {11, 22, 33});
	}
}

TEST_CASE("Native streaming preserves rows through backpressure", "[execution_batch][streaming]") {
	SECTION("single-threaded") {
		VerifyStreamingBackpressure(1);
	}
	SECTION("multi-threaded") {
		VerifyStreamingBackpressure(4);
	}
}
