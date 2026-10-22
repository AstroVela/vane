// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb/execution/executor.hpp"
#include "duckdb/common/dl.hpp"
#include "duckdb/execution/external_block.hpp"
#include "duckdb/execution/operator/scan/physical_empty_result.hpp"
#include "duckdb/execution/operator/persistent/physical_copy_to_file.hpp"
#include "duckdb/execution/operator/exchange/physical_local_exchange.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/execution/udf_executor.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/parallel/interrupt.hpp"
#include "duckdb/parallel/pipeline_executor.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

using namespace duckdb;

namespace {

class RetryBlockingSink : public PhysicalOperator {
public:
	explicit RetryBlockingSink(PhysicalPlan &physical_plan,
	                           PhysicalOperatorType type = PhysicalOperatorType::RESULT_COLLECTOR)
	    : PhysicalOperator(physical_plan, type, {}, 3) {
	}

	bool IsSink() const override {
		return true;
	}

	SinkResultType SinkBatch(ExecutionContext &context, ExecutionBatch &batch,
	                         OperatorSinkInput &input) const override {
		batch_calls++;
		return PhysicalOperator::SinkBatch(context, batch, input);
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
	mutable idx_t batch_calls = 0;
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
		return requires_batch && role == PipelineOperatorRole::INTERMEDIATE ? ExecutionBatchRequirement::BATCH_REQUIRED
		                                                                    : ExecutionBatchRequirement::BATCH_OPTIONAL;
	}

	mutable idx_t execute_calls = 0;
	mutable idx_t batch_calls = 0;
	mutable vector<const DataChunk *> output_addresses;

private:
	bool requires_batch;
};

class MaterializedCountingSink : public PhysicalOperator {
public:
	explicit MaterializedCountingSink(PhysicalPlan &physical_plan, bool requires_batch_p = false,
	                                  PhysicalOperatorType type = PhysicalOperatorType::RESULT_COLLECTOR)
	    : PhysicalOperator(physical_plan, type, {}, 3), requires_batch(requires_batch_p) {
	}

	bool IsSink() const override {
		return true;
	}

	SinkResultType Sink(ExecutionContext &, DataChunk &chunk, OperatorSinkInput &) const override {
		sink_calls++;
		if (fail_consume) {
			throw InvalidInputException("test sink failure");
		}
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
		return requires_batch && role == PipelineOperatorRole::SINK ? ExecutionBatchRequirement::BATCH_REQUIRED
		                                                            : ExecutionBatchRequirement::BATCH_OPTIONAL;
	}

	mutable idx_t sink_calls = 0;
	mutable idx_t batch_calls = 0;
	mutable vector<int64_t> values;
	bool fail_consume = false;

private:
	bool requires_batch;
};

class ExhaustedPartitionedSource : public MaterializedCountingSource {
public:
	explicit ExhaustedPartitionedSource(PhysicalPlan &physical_plan) : MaterializedCountingSource(physical_plan) {
	}

	bool SupportsPartitioning(const OperatorPartitionInfo &partition_info) const override {
		return partition_info.RequiresBatchIndex() && !partition_info.RequiresPartitionColumns();
	}

protected:
	SourceResultType GetDataInternal(ExecutionContext &, DataChunk &, OperatorSourceInput &) const override {
		return SourceResultType::FINISHED;
	}
};

class FinalOutputOperator : public MaterializedCountingOperator {
public:
	explicit FinalOutputOperator(PhysicalPlan &physical_plan) : MaterializedCountingOperator(physical_plan) {
	}

	bool RequiresFinalExecute() const override {
		return true;
	}

	OperatorFinalizeResultType FinalExecute(ExecutionContext &, DataChunk &output, GlobalOperatorState &,
	                                        OperatorState &) const override {
		output.SetCardinality(1);
		output.SetValue(0, 0, Value::BIGINT(42));
		return OperatorFinalizeResultType::FINISHED;
	}
};

class InitialBatchSink : public MaterializedCountingSink {
public:
	InitialBatchSink(PhysicalPlan &physical_plan, bool requires_batch, bool block_initial_p)
	    : MaterializedCountingSink(physical_plan, requires_batch), block_initial(block_initial_p) {
	}

	OperatorPartitionInfo RequiredPartitionInfo() const override {
		return OperatorPartitionInfo::BatchIndex();
	}

	SinkNextBatchType NextBatch(ExecutionContext &, OperatorSinkNextBatchInput &input) const override {
		next_batch_calls++;
		if (block_initial && next_batch_calls == 1) {
			return SinkNextBatchType::BLOCKED;
		}
		current_batch = input.local_state.partition_info.batch_index.GetIndex();
		return SinkNextBatchType::READY;
	}

	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override {
		REQUIRE(current_batch.IsValid());
		REQUIRE(current_batch.GetIndex() == input.local_state.partition_info.batch_index.GetIndex());
		return MaterializedCountingSink::Sink(context, chunk, input);
	}

	mutable idx_t next_batch_calls = 0;
	mutable optional_idx current_batch;

private:
	bool block_initial;
};

class LazyBatchSource : public PhysicalOperator {
public:
	explicit LazyBatchSource(PhysicalPlan &physical_plan, shared_ptr<void> ownership_token_p = nullptr,
	                         shared_ptr<ExternalBlockBackendInterface> materializer_p = nullptr)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::TABLE_SCAN, {LogicalType::BIGINT}, 3),
	      ownership_token(std::move(ownership_token_p)), materializer(std::move(materializer_p)) {
	}

	bool IsSource() const override {
		return true;
	}

	ExecutionBatchRequirement GetExecutionBatchRequirement(PipelineOperatorRole role) const override {
		return role == PipelineOperatorRole::SOURCE ? ExecutionBatchRequirement::BATCH_REQUIRED
		                                            : ExecutionBatchRequirement::BATCH_OPTIONAL;
	}

	SourceResultType GetDataBatch(ExecutionContext &, ExecutionBatch &batch, OperatorSourceInput &) const override {
		batch_calls++;
		batch = ExecutionBatch();
		if (emitted) {
			return SourceResultType::FINISHED;
		}

		ExternalBlockDescriptor block;
		block.materializer = materializer;
		block.metadata.num_rows = 3;
		block.metadata.size_bytes = 24;
		if (ownership_token) {
			block.ownership_tokens.push_back(std::move(ownership_token));
		}
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
	mutable shared_ptr<void> ownership_token;
	shared_ptr<ExternalBlockBackendInterface> materializer;
};

struct BatchBackendGuard {
	BatchBackendGuard() : previous(GetExternalBlockBackend(ExternalBlockBackend::RAY_OBJECT_STORE)) {
	}
	~BatchBackendGuard() {
		SetExternalBlockBackend(ExternalBlockBackend::RAY_OBJECT_STORE, std::move(previous));
	}

	shared_ptr<ExternalBlockBackendInterface> previous;
};

class CountingBatchBackend : public ExternalBlockBackendInterface {
public:
	bool CanMaterialize(const ExternalBlockDescriptor &) override {
		return true;
	}

	unique_ptr<DataChunk> Materialize(ClientContext &context, const LazyDataChunk &chunk) override {
		materialize_calls++;
		if (fail_materialize) {
			throw InvalidInputException("test batch materialization failure");
		}
		auto result = make_uniq<DataChunk>();
		result->Initialize(context, chunk.logical_types);
		result->SetCardinality(chunk.cardinality);
		for (idx_t row = 0; row < chunk.cardinality; row++) {
			result->SetValue(0, row, Value::BIGINT(NumericCast<int64_t>((row + 1) * 11)));
		}
		return result;
	}

	idx_t materialize_calls = 0;
	bool fail_materialize = false;
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

class LazyExtensionSink : public LazyBatchSink {
public:
	LazyExtensionSink(PhysicalPlan &physical_plan, bool required_p)
	    : LazyBatchSink(physical_plan), required(required_p) {
		type = PhysicalOperatorType::EXTENSION;
	}

	ExecutionBatchRequirement GetExecutionBatchRequirement(PipelineOperatorRole role) const override {
		return required && role == PipelineOperatorRole::SINK ? ExecutionBatchRequirement::BATCH_REQUIRED
		                                                      : ExecutionBatchRequirement::BATCH_OPTIONAL;
	}

private:
	bool required;
};

class LazyExtensionOperator : public PhysicalOperator {
public:
	LazyExtensionOperator(PhysicalPlan &plan, bool required_p)
	    : PhysicalOperator(plan, PhysicalOperatorType::EXTENSION, {LogicalType::BIGINT}, 3), required(required_p) {
	}

	ExecutionBatchRequirement GetExecutionBatchRequirement(PipelineOperatorRole role) const override {
		return required && role == PipelineOperatorRole::INTERMEDIATE ? ExecutionBatchRequirement::BATCH_REQUIRED
		                                                              : ExecutionBatchRequirement::BATCH_OPTIONAL;
	}

	OperatorResultType Execute(ExecutionContext &, DataChunk &, DataChunk &, GlobalOperatorState &,
	                           OperatorState &) const override {
		throw InternalException("lazy extension callback was bypassed");
	}

	OperatorResultType ExecuteBatch(ExecutionContext &, ExecutionBatch &input, ExecutionBatch &output,
	                                GlobalOperatorState &, OperatorState &) const override {
		batch_calls++;
		REQUIRE(input.kind == ExecutionBatchKind::LAZY_DATA_CHUNK);
		REQUIRE(input.lazy);
		output = ExecutionBatch();
		output.kind = input.kind;
		output.rows = input.rows;
		output.estimated_bytes = input.estimated_bytes;
		output.lazy = make_uniq<LazyDataChunk>(*input.lazy);
		return OperatorResultType::NEED_MORE_INPUT;
	}

	mutable idx_t batch_calls = 0;

private:
	bool required;
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

static ExecutionBatch MakeLazyExecutionBatch(shared_ptr<ExternalBlockBackendInterface> materializer) {
	ExecutionBatch batch;
	batch.kind = ExecutionBatchKind::LAZY_DATA_CHUNK;
	batch.lazy = make_uniq<LazyDataChunk>();
	batch.lazy->logical_types = {LogicalType::BIGINT};
	batch.lazy->names = {"value"};
	ExternalBlockDescriptor block;
	block.materializer = std::move(materializer);
	block.metadata.num_rows = 3;
	block.metadata.size_bytes = 24;
	batch.lazy->blocks.push_back(std::move(block));
	batch.lazy->RecomputeCardinality();
	batch.rows = 3;
	batch.estimated_bytes = 24;
	return batch;
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

TEST_CASE("Schema-only pipelines preserve empty nested results without dense buffers", "[execution_batch][pipeline]") {
	DuckDB db(nullptr);
	Connection con(db);
	REQUIRE_FALSE(con.Query("SET memory_limit='16MB'")->HasError());
	REQUIRE_FALSE(con.Query("SET enable_profiling='no_output'")->HasError());

	// A full vector of this ordinary nested ARRAY would require 8 GiB. Neither
	// EMPTY_RESULT nor the root RESULT_COLLECTOR needs to allocate those rows.
	auto result = con.Query("SELECT NULL::BIGINT AS id, NULL::FLOAT[1024][1024] AS tensor, "
	                        "NULL::STRUCT(items INTEGER[]) AS payload WHERE false");
	INFO((result->HasError() ? result->GetError() : ""));
	REQUIRE_FALSE(result->HasError());
	REQUIRE(result->RowCount() == 0);
	REQUIRE(result->types ==
	        vector<LogicalType> {LogicalType::BIGINT,
	                             LogicalType::ARRAY(LogicalType::ARRAY(LogicalType::FLOAT, 1024), 1024),
	                             LogicalType::STRUCT({{"items", LogicalType::LIST(LogicalType::INTEGER)}})});
	REQUIRE_FALSE(result->Fetch());
}

TEST_CASE("Schema-only root collectors return their buffered nested rows", "[execution_batch][pipeline]") {
	DuckDB db(nullptr);
	Connection con(db);
	const idx_t row_count = STANDARD_VECTOR_SIZE * 2 + 1;
	auto result = con.Query("SELECT i, [i, NULL]::BIGINT[2], {'items': [i, NULL]} FROM range(" + to_string(row_count) +
	                        ") t(i) ORDER BY i");
	REQUIRE_FALSE(result->HasError());
	REQUIRE(result->RowCount() == row_count);

	idx_t expected_row = 0;
	while (auto chunk = result->Fetch()) {
		for (idx_t row = 0; row < chunk->size(); row++) {
			auto value = Value::BIGINT(NumericCast<int64_t>(expected_row++));
			REQUIRE(chunk->GetValue(0, row) == value);
			REQUIRE(chunk->GetValue(1, row) == Value::ARRAY(LogicalType::BIGINT, {value, Value(LogicalType::BIGINT)}));
			REQUIRE(chunk->GetValue(2, row) ==
			        Value::STRUCT({{"items", Value::LIST(LogicalType::BIGINT, {value, Value(LogicalType::BIGINT)})}}));
		}
	}
	REQUIRE(expected_row == row_count);
}

TEST_CASE("Empty batch sources retain nested schemas without allocating rows", "[execution_batch][pipeline]") {
	DuckDB db(nullptr);
	Connection con(db);
	REQUIRE_FALSE(con.Query("SET memory_limit='16MB'")->HasError());
	ThreadContext thread(*con.context);
	ExecutionContext context(*con.context, thread, nullptr);
	PhysicalPlan physical_plan(Allocator::DefaultAllocator());
	vector<LogicalType> types {LogicalType::BIGINT,
	                           LogicalType::ARRAY(LogicalType::ARRAY(LogicalType::FLOAT, 1024), 1024),
	                           LogicalType::STRUCT({{"items", LogicalType::LIST(LogicalType::INTEGER)}})};
	PhysicalEmptyResult source(physical_plan, types, 0);
	GlobalSourceState global_state;
	LocalSourceState local_state;
	InterruptState interrupt_state;
	OperatorSourceInput input {global_state, local_state, interrupt_state};
	ExecutionBatch batch;
	REQUIRE(source.GetDataBatch(context, batch, input) == SourceResultType::FINISHED);
	REQUIRE(batch.kind == ExecutionBatchKind::MATERIALIZED_CHUNK);
	REQUIRE(batch.rows == 0);
	REQUIRE(batch.estimated_bytes == 0);
	REQUIRE(batch.materialized);
	REQUIRE(batch.materialized->GetTypes() == types);
	batch.materialized->Reset();
	REQUIRE(batch.materialized->size() == 0);
}

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

TEST_CASE("Ordinary extension sinks preserve their optional batch callback", "[execution_batch][pipeline][extension]") {
	DuckDB db(nullptr);
	Connection con(db);
	BatchBackendGuard backend_guard;
	auto backend = make_shared_ptr<CountingBatchBackend>();
	SetExternalBlockBackend(ExternalBlockBackend::RAY_OBJECT_STORE, backend);
	idx_t released = 0;
	auto token = shared_ptr<void>(new int(1), [&released](void *ptr) {
		delete static_cast<int *>(ptr);
		released++;
	});
	Executor executor(*con.context);
	Pipeline pipeline(executor);
	PipelineBuildState build_state;
	PhysicalPlan physical_plan(Allocator::DefaultAllocator());
	LazyBatchSource source(physical_plan, std::move(token), backend);
	MaterializedCountingSink sink(physical_plan, false, PhysicalOperatorType::EXTENSION);

	build_state.SetPipelineSource(pipeline, source);
	build_state.SetPipelineSink(pipeline, sink, 0);
	pipeline.Ready();
	pipeline.Reset();
	PipelineExecutor pipeline_executor(*con.context, pipeline);
	REQUIRE(pipeline_executor.Execute() == PipelineExecuteResult::FINISHED);
	REQUIRE(backend->materialize_calls == 1);
	REQUIRE(sink.batch_calls == 1);
	REQUIRE(sink.sink_calls == 1);
	REQUIRE(sink.values == vector<int64_t> {11, 22, 33});
	REQUIRE(released == 1);
}

TEST_CASE("Optional and required extension sinks retain their lazy callback",
          "[execution_batch][pipeline][extension]") {
	bool required = false;
	SECTION("optional") {
	}
	SECTION("required") {
		required = true;
	}
	DuckDB db(nullptr);
	Connection con(db);
	BatchBackendGuard backend_guard;
	SetExternalBlockBackend(ExternalBlockBackend::RAY_OBJECT_STORE, nullptr);
	Executor executor(*con.context);
	Pipeline pipeline(executor);
	PipelineBuildState build_state;
	PhysicalPlan physical_plan(Allocator::DefaultAllocator());
	LazyBatchSource source(physical_plan);
	LazyExtensionSink sink(physical_plan, required);

	build_state.SetPipelineSource(pipeline, source);
	build_state.SetPipelineSink(pipeline, sink, 0);
	pipeline.Ready();
	pipeline.Reset();
	PipelineExecutor pipeline_executor(*con.context, pipeline);
	REQUIRE(pipeline_executor.Execute() == PipelineExecuteResult::FINISHED);
	REQUIRE(sink.data_calls == 0);
	REQUIRE(sink.batch_calls == 1);
	REQUIRE(sink.accepted_rows == 3);
}

TEST_CASE("Optional and required extension intermediates retain their lazy callback",
          "[execution_batch][pipeline][extension]") {
	bool required = false;
	SECTION("optional") {
	}
	SECTION("required") {
		required = true;
	}
	DuckDB db(nullptr);
	Connection con(db);
	BatchBackendGuard guard;
	SetExternalBlockBackend(ExternalBlockBackend::RAY_OBJECT_STORE, nullptr);
	Executor executor(*con.context);
	Pipeline pipeline(executor);
	PipelineBuildState build_state;
	PhysicalPlan plan(Allocator::DefaultAllocator());
	LazyBatchSource source(plan);
	LazyExtensionOperator op(plan, required);
	LazyExtensionSink sink(plan, false);
	build_state.SetPipelineSource(pipeline, source);
	build_state.AddPipelineOperator(pipeline, op);
	build_state.SetPipelineSink(pipeline, sink, 0);
	pipeline.Ready();
	pipeline.Reset();
	PipelineExecutor pipeline_executor(*con.context, pipeline);
	REQUIRE(pipeline_executor.Execute() == PipelineExecuteResult::FINISHED);
	REQUIRE(op.batch_calls == 1);
	REQUIRE(sink.data_calls == 0);
	REQUIRE(sink.accepted_rows == 3);
}

TEST_CASE("Extension sink retries reuse the bound materialized payload", "[execution_batch][pipeline][extension]") {
	DuckDB db(nullptr);
	Connection con(db);
	BatchBackendGuard backend_guard;
	auto backend = make_shared_ptr<CountingBatchBackend>();
	SetExternalBlockBackend(ExternalBlockBackend::RAY_OBJECT_STORE, backend);
	Executor executor(*con.context);
	Pipeline pipeline(executor);
	PipelineBuildState build_state;
	PhysicalPlan physical_plan(Allocator::DefaultAllocator());
	LazyBatchSource source(physical_plan, nullptr, backend);
	RetryBlockingSink sink(physical_plan, PhysicalOperatorType::EXTENSION);

	build_state.SetPipelineSource(pipeline, source);
	build_state.SetPipelineSink(pipeline, sink, 0);
	pipeline.Ready();
	pipeline.Reset();
	PipelineExecutor pipeline_executor(*con.context, pipeline);
	for (idx_t retry = 0; retry < 2; retry++) {
		REQUIRE(pipeline_executor.Execute() == PipelineExecuteResult::INTERRUPTED);
		REQUIRE(backend->materialize_calls == 1);
		REQUIRE(sink.accepted_calls == 0);
	}
	REQUIRE(pipeline_executor.Execute() == PipelineExecuteResult::FINISHED);
	REQUIRE(backend->materialize_calls == 1);
	REQUIRE(sink.batch_calls == 3);
	REQUIRE(sink.sink_calls == 3);
	REQUIRE(sink.accepted_calls == 1);
	REQUIRE(sink.accepted_values == vector<int64_t> {11, 22, 33});
}

TEST_CASE("Extension materialization and sink failures release descriptor ownership once",
          "[execution_batch][pipeline][extension]") {
	bool fail_materialize = false;
	SECTION("reader throws before replacing the input") {
		fail_materialize = true;
	}
	SECTION("sink throws after materialization") {
	}
	DuckDB db(nullptr);
	Connection con(db);
	BatchBackendGuard backend_guard;
	auto backend = make_shared_ptr<CountingBatchBackend>();
	backend->fail_materialize = fail_materialize;
	SetExternalBlockBackend(ExternalBlockBackend::RAY_OBJECT_STORE, backend);
	idx_t released = 0;
	{
		auto token = shared_ptr<void>(new int(1), [&released](void *ptr) {
			delete static_cast<int *>(ptr);
			released++;
		});
		Executor executor(*con.context);
		Pipeline pipeline(executor);
		PipelineBuildState build_state;
		PhysicalPlan physical_plan(Allocator::DefaultAllocator());
		LazyBatchSource source(physical_plan, std::move(token), backend);
		MaterializedCountingSink sink(physical_plan, false, PhysicalOperatorType::EXTENSION);
		sink.fail_consume = !fail_materialize;
		build_state.SetPipelineSource(pipeline, source);
		build_state.SetPipelineSink(pipeline, sink, 0);
		pipeline.Ready();
		pipeline.Reset();
		PipelineExecutor pipeline_executor(*con.context, pipeline);
		REQUIRE_THROWS_AS(pipeline_executor.Execute(), InvalidInputException);
		REQUIRE(backend->materialize_calls == 1);
		REQUIRE(sink.batch_calls == 1);
		REQUIRE(sink.sink_calls == (fail_materialize ? 0 : 1));
		REQUIRE(sink.values.empty());
		REQUIRE(released == (fail_materialize ? 0 : 1));
	}
	REQUIRE(released == 1);
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

TEST_CASE("Late pipeline tasks initialize sink batches before final output", "[execution_batch][pipeline]") {
	bool requires_batch = false;
	SECTION("DataChunk") {
	}
	SECTION("ExecutionBatch") {
		requires_batch = true;
	}
	for (bool block_initial : {false, true}) {
		DuckDB db(nullptr);
		Connection con(db);
		Executor executor(*con.context);
		Pipeline pipeline(executor);
		PipelineBuildState build_state;
		PhysicalPlan physical_plan(Allocator::DefaultAllocator());
		ExhaustedPartitionedSource source(physical_plan);
		FinalOutputOperator op(physical_plan);
		InitialBatchSink sink(physical_plan, requires_batch, block_initial);

		build_state.SetPipelineSource(pipeline, source);
		build_state.AddPipelineOperator(pipeline, op);
		build_state.SetPipelineSink(pipeline, sink, 3);
		pipeline.Ready();
		REQUIRE(pipeline.GetExecutionMode() ==
		        (requires_batch ? PipelineExecutionMode::EXECUTION_BATCH : PipelineExecutionMode::DATA_CHUNK));
		pipeline.Reset();
		// An earlier task has exhausted the source before this executor is constructed.
		auto initial_batch = pipeline.RegisterNewBatchIndex();
		auto terminal_batch = initial_batch + PipelineBuildState::BATCH_INCREMENT - 1;
		pipeline.UpdateBatchIndex(initial_batch, terminal_batch);
		PipelineExecutor pipeline_executor(*con.context, pipeline);
		if (block_initial) {
			REQUIRE(pipeline_executor.Execute() == PipelineExecuteResult::INTERRUPTED);
			REQUIRE(sink.values.empty());
		}
		REQUIRE(pipeline_executor.Execute() == PipelineExecuteResult::FINISHED);
		REQUIRE(sink.current_batch.GetIndex() == terminal_batch);
		REQUIRE(sink.next_batch_calls == (block_initial ? 2 : 1));
		REQUIRE(sink.values == vector<int64_t> {42});
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

TEST_CASE("Lazy ExecutionBatch materializes once while an intermediate has more output",
          "[execution_batch][operator]") {
	DuckDB db(nullptr);
	Connection con(db);
	ThreadContext thread(*con.context);
	ExecutionContext context(*con.context, thread, nullptr);
	PhysicalPlan plan(Allocator::DefaultAllocator());
	MultiOutputOperator op(plan);
	auto backend = make_shared_ptr<CountingBatchBackend>();
	auto batch = MakeLazyExecutionBatch(backend);
	ExecutionBatch output;
	GlobalOperatorState global_state;
	OperatorState local_state;
	const DataChunk *payload = nullptr;
	for (idx_t call = 0; call < 3; call++) {
		auto result = op.ExecuteBatch(context, batch, output, global_state, local_state);
		REQUIRE(result == (call < 2 ? OperatorResultType::HAVE_MORE_OUTPUT : OperatorResultType::NEED_MORE_INPUT));
		if (!payload) {
			payload = batch.materialized.get();
		}
		RequireRetryablePayload(batch, payload);
		REQUIRE(backend->materialize_calls == 1);
	}
}

TEST_CASE("Hash local exchange retains the materialized batch across budget blocking",
          "[execution_batch][local_exchange]") {
	DBConfig config;
	config.options.local_exchange_buffer_bytes = 1;
	DuckDB db(nullptr, &config);
	Connection con(db);
	ThreadContext thread(*con.context);
	ExecutionContext context(*con.context, thread, nullptr);
	PhysicalPlan plan(Allocator::DefaultAllocator());
	auto spec = std::make_shared<HashRepartitionSpec>(HashRepartitionConfig::create(1, {}));
	PhysicalLocalExchange exchange(plan, {LogicalType::BIGINT}, spec, 6);
	exchange.partition_by.push_back(make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, 0));
	auto global_sink = exchange.GetGlobalSinkState(*con.context);
	auto local_sink = exchange.GetLocalSinkState(context);
	auto global_source = exchange.GetGlobalSourceState(*con.context);
	auto local_source = exchange.GetLocalSourceState(context, *global_source);
	InterruptState interrupt {weak_ptr<Task>()};
	OperatorSinkInput sink_input {*global_sink, *local_sink, interrupt};
	OperatorSourceInput source_input {*global_source, *local_source, interrupt};
	auto backend = make_shared_ptr<CountingBatchBackend>();
	auto first = MakeLazyExecutionBatch(backend);
	REQUIRE(exchange.SinkBatch(context, first, sink_input) == SinkResultType::NEED_MORE_INPUT);
	auto waiting = MakeLazyExecutionBatch(backend);
	REQUIRE(exchange.SinkBatch(context, waiting, sink_input) == SinkResultType::BLOCKED);
	REQUIRE(backend->materialize_calls == 2);
	REQUIRE(waiting.kind == ExecutionBatchKind::MATERIALIZED_CHUNK);
	auto payload = waiting.materialized.get();
	REQUIRE(exchange.SinkBatch(context, waiting, sink_input) == SinkResultType::BLOCKED);
	REQUIRE(backend->materialize_calls == 2);
	RequireRetryablePayload(waiting, payload);
	ExecutionBatch output;
	REQUIRE(exchange.GetDataBatch(context, output, source_input) == SourceResultType::HAVE_MORE_OUTPUT);
	REQUIRE(exchange.SinkBatch(context, waiting, sink_input) == SinkResultType::NEED_MORE_INPUT);
	REQUIRE(backend->materialize_calls == 2);
	REQUIRE(exchange.GetDataBatch(context, output, source_input) == SourceResultType::HAVE_MORE_OUTPUT);
	REQUIRE(output.rows == 3);
}

#ifdef VANE_BATCH_DEMO_PATH
namespace {

unique_ptr<UDFExecutor> RejectHostUDFExecution(ClientContext &, const Value &, UDFConfig &, shared_ptr<void>) {
	throw InvalidInputException("host UDF executor factory reached");
}

struct HostUDFFactoryGuard {
	HostUDFFactoryGuard() : previous(GetUDFExecutorFactory()) {
		SetUDFExecutorFactory(RejectHostUDFExecution);
	}
	~HostUDFFactoryGuard() {
		SetUDFExecutorFactory(previous);
	}
	udf_executor_factory_t previous;
};

} // namespace

TEST_CASE("Private COPY planners retain the host UDF executor", "[execution_batch][extension][udf]") {
	DBConfig config;
#ifndef VANE_BATCH_DEMO_SIGNED
	config.options.allow_unsigned_extensions = true;
#endif
	DuckDB db(nullptr, &config);
	Connection con(db);
	auto loaded = con.Query(string("LOAD '") + VANE_BATCH_DEMO_PATH + "'");
	INFO((loaded->HasError() ? loaded->GetError() : ""));
	REQUIRE_FALSE(loaded->HasError());
	REQUIRE_FALSE(con.Query("SET threads=1")->HasError());
	HostUDFFactoryGuard guard;
	void *library = dlopen(VANE_BATCH_DEMO_PATH, RTLD_NOW | RTLD_LOCAL);
	REQUIRE(library);
	using HasFactory = bool (*)();
	auto has_factory = reinterpret_cast<HasFactory>(dlsym(library, "batch_demo_has_private_udf_factory"));
	REQUIRE(has_factory);
	const auto private_factory_registered = has_factory();
	dlclose(library);
	REQUIRE_FALSE(private_factory_registered);
	const string input = R"(
        SELECT udf(i, {'payload_version': 1, 'udf_name': 'private-copy', 'call_mode': 'map',
                       'execution_backend': 'subprocess_task', 'method_return_type': 'BIGINT',
                       'function_pickle': 'fixture'::BLOB, 'function_pickle_size_bytes': 7,
                       'udf_task_input_max_bytes': 134217728,
                       'udf_output_target_max_bytes': 134217728}) FROM range(4) t(i)
    )";
	SECTION("ordinary SQL still executes") {
		auto result = con.Query("COPY (SELECT 42::BIGINT) TO 'unused' (FORMAT batch_demo_replan)");
		INFO((result->HasError() ? result->GetError() : ""));
		REQUIRE_FALSE(result->HasError());
		REQUIRE(result->GetValue(0, 0) == Value::BIGINT(42));
	}
	SECTION("direct UDF reaches the host factory") {
		auto result = con.Query(input);
		REQUIRE(result->HasError());
		REQUIRE_THAT(result->GetError(), Catch::Matchers::Contains("host UDF executor factory reached"));
	}
	SECTION("UDF replanned inside the private extension reaches the same factory") {
		auto result = con.Query("COPY (" + input + ") TO 'unused' (FORMAT batch_demo_replan)");
		REQUIRE(result->HasError());
		REQUIRE_THAT(result->GetError(), Catch::Matchers::Contains("host UDF executor factory reached"));
	}
	REQUIRE_FALSE(con.Query("SELECT 1")->HasError());
}

TEST_CASE("Private dynamic adapters use the bound host reader", "[execution_batch][extension][external_block]") {
	DBConfig config;
#ifndef VANE_BATCH_DEMO_SIGNED
	config.options.allow_unsigned_extensions = true;
#endif
	DuckDB db(nullptr, &config);
	Connection con(db);
	auto loaded = con.Query(string("LOAD '") + VANE_BATCH_DEMO_PATH + "'");
	INFO((loaded->HasError() ? loaded->GetError() : ""));
	REQUIRE_FALSE(loaded->HasError());
	struct Library {
		explicit Library(void *handle_p) : handle(handle_p) {
		}
		~Library() {
			if (handle) {
				dlclose(handle);
			}
		}
		void *handle;
	} library(dlopen(VANE_BATCH_DEMO_PATH, RTLD_NOW | RTLD_LOCAL));
	REQUIRE(library.handle);
	using HasReader = bool (*)();
	using MakeOperator = PhysicalOperator *(*)(PhysicalPlan &, idx_t, vector<int64_t> &);
	auto has_reader = reinterpret_cast<HasReader>(dlsym(library.handle, "batch_demo_has_private_reader"));
	auto make_operator = reinterpret_cast<MakeOperator>(dlsym(library.handle, "batch_demo_make_operator"));
	REQUIRE(has_reader);
	REQUIRE(make_operator);
	BatchBackendGuard guard;
	auto backend = make_shared_ptr<CountingBatchBackend>();
	SetExternalBlockBackend(ExternalBlockBackend::RAY_OBJECT_STORE, backend);
	REQUIRE_FALSE(has_reader());
	ThreadContext thread(*con.context);
	ExecutionContext context(*con.context, thread, nullptr);
	PhysicalPlan plan(Allocator::DefaultAllocator());
	idx_t object_releases = 0;
	auto batch = MakeLazyExecutionBatch(backend);
	vector<int64_t> values;
	batch.lazy->blocks[0].object_ref = shared_ptr<void>(new int(1), [&object_releases](void *ptr) {
		delete static_cast<int *>(ptr);
		object_releases++;
	});
	idx_t kind = 0;
	SECTION("ordinary sink blocks twice") {
	}
	SECTION("ordinary intermediate has multiple outputs") {
		kind = 1;
	}
	SECTION("extension creates an internal COPY operator") {
		kind = 2;
	}
	unique_ptr<PhysicalOperator> op(make_operator(plan, kind, values));
	REQUIRE(op);
	if (kind == 1) {
		ExecutionBatch output;
		GlobalOperatorState global_state;
		OperatorState state;
		for (idx_t call = 0; call < 3; call++) {
			auto result = op->ExecuteBatch(context, batch, output, global_state, state);
			REQUIRE(result == (call < 2 ? OperatorResultType::HAVE_MORE_OUTPUT : OperatorResultType::NEED_MORE_INPUT));
			REQUIRE(backend->materialize_calls == 1);
		}
		REQUIRE(values == vector<int64_t> {11, 22, 33, 11, 22, 33, 11, 22, 33});
	} else {
		auto global_state = op->GetGlobalSinkState(*con.context);
		auto local_state = op->GetLocalSinkState(context);
		InterruptState interrupt;
		OperatorSinkInput input {*global_state, *local_state, interrupt};
		const DataChunk *payload = nullptr;
		for (idx_t call = 0; call < (kind == 0 ? 3 : 1); call++) {
			auto result = op->SinkBatch(context, batch, input);
			REQUIRE(result == (kind == 0 && call < 2 ? SinkResultType::BLOCKED : SinkResultType::NEED_MORE_INPUT));
			if (!payload) {
				payload = batch.materialized.get();
			}
			RequireRetryablePayload(batch, payload);
			REQUIRE(backend->materialize_calls == 1);
		}
		if (kind == 2) {
			REQUIRE(op->type == PhysicalOperatorType::COPY_TO_FILE);
			op->Cast<PhysicalCopyToFile>().FinalizeInternal(*con.context, *global_state);
		}
		REQUIRE(values == vector<int64_t> {11, 22, 33});
	}
	REQUIRE_FALSE(has_reader());
	REQUIRE(object_releases == 1);
}
#endif

TEST_CASE("Native streaming preserves rows through backpressure", "[execution_batch][streaming]") {
	SECTION("single-threaded") {
		VerifyStreamingBackpressure(1);
	}
	SECTION("multi-threaded") {
		VerifyStreamingBackpressure(4);
	}
}
