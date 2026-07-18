// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/parallel/pipeline_executor.hpp"

#include "duckdb/execution/external_block.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/main/client_context.hpp"

#ifdef DUCKDB_DEBUG_ASYNC_SINK_SOURCE
#include <chrono>
#include <thread>
#endif

namespace duckdb {

namespace {

idx_t ExecutionBatchSize(const ExecutionBatch &batch) {
	if (batch.kind == ExecutionBatchKind::MATERIALIZED_CHUNK) {
		return batch.materialized ? batch.materialized->size() : batch.rows;
	}
	if (batch.kind == ExecutionBatchKind::LAZY_DATA_CHUNK) {
		return batch.lazy ? batch.lazy->cardinality : batch.rows;
	}
	return batch.rows;
}

idx_t ExecutionBatchBytes(const ExecutionBatch &batch) {
	if (batch.estimated_bytes > 0) {
		return batch.estimated_bytes;
	}
	if (batch.kind == ExecutionBatchKind::MATERIALIZED_CHUNK && batch.materialized) {
		return batch.materialized->GetAllocationSize();
	}
	if (batch.kind == ExecutionBatchKind::LAZY_DATA_CHUNK && batch.lazy) {
		return batch.lazy->EstimatedBytes();
	}
	return 0;
}

optional_ptr<DataChunk> ExecutionBatchMaterializedChunk(ExecutionBatch &batch) {
	if (batch.kind == ExecutionBatchKind::MATERIALIZED_CHUNK && batch.materialized) {
		return *batch.materialized;
	}
	return nullptr;
}

unique_ptr<DataChunk> MakeEmptyPipelineBatchChunk(ClientContext &context, const vector<LogicalType> &types) {
	auto chunk = make_uniq<DataChunk>();
	chunk->InitializeEmpty(types);
	chunk->SetCardinality(0);
	return chunk;
}

void StorePipelineBatchMaterialized(ExecutionBatch &batch, unique_ptr<DataChunk> chunk) {
	batch = ExecutionBatch();
	batch.kind = ExecutionBatchKind::MATERIALIZED_CHUNK;
	if (chunk) {
		batch.rows = chunk->size();
		batch.estimated_bytes = chunk->GetAllocationSize();
	}
	batch.materialized = std::move(chunk);
}

DataChunk &MaterializePipelineBatch(ClientContext &context, ExecutionBatch &batch, const vector<LogicalType> &types,
                                    const char *reason) {
	if (batch.kind == ExecutionBatchKind::MATERIALIZED_CHUNK) {
		if (!batch.materialized) {
			StorePipelineBatchMaterialized(batch, MakeEmptyPipelineBatchChunk(context, types));
		}
		return *batch.materialized;
	}
	if (batch.kind == ExecutionBatchKind::LAZY_DATA_CHUNK) {
		if (!batch.lazy) {
			StorePipelineBatchMaterialized(batch, MakeEmptyPipelineBatchChunk(context, types));
			return *batch.materialized;
		}
		auto barrier = MaterializeExternalBlockBarrier(context, *batch.lazy, reason ? string(reason) : string());
		StorePipelineBatchMaterialized(batch, std::move(barrier.chunk));
		return *batch.materialized;
	}
	throw InternalException("unsupported ExecutionBatch kind in pipeline executor");
}

void ReferencePipelineBatch(ExecutionBatch &target, ExecutionBatch &source) {
	target = ExecutionBatch();
	target.kind = source.kind;
	target.rows = ExecutionBatchSize(source);
	target.estimated_bytes = source.estimated_bytes;
	if (source.kind == ExecutionBatchKind::MATERIALIZED_CHUNK) {
		if (source.materialized) {
			target.materialized = make_uniq<DataChunk>();
			target.materialized->InitializeEmpty(source.materialized->GetTypes());
			target.materialized->Reference(*source.materialized);
		}
		return;
	}
	if (source.kind == ExecutionBatchKind::LAZY_DATA_CHUNK) {
		if (source.lazy) {
			target.lazy = make_uniq<LazyDataChunk>(*source.lazy);
		}
		return;
	}
	throw InternalException("unsupported ExecutionBatch kind in pipeline executor");
}

} // namespace

PipelineExecutor::PipelineExecutor(ClientContext &context_p, Pipeline &pipeline_p)
    : pipeline(pipeline_p), thread(context_p), context(context_p, thread, &pipeline_p), use_execution_batches(true) {
	D_ASSERT(pipeline.source_state);
	if (pipeline.sink) {
		local_sink_state = pipeline.sink->GetLocalSinkState(context);
		required_partition_info = pipeline.sink->RequiredPartitionInfo();
		if (required_partition_info.AnyRequired()) {
			D_ASSERT(pipeline.source->SupportsPartitioning(OperatorPartitionInfo::BatchIndex()));
			auto &partition_info = local_sink_state->partition_info;
			D_ASSERT(!partition_info.batch_index.IsValid());
			// batch index is not set yet - initialize before fetching anything
			partition_info.batch_index = pipeline.RegisterNewBatchIndex();
			partition_info.min_batch_index = partition_info.batch_index;
		}
	}
	local_source_state = pipeline.source->GetLocalSourceState(context, *pipeline.source_state);

	intermediate_chunks.reserve(pipeline.operators.size());
	intermediate_batches.reserve(pipeline.operators.size());
	intermediate_states.reserve(pipeline.operators.size());
	for (idx_t i = 0; i < pipeline.operators.size(); i++) {
		auto &current_operator = pipeline.operators[i].get();

		if (!use_execution_batches) {
			auto &prev_operator = i == 0 ? *pipeline.source : pipeline.operators[i - 1].get();
			auto chunk = make_uniq<DataChunk>();
			chunk->Initialize(BufferAllocator::Get(context.client), prev_operator.GetTypes());
			intermediate_chunks.push_back(std::move(chunk));
		}
		intermediate_batches.push_back(make_uniq<ExecutionBatch>());

		auto op_state = current_operator.GetOperatorState(context);
		intermediate_states.push_back(std::move(op_state));

		if (current_operator.IsSink() && current_operator.sink_state->state == SinkFinalizeType::NO_OUTPUT_POSSIBLE) {
			// one of the operators has already figured out no output is possible
			// we can skip executing the pipeline
			FinishProcessing();
		}
	}
	if (!use_execution_batches) {
		InitializeChunk(final_chunk);
	}
}

bool PipelineExecutor::TryFlushCachingOperators(ExecutionBudget &chunk_budget) {
	if (!started_flushing) {
		// Remainder of this method assumes any in process operators are from flushing
		D_ASSERT(in_process_operators.empty());
		started_flushing = true;
		flushing_idx = IsFinished() ? idx_t(finished_processing_idx) : 0;
	}

	// For each operator that supports FinalExecute,
	// extract every chunk from it and push it through the rest of the pipeline
	// before moving onto the next operators' FinalExecute
	while (flushing_idx < pipeline.operators.size()) {
		if (!pipeline.operators[flushing_idx].get().RequiresFinalExecute()) {
			flushing_idx++;
			continue;
		}

		// This slightly awkward way of increasing the flushing idx is to make the code re-entrant: We need to call this
		// method again in the case of a Sink returning BLOCKED.
		if (!should_flush_current_idx && in_process_operators.empty()) {
			should_flush_current_idx = true;
		}

		auto &curr_chunk =
		    flushing_idx + 1 >= intermediate_chunks.size() ? final_chunk : *intermediate_chunks[flushing_idx + 1];
		auto &current_operator = pipeline.operators[flushing_idx].get();

		OperatorFinalizeResultType finalize_result;

		if (in_process_operators.empty()) {
			curr_chunk.Reset();
			if (&curr_chunk == &final_chunk) {
				sink_input_counted = false;
			}
			StartOperator(current_operator);
			finalize_result = current_operator.FinalExecute(context, curr_chunk, *current_operator.op_state,
			                                                *intermediate_states[flushing_idx]);
			EndOperator(current_operator, &curr_chunk, current_operator.op_state.get(),
			            intermediate_states[flushing_idx].get());
			if (finalize_result == OperatorFinalizeResultType::BLOCKED) {
				D_ASSERT(curr_chunk.size() == 0);
				should_flush_current_idx = true;
				blocked_on_finalize = true;
				return false;
			}
		} else {
			// Reset flag and reflush the last chunk we were flushing.
			finalize_result = OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
		}

		auto push_result = ExecutePushInternal(curr_chunk, chunk_budget, flushing_idx + 1);

		if (finalize_result == OperatorFinalizeResultType::HAVE_MORE_OUTPUT) {
			should_flush_current_idx = true;
		} else {
			should_flush_current_idx = false;
		}

		switch (push_result) {
		case OperatorResultType::BLOCKED: {
			if (!blocked_on_operator) {
				remaining_sink_chunk = true;
			}
			return false;
		}
		case OperatorResultType::HAVE_MORE_OUTPUT: {
			D_ASSERT(chunk_budget.IsDepleted());
			// The chunk budget was used up, pushing the chunk through the pipeline created more chunks
			// we need to continue this the next time Execute is called.
			return false;
		}
		case OperatorResultType::NEED_MORE_INPUT:
			if (!should_flush_current_idx) {
				// FinalExecute returned FINISHED for this operator — advance past it
				flushing_idx++;
			}
			continue;
		case OperatorResultType::FINISHED:
			break;
		default:
			throw InternalException("Unexpected OperatorResultType (%s) in TryFlushCachingOperators",
			                        EnumUtil::ToString(push_result));
		}
		break;
	}
	return true;
}

bool PipelineExecutor::TryFlushCachingOperatorsBatch(ExecutionBudget &chunk_budget) {
	if (!started_flushing) {
		D_ASSERT(in_process_operators.empty());
		started_flushing = true;
		flushing_idx = IsFinished() ? idx_t(finished_processing_idx) : 0;
	}

	while (flushing_idx < pipeline.operators.size()) {
		if (!pipeline.operators[flushing_idx].get().RequiresFinalExecute()) {
			flushing_idx++;
			continue;
		}

		if (!should_flush_current_idx && in_process_operators.empty()) {
			should_flush_current_idx = true;
		}

		auto &curr_batch =
		    flushing_idx + 1 >= intermediate_batches.size() ? final_batch : *intermediate_batches[flushing_idx + 1];
		auto &current_operator = pipeline.operators[flushing_idx].get();

		OperatorFinalizeResultType finalize_result;

		if (in_process_operators.empty()) {
			curr_batch = ExecutionBatch();
			if (&curr_batch == &final_batch) {
				sink_input_counted = false;
			}
			StartOperator(current_operator);
			finalize_result = current_operator.FinalExecuteBatch(context, curr_batch, *current_operator.op_state,
			                                                     *intermediate_states[flushing_idx]);
			EndOperator(current_operator, ExecutionBatchMaterializedChunk(curr_batch), current_operator.op_state.get(),
			            intermediate_states[flushing_idx].get());
			if (finalize_result == OperatorFinalizeResultType::BLOCKED) {
				D_ASSERT(ExecutionBatchSize(curr_batch) == 0);
				should_flush_current_idx = true;
				blocked_on_finalize = true;
				return false;
			}
		} else {
			finalize_result = OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
		}

		auto push_result = ExecutePushInternalBatch(curr_batch, chunk_budget, flushing_idx + 1);

		if (finalize_result == OperatorFinalizeResultType::HAVE_MORE_OUTPUT) {
			should_flush_current_idx = true;
		} else {
			should_flush_current_idx = false;
		}

		switch (push_result) {
		case OperatorResultType::BLOCKED: {
			if (!blocked_on_operator) {
				remaining_sink_chunk = true;
			}
			return false;
		}
		case OperatorResultType::HAVE_MORE_OUTPUT: {
			D_ASSERT(chunk_budget.IsDepleted());
			return false;
		}
		case OperatorResultType::NEED_MORE_INPUT:
			if (!should_flush_current_idx) {
				flushing_idx++;
			}
			continue;
		case OperatorResultType::FINISHED:
			break;
		default:
			throw InternalException("Unexpected OperatorResultType (%s) in TryFlushCachingOperatorsBatch",
			                        EnumUtil::ToString(push_result));
		}
		break;
	}
	return true;
}

SinkNextBatchType PipelineExecutor::NextBatch(DataChunk &source_chunk, const bool have_more_output) {
	D_ASSERT(required_partition_info.AnyRequired());
	auto max_batch_index = pipeline.base_batch_index + PipelineBuildState::BATCH_INCREMENT - 1;
	// by default set it to the maximum valid batch index value for the current pipeline
	auto &partition_info = local_sink_state->partition_info;
	OperatorPartitionData next_data(max_batch_index);
	if ((source_chunk.size() > 0)) {
		D_ASSERT(local_source_state);
		D_ASSERT(pipeline.source_state);
		// if we retrieved data - initialize the next batch index
		auto partition_data = pipeline.source->GetPartitionData(context, source_chunk, *pipeline.source_state,
		                                                        *local_source_state, required_partition_info);
		auto batch_index = partition_data.batch_index;
		// we start with the base_batch_index as a valid starting value. Make sure that next batch is called below
		next_data = std::move(partition_data);
		next_data.batch_index = pipeline.base_batch_index + batch_index + 1;
		if (next_data.batch_index >= max_batch_index) {
			throw InternalException("Pipeline batch index - invalid batch index %llu returned by source operator",
			                        batch_index);
		}
	} else if (have_more_output) {
		next_data.batch_index = partition_info.batch_index.GetIndex();
	}
	if (next_data.batch_index == partition_info.batch_index.GetIndex()) {
		// no changes, return
		return SinkNextBatchType::READY;
	}
	// batch index has changed - update it
	if (partition_info.batch_index.GetIndex() > next_data.batch_index) {
		throw InternalException(
		    "Pipeline batch index - gotten lower batch index %llu (down from previous batch index of %llu)",
		    next_data.batch_index, partition_info.batch_index.GetIndex());
	}
#ifdef DUCKDB_DEBUG_ASYNC_SINK_SOURCE
	if (debug_blocked_next_batch_count < debug_blocked_target_count) {
		debug_blocked_next_batch_count++;

		auto &callback_state = interrupt_state;
		std::thread rewake_thread([callback_state] {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			callback_state.Callback();
		});
		rewake_thread.detach();

		return SinkNextBatchType::BLOCKED;
	}
#endif
	auto current_batch = partition_info.batch_index.GetIndex();
	partition_info.batch_index = next_data.batch_index;
	partition_info.partition_data = std::move(next_data.partition_data);
	OperatorSinkNextBatchInput next_batch_input {*pipeline.sink->sink_state, *local_sink_state, interrupt_state};
	// call NextBatch before updating min_batch_index to provide the opportunity to flush the previous batch
	auto next_batch_result = pipeline.sink->NextBatch(context, next_batch_input);

	if (next_batch_result == SinkNextBatchType::BLOCKED) {
		partition_info.batch_index = current_batch; // set batch_index back to what it was before
		return SinkNextBatchType::BLOCKED;
	}

	partition_info.min_batch_index = pipeline.UpdateBatchIndex(current_batch, next_data.batch_index);

	return SinkNextBatchType::READY;
}

SinkNextBatchType PipelineExecutor::NextBatch(ExecutionBatch &source_batch, const bool have_more_output) {
	D_ASSERT(required_partition_info.AnyRequired());
	auto max_batch_index = pipeline.base_batch_index + PipelineBuildState::BATCH_INCREMENT - 1;
	auto &partition_info = local_sink_state->partition_info;
	OperatorPartitionData next_data(max_batch_index);
	if (ExecutionBatchSize(source_batch) > 0) {
		D_ASSERT(local_source_state);
		D_ASSERT(pipeline.source_state);
		auto &source_chunk =
		    MaterializePipelineBatch(context.client, source_batch, pipeline.source->GetTypes(), "next_batch");
		auto partition_data = pipeline.source->GetPartitionData(context, source_chunk, *pipeline.source_state,
		                                                        *local_source_state, required_partition_info);
		auto batch_index = partition_data.batch_index;
		next_data = std::move(partition_data);
		next_data.batch_index = pipeline.base_batch_index + batch_index + 1;
		if (next_data.batch_index >= max_batch_index) {
			throw InternalException("Pipeline batch index - invalid batch index %llu returned by source operator",
			                        batch_index);
		}
	} else if (have_more_output) {
		next_data.batch_index = partition_info.batch_index.GetIndex();
	}
	if (next_data.batch_index == partition_info.batch_index.GetIndex()) {
		return SinkNextBatchType::READY;
	}
	if (partition_info.batch_index.GetIndex() > next_data.batch_index) {
		throw InternalException(
		    "Pipeline batch index - gotten lower batch index %llu (down from previous batch index of %llu)",
		    next_data.batch_index, partition_info.batch_index.GetIndex());
	}
	auto current_batch = partition_info.batch_index.GetIndex();
	partition_info.batch_index = next_data.batch_index;
	partition_info.partition_data = std::move(next_data.partition_data);
	OperatorSinkNextBatchInput next_batch_input {*pipeline.sink->sink_state, *local_sink_state, interrupt_state};
	auto next_batch_result = pipeline.sink->NextBatch(context, next_batch_input);

	if (next_batch_result == SinkNextBatchType::BLOCKED) {
		partition_info.batch_index = current_batch;
		return SinkNextBatchType::BLOCKED;
	}

	partition_info.min_batch_index = pipeline.UpdateBatchIndex(current_batch, next_data.batch_index);
	return SinkNextBatchType::READY;
}

PipelineExecuteResult PipelineExecutor::Execute(idx_t max_chunks) {
	if (use_execution_batches) {
		return ExecuteBatches(max_chunks);
	}
	D_ASSERT(pipeline.sink);
	auto &source_chunk = pipeline.operators.empty() ? final_chunk : *intermediate_chunks[0];
	ExecutionBudget chunk_budget(max_chunks);
	do {
		if (context.client.interrupted) {
			throw InterruptException();
		}

		OperatorResultType result;
		if (blocked_on_operator) {
			auto &resume_chunk = blocked_operator_idx == 0 ? source_chunk : *intermediate_chunks[blocked_operator_idx];
			blocked_on_operator = false;
			result = ExecutePushInternal(resume_chunk, chunk_budget, blocked_operator_idx, true);
		} else if (exhausted_pipeline && done_flushing && !remaining_sink_chunk && !next_batch_blocked &&
		           in_process_operators.empty()) {
			break;
		} else if (remaining_sink_chunk) {
			// The pipeline was interrupted by the Sink. We should retry sinking the final chunk.
			result = ExecutePushInternal(final_chunk, chunk_budget);
			D_ASSERT(result != OperatorResultType::HAVE_MORE_OUTPUT);
			remaining_sink_chunk = false;
		} else if (!in_process_operators.empty() && !started_flushing) {
			// Operator(s) in the pipeline have returned `HAVE_MORE_OUTPUT` in the last Execute call
			// the operators have to be called with the same input chunk to produce the rest of the output
			D_ASSERT(source_chunk.size() > 0);
			result = ExecutePushInternal(source_chunk, chunk_budget);
		} else if (exhausted_pipeline && !next_batch_blocked && !done_flushing) {
			// The pipeline was exhausted, try flushing all operators
			blocked_on_finalize = false;
			auto flush_completed = TryFlushCachingOperators(chunk_budget);
			if (flush_completed) {
				done_flushing = true;
				break;
			} else {
				if (remaining_sink_chunk || blocked_on_operator || blocked_on_finalize) {
					return PipelineExecuteResult::INTERRUPTED;
				} else {
					D_ASSERT(chunk_budget.IsDepleted());
					return PipelineExecuteResult::NOT_FINISHED;
				}
			}
		} else if (!exhausted_pipeline || next_batch_blocked) {
			SourceResultType source_result = SourceResultType::BLOCKED;
			if (!next_batch_blocked) {
				// "Regular" path: fetch a chunk from the source and push it through the pipeline
				source_chunk.Reset();
				source_result = FetchFromSource(source_chunk);
				if (source_result == SourceResultType::BLOCKED) {
					return PipelineExecuteResult::INTERRUPTED;
				}
				if (source_result == SourceResultType::FINISHED) {
					exhausted_pipeline = true;
				}
			}

			if (required_partition_info.AnyRequired()) {
				auto next_batch_result = NextBatch(source_chunk, source_result == SourceResultType::HAVE_MORE_OUTPUT);
				next_batch_blocked = next_batch_result == SinkNextBatchType::BLOCKED;
				if (next_batch_blocked) {
					return PipelineExecuteResult::INTERRUPTED;
				}
			}

			if (exhausted_pipeline && source_chunk.size() == 0) {
				continue;
			}

			result = ExecutePushInternal(source_chunk, chunk_budget);
		} else {
			throw InternalException("Unexpected state reached in pipeline executor");
		}

		// SINK INTERRUPT
		if (result == OperatorResultType::BLOCKED) {
			if (!blocked_on_operator) {
				remaining_sink_chunk = true;
			}
			return PipelineExecuteResult::INTERRUPTED;
		}

		if (result == OperatorResultType::FINISHED) {
			D_ASSERT(in_process_operators.empty());
			exhausted_pipeline = true;
		}
	} while (chunk_budget.Next());

	if ((!exhausted_pipeline || !done_flushing) && !IsFinished()) {
		return PipelineExecuteResult::NOT_FINISHED;
	}

	// When an intermediate operator (e.g. STREAMING_LIMIT) returned FINISHED,
	// IsFinished() is true but exhausted_source may still be false. We must
	// still flush any operators with RequiresFinalExecute (e.g. async UDFs)
	// before finalizing the sink, otherwise their pending results are lost.
	if (!done_flushing) {
		blocked_on_finalize = false;
		auto flush_completed = TryFlushCachingOperators(chunk_budget);
		if (flush_completed) {
			done_flushing = true;
		} else {
			if (remaining_sink_chunk || blocked_on_operator || blocked_on_finalize) {
				return PipelineExecuteResult::INTERRUPTED;
			}
			return PipelineExecuteResult::NOT_FINISHED;
		}
	}

	return PushFinalize();
}

PipelineExecuteResult PipelineExecutor::ExecuteBatches(idx_t max_chunks) {
	D_ASSERT(pipeline.sink);
	auto &source_batch = pipeline.operators.empty() ? final_batch : *intermediate_batches[0];
	ExecutionBudget chunk_budget(max_chunks);
	do {
		if (context.client.interrupted) {
			throw InterruptException();
		}

		OperatorResultType result;
		if (blocked_on_operator) {
			auto &resume_batch = blocked_operator_idx == 0 ? source_batch : *intermediate_batches[blocked_operator_idx];
			blocked_on_operator = false;
			result = ExecutePushInternalBatch(resume_batch, chunk_budget, blocked_operator_idx, true);
		} else if (exhausted_pipeline && done_flushing && !remaining_sink_chunk && !next_batch_blocked &&
		           in_process_operators.empty()) {
			break;
		} else if (remaining_sink_chunk) {
			result = ExecutePushInternalBatch(final_batch, chunk_budget);
			D_ASSERT(result != OperatorResultType::HAVE_MORE_OUTPUT);
			remaining_sink_chunk = false;
		} else if (!in_process_operators.empty() && !started_flushing) {
			D_ASSERT(ExecutionBatchSize(source_batch) > 0);
			result = ExecutePushInternalBatch(source_batch, chunk_budget);
		} else if (exhausted_pipeline && !next_batch_blocked && !done_flushing) {
			blocked_on_finalize = false;
			auto flush_completed = TryFlushCachingOperatorsBatch(chunk_budget);
			if (flush_completed) {
				done_flushing = true;
				break;
			} else {
				if (remaining_sink_chunk || blocked_on_operator || blocked_on_finalize) {
					return PipelineExecuteResult::INTERRUPTED;
				}
				D_ASSERT(chunk_budget.IsDepleted());
				return PipelineExecuteResult::NOT_FINISHED;
			}
		} else if (!exhausted_pipeline || next_batch_blocked) {
			SourceResultType source_result = SourceResultType::BLOCKED;
			if (!next_batch_blocked) {
				source_batch = ExecutionBatch();
				source_result = FetchFromSourceBatch(source_batch);
				if (source_result == SourceResultType::BLOCKED) {
					return PipelineExecuteResult::INTERRUPTED;
				}
				if (source_result == SourceResultType::FINISHED) {
					exhausted_pipeline = true;
				}
			}

			if (required_partition_info.AnyRequired()) {
				auto next_batch_result = NextBatch(source_batch, source_result == SourceResultType::HAVE_MORE_OUTPUT);
				next_batch_blocked = next_batch_result == SinkNextBatchType::BLOCKED;
				if (next_batch_blocked) {
					return PipelineExecuteResult::INTERRUPTED;
				}
			}

			if (exhausted_pipeline && ExecutionBatchSize(source_batch) == 0) {
				continue;
			}

			result = ExecutePushInternalBatch(source_batch, chunk_budget);
		} else {
			throw InternalException("Unexpected state reached in pipeline executor");
		}

		if (result == OperatorResultType::BLOCKED) {
			if (!blocked_on_operator) {
				remaining_sink_chunk = true;
			}
			return PipelineExecuteResult::INTERRUPTED;
		}

		if (result == OperatorResultType::FINISHED) {
			D_ASSERT(in_process_operators.empty());
			exhausted_pipeline = true;
		}
	} while (chunk_budget.Next());

	if ((!exhausted_pipeline || !done_flushing) && !IsFinished()) {
		return PipelineExecuteResult::NOT_FINISHED;
	}

	if (!done_flushing) {
		blocked_on_finalize = false;
		auto flush_completed = TryFlushCachingOperatorsBatch(chunk_budget);
		if (flush_completed) {
			done_flushing = true;
		} else {
			if (remaining_sink_chunk || blocked_on_operator || blocked_on_finalize) {
				return PipelineExecuteResult::INTERRUPTED;
			}
			return PipelineExecuteResult::NOT_FINISHED;
		}
	}

	return PushFinalize();
}

bool PipelineExecutor::RemainingSinkChunk() const {
	return remaining_sink_chunk;
}

PipelineExecuteResult PipelineExecutor::Execute() {
	return Execute(NumericLimits<idx_t>::Maximum());
}

void PipelineExecutor::FinishProcessing(int32_t operator_idx) {
	finished_processing_idx = operator_idx < 0 ? NumericLimits<int32_t>::Maximum() : operator_idx;
	in_process_operators = stack<idx_t>();
	blocked_on_operator = false;
	blocked_on_finalize = false;

	if (pipeline.GetSource()) {
		auto guard = pipeline.source_state->Lock();
		pipeline.source_state->PreventBlocking(guard);
		pipeline.source_state->UnblockTasks(guard);
	}
	if (pipeline.GetSink()) {
		auto guard = pipeline.GetSink()->sink_state->Lock();
		pipeline.GetSink()->sink_state->PreventBlocking(guard);
		pipeline.GetSink()->sink_state->UnblockTasks(guard);
	}
}

bool PipelineExecutor::IsFinished() {
	return finished_processing_idx >= 0;
}

OperatorResultType PipelineExecutor::ExecutePushInternal(DataChunk &input, ExecutionBudget &chunk_budget,
                                                         idx_t initial_idx, bool ignore_in_process) {
	D_ASSERT(pipeline.sink);
	if (input.size() == 0) { // LCOV_EXCL_START
		return OperatorResultType::NEED_MORE_INPUT;
	} // LCOV_EXCL_STOP

	blocked_on_operator = false;
	// this loop will continuously push the input chunk through the pipeline as long as:
	// - the OperatorResultType for the Execute is HAVE_MORE_OUTPUT
	// - the Sink doesn't block
	// - the ExecutionBudget has not been depleted
	OperatorResultType result = OperatorResultType::HAVE_MORE_OUTPUT;
	do {
		// Note: if input is the final_chunk, we don't do any executing, the chunk just needs to be sinked
		if (&input != &final_chunk) {
			final_chunk.Reset();
			sink_input_counted = false;
			// Execute and put the result into 'final_chunk'
			result = Execute(input, final_chunk, initial_idx, ignore_in_process);
			if (result == OperatorResultType::BLOCKED) {
				return OperatorResultType::BLOCKED;
			} else if (result == OperatorResultType::FINISHED) {
				return OperatorResultType::FINISHED;
			}
		} else {
			result = OperatorResultType::NEED_MORE_INPUT;
		}
		auto &sink_chunk = final_chunk;
		if (sink_chunk.size() > 0) {
			if (!sink_input_counted) {
				pipeline.output_rows.fetch_add(sink_chunk.size());
				pipeline.output_bytes.fetch_add(sink_chunk.GetAllocationSize());
				sink_input_counted = true;
			}
			StartOperator(*pipeline.sink);
			D_ASSERT(pipeline.sink);
			D_ASSERT(pipeline.sink->sink_state);
			OperatorSinkInput sink_input {*pipeline.sink->sink_state, *local_sink_state, interrupt_state};

			auto sink_result = Sink(sink_chunk, sink_input);

			EndOperator(*pipeline.sink, nullptr);

			if (sink_result == SinkResultType::BLOCKED) {
				return OperatorResultType::BLOCKED;
			} else if (sink_result == SinkResultType::FINISHED) {
				FinishProcessing();
				return OperatorResultType::FINISHED;
			}
		}
		if (result == OperatorResultType::NEED_MORE_INPUT) {
			return OperatorResultType::NEED_MORE_INPUT;
		}
	} while (chunk_budget.Next());
	return result;
}

OperatorResultType PipelineExecutor::ExecutePushInternalBatch(ExecutionBatch &input, ExecutionBudget &chunk_budget,
                                                              idx_t initial_idx, bool ignore_in_process) {
	D_ASSERT(pipeline.sink);
	if (ExecutionBatchSize(input) == 0) { // LCOV_EXCL_START
		return OperatorResultType::NEED_MORE_INPUT;
	} // LCOV_EXCL_STOP

	blocked_on_operator = false;
	OperatorResultType result = OperatorResultType::HAVE_MORE_OUTPUT;
	do {
		if (&input != &final_batch) {
			final_batch = ExecutionBatch();
			sink_input_counted = false;
			result = ExecuteBatch(input, final_batch, initial_idx, ignore_in_process);
			if (result == OperatorResultType::BLOCKED) {
				return OperatorResultType::BLOCKED;
			} else if (result == OperatorResultType::FINISHED) {
				return OperatorResultType::FINISHED;
			}
		} else {
			result = OperatorResultType::NEED_MORE_INPUT;
		}
		auto &sink_batch = final_batch;
		if (ExecutionBatchSize(sink_batch) > 0) {
			if (!sink_input_counted) {
				pipeline.output_rows.fetch_add(ExecutionBatchSize(sink_batch));
				pipeline.output_bytes.fetch_add(ExecutionBatchBytes(sink_batch));
				sink_input_counted = true;
			}
			StartOperator(*pipeline.sink);
			D_ASSERT(pipeline.sink);
			D_ASSERT(pipeline.sink->sink_state);
			OperatorSinkInput sink_input {*pipeline.sink->sink_state, *local_sink_state, interrupt_state};

			auto sink_result = SinkBatch(sink_batch, sink_input);

			EndOperator(*pipeline.sink, nullptr);

			if (sink_result == SinkResultType::BLOCKED) {
				return OperatorResultType::BLOCKED;
			} else if (sink_result == SinkResultType::FINISHED) {
				FinishProcessing();
				return OperatorResultType::FINISHED;
			}
		}
		if (result == OperatorResultType::NEED_MORE_INPUT) {
			return OperatorResultType::NEED_MORE_INPUT;
		}
	} while (chunk_budget.Next());
	return result;
}

PipelineExecuteResult PipelineExecutor::PushFinalize() {
	if (finalized) {
		throw InternalException("Calling PushFinalize on a pipeline that has been finalized already");
	}

	D_ASSERT(local_sink_state);

	// Run the combine for the sink
	OperatorSinkCombineInput combine_input {*pipeline.sink->sink_state, *local_sink_state, interrupt_state};

#ifdef DUCKDB_DEBUG_ASYNC_SINK_SOURCE
	if (debug_blocked_combine_count < debug_blocked_target_count) {
		debug_blocked_combine_count++;

		auto &callback_state = combine_input.interrupt_state;
		std::thread rewake_thread([callback_state] {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			callback_state.Callback();
		});
		rewake_thread.detach();

		return PipelineExecuteResult::INTERRUPTED;
	}
#endif
	auto result = pipeline.sink->Combine(context, combine_input);

	if (result == SinkCombineResultType::BLOCKED) {
		return PipelineExecuteResult::INTERRUPTED;
	}

	finalized = true;
	// flush all query profiler info
	for (idx_t i = 0; i < intermediate_states.size(); i++) {
		intermediate_states[i]->Finalize(pipeline.operators[i].get(), context);
	}
	pipeline.executor.Flush(thread);
	local_sink_state.reset();

	return PipelineExecuteResult::FINISHED;
}

void PipelineExecutor::GoToSource(idx_t &current_idx, idx_t initial_idx, bool ignore_in_process) {
	// we go back to the first operator (the source)
	current_idx = initial_idx;
	if (!ignore_in_process && !in_process_operators.empty()) {
		// ... UNLESS there is an in process operator
		// if there is an in-process operator, we start executing at the latest one
		// for example, if we have a join operator that has tuples left, we first need to emit those tuples
		current_idx = in_process_operators.top();
		in_process_operators.pop();
	}
	D_ASSERT(current_idx >= initial_idx);
}

OperatorResultType PipelineExecutor::Execute(DataChunk &input, DataChunk &result, idx_t initial_idx,
                                             bool ignore_in_process) {
	if (input.size() == 0) { // LCOV_EXCL_START
		return OperatorResultType::NEED_MORE_INPUT;
	} // LCOV_EXCL_STOP
	D_ASSERT(!pipeline.operators.empty());

	idx_t current_idx;
	GoToSource(current_idx, initial_idx, ignore_in_process);
	if (current_idx == initial_idx) {
		current_idx++;
	}
	if (current_idx > pipeline.operators.size()) {
		result.Reference(input);
		return OperatorResultType::NEED_MORE_INPUT;
	}
	while (true) {
		if (context.client.interrupted) {
			throw InterruptException();
		}
		// now figure out where to put the chunk
		// if current_idx is the last possible index (>= operators.size()) we write to the result
		// otherwise we write to an intermediate chunk
		auto current_intermediate = current_idx;
		auto &current_chunk =
		    current_intermediate >= intermediate_chunks.size() ? result : *intermediate_chunks[current_intermediate];
		current_chunk.Reset();
		if (current_idx == initial_idx) {
			// we went back to the source: we need more input
			return OperatorResultType::NEED_MORE_INPUT;
		} else {
			auto &prev_chunk =
			    current_intermediate == initial_idx + 1 ? input : *intermediate_chunks[current_intermediate - 1];
			auto operator_idx = current_idx - 1;
			auto &current_operator = pipeline.operators[operator_idx].get();

			// if current_idx > source_idx, we pass the previous operators' output through the Execute of the current
			// operator
			StartOperator(current_operator);
			auto result = current_operator.Execute(context, prev_chunk, current_chunk, *current_operator.op_state,
			                                       *intermediate_states[current_intermediate - 1]);
			EndOperator(current_operator, &current_chunk, current_operator.op_state.get(),
			            intermediate_states[current_intermediate - 1].get());
			if (result == OperatorResultType::BLOCKED) {
				D_ASSERT(current_chunk.size() == 0);
				blocked_on_operator = true;
				blocked_operator_idx = operator_idx;
				return OperatorResultType::BLOCKED;
			} else if (result == OperatorResultType::HAVE_MORE_OUTPUT) {
				// more data remains in this operator
				// push in-process marker
				in_process_operators.push(current_idx);
			} else if (result == OperatorResultType::FINISHED) {
				D_ASSERT(current_chunk.size() == 0);
				FinishProcessing(NumericCast<int32_t>(current_idx));
				return OperatorResultType::FINISHED;
			}
			current_chunk.Verify();
		}

		if (current_chunk.size() == 0) {
			// no output from this operator!
			if (current_idx == initial_idx) {
				// if we got no output from the scan, we are done
				break;
			} else {
				// if we got no output from an intermediate op
				// we go back and try to pull data from the source again
				GoToSource(current_idx, initial_idx, false);
				continue;
			}
		} else {
			// we got output! continue to the next operator
			current_idx++;
			if (current_idx > pipeline.operators.size()) {
				// if we got output and are at the last operator, we are finished executing for this output chunk
				// return the data and push it into the chunk
				break;
			}
		}
	}
	return in_process_operators.empty() ? OperatorResultType::NEED_MORE_INPUT : OperatorResultType::HAVE_MORE_OUTPUT;
}

OperatorResultType PipelineExecutor::ExecuteBatch(ExecutionBatch &input, ExecutionBatch &result, idx_t initial_idx,
                                                  bool ignore_in_process) {
	if (ExecutionBatchSize(input) == 0) { // LCOV_EXCL_START
		return OperatorResultType::NEED_MORE_INPUT;
	} // LCOV_EXCL_STOP
	D_ASSERT(!pipeline.operators.empty());

	idx_t current_idx;
	GoToSource(current_idx, initial_idx, ignore_in_process);
	if (current_idx == initial_idx) {
		current_idx++;
	}
	if (current_idx > pipeline.operators.size()) {
		ReferencePipelineBatch(result, input);
		return OperatorResultType::NEED_MORE_INPUT;
	}
	while (true) {
		if (context.client.interrupted) {
			throw InterruptException();
		}
		auto current_intermediate = current_idx;
		auto &current_batch =
		    current_intermediate >= intermediate_batches.size() ? result : *intermediate_batches[current_intermediate];
		current_batch = ExecutionBatch();
		if (current_idx == initial_idx) {
			return OperatorResultType::NEED_MORE_INPUT;
		} else {
			auto &prev_batch =
			    current_intermediate == initial_idx + 1 ? input : *intermediate_batches[current_intermediate - 1];
			auto operator_idx = current_idx - 1;
			auto &current_operator = pipeline.operators[operator_idx].get();

			StartOperator(current_operator);
			auto result = current_operator.ExecuteBatch(context, prev_batch, current_batch, *current_operator.op_state,
			                                            *intermediate_states[current_intermediate - 1]);
			EndOperator(current_operator, ExecutionBatchMaterializedChunk(current_batch),
			            current_operator.op_state.get(), intermediate_states[current_intermediate - 1].get());
			if (result == OperatorResultType::BLOCKED) {
				D_ASSERT(ExecutionBatchSize(current_batch) == 0);
				blocked_on_operator = true;
				blocked_operator_idx = operator_idx;
				return OperatorResultType::BLOCKED;
			} else if (result == OperatorResultType::HAVE_MORE_OUTPUT) {
				in_process_operators.push(current_idx);
			} else if (result == OperatorResultType::FINISHED) {
				D_ASSERT(ExecutionBatchSize(current_batch) == 0);
				FinishProcessing(NumericCast<int32_t>(current_idx));
				return OperatorResultType::FINISHED;
			}
			if (auto chunk = ExecutionBatchMaterializedChunk(current_batch)) {
				chunk->Verify();
			}
		}

		if (ExecutionBatchSize(current_batch) == 0) {
			if (current_idx == initial_idx) {
				break;
			} else {
				GoToSource(current_idx, initial_idx, false);
				continue;
			}
		} else {
			current_idx++;
			if (current_idx > pipeline.operators.size()) {
				break;
			}
		}
	}
	return in_process_operators.empty() ? OperatorResultType::NEED_MORE_INPUT : OperatorResultType::HAVE_MORE_OUTPUT;
}

void PipelineExecutor::SetTaskForInterrupts(weak_ptr<Task> current_task) {
	interrupt_state = InterruptState(std::move(current_task));
	context.interrupt_state = &interrupt_state;
}

SourceResultType PipelineExecutor::GetData(DataChunk &chunk, OperatorSourceInput &input) {
	//! Testing feature to enable async source on every operator
#ifdef DUCKDB_DEBUG_ASYNC_SINK_SOURCE
	if (debug_blocked_source_count < debug_blocked_target_count) {
		debug_blocked_source_count++;

		auto &callback_state = input.interrupt_state;
		std::thread rewake_thread([callback_state] {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			callback_state.Callback();
		});
		rewake_thread.detach();

		return SourceResultType::BLOCKED;
	}
#endif

	return pipeline.source->GetData(context, chunk, input);
}

SourceResultType PipelineExecutor::GetDataBatch(ExecutionBatch &batch, OperatorSourceInput &input) {
	//! Testing feature to enable async source on every operator
#ifdef DUCKDB_DEBUG_ASYNC_SINK_SOURCE
	if (debug_blocked_source_count < debug_blocked_target_count) {
		debug_blocked_source_count++;

		auto &callback_state = input.interrupt_state;
		std::thread rewake_thread([callback_state] {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			callback_state.Callback();
		});
		rewake_thread.detach();

		return SourceResultType::BLOCKED;
	}
#endif

	return pipeline.source->GetDataBatch(context, batch, input);
}

SinkResultType PipelineExecutor::Sink(DataChunk &chunk, OperatorSinkInput &input) {
	//! Testing feature to enable async sink on every operator
#ifdef DUCKDB_DEBUG_ASYNC_SINK_SOURCE
	if (debug_blocked_sink_count < debug_blocked_target_count) {
		debug_blocked_sink_count++;

		auto &callback_state = input.interrupt_state;
		std::thread rewake_thread([callback_state] {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			callback_state.Callback();
		});
		rewake_thread.detach();

		return SinkResultType::BLOCKED;
	}
#endif
	return pipeline.sink->Sink(context, chunk, input);
}

SinkResultType PipelineExecutor::SinkBatch(ExecutionBatch &batch, OperatorSinkInput &input) {
	//! Testing feature to enable async sink on every operator
#ifdef DUCKDB_DEBUG_ASYNC_SINK_SOURCE
	if (debug_blocked_sink_count < debug_blocked_target_count) {
		debug_blocked_sink_count++;

		auto &callback_state = input.interrupt_state;
		std::thread rewake_thread([callback_state] {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			callback_state.Callback();
		});
		rewake_thread.detach();

		return SinkResultType::BLOCKED;
	}
#endif
	return pipeline.sink->SinkBatch(context, batch, input);
}

SourceResultType PipelineExecutor::FetchFromSource(DataChunk &result) {
	StartOperator(*pipeline.source);

	OperatorSourceInput source_input = {*pipeline.source_state, *local_source_state, interrupt_state};
	auto res = GetData(result, source_input);
	if (result.size() > 0) {
		pipeline.input_rows.fetch_add(result.size());
		pipeline.input_bytes.fetch_add(result.GetAllocationSize());
		if (pipeline.operators.empty()) {
			sink_input_counted = false;
		}
	}

	// Ensures sources only return empty results when Blocking or Finished
	D_ASSERT(res != SourceResultType::BLOCKED || result.size() == 0);
	if (res == SourceResultType::FINISHED) {
		// final call into the source - finish source execution
		context.thread.profiler.FinishSource(*pipeline.source_state, *local_source_state);
	}
	EndOperator(*pipeline.source, &result);

	return res;
}

SourceResultType PipelineExecutor::FetchFromSourceBatch(ExecutionBatch &result) {
	StartOperator(*pipeline.source);

	OperatorSourceInput source_input = {*pipeline.source_state, *local_source_state, interrupt_state};
	auto res = GetDataBatch(result, source_input);
	const auto rows = ExecutionBatchSize(result);
	if (rows > 0) {
		pipeline.input_rows.fetch_add(rows);
		pipeline.input_bytes.fetch_add(ExecutionBatchBytes(result));
		if (pipeline.operators.empty()) {
			sink_input_counted = false;
		}
	}

	D_ASSERT(res != SourceResultType::BLOCKED || ExecutionBatchSize(result) == 0);
	if (res == SourceResultType::FINISHED) {
		context.thread.profiler.FinishSource(*pipeline.source_state, *local_source_state);
	}
	EndOperator(*pipeline.source, ExecutionBatchMaterializedChunk(result));

	return res;
}

void PipelineExecutor::InitializeChunk(DataChunk &chunk) {
	auto &last_op = pipeline.operators.empty() ? *pipeline.source : pipeline.operators.back().get();
	chunk.Initialize(BufferAllocator::Get(context.client), last_op.GetTypes());
}

void PipelineExecutor::StartOperator(PhysicalOperator &op) {
	if (context.client.interrupted) {
		throw InterruptException();
	}
	context.thread.profiler.StartOperator(&op);
}

void PipelineExecutor::EndOperator(PhysicalOperator &op, optional_ptr<DataChunk> chunk,
                                   optional_ptr<GlobalOperatorState> gstate, optional_ptr<OperatorState> state) {
	context.thread.profiler.EndOperator(chunk, gstate, state);

	if (chunk) {
		chunk->Verify();
	}
}

} // namespace duckdb
