// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/execution/physical_operator.hpp"

#include "duckdb/common/printer.hpp"
#include "duckdb/common/render_tree.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/tree_renderer.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/execution/operator/set/physical_cte.hpp"
#include "duckdb/execution/operator/set/physical_recursive_cte.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_perfecthash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_partitioned_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_ungrouped_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_window.hpp"
#include "duckdb/execution/operator/aggregate/physical_streaming_window.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/projection/physical_vllm.hpp"

#include "duckdb/execution/operator/projection/physical_pivot.hpp"
#include "duckdb/execution/operator/projection/physical_tableinout_function.hpp"
#include "duckdb/execution/operator/projection/physical_udf_inout.hpp"
#include "duckdb/execution/operator/projection/physical_unnest.hpp"
#include "duckdb/execution/operator/helper/physical_reservoir_sample.hpp"
#include "duckdb/execution/operator/helper/physical_streaming_sample.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/helper/physical_limit.hpp"
#include "duckdb/execution/operator/helper/physical_limit_percent.hpp"
#include "duckdb/execution/operator/helper/physical_streaming_limit.hpp"
#include "duckdb/execution/operator/exchange/physical_remote_exchange_sink.hpp"
#include "duckdb/execution/operator/exchange/physical_remote_exchange_source.hpp"
#include "duckdb/execution/distributed/exchange/flight_exchange_manager.hpp"
#include "duckdb/execution/operator/exchange/physical_local_exchange.hpp"
#include "duckdb/execution/operator/exchange/physical_repartition.hpp"
#include "duckdb/execution/operator/order/physical_order.hpp"
#include "duckdb/execution/operator/order/physical_top_n.hpp"
#include "duckdb/execution/operator/persistent/physical_batch_copy_to_file.hpp"
#include "duckdb/execution/operator/persistent/physical_copy_to_file.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/execution/operator/scan/physical_dummy_scan.hpp"
#include "duckdb/execution/operator/scan/physical_expression_scan.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/join/physical_nested_loop_join.hpp"
#include "duckdb/execution/operator/join/physical_left_delim_join.hpp"
#include "duckdb/execution/operator/join/physical_right_delim_join.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/function/function_serialization.hpp"
#include "duckdb/catalog/catalog_entry/copy_function_catalog_entry.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/function_serialization.hpp"
#include "duckdb/function/scalar/udf_functions.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/storage/buffer/buffer_pool.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serialization_data.hpp"
#include "duckdb/execution/dynamic_filter_serialization.hpp"

namespace duckdb {

namespace {

struct DynamicFilterSerializationGuard {
	explicit DynamicFilterSerializationGuard(SerializationData &data_p) : data(data_p) {
		if (!data.TryGetCustom<DynamicTableFilterSerializationState>()) {
			data.SetCustom(state);
			active = true;
		}
	}

	~DynamicFilterSerializationGuard() {
		if (active) {
			data.UnsetCustom<DynamicTableFilterSerializationState>();
		}
	}

	SerializationData &data;
	DynamicTableFilterSerializationState state;
	bool active = false;
};

struct CTESerializationState : public SerializationData::CustomData {
	static string GetType() {
		return "cte_state";
	}

	void Reset() {
		working_tables.clear();
		cte_ops.clear();
	}

	unordered_map<idx_t, shared_ptr<ColumnDataCollection>> working_tables;
	unordered_map<idx_t, PhysicalCTE *> cte_ops;
};

static CTESerializationState &GetOrCreateCTEState(SerializationData &data) {
	auto state_ptr = data.TryGetCustom<CTESerializationState>();
	if (state_ptr) {
		return *state_ptr;
	}
	static thread_local CTESerializationState state;
	state.Reset();
	data.SetCustom(state);
	return state;
}

static void GatherDelimScans(PhysicalOperator &op, vector<const_reference<PhysicalOperator>> &delim_scans,
                             optional_idx delim_idx) {
	if (op.type == PhysicalOperatorType::DELIM_SCAN) {
		auto &scan = op.Cast<PhysicalColumnDataScan>();
		if (!delim_idx.IsValid() || scan.delim_index == delim_idx) {
			delim_scans.push_back(op);
		}
	}
	for (auto &child : op.children) {
		GatherDelimScans(child, delim_scans, delim_idx);
	}
}

static unique_ptr<DataChunk> MakeEmptyExecutionBatchChunk(ClientContext &context, const vector<LogicalType> &types) {
	auto chunk = make_uniq<DataChunk>();
	chunk->Initialize(BufferAllocator::Get(context), types);
	chunk->SetCardinality(0);
	return chunk;
}

static unique_ptr<DataChunk> TakeMaterializedExecutionBatch(ClientContext &context, ExecutionBatch &batch,
                                                            const vector<LogicalType> &types, const char *reason) {
	if (batch.kind == ExecutionBatchKind::MATERIALIZED_CHUNK) {
		if (batch.materialized) {
			return std::move(batch.materialized);
		}
		return MakeEmptyExecutionBatchChunk(context, types);
	}
	if (batch.kind == ExecutionBatchKind::LAZY_DATA_CHUNK) {
		if (!batch.lazy) {
			return MakeEmptyExecutionBatchChunk(context, types);
		}
		auto barrier = MaterializeExternalBlockBarrier(context, *batch.lazy, reason ? string(reason) : string());
		auto chunk = std::move(barrier.chunk);
		// Once a lazy batch has been materialized, the returned DataChunk owns the
		// materialized Arrow buffers. Drop the original descriptors immediately so
		// their budget tickets do not artificially hold upstream ref-bundle
		// backpressure while the downstream operator processes the chunk.
		batch = ExecutionBatch();
		return chunk;
	}
	throw InternalException("unsupported ExecutionBatch kind");
}

static void StoreMaterializedExecutionBatch(ExecutionBatch &batch, unique_ptr<DataChunk> chunk) {
	batch = ExecutionBatch();
	batch.kind = ExecutionBatchKind::MATERIALIZED_CHUNK;
	if (chunk) {
		batch.rows = chunk->size();
		batch.estimated_bytes = chunk->GetAllocationSize();
	}
	batch.materialized = std::move(chunk);
}

} // namespace

PhysicalOperator::PhysicalOperator(PhysicalPlan &physical_plan, PhysicalOperatorType type, vector<LogicalType> types,
                                   idx_t estimated_cardinality)
    : children(physical_plan.ArenaRef()), type(type), types(std::move(types)),
      estimated_cardinality(estimated_cardinality) {
}

string PhysicalOperator::GetName() const {
	return PhysicalOperatorToString(type);
}

string PhysicalOperator::ToString(ExplainFormat format) const {
	auto renderer = TreeRenderer::CreateRenderer(format);
	stringstream ss;
	auto tree = RenderTree::CreateRenderTree(*this);
	renderer->ToStream(*tree, ss);
	return ss.str();
}

// LCOV_EXCL_START
void PhysicalOperator::Print() const {
	Printer::Print(ToString());
}
// LCOV_EXCL_STOP

vector<const_reference<PhysicalOperator>> PhysicalOperator::GetChildren() const {
	vector<const_reference<PhysicalOperator>> result;
	for (auto &child : children) {
		result.push_back(child.get());
	}
	return result;
}

void PhysicalOperator::SetEstimatedCardinality(InsertionOrderPreservingMap<string> &result,
                                               idx_t estimated_cardinality) {
	result[RenderTreeNode::ESTIMATED_CARDINALITY] = StringUtil::Format("%llu", estimated_cardinality);
}

idx_t PhysicalOperator::EstimatedThreadCount() const {
	idx_t result = 0;
	if (children.empty()) {
		// Terminal operator, e.g., base table, these decide the degree of parallelism of pipelines
		result = MaxValue<idx_t>(estimated_cardinality / (DEFAULT_ROW_GROUP_SIZE * 2), 1);
	} else if (type == PhysicalOperatorType::UNION) {
		// We can run union pipelines in parallel, so we sum up the thread count of the children
		for (auto &child : children) {
			result += child.get().EstimatedThreadCount();
		}
	} else {
		// For other operators we take the maximum of the children
		for (auto &child : children) {
			result = MaxValue(child.get().EstimatedThreadCount(), result);
		}
	}
	return result;
}

bool PhysicalOperator::CanSaturateThreads(ClientContext &context) const {
#ifdef DEBUG
	// In debug mode we always return true here so that the code that depends on it is well-tested
	return true;
#else
	const auto num_threads = NumericCast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads());
	return EstimatedThreadCount() >= num_threads;
#endif
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
// LCOV_EXCL_START
unique_ptr<OperatorState> PhysicalOperator::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<OperatorState>();
}

unique_ptr<GlobalOperatorState> PhysicalOperator::GetGlobalOperatorState(ClientContext &context) const {
	return make_uniq<GlobalOperatorState>();
}

OperatorResultType PhysicalOperator::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                             GlobalOperatorState &gstate, OperatorState &state) const {
	throw InternalException("Calling Execute on a node that is not an operator!");
}

OperatorResultType PhysicalOperator::ExecuteBatch(ExecutionContext &context, ExecutionBatch &input,
                                                  ExecutionBatch &output, GlobalOperatorState &gstate,
                                                  OperatorState &state) const {
	auto input_types = children.empty() ? vector<LogicalType>() : children[0].get().GetTypes();
	auto barrier_reason = GetName();
	auto input_chunk = TakeMaterializedExecutionBatch(context.client, input, input_types, barrier_reason.c_str());
	auto output_chunk = make_uniq<DataChunk>();
	output_chunk->Initialize(BufferAllocator::Get(context.client), types);
	auto result = Execute(context, *input_chunk, *output_chunk, gstate, state);
	StoreMaterializedExecutionBatch(output, std::move(output_chunk));
	return result;
}

OperatorFinalizeResultType PhysicalOperator::FinalExecute(ExecutionContext &context, DataChunk &chunk,
                                                          GlobalOperatorState &gstate, OperatorState &state) const {
	throw InternalException("Calling FinalExecute on a node that is not an operator!");
}

OperatorFinalizeResultType PhysicalOperator::FinalExecuteBatch(ExecutionContext &context, ExecutionBatch &batch,
                                                               GlobalOperatorState &gstate,
                                                               OperatorState &state) const {
	auto chunk = make_uniq<DataChunk>();
	chunk->Initialize(BufferAllocator::Get(context.client), types);
	auto result = FinalExecute(context, *chunk, gstate, state);
	StoreMaterializedExecutionBatch(batch, std::move(chunk));
	return result;
}

OperatorFinalResultType PhysicalOperator::OperatorFinalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                           OperatorFinalizeInput &input) const {
	throw InternalException("Calling FinalExecute on a node that is not an operator!");
}
// LCOV_EXCL_STOP

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
unique_ptr<LocalSourceState> PhysicalOperator::GetLocalSourceState(ExecutionContext &context,
                                                                   GlobalSourceState &gstate) const {
	return make_uniq<LocalSourceState>();
}

unique_ptr<GlobalSourceState> PhysicalOperator::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<GlobalSourceState>();
}

// LCOV_EXCL_START
SourceResultType PhysicalOperator::GetData(ExecutionContext &context, DataChunk &chunk,
                                           OperatorSourceInput &input) const {
	return GetDataInternal(context, chunk, input);
}

SourceResultType PhysicalOperator::GetDataBatch(ExecutionContext &context, ExecutionBatch &batch,
                                                OperatorSourceInput &input) const {
	auto chunk = make_uniq<DataChunk>();
	chunk->Initialize(BufferAllocator::Get(context.client), types);
	auto result = GetData(context, *chunk, input);
	StoreMaterializedExecutionBatch(batch, std::move(chunk));
	return result;
}

SourceResultType PhysicalOperator::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                   OperatorSourceInput &input) const {
	throw InternalException("Calling GetDataInternal on a node that is not a source!");
}

OperatorPartitionData PhysicalOperator::GetPartitionData(ExecutionContext &context, DataChunk &chunk,
                                                         GlobalSourceState &gstate, LocalSourceState &lstate,
                                                         const OperatorPartitionInfo &partition_info) const {
	throw InternalException("Calling GetPartitionData on a node that does not support it");
}

ProgressData PhysicalOperator::GetProgress(ClientContext &context, GlobalSourceState &gstate) const {
	ProgressData res;
	res.SetInvalid();
	return res;
}
// LCOV_EXCL_STOP

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
// LCOV_EXCL_START
SinkResultType PhysicalOperator::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	throw InternalException("Calling Sink on a node that is not a sink!");
}

SinkResultType PhysicalOperator::SinkBatch(ExecutionContext &context, ExecutionBatch &batch,
                                           OperatorSinkInput &input) const {
	auto input_types = children.empty() ? vector<LogicalType>() : children[0].get().GetTypes();
	auto barrier_reason = GetName();
	auto chunk = TakeMaterializedExecutionBatch(context.client, batch, input_types, barrier_reason.c_str());
	return Sink(context, *chunk, input);
}

// LCOV_EXCL_STOP

SinkCombineResultType PhysicalOperator::Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const {
	return SinkCombineResultType::FINISHED;
}

void PhysicalOperator::PrepareFinalize(ClientContext &context, GlobalSinkState &sink_state) const {
}

SinkFinalizeType PhysicalOperator::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                            OperatorSinkFinalizeInput &input) const {
	return SinkFinalizeType::READY;
}

SinkNextBatchType PhysicalOperator::NextBatch(ExecutionContext &context, OperatorSinkNextBatchInput &input) const {
	return SinkNextBatchType::READY;
}

unique_ptr<LocalSinkState> PhysicalOperator::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<LocalSinkState>();
}

unique_ptr<GlobalSinkState> PhysicalOperator::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<GlobalSinkState>();
}

idx_t PhysicalOperator::GetMaxThreadMemory(ClientContext &context) {
	// Memory usage per thread should scale with max mem / num threads
	// We take 1/4th of this, to be conservative
	auto max_memory = BufferManager::GetBufferManager(context).GetQueryMaxMemory();
	auto num_threads = NumericCast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads());
	return (max_memory / num_threads) / 4;
}

OperatorCachingMode PhysicalOperator::SelectOperatorCachingMode(ExecutionContext &context) {
	if (!context.client.config.enable_caching_operators) {
		return OperatorCachingMode::NONE;
	} else if (!context.pipeline) {
		return OperatorCachingMode::NONE;
	} else if (!context.pipeline->GetSink()) {
		return OperatorCachingMode::NONE;
	} else {
		auto partition_info = context.pipeline->GetSink()->RequiredPartitionInfo();
		if (partition_info.AnyRequired()) {
			return OperatorCachingMode::PARTITIONED;
		}
	}
	if (context.pipeline->IsOrderDependent()) {
		return OperatorCachingMode::ORDERED;
	}

	return OperatorCachingMode::UNORDERED;
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalOperator::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	op_state.reset();

	auto &state = meta_pipeline.GetState();
	if (!IsSink() && children.empty()) {
		// Operator is a source.
		state.SetPipelineSource(current, *this);
		return;
	}

	if (children.size() != 1) {
		throw InternalException("Operator not supported in BuildPipelines");
	}

	if (IsSink()) {
		// Operator is a sink.
		sink_state.reset();

		// It becomes the data source of the current pipeline.
		state.SetPipelineSource(current, *this);

		// Create a new pipeline starting at the child.
		auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
		child_meta_pipeline.Build(children[0].get());
		return;
	}

	// Recurse into the child.
	state.AddPipelineOperator(current, *this);
	children[0].get().BuildPipelines(current, meta_pipeline);
}

vector<const_reference<PhysicalOperator>> PhysicalOperator::GetSources() const {
	vector<const_reference<PhysicalOperator>> result;
	if (!IsSink() && children.empty()) {
		// Operator is a source.
		result.push_back(*this);
		return result;
	}

	if (children.size() != 1) {
		throw InternalException("Operator not supported in GetSource");
	}

	if (IsSink()) {
		result.push_back(*this);
		return result;
	}

	// Recurse into the child.
	return children[0].get().GetSources();
}

bool PhysicalOperator::AllSourcesSupportBatchIndex() const {
	auto sources = GetSources();
	for (auto &source : sources) {
		if (!source.get().SupportsPartitioning(OperatorPartitionInfo::BatchIndex())) {
			return false;
		}
	}
	return true;
}

void PhysicalOperator::Verify() {
#ifdef DEBUG
	auto sources = GetSources();
	D_ASSERT(!sources.empty());
	for (auto &child : children) {
		child.get().Verify();
	}
#endif
}

bool CachingPhysicalOperator::CanCacheType(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::LIST:
	case LogicalTypeId::MAP:
	case LogicalTypeId::ARRAY:
		return false;
	case LogicalTypeId::STRUCT: {
		auto &entries = StructType::GetChildTypes(type);
		for (auto &entry : entries) {
			if (!CanCacheType(entry.second)) {
				return false;
			}
		}
		return true;
	}
	default:
		return true;
	}
}

CachingPhysicalOperator::CachingPhysicalOperator(PhysicalPlan &physical_plan, PhysicalOperatorType type,
                                                 vector<LogicalType> types_p, idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, type, std::move(types_p), estimated_cardinality) {
	caching_supported = true;
	for (auto &col_type : types) {
		if (!CanCacheType(col_type)) {
			caching_supported = false;
			break;
		}
	}
}

enum class CachingPhysicalOperatorExecuteMode : uint8_t {
	RETURN_CACHED_APPEND_CHUNK,
	RETURN_CACHED_PLUS_CHUNK,
	RETURN_CACHED_THEN_CHUNK_VIA_CONTINUATION,
	RETURN_CHUNK,
	APPEND_CHUNK,
	RETURN_CACHED
};

static CachingPhysicalOperatorExecuteMode SelectExecutionMode(const DataChunk &chunk,
                                                              const OperatorResultType child_result,
                                                              CachingOperatorState &state,
                                                              ClientContext &client_context) {
	if (state.can_cache_chunk == OperatorCachingMode::NONE) {
		return CachingPhysicalOperatorExecuteMode::RETURN_CHUNK;
	}
	const bool needs_continuation_chunk = (state.can_cache_chunk == OperatorCachingMode::PARTITIONED &&
	                                       child_result != OperatorResultType::HAVE_MORE_OUTPUT) ||
	                                      (child_result == OperatorResultType::FINISHED);
	const bool has_non_empty_cached_chunk = state.cached_chunk && state.cached_chunk->size() > 0;
	const bool has_space_for_chunk_in_cache =
	    !state.cached_chunk || (state.cached_chunk->size() + chunk.size() <= STANDARD_VECTOR_SIZE);

	if (has_non_empty_cached_chunk && needs_continuation_chunk) {
		if (chunk.size() == 0) {
			if (child_result == OperatorResultType::BLOCKED) {
				// First return cached, then empty chunk via continuation that will BLOCK
				return CachingPhysicalOperatorExecuteMode::RETURN_CACHED_THEN_CHUNK_VIA_CONTINUATION;
			}

			// Return cached, and the current result
			return CachingPhysicalOperatorExecuteMode::RETURN_CACHED;
		}
		if (chunk.size() <= CachingPhysicalOperator::CACHE_THRESHOLD && has_space_for_chunk_in_cache) {
			// chunk is small, both fit
			return CachingPhysicalOperatorExecuteMode::RETURN_CACHED_PLUS_CHUNK;
		}

		// First return cached, then chunk via continuation
		return CachingPhysicalOperatorExecuteMode::RETURN_CACHED_THEN_CHUNK_VIA_CONTINUATION;
	} else if (chunk.size() == 0) {
		// Nothing required to be done, this also means that BLOCKED is properly passed through
		// Note that this case works also for unordered cases, given no rows are there

		return CachingPhysicalOperatorExecuteMode::RETURN_CHUNK;
	} else if (chunk.size() <= CachingPhysicalOperator::CACHE_THRESHOLD && !needs_continuation_chunk) {
		// We have filtered out a significant amount of tuples

		if (!state.cached_chunk) {
			// Initialize cached_chunk
			state.cached_chunk = make_uniq<DataChunk>();
			state.cached_chunk->Initialize(Allocator::Get(client_context), chunk.GetTypes());
		}

		if (has_space_for_chunk_in_cache) {
			// We can just append, do and return empty chunk
			return CachingPhysicalOperatorExecuteMode::APPEND_CHUNK;
		}

		// Return what is now cached, and append chunk (via tmp)
		return CachingPhysicalOperatorExecuteMode::RETURN_CACHED_APPEND_CHUNK;
	} else if (state.can_cache_chunk == OperatorCachingMode::UNORDERED) {
		// Chunk is too big to considering caching, order is not required, just return it
		return CachingPhysicalOperatorExecuteMode::RETURN_CHUNK;
	} else if (has_non_empty_cached_chunk) {
		// We need first to return (*state.cached_chunk), then chunk on the continuation
		// NOTE: Both are not empty
		D_ASSERT(chunk.size() > 0);
		D_ASSERT(state.cached_chunk->size() > 0);

		if (chunk.size() <= CachingPhysicalOperator::CACHE_THRESHOLD) {
			// We can consider appening
			if (chunk.size() + state.cached_chunk->size() <= STANDARD_VECTOR_SIZE) {
				// Both fit toghether, append then return
				return CachingPhysicalOperatorExecuteMode::RETURN_CACHED_PLUS_CHUNK;
			}
			if (needs_continuation_chunk) {
				// Both needs to be returned in this step, but cached before current chunk
				return CachingPhysicalOperatorExecuteMode::RETURN_CACHED_THEN_CHUNK_VIA_CONTINUATION;
			}

			// Return now cached, and append chunk (via tmp)
			return CachingPhysicalOperatorExecuteMode::RETURN_CACHED_APPEND_CHUNK;
		}

		// Both needs to be returned in this step, but cached before current chunk
		return CachingPhysicalOperatorExecuteMode::RETURN_CACHED_THEN_CHUNK_VIA_CONTINUATION;
	}
	return CachingPhysicalOperatorExecuteMode::RETURN_CHUNK;
}

OperatorResultType CachingPhysicalOperator::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                    GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &state = state_p.Cast<CachingOperatorState>();

	if (state.initialized && state.must_return_continuation_chunk) {
		chunk.Move(*state.cached_chunk);
		state.cached_chunk->Initialize(Allocator::Get(context.client), chunk.GetTypes());
		if (state.cached_result == OperatorResultType::BLOCKED && chunk.size() > 0) {
			// In case of BLOCKED, first the chunk + HAVE_MORE_OUTPUT, then blocking
			// This should currently be forbidden, so the assertion, but HAVE_MORE_OUTPUT is also a valid solution
			D_ASSERT(false);
			return OperatorResultType::HAVE_MORE_OUTPUT;
		}
		state.must_return_continuation_chunk = false;
		return state.cached_result;
	}

	// Execute child operator
	auto child_result = ExecuteInternal(context, input, chunk, gstate, state);

	if (!state.initialized) {
		state.initialized = true;
		state.must_return_continuation_chunk = false;
		if (caching_supported) {
			state.can_cache_chunk = PhysicalOperator::SelectOperatorCachingMode(context);
		} else {
			state.can_cache_chunk = OperatorCachingMode::NONE;
		}
	}

	const auto execution_mode = SelectExecutionMode(chunk, child_result, state, context.client);

	switch (execution_mode) {
	case CachingPhysicalOperatorExecuteMode::RETURN_CACHED_APPEND_CHUNK: {
		auto tmp = make_uniq<DataChunk>();
		tmp->Move(chunk);
		chunk.Move(*state.cached_chunk);
		state.cached_chunk->Initialize(Allocator::Get(context.client), chunk.GetTypes());
		state.cached_chunk->Append(*tmp);
		break;
	}
	case CachingPhysicalOperatorExecuteMode::RETURN_CACHED_PLUS_CHUNK:
		state.cached_chunk->Append(chunk);
		chunk.Move(*state.cached_chunk);
		state.cached_chunk->Initialize(Allocator::Get(context.client), chunk.GetTypes());
		break;
	case CachingPhysicalOperatorExecuteMode::RETURN_CACHED:
		D_ASSERT(chunk.size() == 0);
		chunk.Move(*state.cached_chunk);
		state.cached_chunk->Initialize(Allocator::Get(context.client), chunk.GetTypes());
		break;
	case CachingPhysicalOperatorExecuteMode::RETURN_CACHED_THEN_CHUNK_VIA_CONTINUATION: {
		// Swap chunk and *state.cached_chunk
		auto tmp = make_uniq<DataChunk>();
		tmp->Move(chunk);
		chunk.Move(*state.cached_chunk);
		state.cached_chunk->Initialize(Allocator::Get(context.client), chunk.GetTypes());
		state.cached_chunk->Move(*tmp);

		// Now chunk holds what was in (*state.cached_chunk), and it's returned directly
		// While what was in chunk will be returned at next iteration via continuation
		state.must_return_continuation_chunk = true;
		state.cached_result = child_result;
		return OperatorResultType::HAVE_MORE_OUTPUT;
	}
	case CachingPhysicalOperatorExecuteMode::APPEND_CHUNK: {
		state.cached_chunk->Append(chunk);
		chunk.Reset();
		break;
	}
	case CachingPhysicalOperatorExecuteMode::RETURN_CHUNK:
		break;
	}

	return child_result;
}

OperatorFinalizeResultType CachingPhysicalOperator::FinalExecute(ExecutionContext &context, DataChunk &chunk,
                                                                 GlobalOperatorState &gstate,
                                                                 OperatorState &state_p) const {
	auto &state = state_p.Cast<CachingOperatorState>();
	if (state.cached_chunk) {
		chunk.Move(*state.cached_chunk);
		state.cached_chunk.reset();
	} else {
		chunk.SetCardinality(0);
	}
	return OperatorFinalizeResultType::FINISHED;
}

void PhysicalOperator::Serialize(Serializer &serializer) const {
	auto &data = serializer.GetSerializationData();
	DynamicFilterSerializationGuard guard(data);

	// Write common fields
	serializer.WriteProperty(100, "type", type);
	serializer.WriteProperty(101, "types", types);
	serializer.WriteProperty(102, "estimated_cardinality", estimated_cardinality);

	// Write operator-specific data (virtual call to derived class)
	SerializeOperatorData(serializer);

	// Write children as a list
	serializer.WriteList(198, "children", children.size(), [&](Serializer::List &list, idx_t i) {
		list.WriteObject([&](Serializer &child_serializer) { children[i].get().Serialize(child_serializer); });
	});
}

void PhysicalOperator::SerializeOperatorData(Serializer &serializer) const {
	// Default implementation: no operator-specific data
}

unique_ptr<PhysicalOperator> PhysicalOperator::Deserialize(Deserializer &deserializer, PhysicalPlan &physical_plan) {
	auto &data = deserializer.GetSerializationData();
	DynamicFilterSerializationGuard guard(data);

	// Read common fields
	auto op_type = deserializer.ReadProperty<PhysicalOperatorType>(100, "type");
	auto types = deserializer.ReadProperty<vector<LogicalType>>(101, "types");
	auto estimated_cardinality = deserializer.ReadProperty<idx_t>(102, "estimated_cardinality");

	// Dispatch to operator-specific deserialization
	auto result =
	    DeserializeOperatorData(deserializer, physical_plan, op_type, std::move(types), estimated_cardinality);

	// Read and deserialize children
	deserializer.ReadList(198, "children", [&](Deserializer::List &list, idx_t /*i*/) {
		list.ReadObject([&](Deserializer &child_deserializer) {
			auto child = Deserialize(child_deserializer, physical_plan);
			PhysicalOperator *child_ptr = child.get();
			physical_plan.TakeOwnership(std::move(child));
			result->children.push_back(*child_ptr);
		});
	});

	return result;
}

unique_ptr<PhysicalOperator> PhysicalOperator::DeserializeOperatorData(Deserializer &deserializer,
                                                                       PhysicalPlan &physical_plan,
                                                                       PhysicalOperatorType op_type,
                                                                       vector<LogicalType> types,
                                                                       idx_t estimated_cardinality) {
	switch (op_type) {
	case PhysicalOperatorType::PROJECTION: {
		// Read projection-specific field: select_list
		auto select_list = deserializer.ReadProperty<vector<unique_ptr<Expression>>>(103, "select_list");
		return make_uniq<PhysicalProjection>(physical_plan, std::move(types), std::move(select_list),
		                                     estimated_cardinality);
	}
	case PhysicalOperatorType::VLLM_PROJECT: {
		unique_ptr<Expression> prompt_expr;
		string model;
		Value options;
		string output_column_name;
		try {
			prompt_expr = deserializer.ReadProperty<unique_ptr<Expression>>(103, "prompt_expr");
		} catch (const std::exception &ex) {
			throw SerializationException("VLLM_PROJECT deserialize failed reading prompt_expr: %s", ex.what());
		}
		try {
			model = deserializer.ReadProperty<string>(104, "model");
		} catch (const std::exception &ex) {
			throw SerializationException("VLLM_PROJECT deserialize failed reading model: %s", ex.what());
		}
		try {
			options = deserializer.ReadProperty<Value>(105, "options");
		} catch (const std::exception &ex) {
			throw SerializationException("VLLM_PROJECT deserialize failed reading options: %s", ex.what());
		}
		try {
			output_column_name = deserializer.ReadProperty<string>(106, "output_column_name");
		} catch (const std::exception &ex) {
			throw SerializationException("VLLM_PROJECT deserialize failed reading output_column_name: %s", ex.what());
		}
		return make_uniq<PhysicalVLLM>(physical_plan, std::move(types), std::move(prompt_expr), std::move(model),
		                               std::move(options), std::move(output_column_name), estimated_cardinality);
	}
	case PhysicalOperatorType::UNNEST: {
		auto select_list = deserializer.ReadProperty<vector<unique_ptr<Expression>>>(103, "select_list");
		return make_uniq<PhysicalUnnest>(physical_plan, std::move(types), std::move(select_list),
		                                 estimated_cardinality);
	}
	case PhysicalOperatorType::RESERVOIR_SAMPLE: {
		auto options = deserializer.ReadProperty<unique_ptr<SampleOptions>>(103, "sample_options");
		return make_uniq<PhysicalReservoirSample>(physical_plan, std::move(types), std::move(options),
		                                          estimated_cardinality);
	}
	case PhysicalOperatorType::STREAMING_SAMPLE: {
		auto options = deserializer.ReadProperty<unique_ptr<SampleOptions>>(103, "sample_options");
		return make_uniq<PhysicalStreamingSample>(physical_plan, std::move(types), std::move(options),
		                                          estimated_cardinality);
	}
	case PhysicalOperatorType::FILTER: {
		// Read filter-specific field: expression
		auto expression = deserializer.ReadProperty<unique_ptr<Expression>>(103, "expression");

		// Create filter with empty select list (expression is set separately)
		vector<unique_ptr<Expression>> empty_select_list;
		auto filter = make_uniq<PhysicalFilter>(physical_plan, std::move(types), std::move(empty_select_list),
		                                        estimated_cardinality);
		filter->expression = std::move(expression);
		return unique_ptr<PhysicalOperator>(std::move(filter));
	}
	case PhysicalOperatorType::LIMIT: {
		auto limit_val = deserializer.ReadProperty<BoundLimitNode>(103, "limit_val");
		auto offset_val = deserializer.ReadProperty<BoundLimitNode>(104, "offset_val");
		return make_uniq<PhysicalLimit>(physical_plan, std::move(types), std::move(limit_val), std::move(offset_val),
		                                estimated_cardinality);
	}
	case PhysicalOperatorType::STREAMING_LIMIT: {
		auto limit_val = deserializer.ReadProperty<BoundLimitNode>(103, "limit_val");
		auto offset_val = deserializer.ReadProperty<BoundLimitNode>(104, "offset_val");
		auto parallel = deserializer.ReadProperty<bool>(105, "parallel");
		return make_uniq<PhysicalStreamingLimit>(physical_plan, std::move(types), std::move(limit_val),
		                                         std::move(offset_val), estimated_cardinality, parallel);
	}
	case PhysicalOperatorType::LIMIT_PERCENT: {
		auto limit_val = deserializer.ReadProperty<BoundLimitNode>(103, "limit_val");
		auto offset_val = deserializer.ReadProperty<BoundLimitNode>(104, "offset_val");
		return make_uniq<PhysicalLimitPercent>(physical_plan, std::move(types), std::move(limit_val),
		                                       std::move(offset_val), estimated_cardinality);
	}
	case PhysicalOperatorType::ORDER_BY: {
		auto orders = deserializer.ReadProperty<vector<BoundOrderByNode>>(103, "orders");
		auto projections = deserializer.ReadProperty<vector<idx_t>>(104, "projections");
		auto is_index_sort = deserializer.ReadProperty<bool>(105, "is_index_sort");
		return make_uniq<PhysicalOrder>(physical_plan, std::move(types), std::move(orders), std::move(projections),
		                                estimated_cardinality, is_index_sort);
	}
	case PhysicalOperatorType::TOP_N: {
		auto orders = deserializer.ReadProperty<vector<BoundOrderByNode>>(103, "orders");
		auto limit = deserializer.ReadProperty<idx_t>(104, "limit");
		auto offset = deserializer.ReadProperty<idx_t>(105, "offset");
		return make_uniq<PhysicalTopN>(physical_plan, std::move(types), std::move(orders), limit, offset, nullptr,
		                               estimated_cardinality);
	}
	case PhysicalOperatorType::DUMMY_SCAN: {
		return make_uniq<PhysicalDummyScan>(physical_plan, std::move(types), estimated_cardinality);
	}
	case PhysicalOperatorType::EXPRESSION_SCAN: {
		auto expressions = deserializer.ReadProperty<vector<vector<unique_ptr<Expression>>>>(103, "expressions");
		return make_uniq<PhysicalExpressionScan>(physical_plan, std::move(types), std::move(expressions),
		                                         estimated_cardinality);
	}
	case PhysicalOperatorType::INOUT_FUNCTION: {
		auto entry = FunctionSerializer::DeserializeBase<TableFunction, TableFunctionCatalogEntry>(
		    deserializer, CatalogType::TABLE_FUNCTION_ENTRY);
		auto function = std::move(entry.first);
		auto has_serialize = entry.second;
		unique_ptr<FunctionData> bind_data;
		if (has_serialize) {
			bind_data = FunctionSerializer::FunctionDeserialize(deserializer, function);
		} else {
			throw SerializationException(
			    "PhysicalTableInOutFunction deserialization requires function serialization for %s", function.name);
		}

		auto column_ids = deserializer.ReadProperty<vector<ColumnIndex>>(200, "column_ids");
		auto projected_input = deserializer.ReadProperty<vector<column_t>>(201, "projected_input");
		auto ordinality_idx = deserializer.ReadPropertyWithDefault<optional_idx>(202, "ordinality_idx");

		auto inout = make_uniq<PhysicalTableInOutFunction>(physical_plan, std::move(types), std::move(function),
		                                                   std::move(bind_data), std::move(column_ids),
		                                                   estimated_cardinality, std::move(projected_input));
		inout->ordinality_idx = ordinality_idx;
		return unique_ptr<PhysicalOperator>(std::move(inout));
	}
	case PhysicalOperatorType::STREAMING_UDF: {
		auto entry = FunctionSerializer::DeserializeBase<TableFunction, TableFunctionCatalogEntry>(
		    deserializer, CatalogType::TABLE_FUNCTION_ENTRY);
		auto function = std::move(entry.first);
		auto has_serialize = entry.second;
		unique_ptr<FunctionData> bind_data;
		if (has_serialize) {
			bind_data = FunctionSerializer::FunctionDeserialize(deserializer, function);
		} else {
			throw SerializationException("PhysicalStreamingUDF deserialization requires function serialization for %s",
			                             function.name);
		}

		auto column_ids = deserializer.ReadProperty<vector<ColumnIndex>>(200, "column_ids");
		auto projected_input = deserializer.ReadProperty<vector<column_t>>(201, "projected_input");
		auto ordinality_idx = deserializer.ReadPropertyWithDefault<optional_idx>(202, "ordinality_idx");

		auto streaming =
		    make_uniq<PhysicalStreamingUDF>(physical_plan, std::move(types), std::move(function), std::move(bind_data),
		                                    std::move(column_ids), estimated_cardinality, std::move(projected_input));
		streaming->ordinality_idx = ordinality_idx;
		return unique_ptr<PhysicalOperator>(std::move(streaming));
	}
	case PhysicalOperatorType::CTE: {
		auto ctename = deserializer.ReadProperty<string>(103, "ctename");
		auto table_index = deserializer.ReadProperty<idx_t>(104, "table_index");
		auto working_table_types =
		    deserializer.ReadPropertyWithDefault<vector<LogicalType>>(105, "working_table_types");
		auto right_types = types;
		if (working_table_types.empty()) {
			working_table_types = right_types;
		}

		auto cte = make_uniq<PhysicalCTE>(physical_plan, std::move(ctename), table_index, std::move(types),
		                                  estimated_cardinality);

		auto &context = deserializer.Get<ClientContext &>();
		auto working_table = make_shared_ptr<ColumnDataCollection>(context, working_table_types);
		cte->working_table = working_table;

		auto &state = GetOrCreateCTEState(deserializer.GetSerializationData());
		state.working_tables[table_index] = working_table;
		state.cte_ops[table_index] = cte.get();
		return unique_ptr<PhysicalOperator>(std::move(cte));
	}
	case PhysicalOperatorType::COLUMN_DATA_SCAN:
	case PhysicalOperatorType::CHUNK_SCAN:
	case PhysicalOperatorType::CTE_SCAN:
	case PhysicalOperatorType::DELIM_SCAN:
	case PhysicalOperatorType::RECURSIVE_CTE_SCAN:
	case PhysicalOperatorType::RECURSIVE_RECURRING_CTE_SCAN: {
		auto cte_index = deserializer.ReadProperty<idx_t>(103, "cte_index");
		auto delim_index = deserializer.ReadProperty<optional_idx>(104, "delim_index");
		auto has_collection = deserializer.ReadProperty<bool>(105, "has_collection");
		auto collection = [&]() {
			if (!has_collection) {
				return optionally_owned_ptr<ColumnDataCollection>();
			}
			auto owned = deserializer.ReadProperty<unique_ptr<ColumnDataCollection>>(106, "collection");
			return optionally_owned_ptr<ColumnDataCollection>(std::move(owned));
		}();
		auto scan = make_uniq<PhysicalColumnDataScan>(physical_plan, std::move(types), op_type, estimated_cardinality,
		                                              std::move(collection));
		scan->cte_index = cte_index;
		scan->delim_index = delim_index;
		scan->source_node_id = deserializer.ReadPropertyWithDefault<optional_idx>(107, "source_node_id");
		if (op_type == PhysicalOperatorType::CTE_SCAN || op_type == PhysicalOperatorType::RECURSIVE_CTE_SCAN ||
		    op_type == PhysicalOperatorType::RECURSIVE_RECURRING_CTE_SCAN) {
			auto &state = GetOrCreateCTEState(deserializer.GetSerializationData());
			auto cte_it = state.cte_ops.find(cte_index);
			if (cte_it != state.cte_ops.end()) {
				cte_it->second->cte_scans.push_back(*scan);
			}
			if (!scan->collection) {
				auto table_it = state.working_tables.find(cte_index);
				if (table_it != state.working_tables.end()) {
					scan->collection = table_it->second.get();
				}
			}
		}
		return unique_ptr<PhysicalOperator>(std::move(scan));
	}
	case PhysicalOperatorType::TABLE_SCAN: {
		auto entry = FunctionSerializer::DeserializeBase<TableFunction, TableFunctionCatalogEntry>(
		    deserializer, CatalogType::TABLE_FUNCTION_ENTRY);
		auto function = std::move(entry.first);
		auto has_serialize = entry.second;
		unique_ptr<FunctionData> bind_data;
		if (has_serialize) {
			bind_data = FunctionSerializer::FunctionDeserialize(deserializer, function);
		} else {
			throw SerializationException(
			    "PhysicalTableScan deserialization requires table function serialization for %s", function.name);
		}

		auto returned_types = deserializer.ReadProperty<vector<LogicalType>>(200, "returned_types");
		auto column_ids = deserializer.ReadProperty<vector<ColumnIndex>>(201, "column_ids");
		auto projection_ids = deserializer.ReadProperty<vector<idx_t>>(202, "projection_ids");
		auto names = deserializer.ReadProperty<vector<string>>(203, "names");

		unique_ptr<TableFilterSet> table_filters;
		auto has_table_filters = deserializer.ReadPropertyWithDefault<bool>(204, "has_table_filters");
		if (has_table_filters) {
			auto filter_set = deserializer.ReadProperty<TableFilterSet>(205, "table_filters");
			table_filters = make_uniq<TableFilterSet>(std::move(filter_set));
		}

		auto extra_info =
		    deserializer.ReadPropertyWithExplicitDefault<ExtraOperatorInfo>(206, "extra_info", ExtraOperatorInfo {});
		auto parameters =
		    deserializer.ReadPropertyWithExplicitDefault<vector<Value>>(207, "parameters", vector<Value>());
		auto virtual_columns = deserializer.ReadPropertyWithExplicitDefault<virtual_column_map_t>(
		    208, "virtual_columns", virtual_column_map_t());
		auto dynamic_filters_id = deserializer.ReadPropertyWithDefault<optional_idx>(209, "dynamic_filters_id");

		auto scan = make_uniq<PhysicalTableScan>(
		    physical_plan, std::move(types), std::move(function), std::move(bind_data), std::move(returned_types),
		    std::move(column_ids), std::move(projection_ids), std::move(names), std::move(table_filters),
		    estimated_cardinality, std::move(extra_info), std::move(parameters), std::move(virtual_columns));
		if (dynamic_filters_id.IsValid()) {
			auto &state = deserializer.GetSerializationData().GetCustom<DynamicTableFilterSerializationState>();
			scan->dynamic_filters = state.GetFilters(dynamic_filters_id);
		}
		return unique_ptr<PhysicalOperator>(std::move(scan));
	}
	case PhysicalOperatorType::HASH_JOIN: {
		auto join_type = deserializer.ReadProperty<JoinType>(103, "join_type");
		auto conditions = deserializer.ReadProperty<vector<JoinCondition>>(104, "conditions");
		auto condition_types = deserializer.ReadProperty<vector<LogicalType>>(105, "condition_types");
		auto payload_col_idxs = deserializer.ReadProperty<vector<idx_t>>(106, "payload_col_idxs");
		auto payload_col_types = deserializer.ReadProperty<vector<LogicalType>>(107, "payload_col_types");
		auto lhs_col_idxs = deserializer.ReadProperty<vector<idx_t>>(108, "lhs_col_idxs");
		auto lhs_col_types = deserializer.ReadProperty<vector<LogicalType>>(109, "lhs_col_types");
		auto rhs_col_idxs = deserializer.ReadProperty<vector<idx_t>>(110, "rhs_col_idxs");
		auto rhs_col_types = deserializer.ReadProperty<vector<LogicalType>>(111, "rhs_col_types");
		auto delim_types = deserializer.ReadPropertyWithDefault<vector<LogicalType>>(112, "delim_types");
		vector<unique_ptr<BaseStatistics>> join_stats;
		deserializer.ReadOptionalList(113, "join_stats", [&](Deserializer::List &list, idx_t /*i*/) {
			list.ReadObject([&](Deserializer &item_deserializer) {
				auto has_stats = item_deserializer.ReadProperty<bool>(0, "has_stats");
				if (!has_stats) {
					join_stats.push_back(nullptr);
					return;
				}
				auto stats_type = item_deserializer.ReadProperty<LogicalType>(1, "type");
				item_deserializer.Set<const LogicalType &>(stats_type);
				auto stats = item_deserializer.ReadProperty<BaseStatistics>(2, "stats");
				item_deserializer.Unset<LogicalType>();
				join_stats.push_back(make_uniq<BaseStatistics>(std::move(stats)));
			});
		});
		auto filter_pushdown =
		    deserializer.ReadPropertyWithDefault<unique_ptr<JoinFilterPushdownInfo>>(114, "filter_pushdown");

		LogicalComparisonJoin dummy_join(join_type);
		dummy_join.types = std::move(types);

		auto join = make_uniq<PhysicalHashJoin>(physical_plan, dummy_join, std::move(conditions), join_type,
		                                        std::move(delim_types), estimated_cardinality, true);
		join->condition_types = std::move(condition_types);
		join->payload_columns.col_idxs = std::move(payload_col_idxs);
		join->payload_columns.col_types = std::move(payload_col_types);
		join->lhs_output_columns.col_idxs = std::move(lhs_col_idxs);
		join->lhs_output_columns.col_types = std::move(lhs_col_types);
		join->rhs_output_columns.col_idxs = std::move(rhs_col_idxs);
		join->rhs_output_columns.col_types = std::move(rhs_col_types);
		join->join_stats = std::move(join_stats);
		join->filter_pushdown = std::move(filter_pushdown);
		return unique_ptr<PhysicalOperator>(std::move(join));
	}
	case PhysicalOperatorType::LEFT_DELIM_JOIN:
	case PhysicalOperatorType::RIGHT_DELIM_JOIN: {
		auto delim_idx = deserializer.ReadPropertyWithDefault<optional_idx>(103, "delim_idx");
		unique_ptr<PhysicalOperator> join;
		unique_ptr<PhysicalOperator> distinct;
		deserializer.ReadObject(104, "join", [&](Deserializer &child_deserializer) {
			join = PhysicalOperator::Deserialize(child_deserializer, physical_plan);
		});
		deserializer.ReadObject(105, "distinct", [&](Deserializer &child_deserializer) {
			distinct = PhysicalOperator::Deserialize(child_deserializer, physical_plan);
		});

		if (!join || !distinct) {
			throw SerializationException("Delim join deserialization failed: missing join or distinct operator");
		}

		auto *join_ptr = join.get();
		auto *distinct_ptr = distinct.get();
		physical_plan.TakeOwnership(std::move(join));
		physical_plan.TakeOwnership(std::move(distinct));

		vector<const_reference<PhysicalOperator>> delim_scans;
		GatherDelimScans(*join_ptr, delim_scans, delim_idx);
		if (!delim_idx.IsValid() && !delim_scans.empty()) {
			auto &scan = delim_scans[0].get().Cast<PhysicalColumnDataScan>();
			delim_idx = scan.delim_index;
		}
		if (delim_scans.empty()) {
			throw SerializationException("Delim join deserialization failed: no DELIM_SCAN nodes found");
		}

		if (op_type == PhysicalOperatorType::LEFT_DELIM_JOIN) {
			return make_uniq<PhysicalLeftDelimJoin>(physical_plan, DelimJoinDeserializeTag {}, std::move(types),
			                                        *join_ptr, *distinct_ptr, delim_scans, estimated_cardinality,
			                                        delim_idx);
		}
		return make_uniq<PhysicalRightDelimJoin>(physical_plan, DelimJoinDeserializeTag {}, std::move(types), *join_ptr,
		                                         *distinct_ptr, delim_scans, estimated_cardinality, delim_idx);
	}
	case PhysicalOperatorType::HASH_GROUP_BY: {
		auto groups = deserializer.ReadProperty<vector<unique_ptr<Expression>>>(103, "groups");
		auto aggregates = deserializer.ReadProperty<vector<unique_ptr<Expression>>>(104, "aggregates");
		auto &context = deserializer.Get<ClientContext &>();
		return make_uniq<PhysicalHashAggregate>(physical_plan, context, std::move(types), std::move(aggregates),
		                                        std::move(groups), estimated_cardinality);
	}
	case PhysicalOperatorType::PERFECT_HASH_GROUP_BY: {
		auto groups = deserializer.ReadProperty<vector<unique_ptr<Expression>>>(103, "groups");
		auto aggregates = deserializer.ReadProperty<vector<unique_ptr<Expression>>>(104, "aggregates");
		auto group_minima = deserializer.ReadProperty<vector<Value>>(105, "group_minima");
		auto required_bits = deserializer.ReadProperty<vector<idx_t>>(106, "required_bits");
		auto &context = deserializer.Get<ClientContext &>();
		return make_uniq<PhysicalPerfectHashAggregate>(physical_plan, context, std::move(types), std::move(aggregates),
		                                               std::move(groups), std::move(group_minima),
		                                               std::move(required_bits), estimated_cardinality);
	}
	case PhysicalOperatorType::PARTITIONED_AGGREGATE: {
		auto groups = deserializer.ReadProperty<vector<unique_ptr<Expression>>>(103, "groups");
		auto aggregates = deserializer.ReadProperty<vector<unique_ptr<Expression>>>(104, "aggregates");
		auto partitions = deserializer.ReadProperty<vector<column_t>>(105, "partitions");
		auto &context = deserializer.Get<ClientContext &>();
		return make_uniq<PhysicalPartitionedAggregate>(physical_plan, context, std::move(types), std::move(aggregates),
		                                               std::move(groups), std::move(partitions), estimated_cardinality);
	}
	case PhysicalOperatorType::UNGROUPED_AGGREGATE: {
		auto aggregates = deserializer.ReadProperty<vector<unique_ptr<Expression>>>(103, "aggregates");
		auto distinct_validity = deserializer.ReadPropertyWithExplicitDefault<TupleDataValidityType>(
		    104, "distinct_validity", TupleDataValidityType::CAN_HAVE_NULL_VALUES);
		return make_uniq<PhysicalUngroupedAggregate>(physical_plan, std::move(types), std::move(aggregates),
		                                             estimated_cardinality, distinct_validity);
	}
	case PhysicalOperatorType::WINDOW: {
		auto select_list = deserializer.ReadProperty<vector<unique_ptr<Expression>>>(103, "select_list");
		return make_uniq<PhysicalWindow>(physical_plan, std::move(types), std::move(select_list),
		                                 estimated_cardinality);
	}
	case PhysicalOperatorType::STREAMING_WINDOW: {
		auto select_list = deserializer.ReadProperty<vector<unique_ptr<Expression>>>(103, "select_list");
		return make_uniq<PhysicalStreamingWindow>(physical_plan, std::move(types), std::move(select_list),
		                                          estimated_cardinality);
	}
	case PhysicalOperatorType::PIVOT: {
		auto bound_pivot = deserializer.ReadProperty<BoundPivotInfo>(103, "bound_pivot");
		return make_uniq<PhysicalPivot>(physical_plan, std::move(types), std::move(bound_pivot), estimated_cardinality);
	}
	case PhysicalOperatorType::EXCHANGE_SINK: {
		auto exchange_id = deserializer.ReadProperty<string>(103, "shuffle_stage_id");
		auto node_id = deserializer.ReadProperty<string>(104, "node_id");
		auto num_partitions = deserializer.ReadProperty<idx_t>(105, "num_partitions");
		auto repartition_type_raw = deserializer.ReadProperty<uint8_t>(106, "repartition_type");
		auto partition_by = deserializer.ReadProperty<vector<unique_ptr<Expression>>>(107, "partition_by");
		auto local_dirs = deserializer.ReadProperty<vector<string>>(108, "local_dirs");
		auto bind_host = deserializer.ReadProperty<string>(109, "flight_bind_host");
		auto port = deserializer.ReadProperty<int>(110, "flight_port");
		auto repartition_type = static_cast<RepartitionSpec::Type>(repartition_type_raw);
		// Create FlightExchangeManager from deserialized config
		distributed::FlightExchangeConfig flight_config;
		flight_config.flight_bind_host = bind_host;
		flight_config.flight_port = port;
		flight_config.local_dirs = std::vector<std::string>(local_dirs.begin(), local_dirs.end());
		flight_config.node_id = node_id;
		auto exchange_mgr = std::make_shared<distributed::FlightExchangeManager>(std::move(flight_config));
		// Create sink handle for this exchange
		distributed::ExchangeSinkInstanceHandle sink_handle;
		sink_handle.sink_handle.task_partition_id =
		    deserializer.ReadPropertyWithDefault<idx_t>(111, "sink_task_partition_id");
		sink_handle.attempt_id = deserializer.ReadPropertyWithDefault<idx_t>(112, "sink_attempt_id");
		sink_handle.output_partition_count = num_partitions;
		sink_handle.output_location =
		    deserializer.ReadPropertyWithExplicitDefault<string>(113, "sink_output_location", exchange_id);
		auto range_boundaries =
		    deserializer.ReadPropertyWithExplicitDefault<vector<string>>(114, "range_boundaries", {});
		auto range_order_modifiers =
		    deserializer.ReadPropertyWithExplicitDefault<vector<string>>(115, "range_order_modifiers", {});
		return make_uniq<PhysicalRemoteExchangeSink>(
		    physical_plan, std::move(types), estimated_cardinality, std::move(exchange_id), num_partitions,
		    repartition_type, std::move(partition_by), std::move(sink_handle), std::move(exchange_mgr),
		    std::move(range_boundaries), std::move(range_order_modifiers));
	}
	case PhysicalOperatorType::EXCHANGE_SOURCE: {
		auto exchange_id = deserializer.ReadProperty<string>(103, "shuffle_stage_id");
		auto partition_indices = deserializer.ReadProperty<vector<idx_t>>(104, "partition_indices");
		auto source_nodes = deserializer.ReadProperty<vector<string>>(105, "source_nodes");
		auto location_template = deserializer.ReadProperty<string>(106, "flight_location_template");
		auto timeout_seconds = deserializer.ReadProperty<double>(107, "flight_timeout_seconds");
		auto source_handle_partition_ids =
		    deserializer.ReadPropertyWithDefault<vector<idx_t>>(108, "source_handle_partition_ids");
		auto source_handle_node_ids =
		    deserializer.ReadPropertyWithDefault<vector<string>>(109, "source_handle_node_ids");
		auto source_handle_paths = deserializer.ReadPropertyWithDefault<vector<string>>(110, "source_handle_paths");
		auto source_handle_flight_ports =
		    deserializer.ReadPropertyWithDefault<vector<int>>(111, "source_handle_flight_ports");
		auto runtime_source_node_id = deserializer.ReadPropertyWithDefault<optional_idx>(112, "runtime_source_node_id");
		auto source_handle_attempt_ids =
		    deserializer.ReadPropertyWithDefault<vector<idx_t>>(113, "source_handle_attempt_ids");
		auto local_dirs = deserializer.ReadPropertyWithDefault<vector<string>>(114, "local_dirs");
		// Create FlightExchangeManager from deserialized config
		distributed::FlightExchangeConfig flight_config;
		flight_config.node_id = distributed::ResolveFlightExchangeNodeIdFromEnv();
		flight_config.flight_location_template = location_template;
		flight_config.flight_timeout_seconds = timeout_seconds;
		flight_config.expected_types = types;
		flight_config.local_dirs = std::vector<std::string>(local_dirs.begin(), local_dirs.end());
		auto exchange_mgr = std::make_shared<distributed::FlightExchangeManager>(std::move(flight_config));
		std::vector<distributed::ExchangeSourceHandle> source_handles;
		if (!source_handle_partition_ids.empty() || !source_handle_node_ids.empty() || !source_handle_paths.empty()) {
			if (source_handle_partition_ids.size() != source_handle_node_ids.size() ||
			    source_handle_partition_ids.size() != source_handle_paths.size()) {
				throw SerializationException("remote exchange source handle metadata is inconsistent");
			}
			if (!source_handle_flight_ports.empty() &&
			    source_handle_flight_ports.size() != source_handle_partition_ids.size()) {
				throw SerializationException("remote exchange source flight port metadata is inconsistent");
			}
			if (!source_handle_attempt_ids.empty() &&
			    source_handle_attempt_ids.size() != source_handle_partition_ids.size()) {
				throw SerializationException("remote exchange source attempt metadata is inconsistent");
			}
			source_handles.reserve(source_handle_partition_ids.size());
			for (idx_t i = 0; i < source_handle_partition_ids.size(); i++) {
				distributed::ExchangeSourceHandle sh;
				sh.partition_id = source_handle_partition_ids[i];
				sh.attempt_id = source_handle_attempt_ids.empty() ? 0 : source_handle_attempt_ids[i];
				sh.node_id = source_handle_node_ids[i];
				sh.flight_port = source_handle_flight_ports.empty() ? 0 : source_handle_flight_ports[i];
				distributed::ExchangeSourceFile file;
				file.path = source_handle_paths[i];
				file.file_size = 0;
				sh.files.push_back(std::move(file));
				source_handles.push_back(std::move(sh));
			}
		} else if (!runtime_source_node_id.IsValid()) {
			// Legacy fallback for plans serialized before explicit source handles
			for (auto partition_idx : partition_indices) {
				for (idx_t i = 0; i < source_nodes.size(); i++) {
					distributed::ExchangeSourceHandle sh;
					sh.partition_id = partition_idx;
					sh.attempt_id = 0;
					sh.node_id = source_nodes[i];
					sh.flight_port = 0;
					distributed::ExchangeSourceFile file;
					file.path = exchange_id;
					file.file_size = 0;
					sh.files.push_back(std::move(file));
					source_handles.push_back(std::move(sh));
				}
			}
		}
		return make_uniq<PhysicalRemoteExchangeSource>(physical_plan, std::move(types), estimated_cardinality,
		                                               std::move(exchange_id), std::move(partition_indices),
		                                               std::move(source_handles), std::move(exchange_mgr), source_nodes,
		                                               runtime_source_node_id);
	}
	case PhysicalOperatorType::REPARTITION: {
		auto repartition_type_raw = deserializer.ReadProperty<uint8_t>(103, "repartition_type");
		bool has_num_partitions = false;
		deserializer.ReadPropertyWithDefault(104, "has_num_partitions", has_num_partitions);
		size_t num_partitions = 0;
		if (has_num_partitions) {
			num_partitions = deserializer.ReadProperty<idx_t>(105, "num_partitions");
		}
		auto partition_by = deserializer.ReadPropertyWithDefault<vector<unique_ptr<Expression>>>(106, "partition_by");
		auto repartition_type = static_cast<RepartitionSpec::Type>(repartition_type_raw);

		std::shared_ptr<RepartitionSpec> spec;
		switch (repartition_type) {
		case RepartitionSpec::Type::Hash: {
			vector<ExprRef> expr_refs;
			expr_refs.reserve(partition_by.size());
			for (auto &expr : partition_by) {
				expr_refs.emplace_back(expr->Copy());
			}
			spec = RepartitionSpec::create_hash(num_partitions, std::move(expr_refs));
			break;
		}
		case RepartitionSpec::Type::Random:
			spec = RepartitionSpec::create_random(num_partitions);
			break;
		case RepartitionSpec::Type::IntoPartitions:
			if (!num_partitions) {
				num_partitions = 1;
			}
			spec = RepartitionSpec::create_into_partitions(num_partitions);
			break;
		case RepartitionSpec::Type::Range:
			throw NotImplementedException("Deserialization not implemented for range repartition");
		}

		auto repartition =
		    make_uniq<PhysicalRepartition>(physical_plan, std::move(types), std::move(spec), estimated_cardinality);
		repartition->partition_by = std::move(partition_by);
		return unique_ptr<PhysicalOperator>(std::move(repartition));
	}
	case PhysicalOperatorType::LOCAL_EXCHANGE: {
		auto repartition_type_raw = deserializer.ReadProperty<uint8_t>(103, "repartition_type");
		bool has_num_partitions = false;
		deserializer.ReadPropertyWithDefault(104, "has_num_partitions", has_num_partitions);
		size_t num_partitions = 0;
		if (has_num_partitions) {
			num_partitions = deserializer.ReadProperty<idx_t>(105, "num_partitions");
		}
		auto partition_by = deserializer.ReadProperty<vector<unique_ptr<Expression>>>(106, "partition_by");
		auto repartition_type = static_cast<RepartitionSpec::Type>(repartition_type_raw);

		std::shared_ptr<RepartitionSpec> spec;
		switch (repartition_type) {
		case RepartitionSpec::Type::Hash: {
			vector<ExprRef> expr_refs;
			expr_refs.reserve(partition_by.size());
			for (auto &expr : partition_by) {
				expr_refs.emplace_back(expr->Copy());
			}
			spec = RepartitionSpec::create_hash(num_partitions, std::move(expr_refs));
			break;
		}
		case RepartitionSpec::Type::Random:
			spec = RepartitionSpec::create_random(num_partitions);
			break;
		case RepartitionSpec::Type::IntoPartitions:
			if (!num_partitions) {
				throw SerializationException("LOCAL_EXCHANGE deserialize missing num_partitions");
			}
			spec = RepartitionSpec::create_into_partitions(num_partitions);
			break;
		case RepartitionSpec::Type::Range:
			throw NotImplementedException("Deserialization not implemented for range repartition");
		}

		auto repartition =
		    make_uniq<PhysicalLocalExchange>(physical_plan, std::move(types), std::move(spec), estimated_cardinality);
		repartition->partition_by = std::move(partition_by);
		return unique_ptr<PhysicalOperator>(std::move(repartition));
	}
	case PhysicalOperatorType::COPY_TO_FILE: {
		auto &context = deserializer.Get<ClientContext &>();
		auto name = deserializer.ReadProperty<string>(500, "name");
		auto catalog_name = deserializer.ReadPropertyWithDefault<string>(505, "catalog_name");
		auto schema_name = deserializer.ReadPropertyWithDefault<string>(506, "schema_name");
		auto has_serialize = deserializer.ReadProperty<bool>(503, "has_serialize");
		if (catalog_name.empty()) {
			catalog_name = SYSTEM_CATALOG;
		}
		if (schema_name.empty()) {
			schema_name = DEFAULT_SCHEMA;
		}
		auto &entry = Catalog::GetEntry(context, CatalogType::COPY_FUNCTION_ENTRY, catalog_name, schema_name, name);
		auto &copy_entry = entry.Cast<CopyFunctionCatalogEntry>();
		auto function = copy_entry.function;
		unique_ptr<FunctionData> bind_data;
		if (has_serialize) {
			if (!function.deserialize) {
				throw SerializationException("Copy function %s is missing deserialize hook", function.name);
			}
			deserializer.ReadObject(504, "function_data",
			                        [&](Deserializer &obj) { bind_data = function.deserialize(obj, function); });
		} else {
			throw SerializationException("PhysicalCopyToFile deserialization requires function serialization for %s",
			                             function.name);
		}

		auto file_path = deserializer.ReadProperty<string>(200, "file_path");
		auto use_tmp_file = deserializer.ReadProperty<bool>(201, "use_tmp_file");
		auto filename_pattern = deserializer.ReadProperty<FilenamePattern>(202, "filename_pattern");
		auto file_extension = deserializer.ReadProperty<string>(203, "file_extension");
		auto overwrite_mode = deserializer.ReadProperty<CopyOverwriteMode>(204, "overwrite_mode");
		auto parallel = deserializer.ReadProperty<bool>(205, "parallel");
		auto per_thread_output = deserializer.ReadProperty<bool>(206, "per_thread_output");
		auto file_size_bytes = deserializer.ReadProperty<optional_idx>(207, "file_size_bytes");
		auto rotate = deserializer.ReadProperty<bool>(208, "rotate");
		auto return_type = deserializer.ReadProperty<CopyFunctionReturnType>(209, "return_type");
		auto partition_output = deserializer.ReadProperty<bool>(210, "partition_output");
		auto write_partition_columns = deserializer.ReadProperty<bool>(211, "write_partition_columns");
		auto write_empty_file = deserializer.ReadProperty<bool>(212, "write_empty_file");
		auto hive_file_pattern = deserializer.ReadProperty<bool>(213, "hive_file_pattern");
		auto partition_columns = deserializer.ReadProperty<vector<idx_t>>(214, "partition_columns");
		auto names = deserializer.ReadProperty<vector<string>>(215, "names");
		auto expected_types = deserializer.ReadProperty<vector<LogicalType>>(216, "expected_types");

		auto copy = make_uniq<PhysicalCopyToFile>(physical_plan, std::move(types), std::move(function),
		                                          std::move(bind_data), estimated_cardinality);
		copy->file_path = std::move(file_path);
		copy->use_tmp_file = use_tmp_file;
		copy->filename_pattern = std::move(filename_pattern);
		copy->file_extension = std::move(file_extension);
		copy->overwrite_mode = overwrite_mode;
		copy->parallel = parallel;
		copy->per_thread_output = per_thread_output;
		copy->file_size_bytes = file_size_bytes;
		copy->rotate = rotate;
		copy->return_type = return_type;
		copy->partition_output = partition_output;
		copy->write_partition_columns = write_partition_columns;
		copy->write_empty_file = write_empty_file;
		copy->hive_file_pattern = hive_file_pattern;
		copy->partition_columns = std::move(partition_columns);
		copy->names = std::move(names);
		copy->expected_types = std::move(expected_types);
		return unique_ptr<PhysicalOperator>(std::move(copy));
	}
	case PhysicalOperatorType::BATCH_COPY_TO_FILE: {
		auto &context = deserializer.Get<ClientContext &>();
		auto name = deserializer.ReadProperty<string>(500, "name");
		auto catalog_name = deserializer.ReadPropertyWithDefault<string>(505, "catalog_name");
		auto schema_name = deserializer.ReadPropertyWithDefault<string>(506, "schema_name");
		auto has_serialize = deserializer.ReadProperty<bool>(503, "has_serialize");
		if (catalog_name.empty()) {
			catalog_name = SYSTEM_CATALOG;
		}
		if (schema_name.empty()) {
			schema_name = DEFAULT_SCHEMA;
		}
		auto &entry = Catalog::GetEntry(context, CatalogType::COPY_FUNCTION_ENTRY, catalog_name, schema_name, name);
		auto &copy_entry = entry.Cast<CopyFunctionCatalogEntry>();
		auto function = copy_entry.function;
		unique_ptr<FunctionData> bind_data;
		if (has_serialize) {
			if (!function.deserialize) {
				throw SerializationException("Copy function %s is missing deserialize hook", function.name);
			}
			deserializer.ReadObject(504, "function_data",
			                        [&](Deserializer &obj) { bind_data = function.deserialize(obj, function); });
		} else {
			throw SerializationException(
			    "PhysicalBatchCopyToFile deserialization requires function serialization for %s", function.name);
		}

		auto file_path = deserializer.ReadProperty<string>(200, "file_path");
		auto use_tmp_file = deserializer.ReadProperty<bool>(201, "use_tmp_file");
		auto return_type = deserializer.ReadProperty<CopyFunctionReturnType>(202, "return_type");
		auto write_empty_file = deserializer.ReadProperty<bool>(203, "write_empty_file");

		auto copy = make_uniq<PhysicalBatchCopyToFile>(physical_plan, std::move(types), std::move(function),
		                                               std::move(bind_data), estimated_cardinality);
		copy->file_path = std::move(file_path);
		copy->use_tmp_file = use_tmp_file;
		copy->return_type = return_type;
		copy->write_empty_file = write_empty_file;
		return unique_ptr<PhysicalOperator>(std::move(copy));
	}
	case PhysicalOperatorType::NESTED_LOOP_JOIN: {
		auto join_type = deserializer.ReadProperty<JoinType>(103, "join_type");
		auto conditions = deserializer.ReadProperty<vector<JoinCondition>>(104, "conditions");
		auto predicate = deserializer.ReadPropertyWithDefault<unique_ptr<Expression>>(105, "predicate");

		LogicalComparisonJoin dummy_join(join_type);
		dummy_join.types = std::move(types);

		auto join = make_uniq<PhysicalNestedLoopJoin>(physical_plan, dummy_join, std::move(conditions), join_type,
		                                              estimated_cardinality, true);
		join->predicate = std::move(predicate);
		return unique_ptr<PhysicalOperator>(std::move(join));
	}
	default:
		throw NotImplementedException("Deserialization not implemented for operator type: %s",
		                              PhysicalOperatorToString(op_type));
	}
}

} // namespace duckdb
