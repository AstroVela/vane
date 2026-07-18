// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/operator/exchange/physical_local_exchange.hpp"
#include "duckdb/execution/distributed/common_types.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/allocator.hpp"
#include "duckdb/common/atomic.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/hash.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"

#include <cstdlib>
#include <deque>

namespace duckdb {

static idx_t EstimateVarlenBytes(const Vector &vec, const idx_t count) {
	if (count == 0) {
		return 0;
	}
	// Vector::GetAllocationSize does not include varlen heap payloads for VARCHAR/BLOB.
	// We approximate the additional heap bytes by summing the non-inlined string lengths.
	Vector scan_vec(vec.GetType());
	scan_vec.Reference(const_cast<Vector &>(vec));
	UnifiedVectorFormat vdata;
	scan_vec.ToUnifiedFormat(count, vdata);
	const auto strings = UnifiedVectorFormat::GetData<string_t>(vdata);
	idx_t total = 0;
	for (idx_t row_idx = 0; row_idx < count; row_idx++) {
		auto idx = vdata.sel->get_index(row_idx);
		if (!vdata.validity.RowIsValid(idx)) {
			continue;
		}
		auto &str = strings[idx];
		if (!str.IsInlined()) {
			total += str.GetSize();
		}
	}
	return total;
}

static idx_t EstimateChunkBytes(const DataChunk &chunk) {
	auto count = chunk.size();
	idx_t total = chunk.GetAllocationSize();
	for (auto &vec : chunk.data) {
		if (vec.GetType().InternalType() == PhysicalType::VARCHAR) {
			total += EstimateVarlenBytes(vec, count);
		}
	}
	return total;
}

static idx_t LocalExchangeBatchRows(const ExecutionBatch &batch) {
	if (batch.kind == ExecutionBatchKind::MATERIALIZED_CHUNK) {
		return batch.materialized ? batch.materialized->size() : batch.rows;
	}
	if (batch.kind == ExecutionBatchKind::LAZY_DATA_CHUNK) {
		return batch.lazy ? batch.lazy->cardinality : batch.rows;
	}
	return batch.rows;
}

static idx_t LocalExchangeBatchBytes(const ExecutionBatch &batch) {
	if (batch.kind == ExecutionBatchKind::MATERIALIZED_CHUNK) {
		return batch.materialized ? EstimateChunkBytes(*batch.materialized) : batch.estimated_bytes;
	}
	if (batch.kind == ExecutionBatchKind::LAZY_DATA_CHUNK) {
		return batch.lazy ? batch.lazy->EstimatedBytes() : batch.estimated_bytes;
	}
	return batch.estimated_bytes;
}

struct LocalExchangeChunk {
	ExecutionBatch batch;
	idx_t bytes = 0;
};

static void StoreLocalExchangeMaterialized(LocalExchangeChunk &entry, unique_ptr<DataChunk> chunk, idx_t bytes) {
	entry = LocalExchangeChunk();
	entry.batch.kind = ExecutionBatchKind::MATERIALIZED_CHUNK;
	if (chunk) {
		entry.batch.rows = chunk->size();
		entry.batch.estimated_bytes = bytes;
	}
	entry.batch.materialized = std::move(chunk);
	entry.bytes = bytes;
}

static void StoreLocalExchangeLazy(LocalExchangeChunk &entry, unique_ptr<LazyDataChunk> lazy) {
	entry = LocalExchangeChunk();
	entry.batch.kind = ExecutionBatchKind::LAZY_DATA_CHUNK;
	if (lazy) {
		lazy->RecomputeCardinality();
		entry.batch.rows = lazy->cardinality;
		entry.batch.estimated_bytes = lazy->EstimatedBytes();
	}
	entry.bytes = entry.batch.estimated_bytes;
	entry.batch.lazy = std::move(lazy);
}

static optional_ptr<DataChunk> LocalExchangeMaterializedChunk(LocalExchangeChunk &entry) {
	if (entry.batch.kind == ExecutionBatchKind::MATERIALIZED_CHUNK && entry.batch.materialized) {
		return *entry.batch.materialized;
	}
	return nullptr;
}

static bool LocalExchangeHasLazyChunk(const LocalExchangeChunk &entry) {
	return entry.batch.kind == ExecutionBatchKind::LAZY_DATA_CHUNK && entry.batch.lazy;
}

struct LocalExchangeState : public StateWithBlockableTasks {
	explicit LocalExchangeState(idx_t num_partitions_p, idx_t max_buffer_bytes_p)
	    : num_partitions(num_partitions_p), max_buffer_bytes(max_buffer_bytes_p), buffered_bytes(0),
	      inflight_reserved_bytes(0), finished(false), queues(num_partitions_p) {
	}

	idx_t num_partitions;
	idx_t max_buffer_bytes;
	idx_t buffered_bytes;
	idx_t inflight_reserved_bytes;
	bool finished;
	vector<std::deque<LocalExchangeChunk>> queues;
};

struct LocalExchangeGlobalSinkState : public GlobalSinkState {
	explicit LocalExchangeGlobalSinkState(std::shared_ptr<LocalExchangeState> exchange_state_p)
	    : exchange_state(std::move(exchange_state_p)) {
	}

	std::shared_ptr<LocalExchangeState> exchange_state;
};

struct LocalExchangeLocalSinkState : public LocalSinkState {
	LocalExchangeLocalSinkState(ClientContext &context, const PhysicalLocalExchange &op, idx_t num_partitions)
	    : executor(context), hash(LogicalType::HASH) {
		if (!op.partition_by.empty()) {
			vector<LogicalType> types;
			types.reserve(op.partition_by.size());
			for (auto &expr : op.partition_by) {
				executor.AddExpression(*expr);
				types.push_back(expr->return_type);
			}
			key_chunk.InitializeEmpty(types);
		}
		selections.reserve(num_partitions);
		for (idx_t i = 0; i < num_partitions; i++) {
			selections.emplace_back(STANDARD_VECTOR_SIZE);
		}
		sel_counts.resize(num_partitions, 0);
	}

	ExpressionExecutor executor;
	DataChunk key_chunk;
	Vector hash;
	vector<SelectionVector> selections;
	vector<idx_t> sel_counts;
	idx_t round_robin = 0;
};

struct LocalExchangeGlobalSourceState : public GlobalSourceState {
	explicit LocalExchangeGlobalSourceState(std::shared_ptr<LocalExchangeState> exchange_state_p)
	    : exchange_state(std::move(exchange_state_p)), next_partition(0) {
	}

	idx_t MaxThreads() override {
		return exchange_state->num_partitions;
	}

	std::shared_ptr<LocalExchangeState> exchange_state;
	atomic<idx_t> next_partition;
};

struct LocalExchangeLocalSourceState : public LocalSourceState {
	explicit LocalExchangeLocalSourceState(idx_t partition_cursor_p) : partition_cursor(partition_cursor_p) {
	}

	idx_t partition_cursor;
};

PhysicalLocalExchange::PhysicalLocalExchange(PhysicalPlan &physical_plan, vector<LogicalType> types,
                                             std::shared_ptr<RepartitionSpec> repartition_spec,
                                             idx_t estimated_cardinality)
    : PhysicalLocalExchange(physical_plan, PhysicalOperatorType::LOCAL_EXCHANGE, std::move(types),
                            std::move(repartition_spec), estimated_cardinality) {
}

PhysicalLocalExchange::PhysicalLocalExchange(PhysicalPlan &physical_plan, PhysicalOperatorType type,
                                             vector<LogicalType> types,
                                             std::shared_ptr<RepartitionSpec> repartition_spec,
                                             idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, type, std::move(types), estimated_cardinality),
      repartition_spec(std::move(repartition_spec)) {
	if (this->repartition_spec) {
		auto exprs = this->repartition_spec->repartition_by();
		partition_by.reserve(exprs.size());
		for (auto &expr_ref : exprs) {
			if (expr_ref) {
				partition_by.push_back(expr_ref->Copy());
			}
		}
	}
}

unique_ptr<OperatorState> PhysicalLocalExchange::GetOperatorState(ExecutionContext &context) const {
	return nullptr;
}

void PhysicalLocalExchange::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	// operator is a sink, build a pipeline
	sink_state.reset();

	if (children.size() != 1) {
		throw InternalException("PhysicalLocalExchange requires exactly one child");
	}

	auto &state = meta_pipeline.GetState();
	// single operator: the operator becomes the data source of the current pipeline
	state.SetPipelineSource(current, *this);

	auto &config = DBConfig::GetConfig(current.GetClientContext());
	const bool user_streaming = config.options.local_exchange_streaming;
	const bool bounded_buffer = config.options.local_exchange_buffer_bytes > 0;
	// A bounded local exchange queue must let source/sink run concurrently.
	const bool streaming = user_streaming || bounded_buffer;
	const bool add_dependency = !streaming;

	// create a new pipeline starting from the child
	auto &child_meta_pipeline =
	    meta_pipeline.CreateChildMetaPipeline(current, *this, MetaPipelineType::REGULAR, add_dependency);
	child_meta_pipeline.Build(children[0].get());
}

idx_t PhysicalLocalExchange::ResolveNumPartitions(ClientContext &context) const {
	if (!repartition_spec) {
		return 1;
	}
	const auto type = repartition_spec->type();
	idx_t resolved = 0;
	switch (type) {
	case RepartitionSpec::Type::Hash: {
		auto *hash_spec = dynamic_cast<HashRepartitionSpec *>(repartition_spec.get());
		if (hash_spec && hash_spec->config()->num_partitions) {
			resolved = NumericCast<idx_t>(hash_spec->config()->num_partitions);
		}
		break;
	}
	case RepartitionSpec::Type::Random: {
		auto *random_spec = dynamic_cast<RandomRepartitionSpec *>(repartition_spec.get());
		if (random_spec && random_spec->config()->num_partitions) {
			resolved = NumericCast<idx_t>(random_spec->config()->num_partitions);
		}
		break;
	}
	case RepartitionSpec::Type::IntoPartitions: {
		auto *into_spec = dynamic_cast<IntoPartitionsRepartitionSpec *>(repartition_spec.get());
		if (!into_spec) {
			throw InternalException("Expected IntoPartitionsRepartitionSpec for repartition");
		}
		resolved = NumericCast<idx_t>(into_spec->config()->num_partitions);
		break;
	}
	case RepartitionSpec::Type::Range:
		throw NotImplementedException("Local repartition does not support range repartition");
	default:
		break;
	}
	auto threads = NumericCast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads());
	auto &config = DBConfig::GetConfig(context);
	auto default_partitions = config.options.local_exchange_default_partitions;
	if (default_partitions == 0) {
		default_partitions = MinValue<idx_t>(NumericCast<idx_t>(threads), 32);
	}
	auto max_partitions = config.options.local_exchange_max_partitions;
	if (max_partitions == 0) {
		max_partitions = NumericCast<idx_t>(threads) * 2;
	}
	if (resolved == 0) {
		resolved = MaxValue<idx_t>(default_partitions, 1);
	}
	if (max_partitions > 0 && resolved > max_partitions) {
		resolved = max_partitions;
	}
	return resolved;
}

std::shared_ptr<LocalExchangeState> PhysicalLocalExchange::GetExchangeState(ClientContext &context) const {
	lock_guard<std::mutex> guard(exchange_lock);
	if (!exchange_state) {
		auto num_partitions = ResolveNumPartitions(context);
		if (num_partitions == 0) {
			throw InvalidInputException("repartition requires at least one partition");
		}
		auto &config = DBConfig::GetConfig(context);
		const idx_t max_buffer_bytes = config.options.local_exchange_buffer_bytes;
		exchange_state = std::make_shared<LocalExchangeState>(num_partitions, max_buffer_bytes);
	}
	return exchange_state;
}

unique_ptr<GlobalSinkState> PhysicalLocalExchange::GetGlobalSinkState(ClientContext &context) const {
	auto exchange = GetExchangeState(context);
	return make_uniq<LocalExchangeGlobalSinkState>(std::move(exchange));
}

unique_ptr<LocalSinkState> PhysicalLocalExchange::GetLocalSinkState(ExecutionContext &context) const {
	auto exchange = GetExchangeState(context.client);
	return make_uniq<LocalExchangeLocalSinkState>(context.client, *this, exchange->num_partitions);
}

static idx_t SelectPartitionHash(const hash_t hash, const idx_t num_partitions) {
	if (num_partitions == 1) {
		return 0;
	}
	return static_cast<idx_t>(hash % num_partitions);
}

SinkResultType PhysicalLocalExchange::Sink(ExecutionContext &context, DataChunk &chunk,
                                           OperatorSinkInput &input) const {
	if (chunk.size() == 0) {
		return SinkResultType::NEED_MORE_INPUT;
	}

	auto &gstate = input.global_state.Cast<LocalExchangeGlobalSinkState>();
	auto &lstate = input.local_state.Cast<LocalExchangeLocalSinkState>();
	auto &exchange = *gstate.exchange_state;

	const auto count = chunk.size();
	const auto partitions = exchange.num_partitions;

	idx_t reserved_bytes = 0;
	if (exchange.max_buffer_bytes > 0) {
		auto estimated_bytes = EstimateChunkBytes(chunk);
		auto guard = exchange.Lock();
		// Include in-flight reservations so concurrent sink threads cannot over-admit chunks.
		const auto total_reserved = exchange.buffered_bytes + exchange.inflight_reserved_bytes;
		const bool allow_oversized_chunk = estimated_bytes > exchange.max_buffer_bytes && total_reserved == 0;
		if (!allow_oversized_chunk && total_reserved + estimated_bytes > exchange.max_buffer_bytes) {
			return exchange.BlockSink(guard, input.interrupt_state);
		}
		exchange.inflight_reserved_bytes += estimated_bytes;
		reserved_bytes = estimated_bytes;
	}

	std::fill(lstate.sel_counts.begin(), lstate.sel_counts.end(), 0);

	const auto repartition_type = repartition_spec ? repartition_spec->type() : RepartitionSpec::Type::Random;
	switch (repartition_type) {
	case RepartitionSpec::Type::Hash: {
		if (partition_by.empty()) {
			throw InvalidInputException("hash repartition requires partition expressions");
		}
		lstate.executor.Execute(chunk, lstate.key_chunk);
		lstate.key_chunk.Hash(lstate.hash);
		lstate.hash.Flatten(count);
		auto hashes = FlatVector::GetData<hash_t>(lstate.hash);
		for (idx_t row_idx = 0; row_idx < count; row_idx++) {
			auto part = SelectPartitionHash(hashes[row_idx], partitions);
			auto &sel_count = lstate.sel_counts[part];
			lstate.selections[part].set_index(sel_count++, row_idx);
		}
		break;
	}
	case RepartitionSpec::Type::Random:
	case RepartitionSpec::Type::IntoPartitions: {
		idx_t current = lstate.round_robin;
		for (idx_t row_idx = 0; row_idx < count; row_idx++) {
			auto part = current++ % partitions;
			auto &sel_count = lstate.sel_counts[part];
			lstate.selections[part].set_index(sel_count++, row_idx);
		}
		lstate.round_robin = current % partitions;
		break;
	}
	case RepartitionSpec::Type::Range:
		throw NotImplementedException("local repartition does not support range repartition");
	default:
		throw NotImplementedException("local repartition encountered unknown repartition type");
	}

	vector<LocalExchangeChunk> ready_chunks;
	ready_chunks.resize(partitions);
	vector<idx_t> ready_rows;
	for (idx_t part_idx = 0; part_idx < partitions; part_idx++) {
		auto part_count = lstate.sel_counts[part_idx];
		if (part_count == 0) {
			continue;
		}
		auto out_chunk = make_uniq<DataChunk>();
		out_chunk->Initialize(BufferAllocator::Get(context.client), chunk.GetTypes(), part_count);
		out_chunk->Append(chunk, false, &lstate.selections[part_idx], part_count);
		out_chunk->SetCardinality(part_count);
		auto bytes = EstimateChunkBytes(*out_chunk);
		LocalExchangeChunk lec;
		StoreLocalExchangeMaterialized(lec, std::move(out_chunk), bytes);
		ready_chunks[part_idx] = std::move(lec);
	}

	idx_t enqueued_chunks = 0;
	idx_t actual_enqueued_bytes = 0;
	idx_t buffered_after = 0;
	{
		auto guard = exchange.Lock();
		if (reserved_bytes > 0) {
			exchange.inflight_reserved_bytes = exchange.inflight_reserved_bytes >= reserved_bytes
			                                       ? exchange.inflight_reserved_bytes - reserved_bytes
			                                       : 0;
		}
		for (idx_t part_idx = 0; part_idx < partitions; part_idx++) {
			auto &entry = ready_chunks[part_idx];
			if (LocalExchangeBatchRows(entry.batch) == 0) {
				continue;
			}
			exchange.buffered_bytes += entry.bytes;
			actual_enqueued_bytes += entry.bytes;
			exchange.queues[part_idx].push_back(std::move(entry));
			enqueued_chunks++;
		}
		buffered_after = exchange.buffered_bytes;
		exchange.UnblockTasks(guard);
	}

	return SinkResultType::NEED_MORE_INPUT;
}

SinkResultType PhysicalLocalExchange::SinkBatch(ExecutionContext &context, ExecutionBatch &batch,
                                                OperatorSinkInput &input) const {
	if (batch.kind != ExecutionBatchKind::LAZY_DATA_CHUNK || !batch.lazy) {
		if (batch.kind == ExecutionBatchKind::MATERIALIZED_CHUNK && batch.materialized) {
			return Sink(context, *batch.materialized, input);
		}
		return PhysicalOperator::SinkBatch(context, batch, input);
	}

	batch.lazy->RecomputeCardinality();
	if (batch.lazy->cardinality == 0) {
		return SinkResultType::NEED_MORE_INPUT;
	}

	const auto repartition_type = repartition_spec ? repartition_spec->type() : RepartitionSpec::Type::Random;
	if (repartition_type != RepartitionSpec::Type::Random &&
	    repartition_type != RepartitionSpec::Type::IntoPartitions) {
		auto barrier = MaterializeExternalBlockBarrier(context.client, *batch.lazy, GetName());
		return Sink(context, *barrier.chunk, input);
	}

	auto &gstate = input.global_state.Cast<LocalExchangeGlobalSinkState>();
	auto &lstate = input.local_state.Cast<LocalExchangeLocalSinkState>();
	auto &exchange = *gstate.exchange_state;

	const auto count = batch.lazy->cardinality;
	const auto partitions = exchange.num_partitions;

	if (partitions == 0) {
		return SinkResultType::NEED_MORE_INPUT;
	}

	auto estimated_bytes = LocalExchangeBatchBytes(batch);
	idx_t reserved_bytes = 0;
	if (exchange.max_buffer_bytes > 0) {
		auto guard = exchange.Lock();
		const auto total_reserved = exchange.buffered_bytes + exchange.inflight_reserved_bytes;
		const bool allow_oversized_chunk = estimated_bytes > exchange.max_buffer_bytes && total_reserved == 0;
		if (!allow_oversized_chunk && total_reserved + estimated_bytes > exchange.max_buffer_bytes) {
			return exchange.BlockSink(guard, input.interrupt_state);
		}
		exchange.inflight_reserved_bytes += estimated_bytes;
		reserved_bytes = estimated_bytes;
	}

	auto part_idx = lstate.round_robin % partitions;
	lstate.round_robin = (lstate.round_robin + 1) % partitions;

	LocalExchangeChunk entry;
	StoreLocalExchangeLazy(entry, std::move(batch.lazy));
	estimated_bytes = entry.bytes;

	idx_t buffered_after = 0;
	{
		auto guard = exchange.Lock();
		if (reserved_bytes > 0) {
			exchange.inflight_reserved_bytes = exchange.inflight_reserved_bytes >= reserved_bytes
			                                       ? exchange.inflight_reserved_bytes - reserved_bytes
			                                       : 0;
		}
		exchange.buffered_bytes += entry.bytes;
		exchange.queues[part_idx].push_back(std::move(entry));
		buffered_after = exchange.buffered_bytes;
		exchange.UnblockTasks(guard);
	}
	batch = ExecutionBatch();

	return SinkResultType::NEED_MORE_INPUT;
}

SinkFinalizeType PhysicalLocalExchange::Finalize(Pipeline &, Event &, ClientContext &context,
                                                 OperatorSinkFinalizeInput &input) const {
	auto &gstate = input.global_state.Cast<LocalExchangeGlobalSinkState>();
	auto &exchange = *gstate.exchange_state;
	auto guard = exchange.Lock();
	exchange.finished = true;
	exchange.UnblockTasks(guard);
	return SinkFinalizeType::READY;
}

unique_ptr<GlobalSourceState> PhysicalLocalExchange::GetGlobalSourceState(ClientContext &context) const {
	auto exchange = GetExchangeState(context);
	return make_uniq<LocalExchangeGlobalSourceState>(std::move(exchange));
}

unique_ptr<LocalSourceState> PhysicalLocalExchange::GetLocalSourceState(ExecutionContext &,
                                                                        GlobalSourceState &gstate) const {
	auto &source_state = gstate.Cast<LocalExchangeGlobalSourceState>();
	auto start = source_state.next_partition.fetch_add(1);
	auto &exchange = *source_state.exchange_state;
	if (exchange.num_partitions > 0) {
		start = start % exchange.num_partitions;
	}
	return make_uniq<LocalExchangeLocalSourceState>(start);
}

static idx_t LocalExchangeCoalesceRows() {
	static idx_t cached = 0;
	static bool resolved = false;
	if (!resolved) {
		resolved = true;
		cached = 64; // default
		auto *value = std::getenv("VANE_LOCAL_EXCHANGE_COALESCE_ROWS");
		if (value && *value) {
			char *endptr = nullptr;
			auto parsed = std::strtoll(value, &endptr, 10);
			if (endptr && *endptr == '\0' && parsed >= 0) {
				cached = static_cast<idx_t>(parsed);
			}
		}
	}
	return cached;
}

SourceResultType PhysicalLocalExchange::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                        OperatorSourceInput &input) const {
	auto &gstate = input.global_state.Cast<LocalExchangeGlobalSourceState>();
	auto &lstate = input.local_state.Cast<LocalExchangeLocalSourceState>();
	auto &exchange = *gstate.exchange_state;

	LocalExchangeChunk entry;
	idx_t picked_partition = DConstants::INVALID_INDEX;

	// Collect extra chunks for coalescing (filled under lock, merged after)
	vector<LocalExchangeChunk> extra_chunks;
	const auto coalesce_target = LocalExchangeCoalesceRows();

	{
		auto guard = exchange.Lock();
		if (exchange.num_partitions == 0) {
			chunk.SetCardinality(0);
			return SourceResultType::FINISHED;
		}
		for (idx_t attempt = 0; attempt < exchange.num_partitions; attempt++) {
			auto partition_idx = (lstate.partition_cursor + attempt) % exchange.num_partitions;
			auto &queue = exchange.queues[partition_idx];
			if (queue.empty()) {
				continue;
			}
			entry = std::move(queue.front());
			queue.pop_front();
			exchange.buffered_bytes =
			    exchange.buffered_bytes >= entry.bytes ? exchange.buffered_bytes - entry.bytes : 0;

			// Coalesce materialized chunks from the same partition until target rows.
			auto entry_chunk = LocalExchangeMaterializedChunk(entry);
			if (coalesce_target > 0 && entry_chunk && entry_chunk->size() < coalesce_target) {
				idx_t accumulated_rows = entry_chunk->size();
				while (!queue.empty() && accumulated_rows < coalesce_target) {
					auto &next = queue.front();
					auto next_chunk = LocalExchangeMaterializedChunk(next);
					if (!next_chunk) {
						break;
					}
					// Don't exceed STANDARD_VECTOR_SIZE
					if (accumulated_rows + next_chunk->size() > STANDARD_VECTOR_SIZE) {
						break;
					}
					auto popped = std::move(queue.front());
					queue.pop_front();
					exchange.buffered_bytes =
					    exchange.buffered_bytes >= popped.bytes ? exchange.buffered_bytes - popped.bytes : 0;
					auto popped_chunk = LocalExchangeMaterializedChunk(popped);
					accumulated_rows += popped_chunk ? popped_chunk->size() : 0;
					extra_chunks.push_back(std::move(popped));
				}
			}

			exchange.UnblockTasks(guard);
			picked_partition = partition_idx;
			lstate.partition_cursor = (partition_idx + 1) % exchange.num_partitions;
			break;
		}
		if (picked_partition == DConstants::INVALID_INDEX) {
			if (exchange.finished) {
				chunk.SetCardinality(0);
				return SourceResultType::FINISHED;
			}
			return exchange.BlockSource(guard, input.interrupt_state);
		}
	}

	if (LocalExchangeHasLazyChunk(entry)) {
		auto barrier = MaterializeExternalBlockBarrier(context.client, *entry.batch.lazy, GetName());
		StoreLocalExchangeMaterialized(entry, std::move(barrier.chunk), entry.bytes);
	}
	auto entry_chunk = LocalExchangeMaterializedChunk(entry);
	if (!entry_chunk) {
		chunk.SetCardinality(0);
		return SourceResultType::FINISHED;
	}

	// Merge coalesced chunks into the first chunk
	if (!extra_chunks.empty()) {
		for (auto &extra : extra_chunks) {
			auto extra_chunk = LocalExchangeMaterializedChunk(extra);
			if (extra_chunk) {
				entry_chunk->Append(*extra_chunk, true);
			}
		}
	}

	chunk.Move(*entry_chunk);
	return SourceResultType::HAVE_MORE_OUTPUT;
}

SourceResultType PhysicalLocalExchange::GetDataBatch(ExecutionContext &context, ExecutionBatch &batch,
                                                     OperatorSourceInput &input) const {
	auto &gstate = input.global_state.Cast<LocalExchangeGlobalSourceState>();
	auto &lstate = input.local_state.Cast<LocalExchangeLocalSourceState>();
	auto &exchange = *gstate.exchange_state;

	LocalExchangeChunk entry;
	idx_t picked_partition = DConstants::INVALID_INDEX;

	{
		auto guard = exchange.Lock();
		if (exchange.num_partitions == 0) {
			batch = ExecutionBatch();
			return SourceResultType::FINISHED;
		}
		for (idx_t attempt = 0; attempt < exchange.num_partitions; attempt++) {
			auto partition_idx = (lstate.partition_cursor + attempt) % exchange.num_partitions;
			auto &queue = exchange.queues[partition_idx];
			if (queue.empty()) {
				continue;
			}
			entry = std::move(queue.front());
			queue.pop_front();
			exchange.buffered_bytes =
			    exchange.buffered_bytes >= entry.bytes ? exchange.buffered_bytes - entry.bytes : 0;
			exchange.UnblockTasks(guard);
			picked_partition = partition_idx;
			lstate.partition_cursor = (partition_idx + 1) % exchange.num_partitions;
			break;
		}
		if (picked_partition == DConstants::INVALID_INDEX) {
			if (exchange.finished) {
				batch = ExecutionBatch();
				return SourceResultType::FINISHED;
			}
			batch = ExecutionBatch();
			return exchange.BlockSource(guard, input.interrupt_state);
		}
	}

	const auto rows = LocalExchangeBatchRows(entry.batch);
	if (rows == 0) {
		batch = ExecutionBatch();
		return SourceResultType::FINISHED;
	}

	const bool lazy = LocalExchangeHasLazyChunk(entry);
	batch = std::move(entry.batch);
	batch.rows = rows;
	batch.estimated_bytes = entry.bytes;
	return SourceResultType::HAVE_MORE_OUTPUT;
}

InsertionOrderPreservingMap<string> PhysicalLocalExchange::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	const auto repartition_type = repartition_spec ? repartition_spec->type() : RepartitionSpec::Type::Random;
	if (repartition_spec) {
		string spec_name = repartition_spec->var_name();
		result["repartition_type"] = spec_name;
	}
	if (repartition_type == RepartitionSpec::Type::Random ||
	    repartition_type == RepartitionSpec::Type::IntoPartitions) {
		result["lazy_ref_local_exchange"] = "whole_block";
	} else {
		result["lazy_ref_local_exchange"] = "materialize";
	}
	if (!partition_by.empty()) {
		string by;
		for (idx_t i = 0; i < partition_by.size(); i++) {
			if (i > 0) {
				by += ", ";
			}
			by += partition_by[i]->ToString();
		}
		result["partition_by"] = by;
	}
	if (exchange_state) {
		result["partitions"] = StringUtil::Format("%llu", exchange_state->num_partitions);
	}
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

void PhysicalLocalExchange::SerializeOperatorData(Serializer &serializer) const {
	uint8_t repartition_type = static_cast<uint8_t>(RepartitionSpec::Type::Random);
	bool has_num_partitions = false;
	idx_t num_partitions = 0;

	if (repartition_spec) {
		repartition_type = static_cast<uint8_t>(repartition_spec->type());
		switch (repartition_spec->type()) {
		case RepartitionSpec::Type::Hash: {
			auto *hash_spec = dynamic_cast<HashRepartitionSpec *>(repartition_spec.get());
			if (hash_spec && hash_spec->config()->num_partitions) {
				has_num_partitions = true;
				num_partitions = hash_spec->config()->num_partitions;
			}
			break;
		}
		case RepartitionSpec::Type::Random: {
			auto *random_spec = dynamic_cast<RandomRepartitionSpec *>(repartition_spec.get());
			if (random_spec && random_spec->config()->num_partitions) {
				has_num_partitions = true;
				num_partitions = random_spec->config()->num_partitions;
			}
			break;
		}
		case RepartitionSpec::Type::IntoPartitions: {
			auto *into_spec = dynamic_cast<IntoPartitionsRepartitionSpec *>(repartition_spec.get());
			if (into_spec) {
				has_num_partitions = true;
				num_partitions = static_cast<idx_t>(into_spec->config()->num_partitions);
			}
			break;
		}
		case RepartitionSpec::Type::Range:
			throw NotImplementedException("Serialization not implemented for range repartition");
		}
	}

	serializer.WriteProperty(103, "repartition_type", repartition_type);
	serializer.WriteProperty(104, "has_num_partitions", has_num_partitions);
	if (has_num_partitions) {
		serializer.WriteProperty(105, "num_partitions", num_partitions);
	}
	serializer.WriteProperty(106, "partition_by", partition_by);
}

} // namespace duckdb
