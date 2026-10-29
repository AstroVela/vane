// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/common/sorting/media_sort.hpp"

#include "duckdb/common/sorting/sorted_run.hpp"
#include "duckdb/common/sorting/sorted_run_merger.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/media_buffer.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/storage/buffer_manager.hpp"

#include <algorithm>
#include <iterator>

namespace duckdb {

namespace {

template <class T>
void ReserveMediaSortEntries(vector<T> &entries, idx_t extra) {
	const auto required = entries.size() + extra;
	if (required > entries.capacity()) {
		entries.reserve(MaxValue<idx_t>(required, entries.capacity() * 2));
	}
}

unique_ptr<ColumnDataCollection> CreateMediaSortStorage(ClientContext &context, const vector<LogicalType> &types) {
	return make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(context), types);
}

struct RetireMediaKeyRuns {
	explicit RetireMediaKeyRuns(SortedRunMerger &merger_p) : merger(merger_p) {
	}
	~RetireMediaKeyRuns() {
		merger.sorted_runs.clear();
	}
	SortedRunMerger &merger;
};

} // namespace

class MediaSortGlobalSinkState : public GlobalSinkState {
public:
	explicit MediaSortGlobalSinkState(idx_t copy_limit_p) : copy_limit(copy_limit_p) {
	}

	const idx_t copy_limit;
	idx_t count = 0;
	vector<idx_t> first_rows;
	vector<unique_ptr<ColumnDataCollection>> storage;
	vector<unique_ptr<SortedRun>> key_runs;
	unique_ptr<ColumnDataCollection> ordered_ids;
};

class MediaSortLocalSinkState : public LocalSinkState {
public:
	MediaSortLocalSinkState(ClientContext &context, Expression &encoder, const vector<LogicalType> &key_input_types)
	    : key_executor(context, encoder) {
		// A cached chunk restores owned vectors on Reset, releasing every
		// borrowed input reference before acknowledging and polling the source.
		key_input.Initialize(context, key_input_types);
		key.Initialize(context, {encoder.return_type});
		row_ids.Initialize(context, {LogicalType::UBIGINT});
	}
	void Reset() {
		key_input.Reset();
		key.Reset();
		row_ids.Reset();
		key_executor.ResetInput();
	}

	ExpressionExecutor key_executor;
	DataChunk key_input;
	DataChunk key;
	DataChunk row_ids;
};

MediaSort::MediaSort(ClientContext &context, const vector<BoundOrderByNode> &orders, vector<LogicalType> input_types_p,
                     vector<idx_t> projections_p)
    : input_types(std::move(input_types_p)), projections(std::move(projections_p)), staging_key_bytes(0) {
	if (projections.empty()) {
		for (idx_t i = 0; i < input_types.size(); i++) {
			projections.push_back(i);
		}
	}
	vector<BoundOrderByNode> key_orders;
	idx_t encoded_width = 0;
	for (auto &order : orders) {
		if (order.expression->GetExpressionClass() != ExpressionClass::BOUND_REF) {
			throw NotImplementedException("Native media sorting requires projected fixed-width sort keys");
		}
		const auto &ref = order.expression->Cast<BoundReferenceExpression>();
		const auto &type = ref.return_type;
		switch (type.InternalType()) {
		case PhysicalType::BOOL:
		case PhysicalType::INT8:
		case PhysicalType::INT16:
		case PhysicalType::INT32:
		case PhysicalType::INT64:
		case PhysicalType::INT128:
		case PhysicalType::UINT8:
		case PhysicalType::UINT16:
		case PhysicalType::UINT32:
		case PhysicalType::UINT64:
		case PhysicalType::UINT128:
		case PhysicalType::FLOAT:
		case PhysicalType::DOUBLE:
		case PhysicalType::INTERVAL:
			break;
		default:
			throw NotImplementedException("Native media sorting requires fixed-width primitive sort keys; got %s",
			                              type.ToString());
		}
		key_orders.emplace_back(order.type, order.null_order,
		                        make_uniq<BoundReferenceExpression>(type, key_columns.size()));
		key_columns.push_back(ref.index);
		key_input_types.push_back(type);
		encoded_width += GetTypeIdSize(type.InternalType()) + 1;
	}
	key_input_types.push_back(LogicalType::UBIGINT);
	key_sort = make_uniq<Sort>(context, key_orders, key_input_types, vector<idx_t> {key_columns.size()});
	// Conservative pre-encoding bound: fixed-width values plus validity bytes,
	// the encoded vector's string slot (if any), and the independent row id.
	if (key_sort->create_sort_key->return_type.id() == LogicalTypeId::BLOB) {
		staging_key_bytes = sizeof(uint64_t) + sizeof(string_t);
		if (encoded_width > string_t::INLINE_LENGTH) {
			staging_key_bytes += encoded_width;
		}
	} else {
		staging_key_bytes = 2 * sizeof(uint64_t);
	}
}

MediaSort::~MediaSort() {
}

unique_ptr<GlobalSinkState> MediaSort::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<MediaSortGlobalSinkState>(ClientConfig::GetConfig(context).streaming_buffer_size);
}

unique_ptr<LocalSinkState> MediaSort::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<MediaSortLocalSinkState>(context.client, *key_sort->create_sort_key, key_input_types);
}

SinkResultType MediaSort::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &state = input.global_state.Cast<MediaSortGlobalSinkState>();
	auto &local = input.local_state.Cast<MediaSortLocalSinkState>();
	vector<unique_ptr<ColumnDataCollection>> storage;
	vector<unique_ptr<SortedRun>> runs;
	vector<idx_t> first_rows;
	idx_t offset = 0;
	while (offset < chunk.size()) {
		const auto row_payload = EstimateMediaFlowInputBytes(chunk, offset, 1);
		if (staging_key_bytes > state.copy_limit || row_payload > state.copy_limit - staging_key_bytes) {
			throw OutOfMemoryException("Native media sort row and encoded keys exceed streaming_buffer_size "
			                           "receiving capacity of %llu bytes",
			                           static_cast<unsigned long long>(state.copy_limit));
		}
		idx_t lower = 1;
		idx_t upper = MinValue<idx_t>(STANDARD_VECTOR_SIZE, chunk.size() - offset);
		while (lower < upper) {
			const auto middle = lower + (upper - lower + 1) / 2;
			const auto bytes = EstimateMediaFlowInputBytes(chunk, offset, middle);
			if (middle <= state.copy_limit / staging_key_bytes &&
			    bytes <= state.copy_limit - middle * staging_key_bytes) {
				lower = middle;
			} else {
				upper = middle - 1;
			}
		}
		auto rows = lower;
		unique_ptr<ColumnDataCollection> segment;
		unique_ptr<SortedRun> run;
		while (true) {
			if (context.client.interrupted) {
				throw InterruptException();
			}
			try {
				DataChunk selected;
				selected.InitializeEmpty(input_types);
				selected.Reference(chunk);
				selected.Slice(offset, rows);
				auto candidate = CreateMediaSortStorage(context.client, input_types);
				candidate->Append(selected); // actual allocation; append pins die here
				for (idx_t i = 0; i < key_columns.size(); i++) {
					local.key_input.data[i].Reference(selected.data[key_columns[i]]);
				}
				auto ids = FlatVector::GetData<uint64_t>(local.row_ids.data[0]);
				for (idx_t row = 0; row < rows; row++) {
					ids[row] = state.count + offset + row;
				}
				local.row_ids.SetCardinality(rows);
				local.key_input.data.back().Reference(local.row_ids.data[0]);
				local.key_input.SetCardinality(rows);
				local.key_executor.Execute(local.key_input, local.key);
				auto candidate_run = make_uniq<SortedRun>(context.client, *key_sort, false);
				candidate_run->Sink(local.key, local.row_ids);
				candidate_run->Finalize(true); // sort and unpin key/id storage
				local.Reset();
				segment = std::move(candidate);
				run = std::move(candidate_run);
				break;
			} catch (const OutOfMemoryException &) {
				local.Reset();
				if (rows == 1) {
					throw;
				}
				rows = MaxValue<idx_t>(1, rows / 2);
			} catch (...) {
				local.Reset();
				throw;
			}
		}
		first_rows.push_back(state.count + offset);
		storage.push_back(std::move(segment));
		runs.push_back(std::move(run));
		offset += rows;
	}
	// All possible directory allocations precede publication of this input.
	ReserveMediaSortEntries(state.storage, storage.size());
	ReserveMediaSortEntries(state.key_runs, runs.size());
	ReserveMediaSortEntries(state.first_rows, first_rows.size());
	state.storage.insert(state.storage.end(), std::make_move_iterator(storage.begin()),
	                     std::make_move_iterator(storage.end()));
	state.key_runs.insert(state.key_runs.end(), std::make_move_iterator(runs.begin()),
	                      std::make_move_iterator(runs.end()));
	state.first_rows.insert(state.first_rows.end(), first_rows.begin(), first_rows.end());
	state.count += chunk.size();
	// Both domains own spillable copies. A callback can re-enter after commit.
	chunk.AcknowledgeMediaFlow();
	return SinkResultType::NEED_MORE_INPUT;
}

SinkFinalizeType MediaSort::Finalize(ClientContext &context, OperatorSinkFinalizeInput &input) const {
	auto &state = input.global_state.Cast<MediaSortGlobalSinkState>();
	if (!state.count) {
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}
	ThreadContext thread(context);
	ExecutionContext execution(context, thread, nullptr);
	// Pairwise passes bound the merge fan-in. Pixels never enter these runs.
	while (state.key_runs.size() > 1) {
		vector<unique_ptr<SortedRun>> next;
		next.reserve((state.key_runs.size() + 1) / 2);
		for (idx_t i = 0; i < state.key_runs.size(); i += 2) {
			if (context.interrupted) {
				throw InterruptException();
			}
			if (i + 1 == state.key_runs.size()) {
				next.push_back(std::move(state.key_runs[i]));
				continue;
			}
			vector<unique_ptr<SortedRun>> pair;
			pair.push_back(std::move(state.key_runs[i]));
			pair.push_back(std::move(state.key_runs[i + 1]));
			SortedRunMerger merger(*key_sort, std::move(pair), STANDARD_VECTOR_SIZE, true, false);
			RetireMediaKeyRuns retire(merger);
			auto global = merger.GetGlobalSourceState(context);
			auto local = merger.GetLocalSourceState(execution, *global);
			OperatorSourceInput source {*global, *local, input.interrupt_state};
			const auto result = merger.MaterializeSortedRun(execution, source);
			if (result != SourceResultType::FINISHED) {
				throw InternalException("Serial native media key merge did not finish");
			}
			next.push_back(merger.GetSortedRun(*global));
			local.reset();
			global.reset();
			// Retire synchronously, before the next pair can allocate its blocks.
			merger.sorted_runs.clear();
		}
		state.key_runs = std::move(next);
	}
	// Materialize the small ordered id stream before looking up pixels. This
	// retires every merge pin, leaving the pixel reader its full working set.
	SortedRunMerger merger(*key_sort, std::move(state.key_runs), STANDARD_VECTOR_SIZE, true, false);
	RetireMediaKeyRuns retire(merger);
	auto global = merger.GetGlobalSourceState(context);
	auto local = merger.GetLocalSourceState(execution, *global);
	OperatorSourceInput source {*global, *local, input.interrupt_state};
	auto ordered_ids = CreateMediaSortStorage(context, {LogicalType::UBIGINT});
	ColumnDataAppendState append;
	ordered_ids->InitializeAppend(append);
	DataChunk ids;
	ids.Initialize(Allocator::DefaultAllocator(), {LogicalType::UBIGINT});
	while (true) {
		if (context.interrupted) {
			throw InterruptException();
		}
		ids.Reset();
		merger.GetData(execution, ids, source);
		if (!ids.size()) {
			break;
		}
		ordered_ids->Append(append, ids);
	}
	D_ASSERT(ordered_ids->Count() == state.count);
	state.ordered_ids = std::move(ordered_ids);
	return SinkFinalizeType::READY;
}

ProgressData MediaSort::GetSinkProgress(GlobalSinkState &, ProgressData source_progress) const {
	return source_progress;
}

class MediaSortGlobalSourceState : public GlobalSourceState {
public:
	explicit MediaSortGlobalSourceState(MediaSortGlobalSinkState &sink_p) : sink(sink_p) {
	}
	idx_t MaxThreads() override {
		return 1;
	}
	MediaSortGlobalSinkState &sink;
	atomic<idx_t> emitted {0};
};

class MediaSortLocalSourceState : public LocalSourceState {
public:
	MediaSortLocalSourceState(MediaSortGlobalSourceState &global, const vector<LogicalType> &input_types) {
		ids.Initialize(Allocator::DefaultAllocator(), {LogicalType::UBIGINT});
		stored.Initialize(Allocator::DefaultAllocator(), input_types);
		if (global.sink.ordered_ids) {
			global.sink.ordered_ids->InitializeScan(ids_scan, ColumnDataScanProperties::DISALLOW_ZERO_COPY);
		}
	}
	ColumnDataScanState ids_scan;
	DataChunk ids;
	idx_t id_offset = 0;
	DataChunk stored;
	ColumnDataScanState scan;
	idx_t stored_segment = DConstants::INVALID_INDEX;
};

unique_ptr<GlobalSourceState> MediaSort::GetGlobalSourceState(ClientContext &context, GlobalSinkState &state) const {
	return make_uniq<MediaSortGlobalSourceState>(state.Cast<MediaSortGlobalSinkState>());
}

unique_ptr<LocalSourceState> MediaSort::GetLocalSourceState(ExecutionContext &context, GlobalSourceState &state) const {
	return make_uniq<MediaSortLocalSourceState>(state.Cast<MediaSortGlobalSourceState>(), input_types);
}

SourceResultType MediaSort::GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const {
	auto &global = input.global_state.Cast<MediaSortGlobalSourceState>();
	auto &local = input.local_state.Cast<MediaSortLocalSourceState>();
	if (context.client.interrupted) {
		throw InterruptException();
	}
	if (local.id_offset == local.ids.size()) {
		local.ids.Reset();
		if (global.sink.ordered_ids) {
			global.sink.ordered_ids->Scan(local.ids_scan, local.ids);
			local.ids_scan.current_chunk_state.handles.clear();
		}
		local.id_offset = 0;
		if (!local.ids.size()) {
			local.stored.Reset();
			local.scan.current_chunk_state.handles.clear();
			return SourceResultType::FINISHED;
		}
	}
	const auto id = local.ids.GetValue(0, local.id_offset++).GetValue<uint64_t>();
	const auto &starts = global.sink.first_rows;
	const auto segment = idx_t(std::upper_bound(starts.begin(), starts.end(), id) - starts.begin() - 1);
	if (segment != local.stored_segment) {
		local.stored.Reset();
		local.scan = ColumnDataScanState();
		auto &storage = *global.sink.storage[segment];
		storage.InitializeScan(local.scan, ColumnDataScanProperties::DISALLOW_ZERO_COPY);
		if (!storage.Scan(local.scan, local.stored)) {
			throw InternalException("Missing sealed native media sort storage");
		}
		// DISALLOW_ZERO_COPY produced an independent bounded cache prefix.
		local.scan.current_chunk_state.handles.clear();
		local.stored_segment = segment;
	}
	// One row avoids gathering a vector of large images before the receiver has
	// capacity. The cached storage prefix has at most copy_limit input payload.
	chunk.ReferenceColumns(local.stored, projections);
	chunk.Slice(id - starts[segment], 1);
	global.emitted++;
	return SourceResultType::HAVE_MORE_OUTPUT;
}

ProgressData MediaSort::GetProgress(GlobalSourceState &state) const {
	auto &global = state.Cast<MediaSortGlobalSourceState>();
	ProgressData result;
	result.done = static_cast<double>(global.emitted.load());
	result.total = static_cast<double>(global.sink.count);
	result.invalid = false;
	return result;
}

} // namespace duckdb
