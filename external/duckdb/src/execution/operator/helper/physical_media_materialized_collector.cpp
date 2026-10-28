// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/operator/helper/physical_media_materialized_collector.hpp"

#include "duckdb/execution/media_buffer.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/storage/buffer_manager.hpp"

namespace duckdb {

namespace {

unique_ptr<ColumnDataCollection> CreateMediaCollection(ClientContext &context, const vector<LogicalType> &types) {
	return make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(*context.db), types,
	                                       ColumnDataCollectionLifetime::THROW_ERROR_AFTER_DATABASE_CLOSES);
}

class MediaMaterializedCollectorState : public GlobalSinkState {
public:
	shared_ptr<ClientContext> context;
	unique_ptr<ColumnDataCollection> collection;
	idx_t copy_limit;
};

struct MediaResultContext {
	explicit MediaResultContext(shared_ptr<ClientContext> context_p) : context(std::move(context_p)) {
	}
	shared_ptr<ClientContext> context;
};

// The context base is destroyed last, after MaterializedQueryResult has released
// its collection and its optional GetValue() scan state (which can retain pins).
class MediaMaterializedQueryResult : private MediaResultContext, public MaterializedQueryResult {
public:
	MediaMaterializedQueryResult(StatementType statement_type, StatementProperties properties, vector<string> names,
	                             unique_ptr<ColumnDataCollection> collection, shared_ptr<ClientContext> context_p)
	    : MediaResultContext(context_p),
	      MaterializedQueryResult(statement_type, std::move(properties), std::move(names), std::move(collection),
	                              context_p->GetClientProperties()) {
	}

protected:
	unique_ptr<DataChunk> FetchInternal() override {
		auto &collection = Collection();
		if (!collection.Count()) {
			return nullptr;
		}
		auto result = make_uniq<DataChunk>();
		// A fetched chunk can outlive the result and its database. The storage
		// allocator belongs to BufferManager; the returned copy must not use it.
		collection.InitializeScanChunk(Allocator::DefaultAllocator(), *result);
		if (!scan_initialized) {
			collection.InitializeScan(scan_state, ColumnDataScanProperties::DISALLOW_ZERO_COPY);
			scan_initialized = true;
		}
		collection.Scan(scan_state, *result);
		if (!result->size()) {
			return nullptr;
		}
		return result;
	}

private:
	ColumnDataScanState scan_state;
	bool scan_initialized = false;
};

} // namespace

PhysicalMediaMaterializedCollector::PhysicalMediaMaterializedCollector(PhysicalPlan &physical_plan,
                                                                       PreparedStatementData &data)
    : PhysicalResultCollector(physical_plan, data) {
}

unique_ptr<GlobalSinkState> PhysicalMediaMaterializedCollector::GetGlobalSinkState(ClientContext &context) const {
	auto state = make_uniq<MediaMaterializedCollectorState>();
	state->context = context.shared_from_this();
	state->copy_limit = ClientConfig::GetConfig(context).streaming_buffer_size;
	state->collection = CreateMediaCollection(context, types);
	return std::move(state);
}

SinkResultType PhysicalMediaMaterializedCollector::Sink(ExecutionContext &context, DataChunk &chunk,
                                                        OperatorSinkInput &input) const {
	auto &state = input.global_state.Cast<MediaMaterializedCollectorState>();
	// Do not publish a partial input or acknowledge it if any allocation/copy fails.
	auto pending = CreateMediaCollection(context.client, types);
	idx_t offset = 0;
	while (offset < chunk.size()) {
		if (context.client.interrupted) {
			throw InterruptException();
		}
		const auto row_bytes = MaxValue<idx_t>(1, EstimateMediaFlowInputBytes(chunk, offset, 1));
		if (row_bytes > state.copy_limit) {
			throw OutOfMemoryException("Native media materialized result row requires %llu payload bytes, exceeding "
			                           "streaming_buffer_size receiving capacity of %llu bytes",
			                           static_cast<unsigned long long>(row_bytes),
			                           static_cast<unsigned long long>(state.copy_limit));
		}
		auto rows = MediaFlowRowsWithinByteLimit(chunk, offset, chunk.size() - offset, state.copy_limit);
		D_ASSERT(rows);
		unique_ptr<ColumnDataCollection> segment;
		while (true) {
			if (context.client.interrupted) {
				throw InterruptException();
			}
			try {
				DataChunk selected;
				selected.InitializeEmpty(types);
				selected.Reference(chunk);
				selected.Slice(offset, rows);
				auto candidate = CreateMediaCollection(context.client, types);
				// Append obtains actual BufferManager blocks before writing. Its local
				// state releases every pin on return. Seal this prefix instead of
				// retaining a partially filled 2048-row image chunk until the next input.
				// Complex-vector flattening has at most copy_limit selected payload.
				candidate->Append(selected);
				segment = std::move(candidate);
				break;
			} catch (const OutOfMemoryException &) {
				// Payload capacity and the pinned storage working set are different
				// limits. Release the failed copy before retrying a smaller prefix.
				// A single row that cannot fit is a permanent error, not a wait or
				// a one-chunk capacity escape. Each retry strictly reduces rows.
				if (rows == 1) {
					throw;
				}
				rows = MaxValue<idx_t>(1, rows / 2);
			}
		}
		pending->Combine(*segment);
		offset += rows;
	}
	state.collection->Combine(*pending);
	// Receipts may synchronously wake/re-enter producers. All target allocation,
	// copying and publication is complete, and no storage pins/locks are held.
	chunk.AcknowledgeMediaFlow();
	return SinkResultType::NEED_MORE_INPUT;
}

unique_ptr<QueryResult> PhysicalMediaMaterializedCollector::GetResult(GlobalSinkState &state) const {
	auto &gstate = state.Cast<MediaMaterializedCollectorState>();
	return make_uniq<MediaMaterializedQueryResult>(statement_type, properties, names, std::move(gstate.collection),
	                                               gstate.context);
}

} // namespace duckdb
