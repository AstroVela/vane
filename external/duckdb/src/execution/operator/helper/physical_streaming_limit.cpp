// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/execution/operator/helper/physical_streaming_limit.hpp"
#include "duckdb/execution/external_block.hpp"
#include "duckdb/execution/operator/helper/physical_limit.hpp"
#include "duckdb/common/serializer/serializer.hpp"

namespace duckdb {

PhysicalStreamingLimit::PhysicalStreamingLimit(PhysicalPlan &physical_plan, vector<LogicalType> types,
                                               BoundLimitNode limit_val_p, BoundLimitNode offset_val_p,
                                               idx_t estimated_cardinality, bool parallel)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::STREAMING_LIMIT, std::move(types), estimated_cardinality),
      limit_val(std::move(limit_val_p)), offset_val(std::move(offset_val_p)), parallel(parallel) {
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
class StreamingLimitOperatorState : public OperatorState {
public:
	explicit StreamingLimitOperatorState(const PhysicalStreamingLimit &op) {
		PhysicalLimit::SetInitialLimits(op.limit_val, op.offset_val, limit, offset);
	}

	optional_idx limit;
	optional_idx offset;
};

class StreamingLimitGlobalState : public GlobalOperatorState {
public:
	StreamingLimitGlobalState() : current_offset(0) {
	}

	std::atomic<idx_t> current_offset;
};

unique_ptr<OperatorState> PhysicalStreamingLimit::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<StreamingLimitOperatorState>(*this);
}

unique_ptr<GlobalOperatorState> PhysicalStreamingLimit::GetGlobalOperatorState(ClientContext &context) const {
	return make_uniq<StreamingLimitGlobalState>();
}

OperatorResultType PhysicalStreamingLimit::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                   GlobalOperatorState &gstate_p, OperatorState &state_p) const {
	auto &gstate = gstate_p.Cast<StreamingLimitGlobalState>();
	auto &state = state_p.Cast<StreamingLimitOperatorState>();
	auto &limit = state.limit;
	auto &offset = state.offset;
	idx_t current_offset = gstate.current_offset.fetch_add(input.size());
	idx_t max_element;
	if (!PhysicalLimit::ComputeOffset(context, input, limit, offset, current_offset, max_element, limit_val,
	                                  offset_val)) {
		return OperatorResultType::FINISHED;
	}
	if (PhysicalLimit::HandleOffset(input, current_offset, offset.GetIndex(), limit.GetIndex())) {
		chunk.Reference(input);
	}
	if (current_offset >= limit.GetIndex() + offset.GetIndex()) {
		return chunk.size() == 0 ? OperatorResultType::FINISHED : OperatorResultType::HAVE_MORE_OUTPUT;
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

namespace {

idx_t StreamingLimitBatchSize(const ExecutionBatch &batch) {
	if (batch.kind == ExecutionBatchKind::MATERIALIZED_CHUNK) {
		return batch.materialized ? batch.materialized->size() : batch.rows;
	}
	if (batch.kind == ExecutionBatchKind::LAZY_DATA_CHUNK) {
		return batch.lazy ? batch.lazy->cardinality : batch.rows;
	}
	return batch.rows;
}

void StoreStreamingLimitLazyBatch(ExecutionBatch &output, unique_ptr<LazyDataChunk> lazy) {
	output = ExecutionBatch();
	output.kind = ExecutionBatchKind::LAZY_DATA_CHUNK;
	if (lazy) {
		lazy->RecomputeCardinality();
		output.rows = lazy->cardinality;
		output.estimated_bytes = lazy->EstimatedBytes();
	}
	output.lazy = std::move(lazy);
}

void StoreStreamingLimitEmptyBatch(ClientContext &context, const vector<LogicalType> &types, ExecutionBatch &output) {
	output = ExecutionBatch();
	output.kind = ExecutionBatchKind::MATERIALIZED_CHUNK;
	output.materialized = make_uniq<DataChunk>();
	output.materialized->Initialize(BufferAllocator::Get(context), types);
	output.materialized->SetCardinality(0);
}

bool HandleLazyStreamingLimit(const LazyDataChunk &input, idx_t &current_offset, idx_t offset, idx_t limit,
                              unique_ptr<LazyDataChunk> &output) {
	idx_t max_element = limit + offset;
	if (limit == DConstants::INVALID_INDEX) {
		max_element = DConstants::INVALID_INDEX;
	}
	auto input_size = input.cardinality;
	if (current_offset < offset) {
		if (current_offset + input_size > offset) {
			auto start_position = offset - current_offset;
			auto chunk_count = MinValue<idx_t>(limit, input_size - start_position);
			output = SliceLazyDataChunk(input, start_position, chunk_count);
		} else {
			current_offset += input_size;
			return false;
		}
	} else {
		idx_t chunk_count;
		if (current_offset + input_size >= max_element) {
			chunk_count = max_element - current_offset;
		} else {
			chunk_count = input_size;
		}
		output = SliceLazyDataChunk(input, 0, chunk_count);
	}
	current_offset += input_size;
	return true;
}

} // namespace

OperatorResultType PhysicalStreamingLimit::ExecuteBatch(ExecutionContext &context, ExecutionBatch &input,
                                                        ExecutionBatch &output, GlobalOperatorState &gstate_p,
                                                        OperatorState &state_p) const {
	if (input.kind != ExecutionBatchKind::LAZY_DATA_CHUNK || !input.lazy) {
		return PhysicalOperator::ExecuteBatch(context, input, output, gstate_p, state_p);
	}

	auto &gstate = gstate_p.Cast<StreamingLimitGlobalState>();
	auto &state = state_p.Cast<StreamingLimitOperatorState>();
	auto &limit = state.limit;
	auto &offset = state.offset;
	if (!limit.IsValid() || !offset.IsValid()) {
		return PhysicalOperator::ExecuteBatch(context, input, output, gstate_p, state_p);
	}

	input.lazy->RecomputeCardinality();
	idx_t current_offset = gstate.current_offset.fetch_add(input.lazy->cardinality);
	auto max_element = limit.GetIndex() + offset.GetIndex();
	if (limit == 0 || current_offset >= max_element) {
		StoreStreamingLimitEmptyBatch(context.client, types, output);
		return OperatorResultType::FINISHED;
	}

	unique_ptr<LazyDataChunk> lazy_output;
	if (HandleLazyStreamingLimit(*input.lazy, current_offset, offset.GetIndex(), limit.GetIndex(), lazy_output)) {
		StoreStreamingLimitLazyBatch(output, std::move(lazy_output));
	} else {
		StoreStreamingLimitEmptyBatch(context.client, types, output);
	}
	if (current_offset >= max_element && StreamingLimitBatchSize(output) == 0) {
		return OperatorResultType::FINISHED;
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

OrderPreservationType PhysicalStreamingLimit::OperatorOrder() const {
	return OrderPreservationType::FIXED_ORDER;
}

bool PhysicalStreamingLimit::ParallelOperator() const {
	return parallel;
}

void PhysicalStreamingLimit::SerializeOperatorData(Serializer &serializer) const {
	serializer.WriteProperty(103, "limit_val", limit_val);
	serializer.WriteProperty(104, "offset_val", offset_val);
	serializer.WriteProperty(105, "parallel", parallel);
}

} // namespace duckdb
