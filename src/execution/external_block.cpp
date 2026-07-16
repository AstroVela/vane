// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/external_block.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/execution/external_block.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"

namespace duckdb {

namespace {

mutex external_block_backend_lock;
shared_ptr<ExternalBlockBackendInterface> ray_object_store_backend;

const char *ExternalBlockBackendName(ExternalBlockBackend backend) {
	switch (backend) {
	case ExternalBlockBackend::RAY_OBJECT_STORE:
		return "ray_object_store";
	default:
		return "unknown";
	}
}

} // namespace

void SetExternalBlockBackend(ExternalBlockBackend backend, shared_ptr<ExternalBlockBackendInterface> impl) {
	lock_guard<mutex> guard(external_block_backend_lock);
	switch (backend) {
	case ExternalBlockBackend::RAY_OBJECT_STORE:
		ray_object_store_backend = std::move(impl);
		return;
	default:
		throw NotImplementedException("unsupported external block backend");
	}
}

shared_ptr<ExternalBlockBackendInterface> GetExternalBlockBackend(ExternalBlockBackend backend) {
	lock_guard<mutex> guard(external_block_backend_lock);
	switch (backend) {
	case ExternalBlockBackend::RAY_OBJECT_STORE:
		return ray_object_store_backend;
	default:
		return nullptr;
	}
}

unique_ptr<DataChunk> MaterializeExternalBlock(ClientContext &context, const LazyDataChunk &chunk) {
	if (chunk.blocks.empty()) {
		auto result = make_uniq<DataChunk>();
		result->Initialize(context, chunk.logical_types);
		result->SetCardinality(0);
		return result;
	}

	auto backend_id = chunk.blocks[0].backend;
	auto backend = GetExternalBlockBackend(backend_id);
	if (!backend) {
		throw InvalidInputException("no external block backend registered for %s",
		                            ExternalBlockBackendName(backend_id));
	}
	for (auto &block : chunk.blocks) {
		if (block.backend != backend_id) {
			throw InvalidInputException("cannot materialize a LazyDataChunk with mixed external block backends");
		}
		if (!backend->CanMaterialize(block)) {
			throw InvalidInputException("external block backend cannot materialize this descriptor");
		}
	}
	return backend->Materialize(context, chunk);
}

ExternalBlockMaterializeResult MaterializeExternalBlockBarrier(ClientContext &context, const LazyDataChunk &chunk,
                                                               string reason) {
	ExternalBlockMaterializeResult result;
	result.stats.reason = std::move(reason);
	result.stats.blocks = chunk.blocks.size();
	result.stats.rows = chunk.cardinality;
	result.stats.estimated_bytes = chunk.EstimatedBytes();
	result.chunk = MaterializeExternalBlock(context, chunk);
	return result;
}

namespace {

bool IsIdentityProjection(const vector<idx_t> &column_ids, idx_t input_column_count) {
	if (column_ids.size() != input_column_count) {
		return false;
	}
	for (idx_t i = 0; i < column_ids.size(); i++) {
		if (column_ids[i] != i) {
			return false;
		}
	}
	return true;
}

vector<idx_t> ComposeColumnProjection(const ExternalBlockDescriptor &block, const vector<idx_t> &column_ids,
                                      idx_t input_column_count) {
	vector<idx_t> result;
	result.reserve(column_ids.size());
	for (auto column_id : column_ids) {
		if (block.column_ids.empty()) {
			result.push_back(column_id);
			continue;
		}
		if (column_id >= block.column_ids.size()) {
			throw InvalidInputException(
			    "lazy data chunk projection references column %d but block only exposes %d columns", column_id,
			    block.column_ids.size());
		}
		result.push_back(block.column_ids[column_id]);
	}
	if (IsIdentityProjection(result, input_column_count)) {
		result.clear();
	}
	return result;
}

unique_ptr<LazyDataChunk> ProjectLazyDataChunkInternal(const LazyDataChunk &input, const vector<idx_t> &column_ids,
                                                       const vector<string> *names) {
	if (column_ids.empty() && !input.logical_types.empty()) {
		throw InvalidInputException("lazy data chunk zero-column projection is not supported");
	}
	if (names && names->size() != column_ids.size()) {
		throw InvalidInputException("lazy data chunk projection names size %d does not match column count %d",
		                            names->size(), column_ids.size());
	}

	auto output = make_uniq<LazyDataChunk>();
	output->wrap_columns_as_struct = input.wrap_columns_as_struct;
	output->logical_types.reserve(column_ids.size());
	output->names.reserve(column_ids.size());
	for (idx_t projected_idx = 0; projected_idx < column_ids.size(); projected_idx++) {
		auto column_id = column_ids[projected_idx];
		if (column_id >= input.logical_types.size()) {
			throw InvalidInputException("lazy data chunk projection references column %d but input has %d columns",
			                            column_id, input.logical_types.size());
		}
		output->logical_types.push_back(input.logical_types[column_id]);
		if (names) {
			output->names.push_back((*names)[projected_idx]);
		} else if (column_id < input.names.size()) {
			output->names.push_back(input.names[column_id]);
		} else {
			output->names.push_back(StringUtil::Format("c%d", projected_idx));
		}
	}

	output->blocks.reserve(input.blocks.size());
	for (auto &block : input.blocks) {
		auto projected = block;
		if (!input.wrap_columns_as_struct) {
			projected.column_ids = ComposeColumnProjection(block, column_ids, input.logical_types.size());
		}
		output->blocks.push_back(std::move(projected));
	}
	output->RecomputeCardinality();
	return output;
}

ExternalBlockDescriptor SliceDescriptor(const ExternalBlockDescriptor &input, idx_t relative_offset, idx_t count) {
	if (relative_offset > input.RowCount() || count > input.RowCount() - relative_offset) {
		throw InvalidInputException("lazy data chunk descriptor slice is outside the block bounds");
	}
	auto output = input;
	output.has_slice = true;
	output.slice.start_offset = input.StartOffset() + relative_offset;
	output.slice.end_offset = output.slice.start_offset + count;
	return output;
}

} // namespace

unique_ptr<LazyDataChunk> ProjectLazyDataChunk(const LazyDataChunk &input, const vector<idx_t> &column_ids) {
	return ProjectLazyDataChunkInternal(input, column_ids, nullptr);
}

unique_ptr<LazyDataChunk> ProjectLazyDataChunk(const LazyDataChunk &input, const vector<idx_t> &column_ids,
                                               const vector<string> &names) {
	return ProjectLazyDataChunkInternal(input, column_ids, &names);
}

unique_ptr<LazyDataChunk> SliceLazyDataChunk(const LazyDataChunk &input, idx_t offset, idx_t count) {
	if (offset > input.cardinality || count > input.cardinality - offset) {
		throw InvalidInputException("lazy data chunk slice [%d, %d) is outside cardinality %d", offset, offset + count,
		                            input.cardinality);
	}

	auto output = make_uniq<LazyDataChunk>();
	output->logical_types = input.logical_types;
	output->names = input.names;
	output->wrap_columns_as_struct = input.wrap_columns_as_struct;
	if (count == 0) {
		output->cardinality = 0;
		return output;
	}

	idx_t rows_to_skip = offset;
	idx_t rows_remaining = count;
	for (auto &block : input.blocks) {
		auto block_rows = block.RowCount();
		if (rows_to_skip >= block_rows) {
			rows_to_skip -= block_rows;
			continue;
		}
		auto take = MinValue<idx_t>(rows_remaining, block_rows - rows_to_skip);
		output->blocks.push_back(SliceDescriptor(block, rows_to_skip, take));
		rows_remaining -= take;
		rows_to_skip = 0;
		if (rows_remaining == 0) {
			break;
		}
	}
	output->RecomputeCardinality();
	if (output->cardinality != count) {
		throw InternalException("lazy data chunk slice emitted row count mismatch");
	}
	return output;
}

} // namespace duckdb
