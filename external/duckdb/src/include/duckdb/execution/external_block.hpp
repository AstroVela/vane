// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/external_block.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/data_chunk.hpp"

#include <deque>
#include <functional>
#include <utility>

namespace duckdb {

class ClientContext;

enum class ExternalBlockBackend : uint8_t { RAY_OBJECT_STORE };

struct ExternalBlockSlice {
	idx_t start_offset = 0;
	idx_t end_offset = 0; // exclusive

	idx_t Count() const {
		return end_offset > start_offset ? end_offset - start_offset : 0;
	}
};

struct ExternalBlockMetadata {
	idx_t num_rows = 0;
	idx_t size_bytes = 0;
	string query_id;
	string operator_id;
	string attempt_id;
};

struct ExternalBlockDescriptor {
	ExternalBlockBackend backend = ExternalBlockBackend::RAY_OBJECT_STORE;
	shared_ptr<void> object_ref;
	// Lifetime credits associated with this block. Descriptor transforms copy
	// these shared tokens so ownership is released only after the final slice,
	// projection, queue entry, or downstream consumer drops the block.
	vector<shared_ptr<void>> ownership_tokens;
	ExternalBlockMetadata metadata;
	bool has_slice = false;
	ExternalBlockSlice slice;
	vector<idx_t> column_ids;

	idx_t StartOffset() const {
		return has_slice ? slice.start_offset : 0;
	}

	idx_t EndOffset() const {
		return has_slice ? slice.end_offset : metadata.num_rows;
	}

	idx_t RowCount() const {
		auto start = StartOffset();
		auto end = EndOffset();
		return end > start ? end - start : 0;
	}

	idx_t EstimatedBytes() const {
		if (metadata.num_rows == 0) {
			return 0;
		}
		auto rows = RowCount();
		if (rows == 0) {
			return 0;
		}
		if (metadata.size_bytes == 0) {
			return rows;
		}
		auto bytes = (metadata.size_bytes * rows) / metadata.num_rows;
		if (bytes == 0) {
			bytes = 1;
		}
		return bytes;
	}
};

struct LazyDataChunk {
	vector<ExternalBlockDescriptor> blocks;
	vector<LogicalType> logical_types;
	vector<string> names;
	idx_t cardinality = 0;
	bool wrap_columns_as_struct = false;

	void RecomputeCardinality() {
		cardinality = 0;
		for (auto &block : blocks) {
			cardinality += block.RowCount();
		}
	}

	idx_t EstimatedBytes() const {
		idx_t total = 0;
		for (auto &block : blocks) {
			total += block.EstimatedBytes();
		}
		return total;
	}

	bool Empty() const {
		return cardinality == 0 || blocks.empty();
	}
};

enum class ExecutionBatchKind : uint8_t { MATERIALIZED_CHUNK, LAZY_DATA_CHUNK, MIXED_CHUNK };

struct ExecutionBatch {
	ExecutionBatchKind kind = ExecutionBatchKind::MATERIALIZED_CHUNK;
	unique_ptr<DataChunk> materialized;
	unique_ptr<LazyDataChunk> lazy;
	idx_t rows = 0;
	idx_t estimated_bytes = 0;
};

struct ExternalBlockMaterializeStats {
	string barrier_name = "MaterializeExternalBlock";
	string reason;
	idx_t blocks = 0;
	idx_t rows = 0;
	idx_t estimated_bytes = 0;
};

struct ExternalBlockMaterializeResult {
	unique_ptr<DataChunk> chunk;
	ExternalBlockMaterializeStats stats;
};

enum class LazyOperatorBehavior : uint8_t {
	PASS_THROUGH,
	PROJECT_ONLY,
	SLICE_ONLY,
	SELECTION_ONLY,
	REF_AWARE,
	MATERIALIZE_REQUIRED
};

class ExternalBlockBackendInterface {
public:
	virtual ~ExternalBlockBackendInterface() = default;

	virtual bool CanMaterialize(const ExternalBlockDescriptor &desc) = 0;
	virtual unique_ptr<DataChunk> Materialize(ClientContext &context, const LazyDataChunk &chunk) = 0;
};

DUCKDB_API void SetExternalBlockBackend(ExternalBlockBackend backend, shared_ptr<ExternalBlockBackendInterface> impl);
DUCKDB_API shared_ptr<ExternalBlockBackendInterface> GetExternalBlockBackend(ExternalBlockBackend backend);
DUCKDB_API unique_ptr<DataChunk> MaterializeExternalBlock(ClientContext &context, const LazyDataChunk &chunk);
DUCKDB_API ExternalBlockMaterializeResult MaterializeExternalBlockBarrier(ClientContext &context,
                                                                          const LazyDataChunk &chunk,
                                                                          string reason = string());
DUCKDB_API unique_ptr<LazyDataChunk> ProjectLazyDataChunk(const LazyDataChunk &input, const vector<idx_t> &column_ids);
DUCKDB_API unique_ptr<LazyDataChunk> ProjectLazyDataChunk(const LazyDataChunk &input, const vector<idx_t> &column_ids,
                                                          const vector<string> &names);
DUCKDB_API unique_ptr<LazyDataChunk> SliceLazyDataChunk(const LazyDataChunk &input, idx_t offset, idx_t count);

class LazyDataChunkBundler {
public:
	explicit LazyDataChunkBundler(idx_t target_rows_p) : target_rows(target_rows_p) {
		if (target_rows == 0) {
			throw InvalidInputException("lazy data chunk target rows must be > 0");
		}
	}

	void Add(unique_ptr<LazyDataChunk> bundle) {
		if (!bundle) {
			return;
		}
		bundle->RecomputeCardinality();
		if (bundle->Empty()) {
			return;
		}
		ValidateSchema(*bundle);
		pending_rows += bundle->cardinality;
		pending.push_back(std::move(bundle));
		BuildReady(false);
	}

	bool HasReady() const {
		return !ready.empty();
	}

	unique_ptr<LazyDataChunk> PopReady() {
		if (ready.empty()) {
			return nullptr;
		}
		auto result = std::move(ready.front());
		ready.pop_front();
		ready_rows = ready_rows >= result->cardinality ? ready_rows - result->cardinality : 0;
		return result;
	}

	void Finish() {
		BuildReady(true);
	}

	idx_t PendingRows() const {
		return pending_rows;
	}

	idx_t ReadyRows() const {
		return ready_rows;
	}

private:
	void ValidateSchema(const LazyDataChunk &bundle) {
		if (!has_schema) {
			logical_types = bundle.logical_types;
			names = bundle.names;
			wrap_columns_as_struct = bundle.wrap_columns_as_struct;
			has_schema = true;
			return;
		}
		if (bundle.logical_types != logical_types || bundle.names != names ||
		    bundle.wrap_columns_as_struct != wrap_columns_as_struct) {
			throw InvalidInputException("lazy data chunk bundler received incompatible schemas");
		}
	}

	static ExternalBlockDescriptor TakeDescriptorPrefix(const ExternalBlockDescriptor &input, idx_t rows) {
		auto available = input.RowCount();
		if (rows > available) {
			throw InternalException("lazy data chunk descriptor split exceeds available rows");
		}
		ExternalBlockDescriptor result = input;
		auto start = input.StartOffset();
		result.has_slice = true;
		result.slice.start_offset = start;
		result.slice.end_offset = start + rows;
		return result;
	}

	static void ConsumeDescriptorPrefix(ExternalBlockDescriptor &input, idx_t rows) {
		auto available = input.RowCount();
		if (rows > available) {
			throw InternalException("lazy data chunk descriptor consume exceeds available rows");
		}
		auto start = input.StartOffset();
		auto end = input.EndOffset();
		input.has_slice = true;
		input.slice.start_offset = start + rows;
		input.slice.end_offset = end;
	}

	void BuildReady(bool allow_tail) {
		while (pending_rows >= target_rows || (allow_tail && pending_rows > 0)) {
			auto rows_to_emit = pending_rows >= target_rows ? target_rows : pending_rows;
			auto out = make_uniq<LazyDataChunk>();
			out->logical_types = logical_types;
			out->names = names;
			out->wrap_columns_as_struct = wrap_columns_as_struct;
			idx_t remaining = rows_to_emit;
			while (remaining > 0 && !pending.empty()) {
				auto &bundle = *pending.front();
				while (remaining > 0 && !bundle.blocks.empty()) {
					auto &block = bundle.blocks.front();
					auto available = block.RowCount();
					if (available == 0) {
						bundle.blocks.erase(bundle.blocks.begin());
						continue;
					}
					auto take = MinValue<idx_t>(remaining, available);
					out->blocks.push_back(TakeDescriptorPrefix(block, take));
					if (take == available) {
						bundle.blocks.erase(bundle.blocks.begin());
					} else {
						ConsumeDescriptorPrefix(block, take);
					}
					bundle.cardinality = bundle.cardinality >= take ? bundle.cardinality - take : 0;
					pending_rows = pending_rows >= take ? pending_rows - take : 0;
					remaining -= take;
				}
				if (bundle.cardinality == 0 || bundle.blocks.empty()) {
					pending.pop_front();
				}
			}
			out->RecomputeCardinality();
			if (out->cardinality != rows_to_emit) {
				throw InternalException("lazy data chunk emitted row count mismatch");
			}
			ready_rows += out->cardinality;
			ready.push_back(std::move(out));
			if (!allow_tail && pending_rows < target_rows) {
				break;
			}
		}
	}

	std::deque<unique_ptr<LazyDataChunk>> pending;
	std::deque<unique_ptr<LazyDataChunk>> ready;
	idx_t pending_rows = 0;
	idx_t ready_rows = 0;
	idx_t target_rows;
	bool has_schema = false;
	vector<LogicalType> logical_types;
	vector<string> names;
	bool wrap_columns_as_struct = false;
};

} // namespace duckdb
