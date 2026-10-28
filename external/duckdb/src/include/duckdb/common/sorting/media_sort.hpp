// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/sorting/sort.hpp"

namespace duckdb {

//! Keep media in sealed spillable columns; sort only fixed-width keys and row ids.
class MediaSort {
public:
	MediaSort(ClientContext &context, const vector<BoundOrderByNode> &orders, vector<LogicalType> input_types,
	          vector<idx_t> projections);
	~MediaSort();

	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const;
	SinkFinalizeType Finalize(ClientContext &context, OperatorSinkFinalizeInput &input) const;
	ProgressData GetSinkProgress(GlobalSinkState &state, ProgressData source_progress) const;

	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context, GlobalSinkState &state) const;
	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context, GlobalSourceState &state) const;
	SourceResultType GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const;
	ProgressData GetProgress(GlobalSourceState &state) const;

private:
	vector<LogicalType> input_types;
	vector<idx_t> projections;
	vector<idx_t> key_columns;
	vector<LogicalType> key_input_types;
	unique_ptr<Sort> key_sort;
	idx_t staging_key_bytes;
};

} // namespace duckdb
