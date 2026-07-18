// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/vllm_executor.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/optional_ptr.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/main/client_context.hpp"

namespace duckdb {

struct VLLMConfig {
	bool do_prefix_routing = true;
	idx_t max_buffer_size = 5000;
	idx_t min_bucket_size = 16;
	double prefix_match_threshold = 0.33;
	idx_t load_balance_threshold = 32;
	optional_idx batch_size {128};
	idx_t inflight_limit = 128;
};

struct VLLMResult {
	vector<string> outputs;
	vector<bool> outputs_validity;
	unique_ptr<DataChunk> rows;
};

class VLLMExecutor {
public:
	virtual ~VLLMExecutor() = default;

	virtual void Submit(optional_ptr<const string> prefix, vector<string> prompts, DataChunk &rows,
	                    ClientContext &context) = 0;
	virtual std::pair<bool, VLLMResult> TakeReadyResult(ClientContext &context) = 0;
	virtual void FinishedSubmitting(ClientContext &context) = 0;
	virtual bool AllTasksFinished(ClientContext &context) = 0;
	virtual void Shutdown() = 0;

	//! Block until at least one result is available, an error occurred, or all tasks finished.
	//! Uses event-driven wakeup.
	virtual void WaitForResult(ClientContext &context) = 0;
};

using vllm_executor_factory_t = unique_ptr<VLLMExecutor> (*)(ClientContext &context, const string &model,
                                                             const Value &options, VLLMConfig &config);

DUCKDB_API void SetVLLMExecutorFactory(vllm_executor_factory_t factory);
DUCKDB_API vllm_executor_factory_t GetVLLMExecutorFactory();

} // namespace duckdb
