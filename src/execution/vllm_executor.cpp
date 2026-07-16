// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/vllm_executor.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/execution/vllm_executor.hpp"

namespace duckdb {

static vllm_executor_factory_t vllm_executor_factory = nullptr;

void SetVLLMExecutorFactory(vllm_executor_factory_t factory) {
	vllm_executor_factory = factory;
}

vllm_executor_factory_t GetVLLMExecutorFactory() {
	return vllm_executor_factory;
}

} // namespace duckdb
