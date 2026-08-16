// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/vllm_executor.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/execution/vllm_executor.hpp"

#include "duckdb/common/types.hpp"

#include <mutex>
#include <unordered_map>

namespace duckdb {

static std::mutex vllm_executor_registry_lock;
static std::unordered_map<string, vllm_executor_factory_t> vllm_executor_registry;

void RegisterEngineExecutorFactory(const string &engine, vllm_executor_factory_t factory) {
	std::lock_guard<std::mutex> lock(vllm_executor_registry_lock);
	vllm_executor_registry[engine] = factory;
}

vllm_executor_factory_t GetEngineExecutorFactory(const string &engine) {
	std::lock_guard<std::mutex> lock(vllm_executor_registry_lock);
	auto entry = vllm_executor_registry.find(engine);
	if (entry == vllm_executor_registry.end()) {
		return nullptr;
	}
	return entry->second;
}

string GetInferenceEngineName(const Value &options) {
	// Envelopes produced before engine dispatch existed carry no engine field.
	if (options.IsNull() || options.type().id() != LogicalTypeId::STRUCT) {
		return "vllm";
	}
	const auto &children = StructValue::GetChildren(options);
	const auto child_count = StructType::GetChildCount(options.type());
	for (idx_t i = 0; i < child_count; i++) {
		if (StructType::GetChildName(options.type(), i) != "engine") {
			continue;
		}
		const auto &value = children[i];
		if (!value.IsNull() && value.type().id() == LogicalTypeId::VARCHAR) {
			return StringValue::Get(value);
		}
		break;
	}
	return "vllm";
}

} // namespace duckdb
