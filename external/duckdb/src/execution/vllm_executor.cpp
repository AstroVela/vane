// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/vllm_executor.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/execution/vllm_executor.hpp"

#include "duckdb/common/exception.hpp"
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
	// The vllm() binder already guarantees options is a non-null STRUCT
	// envelope, and every envelope builder writes the engine field. A missing
	// or malformed engine field is therefore a bug, not a signal to silently
	// fall back to vllm.
	if (options.IsNull() || options.type().id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException("native inference options must use the versioned STRUCT envelope");
	}
	const auto &children = StructValue::GetChildren(options);
	const auto child_count = StructType::GetChildCount(options.type());
	for (idx_t i = 0; i < child_count; i++) {
		if (StructType::GetChildName(options.type(), i) != "engine") {
			continue;
		}
		const auto &value = children[i];
		if (value.IsNull() || value.type().id() != LogicalTypeId::VARCHAR) {
			throw InvalidInputException("native inference options 'engine' field must be a non-null string");
		}
		auto engine = StringValue::Get(value);
		if (engine.empty()) {
			throw InvalidInputException("native inference options 'engine' field must not be empty");
		}
		return engine;
	}
	throw InvalidInputException("native inference options envelope is missing the 'engine' field");
}

} // namespace duckdb
