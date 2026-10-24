// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/mutex.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/execution/udf_executor.hpp"
#include "duckdb/execution/vllm_executor.hpp"
#include "duckdb/function/table/datasource_scan.hpp"

namespace duckdb {

//! Callbacks belong to the database, not to an engine library's static storage.
//! A self-contained extension may execute its own engine code against the same
//! DatabaseInstance, including through a new Connection. That code must resolve
//! the same runtime callbacks as the host. Access is synchronized by lock; copy
//! callbacks before invoking them so no runtime code runs under the lock.
struct ExecutionRuntime {
	//! Embedders install their defaults once, including when connections race
	//! to open a cached database. Later connections must preserve overrides.
	std::once_flag initialization;
	mutex lock;
	udf_executor_factory_t udf_executor_factory = nullptr;
	unordered_map<string, vllm_executor_factory_t> engine_executor_factories;
	DataSourceCallbacks datasource;
};

} // namespace duckdb
