// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/udf_executor.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/execution/udf_executor.hpp"
#include "duckdb/execution/execution_runtime.hpp"
#include "duckdb/main/database.hpp"

namespace duckdb {

void SetUDFExecutorFactory(DatabaseInstance &db, udf_executor_factory_t factory) {
	auto &runtime = db.GetExecutionRuntime();
	lock_guard<mutex> guard(runtime.lock);
	runtime.udf_executor_factory = factory;
}

udf_executor_factory_t GetUDFExecutorFactory(ClientContext &context) {
	auto &runtime = DatabaseInstance::GetDatabase(context).GetExecutionRuntime();
	lock_guard<mutex> guard(runtime.lock);
	return runtime.udf_executor_factory;
}

} // namespace duckdb
