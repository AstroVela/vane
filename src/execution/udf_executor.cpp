// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/udf_executor.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/execution/udf_executor.hpp"

namespace duckdb {

static udf_executor_factory_t udf_executor_factory = nullptr;

void SetUDFExecutorFactory(udf_executor_factory_t factory) {
	udf_executor_factory = factory;
}

udf_executor_factory_t GetUDFExecutorFactory() {
	return udf_executor_factory;
}

} // namespace duckdb
