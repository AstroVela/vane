// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// vane_python/udf_executor.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

namespace duckdb {

void RegisterUDFExecutorFactory();
void ShutdownUDFExecutorDispatcher();
void WakeUDFExecutorSlotsForTesting();

} // namespace duckdb
