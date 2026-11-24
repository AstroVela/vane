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

class DatabaseInstance;

void RegisterUDFExecutorFactory(DatabaseInstance &db);
void RegisterUDFExternalBlockBackend();
void ShutdownUDFExecutorDispatcher();
void WakeUDFExecutorSlotsForTesting();
void PauseUDFDispatcherWaitForTesting();

} // namespace duckdb
