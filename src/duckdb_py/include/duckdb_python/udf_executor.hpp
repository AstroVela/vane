//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/udf_executor.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

namespace duckdb {

void RegisterUDFExecutorFactory();
void ShutdownUDFExecutorDispatcher();

} // namespace duckdb
