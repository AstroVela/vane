//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/table/range.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/function/table_function.hpp"
#include "duckdb/function/built_in_functions.hpp"

namespace duckdb {

struct CheckpointFunction {
	static void RegisterFunction(BuiltinFunctions &set);
};

struct GlobTableFunction {
	static void RegisterFunction(BuiltinFunctions &set);
};

struct RangeTableFunction {
	//! Construct the complete overload set. The core distributed manifest uses
	//! these same function objects so protocol identities cannot drift from
	//! catalog registration.
	static vector<TableFunction> GetFunctions();
	static void RegisterFunction(BuiltinFunctions &set);
};

struct RepeatTableFunction {
	static TableFunction GetFunction();
	static void RegisterFunction(BuiltinFunctions &set);
};

struct RepeatRowTableFunction {
	static TableFunction GetFunction();
	static void RegisterFunction(BuiltinFunctions &set);
};

struct UnnestTableFunction {
	static TableFunction GetFunction();
	static void RegisterFunction(BuiltinFunctions &set);
};

struct CSVSnifferFunction {
	static void RegisterFunction(BuiltinFunctions &set);
};

struct ReadBlobFunction {
	static void RegisterFunction(BuiltinFunctions &set);
};

struct ReadTextFunction {
	static void RegisterFunction(BuiltinFunctions &set);
};

struct QueryTableFunction {
	static void RegisterFunction(BuiltinFunctions &set);
};

} // namespace duckdb
