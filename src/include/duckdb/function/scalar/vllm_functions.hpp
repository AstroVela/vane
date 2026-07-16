// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/scalar/vllm_functions.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/function/function_set.hpp"
#include "duckdb/function/scalar_function.hpp"

namespace duckdb {

struct VLLMFunctionData : public FunctionData {
	VLLMFunctionData(string model_p, Value options_p);

	string model;
	Value options;

	unique_ptr<FunctionData> Copy() const override;
	bool Equals(const FunctionData &other) const override;
};

struct VLLMFunction {
	static ScalarFunctionSet GetFunctions();
};

} // namespace duckdb
