// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/function/scalar_function.hpp"

namespace duckdb {

class DBConfig;

//! The backend is selected while binding. A bound native expression names the
//! extension function explicitly, including when it is serialized to a worker.
struct MediaBackend {
	static void RegisterOption(DBConfig &config);
	static bool UseNative(ClientContext &context, const string &domain);
	static unique_ptr<Expression> BindNative(FunctionBindExpressionInput &input, const string &domain,
	                                         const string &function_name);
};

} // namespace duckdb
