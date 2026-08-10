// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/function/scalar_function.hpp"

namespace duckdb {

struct ImageCropFunction {
	static ScalarFunction GetFunction();
};

struct ImageEncodeFunction {
	static ScalarFunction GetFunction();
};

} // namespace duckdb
