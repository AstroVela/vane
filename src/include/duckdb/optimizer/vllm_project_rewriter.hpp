// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/vllm_project_rewriter.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

class Binder;

class VLLMProjectRewriter {
public:
	explicit VLLMProjectRewriter(Binder &binder);

	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> op);

private:
	Binder &binder;

	unique_ptr<LogicalOperator> Rewrite(unique_ptr<LogicalOperator> op);
};

} // namespace duckdb
