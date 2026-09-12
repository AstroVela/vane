// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/common.hpp"

namespace duckdb {
class ClientContext;
class LogicalOperator;
class QueryNode;

//! True only for a read whose entire bound plan uses client metadata or constants.
DUCKDB_API bool IsClientContextQuery(LogicalOperator &plan, bool client_query_origin = false);
//! Conservative pre-bind proof for native connection reads.
//! Does not bind expressions, expand macros or invoke table-function callbacks.
DUCKDB_API bool IsClientContextQuery(ClientContext &context, QueryNode &query);
} // namespace duckdb
