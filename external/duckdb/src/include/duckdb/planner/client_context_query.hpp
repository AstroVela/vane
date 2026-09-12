// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/common.hpp"

namespace duckdb {
class ClientContext;
class QueryNode;

//! Match the finite allowlist of direct native connection reads before binding.
//! Does not bind expressions, expand macros or invoke table-function callbacks.
DUCKDB_API bool IsClientContextQuery(ClientContext &context, QueryNode &query);
} // namespace duckdb
