// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

namespace duckdb {

//! Dependency introduced by a source, independent of SQL shape and operator name.
//! NONE means the operator derives its values from expressions or child operators.
enum class QuerySourceKind { NONE, DATA, CLIENT_METADATA };

} // namespace duckdb
