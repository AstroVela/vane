// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/common/string.hpp"
#include <pybind11/pybind11.h>

namespace duckdb {

class PhysicalOperator;
class ClientContext;

//! Describe an existing native execution graph without translating or mutating it.
//! UDF binding IDs follow VisitPhysicalExecutionGraph, just like executor injection.
pybind11::dict CollectNativeResourceGraph(PhysicalOperator &root, const string &query_id, ClientContext &context);

} // namespace duckdb
