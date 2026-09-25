// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/parser/parsed_expression.hpp"
#include "duckdb/parser/query_node.hpp"
#include "duckdb/planner/expression/bound_parameter_data.hpp"

namespace duckdb {

//! Capture typed API parameters before passing SQL to native QueryRelation.
//! Each relation owns a complete AST, so composing relations cannot merge or
//! overwrite independently supplied parameter values.
void CaptureQueryParameters(QueryNode &node, const case_insensitive_map_t<BoundParameterData> &parameters);
void CaptureParameters(unique_ptr<ParsedExpression> &expression,
                       const case_insensitive_map_t<BoundParameterData> &parameters);

} // namespace duckdb
