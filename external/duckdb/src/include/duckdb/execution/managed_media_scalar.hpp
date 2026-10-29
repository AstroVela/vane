// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/media_resource.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

namespace duckdb {

//! A backend describes a complete, fixed-shape row before any pixel allocation.
//! The executor additionally reserves its input cache and owned input row.
struct ManagedMediaWorkingSet {
	idx_t output_bytes;
	idx_t scratch_bytes;
};

//! Catalog capability, recovered when expressions are deserialized. Runtime
//! scopes, requests, leases and callbacks are never part of a serialized plan.
struct ManagedMediaScalarFunctionInfo : public ScalarFunctionInfo {
	using working_set_t = std::function<ManagedMediaWorkingSet(ClientContext &, const BoundFunctionExpression &)>;
	using execute_t = std::function<void(ClientContext &, DataChunk &, Vector &, MediaResourceLease, shared_ptr<void>)>;

	ManagedMediaScalarFunctionInfo(working_set_t working_set_p, execute_t execute_p)
	    : working_set(std::move(working_set_p)), execute(std::move(execute_p)) {
	}

	working_set_t working_set;
	execute_t execute;
};

inline const ManagedMediaScalarFunctionInfo *GetManagedMediaScalar(const Expression &expression) {
	if (expression.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		return nullptr;
	}
	return dynamic_cast<const ManagedMediaScalarFunctionInfo *>(
	    expression.Cast<BoundFunctionExpression>().function.function_info.get());
}

} // namespace duckdb
