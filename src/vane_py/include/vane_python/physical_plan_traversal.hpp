// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/execution/physical_operator.hpp"

#include <functional>
#include <unordered_set>

namespace duckdb {

//! Visit each executable operator once, in deterministic pre-order. GetChildren
//! includes owned plans (delim joins, EXECUTE, result collectors) that are not
//! data inputs. UDF binding IDs must use this same order for collection/injection.
inline void VisitPhysicalExecutionGraph(PhysicalOperator &root, const std::function<void(PhysicalOperator &)> &visit) {
	std::unordered_set<const PhysicalOperator *> visited;
	std::function<void(PhysicalOperator &)> walk = [&](PhysicalOperator &op) {
		if (!visited.insert(&op).second) {
			return;
		}
		visit(op);
		for (auto &child : op.GetChildren()) {
			walk(const_cast<PhysicalOperator &>(child.get()));
		}
	};
	walk(root);
}

} // namespace duckdb
