// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/operator/logical_repartition.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/planner/operator/logical_repartition.hpp"

namespace duckdb {

LogicalRepartition::LogicalRepartition(std::shared_ptr<RepartitionSpec> repartition_spec_p)
    : LogicalOperator(LogicalOperatorType::LOGICAL_REPARTITION), repartition_spec(std::move(repartition_spec_p)) {
}

vector<ColumnBinding> LogicalRepartition::GetColumnBindings() {
	return children[0]->GetColumnBindings();
}

idx_t LogicalRepartition::EstimateCardinality(ClientContext &context) {
	return children[0]->EstimateCardinality(context);
}

void LogicalRepartition::ResolveTypes() {
	types = children[0]->types;
}

} // namespace duckdb
