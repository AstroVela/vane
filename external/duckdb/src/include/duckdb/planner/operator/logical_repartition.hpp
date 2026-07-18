// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/operator/logical_repartition.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/execution/operator/exchange/repartition.hpp"

namespace duckdb {

//! LogicalRepartition represents a repartition operation in the logical plan
class LogicalRepartition : public LogicalOperator {
public:
	static constexpr const LogicalOperatorType TYPE = LogicalOperatorType::LOGICAL_REPARTITION;

public:
	explicit LogicalRepartition(std::shared_ptr<RepartitionSpec> repartition_spec);

	std::shared_ptr<RepartitionSpec> repartition_spec;

public:
	vector<ColumnBinding> GetColumnBindings() override;
	idx_t EstimateCardinality(ClientContext &context) override;

	void Serialize(Serializer &serializer) const override;
	static unique_ptr<LogicalOperator> Deserialize(Deserializer &deserializer);

protected:
	void ResolveTypes() override;
};

} // namespace duckdb
