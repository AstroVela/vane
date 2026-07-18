// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/execution/operator/exchange/repartition.hpp"

namespace duckdb {

//! LogicalLocalExchange represents a local (intra-process) exchange in the logical plan.
//! Unlike LogicalRepartition (which becomes a distributed remote exchange),
//! this always maps to PhysicalLocalExchange → LocalExchangePassthroughNode.
class LogicalLocalExchange : public LogicalOperator {
public:
	static constexpr const LogicalOperatorType TYPE = LogicalOperatorType::LOGICAL_LOCAL_EXCHANGE;

public:
	explicit LogicalLocalExchange(std::shared_ptr<RepartitionSpec> repartition_spec);

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
