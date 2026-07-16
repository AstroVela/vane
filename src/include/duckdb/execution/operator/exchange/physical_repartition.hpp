// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/operator/exchange/physical_local_exchange.hpp"

namespace duckdb {

//! PhysicalRepartition is converted by translate.cpp to a distributed RepartitionNode.
//! In native execution, it uses the same in-process exchange machinery as LOCAL_EXCHANGE.
class PhysicalRepartition : public PhysicalLocalExchange {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::REPARTITION;

public:
	PhysicalRepartition(PhysicalPlan &physical_plan, vector<LogicalType> types,
	                    std::shared_ptr<RepartitionSpec> repartition_spec, idx_t estimated_cardinality)
	    : PhysicalLocalExchange(physical_plan, PhysicalOperatorType::REPARTITION, std::move(types),
	                            std::move(repartition_spec), estimated_cardinality) {
	}
};

} // namespace duckdb
