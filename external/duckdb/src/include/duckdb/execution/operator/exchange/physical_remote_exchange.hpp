// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/operator/exchange/repartition.hpp"

namespace duckdb {

//! PhysicalRemoteExchange is a marker operator in the DuckDB physical plan.
//! translate.cpp converts it to a distributed RepartitionNode.
//! It does NOT execute locally — in non-distributed mode, it acts as a passthrough.
class PhysicalRemoteExchange : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::REPARTITION;

public:
	PhysicalRemoteExchange(PhysicalPlan &physical_plan, vector<LogicalType> types,
	                       std::shared_ptr<RepartitionSpec> repartition_spec, idx_t estimated_cardinality)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::REPARTITION, std::move(types), estimated_cardinality),
	      repartition_spec(std::move(repartition_spec)) {
	}

	std::shared_ptr<RepartitionSpec> repartition_spec;

public:
	// In non-distributed mode, just pass through data from child.
	SourceResultType GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const override {
		// Should not be called — translate.cpp replaces this with RepartitionNode
		throw InternalException("PhysicalRemoteExchange should not be executed directly");
	}

	bool IsSource() const override {
		return false;
	}

	bool IsSink() const override {
		return false;
	}

	InsertionOrderPreservingMap<string> ParamsToString() const override {
		InsertionOrderPreservingMap<string> result;
		if (repartition_spec) {
			result["repartition_type"] = repartition_spec->var_name();
		}
		return result;
	}
};

} // namespace duckdb
