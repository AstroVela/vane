// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/execution/operator/order/physical_top_n.hpp"
#include "duckdb/execution/operator/order/physical_order.hpp"
#include "duckdb/execution/operator/helper/physical_streaming_limit.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/planner/filter/dynamic_filter.hpp"

namespace duckdb {

PhysicalOperator &PhysicalPlanGenerator::CreatePlan(LogicalTopN &op) {
	D_ASSERT(op.children.size() == 1);
	auto &plan = CreatePlan(*op.children[0]);
	if (HasNativeMediaFlow(context, plan)) {
		// Media payload enters spillable storage before releasing production
		// credits. A full sort bounds the receiving working set independently
		// of LIMIT/OFFSET; candidate-heap optimization can follow separately.
		if (op.dynamic_filter) {
			// This path does not publish a TopN boundary to optional scan filters.
			op.dynamic_filter->Reset();
		}
		vector<idx_t> projection_map;
		for (idx_t i = 0; i < plan.types.size(); i++) {
			projection_map.push_back(i);
		}
		auto &order = Make<PhysicalOrder>(op.types, std::move(op.orders), std::move(projection_map),
		                                  plan.estimated_cardinality, false, true);
		order.children.push_back(plan);
		auto &limit = Make<PhysicalStreamingLimit>(
		    op.types, BoundLimitNode::ConstantValue(NumericCast<int64_t>(op.limit)),
		    BoundLimitNode::ConstantValue(NumericCast<int64_t>(op.offset)), op.estimated_cardinality, false);
		limit.children.push_back(order);
		return limit;
	}
	auto &top_n =
	    Make<PhysicalTopN>(op.types, std::move(op.orders), NumericCast<idx_t>(op.limit), NumericCast<idx_t>(op.offset),
	                       std::move(op.dynamic_filter), op.estimated_cardinality);
	top_n.children.push_back(plan);
	return top_n;
}

} // namespace duckdb
