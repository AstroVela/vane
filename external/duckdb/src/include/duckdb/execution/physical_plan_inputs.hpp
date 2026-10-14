// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/serializer/serialization_data.hpp"
#include "duckdb/execution/physical_operator.hpp"

#include <functional>

namespace duckdb {

//! Explicit external inputs for a serialized physical subtree. Keeping one
//! native serialization scope preserves CTE and delimiter state, while the
//! caller supplies the execution operator for every named input on load.
struct PhysicalPlanInputBindings : public SerializationData::CustomData {
	static string GetType() {
		return "physical_plan_input_bindings";
	}

	unordered_map<const PhysicalOperator *, string> names;
	std::function<unique_ptr<PhysicalOperator>(PhysicalPlan &, const string &, const vector<LogicalType> &)> bind;
};

} // namespace duckdb
