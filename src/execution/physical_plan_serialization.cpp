// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"

namespace duckdb {

void PhysicalPlan::Serialize(Serializer &serializer) const {
	// If there's a root, serialize it (operator serialization handles children)
	if (HasRoot()) {
		Root().Serialize(serializer);
	}
	// If empty plan: no-op (nothing to serialize)
}

unique_ptr<PhysicalOperator> PhysicalPlan::Deserialize(Deserializer &deserializer) {
	// Delegate to the PhysicalOperator deserializer which handles the recursive tree
	return PhysicalOperator::Deserialize(deserializer, *this);
}

} // namespace duckdb
