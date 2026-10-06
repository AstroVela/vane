// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/serializer/serialization_data.hpp"

namespace duckdb {

// Physical aggregate arguments already refer to materialized input columns.
// Rebinding may reconstruct function data, but must not rewrite those inputs.
class PhysicalPlanDeserializationState : public SerializationData::CustomData {
public:
	explicit PhysicalPlanDeserializationState(SerializationData &data_p) : data(data_p) {
		data.SetCustom(*this);
	}
	~PhysicalPlanDeserializationState() override {
		data.UnsetCustom<PhysicalPlanDeserializationState>();
	}
	PhysicalPlanDeserializationState(const PhysicalPlanDeserializationState &) = delete;
	PhysicalPlanDeserializationState &operator=(const PhysicalPlanDeserializationState &) = delete;

	static string GetType() {
		return "physical_plan_deserialization";
	}

private:
	SerializationData &data;
};

} // namespace duckdb
