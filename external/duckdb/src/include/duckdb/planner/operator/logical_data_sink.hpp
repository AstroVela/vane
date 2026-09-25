// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator_extension.hpp"

namespace duckdb {

class LogicalDataSink : public LogicalExtensionOperator {
public:
	static constexpr const char *EXTENSION_NAME = "vane_data_sink";

	explicit LogicalDataSink(string operation_id);

	string operation_id;

	vector<ColumnBinding> GetColumnBindings() override;
	idx_t EstimateCardinality(ClientContext &context) override;
	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override;
	string GetExtensionName() const override;
	void Serialize(Serializer &serializer) const override;

protected:
	void ResolveTypes() override;
};

class DataSinkOperatorExtension : public OperatorExtension {
public:
	DataSinkOperatorExtension();

	string GetName() override;
	unique_ptr<LogicalExtensionOperator> Deserialize(Deserializer &deserializer) override;
};

} // namespace duckdb
