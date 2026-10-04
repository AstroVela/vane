// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/planner/operator/logical_data_sink.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/execution/operator/helper/physical_data_sink.hpp"
#include "utf8proc_wrapper.hpp"

namespace duckdb {

namespace {

BoundStatement BindDataSinkOperatorExtension(ClientContext &, Binder &, OperatorExtensionInfo *, SQLStatement &) {
	return BoundStatement();
}

} // namespace

LogicalDataSink::LogicalDataSink(string operation_id_p) : operation_id(std::move(operation_id_p)) {
	if (operation_id.empty() || operation_id.size() > 256 ||
	    !Utf8Proc::IsValid(operation_id.c_str(), operation_id.size())) {
		throw InvalidInputException("DataSink operation identity must contain 1 to 256 UTF-8 bytes");
	}
}

vector<ColumnBinding> LogicalDataSink::GetColumnBindings() {
	return children[0]->GetColumnBindings();
}

idx_t LogicalDataSink::EstimateCardinality(ClientContext &context) {
	return children[0]->EstimateCardinality(context);
}

void LogicalDataSink::ResolveTypes() {
	types = children[0]->types;
}

PhysicalOperator &LogicalDataSink::CreatePlan(ClientContext &, PhysicalPlanGenerator &planner) {
	if (children.size() != 1) {
		throw InternalException("LogicalDataSink requires exactly one child");
	}
	auto &child = planner.CreatePlan(*children[0]);
	auto &sink = planner.Make<PhysicalDataSink>(types, operation_id, estimated_cardinality);
	sink.children.push_back(child);
	return sink;
}

string LogicalDataSink::GetExtensionName() const {
	return EXTENSION_NAME;
}

void LogicalDataSink::Serialize(Serializer &serializer) const {
	LogicalExtensionOperator::Serialize(serializer);
	serializer.WriteProperty<string>(201, "operation_id", operation_id);
}

DataSinkOperatorExtension::DataSinkOperatorExtension() {
	Bind = BindDataSinkOperatorExtension;
}

string DataSinkOperatorExtension::GetName() {
	return LogicalDataSink::EXTENSION_NAME;
}

unique_ptr<LogicalExtensionOperator> DataSinkOperatorExtension::Deserialize(Deserializer &deserializer) {
	auto operation_id = deserializer.ReadProperty<string>(201, "operation_id");
	return make_uniq<LogicalDataSink>(std::move(operation_id));
}

} // namespace duckdb
