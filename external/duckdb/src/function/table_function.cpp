// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/function/table_function.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"

namespace duckdb {

GlobalTableFunctionState::~GlobalTableFunctionState() {
}

LocalTableFunctionState::~LocalTableFunctionState() {
}

PartitionStatistics::PartitionStatistics() : row_start(0), count(0), count_type(CountType::COUNT_APPROXIMATE) {
}

TableFunctionInfo::~TableFunctionInfo() {
}

void DistributedScanTask::Validate() const {
	if (task_id.empty()) {
		throw SerializationException("distributed extension scan task has an empty task_id");
	}
}

void DistributedScanTask::Serialize(Serializer &serializer) const {
	Validate();
	serializer.WriteProperty(1, "task_id", task_id);
	serializer.WriteProperty(2, "payload", payload);
	serializer.WriteProperty(3, "estimated_cardinality", estimated_cardinality);
	serializer.WriteProperty(4, "estimated_bytes", estimated_bytes);
}

DistributedScanTask DistributedScanTask::Deserialize(Deserializer &deserializer) {
	DistributedScanTask result;
	result.task_id = deserializer.ReadProperty<string>(1, "task_id");
	result.payload = deserializer.ReadProperty<string>(2, "payload");
	result.estimated_cardinality = deserializer.ReadProperty<optional_idx>(3, "estimated_cardinality");
	result.estimated_bytes = deserializer.ReadProperty<optional_idx>(4, "estimated_bytes");
	result.Validate();
	return result;
}

TableFunction::TableFunction(string name, const vector<LogicalType> &arguments, table_function_t function_,
                             table_function_bind_t bind, table_function_init_global_t init_global,
                             table_function_init_local_t init_local)
    : SimpleNamedParameterFunction(std::move(name), arguments), bind(bind), bind_replace(nullptr),
      bind_operator(nullptr), init_global(init_global), init_local(init_local), function(function_),
      in_out_function(nullptr), in_out_function_final(nullptr), in_out_function_batch(nullptr),
      in_out_function_final_batch(nullptr), statistics(nullptr), statistics_extended(nullptr), dependency(nullptr),
      cardinality(nullptr), rows_scanned(nullptr), get_metrics(nullptr), pushdown_complex_filter(nullptr),
      pushdown_expression(nullptr), to_string(nullptr), dynamic_to_string(nullptr), table_scan_progress(nullptr),
      get_partition_data(nullptr), get_bind_info(nullptr), type_pushdown(nullptr), get_multi_file_reader(nullptr),
      supports_pushdown_type(nullptr), supports_pushdown_extract(nullptr), get_partition_info(nullptr),
      get_partition_stats(nullptr), get_virtual_columns(nullptr), get_row_id_columns(nullptr), set_scan_order(nullptr),
      serialize(nullptr), deserialize(nullptr), distributed_scan(nullptr), projection_pushdown(false),
      filter_pushdown(false), filter_prune(false), sampling_pushdown(false), late_materialization(false) {
}

TableFunction::TableFunction(string name, const vector<LogicalType> &arguments, std::nullptr_t function_,
                             table_function_bind_t bind, table_function_init_global_t init_global,
                             table_function_init_local_t init_local)
    : SimpleNamedParameterFunction(std::move(name), arguments), bind(bind), bind_replace(nullptr),
      bind_operator(nullptr), init_global(init_global), init_local(init_local), function(nullptr),
      in_out_function(nullptr), in_out_function_final(nullptr), in_out_function_batch(nullptr),
      in_out_function_final_batch(nullptr), statistics(nullptr), statistics_extended(nullptr), dependency(nullptr),
      cardinality(nullptr), rows_scanned(nullptr), get_metrics(nullptr), pushdown_complex_filter(nullptr),
      pushdown_expression(nullptr), to_string(nullptr), dynamic_to_string(nullptr), table_scan_progress(nullptr),
      get_partition_data(nullptr), get_bind_info(nullptr), type_pushdown(nullptr), get_multi_file_reader(nullptr),
      supports_pushdown_type(nullptr), supports_pushdown_extract(nullptr), get_partition_info(nullptr),
      get_partition_stats(nullptr), get_virtual_columns(nullptr), get_row_id_columns(nullptr), set_scan_order(nullptr),
      serialize(nullptr), deserialize(nullptr), distributed_scan(nullptr), projection_pushdown(false),
      filter_pushdown(false), filter_prune(false), sampling_pushdown(false), late_materialization(false) {
}

TableFunction::TableFunction(const vector<LogicalType> &arguments, table_function_t function_,
                             table_function_bind_t bind, table_function_init_global_t init_global,
                             table_function_init_local_t init_local)
    : TableFunction("", arguments, function_, bind, init_global, init_local) {
}

TableFunction::TableFunction(const vector<LogicalType> &arguments, std::nullptr_t function_, table_function_bind_t bind,
                             table_function_init_global_t init_global, table_function_init_local_t init_local)
    : TableFunction("", arguments, function_, bind, init_global, init_local) {
}

TableFunction::TableFunction() : TableFunction("", {}, nullptr, nullptr, nullptr, nullptr) {
}

bool TableFunction::operator==(const TableFunction &rhs) const {
	return name == rhs.name && arguments == rhs.arguments && varargs == rhs.varargs && bind == rhs.bind &&
	       bind_replace == rhs.bind_replace && bind_operator == rhs.bind_operator && init_global == rhs.init_global &&
	       init_local == rhs.init_local && function == rhs.function && in_out_function == rhs.in_out_function &&
	       in_out_function_final == rhs.in_out_function_final && in_out_function_batch == rhs.in_out_function_batch &&
	       in_out_function_final_batch == rhs.in_out_function_final_batch && statistics == rhs.statistics &&
	       dependency == rhs.dependency && cardinality == rhs.cardinality &&
	       pushdown_complex_filter == rhs.pushdown_complex_filter && pushdown_expression == rhs.pushdown_expression &&
	       to_string == rhs.to_string && dynamic_to_string == rhs.dynamic_to_string &&
	       table_scan_progress == rhs.table_scan_progress && get_partition_data == rhs.get_partition_data &&
	       get_bind_info == rhs.get_bind_info && type_pushdown == rhs.type_pushdown &&
	       get_multi_file_reader == rhs.get_multi_file_reader && supports_pushdown_type == rhs.supports_pushdown_type &&
	       get_partition_info == rhs.get_partition_info && get_partition_stats == rhs.get_partition_stats &&
	       get_virtual_columns == rhs.get_virtual_columns && get_row_id_columns == rhs.get_row_id_columns &&
	       serialize == rhs.serialize && deserialize == rhs.deserialize &&
	       ((!distributed_scan && !rhs.distributed_scan) ||
	        (distributed_scan && rhs.distributed_scan && *distributed_scan == *rhs.distributed_scan)) &&
	       verify_serialization == rhs.verify_serialization && projection_pushdown == rhs.projection_pushdown &&
	       filter_pushdown == rhs.filter_pushdown && filter_prune == rhs.filter_prune &&
	       sampling_pushdown == rhs.sampling_pushdown && late_materialization == rhs.late_materialization &&
	       global_initialization == rhs.global_initialization;
}

string GetDistributedTableFunctionSignature(const string &function_name, const vector<LogicalType> &arguments,
                                            const LogicalType &varargs) {
	return Function::CallToString(string(), string(), function_name, arguments, varargs);
}

void TableFunctionDistributedScanCallbacks::ValidateDefinition(const string &function_name) const {
	if (!plan || !create_worker_bind || !apply_tasks) {
		throw InvalidInputException("Distributed scan callbacks for table function '%s' must define plan, "
		                            "create_worker_bind, and apply_tasks",
		                            function_name);
	}
	if (protocol_version == 0) {
		throw InvalidInputException(
		    "Distributed scan protocol version for table function '%s' must be greater than zero", function_name);
	}
	task_codec.Validate("Distributed scan task codec for table function '" + function_name + "'");
}

void TableFunctionDistributedScanCallbacks::Validate(const TableFunction &function) const {
	ValidateDefinition(function.name);
	if (capability.extension_name.empty()) {
		throw InvalidInputException("Distributed scan capability for table function '%s' was not bound by its loader",
		                            function.name);
	}
	if (capability.capability.kind != DistributedExtensionCapabilityKind::TABLE_FUNCTION) {
		throw InvalidInputException("Distributed scan capability for table function '%s' must have kind table_function",
		                            function.name);
	}
	if (capability.capability.name != function.name) {
		throw InvalidInputException("Distributed scan capability name '%s' does not match table function '%s'",
		                            capability.capability.name, function.name);
	}
	// BindCapability freezes the declared catalog overload. Bind callbacks may
	// subsequently specialize or erase arguments, so the runtime function shape
	// must not be used to recompute this wire identity.
	capability.Validate();
}

void TableFunctionDistributedScanCallbacks::BindCapability(const string &extension_name,
                                                           const TableFunction &function) {
	ValidateDefinition(function.name);
	DistributedExtensionCapabilityReference bound_capability;
	bound_capability.extension_name = extension_name;
	bound_capability.capability.kind = DistributedExtensionCapabilityKind::TABLE_FUNCTION;
	bound_capability.capability.name = function.name;
	bound_capability.capability.protocol_version = protocol_version;
	bound_capability.capability.function_signature =
	    GetDistributedTableFunctionSignature(function.name, function.arguments, function.varargs);
	if (!capability.extension_name.empty() && capability != bound_capability) {
		throw InvalidInputException("Distributed scan capability for table function '%s' is already bound to '%s'",
		                            function.name, capability.CanonicalIdentity());
	}
	capability = std::move(bound_capability);
	Validate(function);
}

const DistributedExtensionCapabilityReference &TableFunctionDistributedScanCallbacks::GetCapability() const {
	if (capability.extension_name.empty()) {
		throw InternalException("Distributed scan capability has not been bound by its extension loader");
	}
	return capability;
}

bool TableFunctionDistributedScanCallbacks::operator==(const TableFunctionDistributedScanCallbacks &other) const {
	return protocol_version == other.protocol_version && capability == other.capability &&
	       task_codec == other.task_codec && plan == other.plan && create_worker_bind == other.create_worker_bind &&
	       apply_tasks == other.apply_tasks;
}

void TableFunction::SetDistributedScanCallbacks(TableFunctionDistributedScanCallbacks callbacks) {
	callbacks.ValidateDefinition(name);
	distributed_scan = make_shared_ptr<const TableFunctionDistributedScanCallbacks>(std::move(callbacks));
}

void TableFunction::BindDistributedScanCapability(const string &extension_name) {
	if (!distributed_scan) {
		throw InternalException("Table function '%s' has no distributed scan callbacks", name);
	}
	auto callbacks = *distributed_scan;
	callbacks.BindCapability(extension_name, *this);
	distributed_scan = make_shared_ptr<const TableFunctionDistributedScanCallbacks>(std::move(callbacks));
}

const TableFunctionDistributedScanCallbacks &TableFunction::GetDistributedScanCallbacks() const {
	if (!distributed_scan) {
		throw InternalException("Table function '%s' has no distributed scan callbacks", name);
	}
	return *distributed_scan;
}

bool TableFunction::operator!=(const TableFunction &rhs) const {
	return !(*this == rhs);
}

bool TableFunction::Equal(const TableFunction &rhs) const {
	// number of types
	if (this->arguments.size() != rhs.arguments.size()) {
		return false;
	}
	// argument types
	for (idx_t i = 0; i < this->arguments.size(); ++i) {
		if (this->arguments[i] != rhs.arguments[i]) {
			return false;
		}
	}
	// varargs
	if (this->varargs != rhs.varargs) {
		return false;
	}

	return true; // they are equal
}

bool ExtractSourceResultType(AsyncResultType in, SourceResultType &out) {
	switch (in) {
	case AsyncResultType::IMPLICIT:
	case AsyncResultType::INVALID:
		return false;
	case AsyncResultType::HAVE_MORE_OUTPUT:
		out = SourceResultType::HAVE_MORE_OUTPUT;
		break;
	case AsyncResultType::FINISHED:
		out = SourceResultType::FINISHED;
		break;
	case AsyncResultType::BLOCKED:
		out = SourceResultType::BLOCKED;
		break;
	}
	return true;
}

} // namespace duckdb
