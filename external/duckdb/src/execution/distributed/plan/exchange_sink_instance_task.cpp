// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/distributed/plan/exchange_sink_instance_task.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/execution/distributed/plan/exchange_sink_instance_task.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/execution/operator/exchange/physical_remote_exchange_sink.hpp"
#include "duckdb/execution/physical_plan.hpp"

namespace duckdb {
namespace distributed {

namespace {

bool SetValidationError(std::string *error, const std::string &message) {
	if (error) {
		*error = message;
	}
	return false;
}

bool ExtractSinkOutputLocationPrefix(const ExchangeSinkInstanceHandle &handle, std::string &prefix,
                                     std::string *error) {
	const auto sink_suffix = std::string("__sink_") + std::to_string(handle.sink_handle.task_partition_id) +
	                         "__attempt_" + std::to_string(handle.attempt_id);
	if (handle.output_location.size() < sink_suffix.size() ||
	    handle.output_location.compare(handle.output_location.size() - sink_suffix.size(), sink_suffix.size(),
	                                   sink_suffix) != 0) {
		return SetValidationError(error, "remote exchange sink plan has an invalid sink output location");
	}
	prefix = handle.output_location.substr(0, handle.output_location.size() - sink_suffix.size());
	if (prefix.empty()) {
		return SetValidationError(error, "remote exchange sink plan is missing its exchange instance id");
	}
	return true;
}

bool ValidateRuntimeSinkHandle(const PhysicalRemoteExchangeSink &sink, const ExchangeSinkInstanceHandle &runtime_handle,
                               std::string *error) {
	const auto &plan_handle = sink.SinkHandle();
	if (plan_handle.query_id.empty()) {
		return SetValidationError(error, "remote exchange sink plan is missing query ownership");
	}
	if (runtime_handle.query_id.empty() || runtime_handle.query_id != plan_handle.query_id) {
		return SetValidationError(error, "runtime exchange sink query does not match the plan");
	}
	if (plan_handle.output_partition_count != sink.NumPartitions()) {
		return SetValidationError(error, "remote exchange sink plan has an inconsistent output partition count");
	}
	if (runtime_handle.output_partition_count == 0 || runtime_handle.output_partition_count != sink.NumPartitions()) {
		return SetValidationError(error, "runtime exchange sink output partition count does not match the plan");
	}
	if (plan_handle.output_location.empty()) {
		return SetValidationError(error, "remote exchange sink plan is missing its output location");
	}
	std::string exchange_instance_prefix;
	if (!ExtractSinkOutputLocationPrefix(plan_handle, exchange_instance_prefix, error)) {
		return false;
	}
	const auto expected_output_location = exchange_instance_prefix + "__sink_" +
	                                      std::to_string(runtime_handle.sink_handle.task_partition_id) + "__attempt_" +
	                                      std::to_string(runtime_handle.attempt_id);
	if (runtime_handle.output_location != expected_output_location) {
		return SetValidationError(error, "runtime exchange sink output location does not match the plan");
	}
	return true;
}

bool ApplyExchangeSinkInstanceToOperator(PhysicalOperator &op, const ExchangeSinkInstanceTaskDescriptor &task,
                                         std::string *error, idx_t &applied) {
	if (op.type == PhysicalOperatorType::EXCHANGE_SINK) {
		auto *sink = dynamic_cast<PhysicalRemoteExchangeSink *>(&op);
		if (!sink) {
			if (error) {
				*error = "EXCHANGE_SINK operator is not a PhysicalRemoteExchangeSink";
			}
			return false;
		}
		auto sink_handle = task.sink_instance;
		if (!ValidateRuntimeSinkHandle(*sink, sink_handle, error)) {
			return false;
		}
		sink->ApplyRuntimeSinkHandle(std::move(sink_handle));
		applied++;
	}
	for (auto &child : op.children) {
		if (!ApplyExchangeSinkInstanceToOperator(child.get(), task, error, applied)) {
			return false;
		}
	}
	return true;
}

} // namespace

void ExchangeSinkInstanceTaskDescriptor::Serialize(Serializer &serializer) const {
	serializer.WriteProperty(1, "task_partition_id", sink_instance.sink_handle.task_partition_id);
	serializer.WriteProperty(2, "attempt_id", sink_instance.attempt_id);
	serializer.WriteProperty(3, "output_location", sink_instance.output_location);
	serializer.WriteProperty(4, "output_partition_count", sink_instance.output_partition_count);
	serializer.WriteProperty(5, "flight_server_epoch", sink_instance.flight_server_epoch);
	serializer.WriteProperty(6, "query_id", sink_instance.query_id);
}

ExchangeSinkInstanceTaskDescriptor ExchangeSinkInstanceTaskDescriptor::Deserialize(Deserializer &deserializer) {
	ExchangeSinkInstanceTaskDescriptor result;
	result.sink_instance.sink_handle.task_partition_id = deserializer.ReadProperty<idx_t>(1, "task_partition_id");
	result.sink_instance.attempt_id = deserializer.ReadProperty<idx_t>(2, "attempt_id");
	result.sink_instance.output_location =
	    deserializer.ReadPropertyWithExplicitDefault<string>(3, "output_location", "");
	result.sink_instance.output_partition_count =
	    deserializer.ReadPropertyWithDefault<idx_t>(4, "output_partition_count");
	result.sink_instance.flight_server_epoch = deserializer.ReadProperty<string>(5, "flight_server_epoch");
	result.sink_instance.query_id = deserializer.ReadProperty<string>(6, "query_id");
	return result;
}

std::string ExchangeSinkInstanceTaskDescriptor::SerializeToBytes() const {
	MemoryStream stream(Allocator::DefaultAllocator());
	BinarySerializer serializer(stream);
	serializer.Begin();
	Serialize(serializer);
	serializer.End();
	return std::string(reinterpret_cast<const char *>(stream.GetData()), stream.GetPosition());
}

ExchangeSinkInstanceTaskDescriptor ExchangeSinkInstanceTaskDescriptor::DeserializeFromBytes(const std::string &bytes) {
	if (bytes.empty()) {
		return ExchangeSinkInstanceTaskDescriptor();
	}
	auto *data_ptr = reinterpret_cast<data_ptr_t>(const_cast<char *>(bytes.data()));
	MemoryStream stream(data_ptr, bytes.size());
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto result = Deserialize(deserializer);
	deserializer.End();
	return result;
}

bool ApplyExchangeSinkInstanceToPlan(duckdb::PhysicalPlan &plan, const ExchangeSinkInstanceTaskDescriptor &task,
                                     std::string *error) {
	if (!plan.HasRoot()) {
		if (error) {
			*error = "plan has no root";
		}
		return false;
	}
	idx_t applied = 0;
	if (!ApplyExchangeSinkInstanceToOperator(plan.Root(), task, error, applied)) {
		return false;
	}
	if (applied == 0 && error) {
		*error = "no remote exchange sink found in plan";
	}
	return applied > 0;
}

} // namespace distributed
} // namespace duckdb
