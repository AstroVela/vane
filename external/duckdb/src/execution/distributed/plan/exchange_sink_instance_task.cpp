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
		if (sink_handle.output_partition_count == 0) {
			sink_handle.output_partition_count = sink->NumPartitions();
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
}

ExchangeSinkInstanceTaskDescriptor ExchangeSinkInstanceTaskDescriptor::Deserialize(Deserializer &deserializer) {
	ExchangeSinkInstanceTaskDescriptor result;
	result.sink_instance.sink_handle.task_partition_id = deserializer.ReadProperty<idx_t>(1, "task_partition_id");
	result.sink_instance.attempt_id = deserializer.ReadProperty<idx_t>(2, "attempt_id");
	result.sink_instance.output_location =
	    deserializer.ReadPropertyWithExplicitDefault<string>(3, "output_location", "");
	result.sink_instance.output_partition_count =
	    deserializer.ReadPropertyWithDefault<idx_t>(4, "output_partition_count");
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
