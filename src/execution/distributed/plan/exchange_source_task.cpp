// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/distributed/plan/exchange_source_task.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/execution/distributed/plan/exchange_source_task.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/execution/operator/exchange/physical_remote_exchange_source.hpp"
#include "duckdb/execution/physical_plan.hpp"

#include <algorithm>

namespace duckdb {
namespace distributed {

namespace {

bool ApplyExchangeSourceTasksToOperator(PhysicalOperator &op,
                                        const std::unordered_map<idx_t, ExchangeSourceTaskDescriptor> &tasks,
                                        std::string *error, idx_t &applied) {
	if (op.type == PhysicalOperatorType::EXCHANGE_SOURCE) {
		auto *source = dynamic_cast<PhysicalRemoteExchangeSource *>(&op);
		if (source && source->RuntimeSourceNodeId().IsValid()) {
			const auto node_id = source->RuntimeSourceNodeId().GetIndex();
			auto entry = tasks.find(node_id);
			if (entry == tasks.end()) {
				if (error) {
					*error = "missing exchange source task for runtime_source_node_id=" + std::to_string(node_id);
				}
				return false;
			}
			source->ApplyRuntimeTaskDescriptor(entry->second);
			applied++;
		}
	}
	for (auto &child : op.children) {
		if (!ApplyExchangeSourceTasksToOperator(child.get(), tasks, error, applied)) {
			return false;
		}
	}
	return true;
}

} // namespace

void ExchangeSourceTaskDescriptor::Serialize(Serializer &serializer) const {
	serializer.WriteProperty(1, "partition_indices", partition_indices);
	serializer.WriteList(2, "source_handles", source_handles.size(), [&](Serializer::List &list, idx_t i) {
		list.WriteObject([&](Serializer &obj) {
			const auto &handle = source_handles[i];
			obj.WriteProperty(1, "partition_id", handle.partition_id);
			obj.WriteProperty(2, "node_id", handle.node_id);
			obj.WriteProperty(3, "flight_port", handle.flight_port);
			obj.WriteList(4, "files", handle.files.size(), [&](Serializer::List &files, idx_t file_idx) {
				files.WriteObject([&](Serializer &file_obj) {
					const auto &file = handle.files[file_idx];
					file_obj.WriteProperty(1, "path", file.path);
					file_obj.WriteProperty(2, "file_size", file.file_size);
					file_obj.WriteProperty(3, "rows", file.rows);
				});
			});
			obj.WriteProperty(5, "attempt_id", handle.attempt_id);
		});
	});
	serializer.WriteProperty(3, "source_partition_count", source_partition_count);
	serializer.WriteProperty(4, "source_task_count", source_task_count);
	serializer.WritePropertyWithDefault<bool>(5, "replicated", replicated, false);
}

ExchangeSourceTaskDescriptor ExchangeSourceTaskDescriptor::Deserialize(Deserializer &deserializer) {
	ExchangeSourceTaskDescriptor result;
	result.partition_indices = deserializer.ReadProperty<vector<idx_t>>(1, "partition_indices");
	deserializer.ReadList(2, "source_handles", [&](Deserializer::List &list, idx_t) {
		ExchangeSourceHandle handle;
		list.ReadObject([&](Deserializer &obj) {
			handle.partition_id = obj.ReadProperty<idx_t>(1, "partition_id");
			handle.node_id = obj.ReadProperty<string>(2, "node_id");
			handle.flight_port = obj.ReadPropertyWithExplicitDefault<int>(3, "flight_port", 0);
			obj.ReadList(4, "files", [&](Deserializer::List &files, idx_t) {
				ExchangeSourceFile file;
				files.ReadObject([&](Deserializer &file_obj) {
					file.path = file_obj.ReadProperty<string>(1, "path");
					file.file_size = file_obj.ReadProperty<size_t>(2, "file_size");
					file_obj.ReadPropertyWithDefault<idx_t>(3, "rows", file.rows);
				});
				handle.files.push_back(std::move(file));
			});
			handle.attempt_id = obj.ReadPropertyWithExplicitDefault<idx_t>(5, "attempt_id", 0);
		});
		result.source_handles.push_back(std::move(handle));
	});
	deserializer.ReadPropertyWithDefault<idx_t>(3, "source_partition_count", result.source_partition_count);
	deserializer.ReadPropertyWithDefault<idx_t>(4, "source_task_count", result.source_task_count);
	deserializer.ReadPropertyWithDefault<bool>(5, "replicated", result.replicated);
	if (result.source_partition_count == 0 && !result.partition_indices.empty()) {
		for (auto partition_idx : result.partition_indices) {
			result.source_partition_count = std::max(result.source_partition_count, partition_idx + 1);
		}
	}
	if (result.source_task_count == 0 && result.source_partition_count != 0) {
		result.source_task_count = result.source_partition_count;
	}
	return result;
}

std::string ExchangeSourceTaskDescriptor::SerializeToBytes() const {
	MemoryStream stream(Allocator::DefaultAllocator());
	BinarySerializer serializer(stream);
	serializer.Begin();
	Serialize(serializer);
	serializer.End();
	return std::string(reinterpret_cast<const char *>(stream.GetData()), stream.GetPosition());
}

ExchangeSourceTaskDescriptor ExchangeSourceTaskDescriptor::DeserializeFromBytes(const std::string &bytes) {
	if (bytes.empty()) {
		return ExchangeSourceTaskDescriptor();
	}
	auto *data_ptr = reinterpret_cast<data_ptr_t>(const_cast<char *>(bytes.data()));
	MemoryStream stream(data_ptr, bytes.size());
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto result = Deserialize(deserializer);
	deserializer.End();
	return result;
}

bool ApplyExchangeSourceTasksToPlan(duckdb::PhysicalPlan &plan,
                                    const std::unordered_map<idx_t, ExchangeSourceTaskDescriptor> &tasks,
                                    std::string *error) {
	if (!plan.HasRoot()) {
		if (error) {
			*error = "plan has no root";
		}
		return false;
	}
	idx_t applied = 0;
	return ApplyExchangeSourceTasksToOperator(plan.Root(), tasks, error, applied);
}

} // namespace distributed
} // namespace duckdb
