// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/distributed/plan/exchange_source_task.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types.hpp"
#include "duckdb/execution/distributed/exchange/exchange_handles.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace duckdb {

class Deserializer;
class PhysicalPlan;
class Serializer;

namespace distributed {

struct ExchangeSourceTaskDescriptor {
	vector<idx_t> partition_indices;
	std::vector<ExchangeSourceHandle> source_handles;
	idx_t source_partition_count = 0;
	idx_t source_task_count = 0;
	bool replicated = false;

	void Serialize(Serializer &serializer) const;
	static ExchangeSourceTaskDescriptor Deserialize(Deserializer &deserializer);
	std::string SerializeToBytes() const;
	static ExchangeSourceTaskDescriptor DeserializeFromBytes(const std::string &bytes);
};

bool ApplyExchangeSourceTasksToPlan(duckdb::PhysicalPlan &plan,
                                    const std::unordered_map<idx_t, ExchangeSourceTaskDescriptor> &tasks,
                                    std::string *error = nullptr);

} // namespace distributed
} // namespace duckdb
