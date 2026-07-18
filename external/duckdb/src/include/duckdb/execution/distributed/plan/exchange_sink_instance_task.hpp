// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/distributed/plan/exchange_sink_instance_task.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types.hpp"
#include "duckdb/execution/distributed/exchange/exchange_handles.hpp"

#include <string>

namespace duckdb {

class Deserializer;
class PhysicalPlan;
class Serializer;

namespace distributed {

struct ExchangeSinkInstanceTaskDescriptor {
	ExchangeSinkInstanceHandle sink_instance;

	void Serialize(Serializer &serializer) const;
	static ExchangeSinkInstanceTaskDescriptor Deserialize(Deserializer &deserializer);
	std::string SerializeToBytes() const;
	static ExchangeSinkInstanceTaskDescriptor DeserializeFromBytes(const std::string &bytes);
};

bool ApplyExchangeSinkInstanceToPlan(duckdb::PhysicalPlan &plan, const ExchangeSinkInstanceTaskDescriptor &task,
                                     std::string *error = nullptr);

} // namespace distributed
} // namespace duckdb
