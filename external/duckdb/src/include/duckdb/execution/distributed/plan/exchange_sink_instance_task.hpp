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
class PhysicalOperator;
class PhysicalPlan;
class PhysicalRemoteExchangeSink;
class Serializer;

namespace distributed {

struct ExchangeSinkInstanceTaskDescriptor {
	/// Concrete logical-task attempt bound by the task runner.
	ExchangeSinkInstanceHandle sink_instance;

	void Serialize(Serializer &serializer) const;
	static ExchangeSinkInstanceTaskDescriptor Deserialize(Deserializer &deserializer);
	std::string SerializeToBytes() const;
	static ExchangeSinkInstanceTaskDescriptor DeserializeFromBytes(const std::string &bytes);
};

//! Inspect a worker plan without accepting ambiguous sink ownership. A plan
//! may have no remote sink, but it may never have more than one.
bool TryGetUniqueRemoteExchangeSink(const PhysicalOperator &op, const PhysicalRemoteExchangeSink *&sink,
                                    std::string *error = nullptr);

bool ApplyExchangeSinkInstanceToPlan(duckdb::PhysicalPlan &plan, const ExchangeSinkInstanceTaskDescriptor &task,
                                     std::string *error = nullptr);

} // namespace distributed
} // namespace duckdb
