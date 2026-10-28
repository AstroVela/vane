// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/operator/helper/physical_result_collector.hpp"

namespace duckdb {

//! Receives native flow into sealed, spillable BufferManager allocations.
class PhysicalMediaMaterializedCollector : public PhysicalResultCollector {
public:
	PhysicalMediaMaterializedCollector(PhysicalPlan &physical_plan, PreparedStatementData &data);

	unique_ptr<QueryResult> GetResult(GlobalSinkState &state) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;

	bool ParallelSink() const override {
		// One bounded staging copy and no out-of-order partitions holding pins.
		return false;
	}
	bool SinkOrderDependent() const override {
		return true;
	}
	bool SupportsMediaFlowInput() const override {
		return true;
	}
};

} // namespace duckdb
