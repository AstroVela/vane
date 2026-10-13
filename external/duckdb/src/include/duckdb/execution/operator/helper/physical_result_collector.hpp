// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/helper/physical_result_collector.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/common/enums/statement_type.hpp"

namespace duckdb {

class PreparedStatementData;
class ColumnDataCollection;

//! PhysicalResultCollector is an abstract class that is used to generate the final result of a query
class PhysicalResultCollector : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::RESULT_COLLECTOR;

public:
	PhysicalResultCollector(PhysicalPlan &physical_plan, PreparedStatementData &data);

	StatementType statement_type;
	StatementProperties properties;
	QueryResultMemoryType memory_type;
	PhysicalOperator &plan;
	vector<string> names;
	//! Opt in to flushing partial output between source batches, even when
	//! intermediate filters produce no rows. Ordinary collectors do not opt in.
	bool requires_periodic_flush = false;

public:
	static unique_ptr<PhysicalOperator> GetResultCollector(ClientContext &context, PreparedStatementData &data);

public:
	//! The final method used to fetch the query result from this operator
	virtual unique_ptr<QueryResult> GetResult(GlobalSinkState &state) const = 0;

	//! Runs on the pipeline thread, with the same local state as Sink. BLOCKED
	//! must register a wakeup and retain its pending output for a retry. A source
	//! about to block forces a flush regardless of the collector's time budget.
	virtual SinkResultType FlushPending(ExecutionContext &, OperatorSinkInput &, bool force) const {
		return SinkResultType::NEED_MORE_INPUT;
	}

	bool IsSink() const override {
		return true;
	}

public:
	vector<const_reference<PhysicalOperator>> GetChildren() const override;
	void BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) override;

	bool IsSource() const override {
		return true;
	}

public:
	//! Whether this is a streaming result collector
	virtual bool IsStreaming() const {
		return false;
	}

protected:
	unique_ptr<ColumnDataCollection> CreateCollection(ClientContext &context) const;
};

} // namespace duckdb
