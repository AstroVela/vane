// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/projection/physical_udf_inout.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/optional_idx.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/function/table_function.hpp"
#include <mutex>

namespace duckdb {

struct StreamingUDFState;

class PhysicalStreamingUDF : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::STREAMING_UDF;

public:
	PhysicalStreamingUDF(PhysicalPlan &physical_plan, vector<LogicalType> types, TableFunction function_p,
	                     unique_ptr<FunctionData> bind_data_p, vector<ColumnIndex> column_ids_p,
	                     idx_t estimated_cardinality, vector<column_t> projected_input);

public:
	bool ParallelOperator() const override {
		return true;
	}

	bool IsSink() const override {
		return true;
	}

	bool IsSource() const override {
		return true;
	}

	bool ParallelSink() const override {
		return true;
	}

	bool ParallelSource() const override {
		return true;
	}

	OrderPreservationType SourceOrder() const override {
		return OrderPreservationType::NO_ORDER;
	}

	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	ProgressData GetSinkProgress(ClientContext &context, GlobalSinkState &gstate,
	                             const ProgressData source_progress) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkResultType SinkBatch(ExecutionContext &context, ExecutionBatch &batch, OperatorSinkInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;

	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                 GlobalSourceState &gstate) const override;
	ProgressData GetProgress(ClientContext &context, GlobalSourceState &gstate) const override;
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;
	SourceResultType GetDataBatch(ExecutionContext &context, ExecutionBatch &batch,
	                              OperatorSourceInput &input) const override;
	void BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) override;

	const TableFunction &GetFunction() const {
		return function;
	}
	const FunctionData *GetBindData() const {
		return bind_data.get();
	}
	const vector<ColumnIndex> &GetColumnIds() const {
		return column_ids;
	}
	const vector<column_t> &GetProjectedInput() const {
		return projected_input;
	}

	InsertionOrderPreservingMap<string> ParamsToString() const override;
	void SerializeOperatorData(Serializer &serializer) const override;

	optional_idx ordinality_idx;

private:
	std::shared_ptr<StreamingUDFState> GetStreamingState(ClientContext &context) const;

private:
	TableFunction function;
	unique_ptr<FunctionData> bind_data;
	vector<ColumnIndex> column_ids;
	vector<column_t> projected_input;

	mutable std::shared_ptr<StreamingUDFState> streaming_state;
	mutable std::mutex streaming_state_lock;
};

// Create a TableFunction with INOUT callbacks for udf execution.
// Used by the planner to create PhysicalTableInOutFunction.
TableFunction MakeUDFTableFunction(Value payload, const vector<LogicalType> &return_types,
                                   const vector<string> &return_names);

// Create a named, registered TableFunction with INOUT callbacks.
// Used by conn.create_function() to register a named UDF in the catalog.
TableFunction MakeUDFRegisteredTableFunction(string name, Value payload, vector<LogicalType> output_types,
                                             vector<string> output_names);

// Return a minimal udf TableFunction for built-in catalog registration.
// Required so BinaryDeserializer can look up the table function by name when
// reconstructing plans on remote workers (e.g., Ray).
TableFunction GetUDFBuiltinTableFunction();

} // namespace duckdb
