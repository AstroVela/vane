// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// vane_python/pandas_scan.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "vane_python/pandas/pandas_bind.hpp"

#include "vane_python/pybind11/pybind_wrapper.hpp"

namespace duckdb {

struct PandasScanFunction : public TableFunction {
public:
	static constexpr idx_t PANDAS_PARTITION_COUNT = 50 * STANDARD_VECTOR_SIZE;

public:
	PandasScanFunction();

	static unique_ptr<FunctionData> PandasScanBind(ClientContext &context, TableFunctionBindInput &input,
	                                               vector<LogicalType> &return_types, vector<string> &names);

	static unique_ptr<GlobalTableFunctionState> PandasScanInitGlobal(ClientContext &context,
	                                                                 TableFunctionInitInput &input);
	static unique_ptr<LocalTableFunctionState>
	PandasScanInitLocal(ExecutionContext &context, TableFunctionInitInput &input, GlobalTableFunctionState *gstate);

	static idx_t PandasScanMaxThreads(ClientContext &context, const FunctionData *bind_data_p);

	static bool PandasScanParallelStateNext(ClientContext &context, const FunctionData *bind_data_p,
	                                        LocalTableFunctionState *lstate, GlobalTableFunctionState *gstate);

	static double PandasProgress(ClientContext &context, const FunctionData *bind_data_p,
	                             const GlobalTableFunctionState *gstate);

	//! The main pandas scan function: note that this can be called in parallel without the GIL
	//! hence this needs to be GIL-safe, i.e. no methods that create Python objects are allowed
	static void PandasScanFunc(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);

	static unique_ptr<NodeStatistics> PandasScanCardinality(ClientContext &context, const FunctionData *bind_data);

	static OperatorPartitionData PandasScanGetPartitionData(ClientContext &context,
	                                                        TableFunctionGetPartitionInput &input);

	// Helper function that transform pandas df names to make them work with our binder
	static py::object PandasReplaceCopiedNames(const py::object &original_df);

	//! Return the DataFrame retained by bound pandas_scan data.
	//! Used by the Ray planner to create a process-independent Arrow snapshot.
	static py::object GetDataFrame(const FunctionData &bind_data);

	//! Return the original replacement-scan object when available.
	//! Separate bindings create distinct shallow DataFrame copies, so the Ray
	//! planner uses this object to deduplicate snapshots by source identity.
	static py::object GetDataFrameSourceIdentity(const FunctionData &bind_data);

	static void PandasBackendScanSwitch(PandasColumnBindData &bind_data, idx_t count, idx_t offset, Vector &out);

	static void PandasSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data,
	                            const TableFunction &function);

	static unique_ptr<FunctionData> PandasDeserialize(Deserializer &deserializer, TableFunction &function);
};

} // namespace duckdb
