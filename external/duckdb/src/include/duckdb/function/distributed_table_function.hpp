// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/distributed_table_function.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/column_index.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/optional_ptr.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/main/distributed_extension_manager.hpp"

namespace duckdb {

class FunctionData;
class TableFilterSet;

//! One elementary, portable unit of work produced by an extension on the
//! coordinator. Vane never interprets the payload bytes.
struct DistributedScanTask {
	string task_id;
	string payload;
	optional_idx estimated_cardinality;
	optional_idx estimated_bytes;

	DUCKDB_API void Validate() const;
	DUCKDB_API void Serialize(Serializer &serializer) const;
	DUCKDB_API static DistributedScanTask Deserialize(Deserializer &deserializer);
};

//! Physical scan information available while an extension creates its task
//! envelopes or detaches a worker bind-data copy.
struct TableFunctionDistributedScanInput {
	TableFunctionDistributedScanInput(const FunctionData &bind_data_p, const vector<ColumnIndex> &column_ids_p,
	                                  const vector<idx_t> &projection_ids_p,
	                                  optional_ptr<const TableFilterSet> table_filters_p, idx_t estimated_cardinality_p)
	    : bind_data(bind_data_p), column_ids(column_ids_p), projection_ids(projection_ids_p),
	      table_filters(table_filters_p), estimated_cardinality(estimated_cardinality_p) {
	}

	const FunctionData &bind_data;
	const vector<ColumnIndex> &column_ids;
	const vector<idx_t> &projection_ids;
	optional_ptr<const TableFilterSet> table_filters;
	idx_t estimated_cardinality;
};

//! Plan extension-owned elementary tasks as opaque envelopes.
typedef vector<DistributedScanTask> (*table_function_plan_distributed_scan_t)(
    const TableFunctionDistributedScanInput &input);

//! Remove coordinator-only task state from a copied bind object before that
//! object is transported with the worker physical plan. This is not a rebind:
//! the ordinary table-function serialize callback still serializes the copy.
typedef void (*table_function_prepare_distributed_scan_bind_t)(const TableFunctionDistributedScanInput &input,
                                                               FunctionData &worker_bind_data);

//! Decode and install the assigned opaque envelopes into a deserialized worker
//! bind object. An empty vector must install an empty scan.
typedef void (*table_function_apply_distributed_scan_tasks_t)(FunctionData &worker_bind_data,
                                                              const vector<DistributedScanTask> &tasks);

//! Complete distributed scan contract attached to a TableFunction. Extension
//! authors declare only the capability protocol and task codec here. The
//! ExtensionLoader derives the extension and function identity from the normal
//! DuckDB registration and binds the complete capability before publication.
//! The complete capability and codec identity are transported in every task
//! descriptor and must match exactly on the worker.
struct TableFunctionDistributedScanCallbacks {
	idx_t protocol_version = 0;
	DistributedPayloadCodec task_codec;
	table_function_plan_distributed_scan_t plan = nullptr;
	table_function_prepare_distributed_scan_bind_t prepare_bind = nullptr;
	table_function_apply_distributed_scan_tasks_t apply_tasks = nullptr;

	DUCKDB_API void ValidateDefinition(const string &function_name) const;
	DUCKDB_API void Validate(const string &function_name) const;
	DUCKDB_API void BindCapability(const string &extension_name, const string &function_name);
	DUCKDB_API const DistributedExtensionCapabilityReference &GetCapability() const;
	DUCKDB_API bool operator==(const TableFunctionDistributedScanCallbacks &other) const;

private:
	DistributedExtensionCapabilityReference capability;
};

} // namespace duckdb
