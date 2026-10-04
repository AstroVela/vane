// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/execution/distributed/copy_to_file.hpp"
#include "duckdb/function/distributed_write.hpp"

namespace duckdb {

class ClientContext;

namespace distributed {

//! Dynamic coordinator state supplied by one physical extension root. Static
//! mode, protocol, and codec information is resolved from the registered write
//! operator contract and cannot be overridden by a plan.
struct DistributedExtensionWritePlan {
	string extension_name;
	string operator_name;
	string worker_bind_data;

	DUCKDB_API void Validate() const;
};

//! Coordinator-side half of the explicit distributed extension write contract.
//! The ordinary extension operator remains authoritative for native DuckDB
//! execution. Vane replaces it only for Ray execution according to WritePlan().
class ExtensionWriteTaskProvider {
public:
	virtual ~ExtensionWriteTaskProvider() = default;

	//! Immutable extension/operator key and extension-owned worker bind envelope.
	virtual const DistributedExtensionWritePlan &WritePlan() const = 0;

	//! Read-only validation of coordinator/catalog preconditions before any worker
	//! callback can run. This hook must not create artifacts.
	virtual void ValidateDistributedWrite(ClientContext &context) const = 0;

	//! Create coordinator-owned state required before worker callbacks can run.
	//! Vane invokes this hook at most once after translation and coordinator setup.
	//! WritePlan is already frozen, so prepared state must be addressable through
	//! its immutable worker bind. Once invocation starts, a known failure before
	//! coordinator finalization invokes AbortDistributedWrite, including when no
	//! worker envelope exists.
	virtual void PrepareDistributedWrite(ClientContext &context) const {
	}

	//! Commit the selected task results in the active coordinator transaction.
	//! Vane invokes this hook at most once and never retries it. Once invocation
	//! starts, any failure is terminal with an unknown commit outcome and Vane
	//! does not call AbortDistributedWrite. The extension's catalog transaction
	//! is authoritative. Returns the represented affected-row count.
	virtual idx_t FinalizeDistributedWrite(ClientContext &context,
	                                       const vector<DistributedWriteTaskResult> &results) const = 0;

	//! Handle a failure that is known to precede coordinator finalization. The
	//! extension decides whether its catalog contract cleans, compensates, or
	//! retains prepared state and artifacts. This hook is never called once
	//! FinalizeDistributedWrite has started and must accept an empty result set.
	virtual void AbortDistributedWrite(ClientContext &context,
	                                   const vector<DistributedWriteTaskResult> &selected_results) const = 0;
};

//! Coordinator-visible result of an extension write. Callback writes carry
//! only extension-owned task envelopes; file-artifact writes additionally
//! carry DuckDB's explicit output publication lifecycle.
struct DistributedExtensionWriteResult {
	DistributedExtensionWriteInfo info;
	vector<DistributedWriteTaskResult> selected_task_results;
	DistributedCopyResult file_result;
	idx_t rows_written = 0;
	idx_t bytes_written = 0;
	bool catalog_committed = false;
	bool outcome_unknown = false;
	string outcome_error;
};

//! Fixed file adapter used by FILE_ARTIFACT writes.
static constexpr const char *DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC = "duckdb.written-file-statistics";
static constexpr idx_t DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC_VERSION = 1;

//! Resolve the complete immutable worker protocol from the database-local
//! concrete write registration and the physical operator's dynamic plan.
DUCKDB_API DistributedExtensionWriteInfo
ResolveDistributedExtensionWriteInfo(ClientContext &context, const DistributedExtensionWritePlan &plan);

DUCKDB_API vector<DistributedWriteTaskResult>
EncodeDistributedFileWriteResults(const DistributedExtensionWriteInfo &info, const string &query_id,
                                  const vector<DistributedCopyFileInfo> &files);
DUCKDB_API vector<DistributedCopyFileInfo>
DecodeDistributedFileWriteResults(const DistributedExtensionWriteInfo &info,
                                  const vector<DistributedWriteTaskResult> &results);

//! Decode the one-column BLOB output produced by
//! PhysicalDistributedExtensionWrite and enforce the coordinator plan's exact
//! capability and fragment codec.
DUCKDB_API vector<DistributedWriteTaskResult>
ParseDistributedWriteTaskResults(const DistributedExtensionWriteInfo &info, const string &query_id,
                                 const vector<ResultPartitionRef> &partitions);

} // namespace distributed
} // namespace duckdb
