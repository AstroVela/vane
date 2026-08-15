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

//! Stable coordinator identity for one distributed write operation. It remains
//! available even when no worker result envelope was produced, so a provider
//! can identify the speculative artifact namespace for validation, finalize,
//! and abort.
struct DistributedWriteOperationContext {
	string operation_id;

	DUCKDB_API void Validate() const;
};

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
	//! callback can run. This hook neither commits nor reconciles an earlier
	//! attempt and must not create artifacts. A validation failure does not
	//! authorize aborting an earlier attempt with the same operation identity.
	virtual void ValidateDistributedWrite(ClientContext &context,
	                                      const DistributedWriteOperationContext &operation) const = 0;

	//! Commit the selected task results in the active coordinator transaction.
	//! The extension's catalog is authoritative for transaction atomicity and
	//! same-operation idempotence, including after an earlier commit response was
	//! lost. A repeated call may contain new task attempts; it must never replace
	//! artifacts referenced by an existing committed operation. Returns the number
	//! of affected rows represented by the selected results.
	virtual idx_t FinalizeDistributedWrite(ClientContext &context, const DistributedWriteOperationContext &operation,
	                                       const vector<DistributedWriteTaskResult> &results) const = 0;

	//! Remove uncommitted artifacts from the current attempt after a known
	//! pre-commit failure. The extension must consult its authoritative catalog
	//! state and preserve artifacts referenced by any committed attempt with the
	//! same operation identity. It must support cleanup even when no worker
	//! envelope was returned.
	virtual void AbortDistributedWrite(ClientContext &context, const DistributedWriteOperationContext &operation,
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
EncodeDistributedFileWriteResults(const DistributedExtensionWriteInfo &info,
                                  const DistributedWriteOperationContext &operation,
                                  const vector<DistributedCopyFileInfo> &files);
DUCKDB_API vector<DistributedCopyFileInfo>
DecodeDistributedFileWriteResults(const DistributedExtensionWriteInfo &info,
                                  const DistributedWriteOperationContext &operation,
                                  const vector<DistributedWriteTaskResult> &results);

//! Decode the one-column BLOB output produced by
//! PhysicalDistributedExtensionWrite and enforce the coordinator plan's exact
//! capability and fragment codec.
DUCKDB_API vector<DistributedWriteTaskResult>
ParseDistributedWriteTaskResults(const DistributedExtensionWriteInfo &info,
                                 const DistributedWriteOperationContext &operation,
                                 const vector<ResultPartitionRef> &partitions);

} // namespace distributed
} // namespace duckdb
