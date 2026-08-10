// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/execution/distributed/copy_to_file.hpp"

namespace duckdb {

class ClientContext;

namespace distributed {

//! Coordinator-side contract for an extension write whose child produces
//! WRITTEN_FILE_STATISTICS through a distributed COPY operator.
//!
//! Workers only execute the COPY child. The provider remains on the
//! coordinator, validates the operation before any worker writes start, and
//! registers the selected files in the extension's active catalog transaction.
//! Vane owns an auto-commit transaction around finalization and publishes the
//! output marker only after that transaction commits.
class ExtensionWriteTaskProvider {
public:
	virtual ~ExtensionWriteTaskProvider() = default;

	//! Stable, human-readable extension write name used in diagnostics/results.
	virtual string ExtensionWriteName() const = 0;

	//! Validate the write before any distributed task can create output files.
	virtual void ValidateExtensionWrite(ClientContext &context) const = 0;

	//! Register the selected worker files in the active coordinator transaction.
	//! Returns the number of rows affected by the extension write.
	virtual idx_t FinalizeExtensionWrite(ClientContext &context,
	                                     const vector<DistributedCopyFileInfo> &files) const = 0;
};

} // namespace distributed
} // namespace duckdb
