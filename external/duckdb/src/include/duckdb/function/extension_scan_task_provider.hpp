// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/extension_scan_task_provider.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/common/named_parameter_map.hpp"
#include "duckdb/common/open_file_info.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/optional_ptr.hpp"

namespace duckdb {

//! ExtensionScanTaskProvider lets an extension retain ownership of its scan
//! state while Vane schedules portable pieces of that state independently.
//!
//! A task commonly represents a file. Non-file extensions may instead encode
//! an opaque portable token in OpenFileInfo::path and scalar metadata in
//! ExtendedOpenFileInfo::options. The scheduler does not interpret the token.
//!
//! The interface may be implemented directly by table-function bind data or by
//! the custom MultiFileList held by MultiFileBindData.
class ExtensionScanTaskProvider {
public:
	virtual ~ExtensionScanTaskProvider() = default;

	//! Expand the complete logical scan into elementary portable tasks.
	virtual vector<OpenFileInfo> GetScanTasks() const = 0;

	//! Restrict freshly rebound worker state to the assigned tasks. An empty
	//! vector must produce an empty scan rather than remove the restriction.
	virtual void SetScanTasks(const vector<OpenFileInfo> &tasks) = 0;

	//! Optional provider-owned estimates used for task balancing. The engine
	//! never opens provider task paths to infer these values because a path may
	//! be an opaque token rather than a filesystem location.
	virtual optional_idx GetScanTaskEstimatedBytes(const OpenFileInfo &) const {
		return optional_idx();
	}
	virtual optional_idx GetScanTaskEstimatedCardinality(const OpenFileInfo &) const {
		return optional_idx();
	}

	//! Normalize bind parameters before the worker plan is serialized. This is
	//! used to materialize positional inputs for catalog-backed scans and replace
	//! moving references such as "latest" with immutable IDs. Logical-plan and
	//! physical-plan transport are separate boundaries, so the transformation
	//! must be deterministic and safe when its input is already normalized.
	virtual void PrepareWorkerBind(vector<Value> &parameters, named_parameter_map_t &named_parameters) const {
	}
};

inline optional_ptr<ExtensionScanTaskProvider> TryGetExtensionScanTaskProvider(FunctionData &bind_data) {
	auto provider = dynamic_cast<ExtensionScanTaskProvider *>(&bind_data);
	if (provider) {
		return provider;
	}
	auto multi_bind = dynamic_cast<MultiFileBindData *>(&bind_data);
	if (!multi_bind || !multi_bind->file_list) {
		return nullptr;
	}
	return dynamic_cast<ExtensionScanTaskProvider *>(multi_bind->file_list.get());
}

inline optional_ptr<const ExtensionScanTaskProvider> TryGetExtensionScanTaskProvider(const FunctionData &bind_data) {
	auto provider = dynamic_cast<const ExtensionScanTaskProvider *>(&bind_data);
	if (provider) {
		return provider;
	}
	auto multi_bind = dynamic_cast<const MultiFileBindData *>(&bind_data);
	if (!multi_bind || !multi_bind->file_list) {
		return nullptr;
	}
	return dynamic_cast<const ExtensionScanTaskProvider *>(multi_bind->file_list.get());
}

} // namespace duckdb
