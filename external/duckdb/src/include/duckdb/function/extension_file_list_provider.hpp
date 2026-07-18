// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/extension_file_list_provider.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types.hpp"

#include <string>
#include <vector>

namespace duckdb {

//! ExtensionFileListProvider is an interface that extension bind data can
//! implement to participate in the distributed planner's file-based scan
//! task splitting.  Extensions whose table functions operate on a list of
//! files should inherit from this class **in addition to** FunctionData /
//! TableFunctionData so that MakeTableScanTasks and ApplyScanTasksToPlan
//! can discover them when the bind data is not a MultiFileBindData.
class ExtensionFileListProvider {
public:
	virtual ~ExtensionFileListProvider() = default;

	//! Return the full list of files that this scan covers.
	virtual vector<string> GetFileList() const = 0;

	//! Rewrite the bind data so that subsequent scans will read only
	//! the supplied subset of files.
	virtual void SetFileList(const vector<string> &files) = 0;
};

} // namespace duckdb
