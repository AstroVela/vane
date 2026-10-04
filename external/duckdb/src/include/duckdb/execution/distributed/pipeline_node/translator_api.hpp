// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <unordered_map>
#include <string>
#include <vector>

#include "duckdb/common/optional_ptr.hpp"
#include "duckdb/execution/distributed/plan/plan_config.hpp"
#include "duckdb/execution/distributed/plan/distributed_physical_plan.hpp"
#include "duckdb/execution/distributed/plan/scan_split.hpp"

namespace duckdb {
class ClientContext;
class DatabaseInstance;
struct DistributedExtensionWriteInfo;
namespace distributed {

// Lightweight wrapper API to translate a DuckDB physical plan into a
// DistributedPipelineNode. Implemented in translate.cpp and intentionally
// kept as a thin forwarding function so heavy translator header does not
// need to be included in header-only or widely-included files (avoids
// unity-build include ordering issues). A pre-resolved write protocol is
// borrowed only for this call and must come from
// ResolveDistributedExtensionWriteInfo for the physical extension root.
DuckDBResult<std::shared_ptr<DistributedPipelineNode>> physical_plan_to_pipeline_node_wrapper(
    PlanConfig plan_config, DuckPhysicalPlanRef plan, ClientContext *client_context = nullptr,
    optional_ptr<const DistributedExtensionWriteInfo> resolved_extension_write_info = nullptr);

std::unordered_map<idx_t, std::vector<ScanSplit>>
physical_plan_scan_split_map_wrapper(DuckPhysicalPlanRef plan, DuckDBExecutionConfigRef config,
                                     shared_ptr<DatabaseInstance> db, ClientContext *client_context = nullptr);

} // namespace distributed
} // namespace duckdb
