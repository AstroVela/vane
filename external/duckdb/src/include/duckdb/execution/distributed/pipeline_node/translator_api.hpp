// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <unordered_map>
#include <string>
#include <vector>

#include "duckdb/execution/distributed/plan/plan_config.hpp"
#include "duckdb/execution/distributed/plan/distributed_physical_plan.hpp"
#include "duckdb/execution/distributed/plan/scan_task.hpp"

namespace duckdb {
class ClientContext;
class DatabaseInstance;
namespace distributed {

// Lightweight wrapper API to translate a DuckDB physical plan into a
// DistributedPipelineNode. Implemented in translate.cpp and intentionally
// kept as a thin forwarding function so heavy translator header does not
// need to be included in header-only or widely-included files (avoids
// unity-build include ordering issues).
DuckDBResult<std::shared_ptr<DistributedPipelineNode>>
physical_plan_to_pipeline_node_wrapper(PlanConfig plan_config, DuckPhysicalPlanRef plan,
                                       ClientContext *client_context = nullptr);

std::unordered_map<idx_t, std::vector<ScanTaskDescriptor>>
physical_plan_scan_task_map_wrapper(DuckPhysicalPlanRef plan, DuckDBExecutionConfigRef config,
                                    shared_ptr<DatabaseInstance> db);

} // namespace distributed
} // namespace duckdb
