// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>

#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/execution/distributed/plan/plan_config.hpp"
#include "duckdb/execution/distributed/plan/scan_task.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"

namespace duckdb {
class DatabaseInstance;
class PhysicalPlan;

namespace distributed {

DuckPhysicalPlanRef MakeTableScanPlan(const PhysicalTableScan &scan);

std::vector<ScanTaskDescriptor> MakeTableScanTasks(const PhysicalTableScan &scan, const DuckDBExecutionConfig &exec_cfg,
                                                   const shared_ptr<DatabaseInstance> &db);

SchemaRef MakeTableScanSchema(const PhysicalTableScan &scan, const vector<LogicalType> &output_types);

} // namespace distributed
} // namespace duckdb
