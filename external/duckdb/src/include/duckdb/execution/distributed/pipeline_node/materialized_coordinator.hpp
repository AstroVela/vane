// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <memory>

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"

namespace duckdb {
namespace distributed {

class ExchangeManager;

using MaterializedPlanBuilder = std::function<DuckPhysicalPlanRef(DuckPhysicalPlanRef)>;
using PerTaskMaterializedPlanBuilderFactory = std::function<MaterializedPlanBuilder(idx_t)>;

bool ChildHasMultiplePartitions(const PipelineNodeRef &child);

SubmittableTaskStream<WorkerTask> ProduceWithMaterializedCoordinator(
    PlanExecutionContext &plan_context, const PipelineNodeRef &child, const std::shared_ptr<PipelineNodeImpl> &node,
    MaterializedPlanBuilder final_plan_builder, PerTaskMaterializedPlanBuilderFactory per_task_builder_factory = {},
    std::shared_ptr<ExchangeManager> exchange_mgr = nullptr);

} // namespace distributed
} // namespace duckdb
