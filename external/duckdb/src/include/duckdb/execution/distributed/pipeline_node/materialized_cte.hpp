// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"

namespace duckdb {
namespace distributed {

class ExchangeManager;

DistributedPipelineNodeRef MakeMaterializedCTEInput(PipelineNodeContext context, DistributedPipelineNodeRef definition,
                                                    std::shared_ptr<ExchangeManager> manager, bool preserve_order);
DistributedPipelineNodeRef MakeMaterializedCTEScan(PipelineNodeContext context, DistributedPipelineNodeRef input,
                                                   const vector<LogicalType> &scan_types);
DistributedPipelineNodeRef MakeMaterializedCTEScope(PipelineNodeContext context, DistributedPipelineNodeRef input,
                                                    DistributedPipelineNodeRef body);

} // namespace distributed
} // namespace duckdb
