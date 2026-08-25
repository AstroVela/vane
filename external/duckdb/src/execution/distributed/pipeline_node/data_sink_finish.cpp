// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/data_sink_finish.hpp"

#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/execution/operator/helper/physical_data_sink.hpp"

namespace duckdb {
namespace distributed {

SubmittableTaskStream<WorkerTask> DataSinkFinishNode::produce_tasks(PlanExecutionContext &plan_context) {
	auto input_stream = child_->produce_tasks(plan_context);
	auto self = shared_from_this();
	auto operation_id = operation_id_;
	return input_stream.pipeline_instruction(
	    self,
	    [operation_id = std::move(operation_id)](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
		    auto &old_root = input_plan->Root();
		    auto output_types = old_root.GetTypes();
		    auto estimated_cardinality = old_root.estimated_cardinality;
		    auto &sink = input_plan->Make<::duckdb::PhysicalDataSink>(std::move(output_types), operation_id,
		                                                              estimated_cardinality);
		    sink.children.push_back(old_root);
		    input_plan->SetRoot(sink);
		    return input_plan;
	    },
	    plan_context.client_context());
}

} // namespace distributed
} // namespace duckdb
