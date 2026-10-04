// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/empty_result_source.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"

namespace duckdb {
namespace distributed {

EmptyResultSourceNode::EmptyResultSourceNode(PipelineNodeContext context, DuckPhysicalPlanRef empty_plan,
                                             SchemaRef schema, DuckDBExecutionConfigRef execution_config)
    : context_(std::move(context)),
      config_(std::move(schema), std::move(execution_config), ClusteringSpec::unknown_with_num_partitions(1)),
      empty_plan_(std::move(empty_plan)) {
}

SubmittableTaskStream<WorkerTask> EmptyResultSourceNode::produce_tasks(PlanExecutionContext &plan_context) {
	if (!empty_plan_ || !empty_plan_->HasRoot()) {
		throw InternalException("EmptyResultSourceNode requires an empty-result physical plan");
	}

	auto channel = create_channel<SubmittableTask<WorkerTask>>(1);
	TaskContext task_context =
	    TaskContext::from_node_context(context_.query_idx(), node_id(), plan_context.task_id_counter().next());
	WorkerTask task(task_context, empty_plan_, config_.execution_config(), context_.to_hashmap());
	auto send_result = channel.first.send(SubmittableTask<WorkerTask>(std::move(task)));
	channel.first.close();
	if (send_result.is_err()) {
		throw InternalException("EmptyResultSourceNode failed to create its logical empty-input task");
	}
	return SubmittableTaskStream<WorkerTask>::from_receiver(std::move(channel.second));
}

std::vector<std::string> EmptyResultSourceNode::multiline_display(bool /*verbose*/) const {
	return {"EmptyResultSource"};
}

} // namespace distributed
} // namespace duckdb
