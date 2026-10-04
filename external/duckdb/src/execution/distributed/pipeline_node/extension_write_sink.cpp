// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/extension_write_sink.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/execution/operator/persistent/physical_distributed_extension_write.hpp"
#include "duckdb/main/client_context.hpp"

namespace duckdb {
namespace distributed {

namespace {

static DuckPhysicalPlanRef AppendExtensionWriteOperator(DuckPhysicalPlanRef plan,
                                                        const DistributedExtensionWriteInfo &info) {
	if (!plan || !plan->HasRoot()) {
		throw InvalidInputException("ExtensionWriteSinkNode input plan is missing its root");
	}
	auto &old_root = plan->Root();
	auto &write = plan->Make<PhysicalDistributedExtensionWrite>(info, old_root.estimated_cardinality);
	write.children.push_back(old_root);
	plan->SetRoot(write);
	return plan;
}

} // namespace

ExtensionWriteSinkNode::ExtensionWriteSinkNode(NodeID node_id, PipelineNodeRef child,
                                               DistributedExtensionWriteInfo info)
    : ctx_(InheritPipelineNodeContext(child, node_id, "ExtensionWriteSink")),
      config_(child ? child->config() : PipelineNodeConfig()), child_(std::move(child)), info_(std::move(info)) {
	if (!child_) {
		throw InvalidInputException("ExtensionWriteSinkNode requires a child");
	}
	info_.Validate();
	if (info_.mode != DistributedWriteMode::CALLBACK_SINK) {
		throw InvalidInputException("ExtensionWriteSinkNode requires callback write mode");
	}
}

SubmittableTaskStream<WorkerTask> ExtensionWriteSinkNode::produce_tasks(PlanExecutionContext &plan_context) {
	auto input_stream = child_->produce_tasks(plan_context);
	auto self = shared_from_this();
	auto node_id_value = node_id();
	auto node_context = context().to_hashmap();
	auto *client_context = plan_context.client_context();
	auto fragment_plan_cache = make_shared_ptr<unordered_map<DuckPhysicalPlanRef, DuckPhysicalPlanRef>>();
	auto fragment_plan_cache_lock = make_shared_ptr<mutex>();
	if (!client_context) {
		throw InvalidInputException("ExtensionWriteSinkNode requires ClientContext for plan cloning");
	}

	return input_stream.map_tasks([self, node_id_value, node_context, client_context, fragment_plan_cache,
	                               fragment_plan_cache_lock](SubmittableTask<WorkerTask> task) mutable {
		auto *old_task = task.task();
		if (!old_task) {
			throw InvalidInputException("ExtensionWriteSinkNode task is missing");
		}
		auto base_plan = old_task->plan();
		DuckPhysicalPlanRef fragment_plan;
		{
			lock_guard<mutex> guard(*fragment_plan_cache_lock);
			auto entry = fragment_plan_cache->find(base_plan);
			if (entry != fragment_plan_cache->end()) {
				fragment_plan = entry->second;
			}
		}
		if (!fragment_plan) {
			const bool can_reuse_base_plan = base_plan.use_count() <= 2;
			auto cache_key = base_plan;
			DuckPhysicalPlanRef working_plan;
			if (can_reuse_base_plan) {
				working_plan = std::move(base_plan);
			} else {
				working_plan = ClonePhysicalPlanOrThrow(base_plan, "ExtensionWriteSinkNode", client_context);
			}
			auto candidate = AppendExtensionWriteOperator(std::move(working_plan), self->info_);
			lock_guard<mutex> guard(*fragment_plan_cache_lock);
			auto inserted = fragment_plan_cache->emplace(cache_key, candidate);
			fragment_plan = inserted.first->second;
		}

		auto task_context = old_task->task_context();
		task_context.add_node_id(node_id_value);
		auto merged_context = MergeTaskContext(old_task->context(), node_context);
		WorkerTask result(task_context, fragment_plan, old_task->config(), std::move(merged_context));
		result.mutable_inputs() = old_task->inputs();
		return SubmittableTask<WorkerTask>(std::move(result));
	});
}

std::vector<std::string> ExtensionWriteSinkNode::multiline_display(bool) const {
	return {"ExtensionWriteSink: " + info_.Name(), "Codec: " + info_.fragment_codec.CanonicalIdentity()};
}

} // namespace distributed
} // namespace duckdb
