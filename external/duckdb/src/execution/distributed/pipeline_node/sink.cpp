// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/sink.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/copy_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"

#include <mutex>
#include <unordered_map>

namespace duckdb {
namespace distributed {

static DuckPhysicalPlanRef AppendCopyOperator(DuckPhysicalPlanRef plan, DistributedCopySpec spec,
                                              const std::string &task_path) {
	if (!plan || !plan->HasRoot()) {
		throw InvalidInputException("CopySinkNode: input plan missing root");
	}
	if (!spec.bind_data) {
		throw InvalidInputException("CopySinkNode: copy bind_data is null");
	}
	auto &old_root = plan->Root();
	auto worker_return_type = CopyFunctionReturnType::WRITTEN_FILE_STATISTICS;
	auto types = GetCopyFunctionReturnLogicalTypes(worker_return_type);

	if (spec.type == DistributedCopyType::COPY_TO_FILE) {
		auto &copy_op = plan->Make<duckdb::PhysicalCopyToFile>(types, spec.function, std::move(spec.bind_data),
		                                                       old_root.estimated_cardinality);
		auto &cast_copy = copy_op.Cast<duckdb::PhysicalCopyToFile>();
		cast_copy.file_path = task_path;
		// Distributed COPY uses a staging directory; avoid per-task tmp renames.
		cast_copy.use_tmp_file = false;
		cast_copy.filename_pattern = spec.filename_pattern;
		cast_copy.file_extension = spec.file_extension;
		cast_copy.overwrite_mode = spec.overwrite_mode;
		cast_copy.parallel = spec.parallel;
		cast_copy.per_thread_output = spec.per_thread_output;
		cast_copy.file_size_bytes = spec.file_size_bytes;
		cast_copy.rotate = spec.rotate;
		cast_copy.return_type = worker_return_type;
		cast_copy.partition_output = spec.partition_output;
		cast_copy.write_partition_columns = spec.write_partition_columns;
		cast_copy.write_empty_file = spec.write_empty_file;
		cast_copy.hive_file_pattern = spec.hive_file_pattern;
		cast_copy.partition_columns = spec.partition_columns;
		cast_copy.names = spec.names;
		cast_copy.expected_types = spec.expected_types;
		cast_copy.children.push_back(old_root);
		plan->SetRoot(copy_op);
		return plan;
	}

	auto &copy_op = plan->Make<duckdb::PhysicalBatchCopyToFile>(types, spec.function, std::move(spec.bind_data),
	                                                            old_root.estimated_cardinality);
	auto &cast_copy = copy_op.Cast<duckdb::PhysicalBatchCopyToFile>();
	cast_copy.file_path = task_path;
	// Distributed COPY uses a staging directory; avoid per-task tmp renames.
	cast_copy.use_tmp_file = false;
	cast_copy.return_type = worker_return_type;
	cast_copy.write_empty_file = spec.write_empty_file;
	cast_copy.children.push_back(old_root);
	plan->SetRoot(copy_op);
	return plan;
}

SubmittableTaskStream<WorkerTask> CopySinkNode::produce_tasks(PlanExecutionContext &plan_context) {
	auto input_stream = child_->produce_tasks(plan_context);
	auto self = shared_from_this();
	auto node_id_val = this->node_id();
	auto node_ctx = context().to_hashmap();
	auto staging_root_base = staging_root_base_;
	auto staging_run_id = staging_run_id_;
	auto *client_context = plan_context.client_context();
	auto fragment_plan_cache = std::make_shared<std::unordered_map<const PhysicalPlan *, DuckPhysicalPlanRef>>();
	auto fragment_plan_cache_lock = std::make_shared<std::mutex>();

	if (!client_context) {
		throw InvalidInputException("CopySinkNode requires ClientContext for plan cloning");
	}

	return input_stream.map_tasks([self, node_id_val, node_ctx, staging_root_base, staging_run_id, client_context,
	                               fragment_plan_cache,
	                               fragment_plan_cache_lock](SubmittableTask<WorkerTask> task) mutable {
		auto *old_task = task.task();
		if (!old_task) {
			throw InvalidInputException("CopySinkNode: task missing");
		}

		const auto task_id = old_task->task_context().task_id();

		auto base_plan = old_task->plan();
		DuckPhysicalPlanRef fragment_plan;
		{
			std::lock_guard<std::mutex> guard(*fragment_plan_cache_lock);
			auto it = fragment_plan_cache->find(base_plan.get());
			if (it != fragment_plan_cache->end()) {
				fragment_plan = it->second;
			}
		}
		if (!fragment_plan) {
			auto local_spec = self->spec_.Clone();
			auto plan_template_path = BuildCopyPlanTemplatePath(local_spec, node_id_val);
			auto cache_key = base_plan.get();
			DuckPhysicalPlanRef working_plan;
			auto rc = base_plan.use_count();
			if (rc <= 2) {
				working_plan = std::move(base_plan);
			} else {
				working_plan = ClonePhysicalPlanOrThrow(base_plan, "CopySinkNode", client_context);
			}
			auto candidate_plan =
			    AppendCopyOperator(std::move(working_plan), std::move(local_spec), plan_template_path);
			{
				std::lock_guard<std::mutex> guard(*fragment_plan_cache_lock);
				auto emplace_result = fragment_plan_cache->emplace(cache_key, candidate_plan);
				fragment_plan = emplace_result.first->second;
			}
		}

		TaskContext ctx = old_task->task_context();
		ctx.add_node_id(node_id_val);
		auto merged_ctx = MergeTaskContext(old_task->context(), node_ctx);
		merged_ctx["copy_output_base"] = staging_root_base;
		merged_ctx["copy_output_run_id"] = staging_run_id;
		merged_ctx["copy_output_remote_base"] = self->spec_.file_path;
		WorkerTask new_task(ctx, fragment_plan, old_task->config(), std::move(merged_ctx));
		new_task.mutable_inputs() = old_task->inputs();
		return SubmittableTask<WorkerTask>(std::move(new_task));
	});
}

} // namespace distributed
} // namespace duckdb
