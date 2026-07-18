// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/scan_source.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/common/string_util.hpp"
#include <algorithm>
#include <chrono>
#include "duckdb/execution/distributed/utils/optional.hpp"

namespace duckdb {
namespace distributed {

namespace {

size_t ResolveScanTaskSubmissionBacklog(size_t scan_task_count, const DuckDBExecutionConfigRef &exec_cfg) {
	if (scan_task_count == 0) {
		return 1;
	}
	if (!exec_cfg) {
		return scan_task_count;
	}

	size_t backlog = exec_cfg->scan_task_backlog();
	if (backlog == 0) {
		backlog = exec_cfg->distributed_worker_slots();
	}
	if (backlog == 0) {
		backlog = exec_cfg->distributed_node_count();
	}
	if (backlog == 0) {
		backlog = scan_task_count;
	}
	return std::max<size_t>(1, std::min(backlog, scan_task_count));
}

} // namespace

SubmittableTaskStream<WorkerTask> ScanSourceNode::produce_tasks(PlanExecutionContext &plan_context) {
	const auto channel_capacity = ResolveScanTaskSubmissionBacklog(scan_tasks_.size(), config().execution_config());

	auto channel = create_channel<SubmittableTask<WorkerTask>>(channel_capacity);
	auto rx = std::move(channel.second);
	auto tx_ptr = std::make_shared<Sender<SubmittableTask<WorkerTask>>>(std::move(channel.first));

	auto self = shared_from_this();
	Optional<TaskInput> injected_input;
	if (auto *input = plan_context.lookup_initial_input(static_cast<SourceNodeId>(self->node_id()))) {
		injected_input = *input;
	}

	auto injected_input_ptr = std::make_shared<Optional<TaskInput>>(std::move(injected_input));
	auto &task_id_counter = plan_context.task_id_counter();
	plan_context.spawn([self, tx_ptr, &task_id_counter, injected_input_ptr]() mutable -> DuckDBResult<void> {
		auto t_spawn_start = std::chrono::steady_clock::now();
		auto t_wall_start = std::chrono::system_clock::now();
		double t_wall_epoch = std::chrono::duration<double>(t_wall_start.time_since_epoch()).count();

		if (!self->scan_plan_ || !self->scan_plan_->HasRoot()) {
			tx_ptr->state()->close();
			return DuckDBResult<void>::err(DuckDBError("ScanSourceNode received invalid scan plan"));
		}

		if (injected_input_ptr->has_value()) {
			if ((*injected_input_ptr)->kind != TaskInput::Kind::ScanTask) {
				tx_ptr->state()->close();
				return DuckDBResult<void>::err(DuckDBError("ScanSourceNode injected input must be a scan task"));
			}

			TaskContext tctx =
			    TaskContext::from_node_context(self->context().query_idx(), self->node_id(), task_id_counter.next());
			WorkerTask task(tctx, self->scan_plan_, self->config().execution_config(), self->context().to_hashmap());
			task.mutable_inputs()[static_cast<SourceNodeId>(self->node_id())] =
			    TaskInput::make_scan_task((*injected_input_ptr)->scan_task_bytes);
			auto r = tx_ptr->send(SubmittableTask<WorkerTask>(std::move(task)));
			if (r.is_err()) {
				tx_ptr->state()->close();
				return DuckDBResult<void>::ok();
			}
			tx_ptr->state()->close();
			return DuckDBResult<void>::ok();
		}

		if (self->scan_tasks_.empty()) {
			if (self->require_scan_tasks_) {
				tx_ptr->state()->close();
				return DuckDBResult<void>::err(DuckDBError("ScanSourceNode missing scan task partition set"));
			}

			TaskContext tctx =
			    TaskContext::from_node_context(self->context().query_idx(), self->node_id(), task_id_counter.next());
			WorkerTask task(tctx, self->scan_plan_, self->config().execution_config(), self->context().to_hashmap());
			auto r = tx_ptr->send(SubmittableTask<WorkerTask>(std::move(task)));
			if (r.is_err()) {
				tx_ptr->state()->close();
				return DuckDBResult<void>::ok();
			}
			tx_ptr->state()->close();
			return DuckDBResult<void>::ok();
		}

		const size_t num_scan_tasks = self->scan_tasks_.size();

		// Emit one task per descriptor produced by MakeTableScanTasks.
		for (size_t task_idx = 0; task_idx < num_scan_tasks; task_idx++) {
			const auto &descriptor = self->scan_tasks_[task_idx];
			auto t_serialized_start = std::chrono::steady_clock::now();
			auto scan_bytes = descriptor.SerializeToBytes();
			auto t_serialized = std::chrono::steady_clock::now();

			TaskContext tctx =
			    TaskContext::from_node_context(self->context().query_idx(), self->node_id(), task_id_counter.next());
			auto context = self->context().to_hashmap();

			WorkerTask task(tctx, self->scan_plan_, self->config().execution_config(), std::move(context));
			// Populate inputs_ for SourceId-based routing (analogous to Vane's Input::ScanTask)
			task.mutable_inputs()[static_cast<SourceNodeId>(self->node_id())] = TaskInput::make_scan_task(scan_bytes);
			auto t_pre_send = std::chrono::steady_clock::now();
			auto r = tx_ptr->send(SubmittableTask<WorkerTask>(std::move(task)));
			auto t_post_send = std::chrono::steady_clock::now();

			auto serialize_ms =
			    std::chrono::duration_cast<std::chrono::milliseconds>(t_serialized - t_serialized_start).count();
			auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_pre_send - t_serialized).count();
			auto send_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_post_send - t_pre_send).count();
			auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_post_send - t_spawn_start).count();

			if (r.is_err()) {
				tx_ptr->state()->close();
				return DuckDBResult<void>::ok();
			}
		}

		tx_ptr->state()->close();
		return DuckDBResult<void>::ok();
	});

	return SubmittableTaskStream<WorkerTask>::from_receiver(std::move(rx));
}

} // namespace distributed
} // namespace duckdb
