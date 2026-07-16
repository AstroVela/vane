// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"

#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/execution/distributed/utils/stream.hpp"

#include <chrono>
#include <thread>
#include <unordered_set>

namespace duckdb {
namespace distributed {

template <typename TaskT>
SubmittableTaskStream<TaskT> SubmittableTaskStream<TaskT>::from_receiver(Receiver<SubmittableTask<TaskT>> receiver) {
	auto task_stream =
	    boxed<SubmittableTask<TaskT>>(distributed::ChannelStream<SubmittableTask<TaskT>>(std::move(receiver)));
	return SubmittableTaskStream<TaskT> {std::move(task_stream)};
}

template <typename TaskT>
MaterializeResult SubmittableTaskStream<TaskT>::materialize(FteTaskSubmitter *fte_task_submitter,
                                                            MaterializedOutputCallback on_output) {
	auto materialize_profile_start = std::chrono::steady_clock::now();
	MaterializeResult result;
	result.success = true;
	size_t materialize_poll_count = 0;

	if (!fte_task_submitter) {
		result.success = false;
		result.error = "FTE materialize requires an FteTaskSubmitter";
		return result;
	}

	std::unordered_set<SourceNodeId> fte_source_node_ids;
	std::unordered_set<TaskContext, TaskContextHash> materialize_task_contexts;
	std::string query_id;
	size_t fte_event_count = 0;

	auto submit_fte_events = [&](std::vector<WorkerTask> fte_events) -> DuckDBResult<void> {
		for (const auto &task : fte_events) {
			materialize_task_contexts.insert(task.task_context());
			const auto &context = task.context();
			auto query_it = context.find("query_id");
			if (query_it != context.end() && !query_it->second.empty()) {
				if (query_id.empty()) {
					query_id = query_it->second;
				} else if (query_id != query_it->second) {
					return DuckDBResult<void>::err(DuckDBError::invalid_state_error(
					    "FTE materialize received task events from multiple query_id values"));
				}
			}
			for (const auto &entry : task.inputs()) {
				fte_source_node_ids.insert(entry.first);
			}
		}
		fte_event_count += fte_events.size();
		return fte_task_submitter->submit_fte_task_events(std::move(fte_events));
	};

	try {
		while (true) {
			auto poll_profile_start = std::chrono::steady_clock::now();
			auto t = this->poll_next();
			auto poll_profile_end = std::chrono::steady_clock::now();
			auto poll_profile_ms =
			    std::chrono::duration_cast<std::chrono::milliseconds>(poll_profile_end - poll_profile_start).count();
			materialize_poll_count++;
			if (!t.first) {
				break;
			}

			std::vector<WorkerTask> fte_events;
			const size_t burst_limit = FteEventBurstLimit();
			const auto coalesce_delay = FteEventCoalesceDelay();
			fte_events.reserve(burst_limit);
			fte_events.push_back(std::move(t.second).take_task());

			while (fte_events.size() < burst_limit) {
				auto next = this->try_poll_next();
				if (next.first) {
					fte_events.push_back(std::move(next.second).take_task());
					continue;
				}
				if (coalesce_delay.count() <= 0) {
					break;
				}
				const auto deadline = std::chrono::steady_clock::now() + coalesce_delay;
				while (fte_events.size() < burst_limit) {
					const auto now = std::chrono::steady_clock::now();
					if (now >= deadline) {
						break;
					}
					auto sleep_for = std::chrono::duration_cast<std::chrono::microseconds>(deadline - now);
					const auto max_sleep = std::chrono::microseconds(100);
					if (sleep_for > max_sleep) {
						sleep_for = max_sleep;
					}
					std::this_thread::sleep_for(sleep_for);
					auto delayed_next = this->try_poll_next();
					if (!delayed_next.first) {
						continue;
					}
					fte_events.push_back(std::move(delayed_next.second).take_task());
					break;
				}
				if (std::chrono::steady_clock::now() >= deadline) {
					break;
				}
			}

			auto submit_res = submit_fte_events(std::move(fte_events));
			if (submit_res.is_err()) {
				result.success = false;
				result.error = submit_res.error().what();
				return result;
			}
		}

		if (!fte_source_node_ids.empty()) {
			auto exhausted_res = fte_task_submitter->task_input_stream_exhausted(query_id, fte_source_node_ids);
			if (exhausted_res.is_err()) {
				result.success = false;
				result.error = exhausted_res.error().what();
				return result;
			}
		}
		if (!query_id.empty()) {
			auto wait_res = fte_task_submitter->wait_query_finished(query_id, FteQueryWaitTimeoutSeconds(),
			                                                        materialize_task_contexts, on_output);
			if (wait_res.is_err()) {
				result.success = false;
				result.error = wait_res.error().what();
				return result;
			}
			result.outputs = std::move(wait_res).value();
		} else if (fte_event_count > 0) {
			result.success = false;
			result.error = "FTE materialize cannot wait for query completion without query_id";
			return result;
		}
	} catch (const std::exception &ex) {
		result.success = false;
		result.error = ex.what();
	} catch (...) {
		result.success = false;
		result.error = "materialize_fte unknown exception";
	}

	return result;
}

template class SubmittableTaskStream<WorkerTask>;

} // namespace distributed
} // namespace duckdb
