// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file runner.hpp
 * @brief Simplified Plan runner for executing distributed physical plans.
 *
 * Implements the distributed task runner used by Ray-backed execution.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cerrno>
#include <cctype>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

#include <typeinfo>
#include <unordered_set>

#include "duckdb/execution/distributed/plan/distributed_physical_plan.hpp"

#include "duckdb/execution/distributed/common_types.hpp"

#include "duckdb/execution/distributed/utils/channel.hpp"
#include "duckdb/execution/distributed/utils/distributed_task.hpp"
#include "duckdb/execution/distributed/scheduling/worker.hpp"
#include "duckdb/execution/distributed/plan/plan_config.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/distributed/pipeline_node/translator_api.hpp"
#include "duckdb/execution/distributed/pipeline_node/sink.hpp"
#include "duckdb/execution/distributed/pipeline_node/copy_finish.hpp"
#include "duckdb/execution/distributed/copy_finalize.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/hive_partitioning.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {
namespace distributed {

inline std::pair<bool, idx_t> DirectWriteLifecycleCleanupMinAgeMsFromEnv() {
	const char *env = std::getenv("VANE_DISTRIBUTED_COPY_DIRECT_WRITE_CLEANUP_MIN_AGE_MS");
	if (!env || !*env) {
		return std::make_pair(true, static_cast<idx_t>(24ULL * 60ULL * 60ULL * 1000ULL));
	}

	auto lower = StringUtil::Lower(std::string(env));
	if (lower == "0" || lower == "false" || lower == "no" || lower == "off" || lower == "disabled") {
		return std::make_pair(false, idx_t(0));
	}

	errno = 0;
	char *end = nullptr;
	auto value = std::strtoull(env, &end, 10);
	if (errno != 0 || end == env || *end != '\0') {
		return std::make_pair(true, static_cast<idx_t>(24ULL * 60ULL * 60ULL * 1000ULL));
	}
	return std::make_pair(true, static_cast<idx_t>(value));
}

inline size_t FteEventBurstLimit() {
	const char *env = std::getenv("VANE_FTE_EVENT_BURST_LIMIT");
	if (!env || !*env) {
		return 64;
	}
	errno = 0;
	char *end = nullptr;
	auto value = std::strtoull(env, &end, 10);
	if (errno != 0 || end == env || *end != '\0' || value == 0) {
		return 64;
	}
	return static_cast<size_t>(value);
}

inline std::chrono::microseconds FteEventCoalesceDelay() {
	const char *env = std::getenv("VANE_FTE_EVENT_COALESCE_US");
	if (!env || !*env) {
		return std::chrono::microseconds(1000);
	}
	errno = 0;
	char *end = nullptr;
	auto value = std::strtoull(env, &end, 10);
	if (errno != 0 || end == env || *end != '\0') {
		return std::chrono::microseconds(1000);
	}
	return std::chrono::microseconds(value);
}

inline double FteQueryWaitTimeoutSeconds() {
	const char *env = std::getenv("VANE_FTE_QUERY_WAIT_TIMEOUT_S");
	if (!env || !*env) {
		return 0.0;
	}
	errno = 0;
	char *end = nullptr;
	auto value = std::strtod(env, &end);
	if (errno != 0 || end == env || *end != '\0' || value < 0.0) {
		return 0.0;
	}
	return value;
}

inline bool FteRunnerDebugEnabled() {
	for (const char *name : {"VANE_FTE_ADMISSION_DEBUG", "DUCKDB_DISTRIBUTED_DEBUG"}) {
		const char *value = std::getenv(name);
		if (!value || !*value) {
			continue;
		}
		auto lower = StringUtil::Lower(std::string(value));
		if (lower != "0" && lower != "false" && lower != "no" && lower != "off") {
			return true;
		}
	}
	return false;
}

inline std::string FteRunnerFormatField(std::string value) {
	if (value.empty()) {
		return "-";
	}
	for (auto &ch : value) {
		if (std::isspace(static_cast<unsigned char>(ch))) {
			ch = '_';
		}
	}
	return value;
}

inline std::string FteRunnerContextField(const std::unordered_map<std::string, std::string> &context,
                                         const std::string &key) {
	auto it = context.find(key);
	if (it == context.end()) {
		return "-";
	}
	return FteRunnerFormatField(it->second);
}

inline std::string FteRunnerTaskSummary(const WorkerTask &task) {
	const auto &context = task.context();
	const auto task_context = task.task_context();
	std::ostringstream out;
	out << "task_name=" << FteRunnerFormatField(task.name())
	    << " query_id=" << FteRunnerContextField(context, "query_id")
	    << " node_id=" << FteRunnerContextField(context, "node_id")
	    << " fragment_execution_id=" << FteRunnerContextField(context, "fragment_execution_id")
	    << " context_task_id=" << FteRunnerContextField(context, "task_id")
	    << " task_context_task_id=" << task_context.task_id()
	    << " task_context_last_node_id=" << task_context.last_node_id() << " input_count=" << task.inputs().size();
	return out.str();
}

inline int64_t FteRunnerElapsedMs(std::chrono::steady_clock::time_point started_at) {
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started_at).count();
}

inline void FteRunnerDebugLog(const std::string &message) {
	if (!FteRunnerDebugEnabled()) {
		return;
	}
	std::cerr << "[vane-fte-runner tid=" << std::this_thread::get_id() << "] " << message << std::endl;
}

// Result of materializing all pipeline outputs
struct MaterializeResult {
	bool success = true;
	std::string error;
	std::vector<MaterializedOutput> outputs;
};

//==============================================================================
// TaskIDCounter
//==============================================================================
class TaskIDCounter {
private:
	std::shared_ptr<std::atomic<uint32_t>> counter_;

public:
	TaskIDCounter() : counter_(std::make_shared<std::atomic<uint32_t>>(0)) {
	}

	TaskIDCounter(const TaskIDCounter &other) = default;

	uint32_t next() {
		return counter_->fetch_add(1, std::memory_order_relaxed);
	}
};

//==============================================================================
// PlanExecutionContext
//==============================================================================

class PlanExecutionContext {
private:
	duckdb::shared_ptr<::duckdb::ClientContext> client_context_;
	std::shared_ptr<TaskExecutor> task_executor_;
	std::shared_ptr<FteTaskSubmitter> fte_task_submitter_;
	TaskIDCounter task_id_counter_;
	TaskInputs initial_inputs_;

public:
	explicit PlanExecutionContext(std::shared_ptr<TaskExecutor> task_executor,
	                              duckdb::shared_ptr<::duckdb::ClientContext> client_context = nullptr,
	                              TaskInputs initial_inputs = {},
	                              std::shared_ptr<FteTaskSubmitter> fte_task_submitter = nullptr)
	    : client_context_(std::move(client_context)), task_executor_(std::move(task_executor)),
	      fte_task_submitter_(std::move(fte_task_submitter)), task_id_counter_(),
	      initial_inputs_(std::move(initial_inputs)) {
	}

	PlanExecutionContext(PlanExecutionContext &&) noexcept = default;
	PlanExecutionContext &operator=(PlanExecutionContext &&) noexcept = default;
	PlanExecutionContext(const PlanExecutionContext &) = delete;
	PlanExecutionContext &operator=(const PlanExecutionContext &) = delete;

	std::shared_ptr<FteTaskSubmitter> fte_task_submitter_ref() const {
		return fte_task_submitter_;
	}

	template <typename F>
	void spawn(F &&task) {
		typedef typename std::decay<F>::type TaskFunc;
		auto task_ptr = std::make_shared<TaskFunc>(std::forward<F>(task));
		task_executor_->ScheduleTask(make_uniq<DistributedPlanTask>(*task_executor_, [task_ptr]() mutable {
			auto result = (*task_ptr)();
			if (result.is_err()) {
				auto msg = std::string("[PlanExecutionContext::spawn] task error: ") + result.error().what();
				throw InternalException(msg);
			}
		}));
	}

	// Return a new TaskIDCounter reference that allows generating new task ids
	TaskIDCounter &task_id_counter() {
		return task_id_counter_;
	}

	::duckdb::ClientContext *client_context() const {
		return client_context_.get();
	}

	const TaskInput *lookup_initial_input(SourceNodeId node_id) const {
		auto entry = initial_inputs_.find(node_id);
		if (entry == initial_inputs_.end()) {
			return nullptr;
		}
		return &entry->second;
	}
};

class WorkerManagerFteTaskSubmitter final : public FteTaskSubmitter {
public:
	explicit WorkerManagerFteTaskSubmitter(std::shared_ptr<WorkerManager> worker_manager)
	    : worker_manager_(std::move(worker_manager)) {
	}

	DuckDBResult<void> submit_fte_task_events(std::vector<WorkerTask> tasks) override {
		return worker_manager_->submit_fte_task_events(std::move(tasks));
	}

	DuckDBResult<void> task_input_stream_exhausted(const std::string &query_id,
	                                               const std::unordered_set<SourceNodeId> &source_node_ids) override {
		auto submit_res = worker_manager_->task_input_stream_exhausted_for_query(query_id, source_node_ids);
		if (submit_res.is_err()) {
			return DuckDBResult<void>::err(submit_res.error());
		}
		return DuckDBResult<void>::ok();
	}

	DuckDBResult<std::vector<MaterializedOutput>> wait_query_finished(const std::string &query_id,
	                                                                  double timeout_s) override {
		return worker_manager_->wait_fte_query(query_id, timeout_s);
	}

	DuckDBResult<std::vector<MaterializedOutput>> wait_query_finished(const std::string &query_id, double timeout_s,
	                                                                  MaterializedOutputCallback on_output) override {
		return worker_manager_->wait_fte_query(query_id, timeout_s, std::move(on_output));
	}

	DuckDBResult<std::vector<MaterializedOutput>>
	wait_query_finished(const std::string &query_id, double timeout_s,
	                    const std::unordered_set<TaskContext, TaskContextHash> &task_contexts,
	                    MaterializedOutputCallback on_output) override {
		return worker_manager_->wait_fte_query(query_id, timeout_s, task_contexts, std::move(on_output));
	}

private:
	std::shared_ptr<WorkerManager> worker_manager_;
};

//==============================================================================
// PlanRunner (simplified)
//==============================================================================
class PlanRunner : public std::enable_shared_from_this<PlanRunner> {
public:
	explicit PlanRunner(std::shared_ptr<WorkerManager> worker_manager,
	                    duckdb::shared_ptr<::duckdb::ClientContext> client_context = nullptr)
	    : worker_manager_(std::move(worker_manager)), client_context_(std::move(client_context)) {
	}

	// Execute a pipeline node by forwarding its task stream to the FTE
	// coordinator. This runs in a background thread spawned by run_plan.
	DuckDBResult<void> execute_plan(std::shared_ptr<DistributedPipelineNode> pipeline_node,
	                                std::shared_ptr<TaskExecutor> task_executor,
	                                UnboundedSender<MaterializedOutput> output_sender, TaskInputs initial_inputs = {}) {
		auto fte_task_submitter = std::make_shared<WorkerManagerFteTaskSubmitter>(worker_manager_);
		PlanExecutionContext ctx(task_executor, client_context_, std::move(initial_inputs), fte_task_submitter);
		auto tasks_stream = pipeline_node->produce_tasks(ctx);
		std::unordered_set<SourceNodeId> fte_source_node_ids;
		std::string query_id;
		size_t fte_event_count = 0;
		size_t submit_batch_index = 0;
		const auto execute_started_at = std::chrono::steady_clock::now();

		auto submit_fte_events = [&](std::vector<WorkerTask> fte_events) -> DuckDBResult<void> {
			const auto batch_index = submit_batch_index++;
			const auto batch_size = fte_events.size();
			std::ostringstream submit_start_msg;
			submit_start_msg << "event=submit_batch_start"
			                 << " elapsed_ms=" << FteRunnerElapsedMs(execute_started_at)
			                 << " batch_index=" << batch_index << " batch_size=" << batch_size
			                 << " total_events_before=" << fte_event_count;
			if (!fte_events.empty()) {
				submit_start_msg << " " << FteRunnerTaskSummary(fte_events.front());
			}
			FteRunnerDebugLog(submit_start_msg.str());
			for (const auto &task : fte_events) {
				const auto &context = task.context();
				auto query_it = context.find("query_id");
				if (query_it != context.end() && !query_it->second.empty()) {
					if (query_id.empty()) {
						query_id = query_it->second;
					} else if (query_id != query_it->second) {
						return DuckDBResult<void>::err(DuckDBError::invalid_state_error(
						    "FTE runner received task events from multiple query_id values"));
					}
				}
				for (const auto &entry : task.inputs()) {
					fte_source_node_ids.insert(entry.first);
				}
			}
			fte_event_count += fte_events.size();
			const auto submit_started_at = std::chrono::steady_clock::now();
			auto submit_res = fte_task_submitter->submit_fte_task_events(std::move(fte_events));
			std::ostringstream submit_done_msg;
			submit_done_msg << "event=submit_batch_done"
			                << " elapsed_ms=" << FteRunnerElapsedMs(execute_started_at)
			                << " batch_index=" << batch_index << " batch_size=" << batch_size
			                << " total_events_after=" << fte_event_count
			                << " submit_elapsed_ms=" << FteRunnerElapsedMs(submit_started_at)
			                << " result=" << (submit_res.is_err() ? "err" : "ok");
			if (submit_res.is_err()) {
				submit_done_msg << " error=" << FteRunnerFormatField(submit_res.error().what());
			}
			FteRunnerDebugLog(submit_done_msg.str());
			return submit_res;
		};

		try {
			while (true) {
				auto t = tasks_stream.poll_next();
				if (!t.first) {
					break;
				}

				std::vector<WorkerTask> fte_events;
				const size_t burst_limit = FteEventBurstLimit();
				const auto coalesce_delay = FteEventCoalesceDelay();
				fte_events.reserve(burst_limit);
				auto first_task = std::move(t.second).take_task();
				FteRunnerDebugLog(
				    "event=poll_first_task elapsed_ms=" + std::to_string(FteRunnerElapsedMs(execute_started_at)) + " " +
				    FteRunnerTaskSummary(first_task));
				fte_events.push_back(std::move(first_task));

				while (fte_events.size() < burst_limit) {
					auto next = tasks_stream.try_poll_next();
					if (next.first) {
						auto next_task = std::move(next.second).take_task();
						FteRunnerDebugLog("event=coalesce_append elapsed_ms=" +
						                  std::to_string(FteRunnerElapsedMs(execute_started_at)) +
						                  " mode=try_poll batch_size_before=" + std::to_string(fte_events.size()) +
						                  " " + FteRunnerTaskSummary(next_task));
						fte_events.push_back(std::move(next_task));
						continue;
					}
					if (coalesce_delay.count() <= 0) {
						FteRunnerDebugLog("event=coalesce_disabled elapsed_ms=" +
						                  std::to_string(FteRunnerElapsedMs(execute_started_at)) +
						                  " batch_size=" + std::to_string(fte_events.size()));
						break;
					}
					const auto deadline = std::chrono::steady_clock::now() + coalesce_delay;
					bool appended_during_wait = false;
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
						auto delayed_next = tasks_stream.try_poll_next();
						if (!delayed_next.first) {
							continue;
						}
						auto delayed_task = std::move(delayed_next.second).take_task();
						FteRunnerDebugLog("event=coalesce_append elapsed_ms=" +
						                  std::to_string(FteRunnerElapsedMs(execute_started_at)) +
						                  " mode=delayed_try_poll batch_size_before=" +
						                  std::to_string(fte_events.size()) + " " + FteRunnerTaskSummary(delayed_task));
						fte_events.push_back(std::move(delayed_task));
						appended_during_wait = true;
						break;
					}
					if (std::chrono::steady_clock::now() >= deadline) {
						FteRunnerDebugLog("event=coalesce_timeout elapsed_ms=" +
						                  std::to_string(FteRunnerElapsedMs(execute_started_at)) +
						                  " batch_size=" + std::to_string(fte_events.size()) +
						                  " coalesce_us=" + std::to_string(coalesce_delay.count()));
						break;
					}
					if (!appended_during_wait) {
						break;
					}
				}

				auto submit_res = submit_fte_events(std::move(fte_events));
				if (submit_res.is_err()) {
					return DuckDBResult<void>::err(submit_res.error());
				}
			}

			FteRunnerDebugLog(
			    "event=task_stream_exhausted elapsed_ms=" + std::to_string(FteRunnerElapsedMs(execute_started_at)) +
			    " total_events=" + std::to_string(fte_event_count) + " source_count=" +
			    std::to_string(fte_source_node_ids.size()) + " query_id=" + FteRunnerFormatField(query_id));
			if (!fte_source_node_ids.empty()) {
				auto exhausted_res = fte_task_submitter->task_input_stream_exhausted(query_id, fte_source_node_ids);
				if (exhausted_res.is_err()) {
					return DuckDBResult<void>::err(exhausted_res.error());
				}
			}
			if (!query_id.empty()) {
				FteRunnerDebugLog(
				    "event=wait_query_start elapsed_ms=" + std::to_string(FteRunnerElapsedMs(execute_started_at)) +
				    " query_id=" + FteRunnerFormatField(query_id) + " total_events=" + std::to_string(fte_event_count));
				auto wait_res = fte_task_submitter->wait_query_finished(query_id, FteQueryWaitTimeoutSeconds());
				if (wait_res.is_err()) {
					FteRunnerDebugLog(
					    "event=wait_query_done elapsed_ms=" + std::to_string(FteRunnerElapsedMs(execute_started_at)) +
					    " query_id=" + FteRunnerFormatField(query_id) +
					    " result=err error=" + FteRunnerFormatField(wait_res.error().what()));
					return DuckDBResult<void>::err(wait_res.error());
				}
				auto outputs = std::move(wait_res).value();
				FteRunnerDebugLog(
				    "event=wait_query_done elapsed_ms=" + std::to_string(FteRunnerElapsedMs(execute_started_at)) +
				    " query_id=" + FteRunnerFormatField(query_id) +
				    " result=ok output_count=" + std::to_string(outputs.size()));
				for (auto &output : outputs) {
					auto send_res = output_sender.send(std::move(output));
					if (send_res.is_err()) {
						return DuckDBResult<void>::err(send_res.error());
					}
				}
			} else if (fte_event_count > 0) {
				return DuckDBResult<void>::err(
				    DuckDBError::invalid_state_error("FTE runner cannot wait for query completion without query_id"));
			}
		} catch (const std::exception &ex) {
			return DuckDBResult<void>::err(DuckDBError::external_error(ex.what()));
		} catch (...) {
			return DuckDBResult<void>::err(DuckDBError::external_error("execute_plan unknown exception"));
		}

		return DuckDBResult<void>::ok();
	}

	/// Unified result type: streaming (SELECT) or finalized (COPY).
	struct PlanResult {
		enum Tag { STREAMING, COPY };
		Tag tag;
		// Only one of these is valid depending on tag
		PlanResultStream stream;
		DistributedCopyResult copy_result;

		// Streaming constructor
		static PlanResult make_streaming(std::shared_ptr<TaskExecutor> te, UnboundedReceiver<MaterializedOutput> recv,
		                                 std::shared_ptr<PlanExecutionStatus> status) {
			PlanResult r;
			r.tag = STREAMING;
			r.stream = PlanResultStream(std::move(te), std::move(recv), std::move(status));
			return r;
		}
		// Copy constructor
		static PlanResult make_copy(DistributedCopyResult cr) {
			PlanResult r;
			r.tag = COPY;
			r.copy_result = std::move(cr);
			return r;
		}
	};

	/// Unified run_plan: auto-detects sink nodes and handles both streaming and finalize paths.
	/// - Non-sink plans → returns PlanResultStream (streaming pull)
	/// - Sink plans (CopyFinish) → collects all outputs, calls finalize(), returns DistributedCopyResult
	DuckDBResult<PlanResult> run_plan(std::shared_ptr<DistributedPhysicalPlan> plan, TaskInputs initial_inputs = {}) {
		if (!client_context_) {
			return DuckDBResult<PlanResult>::err(DuckDBError("run_plan requires a ClientContext"));
		}

		// ── Step 1: Translate physical plan → pipeline node ──
		auto physical_plan = plan->physical_plan();
		auto exec_cfg = plan->execution_config();
		if (exec_cfg && worker_manager_ &&
		    (exec_cfg->distributed_node_count() == 0 || exec_cfg->distributed_worker_slots() == 0)) {
			auto snapshots_res = worker_manager_->worker_snapshots();
			if (snapshots_res.is_ok()) {
				const auto &snapshots = snapshots_res.value();
				const auto worker_count = snapshots.size();
				if (exec_cfg->distributed_node_count() == 0 && worker_count > 0) {
					exec_cfg->set_distributed_node_count(worker_count);
				}
				if (exec_cfg->distributed_worker_slots() == 0 && worker_count > 0) {
					const int min_cpu_per_task = exec_cfg->min_cpu_per_task() > 0 ? exec_cfg->min_cpu_per_task() : 1;
					size_t total_worker_slots = 0;
					for (const auto &snapshot : snapshots) {
						const double total_num_cpus = snapshot.total_num_cpus();
						if (total_num_cpus <= 0) {
							continue;
						}
						size_t worker_slots =
						    static_cast<size_t>(total_num_cpus / static_cast<double>(min_cpu_per_task));
						if (worker_slots == 0) {
							worker_slots = 1;
						}
						total_worker_slots += worker_slots;
					}
					if (total_worker_slots == 0) {
						total_worker_slots = worker_count;
					}
					exec_cfg->set_distributed_worker_slots(total_worker_slots);
				}
			}
		}
		PlanConfig cfg(plan->idx(), plan->query_id(), exec_cfg);
		if (client_context_ && client_context_->db) {
			cfg.db = client_context_->db;
		}

		DuckDBResult<std::shared_ptr<DistributedPipelineNode>> pipeline_res;
		try {
			pipeline_res = physical_plan_to_pipeline_node_wrapper(cfg, physical_plan, client_context_.get());
		} catch (const std::exception &ex) {
			return DuckDBResult<PlanResult>::err(DuckDBError(std::string("Failed to translate plan: ") + ex.what()));
		}
		if (pipeline_res.is_err()) {
			return DuckDBResult<PlanResult>::err(pipeline_res.error());
		}
		if (!pipeline_res.value()) {
			return DuckDBResult<PlanResult>::err(DuckDBError("Pipeline translation returned null"));
		}
		auto pipeline_node = pipeline_res.value();

		// ── Step 2: Find sink node (if any) ──
		std::shared_ptr<CopyFinishNode> sink_node;
		std::function<void(const DistributedPipelineNodeRef &)> find_sink = [&](const DistributedPipelineNodeRef &n) {
			if (!n || sink_node)
				return;
			auto impl = n->inner();
			if (impl && impl->is_sink()) {
				if (auto copy_finish = std::dynamic_pointer_cast<CopyFinishNode>(impl)) {
					sink_node = copy_finish;
					return;
				}
			}
			for (auto &c : n->arc_children()) {
				find_sink(c);
				if (sink_node)
					return;
			}
		};
		find_sink(pipeline_node);

		if (sink_node && sink_node->staging_root_base().empty()) {
			auto &fs = FileSystem::GetFileSystem(*client_context_);
			auto cleanup_policy = DirectWriteLifecycleCleanupMinAgeMsFromEnv();
			if (cleanup_policy.first) {
				(void)CleanupExpiredDistributedCopyDirectWriteRuns(fs, sink_node->spec().file_path,
				                                                   cleanup_policy.second);
			}

			auto lifecycle_res =
			    WriteDistributedCopyDirectWriteLifecycle(fs, sink_node->spec().file_path, sink_node->staging_run_id());
			if (lifecycle_res.is_err()) {
				return DuckDBResult<PlanResult>::err(lifecycle_res.error());
			}
		}

		// ── Step 3: Common setup — result channel + FTE execution ──
		auto channel_pair = create_unbounded_channel<MaterializedOutput>();
		auto sender = std::move(channel_pair.first);
		auto receiver = std::move(channel_pair.second);
		auto execute_status = std::make_shared<PlanExecutionStatus>();
		auto output_state = sender.state();
		if (worker_manager_) {
			worker_manager_->set_streaming_results_channel_state(sender.state());
		}
		auto task_executor = std::make_shared<TaskExecutor>(*client_context_);

		auto self = std::shared_ptr<PlanRunner>(this->shared_from_this());
		if (!self) {
			return DuckDBResult<PlanResult>::err(
			    DuckDBError("PlanRunner requires shared_ptr ownership; create via std::make_shared"));
		}
		auto sender_ptr = std::make_shared<UnboundedSender<MaterializedOutput>>(std::move(sender));
		auto initial_inputs_ptr = std::make_shared<TaskInputs>(std::move(initial_inputs));
		task_executor->ScheduleTask(make_uniq<distributed::DistributedPlanTask>(
		    *task_executor, [self, pipeline_node, sender_ptr, output_state, execute_status, task_executor,
		                     initial_inputs_ptr]() mutable {
			    std::unique_ptr<UnboundedSender<MaterializedOutput>> output_lifetime_guard;
			    auto publish_error = [&](const DuckDBError &error) {
				    execute_status->RecordError(error);
				    if (output_state) {
					    output_state->close();
				    }
			    };
			    auto clear_worker_channel = [&]() {
				    if (self->worker_manager_) {
					    self->worker_manager_->clear_streaming_results_channel_state();
				    }
			    };
			    try {
				    output_lifetime_guard = make_uniq<UnboundedSender<MaterializedOutput>>(sender_ptr->clone());
				    auto result = self->execute_plan(pipeline_node, task_executor, std::move(*sender_ptr),
				                                     std::move(*initial_inputs_ptr));
				    clear_worker_channel();
				    if (result.is_err()) {
					    publish_error(result.error());
				    }
				    output_lifetime_guard.reset();
			    } catch (const std::exception &ex) {
				    clear_worker_channel();
				    DuckDBError error =
				        DuckDBError::external_error(std::string("execute_plan task threw: ") + ex.what());
				    publish_error(error);
				    output_lifetime_guard.reset();
			    } catch (...) {
				    clear_worker_channel();
				    DuckDBError error = DuckDBError::external_error("execute_plan task threw unknown exception");
				    publish_error(error);
				    output_lifetime_guard.reset();
			    }
		    }));

		// ── Step 4: Dispatch based on sink presence ──
		if (!sink_node) {
			// Streaming path: return pull-based stream
			return DuckDBResult<PlanResult>::ok(
			    PlanResult::make_streaming(std::move(task_executor), std::move(receiver), std::move(execute_status)));
		}

		// Sink path: collect all outputs, then finalize
		auto sink_node_id = sink_node->copy_sink()->node_id();
		auto cleanup_sink_output = [&]() {
			auto &fs = FileSystem::GetFileSystem(*client_context_);
			if (sink_node->staging_root_base().empty()) {
				CleanupDistributedCopyUncommittedDirectWriteRun(fs, sink_node->spec().file_path,
				                                                sink_node->staging_run_id());
				return;
			}
			auto staging_root = fs.JoinPath(sink_node->staging_root_base(), sink_node->staging_run_id());
			RemoveDistributedCopyDirectoryTree(fs, staging_root);
			RemoveDistributedCopyDirectoryIfEmpty(fs, sink_node->staging_root_base());
		};
		std::vector<ResultPartitionRef> partitions;
		auto staging_write_started = std::chrono::steady_clock::now();
		while (true) {
			auto item = receiver.recv();
			if (auto execute_error = execute_status->GetError()) {
				cleanup_sink_output();
				return DuckDBResult<PlanResult>::err(*execute_error);
			}
			if (!item.first) {
				break;
			}
			if (!item.second.has_node_id(sink_node_id))
				continue;
			for (auto &part : item.second.fragments()) {
				partitions.push_back(part);
			}
		}

		if (auto execute_error = execute_status->GetError()) {
			cleanup_sink_output();
			return DuckDBResult<PlanResult>::err(*execute_error);
		}

		auto staging_write_ms = DistributedCopyElapsedMillis(staging_write_started);
		auto finalize_res = sink_node->finalize(partitions, *client_context_);
		if (finalize_res.is_err()) {
			return DuckDBResult<PlanResult>::err(finalize_res.error());
		}
		auto copy_result = std::move(finalize_res).value();
		copy_result.staging_write_ms = staging_write_ms;
		return DuckDBResult<PlanResult>::ok(PlanResult::make_copy(std::move(copy_result)));
	}

	/// Legacy finalize_copy — kept for Python callers that use the streaming + manual finalize path.
	DuckDBResult<DistributedCopyResult> finalize_copy(const DistributedCopySpec &spec, const std::string &staging_root,
	                                                  std::vector<DistributedCopyFileInfo> files) {
		if (!client_context_) {
			return DuckDBResult<DistributedCopyResult>::err(DuckDBError("finalize_copy requires a ClientContext"));
		}
		return FinalizeCopyFiles(spec, staging_root, std::move(files), *client_context_);
	}

private:
	std::shared_ptr<WorkerManager> worker_manager_;
	duckdb::shared_ptr<::duckdb::ClientContext> client_context_;
};

} // namespace distributed
} // namespace duckdb
