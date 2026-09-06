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

#include "duckdb/common/limits.hpp"
#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/execution/distributed/data_sink.hpp"

#include "duckdb/execution/distributed/utils/channel.hpp"
#include "duckdb/execution/distributed/scheduling/worker.hpp"
#include "duckdb/execution/distributed/plan/plan_config.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/distributed/pipeline_node/translator_api.hpp"
#include "duckdb/execution/distributed/pipeline_node/sink.hpp"
#include "duckdb/execution/distributed/pipeline_node/copy_finish.hpp"
#include "duckdb/execution/distributed/pipeline_node/data_sink_finish.hpp"
#include "duckdb/execution/distributed/pipeline_node/extension_write_sink.hpp"
#include "duckdb/execution/distributed/copy_finalize.hpp"
#include "duckdb/execution/distributed/extension_write_task_provider.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/hive_partitioning.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {
namespace distributed {

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
	std::shared_ptr<PlanTaskExecutor> task_executor_;
	std::shared_ptr<FteTaskSubmitter> fte_task_submitter_;
	TaskIDCounter task_id_counter_;
	TaskInputs initial_inputs_;

public:
	explicit PlanExecutionContext(std::shared_ptr<PlanTaskExecutor> task_executor,
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
		task_executor_->ScheduleTask([task_ptr]() mutable {
			auto result = (*task_ptr)();
			if (result.is_err()) {
				auto msg = std::string("[PlanExecutionContext::spawn] task error: ") + result.error().what();
				throw InternalException(msg);
			}
		});
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

	DuckDBResult<void> materialization_barrier_completed(const std::string &query_id, NodeID node_id) override {
		return worker_manager_->materialization_barrier_completed(query_id, node_id);
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
	wait_query_finished_streaming(const std::string &query_id, double timeout_s,
	                              MaterializedOutputCallback on_output) override {
		return worker_manager_->wait_fte_query_streaming(query_id, timeout_s, std::move(on_output));
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
	                                std::shared_ptr<PlanTaskExecutor> task_executor,
	                                UnboundedSender<MaterializedOutput> output_sender, TaskInputs initial_inputs = {},
	                                MaterializedOutputCallback validate_incremental_output = {}) {
		const bool bound_execution_errors = static_cast<bool>(validate_incremental_output);
		auto execution_error = [&](const DuckDBError &error) -> DuckDBResult<void> {
			if (!bound_execution_errors) {
				return DuckDBResult<void>::err(error);
			}
			return DuckDBResult<void>::err(DuckDBError::external_error(BoundDataSinkOutcomeError(error.what())));
		};
		auto execution_error_detail = [&](const DuckDBError &error) {
			return bound_execution_errors ? BoundDataSinkOutcomeError(error.what()) : string(error.what());
		};
		if (pipeline_node->is_statically_empty_result()) {
			return DuckDBResult<void>::ok();
		}
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
				submit_done_msg << " error=" << FteRunnerFormatField(execution_error_detail(submit_res.error()));
			}
			FteRunnerDebugLog(submit_done_msg.str());
			return submit_res.is_err() ? execution_error(submit_res.error()) : DuckDBResult<void>::ok();
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
					return submit_res;
				}
			}

			FteRunnerDebugLog(
			    "event=task_stream_exhausted elapsed_ms=" + std::to_string(FteRunnerElapsedMs(execute_started_at)) +
			    " total_events=" + std::to_string(fte_event_count) + " source_count=" +
			    std::to_string(fte_source_node_ids.size()) + " query_id=" + FteRunnerFormatField(query_id));
			if (!fte_source_node_ids.empty()) {
				auto exhausted_res = fte_task_submitter->task_input_stream_exhausted(query_id, fte_source_node_ids);
				if (exhausted_res.is_err()) {
					return execution_error(exhausted_res.error());
				}
			}
			if (!query_id.empty()) {
				FteRunnerDebugLog(
				    "event=wait_query_start elapsed_ms=" + std::to_string(FteRunnerElapsedMs(execute_started_at)) +
				    " query_id=" + FteRunnerFormatField(query_id) + " total_events=" + std::to_string(fte_event_count));
				auto publish_output = [&output_sender, &validate_incremental_output](
				                          const MaterializedOutput &output) -> DuckDBResult<void> {
					if (validate_incremental_output) {
						auto validation_res = validate_incremental_output(output);
						if (validation_res.is_err()) {
							return validation_res;
						}
					}
					return output_sender.send(output);
				};
				auto wait_res = validate_incremental_output
				                    ? fte_task_submitter->wait_query_finished_streaming(
				                          query_id, FteQueryWaitTimeoutSeconds(), publish_output)
				                    : fte_task_submitter->wait_query_finished(query_id, FteQueryWaitTimeoutSeconds());
				if (wait_res.is_err()) {
					FteRunnerDebugLog(
					    "event=wait_query_done elapsed_ms=" + std::to_string(FteRunnerElapsedMs(execute_started_at)) +
					    " query_id=" + FteRunnerFormatField(query_id) +
					    " result=err error=" + FteRunnerFormatField(execution_error_detail(wait_res.error())));
					return execution_error(wait_res.error());
				}
				auto outputs = std::move(wait_res).value();
				FteRunnerDebugLog(
				    "event=wait_query_done elapsed_ms=" + std::to_string(FteRunnerElapsedMs(execute_started_at)) +
				    " query_id=" + FteRunnerFormatField(query_id) +
				    " result=ok output_count=" + std::to_string(outputs.size()));
				if (validate_incremental_output && !outputs.empty()) {
					return DuckDBResult<void>::err(DuckDBError::invalid_state_error(
					    "streaming FTE result drain returned outputs outside its publication callback"));
				}
				if (!validate_incremental_output) {
					for (auto &output : outputs) {
						auto send_res = output_sender.send(std::move(output));
						if (send_res.is_err()) {
							return execution_error(send_res.error());
						}
					}
				}
			} else if (fte_event_count > 0) {
				return DuckDBResult<void>::err(
				    DuckDBError::invalid_state_error("FTE runner cannot wait for query completion without query_id"));
			}
		} catch (const std::exception &ex) {
			if (validate_incremental_output) {
				return DuckDBResult<void>::err(DuckDBError::external_error(BoundDataSinkOutcomeError(ex.what())));
			}
			return DuckDBResult<void>::err(DuckDBError::external_error(ex.what()));
		} catch (...) {
			return DuckDBResult<void>::err(DuckDBError::external_error("execute_plan unknown exception"));
		}

		return DuckDBResult<void>::ok();
	}

	/// Unified terminal result type. Each tag owns an independent protocol.
	struct PlanResult {
		enum Tag { STREAMING, COPY, EXTENSION_WRITE, DATA_SINK };
		Tag tag;
		// Only one of these is valid depending on tag
		PlanResultStream stream;
		DistributedCopyResult copy_result;
		DistributedExtensionWriteResult extension_write_result;
		DistributedDataSinkResult data_sink_result;

		// Streaming constructor
		static PlanResult make_streaming(std::shared_ptr<PlanTaskExecutor> te,
		                                 UnboundedReceiver<MaterializedOutput> recv,
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
		static PlanResult make_extension_write(DistributedExtensionWriteResult result) {
			PlanResult r;
			r.tag = EXTENSION_WRITE;
			r.extension_write_result = std::move(result);
			return r;
		}
		static PlanResult make_data_sink(DistributedDataSinkResult result) {
			PlanResult r;
			r.tag = DATA_SINK;
			r.data_sink_result = std::move(result);
			return r;
		}
	};

	/// Unified run_plan: auto-detects terminal protocols and handles streaming and finalized results.
	/// - Non-terminal plans → returns PlanResultStream (streaming pull)
	/// - Terminal plans → collect selected outputs and finalize their protocol-specific result
	DuckDBResult<PlanResult> run_plan(std::shared_ptr<DistributedPhysicalPlan> plan, TaskInputs initial_inputs = {},
	                                  std::function<void()> on_execution_started = {}) {
		if (!client_context_) {
			return DuckDBResult<PlanResult>::err(DuckDBError("run_plan requires a ClientContext"));
		}
		if (!plan) {
			return DuckDBResult<PlanResult>::err(DuckDBError("run_plan requires a distributed physical plan"));
		}

		// ── Step 1: Validate extension writes, then translate the physical plan ──
		auto physical_plan = plan->physical_plan();
		if (!physical_plan || !physical_plan->HasRoot()) {
			return DuckDBResult<PlanResult>::err(DuckDBError("run_plan requires a physical plan root"));
		}
		auto extension_write_provider = physical_plan->Root().GetExtensionWriteTaskProvider();
		const bool has_data_sink_root = physical_plan->Root().type == PhysicalOperatorType::DATA_SINK;
		if (has_data_sink_root && !client_context_->transaction.IsAutoCommit()) {
			return DuckDBResult<PlanResult>::err(DuckDBError::invalid_state_error(
			    "distributed DataSink writes require DuckDB auto-commit mode and cannot participate in an explicit "
			    "transaction"));
		}
		unique_ptr<DistributedExtensionWriteInfo> extension_write_info;
		string extension_write_query_id;
		vector<DistributedWriteTaskResult> selected_task_results;
		bool extension_abort_attempted = false;
		bool extension_prepare_started = false;
		bool extension_finalize_started = false;
		auto extension_outcome_unknown = [&](DistributedExtensionWriteResult result,
		                                     string error) -> DuckDBResult<PlanResult> {
			result.catalog_committed = false;
			result.outcome_unknown = true;
			result.outcome_error = std::move(error);
			if (result.info.mode == DistributedWriteMode::FILE_ARTIFACT) {
				result.file_result.output_outcome_unknown = true;
				result.file_result.output_outcome_error = result.outcome_error;
			}
			return DuckDBResult<PlanResult>::ok(PlanResult::make_extension_write(std::move(result)));
		};
		auto abort_extension_write = [&]() -> string {
			if (!extension_write_provider || !extension_prepare_started || extension_abort_attempted ||
			    extension_finalize_started) {
				return string();
			}
			extension_abort_attempted = true;
			try {
				extension_write_provider->AbortDistributedWrite(*client_context_, selected_task_results);
				return string();
			} catch (const std::exception &ex) {
				return ex.what();
			} catch (...) {
				return "unknown abort failure";
			}
		};
		if (extension_write_provider) {
			if (physical_plan->Root().type != PhysicalOperatorType::EXTENSION) {
				return DuckDBResult<PlanResult>::err(DuckDBError::invalid_state_error(
				    "distributed extension write provider must be exposed by an EXTENSION physical root"));
			}
			try {
				extension_write_query_id = plan->query_id();
				if (extension_write_query_id.empty()) {
					throw InvalidInputException("distributed extension write requires a non-empty query identity");
				}
				extension_write_info = make_uniq<DistributedExtensionWriteInfo>(
				    ResolveDistributedExtensionWriteInfo(*client_context_, extension_write_provider->WritePlan()));
			} catch (const std::exception &ex) {
				return DuckDBResult<PlanResult>::err(DuckDBError::invalid_state_error(
				    StringUtil::Format("distributed extension write protocol validation failed: %s", ex.what())));
			}
			if (!client_context_->transaction.IsAutoCommit()) {
				return DuckDBResult<PlanResult>::err(DuckDBError::invalid_state_error(
				    "distributed extension write requires DuckDB auto-commit mode so Vane can own its catalog "
				    "transaction boundary"));
			}
			if (!client_context_->transaction.HasActiveTransaction()) {
				return DuckDBResult<PlanResult>::err(DuckDBError::invalid_state_error(
				    "distributed extension write requires an active Vane-owned auto-commit transaction"));
			}
			try {
				extension_write_provider->ValidateDistributedWrite(*client_context_);
			} catch (const std::exception &ex) {
				return DuckDBResult<PlanResult>::err(DuckDBError(ex.what()));
			} catch (...) {
				return DuckDBResult<PlanResult>::err(
				    DuckDBError::external_error("distributed extension write validation threw an unknown exception"));
			}
		}
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
			pipeline_res = physical_plan_to_pipeline_node_wrapper(cfg, physical_plan, client_context_.get(),
			                                                      extension_write_info.get());
		} catch (const std::exception &ex) {
			return DuckDBResult<PlanResult>::err(DuckDBError(std::string("Failed to translate plan: ") + ex.what()));
		} catch (...) {
			return DuckDBResult<PlanResult>::err(
			    DuckDBError::external_error("physical plan translation threw an unknown exception"));
		}
		if (pipeline_res.is_err()) {
			return DuckDBResult<PlanResult>::err(pipeline_res.error());
		}
		if (!pipeline_res.value()) {
			return DuckDBResult<PlanResult>::err(DuckDBError("Pipeline translation returned null"));
		}
		auto pipeline_node = pipeline_res.value();

		// ── Step 2: Resolve the exact sink protocol ──
		std::shared_ptr<CopyFinishNode> copy_sink_node;
		std::shared_ptr<ExtensionWriteSinkNode> callback_sink_node;
		std::shared_ptr<DataSinkFinishNode> data_sink_node;
		idx_t copy_sink_count = 0;
		idx_t callback_sink_count = 0;
		idx_t data_sink_count = 0;
		std::function<void(const DistributedPipelineNodeRef &)> find_sinks = [&](const DistributedPipelineNodeRef &n) {
			if (!n) {
				return;
			}
			auto impl = n->inner();
			if (impl && impl->is_sink()) {
				if (auto copy_sink = std::dynamic_pointer_cast<CopyFinishNode>(impl)) {
					copy_sink_count++;
					if (!copy_sink_node) {
						copy_sink_node = std::move(copy_sink);
					}
				}
				if (auto callback_sink = std::dynamic_pointer_cast<ExtensionWriteSinkNode>(impl)) {
					callback_sink_count++;
					if (!callback_sink_node) {
						callback_sink_node = std::move(callback_sink);
					}
				}
				if (auto data_sink = std::dynamic_pointer_cast<DataSinkFinishNode>(impl)) {
					data_sink_count++;
					if (!data_sink_node) {
						data_sink_node = std::move(data_sink);
					}
				}
			}
			for (auto &child : n->arc_children()) {
				find_sinks(child);
			}
		};
		find_sinks(pipeline_node);
		if (extension_write_provider) {
			if (extension_write_info->mode == DistributedWriteMode::FILE_ARTIFACT &&
			    (copy_sink_count != 1 || callback_sink_count != 0 || data_sink_count != 0)) {
				return DuckDBResult<PlanResult>::err(DuckDBError::invalid_state_error(
				    StringUtil::Format("distributed file-artifact write %s did not translate to exactly one COPY sink",
				                       extension_write_info->Name())));
			}
			if (extension_write_info->mode == DistributedWriteMode::CALLBACK &&
			    (callback_sink_count != 1 || copy_sink_count != 0 || data_sink_count != 0)) {
				return DuckDBResult<PlanResult>::err(DuckDBError::invalid_state_error(
				    StringUtil::Format("distributed callback write %s did not translate to exactly one callback sink",
				                       extension_write_info->Name())));
			}
			if (copy_sink_node && !copy_sink_node->staging_root_base().empty()) {
				return DuckDBResult<PlanResult>::err(DuckDBError::invalid_state_error(
				    StringUtil::Format("distributed file-artifact write %s requires worker direct-write output",
				                       extension_write_info->Name())));
			}
		}
		if ((has_data_sink_root &&
		     (data_sink_count != 1 || copy_sink_count != 0 || callback_sink_count != 0 || extension_write_provider)) ||
		    (!has_data_sink_root && data_sink_count != 0)) {
			return DuckDBResult<PlanResult>::err(DuckDBError::invalid_state_error(
			    "distributed DataSink must be the only terminal sink and the physical plan root"));
		}

		std::string sink_base_path;
		std::string sink_worker_base_path;
		if (copy_sink_node) {
			auto &fs = FileSystem::GetFileSystem(*client_context_);
			auto canonical_res = CanonicalDistributedCopyBasePath(fs, copy_sink_node->spec());
			if (canonical_res.is_err()) {
				return DuckDBResult<PlanResult>::err(canonical_res.error());
			}
			sink_base_path = std::move(canonical_res).value();
			auto worker_base_res = CanonicalDistributedCopyBasePath(fs, copy_sink_node->spec().file_path);
			if (worker_base_res.is_err()) {
				return DuckDBResult<PlanResult>::err(worker_base_res.error());
			}
			sink_worker_base_path = std::move(worker_base_res).value();
		}

		auto cleanup_copy_output = [&]() -> DuckDBResult<void> {
			try {
				if (!copy_sink_node) {
					return DuckDBResult<void>::ok();
				}
				auto &fs = FileSystem::GetFileSystem(*client_context_);
				if (copy_sink_node->staging_root_base().empty()) {
					auto cleanup_res =
					    extension_write_provider
					        ? CleanupDistributedCopyUncommittedDirectWriteRunWithWorkerBase(
					              fs, sink_base_path, sink_worker_base_path, copy_sink_node->staging_run_id())
					        : CleanupDistributedCopyUncommittedDirectWriteRun(fs, sink_base_path,
					                                                          copy_sink_node->staging_run_id());
					if (cleanup_res.is_err()) {
						return DuckDBResult<void>::err(cleanup_res.error());
					}
					return DuckDBResult<void>::ok();
				}
				auto staging_root = fs.JoinPath(copy_sink_node->staging_root_base(), copy_sink_node->staging_run_id());
				RemoveDistributedCopyDirectoryTree(fs, staging_root);
				RemoveDistributedCopyDirectoryIfEmpty(fs, copy_sink_node->staging_root_base());
				return DuckDBResult<void>::ok();
			} catch (const std::exception &ex) {
				return DuckDBResult<void>::err(
				    DuckDBError::io_error("distributed write output cleanup threw: " + string(ex.what())));
			} catch (...) {
				return DuckDBResult<void>::err(
				    DuckDBError::io_error("distributed write output cleanup threw an unknown exception"));
			}
		};
		auto fail_after_copy_cleanup = [&](const DuckDBError &primary_error) -> DuckDBResult<PlanResult> {
			auto copy_cleanup_res = cleanup_copy_output();
			if (copy_cleanup_res.is_err()) {
				return DuckDBResult<PlanResult>::err(DuckDBError::io_error(StringUtil::Format(
				    "%s; cleanup failed: %s", primary_error.what(), copy_cleanup_res.error().what())));
			}
			return DuckDBResult<PlanResult>::err(primary_error);
		};
		auto fail_after_write_cleanup = [&](const DuckDBError &primary_error) -> DuckDBResult<PlanResult> {
			if (extension_finalize_started) {
				return DuckDBResult<PlanResult>::err(primary_error);
			}
			auto abort_error = abort_extension_write();
			auto copy_cleanup_res = cleanup_copy_output();
			vector<string> cleanup_errors;
			if (copy_cleanup_res.is_err()) {
				cleanup_errors.push_back(copy_cleanup_res.error().what());
			}
			if (!abort_error.empty()) {
				cleanup_errors.push_back("extension abort failed: " + abort_error);
			}
			if (!cleanup_errors.empty()) {
				return DuckDBResult<PlanResult>::err(DuckDBError::io_error(StringUtil::Format(
				    "%s; cleanup failed: %s", primary_error.what(), StringUtil::Join(cleanup_errors, "; "))));
			}
			return DuckDBResult<PlanResult>::err(primary_error);
		};
		auto abort_write_workers = [&]() -> DuckDBResult<void> {
			if (!worker_manager_) {
				return DuckDBResult<void>::err(
				    DuckDBError::invalid_state_error("distributed write has no worker manager to quiesce"));
			}
			if (plan->query_id().empty()) {
				return DuckDBResult<void>::err(
				    DuckDBError::invalid_state_error("distributed write cannot quiesce workers without a query id"));
			}
			try {
				return worker_manager_->abort_and_quiesce_query(plan->query_id());
			} catch (const std::exception &ex) {
				if (data_sink_node) {
					return DuckDBResult<void>::err(DuckDBError::external_error(BoundDataSinkOutcomeError(
					    "worker manager threw while aborting FTE query: " + BoundDataSinkOutcomeError(ex.what()))));
				}
				return DuckDBResult<void>::err(DuckDBError::external_error(
				    "worker manager threw while aborting FTE query: " + std::string(ex.what())));
			} catch (...) {
				return DuckDBResult<void>::err(
				    DuckDBError::external_error("worker manager threw while aborting FTE query: unknown exception"));
			}
		};
		auto fail_after_worker_abort = [&](const DuckDBError &primary_error) -> DuckDBResult<PlanResult> {
			auto abort_res = abort_write_workers();
			if (abort_res.is_err()) {
				return DuckDBResult<PlanResult>::err(DuckDBError::io_error(
				    StringUtil::Format("%s; worker abort barrier failed: %s; current-execution output is retained",
				                       primary_error.what(), abort_res.error().what())));
			}
			return fail_after_write_cleanup(primary_error);
		};
		vector<ResultPartitionRef> collected_partitions;
		auto data_sink_results =
		    std::make_shared<DataSinkResultCollector>(data_sink_node ? data_sink_node->operation_id() : string());
		std::shared_ptr<UnboundedChannelState<MaterializedOutput>> data_sink_output_state;
		auto disconnect_data_sink_output = [&]() {
			if (data_sink_output_state) {
				data_sink_output_state->disconnect_receiver();
			}
		};
		auto data_sink_unknown_after_worker_abort = [&](const DuckDBError &primary_error) -> DuckDBResult<PlanResult> {
			D_ASSERT(data_sink_node);
			vector<string> errors {BoundDataSinkOutcomeError(primary_error.what())};
			auto abort_res = abort_write_workers();
			if (abort_res.is_err()) {
				errors.push_back(BoundDataSinkOutcomeError("worker abort barrier failed: " +
				                                           BoundDataSinkOutcomeError(abort_res.error().what())));
			}
			DistributedDataSinkResult result;
			result.operation_id = data_sink_node->operation_id();
			auto parsed = data_sink_results->Finalize();
			if (parsed.is_ok()) {
				result = std::move(parsed).value();
			} else {
				errors.push_back(BoundDataSinkOutcomeError("selected worker results could not be parsed: " +
				                                           BoundDataSinkOutcomeError(parsed.error().what())));
			}
			result.outcome_aborted = false;
			result.outcome_unknown = true;
			result.outcome_error = BoundDataSinkOutcomeError(StringUtil::Join(errors, "; "));
			return DuckDBResult<PlanResult>::ok(PlanResult::make_data_sink(std::move(result)));
		};

		bool direct_write_lifecycle_may_exist = false;
		bool execution_may_have_started = false;
		try {
			if (copy_sink_node && copy_sink_node->staging_root_base().empty()) {
				// Persist lifecycle metadata for explicit operator-managed cleanup. Starting a COPY must not age out
				// other runs: elapsed time alone does not establish that another run is abandoned.
				auto &fs = FileSystem::GetFileSystem(*client_context_);
				direct_write_lifecycle_may_exist = true;
				auto lifecycle_res = WriteDistributedCopyDirectWriteLifecycle(
				    fs, sink_base_path, copy_sink_node->staging_run_id(), 0, sink_worker_base_path);
				if (lifecycle_res.is_err()) {
					return fail_after_copy_cleanup(lifecycle_res.error());
				}
			}

			// ── Step 3: Common setup — result channel + FTE execution ──
			// DataSink result parsing runs synchronously in the publication callback,
			// before the selected handle is acknowledged. Keep at most one validated
			// output queued so the coordinator also bounds RefBundle ownership.
			auto channel_pair = create_unbounded_channel<MaterializedOutput>(data_sink_node ? 1 : 0);
			auto sender = std::move(channel_pair.first);
			auto receiver = std::move(channel_pair.second);
			auto execute_status = std::make_shared<PlanExecutionStatus>();
			auto output_state = sender.state();
			if (data_sink_node) {
				data_sink_output_state = output_state;
			}
			auto task_executor = std::make_shared<PlanTaskExecutor>(client_context_, execute_status);

			auto self = this->weak_from_this().lock();
			if (!self) {
				throw InternalException("PlanRunner requires shared_ptr ownership; create via std::make_shared");
			}
			auto sender_ptr = std::make_shared<UnboundedSender<MaterializedOutput>>(std::move(sender));
			auto initial_inputs_ptr = std::make_shared<TaskInputs>(std::move(initial_inputs));
			if (!extension_write_provider) {
				execution_may_have_started = true;
			}
			MaterializedOutputCallback validate_incremental_output;
			if (data_sink_node) {
				const auto data_sink_node_id = data_sink_node->result_node_id();
				validate_incremental_output =
				    [data_sink_results, data_sink_node_id](const MaterializedOutput &output) -> DuckDBResult<void> {
					if (!output.has_node_id(data_sink_node_id)) {
						return DuckDBResult<void>::ok();
					}
					return data_sink_results->Append(output.fragments());
				};
			}
			const bool bound_execution_errors = static_cast<bool>(validate_incremental_output);
			if (extension_write_provider) {
				extension_prepare_started = true;
				try {
					extension_write_provider->PrepareDistributedWrite(*client_context_);
				} catch (const std::exception &ex) {
					return fail_after_write_cleanup(DuckDBError::external_error(
					    "distributed extension write preparation failed: " + string(ex.what())));
				} catch (...) {
					return fail_after_write_cleanup(DuckDBError::external_error(
					    "distributed extension write preparation threw an unknown exception"));
				}
			}
			execution_may_have_started = true;
			task_executor->ScheduleTask(
			    [self, pipeline_node, sender_ptr, output_state, execute_status, task_executor, initial_inputs_ptr,
			     bound_execution_errors,
			     validate_incremental_output = std::move(validate_incremental_output)]() mutable {
				    std::unique_ptr<UnboundedSender<MaterializedOutput>> output_lifetime_guard;
				    auto publish_error = [&](const DuckDBError &error) {
					    execute_status->RecordError(error);
					    if (output_state) {
						    output_state->close();
					    }
				    };
				    try {
					    output_lifetime_guard = make_uniq<UnboundedSender<MaterializedOutput>>(sender_ptr->clone());
					    auto result =
					        self->execute_plan(pipeline_node, task_executor, std::move(*sender_ptr),
					                           std::move(*initial_inputs_ptr), std::move(validate_incremental_output));
					    if (result.is_err()) {
						    publish_error(result.error());
					    }
					    output_lifetime_guard.reset();
				    } catch (const std::exception &ex) {
					    DuckDBError error =
					        bound_execution_errors
					            ? DuckDBError::external_error(BoundDataSinkOutcomeError(
					                  "execute_plan task threw: " + BoundDataSinkOutcomeError(ex.what())))
					            : DuckDBError::external_error(string("execute_plan task threw: ") + ex.what());
					    publish_error(error);
					    output_lifetime_guard.reset();
				    } catch (...) {
					    DuckDBError error = DuckDBError::external_error("execute_plan task threw unknown exception");
					    publish_error(error);
					    output_lifetime_guard.reset();
				    }
			    });
			if (on_execution_started) {
				on_execution_started();
			}

			// ── Step 4: Dispatch based on the exact sink protocol ──
			if (!copy_sink_node && !callback_sink_node && !data_sink_node) {
				// Streaming path: return pull-based stream
				return DuckDBResult<PlanResult>::ok(PlanResult::make_streaming(
				    std::move(task_executor), std::move(receiver), std::move(execute_status)));
			}

			const auto sink_node_id = data_sink_node       ? data_sink_node->result_node_id()
			                          : callback_sink_node ? callback_sink_node->node_id()
			                                               : copy_sink_node->copy_sink()->node_id();
			std::shared_ptr<DuckDBError> deferred_collection_error;
			auto capture_execution_error = [&]() {
				auto execution_error = execute_status->GetError();
				if (execution_error && !deferred_collection_error) {
					deferred_collection_error = std::move(execution_error);
				}
			};
			auto worker_write_started = std::chrono::steady_clock::now();
			try {
				while (true) {
					auto item = receiver.recv();
					capture_execution_error();
					if (!item.first) {
						break;
					}
					// Drain the plan-control channel to its terminal state before
					// entering the worker abort barrier. Queued results are irrelevant
					// after the first execution or result-contract failure.
					if (deferred_collection_error && !data_sink_node) {
						continue;
					}
					if (!item.second.has_node_id(sink_node_id)) {
						continue;
					}
					if (callback_sink_node && item.second.fragments().size() != 1) {
						deferred_collection_error = std::make_shared<DuckDBError>(DuckDBError::invalid_state_error(
						    StringUtil::Format("distributed callback write %s worker output must contain exactly one "
						                       "result partition",
						                       extension_write_info->Name())));
						continue;
					}
					if (!data_sink_node) {
						for (auto &part : item.second.fragments()) {
							collected_partitions.push_back(part);
						}
					}
				}
			} catch (const std::exception &ex) {
				if (data_sink_node) {
					receiver.close();
				}
				capture_execution_error();
				if (!deferred_collection_error) {
					deferred_collection_error = std::make_shared<DuckDBError>(DuckDBError::external_error(
					    data_sink_node ? BoundDataSinkOutcomeError(ex.what()) : string(ex.what())));
				}
			} catch (...) {
				if (data_sink_node) {
					receiver.close();
				}
				capture_execution_error();
				if (!deferred_collection_error) {
					deferred_collection_error = std::make_shared<DuckDBError>(
					    DuckDBError::external_error("distributed write result collection threw an unknown exception"));
				}
			}

			capture_execution_error();
			if (deferred_collection_error) {
				if (data_sink_node) {
					return data_sink_unknown_after_worker_abort(*deferred_collection_error);
				}
				return fail_after_worker_abort(*deferred_collection_error);
			}

			if (data_sink_node) {
				auto finalize_res = data_sink_results->Finalize();
				if (finalize_res.is_err()) {
					return data_sink_unknown_after_worker_abort(finalize_res.error());
				}
				return DuckDBResult<PlanResult>::ok(PlanResult::make_data_sink(std::move(finalize_res).value()));
			}

			if (copy_sink_node) {
				auto worker_write_ms = DistributedCopyElapsedMillis(worker_write_started);
				auto finalize_res = copy_sink_node->finalize(collected_partitions, *client_context_);
				if (finalize_res.is_err()) {
					if (extension_write_provider) {
						return fail_after_write_cleanup(finalize_res.error());
					}
					return DuckDBResult<PlanResult>::err(finalize_res.error());
				}
				auto copy_result = std::move(finalize_res).value();
				copy_result.staging_write_ms = worker_write_ms;
				if (!extension_write_provider) {
					return DuckDBResult<PlanResult>::ok(PlanResult::make_copy(std::move(copy_result)));
				}
				if (copy_result.output_outcome_unknown) {
					DistributedExtensionWriteResult result;
					result.info = *extension_write_info;
					result.rows_written = copy_result.rows_copied;
					result.file_result = std::move(copy_result);
					return extension_outcome_unknown(std::move(result),
					                                 "worker artifact publication outcome is unknown; extension "
					                                 "finalization was not invoked and artifacts "
					                                 "were retained");
				}
				if (!copy_result.output_committed) {
					return fail_after_write_cleanup(DuckDBError::invalid_state_error(StringUtil::Format(
					    "distributed file-artifact write %s did not publish worker artifacts definitively: %s",
					    extension_write_info->Name(), copy_result.output_outcome_error)));
				}
				try {
					vector<DistributedCopyFileInfo> extension_files(copy_result.files.begin(), copy_result.files.end());
					selected_task_results = EncodeDistributedFileWriteResults(
					    *extension_write_info, extension_write_query_id, extension_files);
				} catch (const std::exception &ex) {
					return fail_after_write_cleanup(DuckDBError(ex.what()));
				} catch (...) {
					return fail_after_write_cleanup(
					    DuckDBError::external_error("distributed file result encoding threw an unknown exception"));
				}
				DistributedExtensionWriteResult result;
				result.info = *extension_write_info;
				result.rows_written = copy_result.rows_copied;
				for (const auto &task_result : selected_task_results) {
					if (task_result.ByteCount() > NumericLimits<idx_t>::Maximum() - result.bytes_written) {
						return fail_after_write_cleanup(
						    DuckDBError::invalid_state_error("distributed extension write byte count overflow"));
					}
					result.bytes_written += task_result.ByteCount();
				}
				result.selected_task_results = std::move(selected_task_results);
				result.file_result = std::move(copy_result);
				extension_finalize_started = true;
				try {
					auto affected_rows = extension_write_provider->FinalizeDistributedWrite(
					    *client_context_, result.selected_task_results);
					if (affected_rows != result.rows_written) {
						auto mismatch_error = StringUtil::Format("extension coordinator finalization returned %llu "
						                                         "rows; worker fragments contained %llu rows; "
						                                         "artifacts were retained",
						                                         static_cast<unsigned long long>(affected_rows),
						                                         static_cast<unsigned long long>(result.rows_written));
						return extension_outcome_unknown(std::move(result), std::move(mismatch_error));
					}
					return DuckDBResult<PlanResult>::ok(PlanResult::make_extension_write(std::move(result)));
				} catch (const std::exception &ex) {
					return extension_outcome_unknown(
					    std::move(result),
					    "extension coordinator finalization outcome is unknown; artifacts were retained: " +
					        string(ex.what()));
				} catch (...) {
					return extension_outcome_unknown(
					    std::move(result),
					    "extension coordinator finalization outcome is unknown; artifacts were retained");
				}
			}

			DistributedExtensionWriteResult result;
			result.info = *extension_write_info;
			try {
				vector<ResultPartitionRef> extension_partitions(collected_partitions.begin(),
				                                                collected_partitions.end());
				selected_task_results = ParseDistributedWriteTaskResults(
				    *extension_write_info, extension_write_query_id, extension_partitions);
				for (const auto &task_result : selected_task_results) {
					auto task_rows = task_result.RowCount();
					auto task_bytes = task_result.ByteCount();
					if (task_rows > NumericLimits<idx_t>::Maximum() - result.rows_written ||
					    task_bytes > NumericLimits<idx_t>::Maximum() - result.bytes_written) {
						throw InvalidInputException("distributed extension write '%s' result counts overflow",
						                            extension_write_info->Name());
					}
					result.rows_written += task_rows;
					result.bytes_written += task_bytes;
				}
			} catch (const std::exception &ex) {
				return fail_after_write_cleanup(DuckDBError(ex.what()));
			} catch (...) {
				return fail_after_write_cleanup(
				    DuckDBError::external_error("distributed callback result parsing threw an unknown exception"));
			}
			result.selected_task_results = std::move(selected_task_results);
			extension_finalize_started = true;
			try {
				auto affected_rows =
				    extension_write_provider->FinalizeDistributedWrite(*client_context_, result.selected_task_results);
				if (affected_rows != result.rows_written) {
					auto mismatch_error = StringUtil::Format(
					    "extension coordinator finalization returned %llu rows; worker fragments contained %llu rows; "
					    "artifacts were retained",
					    static_cast<unsigned long long>(affected_rows),
					    static_cast<unsigned long long>(result.rows_written));
					return extension_outcome_unknown(std::move(result), std::move(mismatch_error));
				}
				return DuckDBResult<PlanResult>::ok(PlanResult::make_extension_write(std::move(result)));
			} catch (const std::exception &ex) {
				return extension_outcome_unknown(
				    std::move(result),
				    "extension coordinator finalization outcome is unknown; artifacts were retained: " +
				        string(ex.what()));
			} catch (...) {
				return extension_outcome_unknown(
				    std::move(result),
				    "extension coordinator finalization outcome is unknown; artifacts were retained");
			}
		} catch (const std::exception &ex) {
			const auto execution_error =
			    data_sink_node
			        ? DuckDBError::external_error(BoundDataSinkOutcomeError(
			              "distributed DataSink setup or execution threw: " + BoundDataSinkOutcomeError(ex.what())))
			        : DuckDBError::external_error("distributed write setup or execution threw: " + string(ex.what()));
			if (extension_finalize_started) {
				DistributedExtensionWriteResult result;
				result.info = *extension_write_info;
				return extension_outcome_unknown(
				    std::move(result),
				    "extension coordinator finalization outcome is unknown; artifacts were retained: " +
				        string(ex.what()));
			}
			if (execution_may_have_started) {
				disconnect_data_sink_output();
				auto result = data_sink_node ? data_sink_unknown_after_worker_abort(execution_error)
				                             : fail_after_worker_abort(execution_error);
				return result;
			}
			if (direct_write_lifecycle_may_exist) {
				return fail_after_copy_cleanup(execution_error);
			}
			return DuckDBResult<PlanResult>::err(execution_error);
		} catch (...) {
			const auto execution_error =
			    DuckDBError::external_error("distributed write setup or execution threw an unknown exception");
			if (extension_finalize_started) {
				DistributedExtensionWriteResult result;
				result.info = *extension_write_info;
				return extension_outcome_unknown(
				    std::move(result),
				    "extension coordinator finalization outcome is unknown; artifacts were retained");
			}
			if (execution_may_have_started) {
				disconnect_data_sink_output();
				auto result = data_sink_node ? data_sink_unknown_after_worker_abort(execution_error)
				                             : fail_after_worker_abort(execution_error);
				return result;
			}
			if (direct_write_lifecycle_may_exist) {
				return fail_after_copy_cleanup(execution_error);
			}
			return DuckDBResult<PlanResult>::err(execution_error);
		}
	}

	/// Legacy finalize_copy — kept for Python callers that use the streaming + manual finalize path.
	DuckDBResult<DistributedCopyResult> finalize_copy(const DistributedCopySpec &spec, const string &staging_root,
	                                                  vector<DistributedCopyFileInfo> files) {
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
