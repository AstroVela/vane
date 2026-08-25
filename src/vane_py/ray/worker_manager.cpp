// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "worker_manager.hpp"
#include "bounded_diagnostics.hpp"
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include "duckdb/common/types/uuid.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"

namespace py = pybind11;
using namespace duckdb::distributed::python::ray;
using duckdb::distributed::DuckDBError;
using duckdb::distributed::DuckDBResult;
using duckdb::distributed::TaskResourceRequest;
using duckdb::distributed::WorkerSnapshot;

static constexpr auto REFRESH_INTERVAL = std::chrono::seconds(5);

RayWorkerManager::RayWorkerManager(QueryCleanup query_cleanup)
    : manager_instance_id_(duckdb::UUID::ToString(duckdb::UUID::GenerateRandomUUID())),
      query_cleanup_(std::move(query_cleanup)), query_lifecycles_("Ray worker manager") {
}

static std::vector<std::string> AbortWorkers(const std::vector<std::shared_ptr<RayWorkerRuntime>> &workers) {
	std::vector<std::string> errors;
	for (auto &worker : workers) {
		const auto worker_id = worker && worker->Id() ? *worker->Id() : std::string("<unknown>");
		try {
			if (worker) {
				worker->AbortShutdown();
			}
		} catch (const std::exception &ex) {
			errors.push_back(worker_id + ": " + ex.what());
		} catch (...) {
			errors.push_back(worker_id + ": unknown abort error");
		}
	}
	return errors;
}

static std::string ExceptionMessage(const std::exception_ptr &error) {
	try {
		std::rethrow_exception(error);
	} catch (const std::exception &ex) {
		return ex.what();
	} catch (...) {
		return "unknown exception";
	}
}

bool RayWorkerManager::BeginOperation() const {
	lock_guard<mutex> guard(mutex_);
	if (state_.shutdown_started) {
		return false;
	}
	state_.active_operations++;
	return true;
}

void RayWorkerManager::EndOperation() const {
	{
		lock_guard<mutex> guard(mutex_);
		D_ASSERT(state_.active_operations > 0);
		state_.active_operations--;
	}
	shutdown_cv_.notify_all();
}

std::optional<QueryLifecycleCoordinator::Operation>
RayWorkerManager::BeginQueryOperation(const string &query_id, const string &requested_owner_query_id) {
	return query_lifecycles_.BeginOperation(query_id, requested_owner_query_id, !requested_owner_query_id.empty());
}

void RayWorkerManager::EndQueryOperation(const QueryLifecycleCoordinator::Operation &operation) {
	query_lifecycles_.EndOperation(operation);
}

void RayWorkerManager::register_query_owner(const string &query_id, const string &owner_query_id) {
	OperationGuard operation(*this);
	if (!operation) {
		throw std::runtime_error("cannot register FTE query ownership after Ray worker manager shutdown");
	}
	query_lifecycles_.RegisterImmediate(query_id, owner_query_id);
}

void RayWorkerManager::WaitForQueryOperations(const QueryLifecycleCoordinator::LifecycleRef &lifecycle) {
	if (Py_IsInitialized() && !duckdb::PythonIsFinalizing() && PyGILState_Check()) {
		py::gil_scoped_release release;
		query_lifecycles_.WaitForOperations(lifecycle);
		return;
	}
	query_lifecycles_.WaitForOperations(lifecycle);
}

void RayWorkerManager::RecordQueryWorkers(const string &owner_query_id,
                                          const std::vector<std::shared_ptr<RayWorkerRuntime>> &workers) {
	lock_guard<mutex> guard(mutex_);
	auto &owned_workers = state_.workers_by_query_owner[owner_query_id];
	for (const auto &worker : workers) {
		if (!worker || std::find(owned_workers.begin(), owned_workers.end(), worker) != owned_workers.end()) {
			continue;
		}
		owned_workers.push_back(worker);
	}
}
std::optional<QueryLifecycleCoordinator::Abort> RayWorkerManager::BeginQueryAbort(const string &query_id) {
	if (Py_IsInitialized() && !duckdb::PythonIsFinalizing() && PyGILState_Check()) {
		py::gil_scoped_release release;
		return query_lifecycles_.BeginAbort(query_id);
	}
	return query_lifecycles_.BeginAbort(query_id);
}

std::optional<QueryLifecycleCoordinator::Abort>
RayWorkerManager::BeginQueryAbort(const QueryLifecycleCoordinator::Teardown &teardown) {
	if (Py_IsInitialized() && !duckdb::PythonIsFinalizing() && PyGILState_Check()) {
		py::gil_scoped_release release;
		return query_lifecycles_.BeginAbort(teardown);
	}
	return query_lifecycles_.BeginAbort(teardown);
}

std::optional<QueryLifecycleCoordinator::Teardown> RayWorkerManager::BeginQueryTeardown(const string &query_id) {
	if (Py_IsInitialized() && !duckdb::PythonIsFinalizing() && PyGILState_Check()) {
		py::gil_scoped_release release;
		return query_lifecycles_.BeginTeardown(query_id);
	}
	return query_lifecycles_.BeginTeardown(query_id);
}

std::vector<std::shared_ptr<RayWorkerRuntime>> RayWorkerManager::QueryWorkers(const string &owner_query_id) const {
	std::vector<std::shared_ptr<RayWorkerRuntime>> workers;
	lock_guard<mutex> guard(mutex_);
	auto add_worker = [&](const std::shared_ptr<RayWorkerRuntime> &worker) {
		if (worker && std::find(workers.begin(), workers.end(), worker) == workers.end()) {
			workers.push_back(worker);
		}
	};
	auto owned_workers = state_.workers_by_query_owner.find(owner_query_id);
	if (owned_workers != state_.workers_by_query_owner.end()) {
		for (const auto &worker : owned_workers->second) {
			add_worker(worker);
		}
	}
	for (const auto &entry : state_.ray_workers) {
		add_worker(entry.second);
	}
	return workers;
}

bool RayWorkerManager::ShutdownStarted() const {
	lock_guard<mutex> guard(mutex_);
	return state_.shutdown_started;
}

bool RayWorkerManager::RetireWorkerForFailure(const string &worker_id, const std::shared_ptr<RayWorkerRuntime> &worker,
                                              const std::shared_ptr<std::atomic<bool>> &retired) const {
	std::shared_ptr<RayWorkerRuntime> retired_worker;
	{
		lock_guard<mutex> guard(mutex_);
		if (state_.shutdown_started) {
			return false;
		}
		retired->store(true);
		auto entry = state_.ray_workers.find(duckdb::distributed::make_worker_id(worker_id));
		if (entry != state_.ray_workers.end() && entry->second == worker) {
			retired_worker = std::move(entry->second);
			state_.ray_workers.erase(entry);
			state_.worker_membership_version++;
		}
		state_.last_refresh = {};
	}
	return true;
}

std::string
duckdb::distributed::python::ray::SubmissionErrorOwnerQueryId(const std::vector<duckdb::distributed::WorkerTask> &tasks,
                                                              const std::string &execution_query_id) {
	if (tasks.empty()) {
		return execution_query_id;
	}
	std::string resource_query_id;
	for (const auto &task : tasks) {
		const auto &context = task.context();
		auto it = context.find("resource_query_id");
		if (it == context.end() || it->second.empty()) {
			throw std::runtime_error("FTE submit task requires a non-empty resource_query_id");
		}
		if (resource_query_id.empty()) {
			resource_query_id = it->second;
			continue;
		}
		if (resource_query_id != it->second) {
			throw std::runtime_error("FTE submit batch contains multiple resource_query_id values");
		}
	}
	return resource_query_id;
}

std::string RayWorkerManager::QueryIdFromTaskEvents(const std::vector<duckdb::distributed::WorkerTask> &tasks) {
	std::string query_id;
	for (const auto &task : tasks) {
		const auto &context = task.context();
		auto it = context.find("query_id");
		if (it == context.end() || it->second.empty()) {
			continue;
		}
		if (query_id.empty()) {
			query_id = it->second;
			continue;
		}
		if (query_id != it->second) {
			throw std::runtime_error("FTE submit batch contains multiple query_id values");
		}
	}
	return query_id;
}

void RayWorkerManager::StoreFteResultHandles(
    const string &query_id, std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> handles) {
	if (query_id.empty() || handles.empty()) {
		return;
	}
	std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> live_handles;
	live_handles.reserve(handles.size());
	for (auto &handle : handles) {
		if (handle) {
			live_handles.push_back(std::move(handle));
		}
	}
	if (live_handles.empty()) {
		return;
	}
	lock_guard<mutex> guard(mutex_);
	auto &stored = state_.fte_result_handles_by_query[query_id];
	stored.reserve(stored.size() + live_handles.size());
	for (auto &handle : live_handles) {
		stored.push_back(std::move(handle));
	}
}

void RayWorkerManager::RetainFteResultHandles(
    const string &query_id, std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> handles) {
	if (query_id.empty() || handles.empty()) {
		return;
	}
	lock_guard<mutex> guard(mutex_);
	auto &retained = state_.retained_fte_result_handles_by_query[query_id];
	retained.reserve(retained.size() + handles.size());
	for (auto &handle : handles) {
		retained.push_back(std::move(handle));
	}
}

void RayWorkerManager::ClearFteResultHandles(const string &query_id) {
	if (query_id.empty()) {
		return;
	}
	std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> handles;
	std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> retained_handles;
	{
		lock_guard<mutex> guard(mutex_);
		auto it = state_.fte_result_handles_by_query.find(query_id);
		if (it != state_.fte_result_handles_by_query.end()) {
			handles = std::move(it->second);
			state_.fte_result_handles_by_query.erase(it);
		}
		auto retained_it = state_.retained_fte_result_handles_by_query.find(query_id);
		if (retained_it != state_.retained_fte_result_handles_by_query.end()) {
			retained_handles = std::move(retained_it->second);
			state_.retained_fte_result_handles_by_query.erase(retained_it);
		}
	}
	vane::BoundedErrorDetails errors;
	std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> retry_handles;
	auto release_all = [&](auto &owned_handles, const char *kind) {
		for (size_t index = 0; index < owned_handles.size(); index++) {
			try {
				owned_handles[index]->ReleasePollResult();
			} catch (const std::exception &ex) {
				errors.Add(std::string(kind) + "[" + std::to_string(index) + "]", ex.what());
				retry_handles.push_back(std::move(owned_handles[index]));
			} catch (...) {
				errors.Add(std::string(kind) + "[" + std::to_string(index) + "]", "unknown release error");
				retry_handles.push_back(std::move(owned_handles[index]));
			}
		}
	};
	release_all(handles, "pending");
	release_all(retained_handles, "retained");
	StoreFteResultHandles(query_id, std::move(retry_handles));
	if (errors) {
		throw std::runtime_error(
		    errors.AppendTo("failed to release " + std::to_string(errors.Count()) + " FTE result handle(s)"));
	}
}

DuckDBResult<void> RayWorkerManager::CollectFteResultHandles(const string &query_id) {
	std::vector<std::shared_ptr<RayWorkerRuntime>> workers;
	{
		lock_guard<mutex> guard(mutex_);
		workers.reserve(state_.ray_workers.size());
		for (auto &kv : state_.ray_workers) {
			workers.push_back(kv.second);
		}
	}
	try {
		for (auto &worker : workers) {
			auto handles = worker->PopFteResultHandles(query_id);
			std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> wrapped;
			wrapped.reserve(handles.size());
			for (auto &handle : handles) {
				wrapped.push_back(make_uniq<RayWorkerRuntime::TaskResultHandleType>(std::move(handle)));
			}
			{
				lock_guard<mutex> guard(mutex_);
				auto &counts = state_.fte_result_handle_counts_by_query[query_id];
				for (const auto &handle : wrapped) {
					auto &count = counts[handle->GetFteTaskId()];
					if (count < 2) {
						count++;
					}
				}
			}
			StoreFteResultHandles(query_id, std::move(wrapped));
		}
	} catch (const std::exception &e) {
		return DuckDBResult<void>::err(DuckDBError(
		    vane::BoundedErrorDetails::FormatDetail("Python error while collecting FTE result handles", e.what())));
	}
	return DuckDBResult<void>::ok();
}

DuckDBResult<void>
RayWorkerManager::ValidateFteResultHandleCoverage(const string &query_id,
                                                  const std::unordered_set<string> &selected_attempt_task_ids) const {
	size_t missing = 0;
	size_t duplicate = 0;
	{
		lock_guard<mutex> guard(mutex_);
		auto query_it = state_.fte_result_handle_counts_by_query.find(query_id);
		for (const auto &task_id : selected_attempt_task_ids) {
			size_t count = 0;
			if (query_it != state_.fte_result_handle_counts_by_query.end()) {
				auto task_it = query_it->second.find(task_id);
				if (task_it != query_it->second.end()) {
					count = task_it->second;
				}
			}
			missing += count == 0 ? 1 : 0;
			duplicate += count > 1 ? 1 : 0;
		}
	}
	if (duplicate > 0) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error(
		    "FTE query returned multiple result handles for one selected attempt; duplicate selected attempt count=" +
		    std::to_string(duplicate)));
	}
	if (missing > 0) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error(
		    "FTE query returned no result handle for " + std::to_string(missing) + " selected attempt(s)"));
	}
	return DuckDBResult<void>::ok();
}

DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>> RayWorkerManager::DrainFteResultHandles(
    const string &query_id, double timeout_s, const RayWorkerRuntime::QueryStatus *finished_status,
    const std::unordered_set<duckdb::distributed::TaskContext, duckdb::distributed::TaskContextHash>
        *task_context_filter,
    bool release_payloads, duckdb::distributed::MaterializedOutputCallback on_output, bool selected_only) {
	std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> handles;
	{
		lock_guard<mutex> guard(mutex_);
		auto it = state_.fte_result_handles_by_query.find(query_id);
		if (it != state_.fte_result_handles_by_query.end()) {
			auto stored_handles = std::move(it->second);
			state_.fte_result_handles_by_query.erase(it);
			if (task_context_filter && !task_context_filter->empty()) {
				std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> retained_handles;
				retained_handles.reserve(stored_handles.size());
				handles.reserve(stored_handles.size());
				for (auto &handle : stored_handles) {
					if (task_context_filter->find(handle->GetTaskContext()) == task_context_filter->end()) {
						retained_handles.push_back(std::move(handle));
					} else {
						handles.push_back(std::move(handle));
					}
				}
				if (!retained_handles.empty()) {
					auto &retained = state_.fte_result_handles_by_query[query_id];
					retained.reserve(retained.size() + retained_handles.size());
					for (auto &handle : retained_handles) {
						retained.push_back(std::move(handle));
					}
				}
			} else {
				handles = std::move(stored_handles);
			}
		}
	}
	if (selected_only) {
		std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> selected_handles;
		std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> pending_handles;
		const auto *selected_attempts = finished_status ? &finished_status->selected_attempt_task_ids : nullptr;
		selected_handles.reserve(handles.size());
		pending_handles.reserve(handles.size());
		for (auto &handle : handles) {
			const auto &fte_task_id = handle->GetFteTaskId();
			if (selected_attempts && !fte_task_id.empty() &&
			    selected_attempts->find(fte_task_id) != selected_attempts->end()) {
				selected_handles.push_back(std::move(handle));
			} else {
				pending_handles.push_back(std::move(handle));
			}
		}
		StoreFteResultHandles(query_id, std::move(pending_handles));
		handles = std::move(selected_handles);
	}

	struct DrainedOutput {
		duckdb::distributed::TaskContext task_context;
		size_t ordinal;
		duckdb::distributed::MaterializedOutput output;
	};
	std::vector<DrainedOutput> drained_outputs;
	size_t output_ordinal = 0;
	std::vector<duckdb::distributed::MaterializedOutput> outputs;
	if (handles.empty()) {
		return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::ok(std::move(outputs));
	}
	if (!selected_only && finished_status && !finished_status->selected_attempt_task_ids.empty()) {
		std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> selected_handles;
		std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> retry_handles;
		vane::BoundedErrorDetails release_errors;
		selected_handles.reserve(handles.size());
		for (size_t index = 0; index < handles.size(); index++) {
			auto &handle = handles[index];
			const auto &fte_task_id = handle->GetFteTaskId();
			if (finished_status->selected_attempt_task_ids.find(fte_task_id) ==
			    finished_status->selected_attempt_task_ids.end()) {
				try {
					// ACK transfers output-lease ownership to the consumer. A losing
					// attempt has no consumer, so release its lease-owning handle directly.
					handle->ReleasePollResult();
				} catch (const std::exception &ex) {
					release_errors.Add("unselected[" + std::to_string(index) + "]", ex.what());
					retry_handles.push_back(std::move(handle));
				} catch (...) {
					release_errors.Add("unselected[" + std::to_string(index) + "]", "unknown release error");
					retry_handles.push_back(std::move(handle));
				}
				continue;
			}
			selected_handles.push_back(std::move(handle));
		}
		handles = std::move(selected_handles);
		StoreFteResultHandles(query_id, std::move(retry_handles));
		if (release_errors) {
			StoreFteResultHandles(query_id, std::move(handles));
			return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(DuckDBError::external_error(
			    release_errors.AppendTo("failed to release unselected FTE result handle(s)")));
		}
		if (handles.empty()) {
			return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::ok(std::move(outputs));
		}
	}
	const bool discard_unselected_outputs =
	    !selected_only && finished_status && finished_status->selected_attempt_task_ids.empty();
	if (on_output) {
		std::stable_sort(handles.begin(), handles.end(), [](const auto &lhs, const auto &rhs) {
			const auto lhs_context = lhs->GetTaskContext();
			const auto rhs_context = rhs->GetTaskContext();
			if (lhs_context.query_idx() != rhs_context.query_idx()) {
				return lhs_context.query_idx() < rhs_context.query_idx();
			}
			if (lhs_context.last_node_id() != rhs_context.last_node_id()) {
				return lhs_context.last_node_id() < rhs_context.last_node_id();
			}
			if (lhs_context.task_id() != rhs_context.task_id()) {
				return lhs_context.task_id() < rhs_context.task_id();
			}
			return lhs->GetFteTaskId() < rhs->GetFteTaskId();
		});
	}

	std::vector<bool> retain_payload_until_query_cleanup(handles.size(), false);
	std::vector<bool> finished(handles.size(), false);
	size_t remaining = handles.size();
	auto retain_finished_stream_handles = [&]() {
		std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> delivered_handles;
		std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> pending_handles;
		for (size_t index = 0; index < handles.size(); index++) {
			if (!handles[index]) {
				continue;
			}
			if (finished[index]) {
				delivered_handles.push_back(std::move(handles[index]));
			} else {
				pending_handles.push_back(std::move(handles[index]));
			}
		}
		// Once publication was attempted, retrying this wait must not invoke the
		// callback again: validation or channel send may already have had effects.
		// Keep finished handles cleanup-only while preserving later work normally.
		RetainFteResultHandles(query_id, std::move(delivered_handles));
		StoreFteResultHandles(query_id, std::move(pending_handles));
	};
	// Internal helper convention: negative timeout means no deadline; zero means poll once then time out.
	const auto deadline = timeout_s >= 0.0 ? std::chrono::steady_clock::now() +
	                                             std::chrono::duration_cast<std::chrono::steady_clock::duration>(
	                                                 std::chrono::duration<double>(timeout_s))
	                                       : std::chrono::steady_clock::time_point::max();

	while (remaining > 0) {
		bool had_progress = false;
		for (size_t i = 0; i < handles.size(); i++) {
			if (finished[i]) {
				continue;
			}
			auto poll_res = handles[i]->poll();
			if (!poll_res.first) {
				if (on_output) {
					break;
				}
				continue;
			}

			finished[i] = true;
			remaining--;
			had_progress = true;
			auto task_context = handles[i]->GetTaskContext();
			auto task_result = std::move(poll_res.second);
			if (task_result.is_err()) {
				auto error = task_result.error();
				StoreFteResultHandles(query_id, std::move(handles));
				return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(error);
			}

			auto maybe_output = std::move(task_result.value());
			if (maybe_output.first && !discard_unselected_outputs) {
				if (!on_output && !release_payloads && !maybe_output.second.has_exchange_sink_instance()) {
					retain_payload_until_query_cleanup[i] = true;
				}
				if (on_output) {
					try {
						auto callback_res = on_output(maybe_output.second);
						if (callback_res.is_err()) {
							retain_finished_stream_handles();
							return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
							    callback_res.error());
						}
					} catch (const std::exception &ex) {
						retain_finished_stream_handles();
						return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
						    DuckDBError::external_error(vane::BoundedErrorDetails::FormatDetail(
						        "streaming FTE output callback threw", ex.what())));
					} catch (...) {
						retain_finished_stream_handles();
						return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
						    DuckDBError::external_error("streaming FTE output callback threw an unknown exception"));
					}
				}
				if (!on_output) {
					drained_outputs.push_back({task_context, output_ordinal++, std::move(maybe_output.second)});
				}
			}
			if (on_output && release_payloads) {
				try {
					// An empty final selection makes every drained handle a loser.
					if (!discard_unselected_outputs) {
						handles[i]->AckPollResult();
					}
					handles[i]->ReleasePollResult();
					handles[i].reset();
				} catch (const std::exception &ex) {
					retain_finished_stream_handles();
					return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
					    DuckDBError::external_error(vane::BoundedErrorDetails::FormatDetail(
					        "failed to finalize streamed FTE result handle[" + std::to_string(i) + "]", ex.what())));
				} catch (...) {
					retain_finished_stream_handles();
					return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
					    DuckDBError::external_error("failed to finalize streamed FTE result handle[" +
					                                std::to_string(i) + "]: unknown error"));
				}
			}
		}
		if (remaining == 0) {
			break;
		}
		if (!had_progress) {
			if (std::chrono::steady_clock::now() >= deadline) {
				StoreFteResultHandles(query_id, std::move(handles));
				return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
				    DuckDBError::external_error("timed out draining FTE result handles"));
			}
			if (PyGILState_Check()) {
				py::gil_scoped_release gil_release;
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			} else {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
		}
	}
	std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> retained_handles;
	std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> retry_handles;
	vane::BoundedErrorDetails release_errors;
	for (size_t idx = 0; idx < handles.size(); idx++) {
		if (!handles[idx]) {
			continue;
		}
		auto &handle = handles[idx];
		bool handle_failed = false;
		// Only selected results were handed downstream and may be acknowledged.
		if (!discard_unselected_outputs) {
			try {
				handle->AckPollResult();
			} catch (const std::exception &ex) {
				handle_failed = true;
				release_errors.Add("ack[" + std::to_string(idx) + "]", ex.what());
			} catch (...) {
				handle_failed = true;
				release_errors.Add("ack[" + std::to_string(idx) + "]", "unknown error");
			}
		}
		if (retain_payload_until_query_cleanup[idx]) {
			if (handle_failed) {
				retry_handles.push_back(std::move(handle));
			} else {
				retained_handles.push_back(std::move(handle));
			}
		} else {
			try {
				handle->ReleasePollResult();
			} catch (const std::exception &ex) {
				handle_failed = true;
				release_errors.Add("release[" + std::to_string(idx) + "]", ex.what());
			} catch (...) {
				handle_failed = true;
				release_errors.Add("release[" + std::to_string(idx) + "]", "unknown error");
			}
			if (handle_failed) {
				retry_handles.push_back(std::move(handle));
			}
		}
	}
	RetainFteResultHandles(query_id, std::move(retained_handles));
	StoreFteResultHandles(query_id, std::move(retry_handles));
	if (release_errors) {
		return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
		    DuckDBError::external_error(release_errors.AppendTo("failed to finalize FTE result handle(s)")));
	}
	std::sort(drained_outputs.begin(), drained_outputs.end(), [](const DrainedOutput &lhs, const DrainedOutput &rhs) {
		if (lhs.task_context.query_idx() != rhs.task_context.query_idx()) {
			return lhs.task_context.query_idx() < rhs.task_context.query_idx();
		}
		if (lhs.task_context.last_node_id() != rhs.task_context.last_node_id()) {
			return lhs.task_context.last_node_id() < rhs.task_context.last_node_id();
		}
		if (lhs.task_context.task_id() != rhs.task_context.task_id()) {
			return lhs.task_context.task_id() < rhs.task_context.task_id();
		}
		return lhs.ordinal < rhs.ordinal;
	});
	outputs.reserve(drained_outputs.size());
	for (auto &entry : drained_outputs) {
		outputs.push_back(std::move(entry.output));
	}
	return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::ok(std::move(outputs));
}

DuckDBResult<void> RayWorkerManager::submit_fte_task_events(std::vector<duckdb::distributed::WorkerTask> tasks) {
	OperationGuard operation(*this);
	if (!operation) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error("Ray worker manager is shut down"));
	}
	string query_id;
	string submission_error_owner;
	try {
		query_id = QueryIdFromTaskEvents(tasks);
		submission_error_owner = SubmissionErrorOwnerQueryId(tasks, query_id);
		if (!tasks.empty() && query_id.empty()) {
			return DuckDBResult<void>::err(DuckDBError::value_error("FTE task events require non-empty query_id"));
		}
		if (tasks.empty()) {
			return DuckDBResult<void>::ok();
		}
		QueryOperationGuard query_operation(*this, query_id, submission_error_owner);
		if (!query_operation) {
			return DuckDBResult<void>::err(DuckDBError::invalid_state_error(
			    "FTE task submission rejected because its resource query is closing: " + submission_error_owner));
		}
		try {
			auto fail_submission = [&](const DuckDBError &error) {
				query_lifecycles_.Close(query_operation.lifecycle());
				return DuckDBResult<void>::err(error);
			};
			auto collect_workers = [&]() {
				std::vector<std::shared_ptr<RayWorkerRuntime>> workers;
				lock_guard<mutex> guard(mutex_);
				if (state_.shutdown_started) {
					throw std::runtime_error("Ray worker manager is shut down");
				}
				workers.reserve(state_.ray_workers.size());
				for (auto &kv : state_.ray_workers) {
					workers.push_back(kv.second);
				}
				return workers;
			};
			std::vector<std::shared_ptr<RayWorkerRuntime>> workers = collect_workers();
			if (workers.empty()) {
				auto snapshots_res = worker_snapshots();
				if (snapshots_res.is_err()) {
					return fail_submission(snapshots_res.error());
				}
				workers = collect_workers();
			}
			if (workers.empty()) {
				return fail_submission(
				    DuckDBError::invalid_state_error("No Ray workers available for FTE task events"));
			}
			RecordQueryWorkers(query_operation.owner_query_id(), workers);

			std::vector<std::vector<duckdb::distributed::WorkerTask>> tasks_per_worker(workers.size());
			for (size_t i = 0; i < tasks.size(); i++) {
				tasks_per_worker[i % workers.size()].push_back(std::move(tasks[i]));
			}

			for (size_t worker_idx = 0; worker_idx < workers.size(); worker_idx++) {
				auto &worker_tasks = tasks_per_worker[worker_idx];
				if (worker_tasks.empty()) {
					continue;
				}
				workers[worker_idx]->SubmitFteTaskEvents(worker_tasks);
			}
			return DuckDBResult<void>::ok();
		} catch (const py::error_already_set &e) {
			// Publish the Python exception before the query operation is allowed
			// to quiesce or shutdown to discard its owner state.
			query_lifecycles_.Close(query_operation.lifecycle());
			submission_errors_.Store(query_operation.owner_query_id(), e);
			return DuckDBResult<void>::err(
			    DuckDBError(string("Python error during submit_fte_task_events: ") + e.what()));
		} catch (const std::exception &e) {
			query_lifecycles_.Close(query_operation.lifecycle());
			return DuckDBResult<void>::err(DuckDBError(string("submit_fte_task_events failed: ") + e.what()));
		}
	} catch (const py::error_already_set &e) {
		return DuckDBResult<void>::err(DuckDBError(string("Python error during submit_fte_task_events: ") + e.what()));
	} catch (const std::exception &e) {
		return DuckDBResult<void>::err(DuckDBError(string("submit_fte_task_events failed: ") + e.what()));
	}
}

void RayWorkerManager::rethrow_submission_error(const string &query_id, const string &details) {
	const auto owner_query_id = query_lifecycles_.OwnerForQuery(query_id);
	auto message = string("distributed worker task submission failed for query_id=") + query_id;
	if (!details.empty()) {
		message += "; execution error: " + details;
	}
	submission_errors_.RethrowAsCause(owner_query_id, std::move(message));
}

DuckDBResult<std::vector<duckdb::distributed::WorkerSnapshot>> RayWorkerManager::worker_snapshots() const {
	// State locks and refresh waits must never retain the GIL. The creator
	// reacquires it only around Python/Ray startup and actor cleanup.
	if (PyGILState_Check()) {
		py::gil_scoped_release release;
		return WorkerSnapshotsWithoutGIL();
	}
	return WorkerSnapshotsWithoutGIL();
}

RayWorkerManager::WorkerSnapshotResult RayWorkerManager::WorkerSnapshotsWithoutGIL() const {
	OperationGuard operation(*this);
	if (!operation) {
		return DuckDBResult<std::vector<duckdb::distributed::WorkerSnapshot>>::err(
		    DuckDBError::invalid_state_error("Ray worker manager is shut down"));
	}
	std::promise<WorkerSnapshotResult> refresh_completion;
	std::shared_ptr<WorkerRefreshFlight> refresh;
	bool refresh_creator = false;
	std::vector<string> existing_ids;
	idx_t refresh_membership_version = 0;
	{
		lock_guard<mutex> guard(mutex_);
		if (state_.shutdown_started) {
			return DuckDBResult<std::vector<duckdb::distributed::WorkerSnapshot>>::err(
			    DuckDBError::invalid_state_error("Ray worker manager is shut down"));
		}
		const bool should_refresh = !state_.last_refresh.first ||
		                            (std::chrono::steady_clock::now() - state_.last_refresh.second) > REFRESH_INTERVAL;
		if (!should_refresh) {
			std::vector<duckdb::distributed::WorkerSnapshot> snapshots;
			snapshots.reserve(state_.ray_workers.size());
			for (auto &kv : state_.ray_workers) {
				snapshots.emplace_back(kv.first, kv.second->TotalNumCpus(), kv.second->TotalNumGpus(),
				                       kv.second->TotalMemoryBytes());
			}
			return DuckDBResult<std::vector<duckdb::distributed::WorkerSnapshot>>::ok(std::move(snapshots));
		}
		if (state_.worker_refresh) {
			refresh = state_.worker_refresh;
		} else {
			refresh_membership_version = state_.worker_membership_version;
			existing_ids.reserve(state_.ray_workers.size());
			for (auto &kv : state_.ray_workers) {
				if (kv.first) {
					existing_ids.push_back(*kv.first);
				}
			}
			auto result = refresh_completion.get_future().share();
			refresh = std::make_shared<WorkerRefreshFlight>(std::move(result));
			state_.worker_refresh = refresh;
			refresh_creator = true;
		}
	}

	if (!refresh_creator) {
		try {
			return refresh->result.get();
		} catch (const std::exception &ex) {
			return WorkerSnapshotResult::err(
			    DuckDBError::internal_error(string("worker refresh synchronization failed: ") + ex.what()));
		} catch (...) {
			return WorkerSnapshotResult::err(
			    DuckDBError::internal_error("worker refresh synchronization failed: unknown exception"));
		}
	}

	WorkerSnapshotResult refresh_result;
	std::vector<std::shared_ptr<RayWorkerRuntime>> new_workers;
	std::vector<std::shared_ptr<std::atomic<bool>>> new_worker_retirement_states;
	bool worker_creation_succeeded = false;
	auto weak_manager = weak_from_this();
	{
		duckdb::PythonGILWrapper gil;
		try {
			py::module_ worker_pool_obj = py::module_::import("vane.runners.ray.worker_pool");
			py::object py_workers_obj = worker_pool_obj.attr("start_ray_workers")(existing_ids, manager_instance_id_);

			py::iterable workers_iter;
			try {
				workers_iter = py_workers_obj.cast<py::iterable>();
			} catch (const py::cast_error &e) {
				throw std::runtime_error(string("start_ray_workers must return an iterable of RayWorkerRuntime: ") +
				                         e.what());
			}

			string worker_validation_error;
			for (auto item : workers_iter) {
				std::shared_ptr<RayWorkerRuntime> worker;
				try {
					worker = item.cast<std::shared_ptr<RayWorkerRuntime>>();
				} catch (const py::cast_error &e) {
					if (worker_validation_error.empty()) {
						worker_validation_error = e.what();
					}
					continue;
				}
				if (!worker) {
					if (worker_validation_error.empty()) {
						worker_validation_error = "start_ray_workers returned null RayWorkerRuntime";
					}
					continue;
				}
				auto worker_id = worker->Id();
				new_workers.push_back(std::move(worker));
				if (!worker_id || worker_id->empty()) {
					if (worker_validation_error.empty()) {
						worker_validation_error = "start_ray_workers returned worker without id";
					}
					continue;
				}
			}
			if (!worker_validation_error.empty()) {
				throw std::runtime_error(std::move(worker_validation_error));
			}
			for (auto &worker : new_workers) {
				auto worker_id = *worker->Id();
				auto weak_worker = std::weak_ptr<RayWorkerRuntime>(worker);
				auto retired = std::make_shared<std::atomic<bool>>(false);
				auto retire_callback = py::cpp_function([weak_manager, weak_worker, worker_id, retired]() {
					auto manager = weak_manager.lock();
					auto worker = weak_worker.lock();
					if (!manager || !worker) {
						retired->store(true);
						return true;
					}
					return manager->RetireWorkerForFailure(worker_id, worker, retired);
				});
				if (!worker->InstallFailureRetirementCallback(std::move(retire_callback))) {
					retired->store(true);
				}
				new_worker_retirement_states.push_back(std::move(retired));
			}
			worker_creation_succeeded = true;
		} catch (const py::error_already_set &e) {
			refresh_result = WorkerSnapshotResult::err(
			    DuckDBError::external_error(string("refresh_workers python error: ") + e.what()));
		} catch (const std::exception &e) {
			refresh_result = WorkerSnapshotResult::err(
			    DuckDBError::external_error(string("refresh_workers exception: ") + e.what()));
		} catch (...) {
			refresh_result =
			    WorkerSnapshotResult::err(DuckDBError::external_error("refresh_workers unknown exception"));
		}
	}

	if (worker_creation_succeeded) {
		try {
			std::unordered_set<string> worker_ids;
			struct NewWorkerEntry {
				WorkerId id;
				std::shared_ptr<RayWorkerRuntime> worker;
				std::shared_ptr<std::atomic<bool>> retired;
			};
			std::vector<NewWorkerEntry> new_entries;
			new_entries.reserve(new_workers.size());
			for (idx_t worker_idx = 0; worker_idx < new_workers.size(); worker_idx++) {
				auto &worker = new_workers[worker_idx];
				const auto &worker_id = *worker->Id();
				if (!worker_ids.insert(worker_id).second) {
					throw std::runtime_error("start_ray_workers returned duplicate worker id: " + worker_id);
				}
				new_entries.push_back(
				    {duckdb::distributed::make_worker_id(worker_id), worker, new_worker_retirement_states[worker_idx]});
			}

			{
				lock_guard<mutex> guard(mutex_);
				if (state_.shutdown_started) {
					refresh_result = WorkerSnapshotResult::err(
					    DuckDBError::invalid_state_error("Ray worker manager shut down during worker refresh"));
				} else {
					const bool membership_changed = state_.worker_membership_version != refresh_membership_version;
					auto updated_workers = state_.ray_workers;
					idx_t inserted_workers = 0;
					bool skipped_retired_worker = false;
					for (auto &entry : new_entries) {
						if (entry.retired->load()) {
							skipped_retired_worker = true;
							continue;
						}
						if (updated_workers.find(entry.id) != updated_workers.end()) {
							throw std::runtime_error("start_ray_workers returned existing worker id: " + *entry.id);
						}
						auto inserted = updated_workers.emplace(entry.id, entry.worker);
						if (!inserted.second) {
							throw std::runtime_error("failed to stage worker id: " + *entry.id);
						}
						inserted_workers++;
					}

					std::vector<duckdb::distributed::WorkerSnapshot> snapshots;
					snapshots.reserve(updated_workers.size());
					for (auto &kv : updated_workers) {
						snapshots.emplace_back(kv.first, kv.second->TotalNumCpus(), kv.second->TotalNumGpus(),
						                       kv.second->TotalMemoryBytes());
					}
					state_.ray_workers.swap(updated_workers);
					if (inserted_workers > 0) {
						state_.worker_membership_version++;
					}
					state_.last_refresh = membership_changed || skipped_retired_worker
					                          ? std::pair<bool, std::chrono::steady_clock::time_point> {}
					                          : std::make_pair(true, std::chrono::steady_clock::now());
					refresh_result = WorkerSnapshotResult::ok(std::move(snapshots));
				}
			}
		} catch (const std::exception &ex) {
			refresh_result = WorkerSnapshotResult::err(
			    DuckDBError::external_error(string("refresh_workers commit failed: ") + ex.what()));
		} catch (...) {
			refresh_result = WorkerSnapshotResult::err(
			    DuckDBError::external_error("refresh_workers commit failed: unknown exception"));
		}
	}

	if (refresh_result.is_err() && !new_workers.empty()) {
		try {
			auto cleanup_errors = AbortWorkers(new_workers);
			if (!cleanup_errors.empty()) {
				string message = refresh_result.error().what();
				for (auto &error : cleanup_errors) {
					message += "; worker refresh cleanup failed: " + error;
				}
				refresh_result = WorkerSnapshotResult::err(DuckDBError::external_error(std::move(message)));
			}
		} catch (const std::exception &ex) {
			refresh_result = WorkerSnapshotResult::err(DuckDBError::external_error(
			    string(refresh_result.error().what()) + "; worker refresh cleanup failed: " + ex.what()));
		} catch (...) {
			refresh_result = WorkerSnapshotResult::err(DuckDBError::external_error(
			    string(refresh_result.error().what()) + "; worker refresh cleanup failed: unknown exception"));
		}
	}

	try {
		refresh_completion.set_value(refresh_result);
	} catch (...) {
		// The creator owns the only promise. Its destructor makes the shared
		// future ready with broken_promise if publication unexpectedly fails.
	}
	{
		lock_guard<mutex> guard(mutex_);
		if (state_.worker_refresh == refresh) {
			state_.worker_refresh.reset();
		}
	}
	return refresh_result;
}

DuckDBResult<void> RayWorkerManager::shutdown() {
	std::vector<std::shared_ptr<RayWorkerRuntime>> workers;
	{
		std::unique_lock<mutex> guard(mutex_);
		if (state_.shutdown_started) {
			shutdown_cv_.wait(guard, [&]() { return state_.shutdown_finished; });
			// The caller that performed shutdown already received the aggregated
			// error. Every worker was either finished or force-terminated and all
			// manager-owned state was released, so a retry observes the completed
			// terminal state instead of replaying an unrecoverable error forever.
			return DuckDBResult<void>::ok();
		}
		state_.shutdown_started = true;
	}
	if (Py_IsInitialized() && !duckdb::PythonIsFinalizing() && PyGILState_Check()) {
		py::gil_scoped_release release;
		query_lifecycles_.BeginShutdown();
	} else {
		query_lifecycles_.BeginShutdown();
	}
	{
		lock_guard<mutex> guard(mutex_);
		workers.reserve(state_.ray_workers.size());
		for (auto &kv : state_.ray_workers) {
			workers.push_back(kv.second);
		}
	}
	std::vector<std::string> errors;
	std::vector<std::string> prepare_errors;
	for (auto &worker : workers) {
		const auto worker_id = worker->Id() ? *worker->Id() : std::string("<unknown>");
		try {
			worker->PrepareShutdown();
		} catch (const std::exception &ex) {
			prepare_errors.push_back(worker_id + ": " + ex.what());
		} catch (...) {
			prepare_errors.push_back(worker_id + ": unknown prepare-shutdown error");
		}
	}
	errors.insert(errors.end(), prepare_errors.begin(), prepare_errors.end());
	// Flight shutdown waits for in-flight RPCs. Stop services only after every
	// worker has canceled and joined native work, so cross-worker readers cannot
	// deadlock a server that is being stopped earlier in this loop.
	if (prepare_errors.empty()) {
		for (auto &worker : workers) {
			const auto worker_id = worker->Id() ? *worker->Id() : std::string("<unknown>");
			try {
				worker->FinishShutdown();
			} catch (const std::exception &ex) {
				errors.push_back(worker_id + ": " + ex.what());
			} catch (...) {
				errors.push_back(worker_id + ": unknown finish-shutdown error");
			}
		}
	} else {
		for (auto &worker : workers) {
			const auto worker_id = worker->Id() ? *worker->Id() : std::string("<unknown>");
			try {
				worker->AbortShutdown();
			} catch (const std::exception &ex) {
				errors.push_back(worker_id + " force termination: " + ex.what());
			} catch (...) {
				errors.push_back(worker_id + ": unknown force-termination error");
			}
		}
	}
	decltype(state_.fte_result_handles_by_query) result_handles;
	decltype(state_.retained_fte_result_handles_by_query) retained_result_handles;
	decltype(state_.workers_by_query_owner) query_workers;
	{
		std::unique_lock<mutex> guard(mutex_);
		shutdown_cv_.wait(guard, [&]() { return state_.active_operations == 0; });
		state_.ray_workers.clear();
		result_handles = std::move(state_.fte_result_handles_by_query);
		retained_result_handles = std::move(state_.retained_fte_result_handles_by_query);
		state_.fte_result_handle_counts_by_query.clear();
		query_workers = std::move(state_.workers_by_query_owner);
		state_.last_refresh = {};
	}
	if (Py_IsInitialized() && !duckdb::PythonIsFinalizing() && PyGILState_Check()) {
		py::gil_scoped_release release;
		query_lifecycles_.WaitForAllOperations();
	} else {
		query_lifecycles_.WaitForAllOperations();
	}
	const auto owner_query_ids = query_lifecycles_.OwnerQueryIds();
	result_handles.clear();
	retained_result_handles.clear();
	query_workers.clear();
	for (const auto &owner_query_id : owner_query_ids) {
		if (query_cleanup_) {
			try {
				query_cleanup_(owner_query_id);
			} catch (const std::exception &ex) {
				errors.push_back("query owner " + owner_query_id + " cleanup: " + ex.what());
			} catch (...) {
				errors.push_back("query owner " + owner_query_id + " cleanup: unknown error");
			}
		}
		submission_errors_.Discard(owner_query_id);
	}
	try {
		query_lifecycles_.FinishShutdown(true);
	} catch (const std::exception &ex) {
		errors.push_back(string("query lifecycle shutdown: ") + ex.what());
		try {
			query_lifecycles_.FinishShutdown(false);
		} catch (const std::exception &finish_ex) {
			errors.push_back(string("query lifecycle shutdown recovery: ") + finish_ex.what());
		} catch (...) {
			errors.push_back("query lifecycle shutdown recovery: unknown error");
		}
	} catch (...) {
		errors.push_back("query lifecycle shutdown: unknown error");
		try {
			query_lifecycles_.FinishShutdown(false);
		} catch (const std::exception &finish_ex) {
			errors.push_back(string("query lifecycle shutdown recovery: ") + finish_ex.what());
		} catch (...) {
			errors.push_back("query lifecycle shutdown recovery: unknown error");
		}
	}
	std::string error_message;
	if (!errors.empty()) {
		error_message = "Ray worker shutdown failed with " + std::to_string(errors.size()) + " error(s)";
		for (const auto &error : errors) {
			error_message += "; " + error;
		}
	}
	{
		lock_guard<mutex> guard(mutex_);
		state_.shutdown_finished = true;
	}
	shutdown_cv_.notify_all();
	if (!error_message.empty()) {
		return DuckDBResult<void>::err(DuckDBError::external_error(std::move(error_message)));
	}
	return DuckDBResult<void>::ok();
}

DuckDBResult<void> RayWorkerManager::close_session(const string &session_id) {
	if (session_id.empty()) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error("Ray worker session_id is empty"));
	}
	OperationGuard operation(*this);
	if (!operation) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error("Ray worker manager is shut down"));
	}
	std::vector<std::shared_ptr<RayWorkerRuntime>> workers;
	{
		lock_guard<mutex> guard(mutex_);
		workers.reserve(state_.ray_workers.size());
		for (auto &entry : state_.ray_workers) {
			workers.push_back(entry.second);
		}
	}
	std::vector<std::string> errors;
	for (auto &worker : workers) {
		const auto worker_id = worker->Id() ? *worker->Id() : std::string("<unknown>");
		try {
			worker->CloseSession(session_id);
		} catch (const std::exception &ex) {
			errors.push_back(worker_id + ": " + ex.what());
		} catch (...) {
			errors.push_back(worker_id + ": unknown close-session error");
		}
	}
	if (!errors.empty()) {
		std::string message =
		    "Failed to close Ray worker session " + session_id + " with " + std::to_string(errors.size()) + " error(s)";
		for (const auto &error : errors) {
			message += "; " + error;
		}
		return DuckDBResult<void>::err(DuckDBError::external_error(std::move(message)));
	}
	return DuckDBResult<void>::ok();
}

DuckDBResult<void> RayWorkerManager::ExecuteQueryAbort(std::optional<QueryLifecycleCoordinator::Abort> active_abort) {
	if (!active_abort) {
		return DuckDBResult<void>::ok();
	}

	std::optional<string> failure;
	try {
		std::vector<string> errors;
		auto prepare_workers = [&]() {
			const auto workers = QueryWorkers(active_abort->lifecycle.owner_query_id);
			for (const auto &execution_query_id : active_abort->execution_query_ids) {
				for (const auto &worker : workers) {
					const auto worker_id = worker && worker->Id() ? *worker->Id() : string("<unknown>");
					try {
						worker->PrepareDropQuery(execution_query_id);
					} catch (const std::exception &ex) {
						errors.push_back(execution_query_id + "@" + worker_id + ": " + ex.what());
					} catch (...) {
						errors.push_back(execution_query_id + "@" + worker_id + ": unknown abort error");
					}
				}
			}
		};
		prepare_workers();
		if (errors.empty()) {
			WaitForQueryOperations(active_abort->lifecycle);
			if (active_abort->had_active_operations) {
				prepare_workers();
			}
		}
		if (!errors.empty()) {
			failure = "resource query abort barrier failed with " + std::to_string(errors.size()) + " error(s)";
			for (const auto &error : errors) {
				*failure += "; " + error;
			}
		}
	} catch (const std::exception &ex) {
		failure = string("resource query abort orchestration failed: ") + ex.what();
	} catch (...) {
		failure = "resource query abort orchestration failed: unknown error";
	}
	try {
		query_lifecycles_.CompleteAbort(*active_abort, failure);
	} catch (const std::exception &ex) {
		if (failure) {
			*failure += "; lifecycle completion: " + string(ex.what());
		} else {
			failure = string("resource query abort lifecycle completion failed: ") + ex.what();
		}
	} catch (...) {
		if (failure) {
			*failure += "; lifecycle completion: unknown error";
		} else {
			failure = "resource query abort lifecycle completion failed: unknown error";
		}
	}
	if (failure) {
		return DuckDBResult<void>::err(DuckDBError::external_error(std::move(*failure)));
	}
	return DuckDBResult<void>::ok();
}

DuckDBResult<void> RayWorkerManager::abort_and_quiesce_query(const string &query_id) {
	if (query_id.empty()) {
		return DuckDBResult<void>::err(DuckDBError::value_error("FTE query abort requires non-empty query_id"));
	}
	OperationGuard operation(*this);
	if (!operation) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error("Ray worker manager is shut down"));
	}
	try {
		return ExecuteQueryAbort(BeginQueryAbort(query_id));
	} catch (const std::exception &ex) {
		return DuckDBResult<void>::err(DuckDBError::external_error(ex.what()));
	} catch (...) {
		return DuckDBResult<void>::err(
		    DuckDBError::external_error("unknown error while starting resource query abort"));
	}
}

void RayWorkerManager::drop_query_fragments(const string &query_id) {
	if (query_id.empty()) {
		return;
	}
	OperationGuard operation(*this);
	if (!operation) {
		throw std::runtime_error("Ray worker manager is shut down");
	}
	auto teardown = BeginQueryTeardown(query_id);
	if (!teardown) {
		return;
	}
	try {
		auto abort_res = ExecuteQueryAbort(BeginQueryAbort(*teardown));
		if (abort_res.is_err()) {
			throw std::runtime_error(abort_res.error().what());
		}
		query_lifecycles_.MarkDropping(*teardown);
		const auto workers = QueryWorkers(teardown->lifecycle.owner_query_id);
		std::vector<string> errors;
		for (const auto &execution_query_id : teardown->execution_query_ids) {
			for (const auto &worker : workers) {
				const auto worker_id = worker && worker->Id() ? *worker->Id() : string("<unknown>");
				try {
					worker->CleanupQuery(execution_query_id);
				} catch (const std::exception &ex) {
					errors.push_back("worker storage " + execution_query_id + "@" + worker_id + ": " + ex.what());
				} catch (...) {
					errors.push_back("worker storage " + execution_query_id + "@" + worker_id +
					                 ": unknown cleanup error");
				}
			}
		}
		if (!errors.empty()) {
			string message = "resource query final teardown failed with " + std::to_string(errors.size()) + " error(s)";
			for (const auto &error : errors) {
				message += "; " + error;
			}
			throw std::runtime_error(std::move(message));
		}

		for (const auto &execution_query_id : teardown->execution_query_ids) {
			try {
				ClearFteResultHandles(execution_query_id);
			} catch (const std::exception &ex) {
				errors.push_back("result handles " + execution_query_id + ": " + ex.what());
			} catch (...) {
				errors.push_back("result handles " + execution_query_id + ": unknown cleanup error");
			}
		}
		if (!errors.empty()) {
			string message = "resource query result cleanup failed with " + std::to_string(errors.size()) + " error(s)";
			for (const auto &error : errors) {
				message += "; " + error;
			}
			throw std::runtime_error(std::move(message));
		}
		{
			lock_guard<mutex> guard(mutex_);
			for (const auto &execution_query_id : teardown->execution_query_ids) {
				state_.fte_result_handle_counts_by_query.erase(execution_query_id);
			}
		}
		if (query_cleanup_) {
			query_cleanup_(teardown->lifecycle.owner_query_id);
		}
		submission_errors_.Discard(teardown->lifecycle.owner_query_id);
		{
			lock_guard<mutex> guard(mutex_);
			state_.workers_by_query_owner.erase(teardown->lifecycle.owner_query_id);
		}
		query_lifecycles_.CompleteTeardown(*teardown, std::nullopt);
	} catch (...) {
		auto failure = ExceptionMessage(std::current_exception());
		try {
			query_lifecycles_.CompleteTeardown(*teardown, failure);
		} catch (const std::exception &ex) {
			failure += "; lifecycle completion: " + string(ex.what());
		} catch (...) {
			failure += "; lifecycle completion: unknown error";
		}
		throw std::runtime_error(std::move(failure));
	}
}

DuckDBResult<void> RayWorkerManager::task_input_stream_exhausted_for_query(
    const string &query_id, const std::unordered_set<duckdb::distributed::SourceNodeId> &source_node_ids) {
	if (query_id.empty()) {
		return DuckDBResult<void>::err(
		    DuckDBError::value_error("FTE task input exhaustion requires non-empty query_id"));
	}
	OperationGuard operation(*this);
	if (!operation) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error("Ray worker manager is shut down"));
	}
	QueryOperationGuard query_operation(*this, query_id);
	if (!query_operation) {
		return DuckDBResult<void>::err(
		    DuckDBError::invalid_state_error("FTE query input stream is closing: " + query_id));
	}

	std::vector<std::shared_ptr<RayWorkerRuntime>> workers;
	{
		lock_guard<mutex> guard(mutex_);
		workers.reserve(state_.ray_workers.size());
		for (auto &kv : state_.ray_workers) {
			workers.push_back(kv.second);
		}
	}
	try {
		for (auto &worker : workers) {
			worker->TaskInputStreamExhaustedForQuery(query_id, source_node_ids);
		}
	} catch (const std::exception &e) {
		return DuckDBResult<void>::err(
		    DuckDBError(string("Python error during task_input_stream_exhausted_for_query: ") + e.what()));
	}
	return DuckDBResult<void>::ok();
}

DuckDBResult<void> RayWorkerManager::materialization_barrier_completed(const string &query_id,
                                                                       duckdb::distributed::NodeID node_id) {
	if (query_id.empty()) {
		return DuckDBResult<void>::err(
		    DuckDBError::value_error("materialization barrier completion requires non-empty query_id"));
	}
	OperationGuard operation(*this);
	if (!operation) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error("Ray worker manager is shut down"));
	}
	QueryOperationGuard query_operation(*this, query_id);
	if (!query_operation) {
		return DuckDBResult<void>::err(
		    DuckDBError::invalid_state_error("FTE query materialization barrier is closing: " + query_id));
	}

	try {
		duckdb::PythonGILWrapper gil;
		py::module_ resource_runtime = py::module_::import("vane.runners.ray.query_resource_runtime");
		resource_runtime.attr("mark_materialization_barrier_completed")(query_id, std::to_string(node_id));
	} catch (const std::exception &e) {
		return DuckDBResult<void>::err(
		    DuckDBError(string("Python error during materialization_barrier_completed: ") + e.what()));
	}
	return DuckDBResult<void>::ok();
}

DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>
RayWorkerManager::wait_fte_query(const string &query_id, double timeout_s) {
	return wait_fte_query(query_id, timeout_s, {});
}

DuckDBResult<RayWorkerRuntime::QueryStatus> RayWorkerManager::FteQueryStatus(
    const string &query_id,
    const std::unordered_set<duckdb::distributed::TaskContext, duckdb::distributed::TaskContextHash>
        *task_context_filter) {
	if (query_id.empty()) {
		return DuckDBResult<RayWorkerRuntime::QueryStatus>::err(DuckDBError::value_error("query_id must be non-empty"));
	}
	std::vector<std::shared_ptr<RayWorkerRuntime>> workers;
	{
		lock_guard<mutex> guard(mutex_);
		workers.reserve(state_.ray_workers.size());
		for (auto &kv : state_.ray_workers) {
			workers.push_back(kv.second);
		}
	}
	try {
		for (auto &worker : workers) {
			auto status = worker->FteQueryStatus(query_id, task_context_filter);
			return DuckDBResult<RayWorkerRuntime::QueryStatus>::ok(std::move(status));
		}
	} catch (const std::exception &e) {
		return DuckDBResult<RayWorkerRuntime::QueryStatus>::err(
		    DuckDBError(string("Python error during fte_query_status: ") + e.what()));
	}
	return DuckDBResult<RayWorkerRuntime::QueryStatus>::err(
	    DuckDBError::invalid_state_error("No Ray workers available for fte_query_status"));
}

DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>
RayWorkerManager::wait_fte_query(const string &query_id, double timeout_s,
                                 duckdb::distributed::MaterializedOutputCallback on_output) {
	const std::unordered_set<duckdb::distributed::TaskContext, duckdb::distributed::TaskContextHash> empty_contexts;
	return wait_fte_query(query_id, timeout_s, empty_contexts, std::move(on_output));
}

DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>> RayWorkerManager::wait_fte_query(
    const string &query_id, double timeout_s,
    const std::unordered_set<duckdb::distributed::TaskContext, duckdb::distributed::TaskContextHash> &task_contexts,
    duckdb::distributed::MaterializedOutputCallback on_output) {
	return WaitFteQuery(query_id, timeout_s, task_contexts, std::move(on_output), false);
}

DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>
RayWorkerManager::wait_fte_query_streaming(const string &query_id, double timeout_s,
                                           duckdb::distributed::MaterializedOutputCallback on_output) {
	const std::unordered_set<duckdb::distributed::TaskContext, duckdb::distributed::TaskContextHash> empty_contexts;
	return WaitFteQuery(query_id, timeout_s, empty_contexts, std::move(on_output), true);
}

DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>> RayWorkerManager::WaitFteQuery(
    const string &query_id, double timeout_s,
    const std::unordered_set<duckdb::distributed::TaskContext, duckdb::distributed::TaskContextHash> &task_contexts,
    duckdb::distributed::MaterializedOutputCallback on_output, bool stream_outputs) {
	if (query_id.empty()) {
		return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
		    DuckDBError::value_error("query_id must be non-empty"));
	}
	if (stream_outputs && !on_output) {
		return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
		    DuckDBError::value_error("streaming FTE result drain requires an output callback"));
	}
	OperationGuard operation(*this);
	if (!operation) {
		return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
		    DuckDBError::invalid_state_error("Ray worker manager is shut down"));
	}
	QueryOperationGuard query_operation(*this, query_id);
	if (!query_operation) {
		return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
		    DuckDBError::invalid_state_error("FTE query is closing: " + query_id));
	}

	std::vector<duckdb::distributed::MaterializedOutput> outputs;
	RayWorkerRuntime::QueryStatus finished_status;
	bool has_finished_status = false;
	const bool has_deadline = timeout_s > 0.0;
	const auto deadline = has_deadline ? std::chrono::steady_clock::now() +
	                                         std::chrono::duration_cast<std::chrono::steady_clock::duration>(
	                                             std::chrono::duration<double>(timeout_s))
	                                   : std::chrono::steady_clock::time_point::max();
	auto fail_after_result_cleanup = [&](const string &stage, const char *detail) {
		vane::BoundedErrorDetails errors;
		errors.Add(stage, detail);
		try {
			ClearFteResultHandles(query_id);
		} catch (const std::exception &cleanup_error) {
			errors.Add("FTE result cleanup", cleanup_error.what());
		} catch (...) {
			errors.Add("FTE result cleanup", "unknown error");
		}
		return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
		    DuckDBError(errors.AppendTo("FTE query wait failed")));
	};

	try {
		while (true) {
			if (ShutdownStarted()) {
				return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
				    DuckDBError::invalid_state_error("Ray worker manager is shutting down"));
			}
			const auto *task_context_filter = task_contexts.empty() ? nullptr : &task_contexts;
			auto status_res = FteQueryStatus(query_id, task_context_filter);
			if (status_res.is_err()) {
				return fail_after_result_cleanup("FTE query status", status_res.error().what());
			}
			const auto &status = status_res.value();
			if (status.failed) {
				return fail_after_result_cleanup("FTE query failed", status.message.c_str());
			}
			if (status.canceled) {
				return fail_after_result_cleanup("FTE query canceled", status.message.c_str());
			}
			auto collect_res = CollectFteResultHandles(query_id);
			if (collect_res.is_err()) {
				return fail_after_result_cleanup("FTE result handle collection", collect_res.error().what());
			}
			if (stream_outputs && !status.selected_attempt_task_ids.empty()) {
				auto coverage_res = ValidateFteResultHandleCoverage(query_id, status.selected_attempt_task_ids);
				if (coverage_res.is_err()) {
					return fail_after_result_cleanup("FTE selected-attempt/result-handle validation",
					                                 coverage_res.error().what());
				}
				const double remaining_timeout_s =
				    has_deadline
				        ? std::max(0.0,
				                   std::chrono::duration<double>(deadline - std::chrono::steady_clock::now()).count())
				        : -1.0;
				const auto *task_context_filter = task_contexts.empty() ? nullptr : &task_contexts;
				auto stream_res = DrainFteResultHandles(query_id, remaining_timeout_s, &status, task_context_filter,
				                                        true, on_output, true);
				if (stream_res.is_err()) {
					return stream_res;
				}
			}
			// Registry operations fence every ingress path that can publish a
			// fragment. Once no such operation remains, an unmatched materializer
			// scope cannot appear later and must not poll indefinitely.
			if (task_context_filter && !status.matched) {
				if (!status.registration_pending) {
					return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
					    DuckDBError::external_error(vane::BoundedErrorDetails::FormatDetail(
					        "FTE query scope did not match any registered fragment", status.message.c_str())));
				}
			} else if (status.finished) {
				finished_status = status;
				has_finished_status = true;
				break;
			}
			if (has_deadline && std::chrono::steady_clock::now() >= deadline) {
				return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
				    DuckDBError::external_error(vane::BoundedErrorDetails::FormatDetail(
				        "timed out waiting for FTE query", status.message.c_str())));
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}

		auto collect_res = CollectFteResultHandles(query_id);
		if (collect_res.is_err()) {
			return fail_after_result_cleanup("FTE result handle collection", collect_res.error().what());
		}
		auto coverage_res = ValidateFteResultHandleCoverage(query_id, finished_status.selected_attempt_task_ids);
		if (coverage_res.is_err()) {
			return fail_after_result_cleanup("FTE selected-attempt/result-handle validation",
			                                 coverage_res.error().what());
		}
		const double remaining_timeout_s =
		    has_deadline
		        ? std::max(0.0, std::chrono::duration<double>(deadline - std::chrono::steady_clock::now()).count())
		        : -1.0;
		const auto *task_context_filter = task_contexts.empty() ? nullptr : &task_contexts;
		auto drain_res = DrainFteResultHandles(
		    query_id, remaining_timeout_s, has_finished_status ? &finished_status : nullptr, task_context_filter,
		    stream_outputs, stream_outputs ? on_output : duckdb::distributed::MaterializedOutputCallback {}, false);
		if (drain_res.is_err()) {
			return drain_res;
		}
		for (auto &output : drain_res.value()) {
			if (!stream_outputs && on_output) {
				auto callback_res = on_output(output);
				if (callback_res.is_err()) {
					return fail_after_result_cleanup("FTE output callback", callback_res.error().what());
				}
			}
			outputs.push_back(std::move(output));
		}
		return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::ok(std::move(outputs));
	} catch (const std::exception &e) {
		return fail_after_result_cleanup("Python error during wait_fte_query", e.what());
	} catch (...) {
		return fail_after_result_cleanup("wait_fte_query", "unknown error");
	}
}

std::unordered_map<std::string, std::unordered_map<std::string, duckdb::idx_t>>
RayWorkerManager::fragment_stats_by_worker() const {
	OperationGuard operation(*this);
	if (!operation) {
		throw std::runtime_error("Ray worker manager is shut down");
	}
	std::vector<std::pair<std::string, std::shared_ptr<RayWorkerRuntime>>> workers;
	{
		lock_guard<mutex> guard(mutex_);
		workers.reserve(state_.ray_workers.size());
		for (const auto &kv : state_.ray_workers) {
			if (!kv.first || kv.first->empty()) {
				continue;
			}
			workers.emplace_back(*kv.first, kv.second);
		}
	}

	std::unordered_map<std::string, std::unordered_map<std::string, duckdb::idx_t>> out;
	for (const auto &entry : workers) {
		out.emplace(entry.first, entry.second->FragmentStats());
	}
	return out;
}

DuckDBResult<void> RayWorkerManager::try_autoscale(const std::vector<TaskResourceRequest> &bundles) {
	OperationGuard operation(*this);
	if (!operation) {
		return DuckDBResult<void>::err(DuckDBError::invalid_state_error("Ray worker manager is shut down"));
	}
	try {
		double req_cpus = 0, req_gpus = 0;
		size_t req_mem = 0;
		for (auto &b : bundles) {
			req_cpus += b.resource_request().num_cpus();
			req_gpus += b.resource_request().num_gpus();
			req_mem += b.resource_request().memory_bytes();
		}

		double cluster_cpus = 0, cluster_gpus = 0;
		size_t cluster_mem = 0;
		{
			lock_guard<mutex> guard(mutex_);
			if (state_.shutdown_started) {
				return DuckDBResult<void>::err(DuckDBError::invalid_state_error("Ray worker manager is shut down"));
			}
			for (auto &kv : state_.ray_workers) {
				cluster_cpus += kv.second->TotalNumCpus();
				cluster_gpus += kv.second->TotalNumGpus();
				cluster_mem += kv.second->TotalMemoryBytes();
			}
		}

		bool need_more = req_cpus > cluster_cpus || req_gpus > cluster_gpus || req_mem > cluster_mem;
		if (!need_more) {
			return DuckDBResult<void>::ok();
		}

		duckdb::PythonGILWrapper gil;
		py::module_ worker_pool = py::module_::import("vane.runners.ray.worker_pool");
		py::list python_bundles;
		for (auto &b : bundles) {
			py::dict d;
			d["CPU"] = (int64_t)std::ceil(b.num_cpus());
			d["GPU"] = (int64_t)std::ceil(b.num_gpus());
			d["memory"] = (int64_t)b.memory_bytes();
			python_bundles.append(d);
		}
		worker_pool.attr("try_autoscale")(python_bundles);
		return DuckDBResult<void>::ok();
	} catch (const std::exception &e) {
		return DuckDBResult<void>::err(DuckDBError::external_error(string("try_autoscale failed: ") + e.what()));
	} catch (...) {
		return DuckDBResult<void>::err(DuckDBError::external_error("try_autoscale failed: unknown exception"));
	}
}
