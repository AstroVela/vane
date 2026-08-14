// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "worker_manager.hpp"
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

RayWorkerManager::RayWorkerManager()
    : manager_instance_id_(duckdb::UUID::ToString(duckdb::UUID::GenerateRandomUUID())) {
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

static bool IsUnselectedFteHandle(const RayWorkerRuntime::TaskResultHandleType &handle,
                                  const RayWorkerRuntime::QueryStatus *finished_status) {
	if (!finished_status || finished_status->selected_attempt_task_ids.empty()) {
		return false;
	}
	const auto &fte_task_id = handle.GetFteTaskId();
	if (fte_task_id.empty()) {
		return false;
	}
	return finished_status->selected_attempt_task_ids.find(fte_task_id) ==
	       finished_status->selected_attempt_task_ids.end();
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
	lock_guard<mutex> guard(mutex_);
	auto &stored = state_.fte_result_handles_by_query[query_id];
	stored.reserve(stored.size() + handles.size());
	for (auto &handle : handles) {
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
	std::vector<std::string> errors;
	std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> retry_handles;
	auto release_all = [&](auto &owned_handles, const char *kind) {
		for (size_t index = 0; index < owned_handles.size(); index++) {
			try {
				owned_handles[index]->ReleasePollResult();
			} catch (const std::exception &ex) {
				errors.push_back(std::string(kind) + "[" + std::to_string(index) + "]: " + ex.what());
				retry_handles.push_back(std::move(owned_handles[index]));
			} catch (...) {
				errors.push_back(std::string(kind) + "[" + std::to_string(index) + "]: unknown release error");
				retry_handles.push_back(std::move(owned_handles[index]));
			}
		}
	};
	release_all(handles, "pending");
	release_all(retained_handles, "retained");
	StoreFteResultHandles(query_id, std::move(retry_handles));
	if (!errors.empty()) {
		std::string message = "failed to release " + std::to_string(errors.size()) + " FTE result handle(s)";
		for (const auto &error : errors) {
			message += "; " + error;
		}
		throw std::runtime_error(message);
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
			StoreFteResultHandles(query_id, std::move(wrapped));
		}
	} catch (const std::exception &e) {
		return DuckDBResult<void>::err(
		    DuckDBError(string("Python error while collecting FTE result handles: ") + e.what()));
	}
	return DuckDBResult<void>::ok();
}

DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>> RayWorkerManager::DrainFteResultHandles(
    const string &query_id, double timeout_s, const RayWorkerRuntime::QueryStatus *finished_status,
    const std::unordered_set<duckdb::distributed::TaskContext, duckdb::distributed::TaskContextHash>
        *task_context_filter,
    bool release_payloads) {
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
	std::vector<bool> retain_payload_until_query_cleanup(handles.size(), false);
	bool has_duplicate_task_context = false;
	if (finished_status && !finished_status->selected_attempt_task_ids.empty()) {
		std::unordered_set<duckdb::distributed::TaskContext, duckdb::distributed::TaskContextHash> seen_contexts;
		for (auto &handle : handles) {
			if (!seen_contexts.insert(handle->GetTaskContext()).second) {
				has_duplicate_task_context = true;
				break;
			}
		}
	}
	if (has_duplicate_task_context && finished_status && !finished_status->selected_attempt_task_ids.empty()) {
		std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> selected_handles;
		std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> retry_handles;
		std::vector<std::string> release_errors;
		selected_handles.reserve(handles.size());
		for (size_t index = 0; index < handles.size(); index++) {
			auto &handle = handles[index];
			if (IsUnselectedFteHandle(*handle, finished_status)) {
				try {
					handle->AckPollResult();
					handle->ReleasePollResult();
				} catch (const std::exception &ex) {
					release_errors.push_back("unselected[" + std::to_string(index) + "]: " + ex.what());
					retry_handles.push_back(std::move(handle));
				} catch (...) {
					release_errors.push_back("unselected[" + std::to_string(index) + "]: unknown release error");
					retry_handles.push_back(std::move(handle));
				}
				continue;
			}
			selected_handles.push_back(std::move(handle));
		}
		handles = std::move(selected_handles);
		StoreFteResultHandles(query_id, std::move(retry_handles));
		if (!release_errors.empty()) {
			StoreFteResultHandles(query_id, std::move(handles));
			std::string message = "failed to release unselected FTE result handle(s)";
			for (const auto &error : release_errors) {
				message += "; " + error;
			}
			return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
			    DuckDBError::external_error(message));
		}
		if (handles.empty()) {
			return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::ok(std::move(outputs));
		}
	}

	std::vector<bool> finished(handles.size(), false);
	size_t remaining = handles.size();
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
				continue;
			}

			finished[i] = true;
			remaining--;
			had_progress = true;
			auto task_context = handles[i]->GetTaskContext();
			auto task_result = std::move(poll_res.second);
			if (task_result.is_err()) {
				if (IsUnselectedFteHandle(*handles[i], finished_status)) {
					continue;
				}
				auto error = task_result.error();
				StoreFteResultHandles(query_id, std::move(handles));
				return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(error);
			}

			auto maybe_output = std::move(task_result.value());
			if (maybe_output.first) {
				if (!release_payloads && !maybe_output.second.has_exchange_sink_instance()) {
					retain_payload_until_query_cleanup[i] = true;
				}
				drained_outputs.push_back({task_context, output_ordinal++, std::move(maybe_output.second)});
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
			duckdb::SleepWithGILReleasedIfHeld(std::chrono::milliseconds(1));
		}
	}
	std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> retained_handles;
	std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> retry_handles;
	std::vector<std::string> release_errors;
	for (size_t idx = 0; idx < handles.size(); idx++) {
		auto &handle = handles[idx];
		bool handle_failed = false;
		try {
			handle->AckPollResult();
		} catch (const std::exception &ex) {
			handle_failed = true;
			release_errors.push_back("ack[" + std::to_string(idx) + "]: " + ex.what());
		} catch (...) {
			handle_failed = true;
			release_errors.push_back("ack[" + std::to_string(idx) + "]: unknown error");
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
				release_errors.push_back("release[" + std::to_string(idx) + "]: " + ex.what());
			} catch (...) {
				handle_failed = true;
				release_errors.push_back("release[" + std::to_string(idx) + "]: unknown error");
			}
			if (handle_failed) {
				retry_handles.push_back(std::move(handle));
			}
		}
	}
	RetainFteResultHandles(query_id, std::move(retained_handles));
	StoreFteResultHandles(query_id, std::move(retry_handles));
	if (!release_errors.empty()) {
		std::string message = "failed to finalize FTE result handle(s)";
		for (const auto &error : release_errors) {
			message += "; " + error;
		}
		return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
		    DuckDBError::external_error(message));
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
				return DuckDBResult<void>::err(snapshots_res.error());
			}
			workers = collect_workers();
		}
		if (workers.empty()) {
			return DuckDBResult<void>::err(
			    DuckDBError::invalid_state_error("No Ray workers available for FTE task events"));
		}

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
		submission_errors_.Store(submission_error_owner, e);
		return DuckDBResult<void>::err(DuckDBError(string("Python error during submit_fte_task_events: ") + e.what()));
	} catch (const std::exception &e) {
		return DuckDBResult<void>::err(DuckDBError(string("submit_fte_task_events failed: ") + e.what()));
	}
}

void RayWorkerManager::rethrow_submission_error(const string &query_id) {
	submission_errors_.RethrowAsCause(query_id,
	                                  string("distributed worker task submission failed for query_id=") + query_id);
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
	{
		std::unique_lock<mutex> guard(mutex_);
		shutdown_cv_.wait(guard, [&]() { return state_.active_operations == 0; });
		state_.ray_workers.clear();
		result_handles = std::move(state_.fte_result_handles_by_query);
		retained_result_handles = std::move(state_.retained_fte_result_handles_by_query);
		state_.last_refresh = {};
	}
	result_handles.clear();
	retained_result_handles.clear();
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

void RayWorkerManager::drop_query_fragments(const string &query_id) {
	if (query_id.empty()) {
		return;
	}
	OperationGuard operation(*this);
	if (!operation) {
		throw std::runtime_error("Ray worker manager is shut down");
	}
	submission_errors_.Discard(query_id);
	std::vector<std::string> errors;
	std::vector<std::string> prepare_errors;
	try {
		ClearFteResultHandles(query_id);
	} catch (const std::exception &ex) {
		errors.push_back(std::string("result handles: ") + ex.what());
	} catch (...) {
		errors.push_back("result handles: unknown cleanup error");
	}
	std::vector<std::shared_ptr<RayWorkerRuntime>> workers;
	{
		lock_guard<mutex> guard(mutex_);
		workers.reserve(state_.ray_workers.size());
		for (auto &kv : state_.ray_workers) {
			workers.push_back(kv.second);
		}
	}
	for (auto &worker : workers) {
		const auto worker_id = worker->Id() ? *worker->Id() : std::string("<unknown>");
		try {
			worker->PrepareDropQuery(query_id);
		} catch (const std::exception &ex) {
			prepare_errors.push_back(worker_id + ": " + ex.what());
		} catch (...) {
			prepare_errors.push_back(worker_id + ": unknown prepare-teardown error");
		}
	}
	errors.insert(errors.end(), prepare_errors.begin(), prepare_errors.end());
	// Storage deletion is a distributed barrier: no worker may clean its
	// published attempts until every worker has fenced and drained native work.
	if (prepare_errors.empty()) {
		for (auto &worker : workers) {
			const auto worker_id = worker->Id() ? *worker->Id() : std::string("<unknown>");
			try {
				worker->CleanupQuery(query_id);
			} catch (const std::exception &ex) {
				errors.push_back(worker_id + ": " + ex.what());
			} catch (...) {
				errors.push_back(worker_id + ": unknown storage-cleanup error");
			}
		}
	}
	if (!errors.empty()) {
		std::string message = "query teardown failed with " + std::to_string(errors.size()) + " error(s)";
		for (const auto &error : errors) {
			message += "; " + error;
		}
		throw std::runtime_error(message);
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
	if (query_id.empty()) {
		return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
		    DuckDBError::value_error("query_id must be non-empty"));
	}
	OperationGuard operation(*this);
	if (!operation) {
		return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
		    DuckDBError::invalid_state_error("Ray worker manager is shut down"));
	}

	std::vector<duckdb::distributed::MaterializedOutput> outputs;
	RayWorkerRuntime::QueryStatus finished_status;
	bool has_finished_status = false;
	const bool has_deadline = timeout_s > 0.0;
	const auto deadline = has_deadline ? std::chrono::steady_clock::now() +
	                                         std::chrono::duration_cast<std::chrono::steady_clock::duration>(
	                                             std::chrono::duration<double>(timeout_s))
	                                   : std::chrono::steady_clock::time_point::max();

	try {
		while (true) {
			if (ShutdownStarted()) {
				return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
				    DuckDBError::invalid_state_error("Ray worker manager is shutting down"));
			}
			const auto *task_context_filter = task_contexts.empty() ? nullptr : &task_contexts;
			auto status_res = FteQueryStatus(query_id, task_context_filter);
			if (status_res.is_err()) {
				ClearFteResultHandles(query_id);
				return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(status_res.error());
			}
			const auto &status = status_res.value();
			if (status.failed) {
				ClearFteResultHandles(query_id);
				return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
				    DuckDBError::external_error("FTE query failed: " + status.message));
			}
			if (status.canceled) {
				ClearFteResultHandles(query_id);
				return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
				    DuckDBError::external_error("FTE query canceled: " + status.message));
			}
			auto collect_res = CollectFteResultHandles(query_id);
			if (collect_res.is_err()) {
				ClearFteResultHandles(query_id);
				return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(collect_res.error());
			}
			// Registry operations fence every ingress path that can publish a
			// fragment. Once no such operation remains, an unmatched materializer
			// scope cannot appear later and must not poll indefinitely.
			if (task_context_filter && !status.matched) {
				if (!status.registration_pending) {
					return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
					    DuckDBError::external_error("FTE query scope did not match any registered fragment: " +
					                                status.message));
				}
			} else if (status.finished) {
				finished_status = status;
				has_finished_status = true;
				break;
			}
			if (has_deadline && std::chrono::steady_clock::now() >= deadline) {
				return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
				    DuckDBError::external_error("timed out waiting for FTE query: " + status.message));
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}

		auto collect_res = CollectFteResultHandles(query_id);
		if (collect_res.is_err()) {
			ClearFteResultHandles(query_id);
			return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(collect_res.error());
		}
		const double remaining_timeout_s =
		    has_deadline
		        ? std::max(0.0, std::chrono::duration<double>(deadline - std::chrono::steady_clock::now()).count())
		        : -1.0;
		const auto *task_context_filter = task_contexts.empty() ? nullptr : &task_contexts;
		auto drain_res =
		    DrainFteResultHandles(query_id, remaining_timeout_s, has_finished_status ? &finished_status : nullptr,
		                          task_context_filter, false);
		if (drain_res.is_err()) {
			return drain_res;
		}
		for (auto &output : drain_res.value()) {
			if (on_output) {
				auto callback_res = on_output(output);
				if (callback_res.is_err()) {
					ClearFteResultHandles(query_id);
					return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
					    callback_res.error());
				}
			}
			outputs.push_back(std::move(output));
		}
		return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::ok(std::move(outputs));
	} catch (const std::exception &e) {
		ClearFteResultHandles(query_id);
		return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(
		    DuckDBError(string("Python error during wait_fte_query: ") + e.what()));
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
