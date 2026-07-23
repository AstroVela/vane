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
#include "duckdb_python/pybind11/gil_wrapper.hpp"

namespace py = pybind11;
using namespace duckdb::distributed::python::ray;
using duckdb::distributed::DuckDBError;
using duckdb::distributed::DuckDBResult;
using duckdb::distributed::TaskResourceRequest;
using duckdb::distributed::WorkerSnapshot;

static constexpr auto REFRESH_INTERVAL = std::chrono::seconds(5);

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
	string query_id;
	try {
		query_id = QueryIdFromTaskEvents(tasks);
		if (!tasks.empty() && query_id.empty()) {
			return DuckDBResult<void>::err(DuckDBError::value_error("FTE task events require non-empty query_id"));
		}
		auto collect_workers = [&]() {
			std::vector<std::shared_ptr<RayWorkerRuntime>> workers;
			lock_guard<mutex> guard(mutex_);
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
		submission_errors_.Store(query_id, e);
		return DuckDBResult<void>::err(DuckDBError(string("Python error during submit_fte_task_events: ") + e.what()));
	} catch (const std::exception &e) {
		return DuckDBResult<void>::err(DuckDBError(string("Python error during submit_fte_task_events: ") + e.what()));
	}
}

void RayWorkerManager::rethrow_submission_error(const string &query_id) {
	submission_errors_.RethrowAsCause(query_id,
	                                  string("distributed worker task submission failed for query_id=") + query_id);
}

DuckDBResult<std::vector<duckdb::distributed::WorkerSnapshot>> RayWorkerManager::worker_snapshots() const {
	bool should_refresh = false;
	std::vector<string> existing_ids;
	{
		lock_guard<mutex> guard(mutex_);
		should_refresh = !state_.last_refresh.first ||
		                 (std::chrono::steady_clock::now() - state_.last_refresh.second) > REFRESH_INTERVAL;
		if (should_refresh) {
			existing_ids.reserve(state_.ray_workers.size());
			for (auto &kv : state_.ray_workers) {
				if (kv.first) {
					existing_ids.push_back(*kv.first);
				}
			}
		}
	}

	if (should_refresh) {
		duckdb::PythonGILWrapper gil;
		try {
			py::module_ worker_pool_obj = py::module_::import("duckdb.runners.ray.worker_pool");
			py::object py_workers_obj = worker_pool_obj.attr("start_ray_workers")(existing_ids);

			py::iterable workers_iter;
			try {
				workers_iter = py_workers_obj.cast<py::iterable>();
			} catch (const py::cast_error &e) {
				return DuckDBResult<std::vector<duckdb::distributed::WorkerSnapshot>>::err(DuckDBError::external_error(
				    string("start_ray_workers must return an iterable of RayWorkerRuntime: ") + e.what()));
			}

			std::vector<std::shared_ptr<RayWorkerRuntime>> new_workers;
			for (auto item : workers_iter) {
				auto worker = item.cast<std::shared_ptr<RayWorkerRuntime>>();
				if (!worker) {
					return DuckDBResult<std::vector<duckdb::distributed::WorkerSnapshot>>::err(
					    DuckDBError::invalid_state_error("start_ray_workers returned null RayWorkerRuntime"));
				}
				auto worker_id = worker->Id();
				if (!worker_id) {
					return DuckDBResult<std::vector<duckdb::distributed::WorkerSnapshot>>::err(
					    DuckDBError::invalid_state_error("start_ray_workers returned worker without id"));
				}
				new_workers.push_back(std::move(worker));
			}

			{
				lock_guard<mutex> guard(mutex_);
				for (auto &worker : new_workers) {
					state_.ray_workers.emplace(duckdb::distributed::make_worker_id(*worker->Id()), worker);
				}
				state_.last_refresh = std::make_pair(true, std::chrono::steady_clock::now());
			}
		} catch (const py::error_already_set &e) {
			return DuckDBResult<std::vector<duckdb::distributed::WorkerSnapshot>>::err(
			    DuckDBError::external_error(string("refresh_workers python error: ") + e.what()));
		} catch (const std::exception &e) {
			return DuckDBResult<std::vector<duckdb::distributed::WorkerSnapshot>>::err(
			    DuckDBError::external_error(string("refresh_workers exception: ") + e.what()));
		} catch (...) {
			return DuckDBResult<std::vector<duckdb::distributed::WorkerSnapshot>>::err(
			    DuckDBError::external_error("refresh_workers unknown exception"));
		}
	}

	std::vector<duckdb::distributed::WorkerSnapshot> snapshots;
	{
		lock_guard<mutex> guard(mutex_);
		snapshots.reserve(state_.ray_workers.size());
		for (auto &kv : state_.ray_workers) {
			snapshots.emplace_back(kv.first, kv.second->TotalNumCpus(), kv.second->TotalNumGpus(),
			                       kv.second->TotalMemoryBytes());
		}
	}
	return DuckDBResult<std::vector<duckdb::distributed::WorkerSnapshot>>::ok(std::move(snapshots));
}

DuckDBResult<void> RayWorkerManager::shutdown() {
	std::vector<std::shared_ptr<RayWorkerRuntime>> workers;
	{
		lock_guard<mutex> guard(mutex_);
		workers.reserve(state_.ray_workers.size());
		for (auto &kv : state_.ray_workers) {
			workers.push_back(kv.second);
		}
	}
	for (auto &worker : workers) {
		worker->Shutdown();
	}
	return DuckDBResult<void>::ok();
}

void RayWorkerManager::drop_query_fragments(const string &query_id) {
	if (query_id.empty()) {
		return;
	}
	submission_errors_.Discard(query_id);
	std::vector<std::string> errors;
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
			worker->DropQueryFragments(query_id);
		} catch (const std::exception &ex) {
			errors.push_back(worker_id + ": " + ex.what());
		} catch (...) {
			errors.push_back(worker_id + ": unknown teardown error");
		}
	}
	if (!errors.empty()) {
		std::string message = "failed to drop query fragments on " + std::to_string(errors.size()) + " worker(s)";
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

DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>
RayWorkerManager::wait_fte_query(const string &query_id, double timeout_s) {
	return wait_fte_query(query_id, timeout_s, {});
}

DuckDBResult<RayWorkerRuntime::QueryStatus> RayWorkerManager::FteQueryStatus(const string &query_id) {
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
			auto status = worker->FteQueryStatus(query_id);
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
			auto status_res = FteQueryStatus(query_id);
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
			auto collect_res = CollectFteResultHandles(query_id);
			if (collect_res.is_err()) {
				ClearFteResultHandles(query_id);
				return DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>::err(collect_res.error());
			}
			if (status.finished) {
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
		py::module_ worker_pool = py::module_::import("duckdb.runners.ray.worker_pool");
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
