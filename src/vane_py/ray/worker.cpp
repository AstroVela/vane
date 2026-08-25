// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "worker.hpp"
#include "bounded_diagnostics.hpp"
#include "python_bounded_diagnostics.hpp"
#include <pybind11/pybind11.h>
#include "vane_python/pybind11/gil_wrapper.hpp"

namespace py = pybind11;
using namespace duckdb::distributed::python::ray;

namespace {

bool TryWorkerIdAttr(const py::object &obj, WorkerId &worker_id_out) {
	if (!py::hasattr(obj, "worker_id")) {
		return false;
	}
	auto value = obj.attr("worker_id");
	if (value.is_none()) {
		return false;
	}
	auto worker_id = value.cast<std::string>();
	if (worker_id.empty()) {
		return false;
	}
	worker_id_out = duckdb::distributed::make_worker_id(worker_id);
	return true;
}

WorkerId WorkerIdFromPythonHandle(const py::object &handle) {
	WorkerId worker_id;
	if (TryWorkerIdAttr(handle, worker_id)) {
		return worker_id;
	}
	throw duckdb::InternalException("FTE result handle must provide non-empty worker_id");
}

TaskContext TaskContextForFteHandle(const py::object &handle) {
	if (!py::hasattr(handle, "task_context_info")) {
		throw duckdb::InternalException("FTE result handle must provide task_context_info");
	}
	auto info_obj = handle.attr("task_context_info");
	if (info_obj.is_none() || !py::isinstance<py::dict>(info_obj)) {
		throw duckdb::InternalException("FTE result handle task_context_info must be a dict");
	}
	auto info = info_obj.cast<py::dict>();
	for (auto key : {"query_idx", "last_node_id", "task_id", "node_ids"}) {
		if (!info.contains(key)) {
			throw duckdb::InternalException(std::string("FTE result handle task_context_info missing ") + key);
		}
	}

	uint16_t query_idx = static_cast<uint16_t>(info["query_idx"].cast<uint64_t>());
	auto last_node_id = static_cast<duckdb::distributed::NodeID>(info["last_node_id"].cast<uint64_t>());
	auto original_task_id = static_cast<duckdb::distributed::TaskID>(info["task_id"].cast<uint64_t>());
	std::vector<duckdb::distributed::NodeID> node_ids;
	for (auto node_id : info["node_ids"]) {
		node_ids.push_back(
		    static_cast<duckdb::distributed::NodeID>(py::reinterpret_borrow<py::object>(node_id).cast<uint64_t>()));
	}
	if (node_ids.empty()) {
		throw duckdb::InternalException("FTE result handle task_context_info node_ids must not be empty");
	}
	return TaskContext(query_idx, last_node_id, original_task_id, std::move(node_ids));
}

std::string FteTaskIdStringFromPythonHandle(const py::object &handle) {
	if (!py::hasattr(handle, "task_id")) {
		throw duckdb::InternalException("FTE result handle must provide task_id");
	}
	auto task_id = handle.attr("task_id");
	if (task_id.is_none()) {
		throw duckdb::InternalException("FTE result handle task_id must not be None");
	}
	if (!(py::hasattr(task_id, "query_id") && py::hasattr(task_id, "fragment_execution_id") &&
	      py::hasattr(task_id, "partition_id") && py::hasattr(task_id, "attempt_id"))) {
		throw duckdb::InternalException(
		    "FTE result handle task_id must expose query_id, fragment_execution_id, partition_id, and attempt_id");
	}

	auto query_id = py::str(task_id.attr("query_id")).cast<std::string>();
	if (query_id.empty()) {
		throw duckdb::InternalException("FTE result handle task_id query_id must be non-empty");
	}
	auto fragment_execution_id = task_id.attr("fragment_execution_id").cast<uint64_t>();
	auto partition_id = task_id.attr("partition_id").cast<uint64_t>();
	auto attempt_id = task_id.attr("attempt_id").cast<uint64_t>();
	return query_id + "." + std::to_string(fragment_execution_id) + "." + std::to_string(partition_id) + "." +
	       std::to_string(attempt_id);
}

void ReleasePoppedFteResultHandles(const py::list &py_handles, vane::BoundedErrorDetails &errors) {
	for (size_t index = 0; index < py_handles.size(); ++index) {
		try {
			auto handle = py::reinterpret_borrow<py::object>(py_handles[index]);
			handle.attr("release_result_payload")();
		} catch (const std::exception &ex) {
			errors.Add("release[" + std::to_string(index) + "]", ex.what());
		} catch (...) {
			errors.Add("release[" + std::to_string(index) + "]", "unknown release error");
		}
	}
}

bool RequiredStatusBool(const py::dict &status, const char *field_name) {
	auto key = py::str(field_name);
	if (!status.contains(key)) {
		throw duckdb::InternalException("FTE query status must include boolean '%s'", field_name);
	}
	auto value = py::reinterpret_borrow<py::object>(status[key]);
	if (!py::isinstance<py::bool_>(value)) {
		throw duckdb::InternalException("FTE query status field '%s' must be boolean", field_name);
	}
	return value.cast<bool>();
}

bool OptionalStatusBool(const py::dict &status, const char *field_name, bool default_value = false) {
	auto key = py::str(field_name);
	if (!status.contains(key)) {
		return default_value;
	}
	auto value = py::reinterpret_borrow<py::object>(status[key]);
	if (!py::isinstance<py::bool_>(value)) {
		throw duckdb::InternalException("FTE query status field '%s' must be boolean", field_name);
	}
	return value.cast<bool>();
}

bool TryBoundedTextValue(const py::object &value, const char *field_name, std::string &result) {
	if (!py::isinstance<py::str>(value)) {
		throw duckdb::InternalException("FTE query status field '%s' must be a string", field_name);
	}
	result = vane::BoundedPythonDiagnosticText(value);
	return result.find_first_not_of(" \t\n\r\f\v") != std::string::npos;
}

bool TryBoundedStatusText(const py::dict &status, const char *field_name, std::string &result) {
	auto key = py::str(field_name);
	if (!status.contains(key) || py::reinterpret_borrow<py::object>(status[key]).is_none()) {
		return false;
	}
	return TryBoundedTextValue(py::reinterpret_borrow<py::object>(status[key]), field_name, result);
}

bool TryBoundedFailedPartitionText(const py::dict &status, std::string &result) {
	auto failed_key = py::str("failed_partitions");
	if (!status.contains(failed_key) || py::reinterpret_borrow<py::object>(status[failed_key]).is_none()) {
		return false;
	}
	auto failed_obj = py::reinterpret_borrow<py::object>(status[failed_key]);
	if (!py::isinstance<py::list>(failed_obj)) {
		throw duckdb::InternalException("FTE query status field 'failed_partitions' must be a list");
	}
	auto failed_partitions = failed_obj.cast<py::list>();
	const auto detail_limit = vane::BoundedErrorDetails::MAX_DETAILS;
	for (size_t index = 0; index < failed_partitions.size() && index < detail_limit; ++index) {
		auto partition_obj = py::reinterpret_borrow<py::object>(failed_partitions[index]);
		if (!py::isinstance<py::dict>(partition_obj)) {
			throw duckdb::InternalException("FTE query status failed_partitions entries must be dicts");
		}
		auto partition = partition_obj.cast<py::dict>();
		auto latest_key = py::str("latest_failure");
		if (!partition.contains(latest_key) || py::reinterpret_borrow<py::object>(partition[latest_key]).is_none()) {
			continue;
		}
		auto latest = py::reinterpret_borrow<py::object>(partition[latest_key]);
		if (py::isinstance<py::str>(latest)) {
			if (TryBoundedTextValue(latest, "failed_partitions.latest_failure", result)) {
				return true;
			}
			continue;
		}
		if (!py::isinstance<py::dict>(latest)) {
			continue;
		}
		auto failure = latest.cast<py::dict>();
		for (const char *field_name : {"message", "failure_reason"}) {
			auto key = py::str(field_name);
			if (failure.contains(key) && !py::reinterpret_borrow<py::object>(failure[key]).is_none() &&
			    TryBoundedTextValue(py::reinterpret_borrow<py::object>(failure[key]), field_name, result)) {
				return true;
			}
		}
	}
	return false;
}

std::string QueryStatusMessage(const py::dict &status, const RayWorkerRuntime::QueryStatus &result) {
	std::string message;
	if (TryBoundedStatusText(status, "message", message) ||
	    TryBoundedStatusText(status, "scheduler_failure", message) || TryBoundedFailedPartitionText(status, message)) {
		return message;
	}
	return "failed=" + std::string(result.failed ? "true" : "false") +
	       ", finished=" + (result.finished ? "true" : "false") + ", canceled=" + (result.canceled ? "true" : "false") +
	       ", matched=" + (result.matched ? "true" : "false") +
	       ", registration_pending=" + (result.registration_pending ? "true" : "false") +
	       ", selected_attempt_count=" + std::to_string(result.selected_attempt_task_ids.size());
}

std::string RequiredSelectedAttemptTaskId(const py::handle &item) {
	auto value = py::reinterpret_borrow<py::object>(item);
	if (!py::isinstance<py::str>(value)) {
		throw duckdb::InternalException("FTE query status selected_attempt_task_ids entries must be strings");
	}
	const auto character_count = PyUnicode_GetLength(value.ptr());
	if (character_count < 0) {
		throw py::error_already_set();
	}
	if (static_cast<size_t>(character_count) > vane::BoundedErrorDetails::MAX_DETAIL_BYTES) {
		throw duckdb::InternalException("FTE query status selected_attempt_task_ids entry exceeds 4096 characters");
	}
	auto result = value.cast<std::string>();
	if (result.size() > vane::BoundedErrorDetails::MAX_DETAIL_BYTES) {
		throw duckdb::InternalException("FTE query status selected_attempt_task_ids entry exceeds 4096 bytes");
	}
	if (result.empty()) {
		throw duckdb::InternalException("FTE query status selected_attempt_task_ids entries must be non-empty");
	}
	return result;
}

void FillSelectedAttemptTaskIds(const py::dict &status, RayWorkerRuntime::QueryStatus &result) {
	auto key = py::str("selected_attempt_task_ids");
	if (!status.contains(key)) {
		throw duckdb::InternalException("FTE query status must include 'selected_attempt_task_ids'");
	}
	auto selected_obj = py::reinterpret_borrow<py::object>(status[key]);
	if (selected_obj.is_none()) {
		throw duckdb::InternalException("FTE query status selected_attempt_task_ids must be a list");
	}
	if (!py::isinstance<py::list>(selected_obj)) {
		throw duckdb::InternalException("FTE query status selected_attempt_task_ids must be a list");
	}
	for (auto item : selected_obj) {
		auto value = RequiredSelectedAttemptTaskId(item);
		if (!result.selected_attempt_task_ids.insert(std::move(value)).second) {
			throw duckdb::InternalException("FTE query status selected_attempt_task_ids entries must be unique");
		}
	}
}

RayWorkerRuntime::QueryStatus ParseFteQueryStatus(const py::object &status_obj, bool scoped) {
	RayWorkerRuntime::QueryStatus result;
	if (!py::isinstance<py::dict>(status_obj)) {
		throw duckdb::InternalException("FTE query status must be a dict");
	}
	auto status = status_obj.cast<py::dict>();
	result.failed = RequiredStatusBool(status, "failed");
	result.finished = RequiredStatusBool(status, "finished");
	result.canceled = OptionalStatusBool(status, "canceled");
	result.registration_pending = OptionalStatusBool(status, "registration_pending");
	if (scoped) {
		result.matched = RequiredStatusBool(status, "matched");
	}
	FillSelectedAttemptTaskIds(status, result);
	result.message = QueryStatusMessage(status, result);
	return result;
}

} // namespace

RayWorkerRuntime::RayWorkerRuntime(string worker_id, py::object ray_worker_handle, double num_cpus, double num_gpus,
                                   size_t total_memory_bytes)
    : worker_id_(std::make_shared<string>(std::move(worker_id))), ray_worker_handle_(std::move(ray_worker_handle)),
      num_cpus_(num_cpus), num_gpus_(num_gpus), total_memory_bytes_(total_memory_bytes) {
}

void RayWorkerRuntime::SubmitFteTaskEvents(const std::vector<WorkerTask> &tasks) {
	if (tasks.empty()) {
		return;
	}

	duckdb::PythonGILWrapper gil;
	py::list py_tasks;
	for (const auto &task : tasks) {
		RayWorkerTask py_task_wrapper(task);
		py_tasks.append(py::cast(std::move(py_task_wrapper), py::return_value_policy::move));
	}

	ray_worker_handle_.attr("submit_tasks")(py_tasks);
}

std::vector<RayTaskResultHandle> RayWorkerRuntime::WrapFtePythonHandles(const py::list &py_handles) {
	std::vector<RayTaskResultHandle> handles;
	handles.reserve(py_handles.size());
	try {
		for (size_t i = 0; i < py_handles.size(); ++i) {
			py::object py_task_handle = py::reinterpret_borrow<py::object>(py_handles[i]);
			auto task_context = TaskContextForFteHandle(py_task_handle);
			auto actual_worker_id = WorkerIdFromPythonHandle(py_task_handle);
			auto fte_task_id = FteTaskIdStringFromPythonHandle(py_task_handle);
			RayTaskResultHandle rh(task_context, py_task_handle, actual_worker_id, std::move(fte_task_id));
			handles.push_back(std::move(rh));
		}
	} catch (const std::exception &ex) {
		// pop_fte_result_handles transfers the whole batch out of the Python
		// registry. If one item is malformed, every item in that batch still
		// needs an explicit payload release, including handles wrapped before
		// the validation failure and handles not reached yet.
		handles.clear();
		vane::BoundedErrorDetails errors;
		errors.Add("conversion", ex.what());
		ReleasePoppedFteResultHandles(py_handles, errors);
		throw std::runtime_error(errors.AppendTo("failed to adopt popped FTE result handle batch"));
	} catch (...) {
		handles.clear();
		vane::BoundedErrorDetails errors;
		errors.Add("conversion", "unknown conversion error");
		ReleasePoppedFteResultHandles(py_handles, errors);
		throw std::runtime_error(errors.AppendTo("failed to adopt popped FTE result handle batch"));
	}
	return handles;
}

void RayWorkerRuntime::PrepareDropQuery(const string &query_id) {
	if (query_id.empty()) {
		return;
	}
	duckdb::PythonGILWrapper gil;
	ray_worker_handle_.attr("fte_prepare_drop_query")(query_id);
}

void RayWorkerRuntime::CleanupQuery(const string &query_id) {
	if (query_id.empty()) {
		return;
	}
	duckdb::PythonGILWrapper gil;
	ray_worker_handle_.attr("fte_cleanup_query")(query_id);
}

void RayWorkerRuntime::TaskInputStreamExhaustedForQuery(
    const string &query_id, const std::unordered_set<duckdb::distributed::SourceNodeId> &source_node_ids) {
	duckdb::PythonGILWrapper gil;
	py::list py_source_node_ids;
	for (auto source_node_id : source_node_ids) {
		py_source_node_ids.append(std::to_string(source_node_id));
	}
	ray_worker_handle_.attr("task_input_stream_exhausted_for_query")(query_id, py_source_node_ids);
}

RayWorkerRuntime::QueryStatus RayWorkerRuntime::FteQueryStatus(
    const string &query_id,
    const std::unordered_set<duckdb::distributed::TaskContext, duckdb::distributed::TaskContextHash>
        *task_context_filter) {
	QueryStatus result;
	if (query_id.empty()) {
		result.message = "query_id is empty";
		return result;
	}
	duckdb::PythonGILWrapper gil;
	const bool scoped = task_context_filter && !task_context_filter->empty();
	py::object status_obj;
	if (scoped) {
		py::list py_task_contexts;
		for (const auto &task_context : *task_context_filter) {
			py::dict info;
			info["query_idx"] = task_context.query_idx();
			info["last_node_id"] = task_context.last_node_id();
			info["task_id"] = task_context.task_id();
			py::list node_ids;
			for (auto node_id : task_context.node_ids()) {
				node_ids.append(node_id);
			}
			info["node_ids"] = std::move(node_ids);
			py_task_contexts.append(std::move(info));
		}
		status_obj = ray_worker_handle_.attr("fte_query_status")(query_id, std::move(py_task_contexts));
	} else {
		status_obj = ray_worker_handle_.attr("fte_query_status")(query_id);
	}
	return ParseFteQueryStatus(status_obj, scoped);
}

std::vector<RayTaskResultHandle> RayWorkerRuntime::PopFteResultHandles(const string &query_id) {
	std::vector<RayTaskResultHandle> handles;
	if (query_id.empty()) {
		return handles;
	}
	duckdb::PythonGILWrapper gil;
	py::object py_handles_obj = ray_worker_handle_.attr("pop_fte_result_handles")(query_id);
	if (py_handles_obj.is_none()) {
		return handles;
	}
	py::list py_handles = py_handles_obj.cast<py::list>();
	return WrapFtePythonHandles(py_handles);
}

std::unordered_map<std::string, duckdb::idx_t> RayWorkerRuntime::FragmentStats() const {
	duckdb::PythonGILWrapper gil;
	py::object stats_obj = ray_worker_handle_.attr("stats_fragments")();
	if (!py::isinstance<py::dict>(stats_obj)) {
		throw duckdb::InternalException("Ray worker handle stats_fragments() must return a dict");
	}
	std::unordered_map<std::string, duckdb::idx_t> stats;
	py::dict stats_dict = stats_obj.cast<py::dict>();
	for (auto item : stats_dict) {
		auto key = py::reinterpret_borrow<py::object>(item.first).cast<std::string>();
		auto value = py::reinterpret_borrow<py::object>(item.second).cast<duckdb::idx_t>();
		stats.emplace(std::move(key), value);
	}
	return stats;
}

void RayWorkerRuntime::CloseSession(const string &session_id) {
	if (session_id.empty()) {
		throw duckdb::InvalidInputException("Ray worker close session requires a non-empty session_id");
	}
	duckdb::PythonGILWrapper gil;
	ray_worker_handle_.attr("close_session")(session_id);
}

void RayWorkerRuntime::PrepareShutdown() {
	duckdb::PythonGILWrapper gil;
	ray_worker_handle_.attr("prepare_shutdown")();
}

void RayWorkerRuntime::FinishShutdown() {
	duckdb::PythonGILWrapper gil;
	ray_worker_handle_.attr("finish_shutdown")();
}

void RayWorkerRuntime::AbortShutdown() {
	duckdb::PythonGILWrapper gil;
	ray_worker_handle_.attr("abort_shutdown")();
}

bool RayWorkerRuntime::InstallFailureRetirementCallback(py::object callback) {
	duckdb::PythonGILWrapper gil;
	ray_worker_handle_.attr("_retire_from_manager_for_failure") = std::move(callback);
	return !py::hasattr(ray_worker_handle_, "_fte_failure_retirement_completed") ||
	       !py::cast<bool>(ray_worker_handle_.attr("_fte_failure_retirement_completed"));
}
