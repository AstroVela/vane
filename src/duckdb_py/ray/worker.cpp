// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "worker.hpp"
#include <pybind11/pybind11.h>
#include "duckdb_python/pybind11/gil_wrapper.hpp"

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

void FillSelectedAttemptTaskIds(const py::dict &status, RayWorkerRuntime::QueryStatus &result) {
	auto key = py::str("selected_attempt_task_ids");
	if (!status.contains(key)) {
		return;
	}
	auto selected_obj = py::reinterpret_borrow<py::object>(status[key]);
	if (selected_obj.is_none()) {
		return;
	}
	if (!py::isinstance<py::list>(selected_obj)) {
		throw duckdb::InternalException("FTE query status selected_attempt_task_ids must be a list");
	}
	for (auto item : selected_obj) {
		auto value = py::str(py::reinterpret_borrow<py::object>(item)).cast<std::string>();
		if (!value.empty()) {
			result.selected_attempt_task_ids.insert(std::move(value));
		}
	}
}

RayWorkerRuntime::QueryStatus ParseFteQueryStatus(const py::object &status_obj) {
	RayWorkerRuntime::QueryStatus result;
	result.message = py::str(status_obj).cast<std::string>();
	if (!py::isinstance<py::dict>(status_obj)) {
		throw duckdb::InternalException("FTE query status must be a dict");
	}
	auto status = status_obj.cast<py::dict>();
	result.failed = RequiredStatusBool(status, "failed");
	result.finished = RequiredStatusBool(status, "finished");
	FillSelectedAttemptTaskIds(status, result);
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
	for (size_t i = 0; i < py_handles.size(); ++i) {
		py::object py_task_handle = py::reinterpret_borrow<py::object>(py_handles[i]);
		auto task_context = TaskContextForFteHandle(py_task_handle);
		auto actual_worker_id = WorkerIdFromPythonHandle(py_task_handle);
		auto fte_task_id = FteTaskIdStringFromPythonHandle(py_task_handle);
		RayTaskResultHandle rh(task_context, py_task_handle, actual_worker_id, std::move(fte_task_id));
		handles.push_back(std::move(rh));
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

RayWorkerRuntime::QueryStatus RayWorkerRuntime::FteQueryStatus(const string &query_id) {
	QueryStatus result;
	if (query_id.empty()) {
		result.message = "query_id is empty";
		return result;
	}
	duckdb::PythonGILWrapper gil;
	py::object status_obj = ray_worker_handle_.attr("fte_query_status")(query_id);
	return ParseFteQueryStatus(status_obj);
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
