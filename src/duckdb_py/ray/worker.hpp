// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "duckdb_python/pybind11/gil_wrapper.hpp"

#include <pybind11/pybind11.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "task.hpp"
#include "duckdb/execution/distributed/scheduling/worker.hpp"
#include "duckdb/execution/distributed/scheduling/task.hpp"
#include "duckdb/execution/distributed/utils/channel.hpp"

namespace duckdb {
namespace distributed {
namespace python {
namespace ray {

namespace py = pybind11;
using WorkerId = duckdb::distributed::WorkerId;
using TaskContext = duckdb::distributed::TaskContext;
using WorkerTask = duckdb::distributed::WorkerTask;

class RayWorkerRuntime {
public:
	using TaskResultHandleType = duckdb::distributed::python::ray::RayTaskResultHandle;

	struct QueryStatus {
		bool failed = false;
		bool finished = false;
		std::string message;
		std::unordered_set<string> selected_attempt_task_ids;
	};

	RayWorkerRuntime(string worker_id, py::object ray_worker_handle, double num_cpus, double num_gpus,
	                 size_t total_memory_bytes);
	RayWorkerRuntime(const RayWorkerRuntime &) = delete;
	RayWorkerRuntime &operator=(const RayWorkerRuntime &) = delete;
	RayWorkerRuntime(RayWorkerRuntime &&other) noexcept = delete;
	RayWorkerRuntime &operator=(RayWorkerRuntime &&other) noexcept = delete;

	~RayWorkerRuntime() {
		// Ensure any Python objects are destroyed with the GIL
		PythonGILWrapper acquire;
		// Explicitly clear the Python handle while the GIL is held
		ray_worker_handle_ = py::none();
	}

	size_t TotalMemoryBytes() const {
		return total_memory_bytes_;
	}

	void SubmitFteTaskEvents(const std::vector<WorkerTask> &tasks);
	void TaskInputStreamExhaustedForQuery(const string &query_id,
	                                      const std::unordered_set<duckdb::distributed::SourceNodeId> &source_node_ids);
	QueryStatus FteQueryStatus(const string &query_id);
	std::vector<RayTaskResultHandle> PopFteResultHandles(const string &query_id);
	void PrepareDropQuery(const string &query_id);
	void CleanupQuery(const string &query_id);
	std::unordered_map<string, idx_t> FragmentStats() const;

	void PrepareShutdown();
	void FinishShutdown();
	void AbortShutdown();

	// worker interface
	const WorkerId &Id() const {
		return worker_id_;
	}
	double TotalNumCpus() const {
		return num_cpus_;
	}
	double TotalNumGpus() const {
		return num_gpus_;
	}

private:
	std::vector<RayTaskResultHandle> WrapFtePythonHandles(const py::list &py_handles);

	WorkerId worker_id_;
	py::object ray_worker_handle_;
	double num_cpus_;
	double num_gpus_;
	size_t total_memory_bytes_;
};

} // namespace ray
} // namespace python
} // namespace distributed
} // namespace duckdb
