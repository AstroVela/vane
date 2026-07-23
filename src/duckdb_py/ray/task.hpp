// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <atomic>
#include <exception>
#include <memory>
#include <vector>

#include <string>
#include <cstdint>
#include <unordered_map>
#include <mutex>
#include <optional>

#include <duckdb/execution/distributed/common_types.hpp>
#include <duckdb/execution/distributed/exchange/exchange_handles.hpp>
#include <duckdb/execution/distributed/scheduling/task.hpp>
#include <duckdb/execution/distributed/scheduling/worker.hpp>
#include "safe_pyobject.hpp"

namespace duckdb {
class ClientContext;
}

namespace duckdb {
namespace distributed {
namespace python {
namespace ray {

namespace py = pybind11;

struct RayTaskPollState;

// RayTaskResult mirrors the Rust enum with an extra task-level result_schema
// carrier used by the distributed Python runner.
struct RayTaskResult {
	enum class Tag { Success, NoOutput, WorkerDied, WorkerUnavailable } tag;
	std::vector<duckdb::distributed::python::ray::SafePyObject>
	    parts; // each part is a Python object; wrapper RayResultPartitionRef may be used instead
	std::vector<uint8_t> stats_serialized;
	duckdb::distributed::python::ray::SafePyObject result_schema;
	duckdb::distributed::python::ray::SafePyObject exchange_sink_instance;
	int flight_port = 0;

	RayTaskResult(Tag tag_p, std::vector<duckdb::distributed::python::ray::SafePyObject> parts_p,
	              std::vector<uint8_t> stats_serialized_p,
	              duckdb::distributed::python::ray::SafePyObject result_schema_p, int flight_port_p = 0,
	              duckdb::distributed::python::ray::SafePyObject exchange_sink_instance_p =
	                  duckdb::distributed::python::ray::SafePyObject())
	    : tag(tag_p), parts(std::move(parts_p)), stats_serialized(std::move(stats_serialized_p)),
	      result_schema(std::move(result_schema_p)), exchange_sink_instance(std::move(exchange_sink_instance_p)),
	      flight_port(flight_port_p) {
	}

	~RayTaskResult() {
		// SafePyObject destructor ensures Python teardown happens with the GIL
		parts.clear();
		result_schema.reset_with_gil();
		exchange_sink_instance.reset_with_gil();
	}

	static RayTaskResult Success(std::vector<py::object> parts_in, std::vector<uint8_t> stats,
	                             py::object result_schema_in = py::none(), int flight_port = 0,
	                             py::object exchange_sink_instance_in = py::none()) {
		std::vector<duckdb::distributed::python::ray::SafePyObject> safe_parts;
		safe_parts.reserve(parts_in.size());
		for (auto &p : parts_in) {
			safe_parts.emplace_back(std::move(p));
		}
		return RayTaskResult(Tag::Success, std::move(safe_parts), std::move(stats),
		                     duckdb::distributed::python::ray::SafePyObject(std::move(result_schema_in)), flight_port,
		                     duckdb::distributed::python::ray::SafePyObject(std::move(exchange_sink_instance_in)));
	}
	static RayTaskResult WorkerDied() {
		return RayTaskResult(Tag::WorkerDied, {}, {}, duckdb::distributed::python::ray::SafePyObject());
	}
	static RayTaskResult WorkerUnavailable() {
		return RayTaskResult(Tag::WorkerUnavailable, {}, {}, duckdb::distributed::python::ray::SafePyObject());
	}
	static RayTaskResult NoOutput() {
		return RayTaskResult(Tag::NoOutput, {}, {}, duckdb::distributed::python::ray::SafePyObject());
	}

	py::object ResultSchema() const {
		return result_schema.get();
	}
	py::object ExchangeSinkInstanceObject() const {
		return exchange_sink_instance.get();
	}
};

// A thin wrapper that holds the Python object reference and metadata
class RayResultPartitionRef {
public:
	RayResultPartitionRef(py::object obj, size_t num_rows, size_t size_bytes, py::object lease_owner)
	    : object_ref(std::move(obj)), lease_owner_(std::move(lease_owner)), num_rows_(num_rows),
	      size_bytes_(size_bytes) {
	}

	~RayResultPartitionRef() {
		// Ensure Python refcount operations happen with the GIL
		object_ref.reset_with_gil();
		lease_owner_.reset_with_gil();
	}

	py::object GetObjectRef() const {
		return object_ref.get();
	}
	size_t GetNumRows() const {
		return num_rows_;
	}
	size_t GetSizeBytes() const {
		return size_bytes_;
	}
	py::object GetLeaseOwner() const {
		return lease_owner_.get();
	}

private:
	duckdb::distributed::python::ray::SafePyObject object_ref;
	duckdb::distributed::python::ray::SafePyObject lease_owner_;
	size_t num_rows_;
	size_t size_bytes_;
};

class RayBackedResultPartition : public duckdb::distributed::ResultPartition {
public:
	RayBackedResultPartition(py::object object_ref, size_t num_rows, size_t size_bytes, py::object lease_owner);
	~RayBackedResultPartition() override;

	DuckDBResult<size_t> size_bytes() const override;
	DuckDBResult<size_t> num_rows() const override;

	std::shared_ptr<duckdb::ColumnDataCollection> to_column_data() const override;

	py::object GetObjectRef() const;
	py::object GetLeaseOwner() const;
	size_t GetNumRowsMetadata() const {
		return num_rows_;
	}
	size_t GetSizeBytesMetadata() const {
		return size_bytes_;
	}

private:
	duckdb::distributed::python::ray::SafePyObject object_ref_;
	duckdb::distributed::python::ray::SafePyObject lease_owner_;
	size_t num_rows_;
	size_t size_bytes_;
	mutable std::once_flag materialize_once_;
	mutable std::atomic<std::shared_ptr<duckdb::ColumnDataCollection>> materialized_collection_;
	mutable std::exception_ptr materialize_error_;
};

std::shared_ptr<duckdb::ColumnDataCollection>
MaterializePyPayloadToCollection(const py::object &obj, duckdb::ClientContext *context = nullptr);
py::object ResultPartitionToPyObject(const std::shared_ptr<duckdb::distributed::ResultPartition> &part);

// RayTaskResultHandle: wraps a Python task handle for polling results.
class RayTaskResultHandle {
public:
	using TaskContext = duckdb::distributed::TaskContext;
	using WorkerId = duckdb::distributed::WorkerId;
	using PollResult = DuckDBResult<std::pair<bool, duckdb::distributed::MaterializedOutput>>;

	RayTaskResultHandle(TaskContext task_context, py::object handle, WorkerId worker_id, std::string fte_task_id = "");
	RayTaskResultHandle(const RayTaskResultHandle &) = delete;
	RayTaskResultHandle &operator=(const RayTaskResultHandle &) = delete;
	RayTaskResultHandle(RayTaskResultHandle &&) = default;
	RayTaskResultHandle &operator=(RayTaskResultHandle &&) = default;

	~RayTaskResultHandle();

	TaskContext GetTaskContext() const;
	const std::string &GetFteTaskId() const;

	// Non-blocking poll() to check if result is ready and return an optional
	// MaterializedOutput.  The inner pair mirrors the FTE result handle:
	// first=false means the task/update completed without output.
	std::pair<bool, PollResult> poll();
	void AckPollResult();
	void ReleasePollResult();

private:
	struct PollResultCache {
		std::mutex mutex;
		std::optional<PollResult> result;
	};

	TaskContext task_context_;
	std::string fte_task_id_;
	std::shared_ptr<RayTaskPollState> poll_state_;
	std::shared_ptr<PollResultCache> poll_result_cache_;
	bool released_ = false;
};

// Backend-neutral Python task result handle. This has the same result contract
// as RayTaskResultHandle but polls the Python handle directly instead of going
// through the Ray driver batch-wait helper.
class PythonTaskResultHandle {
public:
	using TaskContext = duckdb::distributed::TaskContext;
	using WorkerId = duckdb::distributed::WorkerId;
	using PollResult = DuckDBResult<std::pair<bool, duckdb::distributed::MaterializedOutput>>;

	PythonTaskResultHandle(TaskContext task_context, py::object handle, WorkerId worker_id,
	                       std::string fte_task_id = "");
	PythonTaskResultHandle(const PythonTaskResultHandle &) = delete;
	PythonTaskResultHandle &operator=(const PythonTaskResultHandle &) = delete;
	PythonTaskResultHandle(PythonTaskResultHandle &&) = default;
	PythonTaskResultHandle &operator=(PythonTaskResultHandle &&) = default;

	TaskContext GetTaskContext() const;
	const std::string &GetFteTaskId() const;

	std::pair<bool, PollResult> poll();
	void AckPollResult();
	void ReleasePollResult();

private:
	struct PollResultCache {
		std::mutex mutex;
		std::optional<PollResult> result;
	};

	TaskContext task_context_;
	WorkerId worker_id_;
	std::string fte_task_id_;
	duckdb::distributed::python::ray::SafePyObject handle_;
	std::shared_ptr<PollResultCache> poll_result_cache_;
	bool acked_ = false;
	bool released_ = false;
};

PythonTaskResultHandle MakePythonTaskResultHandle(py::object handle);

// RayWorkerTask: thin wrapper around distributed::WorkerTask
class RayWorkerTask {
public:
	explicit RayWorkerTask(duckdb::distributed::WorkerTask task);

	std::unordered_map<string, string> Context() const;
	py::dict TaskContextInfo() const;
	string Name() const;

	// Return plan (opaque)
	py::object Plan() const;

	// Return task inputs keyed by source node id. Each value contains a
	// typed payload such as scan-task bytes or exchange-source-task bytes.
	py::dict Inputs() const;

	// Return the task-local remote exchange sink instance, if the plan has one.
	py::object ExchangeSinkInstance() const;

private:
	duckdb::distributed::WorkerTask task_;
};

} // namespace ray
} // namespace python
} // namespace distributed
} // namespace duckdb
