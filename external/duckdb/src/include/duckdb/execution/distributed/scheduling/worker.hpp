// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file worker.hpp
 * @brief Worker types and interfaces for distributed execution
 *
 * Translated from DuckDB's vane-distributed/src/scheduling/worker.rs to C++20.
 * Provides Worker and WorkerManager interfaces.
 */

#pragma once

#include <memory>
#include <unordered_set>
#include <string>
#include <vector>

#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/execution/distributed/scheduling/task.hpp"

namespace duckdb {
namespace distributed {

// Forward declarations
class WorkerSnapshot;

//------------------------------------------------------------------------------
// Worker Snapshot
//------------------------------------------------------------------------------

/**
 * @brief WorkerSnapshot - point-in-time snapshot of worker state
 * (Rust: WorkerSnapshot in scheduler/mod.rs)
 */
class WorkerSnapshot {
public:
	WorkerSnapshot() = default;

	WorkerSnapshot(WorkerId worker_id, double total_num_cpus, double total_num_gpus,
	               size_t total_memory_bytes = 4ULL * 1024 * 1024 * 1024)
	    : worker_id_(std::move(worker_id)), total_num_cpus_(total_num_cpus), total_num_gpus_(total_num_gpus),
	      total_memory_bytes_(total_memory_bytes) {
	}

	/// Get worker ID
	const WorkerId &worker_id() const {
		return worker_id_;
	}

	/// Get total CPUs
	double total_num_cpus() const {
		return total_num_cpus_;
	}

	/// Get total GPUs
	double total_num_gpus() const {
		return total_num_gpus_;
	}

	/// Get available CPUs
	double available_num_cpus() const {
		return total_num_cpus_;
	}

	/// Get available GPUs
	double available_num_gpus() const {
		return total_num_gpus_;
	}

	/// Get total memory bytes
	size_t total_memory_bytes() const {
		return total_memory_bytes_;
	}

	/// Get available memory bytes
	size_t available_memory_bytes() const {
		return total_memory_bytes_;
	}

private:
	WorkerId worker_id_;
	double total_num_cpus_ = 0.0;
	double total_num_gpus_ = 0.0;
	size_t total_memory_bytes_ = 4ULL * 1024 * 1024 * 1024;
};

//------------------------------------------------------------------------------
// Worker Manager Interface
//------------------------------------------------------------------------------

/**
 * @brief WorkerManager - interface for managing workers
 * (Rust: WorkerManager trait)
 */
class WorkerManager {
public:
	virtual ~WorkerManager() = default;

	/// Get snapshots of all workers
	virtual DuckDBResult<std::vector<WorkerSnapshot>> worker_snapshots() const = 0;

	/// Try to autoscale based on resource requests
	virtual DuckDBResult<void> try_autoscale(const std::vector<TaskResourceRequest> &resource_requests) = 0;

	/// Shutdown all workers
	virtual DuckDBResult<void> shutdown() = 0;

	/// Fired when the FTE task event stream for a query has been fully
	/// consumed. This is the production no-more-input signal for dynamic task
	/// inputs.
	virtual DuckDBResult<void>
	task_input_stream_exhausted_for_query(const std::string &query_id,
	                                      const std::unordered_set<SourceNodeId> &source_node_ids) {
		return DuckDBResult<void>::err(
		    DuckDBError::invalid_state_error("worker manager does not support FTE source exhaustion"));
	}

	/// Notify the query resource controller that a true distributed
	/// materialization boundary has finished and its downstream may start.
	virtual DuckDBResult<void> materialization_barrier_completed(const std::string &query_id, NodeID node_id) {
		return DuckDBResult<void>::err(
		    DuckDBError::invalid_state_error("worker manager does not support materialization barriers"));
	}

	/// Submit task-stream events directly to the FTE task-update coordinator.
	virtual DuckDBResult<void> submit_fte_task_events(std::vector<WorkerTask> tasks) {
		return DuckDBResult<void>::err(
		    DuckDBError::invalid_state_error("worker manager does not support FTE task events"));
	}

	/// Wait for a FTE query to complete in the coordinator.
	virtual DuckDBResult<std::vector<MaterializedOutput>> wait_fte_query(const std::string &query_id,
	                                                                     double timeout_s) {
		return DuckDBResult<std::vector<MaterializedOutput>>::err(
		    DuckDBError::invalid_state_error("worker manager does not support wait_fte_query"));
	}

	virtual DuckDBResult<std::vector<MaterializedOutput>> wait_fte_query(const std::string &query_id, double timeout_s,
	                                                                     MaterializedOutputCallback on_output) {
		auto res = wait_fte_query(query_id, timeout_s);
		if (res.is_err()) {
			return res;
		}
		if (on_output) {
			for (const auto &output : res.value()) {
				auto callback_res = on_output(output);
				if (callback_res.is_err()) {
					return DuckDBResult<std::vector<MaterializedOutput>>::err(callback_res.error());
				}
			}
		}
		return res;
	}

	/// Wait for a query while publishing each selected output as it is drained.
	/// Implementations must stop draining immediately when the callback fails.
	/// DataSink uses this method to enforce coordinator result bounds before all
	/// selected outputs are retained, so collecting first is not a safe fallback.
	/// A successful implementation must return an empty vector: every selected
	/// output is transferred exactly once through the callback.
	virtual DuckDBResult<std::vector<MaterializedOutput>>
	wait_fte_query_streaming(const std::string &query_id, double timeout_s, MaterializedOutputCallback on_output) {
		(void)query_id;
		(void)timeout_s;
		(void)on_output;
		return DuckDBResult<std::vector<MaterializedOutput>>::err(
		    DuckDBError::invalid_state_error("worker manager does not support streaming FTE result drain"));
	}

	virtual DuckDBResult<std::vector<MaterializedOutput>>
	wait_fte_query(const std::string &query_id, double timeout_s,
	               const std::unordered_set<TaskContext, TaskContextHash> &task_contexts,
	               MaterializedOutputCallback on_output) {
		(void)task_contexts;
		auto res = wait_fte_query(query_id, timeout_s);
		if (res.is_err()) {
			return res;
		}
		if (on_output) {
			for (const auto &output : res.value()) {
				auto callback_res = on_output(output);
				if (callback_res.is_err()) {
					return DuckDBResult<std::vector<MaterializedOutput>>::err(callback_res.error());
				}
			}
		}
		return res;
	}

	/// Resolve query_id to its resource-query lifecycle, permanently fence task
	/// ingress, and wait until every owned execution query can no longer produce
	/// side effects. This is an abort barrier, not final manager teardown:
	/// adapter-owned lifecycle and result ownership are finalized separately.
	virtual DuckDBResult<void> abort_and_quiesce_query(const std::string &query_id) = 0;
};

} // namespace distributed
} // namespace duckdb
