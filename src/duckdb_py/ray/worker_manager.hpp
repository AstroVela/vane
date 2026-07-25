// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <memory>

#include <vector>
#include <string>

#include "worker.hpp"
#include "task.hpp"
#include "duckdb/execution/distributed/utils/channel.hpp"

namespace duckdb {
namespace distributed {
namespace python {
namespace ray {

class RayWorkerManager : public duckdb::distributed::WorkerManager {
public:
	DuckDBResult<void> submit_fte_task_events(std::vector<duckdb::distributed::WorkerTask> tasks) override;

	// WorkerManager interface implementations (one-to-one with Rust trait)
	DuckDBResult<std::vector<duckdb::distributed::WorkerSnapshot>> worker_snapshots() const override;
	DuckDBResult<void> try_autoscale(const std::vector<duckdb::distributed::TaskResourceRequest> &bundles) override;
	DuckDBResult<void> shutdown() override;
	DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>> wait_fte_query(const string &query_id,
	                                                                                  double timeout_s) override;
	DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>
	wait_fte_query(const string &query_id, double timeout_s,
	               duckdb::distributed::MaterializedOutputCallback on_output) override;
	DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>> wait_fte_query(
	    const string &query_id, double timeout_s,
	    const std::unordered_set<duckdb::distributed::TaskContext, duckdb::distributed::TaskContextHash> &task_contexts,
	    duckdb::distributed::MaterializedOutputCallback on_output) override;
	DuckDBResult<void> task_input_stream_exhausted_for_query(
	    const string &query_id, const std::unordered_set<duckdb::distributed::SourceNodeId> &source_node_ids) override;

	void drop_query_fragments(const string &query_id);
	std::unordered_map<string, std::unordered_map<string, idx_t>> fragment_stats_by_worker() const;

private:
	struct State {
		std::unordered_map<WorkerId, std::shared_ptr<RayWorkerRuntime>, WorkerIdHash, WorkerIdEqual> ray_workers;
		std::pair<bool, std::chrono::steady_clock::time_point> last_refresh;
		std::unordered_map<string, std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>>>
		    fte_result_handles_by_query;
		std::unordered_map<string, std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>>>
		    retained_fte_result_handles_by_query;
		idx_t active_operations = 0;
		bool shutdown_started = false;
		bool shutdown_finished = false;
		std::string shutdown_error;
	};

	class OperationGuard {
	public:
		explicit OperationGuard(const RayWorkerManager &manager)
		    : manager_(manager), active_(manager_.BeginOperation()) {
		}
		OperationGuard(const OperationGuard &) = delete;
		OperationGuard &operator=(const OperationGuard &) = delete;
		~OperationGuard() {
			if (active_) {
				manager_.EndOperation();
			}
		}
		explicit operator bool() const {
			return active_;
		}

	private:
		const RayWorkerManager &manager_;
		bool active_;
	};

	mutable mutex mutex_;
	mutable std::condition_variable shutdown_cv_;
	mutable State state_;

	bool BeginOperation() const;
	void EndOperation() const;
	static string QueryIdFromTaskEvents(const std::vector<duckdb::distributed::WorkerTask> &tasks);
	void StoreFteResultHandles(const string &query_id,
	                           std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> handles);
	void RetainFteResultHandles(const string &query_id,
	                            std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>> handles);
	void ClearFteResultHandles(const string &query_id);
	DuckDBResult<void> CollectFteResultHandles(const string &query_id);
	DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>> DrainFteResultHandles(
	    const string &query_id, double timeout_s, const RayWorkerRuntime::QueryStatus *finished_status = nullptr,
	    const std::unordered_set<duckdb::distributed::TaskContext, duckdb::distributed::TaskContextHash>
	        *task_context_filter = nullptr,
	    bool release_payloads = true);
	DuckDBResult<RayWorkerRuntime::QueryStatus> FteQueryStatus(const string &query_id);
};

} // namespace ray
} // namespace python
} // namespace distributed
} // namespace duckdb
