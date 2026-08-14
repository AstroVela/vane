// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <optional>

#include <vector>
#include <string>

#include "safe_pyobject.hpp"
#include "worker.hpp"
#include "task.hpp"
#include "duckdb/execution/distributed/utils/channel.hpp"

namespace duckdb {
namespace distributed {
namespace python {
namespace ray {

string SubmissionErrorOwnerQueryId(const std::vector<duckdb::distributed::WorkerTask> &tasks,
                                   const string &execution_query_id);

class RayWorkerManager : public duckdb::distributed::WorkerManager,
                         public std::enable_shared_from_this<RayWorkerManager> {
public:
	using QueryCleanup = std::function<void(const string &)>;
	explicit RayWorkerManager(QueryCleanup query_cleanup = {});

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
	DuckDBResult<void> materialization_barrier_completed(const string &query_id,
	                                                     duckdb::distributed::NodeID node_id) override;
	DuckDBResult<void> abort_and_quiesce_query(const string &query_id) override;

	void register_query_owner(const string &query_id, const string &owner_query_id);
	void drop_query_fragments(const string &query_id);
	void rethrow_submission_error(const string &query_id, const string &details = string());
	DuckDBResult<void> close_session(const string &session_id);
	std::unordered_map<string, std::unordered_map<string, idx_t>> fragment_stats_by_worker() const;

private:
	using WorkerSnapshotResult = DuckDBResult<std::vector<duckdb::distributed::WorkerSnapshot>>;

	struct WorkerRefreshFlight {
		explicit WorkerRefreshFlight(std::shared_future<WorkerSnapshotResult> result_p) : result(std::move(result_p)) {
		}

		std::shared_future<WorkerSnapshotResult> result;
	};

	struct State {
		std::unordered_map<WorkerId, std::shared_ptr<RayWorkerRuntime>, WorkerIdHash, WorkerIdEqual> ray_workers;
		std::pair<bool, std::chrono::steady_clock::time_point> last_refresh;
		std::shared_ptr<WorkerRefreshFlight> worker_refresh;
		idx_t worker_membership_version = 0;
		std::unordered_map<string, std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>>>
		    fte_result_handles_by_query;
		std::unordered_map<string, std::vector<std::unique_ptr<RayWorkerRuntime::TaskResultHandleType>>>
		    retained_fte_result_handles_by_query;
		std::unordered_map<string, string> query_owner_by_query;
		std::unordered_map<string, idx_t> active_query_operations_by_owner;
		std::unordered_map<string, std::vector<std::shared_ptr<RayWorkerRuntime>>> workers_by_query_owner;
		std::unordered_set<string> closed_query_owners;
		std::unordered_set<string> quiesced_query_owners;
		std::unordered_map<string, uint64_t> quiescing_query_owners;
		std::unordered_map<string, idx_t> quiesce_waiters_by_owner;
		std::unordered_map<string, uint64_t> dropping_query_owners;
		uint64_t next_query_quiesce_token = 1;
		uint64_t next_query_drop_token = 1;
		idx_t active_operations = 0;
		bool shutdown_started = false;
		bool shutdown_finished = false;
	};

	struct QueryAbort {
		string owner_query_id;
		std::vector<string> execution_query_ids;
		std::vector<std::shared_ptr<RayWorkerRuntime>> workers;
		uint64_t token;
		bool had_active_operations;
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

	class QueryOperationGuard {
	public:
		QueryOperationGuard(RayWorkerManager &manager, const string &query_id,
		                    const string &requested_owner_query_id = string())
		    : manager_(manager), owner_query_id_(manager_.BeginQueryOperation(query_id, requested_owner_query_id)) {
		}
		QueryOperationGuard(const QueryOperationGuard &) = delete;
		QueryOperationGuard &operator=(const QueryOperationGuard &) = delete;
		~QueryOperationGuard() {
			if (owner_query_id_) {
				manager_.EndQueryOperation(*owner_query_id_);
			}
		}
		explicit operator bool() const {
			return owner_query_id_.has_value();
		}
		const string &owner_query_id() const {
			return *owner_query_id_;
		}

	private:
		RayWorkerManager &manager_;
		std::optional<string> owner_query_id_;
	};

	const string manager_instance_id_;
	const QueryCleanup query_cleanup_;
	mutable mutex mutex_;
	mutable std::condition_variable shutdown_cv_;
	mutable State state_;
	PythonExceptionStore submission_errors_;

	bool BeginOperation() const;
	void EndOperation() const;
	std::optional<string> BeginQueryOperation(const string &query_id, const string &requested_owner_query_id);
	void EndQueryOperation(const string &owner_query_id);
	void CloseQueryOwnerIngress(const string &owner_query_id);
	void WaitForQueryOperationsWithoutGIL(const string &owner_query_id);
	void WaitForQueryOperations(const string &owner_query_id);
	void RecordQueryWorkers(const string &owner_query_id,
	                        const std::vector<std::shared_ptr<RayWorkerRuntime>> &workers);
	std::optional<QueryAbort> BeginQueryAbortWithoutGIL(const string &query_id);
	std::optional<QueryAbort> BeginQueryAbort(const string &query_id);
	void EndQueryAbort(const string &owner_query_id, uint64_t token, bool succeeded);
	std::optional<QueryAbort> BeginQueryDropWithoutGIL(const string &query_id);
	std::optional<QueryAbort> BeginQueryDrop(const string &query_id);
	void EndQueryDrop(const string &owner_query_id, uint64_t token);
	void FinishQueryLifecycle(const string &owner_query_id, uint64_t drop_token);
	bool ShutdownStarted() const;
	bool RetireWorkerForFailure(const string &worker_id, const std::shared_ptr<RayWorkerRuntime> &worker,
	                            const std::shared_ptr<std::atomic<bool>> &retired) const;
	WorkerSnapshotResult WorkerSnapshotsWithoutGIL() const;
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
	DuckDBResult<RayWorkerRuntime::QueryStatus>
	FteQueryStatus(const string &query_id,
	               const std::unordered_set<duckdb::distributed::TaskContext, duckdb::distributed::TaskContextHash>
	                   *task_context_filter = nullptr);
};

} // namespace ray
} // namespace python
} // namespace distributed
} // namespace duckdb
