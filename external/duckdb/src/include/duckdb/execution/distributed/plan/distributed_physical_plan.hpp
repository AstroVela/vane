// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

// Distributed physical plan and result stream interfaces.

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/execution/distributed/plan/plan_task_executor.hpp"
#include "duckdb/execution/distributed/utils/channel.hpp"
#include "duckdb/execution/distributed/utils/stream.hpp"

#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"

namespace duckdb {
// Forward-declare Relation at outer `duckdb` namespace so distributed types can reference it without including heavy
// headers
class Relation;

namespace distributed {

// Uses the global query index counter in common_types.hpp.

// Forward declare DuckDB execution config and logical plan placeholder types.
class DuckDBExecutionConfig; // defined elsewhere
class LogicalPlan;           // Placeholder - actual logical plan implementation is not part of duckdb2

class DistributedPhysicalPlan {
private:
	uint16_t query_idx_;
	std::string query_id_;
	std::shared_ptr<duckdb::PhysicalPlan> physical_plan_;
	DuckDBExecutionConfigRef config_;

public:
	DistributedPhysicalPlan(uint16_t query_idx, std::string query_id,
	                        std::shared_ptr<duckdb::PhysicalPlan> physical_plan, DuckDBExecutionConfigRef config)
	    : query_idx_(query_idx), query_id_(std::move(query_id)), physical_plan_(std::move(physical_plan)),
	      config_(std::move(config)) {
	}

	uint16_t idx() const {
		return query_idx_;
	}

	const std::string &query_id() const {
		return query_id_;
	}

	const std::shared_ptr<duckdb::PhysicalPlan> &physical_plan() const {
		return physical_plan_;
	}

	DuckDBExecutionConfigRef execution_config() const {
		return config_;
	}

	// Build from an already-constructed logical plan (acts as 'from_logical_plan_builder')
	static DuckDBResult<std::shared_ptr<DistributedPhysicalPlan>>
	from_logical_plan_builder(const std::shared_ptr<LogicalPlan> &builder_plan, std::string query_id,
	                          DuckDBExecutionConfigRef config);

	// Construct a DistributedPhysicalPlan directly from a DuckDB Relation (non-materialized path).
	static DuckDBResult<std::shared_ptr<DistributedPhysicalPlan>>
	from_duckdb_relation(const shared_ptr<duckdb::Relation> &relation, std::string query_id,
	                     DuckDBExecutionConfigRef config = nullptr);

private:
	DistributedPhysicalPlan() = delete;
};

// PlanResultStream — flattens MaterializedOutput → ResultPartitionRef.
//
// Blocking plan-control tasks use PlanTaskExecutor rather than DuckDB's
// compute scheduler. The stream owns the executor while results are consumed.
class PlanResultStream {
public:
	enum class PollState : uint8_t { READY, PENDING, EXHAUSTED };

	struct PollResult {
		PollState state;
		ResultPartitionRef partition;
	};

	PlanResultStream() = default;
	PlanResultStream(std::shared_ptr<PlanTaskExecutor> executor, UnboundedReceiver<MaterializedOutput> rx,
	                 std::shared_ptr<PlanExecutionStatus> status = nullptr)
	    : executor_(std::move(executor)), receiver_(std::move(rx)), status_(std::move(status)) {
	}

	~PlanResultStream() {
		// Disconnect the receiver so background plan tasks detect send() failure
		// and queued outputs are released. We intentionally do not
		// wait here because the destructor may run on the Python asyncio event-loop
		// thread (during coroutine-frame GC).
		// Waiting for control tasks here would block the event loop and prevent
		// Ray from delivering the final result back to the client. Each detached
		// task retains the ClientContext and status until it finishes naturally.
		ClearReadyCallback();
		receiver_.close();
	}

	PlanResultStream(PlanResultStream &&) = default;
	PlanResultStream &operator=(PlanResultStream &&) = default;
	PlanResultStream(const PlanResultStream &) = delete;
	PlanResultStream &operator=(const PlanResultStream &) = delete;

	/// Poll for next ResultPartitionRef (blocking). Flattens MaterializedOutput
	/// partition vectors — buffers partitions from one MaterializedOutput
	/// and yields them one at a time.
	std::pair<bool, ResultPartitionRef> next() {
		if (status_) {
			status_->ThrowIfError();
		}
		// Yield buffered partitions first
		while (curr_index_ < curr_fragments_.size()) {
			if (status_) {
				status_->ThrowIfError();
			}
			return std::make_pair(true, curr_fragments_[curr_index_++]);
		}
		while (true) {
			// Fetch next MaterializedOutput from receiver
			auto opt = receiver_.recv();
			if (status_) {
				status_->ThrowIfError();
			}
			if (!opt.first)
				return std::make_pair(false, ResultPartitionRef());
			curr_fragments_ = opt.second.fragments();
			curr_index_ = 0;
			if (curr_fragments_.empty()) {
				if (status_) {
					status_->ThrowIfError();
				}
				continue;
			}
			if (status_) {
				status_->ThrowIfError();
			}
			return std::make_pair(true, curr_fragments_[curr_index_++]);
		}
	}

	/// Poll without occupying a thread. PENDING callers can install a one-shot
	/// readiness callback and poll again after the notification.
	PollResult try_next() {
		if (status_) {
			status_->ThrowIfError();
		}
		while (curr_index_ < curr_fragments_.size()) {
			if (status_) {
				status_->ThrowIfError();
			}
			return {PollState::READY, curr_fragments_[curr_index_++]};
		}
		while (true) {
			auto opt = receiver_.try_recv();
			if (status_) {
				status_->ThrowIfError();
			}
			if (!opt.first) {
				if (receiver_.is_disconnected()) {
					return {PollState::EXHAUSTED, ResultPartitionRef()};
				}
				return {PollState::PENDING, ResultPartitionRef()};
			}
			curr_fragments_ = opt.second.fragments();
			curr_index_ = 0;
			if (curr_fragments_.empty()) {
				continue;
			}
			return {PollState::READY, curr_fragments_[curr_index_++]};
		}
	}

	void NotifyWhenReady(std::function<void()> callback) {
		auto notified = std::make_shared<std::atomic<bool>>(false);
		auto notify_once = [notified, callback = std::move(callback)]() {
			if (!notified->exchange(true)) {
				callback();
			}
		};
		receiver_.notify_when_ready(notify_once);
		if (status_) {
			status_->NotifyWhenError(std::move(notify_once));
		}
	}

	void ClearReadyCallback() {
		receiver_.clear_ready_callback();
		if (status_) {
			status_->ClearErrorCallback();
		}
	}

	bool is_exhausted() const {
		return receiver_.is_disconnected() && curr_index_ >= curr_fragments_.size();
	}

private:
	std::shared_ptr<PlanTaskExecutor> executor_;
	UnboundedReceiver<MaterializedOutput> receiver_;
	std::shared_ptr<PlanExecutionStatus> status_;
	std::vector<ResultPartitionRef> curr_fragments_;
	size_t curr_index_ = 0;
};

} // namespace distributed
} // namespace duckdb
