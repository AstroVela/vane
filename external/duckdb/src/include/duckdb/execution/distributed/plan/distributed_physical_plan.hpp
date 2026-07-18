// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

// Distributed physical plan and result stream interfaces.

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "duckdb/parallel/task_executor.hpp"

#include "duckdb/execution/distributed/common_types.hpp"
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
	// 构造函数
	DistributedPhysicalPlan(uint16_t query_idx, std::string query_id,
	                        std::shared_ptr<duckdb::PhysicalPlan> physical_plan, DuckDBExecutionConfigRef config)
	    : query_idx_(query_idx), query_id_(std::move(query_id)), physical_plan_(std::move(physical_plan)),
	      config_(std::move(config)) {
	}

	// 获取查询索引
	uint16_t idx() const {
		return query_idx_;
	}

	// 获取查询ID
	const std::string &query_id() const {
		return query_id_;
	}

	// 获取物理计划（返回引用避免不必要的拷贝）
	const std::shared_ptr<duckdb::PhysicalPlan> &physical_plan() const {
		return physical_plan_;
	}

	// 获取执行配置
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
	// 禁用默认构造函数
	DistributedPhysicalPlan() = delete;
};

// Shared status for background plan execution. The execute task can finish
// after run_plan() has returned a stream, so errors need an explicit side
// channel rather than a local return value.
class PlanExecutionStatus {
public:
	void RecordError(const DuckDBError &error) {
		std::lock_guard<std::mutex> lock(mutex_);
		if (!error_) {
			error_ = std::make_shared<DuckDBError>(error);
		}
	}

	std::shared_ptr<DuckDBError> GetError() const {
		std::lock_guard<std::mutex> lock(mutex_);
		return error_;
	}

	void ThrowIfError() const {
		auto error = GetError();
		if (error) {
			throw *error;
		}
	}

private:
	mutable std::mutex mutex_;
	std::shared_ptr<DuckDBError> error_;
};

// PlanResultStream — flattens MaterializedOutput → ResultPartitionRef.
//
// Background plan tasks run on DuckDB's
// TaskScheduler thread pool via TaskExecutor. Destructor explicitly
// closes receiver (→ tasks detect send failure and exit), then calls
// WorkOnTasks() to wait for completion.
class PlanResultStream {
public:
	PlanResultStream() = default;
	PlanResultStream(std::shared_ptr<TaskExecutor> executor, UnboundedReceiver<MaterializedOutput> rx,
	                 std::shared_ptr<PlanExecutionStatus> status = nullptr)
	    : executor_(std::move(executor)), receiver_(std::move(rx)), status_(std::move(status)) {
	}

	~PlanResultStream() {
		// Close the channel so background plan tasks
		// detect send() failure and exit on their own.  We intentionally do NOT
		// call executor_->WorkOnTasks() here because the destructor may run on
		// the Python asyncio event-loop thread (during coroutine-frame GC).
		// WorkOnTasks() spin-waits for ALL executor tasks to finish, which would
		// block the event loop and prevent Ray from delivering the final result
		// back to the client — causing a hang.
		//
		// Safety: both background tasks capture a shared_ptr<TaskExecutor>,
		// keeping the executor alive until they finish naturally.
		receiver_ = UnboundedReceiver<MaterializedOutput>();
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
			return next(); // skip empty outputs
		}
		if (status_) {
			status_->ThrowIfError();
		}
		return std::make_pair(true, curr_fragments_[curr_index_++]);
	}

	bool is_exhausted() const {
		return receiver_.is_disconnected() && curr_index_ >= curr_fragments_.size();
	}

private:
	std::shared_ptr<TaskExecutor> executor_;
	UnboundedReceiver<MaterializedOutput> receiver_;
	std::shared_ptr<PlanExecutionStatus> status_;
	std::vector<ResultPartitionRef> curr_fragments_;
	size_t curr_index_ = 0;
};

} // namespace distributed
} // namespace duckdb
