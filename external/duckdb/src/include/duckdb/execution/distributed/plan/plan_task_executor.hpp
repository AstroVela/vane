// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parallel/task_notifier.hpp"

namespace duckdb {
namespace distributed {

// Shared status for background plan execution. Plan-control tasks can finish
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

// Distributed plan-control tasks block on channels and remote FTE completion.
// Running them on DuckDB's fixed-size compute scheduler can exhaust every
// scheduler thread before a downstream coordinator starts. Give each control
// task its own thread; their count is bounded by the plan's control nodes, not
// by input partitions or worker tasks.
class PlanTaskExecutor {
public:
	explicit PlanTaskExecutor(duckdb::shared_ptr<ClientContext> client_context)
	    : PlanTaskExecutor(std::move(client_context), std::make_shared<PlanExecutionStatus>()) {
	}

	PlanTaskExecutor(duckdb::shared_ptr<ClientContext> client_context, std::shared_ptr<PlanExecutionStatus> status)
	    : client_context_(std::move(client_context)), status_(std::move(status)) {
		if (!client_context_) {
			throw InvalidInputException("PlanTaskExecutor requires a ClientContext");
		}
		if (!status_) {
			throw InvalidInputException("PlanTaskExecutor requires a PlanExecutionStatus");
		}
	}

	template <typename F>
	void ScheduleTask(F &&task) {
		using TaskFunc = typename std::decay<F>::type;
		auto task_ptr = std::make_shared<TaskFunc>(std::forward<F>(task));
		auto client_context = client_context_;
		auto status = status_;
		try {
			std::thread worker([task_ptr, client_context, status]() mutable {
				try {
					TaskNotifier task_notifier(client_context.get());
					(*task_ptr)();
				} catch (const std::exception &ex) {
					status->RecordError(DuckDBError::external_error(ex.what()));
				} catch (...) {
					status->RecordError(DuckDBError::external_error("plan-control task threw unknown exception"));
				}
			});
			worker.detach();
		} catch (const std::exception &ex) {
			status->RecordError(
			    DuckDBError::external_error(std::string("failed to start plan-control task: ") + ex.what()));
			throw;
		}
	}

private:
	duckdb::shared_ptr<ClientContext> client_context_;
	std::shared_ptr<PlanExecutionStatus> status_;
};

} // namespace distributed
} // namespace duckdb
