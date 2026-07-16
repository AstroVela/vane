// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file distributed_task.hpp
 * @brief Task wrapper for scheduling distributed plan work on DuckDB's TaskScheduler.
 *
 * DistributedPlanTask extends BaseExecutorTask to wrap a callable,
 * allowing background plan work to run on the database-level thread pool
 * instead of spawning per-plan jthreads.
 */

#pragma once

#include "duckdb/parallel/task_executor.hpp"
#include <functional>
#include <memory>
#include <utility>

namespace duckdb {
namespace distributed {

/// Type-erased move-only callable wrapper (avoids std::function's copy requirement).
class MoveOnlyCallable {
public:
	template <typename F>
	MoveOnlyCallable(F &&f) : impl_(new Model<typename std::decay<F>::type>(std::forward<F>(f))) {
	}
	void operator()() {
		impl_->call();
	}

private:
	struct Concept {
		virtual ~Concept() = default;
		virtual void call() = 0;
	};
	template <typename F>
	struct Model : Concept {
		F f_;
		Model(F f) : f_(std::move(f)) {
		}
		void call() override {
			f_();
		}
	};
	std::unique_ptr<Concept> impl_;
};

class DistributedPlanTask : public BaseExecutorTask {
public:
	template <typename F>
	DistributedPlanTask(TaskExecutor &executor, F &&fn) : BaseExecutorTask(executor), fn_(std::forward<F>(fn)) {
	}

	void ExecuteTask() override {
		fn_();
	}

	string TaskType() const override {
		return "DistributedPlanTask";
	}

private:
	MoveOnlyCallable fn_;
};

} // namespace distributed
} // namespace duckdb
