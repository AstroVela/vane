//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/parallel/executor_task.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/parallel/task.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/optional_ptr.hpp"

namespace duckdb {
class Event;
class PhysicalOperator;
class ThreadContext;

//! Execute a task within an executor, including exception handling
//! This should be used within queries
class ExecutorTask : public Task {
public:
	ExecutorTask(Executor &executor, shared_ptr<Event> event);
	ExecutorTask(ClientContext &context, shared_ptr<Event> event, const PhysicalOperator &op);
	~ExecutorTask() override;

public:
	void Deschedule() override;
	uint64_t CurrentInterruptEpoch() const override;
	void Reschedule(uint64_t interrupt_epoch) override;

public:
	Executor &executor;
	shared_ptr<Event> event;
	unique_ptr<ThreadContext> thread_context;
	optional_ptr<const PhysicalOperator> op;

private:
	enum class InterruptExecutionState : uint8_t {
		IDLE,
		RUNNING,
		BLOCKING,
		DESCHEDULING,
		DESCHEDULED,
		RESCHEDULED,
		FINISHED
	};

	void BeginInterruptExecution();
	void FinishInterruptExecution(TaskExecutionResult result);

	mutable mutex interrupt_lock;
	uint64_t interrupt_epoch = 0;
	InterruptExecutionState interrupt_execution_state = InterruptExecutionState::IDLE;
	bool interrupt_reschedule_requested = false;
	ClientContext &context;

public:
	virtual TaskExecutionResult ExecuteTask(TaskExecutionMode mode) = 0;
	TaskExecutionResult Execute(TaskExecutionMode mode) override;
};

} // namespace duckdb
