#include "duckdb/parallel/executor_task.hpp"
#include "duckdb/parallel/task_notifier.hpp"
#include "duckdb/execution/executor.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parallel/thread_context.hpp"

namespace duckdb {

ExecutorTask::ExecutorTask(Executor &executor_p, shared_ptr<Event> event_p)
    : executor(executor_p), event(std::move(event_p)), context(executor_p.context) {
	executor.RegisterTask();
}

ExecutorTask::ExecutorTask(ClientContext &context_p, shared_ptr<Event> event_p, const PhysicalOperator &op_p)
    : executor(Executor::Get(context_p)), event(std::move(event_p)), op(&op_p), context(context_p) {
	thread_context = make_uniq<ThreadContext>(context_p);
	executor.RegisterTask();
}

ExecutorTask::~ExecutorTask() {
	if (thread_context) {
		executor.Flush(*thread_context);
	}
	executor.UnregisterTask();
}

void ExecutorTask::Deschedule() {
	auto this_ptr = shared_from_this();
	{
		lock_guard<mutex> guard(interrupt_lock);
		D_ASSERT(interrupt_execution_state == InterruptExecutionState::BLOCKING);
		interrupt_execution_state = InterruptExecutionState::DESCHEDULING;
	}
	executor.AddToBeRescheduled(this_ptr);
	bool reschedule = false;
	{
		lock_guard<mutex> guard(interrupt_lock);
		D_ASSERT(interrupt_execution_state == InterruptExecutionState::DESCHEDULING);
		if (interrupt_reschedule_requested) {
			interrupt_execution_state = InterruptExecutionState::RESCHEDULED;
			reschedule = true;
		} else {
			interrupt_execution_state = InterruptExecutionState::DESCHEDULED;
		}
	}
	if (reschedule) {
		auto reschedule_task = shared_from_this();
		executor.RescheduleTask(reschedule_task);
	}
}

uint64_t ExecutorTask::CurrentInterruptEpoch() const {
	lock_guard<mutex> guard(interrupt_lock);
	D_ASSERT(interrupt_execution_state == InterruptExecutionState::RUNNING);
	return interrupt_epoch;
}

void ExecutorTask::Reschedule(uint64_t callback_epoch) {
	bool reschedule = false;
	{
		lock_guard<mutex> guard(interrupt_lock);
		if (callback_epoch != interrupt_epoch) {
			return;
		}
		switch (interrupt_execution_state) {
		case InterruptExecutionState::RUNNING:
		case InterruptExecutionState::BLOCKING:
		case InterruptExecutionState::DESCHEDULING:
			interrupt_reschedule_requested = true;
			return;
		case InterruptExecutionState::DESCHEDULED:
			interrupt_execution_state = InterruptExecutionState::RESCHEDULED;
			reschedule = true;
			break;
		case InterruptExecutionState::IDLE:
		case InterruptExecutionState::RESCHEDULED:
		case InterruptExecutionState::FINISHED:
			return;
		}
	}
	if (reschedule) {
		auto this_ptr = shared_from_this();
		executor.RescheduleTask(this_ptr);
	}
}

void ExecutorTask::BeginInterruptExecution() {
	lock_guard<mutex> guard(interrupt_lock);
	D_ASSERT(interrupt_execution_state == InterruptExecutionState::IDLE ||
	         interrupt_execution_state == InterruptExecutionState::RESCHEDULED ||
	         interrupt_execution_state == InterruptExecutionState::FINISHED);
	interrupt_epoch++;
	if (interrupt_epoch == 0) {
		interrupt_epoch++;
	}
	interrupt_execution_state = InterruptExecutionState::RUNNING;
	interrupt_reschedule_requested = false;
}

void ExecutorTask::FinishInterruptExecution(TaskExecutionResult result) {
	lock_guard<mutex> guard(interrupt_lock);
	D_ASSERT(interrupt_execution_state == InterruptExecutionState::RUNNING);
	if (result == TaskExecutionResult::TASK_BLOCKED) {
		interrupt_execution_state = InterruptExecutionState::BLOCKING;
		return;
	}
	interrupt_execution_state = InterruptExecutionState::FINISHED;
	interrupt_reschedule_requested = false;
}

TaskExecutionResult ExecutorTask::Execute(TaskExecutionMode mode) {
	BeginInterruptExecution();
	TaskExecutionResult result = TaskExecutionResult::TASK_ERROR;
	try {
		if (thread_context) {
			do {
				TaskNotifier task_notifier {context};
				thread_context->profiler.StartOperator(op);
				// to allow continuous profiling, always execute in small steps
				result = ExecuteTask(TaskExecutionMode::PROCESS_PARTIAL);
				thread_context->profiler.EndOperator(nullptr);
				executor.Flush(*thread_context);
			} while (mode == TaskExecutionMode::PROCESS_ALL && result == TaskExecutionResult::TASK_NOT_FINISHED);
		} else {
			TaskNotifier task_notifier {context};
			result = ExecuteTask(mode);
		}
	} catch (std::exception &ex) {
		executor.PushError(ErrorData(ex));
	} catch (...) { // LCOV_EXCL_START
		executor.PushError(ErrorData("Unknown exception in ExecutorTask::Execute"));
	} // LCOV_EXCL_STOP
	FinishInterruptExecution(result);
	return result;
}

} // namespace duckdb
