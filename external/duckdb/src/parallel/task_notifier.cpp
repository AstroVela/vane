#include "duckdb/parallel/task_notifier.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_context_state.hpp"

namespace duckdb {

thread_local TaskNotifier *TaskNotifier::current = nullptr;

TaskNotifier::TaskNotifier(optional_ptr<ClientContext> context_p) : context(context_p), previous(current) {
	current = this;
	try {
		if (context) {
			for (auto &state : context->registered_state->States()) {
				state->OnTaskStart(*context);
			}
		}
	} catch (...) {
		// A failed constructor has no destructor to restore the previous task.
		current = previous;
		throw;
	}
}

TaskNotifier::~TaskNotifier() {
	if (context) {
		for (auto &state : context->registered_state->States()) {
			state->OnTaskStop(*context);
		}
	}
	current = previous;
}

optional_ptr<ClientContext> TaskNotifier::GetCurrentContext() {
	for (auto task = current; task; task = task->previous) {
		if (task->context) {
			return task->context;
		}
	}
	return nullptr;
}

} // namespace duckdb
