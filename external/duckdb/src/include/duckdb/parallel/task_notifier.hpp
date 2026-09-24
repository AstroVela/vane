//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/parallel/task_notifier.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/optional_ptr.hpp"
#include "duckdb/common/winapi.hpp"

namespace duckdb {
class ClientContext;

//! The TaskNotifier notifies ClientContextState listener about started / stopped tasks
class TaskNotifier {
public:
	explicit TaskNotifier(optional_ptr<ClientContext> context_p);

	~TaskNotifier();
	TaskNotifier(const TaskNotifier &) = delete;
	TaskNotifier &operator=(const TaskNotifier &) = delete;

	//! The executing task's context, including callbacks using shared file handles.
	DUCKDB_API static optional_ptr<ClientContext> GetCurrentContext();

private:
	optional_ptr<ClientContext> context;
	TaskNotifier *previous;
	static thread_local TaskNotifier *current;
};

} // namespace duckdb
