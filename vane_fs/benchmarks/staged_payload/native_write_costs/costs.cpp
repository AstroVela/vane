// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "costs.hpp"
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <ostream>
#include <sqlite3.h>
#include <string_view>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

namespace write_costs {
thread_local Counters counters {};
thread_local bool enabled = false;
static thread_local Scope *active = nullptr;
static std::atomic<int> phase {-1};
static const char *NAMES[] = {"root",
                              "write_bytes",
                              "batch",
                              "build_sql",
                              "concatenate",
                              "bind_batch",
                              "statement_acquire",
                              "statement_release",
                              "prepare",
                              "step_payload",
                              "step_version",
                              "step_other",
                              "exec",
                              "payload_append",
                              "pwrite",
                              "pwrite64",
                              "pread",
                              "sync",
                              "checkpoint"};
static_assert(sizeof(NAMES) / sizeof(*NAMES) == Count);
int64_t Now(int clock) {
	int saved = errno;
	timespec value {};
	if (clock_gettime(clock, &value))
		std::abort();
	errno = saved;
	return int64_t(value.tv_sec) * 1000000000 + value.tv_nsec;
}
Scope::Scope(Stage stage) : stage(stage), running(enabled) {
	if (!running)
		return;
	parent = active;
	wall = Now(CLOCK_MONOTONIC);
	cpu = Now(CLOCK_THREAD_CPUTIME_ID);
	active = this;
}
Scope::~Scope() {
	Finish();
}
void Scope::Finish() {
	if (!running)
		return;
	if (active != this)
		std::abort();
	auto elapsed_cpu = Now(CLOCK_THREAD_CPUTIME_ID) - cpu;
	auto elapsed_wall = Now(CLOCK_MONOTONIC) - wall;
	auto &value = counters[stage];
	++value.calls;
	value.inclusive_wall_ns += elapsed_wall;
	value.exclusive_wall_ns += elapsed_wall - child_wall;
	value.inclusive_cpu_ns += elapsed_cpu;
	value.exclusive_cpu_ns += elapsed_cpu - child_cpu;
	if (parent) {
		parent->child_wall += elapsed_wall;
		parent->child_cpu += elapsed_cpu;
	}
	active = parent;
	running = false;
}
void Allocated(size_t bytes) {
	if (enabled && active) {
		auto &value = counters[active->stage];
		++value.cpp_allocations;
		value.cpp_allocated_bytes += int64_t(bytes);
	}
}
void AddBytes(Stage stage, int64_t bytes) {
	if (enabled)
		counters[stage].bytes += bytes;
}
void Phase(int value) {
	phase.store(value, std::memory_order_relaxed);
}
void Begin(int value) {
	if (active || enabled)
		std::abort();
	counters = {};
	Phase(value);
	enabled = true;
}
Counters End() {
	if (active || !enabled)
		std::abort();
	enabled = false;
	return counters;
}
void Print(std::ostream &output, const Counters &values) {
	output << '{';
	for (int i = 0; i < Count; ++i) {
		auto &v = values[i];
		output << (i ? "," : "") << '"' << NAMES[i] << "\":[" << v.calls << ',' << v.inclusive_wall_ns << ','
		       << v.exclusive_wall_ns << ',' << v.inclusive_cpu_ns << ',' << v.exclusive_cpu_ns << ','
		       << v.cpp_allocations << ',' << v.cpp_allocated_bytes << ',' << v.bytes << ',' << v.vm_steps << ']';
	}
	output << '}';
}
struct Event {
	const char *kind = nullptr, *file = nullptr;
	int phase = -1, code = 0, error = 0, mode = -1, frames = -1, copied = -1;
	long tid = 0;
	int64_t begin = 0, end = 0, cpu = 0;
};
static std::array<Event, 16384> events;
static std::atomic<unsigned> next_event {0};
static bool Ends(std::string_view path, std::string_view suffix) {
	return path.size() >= suffix.size() && path.substr(path.size() - suffix.size()) == suffix;
}
static const char *FileKind(int fd) {
	char descriptor[64], path[4096];
	std::snprintf(descriptor, sizeof(descriptor), "/proc/self/fd/%d", fd);
	auto size = readlink(descriptor, path, sizeof(path));
	if (size < 0)
		return "unknown";
	std::string_view name(path, size_t(size));
	return Ends(name, ".payload")       ? "payload"
	       : Ends(name, "-wal")         ? "wal"
	       : Ends(name, ".stage-fence") ? "fence"
	       : Ends(name, ".sqlite")      ? "database"
	                                    : "other";
}
static void Record(Event event) {
	auto index = next_event.fetch_add(1);
	if (index >= events.size())
		std::abort();
	events[index] = event;
}
void PrintEvents(std::ostream &output) {
	output << '[';
	for (unsigned i = 0; i < next_event; ++i) {
		auto &e = events[i];
		output << (i ? "," : "") << "[\"" << e.kind << "\",\"" << e.file << "\"," << e.phase << ',' << e.tid << ','
		       << e.begin << ',' << e.end << ',' << e.cpu << ',' << e.code << ',' << e.error << ',' << e.mode << ','
		       << e.frames << ',' << e.copied << ']';
	}
	output << ']';
}
} // namespace write_costs

void *operator new(size_t bytes) {
	auto *result = std::malloc(bytes ? bytes : 1);
	if (!result)
		throw std::bad_alloc();
	write_costs::Allocated(bytes);
	return result;
}
void operator delete(void *pointer) noexcept {
	std::free(pointer);
}
void operator delete(void *pointer, size_t) noexcept {
	std::free(pointer);
}

using namespace write_costs;
extern "C" int __real_sqlite3_prepare_v2(sqlite3 *, const char *, int, sqlite3_stmt **, const char **);
extern "C" int __wrap_sqlite3_prepare_v2(sqlite3 *db, const char *sql, int bytes, sqlite3_stmt **statement,
                                         const char **tail) {
	Scope scope(Prepare);
	return __real_sqlite3_prepare_v2(db, sql, bytes, statement, tail);
}
extern "C" int __real_sqlite3_step(sqlite3_stmt *);
extern "C" int __wrap_sqlite3_step(sqlite3_stmt *statement) {
	if (!enabled)
		return __real_sqlite3_step(statement);
	auto *sql = sqlite3_sql(statement);
	auto stage = sql && !std::strncmp(sql, "INSERT INTO block_payloads(id,data)", 34)      ? StepPayload
	             : sql && !std::strncmp(sql, "INSERT INTO block_versions(inode,block", 37) ? StepVersion
	                                                                                       : StepOther;
	Scope scope(stage);
	sqlite3_stmt_status(statement, SQLITE_STMTSTATUS_VM_STEP, 1);
	auto code = __real_sqlite3_step(statement);
	counters[stage].vm_steps += sqlite3_stmt_status(statement, SQLITE_STMTSTATUS_VM_STEP, 1);
	return code;
}
extern "C" int __real_sqlite3_exec(sqlite3 *, const char *, int (*)(void *, int, char **, char **), void *, char **);
extern "C" int __wrap_sqlite3_exec(sqlite3 *db, const char *sql, int (*callback)(void *, int, char **, char **),
                                   void *context, char **error) {
	Scope scope(Exec);
	return __real_sqlite3_exec(db, sql, callback, context, error);
}
extern "C" ssize_t __real_pwrite(int, const void *, size_t, off_t);
extern "C" ssize_t __real_pwrite64(int, const void *, size_t, off64_t);
extern "C" ssize_t __real_pread(int, void *, size_t, off_t);
extern "C" ssize_t __real_pread64(int, void *, size_t, off64_t);
extern "C" ssize_t __wrap_pwrite(int fd, const void *data, size_t size, off_t offset) {
	Scope scope(Pwrite);
	auto code = __real_pwrite(fd, data, size, offset);
	if (code > 0)
		AddBytes(Pwrite, code);
	return code;
}
extern "C" ssize_t __wrap_pwrite64(int fd, const void *data, size_t size, off64_t offset) {
	Scope scope(Pwrite64);
	auto code = __real_pwrite64(fd, data, size, offset);
	if (code > 0)
		AddBytes(Pwrite64, code);
	return code;
}
extern "C" ssize_t __wrap_pread(int fd, void *data, size_t size, off_t offset) {
	Scope scope(Pread);
	return __real_pread(fd, data, size, offset);
}
extern "C" ssize_t __wrap_pread64(int fd, void *data, size_t size, off64_t offset) {
	Scope scope(Pread);
	return __real_pread64(fd, data, size, offset);
}
extern "C" int __real_fsync(int);
extern "C" int __real_fdatasync(int);
static int Synchronize(int fd, bool data_only) {
	Event event;
	event.kind = data_only ? "fdatasync" : "fsync";
	event.phase = phase.load(std::memory_order_relaxed);
	event.tid = syscall(SYS_gettid);
	Scope scope(Sync);
	event.begin = Now(CLOCK_MONOTONIC);
	auto cpu = Now(CLOCK_THREAD_CPUTIME_ID);
	auto code = data_only ? __real_fdatasync(fd) : __real_fsync(fd);
	auto saved = errno;
	event.cpu = Now(CLOCK_THREAD_CPUTIME_ID) - cpu;
	event.end = Now(CLOCK_MONOTONIC);
	scope.Finish();
	event.code = code;
	event.error = code ? saved : 0;
	event.file = FileKind(fd);
	Record(event);
	errno = saved;
	return code;
}
extern "C" int __wrap_fsync(int fd) {
	return Synchronize(fd, false);
}
extern "C" int __wrap_fdatasync(int fd) {
	return Synchronize(fd, true);
}
extern "C" int __real_sqlite3_wal_checkpoint_v2(sqlite3 *, const char *, int, int *, int *);
extern "C" int __wrap_sqlite3_wal_checkpoint_v2(sqlite3 *db, const char *name, int mode, int *frames, int *copied) {
	Event event;
	event.kind = "checkpoint";
	event.file = "sqlite";
	event.phase = phase.load(std::memory_order_relaxed);
	event.tid = syscall(SYS_gettid);
	Scope scope(Checkpoint);
	event.begin = Now(CLOCK_MONOTONIC);
	auto cpu = Now(CLOCK_THREAD_CPUTIME_ID);
	auto code = __real_sqlite3_wal_checkpoint_v2(db, name, mode, frames, copied);
	auto saved = errno;
	event.cpu = Now(CLOCK_THREAD_CPUTIME_ID) - cpu;
	event.end = Now(CLOCK_MONOTONIC);
	scope.Finish();
	event.code = code;
	event.mode = mode;
	event.frames = frames ? *frames : -1;
	event.copied = copied ? *copied : -1;
	Record(event);
	errno = saved;
	return code;
}
