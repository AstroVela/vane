// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "wait_trace.hpp"
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>
#include <string_view>
#include <vector>
#include <sqlite3.h>
#include <sys/file.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

namespace fuse_waits {
struct Counter {
	int64_t calls = 0, wall = 0, exclusive_wall = 0, cpu = 0, exclusive_cpu = 0;
};
struct Row {
	const char *name;
	int64_t id, offset, bytes, inode, caller, begin = 0, end = 0, cpu = 0, steps = 0;
	std::array<Counter, Count> counters {};
};
struct Event {
	const char *kind, *file = "none";
	int64_t root = -1, begin = 0, end = 0, cpu = 0, result = 0, offset = -1, bytes = 0;
	int error = 0, fd = -1, mode = -1, frames = -1, copied = -1;
};
struct Thread {
	long tid = 0;
	std::vector<Row> rows;
	std::vector<Event> events;
};
static std::array<Thread, 16> threads;
static std::atomic<unsigned> next {0};
static thread_local Thread *thread = nullptr;
static thread_local Row *row = nullptr;
static thread_local Span *active = nullptr;
static bool enabled = false;
static std::string output;
static constexpr const char *NAMES[] = {
    "root",        "database_wait", "admission",  "checkpoint_wait", "publication_wait",
    "lease_wait",  "payload_guard", "flock",      "begin",           "commit",
    "publish",     "before_full",   "after_full", "begin_batch",     "prepare_external",
    "write_bytes", "step",          "exec",       "notify",          "pwrite",
    "pread",       "fsync",         "fdatasync",  "checkpoint"};
static_assert(sizeof(NAMES) / sizeof(*NAMES) == Count);
int64_t Now(int clock) {
	int saved = errno;
	timespec value {};
	if (clock_gettime(clock, &value))
		std::abort();
	errno = saved;
	return int64_t(value.tv_sec) * 1000000000 + value.tv_nsec;
}
static Thread &Local() {
	if (!thread) {
		auto slot = next.fetch_add(1);
		if (slot >= threads.size())
			std::abort();
		thread = &threads[slot];
		thread->tid = syscall(SYS_gettid);
		thread->rows.reserve(65536);
		thread->events.reserve(131072);
	}
	return *thread;
}
static void Record(Event event) {
	auto &events = Local().events;
	if (events.size() == events.capacity())
		std::abort();
	events.push_back(event);
}
Span::Span(Stage stage) : stage(stage), running(row != nullptr) {
	if (!running)
		return;
	parent = active;
	begin = Now(CLOCK_MONOTONIC);
	cpu = Now(CLOCK_THREAD_CPUTIME_ID);
	active = this;
}
Span::~Span() {
	Finish();
}
void Span::Finish() {
	if (!running)
		return;
	if (active != this)
		std::abort();
	cpu_ns = Now(CLOCK_THREAD_CPUTIME_ID) - cpu;
	end = Now(CLOCK_MONOTONIC);
	auto &c = row->counters[stage];
	++c.calls;
	c.wall += end - begin;
	c.exclusive_wall += end - begin - child_wall;
	c.cpu += cpu_ns;
	c.exclusive_cpu += cpu_ns - child_cpu;
	if (parent) {
		parent->child_wall += end - begin;
		parent->child_cpu += cpu_ns;
	}
	active = parent;
	running = false;
	if ((stage == DatabaseWait || stage == CheckpointWait || stage == PublicationWait || stage == LeaseWait ||
	     stage == PayloadGuard || stage == Admission) &&
	    end - begin >= 1000000) {
		Event event {NAMES[stage]};
		event.root = row->id;
		event.begin = begin;
		event.end = end;
		event.cpu = cpu_ns;
		Record(event);
	}
}
Root::Root(const char *name, int64_t offset, int64_t bytes, int64_t inode, int64_t caller) {
	if (!enabled)
		return;
	if (row || active)
		std::abort();
	auto &rows = Local().rows;
	if (rows.size() == rows.capacity())
		std::abort();
	rows.push_back({name, int64_t(rows.size()), offset, bytes, inode, caller});
	row = &rows.back();
	span.emplace(RootStage);
}
Root::~Root() {
	if (!span)
		return;
	span->Finish();
	row->begin = span->begin;
	row->end = span->end;
	row->cpu = span->cpu_ns;
	span.reset();
	row = nullptr;
}
static const char *FileKind(int fd) {
	char descriptor[64], path[4096];
	std::snprintf(descriptor, sizeof(descriptor), "/proc/self/fd/%d", fd);
	auto size = readlink(descriptor, path, sizeof(path));
	if (size < 0)
		return "unknown";
	std::string_view name(path, size_t(size));
	auto ends = [&](std::string_view suffix) {
		return name.size() >= suffix.size() && name.substr(name.size() - suffix.size()) == suffix;
	};
	return ends(".payload")            ? "payload"
	       : ends("-wal")              ? "wal"
	       : ends(".stage-fence")      ? "fence"
	       : ends(".stage-checkpoint") ? "checkpoint_lease"
	       : ends(".stage-publish")    ? "publication_lease"
	       : ends(".sqlite")           ? "database"
	                                   : "other";
}
static void Report() {
	std::ofstream file(output);
	file << "{\"clock\":\"CLOCK_MONOTONIC\",\"slow_event_threshold_ns\":1000000,\"stages\":[";
	for (int i = 0; i < Count; ++i)
		file << (i ? "," : "") << std::quoted(NAMES[i]);
	file << "],\"counter_columns\":[\"calls\",\"wall_ns\",\"exclusive_wall_ns\",\"cpu_ns\",\"exclusive_cpu_ns\"],"
	        "\"root_columns\":[\"name\",\"id\",\"begin_ns\",\"end_ns\",\"cpu_ns\",\"offset\",\"bytes\",\"inode\","
	        "\"caller\",\"vm_steps\",\"counters\"],"
	        "\"event_columns\":[\"kind\",\"file\",\"root\",\"begin_ns\",\"end_ns\",\"cpu_ns\",\"result\",\"errno\","
	        "\"fd\",\"offset\",\"bytes\",\"mode\",\"frames\",\"copied\"],\"threads\":[";
	for (unsigned i = 0; i < next; ++i) {
		auto &t = threads[i];
		file << (i ? "," : "") << "{\"tid\":" << t.tid << ",\"roots\":[";
		for (size_t j = 0; j < t.rows.size(); ++j) {
			auto &r = t.rows[j];
			file << (j ? "," : "") << '[' << std::quoted(r.name) << ',' << r.id << ',' << r.begin << ',' << r.end << ','
			     << r.cpu << ',' << r.offset << ',' << r.bytes << ',' << r.inode << ',' << r.caller << ',' << r.steps
			     << ",[";
			for (int k = 0; k < Count; ++k) {
				auto &c = r.counters[k];
				file << (k ? "," : "") << '[' << c.calls << ',' << c.wall << ',' << c.exclusive_wall << ',' << c.cpu
				     << ',' << c.exclusive_cpu << ']';
			}
			file << "]]";
		}
		file << "],\"events\":[";
		for (size_t j = 0; j < t.events.size(); ++j) {
			auto &e = t.events[j];
			file << (j ? "," : "") << '[' << std::quoted(e.kind) << ',' << std::quoted(e.file) << ',' << e.root << ','
			     << e.begin << ',' << e.end << ',' << e.cpu << ',' << e.result << ',' << e.error << ',' << e.fd << ','
			     << e.offset << ',' << e.bytes << ',' << e.mode << ',' << e.frames << ',' << e.copied << ']';
		}
		file << "]}";
	}
	file << "]}\n";
	if (!file)
		std::abort();
}
void Install(const char *database) {
	if (enabled)
		std::abort();
	output = std::string(database) + ".waits.json";
	enabled = true;
	std::atexit(Report);
}
template <class F>
static auto Call(Stage stage, const char *kind, int fd, int64_t offset, int64_t bytes, int mode, F call) {
	if (!enabled)
		return call();
	Span span(stage);
	Event e {kind};
	e.root = row ? row->id : -1;
	e.fd = fd;
	e.offset = offset;
	e.bytes = bytes;
	e.mode = mode;
	e.begin = Now(CLOCK_MONOTONIC);
	auto cpu = Now(CLOCK_THREAD_CPUTIME_ID);
	auto result = call();
	auto saved = errno;
	e.cpu = Now(CLOCK_THREAD_CPUTIME_ID) - cpu;
	e.end = Now(CLOCK_MONOTONIC);
	span.Finish();
	if (stage == Fsync || stage == Fdatasync || e.end - e.begin >= 1000000 || result < 0) {
		e.file = FileKind(fd);
		e.result = result;
		e.error = result < 0 ? saved : 0;
		Record(e);
	}
	errno = saved;
	return result;
}
} // namespace fuse_waits

using namespace fuse_waits;
extern "C" int __real_fsync(int);
extern "C" int __real_fdatasync(int);
extern "C" int __real_flock(int, int);
extern "C" ssize_t __real_pwrite(int, const void *, size_t, off_t);
extern "C" ssize_t __real_pwrite64(int, const void *, size_t, off64_t);
extern "C" ssize_t __real_pread(int, void *, size_t, off_t);
extern "C" ssize_t __real_pread64(int, void *, size_t, off64_t);
extern "C" int __wrap_fsync(int fd) {
	return Call(Fsync, "fsync", fd, -1, 0, -1, [&] { return __real_fsync(fd); });
}
extern "C" int __wrap_fdatasync(int fd) {
	return Call(Fdatasync, "fdatasync", fd, -1, 0, -1, [&] { return __real_fdatasync(fd); });
}
extern "C" int __wrap_flock(int fd, int mode) {
	return Call(Flock, "flock", fd, -1, 0, mode, [&] { return __real_flock(fd, mode); });
}
extern "C" ssize_t __wrap_pwrite(int fd, const void *data, size_t size, off_t offset) {
	return Call(Pwrite, "pwrite", fd, offset, int64_t(size), -1, [&] { return __real_pwrite(fd, data, size, offset); });
}
extern "C" ssize_t __wrap_pwrite64(int fd, const void *data, size_t size, off64_t offset) {
	return Call(Pwrite, "pwrite64", fd, offset, int64_t(size), -1,
	            [&] { return __real_pwrite64(fd, data, size, offset); });
}
extern "C" ssize_t __wrap_pread(int fd, void *data, size_t size, off_t offset) {
	return Call(Pread, "pread", fd, offset, int64_t(size), -1, [&] { return __real_pread(fd, data, size, offset); });
}
extern "C" ssize_t __wrap_pread64(int fd, void *data, size_t size, off64_t offset) {
	return Call(Pread, "pread64", fd, offset, int64_t(size), -1,
	            [&] { return __real_pread64(fd, data, size, offset); });
}
extern "C" int __real_sqlite3_step(sqlite3_stmt *);
extern "C" int __wrap_sqlite3_step(sqlite3_stmt *statement) {
	Span span(Step);
	auto rc = __real_sqlite3_step(statement);
	if (row)
		row->steps += sqlite3_stmt_status(statement, SQLITE_STMTSTATUS_VM_STEP, 1);
	return rc;
}
extern "C" int __real_sqlite3_exec(sqlite3 *, const char *, int (*)(void *, int, char **, char **), void *, char **);
extern "C" int __wrap_sqlite3_exec(sqlite3 *db, const char *sql, int (*callback)(void *, int, char **, char **),
                                   void *context, char **error) {
	Span span(Exec);
	return __real_sqlite3_exec(db, sql, callback, context, error);
}
extern "C" int __real_sqlite3_wal_checkpoint_v2(sqlite3 *, const char *, int, int *, int *);
extern "C" int __wrap_sqlite3_wal_checkpoint_v2(sqlite3 *db, const char *name, int mode, int *frames, int *copied) {
	if (!enabled)
		return __real_sqlite3_wal_checkpoint_v2(db, name, mode, frames, copied);
	Span span(Checkpoint);
	Event e {"checkpoint", "sqlite"};
	e.root = row ? row->id : -1;
	e.mode = mode;
	e.begin = Now(CLOCK_MONOTONIC);
	auto cpu = Now(CLOCK_THREAD_CPUTIME_ID);
	auto rc = __real_sqlite3_wal_checkpoint_v2(db, name, mode, frames, copied);
	auto saved = errno;
	e.cpu = Now(CLOCK_THREAD_CPUTIME_ID) - cpu;
	e.end = Now(CLOCK_MONOTONIC);
	span.Finish();
	e.result = rc;
	e.frames = frames ? *frames : -1;
	e.copied = copied ? *copied : -1;
	Record(e);
	errno = saved;
	return rc;
}
