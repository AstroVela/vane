// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "trace.hpp"
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <sqlite3.h>

namespace request_trace {
namespace {
constexpr size_t N = size_t(Stage::Count);
constexpr const char *NAMES[] = {"database_wait",
                                 "admission",
                                 "publication_wait",
                                 "payload_guard",
                                 "pins_wait",
                                 "begin",
                                 "commit",
                                 "statement_acquire",
                                 "statement_step",
                                 "statement_release",
                                 "exec",
                                 "writeback_error",
                                 "pause_enter",
                                 "pause_exit",
                                 "publication",
                                 "notify",
                                 "checkpoint",
                                 "wal_hook",
                                 "pread",
                                 "pwrite",
                                 "fsync",
                                 "fdatasync",
                                 "hint"};
static_assert(sizeof(NAMES) / sizeof(*NAMES) == N);
struct Row {
	const char *name;
	long long begin, end = 0, cpu_begin, cpu_end = 0;
	int64_t offset, bytes;
	uint64_t steps = 0;
	std::array<int, 4> wal {-1, -1, -1, 0};
	int committed_frames = -1;
	std::array<uint64_t, N> counts {}, ns {};
};
struct Thread {
	long tid = 0;
	std::vector<Row> rows;
};
Thread threads[64];
std::atomic<unsigned> next {0};
thread_local int index = -1;
thread_local Row *active = nullptr;
std::chrono::steady_clock::time_point origin;
long long epoch;
std::string output;
long long Cpu() {
	int saved = errno;
	struct timespec time {};
	if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time))
		std::abort();
	auto result = int64_t(time.tv_sec) * 1000000000 + time.tv_nsec;
	errno = saved;
	return result;
}
void Report() {
	std::ofstream file(output);
	file << "{\"origin_epoch_ns\":" << epoch << ",\"stages\":[";
	for (size_t i = 0; i < N; ++i) {
		if (i)
			file << ',';
		file << std::quoted(NAMES[i]);
	}
	file << "],\"columns\":[\"name\",\"begin_ns\",\"end_ns\",\"cpu_ns\",\"offset\",\"bytes\",\"vm_steps\",\"counts\","
	        "\"stage_ns\",\"last_checkpoint\",\"committed_frames\"],\"threads\":[";
	for (unsigned i = 0; i < next; ++i) {
		if (i)
			file << ',';
		file << "{\"tid\":" << threads[i].tid << ",\"rows\":[";
		bool first = true;
		for (const auto &row : threads[i].rows) {
			if (!first)
				file << ',';
			first = false;
			file << '[' << std::quoted(row.name) << ',' << row.begin << ',' << row.end << ','
			     << row.cpu_end - row.cpu_begin << ',' << row.offset << ',' << row.bytes << ',' << row.steps;
			for (auto values : {&row.counts, &row.ns}) {
				file << ",[";
				for (size_t j = 0; j < N; ++j) {
					if (j)
						file << ',';
					file << (*values)[j];
				}
				file << ']';
			}
			file << ",[" << row.wal[0] << ',' << row.wal[1] << ',' << row.wal[2] << ',' << row.wal[3] << "],"
			     << row.committed_frames << ']';
		}
		file << "]}";
	}
	file << "]}\n";
	if (!file)
		std::abort();
}
} // namespace
long long Now() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - origin).count();
}
void Add(Stage stage, long long begin) {
	if (!active)
		return;
	auto end = Now();
	++active->counts[size_t(stage)];
	active->ns[size_t(stage)] += end - begin;
}
void Steps(int value) {
	if (active && value >= 0)
		active->steps += value;
}
void Wal(int mode, int frames, int copied, int rc) {
	if (active)
		active->wal = {mode, frames, copied, rc};
}
void CommitFrames(int frames) {
	if (active)
		active->committed_frames = frames;
}
Scope::Scope(const char *name, int64_t offset, int64_t bytes) : previous(active) {
	if (index < 0) {
		index = int(next.fetch_add(1));
		if (index >= 64)
			std::abort();
		threads[index].tid = syscall(SYS_gettid);
		// Scope pointers survive child spans and possible nested scopes.
		threads[index].rows.reserve(65536);
	}
	auto &rows = threads[index].rows;
	if (rows.size() == rows.capacity())
		std::abort();
	rows.push_back({name, Now(), 0, Cpu(), 0, offset, bytes});
	record = active = &rows.back();
}
Scope::~Scope() {
	auto &row = *static_cast<Row *>(record);
	row.cpu_end = Cpu();
	row.end = Now();
	active = static_cast<Row *>(previous);
}
void Install(const char *database) {
	output = std::string(database) + ".request-costs.json";
	origin = std::chrono::steady_clock::now();
	epoch = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
	            .count();
	std::atexit(Report);
}
} // namespace request_trace

// Wrappers preserve the original syscall and errno. Stage times are inclusive:
// SQLite step/commit/admission can contain the measured I/O/checkpoint time.
extern "C" int __real_fsync(int);
extern "C" int __real_fdatasync(int);
extern "C" int __wrap_fsync(int fd) {
	request_trace::Span span(request_trace::Stage::Fsync);
	return __real_fsync(fd);
}
extern "C" int __wrap_fdatasync(int fd) {
	request_trace::Span span(request_trace::Stage::Fdatasync);
	return __real_fdatasync(fd);
}
extern "C" ssize_t __real_pread(int, void *, size_t, off_t);
extern "C" ssize_t __real_pwrite(int, const void *, size_t, off_t);
extern "C" ssize_t __real_pread64(int, void *, size_t, off64_t);
extern "C" ssize_t __real_pwrite64(int, const void *, size_t, off64_t);
extern "C" ssize_t __wrap_pread(int fd, void *data, size_t bytes, off_t offset) {
	request_trace::Span span(request_trace::Stage::Pread);
	return __real_pread(fd, data, bytes, offset);
}
extern "C" ssize_t __wrap_pwrite(int fd, const void *data, size_t bytes, off_t offset) {
	request_trace::Span span(request_trace::Stage::Pwrite);
	return __real_pwrite(fd, data, bytes, offset);
}
extern "C" ssize_t __wrap_pread64(int fd, void *data, size_t bytes, off64_t offset) {
	request_trace::Span span(request_trace::Stage::Pread);
	return __real_pread64(fd, data, bytes, offset);
}
extern "C" ssize_t __wrap_pwrite64(int fd, const void *data, size_t bytes, off64_t offset) {
	request_trace::Span span(request_trace::Stage::Pwrite);
	return __real_pwrite64(fd, data, bytes, offset);
}
extern "C" int __real_sqlite3_wal_checkpoint_v2(sqlite3 *, const char *, int, int *, int *);
extern "C" int __wrap_sqlite3_wal_checkpoint_v2(sqlite3 *db, const char *name, int mode, int *frames, int *copied) {
	request_trace::Span span(request_trace::Stage::Checkpoint);
	auto rc = __real_sqlite3_wal_checkpoint_v2(db, name, mode, frames, copied);
	request_trace::Wal(mode, frames ? *frames : -1, copied ? *copied : -1, rc);
	return rc;
}
extern "C" int __real_sync_file_range(int, off64_t, off64_t, unsigned);
extern "C" int __wrap_sync_file_range(int fd, off64_t offset, off64_t bytes, unsigned flags) {
	request_trace::Scope request("hint", offset, bytes);
	request_trace::Span span(request_trace::Stage::Hint);
	return __real_sync_file_range(fd, offset, bytes, flags);
}
