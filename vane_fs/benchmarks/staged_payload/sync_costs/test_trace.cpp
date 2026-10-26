// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "trace.hpp"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <string>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>

extern "C" int __wrap_fsync(int);
extern "C" int __wrap_fdatasync(int);
namespace {
thread_local int injected_error = 0;
int Call(int fd, long number) {
	if (injected_error) {
		errno = injected_error;
		injected_error = 0;
		return -1;
	}
	return int(syscall(number, fd));
}
void Require(bool condition) {
	if (!condition) {
		std::perror("sync recorder assertion");
		std::exit(1);
	}
}
int Open(const std::string &path) {
	auto fd = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
	Require(fd >= 0);
	Require(write(fd, "data", 4) == 4);
	return fd;
}
void Success(const char *name, int fd, bool data = true) {
	request_trace::Scope scope(name);
	request_trace::SyncSite site("success");
	errno = EDOM;
	Require((data ? __wrap_fdatasync(fd) : __wrap_fsync(fd)) == 0);
	Require(errno == EDOM);
}
} // namespace
// Call the real Linux syscall in the success case, with explicit failure
// injection for the wrapper contract. Only the other I/O symbols are wrapped
// by the test linker; these two entry points are supplied here.
extern "C" int __real_fsync(int fd) {
	return Call(fd, SYS_fsync);
}
extern "C" int __real_fdatasync(int fd) {
	return Call(fd, SYS_fdatasync);
}
int main(int argc, char **argv) {
	Require(argc == 2);
	std::string path = argv[1];
	request_trace::Install(argv[1]);
	int database = Open(path), payload = Open(path + ".payload"), wal = Open(path + "-wal"),
	    fence = Open(path + ".stage-fence"), other = Open(path + ".payload-copy");
	int directory = open(path.substr(0, path.rfind('/')).c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
	Require(directory >= 0);
	// Startup and shutdown calls have no active request, but remain attributable.
	Require(__wrap_fdatasync(payload) == 0);
	Success("database_fsync", database, false);
	Success("database_fdatasync", database);
	Success("payload_fsync", payload, false);
	Success("payload_fdatasync", payload);
	Success("wal_fsync", wal, false);
	Success("wal_fdatasync", wal);
	Success("fence_fsync", fence, false);
	Success("fence_fdatasync", fence);
	Success("directory_fsync", directory, false);
	Success("directory_fdatasync", directory);
	Success("other_fsync", other, false);
	Success("other_fdatasync", other);
	{
		request_trace::Scope scope("error_payload");
		injected_error = EIO;
		Require(__wrap_fdatasync(payload) == -1 && errno == EIO);
	}
	{
		request_trace::Scope scope("eintr_wal");
		injected_error = EINTR;
		Require(__wrap_fsync(wal) == -1 && errno == EINTR);
		Require(__wrap_fsync(wal) == 0);
	}
	{
		request_trace::Scope scope("invalid_other");
		Require(__wrap_fdatasync(-1) == -1 && errno == EBADF);
	}
	{
		request_trace::Scope outer("nested_payload");
		request_trace::SyncSite site("nested_outer");
		Require(__wrap_fdatasync(payload) == 0);
		Success("nested_fence", fence);
		Require(__wrap_fdatasync(payload) == 0);
	}
	std::thread a([&] { Success("thread_payload", payload); });
	std::thread b([&] { Success("thread_fence", fence); });
	a.join();
	b.join();
	Require(unlink((path + ".payload").c_str()) == 0);
	Success("deleted_payload", payload);
	// The same descriptor number now refers to the fence, without cached state.
	Require(dup2(fence, payload) == payload);
	Success("reused_fence", payload);
	Require(__wrap_fsync(database) == 0);
	for (int fd : {database, payload, wal, fence, other, directory})
		Require(close(fd) == 0);
	std::puts("sync recorder contract PASS");
}
