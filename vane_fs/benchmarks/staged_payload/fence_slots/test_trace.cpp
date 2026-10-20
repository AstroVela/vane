// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "wait_trace.hpp"
#include <cerrno>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>

static int injected = 0;
static void Require(bool value) {
	if (!value)
		throw std::runtime_error("File-size recorder contract failed");
}
extern "C" int __real_fdatasync(int fd) {
	if (injected) {
		errno = injected;
		injected = 0;
		return -1;
	}
	return int(syscall(SYS_fdatasync, fd));
}
extern "C" int __wrap_fdatasync(int);
extern "C" ssize_t __wrap_pwrite(int, const void *, size_t, off_t);
int main(int argc, char **argv) {
	Require(argc == 2);
	fuse_waits::Install(argv[1]);
	auto fd = open((std::string(argv[1]) + ".stage-fence").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
	Require(fd >= 0);
	{
		fuse_waits::Root root("growth");
		errno = EDOM;
		Require(__wrap_pwrite(fd, "hello", 5, 4096) == 5 && errno == EDOM);
		Require(__wrap_fdatasync(fd) == 0 && errno == EDOM);
	}
	{
		fuse_waits::Root root("errors");
		injected = EIO;
		Require(__wrap_fdatasync(fd) == -1 && errno == EIO);
		injected = EINTR;
		Require(__wrap_fdatasync(fd) == -1 && errno == EINTR);
		Require(__wrap_fdatasync(-1) == -1 && errno == EBADF);
	}
	Require(close(fd) == 0);
}
