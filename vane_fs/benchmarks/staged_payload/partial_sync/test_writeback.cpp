// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Separate fault/control executable; never used for performance measurements.
#include "vane_fs/workspace.hpp"
#include <atomic>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>

using namespace vane_fs;
static std::atomic<int> calls {0}, interrupted {0}, payload_syncs {0}, wal_syncs {0};
static std::atomic<int> watched {-1};
static bool watched_closed = false, fail_barrier = false;
static int hint_error = 0;
static std::string Name(int fd) {
	char text[4096];
	auto link = "/proc/self/fd/" + std::to_string(fd);
	auto n = readlink(link.c_str(), text, sizeof(text));
	return n > 0 ? std::string(text, size_t(n)) : "";
}
static bool Ends(const std::string &name, const char *suffix) {
	auto n = std::strlen(suffix);
	return name.size() >= n && name.compare(name.size() - n, n, suffix) == 0;
}
static void Require(bool value, const char *message) {
	if (!value)
		throw std::runtime_error(message);
}
extern "C" int __wrap_sync_file_range(int fd, off64_t offset, off64_t bytes, unsigned flags) {
	Require(Ends(Name(fd), ".payload"), "Writeback must target payload");
	Require(flags == SYNC_FILE_RANGE_WRITE, "Writeback must never add WAIT flags");
	Require(offset >= 4096 && offset % 4096 == 0 && bytes >= 16 * 1024 * 1024, "Bounded writeback range");
	++calls;
	watched = fd;
	Require(!watched_closed, "A new hint closed its error-tracking description");
	if (interrupted.exchange(0)) {
		errno = EINTR;
		return -1;
	}
	if (hint_error) {
		errno = hint_error;
		return -1;
	}
	return 0; // Submission fault model; the actual data barrier still runs.
}
extern "C" int __real_fdatasync(int);
extern "C" int __wrap_fdatasync(int fd) {
	if (watched >= 0 && Ends(Name(fd), ".payload")) {
		Require(fd == watched && !watched_closed, "Publication reopened after asynchronous writeback");
		++payload_syncs;
		if (fail_barrier) {
			fail_barrier = false;
			errno = EIO;
			return -1;
		}
	}
	return __real_fdatasync(fd);
}
extern "C" int __real_fsync(int);
extern "C" int __wrap_fsync(int fd) {
	if (watched >= 0 && Ends(Name(fd), "-wal")) {
		Require(payload_syncs > 0, "WAL barrier must follow payload barrier");
		++wal_syncs;
	}
	return __real_fsync(fd);
}
extern "C" int __real_close(int);
extern "C" int __wrap_close(int fd) {
	if (watched >= 0 && fd == watched)
		watched_closed = true;
	return __real_close(fd);
}
template <class F>
static void ExpectStorage(F action) {
	try {
		action();
	} catch (const Error &e) {
		Require(e.code == ErrorCode::Storage, "Wrong storage error");
		return;
	}
	throw std::runtime_error("Expected storage error");
}
int main(int argc, char **argv) {
	try {
		Require(argc == 3, "Expected fresh database and case");
		std::string mode(argv[2]);
		hint_error = mode == "unsupported"     ? ENOSYS
		             : mode == "invalid"       ? EINVAL
		             : mode == "not_supported" ? EOPNOTSUPP
		             : mode == "io_error"      ? EIO
		             : mode == "no_space"      ? ENOSPC
		                                       : 0;
		interrupted = mode == "interrupt" ? 1 : 0;
		fail_barrier = mode == "barrier_error";
		std::string bytes(32 * 1024 * 1024, 'a');
		std::string expected = "s" + bytes;
		{
			Workspace ws(argv[1]);
			auto session = ws.Checkout();
			session->WriteFile("/file", "stable");
			auto saved = ws.Snapshot();
			if (mode == "small") {
				session->WriteFile("/small", std::string(1024 * 1024, 's'));
				Require(calls == 0, "Small external writes must not submit hints");
			} else if (mode == "io_error" || mode == "no_space" || mode == "barrier_error") {
				ExpectStorage([&] { session->Write("/file", bytes, 1); });
				Require(session->Read("/file") == "stable", "Failed write changed content");
				hint_error = 0;
				session->Write("/file", bytes, 1);
				Require(session->Read("/file") == expected, "Retry content");
			} else {
				session->Write("/file", bytes, 1);
				Require(calls >= 1, "Expected early payload writeback");
				if (mode == "unsupported" || mode == "invalid" || mode == "not_supported")
					Require(calls == 1, "Unsupported hint must be disabled");
				if (mode == "interrupt")
					Require(calls >= 2, "Interrupted submission must retry");
				Require(session->Read("/file") == expected, "Published content");
			}
			auto frozen = ws.OpenSnapshot(saved);
			Require(frozen->Read("/file") == "stable", "Snapshot content");
			frozen->Close();
			session->Close();
			ws.Close();
			if (mode != "small")
				Require(payload_syncs >= 1 && wal_syncs >= 1 && !watched_closed,
				        "Durability ordering and live descriptor");
		}
		Require(watched < 0 || watched_closed, "Descriptor leak");
		// Reopen is a separate error-observation lifetime.
		watched = -1;
		Workspace reopened(argv[1]);
		Require(reopened.Checkout()->Read("/file") == (mode == "small" ? "stable" : expected), "Reopen content");
		reopened.Close();
		std::cout << "{\"case\":\"" << mode << "\",\"hint_calls\":" << calls << ",\"content_verified\":true}\n";
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return 1;
	}
}
