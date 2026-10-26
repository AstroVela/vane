// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <thread>
#include <cerrno>
#include <poll.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace vane_fs;
static constexpr size_t MIB = 1024 * 1024;
enum class Fault { None, Write, PartialWrite, Sync, Read, Punch };
static std::atomic<Fault> fault {Fault::None};
static std::atomic<bool> pause_read {false}, paused {false}, resume_read {false};
static bool short_io = false;
static int crash_point = 0, notification = -1, payload_syncs = 0;

static void Require(bool condition, const char *message) {
	if (!condition)
		throw std::runtime_error(message);
}
template <class F>
static void Expect(ErrorCode code, F operation) {
	try {
		operation();
	} catch (const Error &error) {
		Require(error.code == code, "Wrong error code");
		return;
	}
	throw std::runtime_error("Expected an error");
}
static bool Payload(int fd) {
	char target[4096];
	auto link = "/proc/self/fd/" + std::to_string(fd);
	auto n = readlink(link.c_str(), target, sizeof(target));
	return n >= 8 && std::string(target, size_t(n)).substr(size_t(n) - 8) == ".payload";
}
static bool Fail(Fault expected) {
	if (fault.compare_exchange_strong(expected, Fault::None)) {
		errno = EIO;
		return true;
	}
	return false;
}
static void Crash(int point) {
	if (crash_point != point)
		return;
	char ready = 'x';
	if (write(notification, &ready, 1) != 1)
		_exit(81);
	raise(SIGSTOP);
}
extern "C" ssize_t __real_pwrite(int, const void *, size_t, off_t);
extern "C" ssize_t __real_pread(int, void *, size_t, off_t);
extern "C" int __real_fdatasync(int);
extern "C" int __real_fallocate(int, int, off_t, off_t);
extern "C" ssize_t __wrap_pwrite(int fd, const void *data, size_t size, off_t offset) {
	if (Payload(fd)) {
		auto partial = Fault::PartialWrite;
		if (fault.compare_exchange_strong(partial, Fault::Write))
			return __real_pwrite(fd, data, std::min<size_t>(size, 4001), offset);
		if (Fail(Fault::Write))
			return -1;
		if (short_io)
			size = std::min<size_t>(size, 1021);
	}
	return __real_pwrite(fd, data, size, offset);
}
extern "C" ssize_t __wrap_pread(int fd, void *data, size_t size, off_t offset) {
	if (Payload(fd)) {
		if (Fail(Fault::Read))
			return -1;
		if (offset >= 4096 && pause_read.exchange(false)) {
			paused = true;
			while (!resume_read)
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		if (short_io)
			size = std::min<size_t>(size, 1021);
	}
	return __real_pread(fd, data, size, offset);
}
extern "C" int __wrap_fdatasync(int fd) {
	if (!Payload(fd))
		return __real_fdatasync(fd);
	++payload_syncs;
	if (Fail(Fault::Sync))
		return -1;
	Crash(1);
	auto rc = __real_fdatasync(fd);
	if (!rc)
		Crash(2);
	return rc;
}
extern "C" int __wrap_fallocate(int fd, int mode, off_t offset, off_t size) {
	if (Payload(fd)) {
		if (Fail(Fault::Punch))
			return -1;
		Crash(3);
	}
	return __real_fallocate(fd, mode, offset, size);
}

static void Failures(const std::string &path) {
	Workspace workspace(path, 100, Durability::Fsync);
	auto session = workspace.Checkout();
	short_io = true;
	session->WriteFile("/stable", std::string(MIB, 's'));
	Require(session->Read("/stable") == std::string(MIB, 's'), "Short I/O was not retried");
	short_io = false;
	for (auto value : {Fault::Write, Fault::PartialWrite, Fault::Sync}) {
		fault = value;
		Expect(ErrorCode::Storage, [&] { session->WriteFile("/failed", std::string(MIB, 'f')); });
		Require(fault == Fault::None, "Fault did not reach payload I/O");
		Expect(ErrorCode::NotFound, [&] { session->Stat("/failed"); });
		Require(session->Read("/stable") == std::string(MIB, 's'), "Failure changed durable data");
		session->WriteFile("/retry", std::string(MIB, 'r'));
		Require(session->Read("/retry") == std::string(MIB, 'r'), "Append after a failed/partial write differs");
		session->Unlink("/retry");
		workspace.CollectGarbage();
		Require(std::filesystem::file_size(path + ".payload") == MIB + 4096, "Failed append leaked after GC");
	}
	fault = Fault::Read;
	Expect(ErrorCode::Storage, [&] { session->Read("/stable"); });
	Require(session->Read("/stable") == std::string(MIB, 's'), "Read error was not retryable");
	auto before = payload_syncs;
	session->WriteFile("/middle", std::string(MIB, 'm'));
	Require(payload_syncs == before + 1, "Expected one data barrier per bulk transaction");
	before = payload_syncs;
	session->WriteFile("/small", std::string(4096, 'i'));
	Require(payload_syncs == before, "Small inline write synchronized the payload file");
	session->WriteFile("/tail", std::string(MIB, 't'));
	session->Unlink("/middle");
	fault = Fault::Punch;
	Expect(ErrorCode::Storage, [&] { workspace.CollectGarbage(); });
	Require(fault == Fault::None, "Hole-punch fault was not reached");
	Require(session->Read("/stable") == std::string(MIB, 's'), "Failed GC changed first extent");
	Require(session->Read("/tail") == std::string(MIB, 't'), "Failed GC changed last extent");
	workspace.CollectGarbage();
	struct stat st {};
	Require(stat((path + ".payload").c_str(), &st) == 0 && st.st_blocks * 512 < 3 * MIB,
	        "Retry failed to reclaim the dead middle extent");
	workspace.Sync();
}

static void ReaderAndGC(const std::string &path) {
	Workspace first(path, 20, Durability::Fsync), second(path, 20, Durability::Fsync);
	auto a = first.Checkout(), b = second.Checkout();
	a->WriteFile("/file", std::string(MIB, 'a'));
	std::exception_ptr failure;
	paused = false;
	resume_read = false;
	pause_read = true;
	std::thread reader([&] {
		try {
			Require(a->Read("/file") == std::string(MIB, 'a'), "Concurrent overwrite/GC changed an old reader");
		} catch (...) {
			failure = std::current_exception();
		}
	});
	try {
		auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
		while (!paused) {
			Require(std::chrono::steady_clock::now() < deadline, "Reader did not pause in the external extent");
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		b->WriteFile("/file", std::string(MIB, 'b'));
		Expect(ErrorCode::Busy, [&] { second.CollectGarbage(); });
	} catch (...) {
		resume_read = true;
		reader.join();
		throw;
	}
	resume_read = true;
	reader.join();
	if (failure)
		std::rethrow_exception(failure);
	second.CollectGarbage();
	Require(b->Read("/file") == std::string(MIB, 'b'), "GC lost the replacement");
}

static void CrashRecovery(const std::string &path, int point) {
	{
		Workspace workspace(path);
		auto session = workspace.Checkout();
		session->WriteFile("/stable", std::string(MIB, 's'));
		if (point == 3) {
			session->WriteFile("/dead", std::string(MIB, 'd'));
			session->WriteFile("/tail", std::string(MIB, 't'));
			session->Unlink("/dead");
		}
	}
	int pipefd[2];
	Require(pipe(pipefd) == 0, "Creating crash pipe");
	auto child = fork();
	Require(child >= 0, "Forking crash child");
	if (!child) {
		close(pipefd[0]);
		notification = pipefd[1];
		try {
			Workspace workspace(path, 100, Durability::Fsync);
			crash_point = point;
			if (point == 3)
				workspace.CollectGarbage();
			else
				workspace.Checkout()->WriteFile("/unpublished", std::string(MIB, 'u'));
		} catch (...) {
		}
		_exit(82);
	}
	close(pipefd[1]);
	pollfd waiting {pipefd[0], POLLIN, 0};
	char ready = 0;
	bool reached = poll(&waiting, 1, 15000) == 1 && read(pipefd[0], &ready, 1) == 1 && ready == 'x';
	close(pipefd[0]);
	kill(child, SIGKILL);
	int status = 0;
	waitpid(child, &status, 0);
	Require(reached && WIFSIGNALED(status), "Crash boundary was not reached");
	Workspace reopened(path);
	reopened.RecoverOwners();
	Require(reopened.Checkout()->Read("/stable") == std::string(MIB, 's'), "Crash lost existing data");
	Expect(ErrorCode::NotFound, [&] { reopened.Checkout()->Stat("/unpublished"); });
	if (point == 3)
		Require(reopened.Checkout()->Read("/tail") == std::string(MIB, 't'), "Interrupted sweep lost surviving data");
	reopened.CollectGarbage();
	if (point != 3)
		Require(std::filesystem::file_size(path + ".payload") == MIB + 4096, "Crash orphan was not reclaimed");
}

int main(int argc, char **argv) {
	try {
		Require(argc == 2, "Expected an empty owned test directory");
		auto root = std::filesystem::path(argv[1]);
		std::filesystem::create_directory(root);
		Failures((root / "failures.sqlite").string());
		ReaderAndGC((root / "reader.sqlite").string());
		for (int point : {1, 2, 3})
			CrashRecovery((root / ("crash-" + std::to_string(point) + ".sqlite")).string(), point);
		std::cout << "PASS: short I/O, errors/retry, read/GC exclusion and three SIGKILL boundaries\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
