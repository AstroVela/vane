// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "prewriteback.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace vane_fs;
namespace fs = std::filesystem;
static constexpr int64_t MIB = 1024 * 1024;
static std::mutex calls_mutex;
struct Call {
	int64_t begin, end, offset, bytes;
	int error;
};
static std::vector<Call> calls;
static std::atomic<bool> hold {false}, entered {false}, lease_waiting {false};
static std::atomic<int> fault {0};
static int64_t Now() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}
static void Require(bool value, const char *message) {
	if (!value)
		throw std::runtime_error(message);
}
template <class F>
static void Wait(F test) {
	auto end = Now() + 10000000000;
	while (!test()) {
		Require(Now() < end, "Timed out waiting for worker");
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}
static std::vector<Call> Calls() {
	std::lock_guard<std::mutex> lock(calls_mutex);
	return calls;
}
extern "C" int __real_sync_file_range(int, off64_t, off64_t, unsigned);
extern "C" int __real_flock(int, int);
extern "C" int __wrap_flock(int fd, int flags) {
	auto rc = __real_flock(fd, flags), saved = errno;
	if (rc && (flags & LOCK_SH) && (saved == EAGAIN || saved == EWOULDBLOCK))
		lease_waiting = true;
	errno = saved;
	return rc;
}
extern "C" int __wrap_sync_file_range(int fd, off64_t offset, off64_t bytes, unsigned flags) {
	entered = true;
	auto begin = Now();
	while (hold)
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	int code = fault.exchange(0), rc;
	struct stat st {};
	if (flags != SYNC_FILE_RANGE_WRITE || bytes <= 0 || offset < 4096 || fstat(fd, &st) || bytes > st.st_size - offset)
		code = EINVAL;
	if (code) {
		rc = -1;
	} else {
		rc = __real_sync_file_range(fd, offset, bytes, flags);
		code = rc ? errno : 0;
	}
	{
		std::lock_guard<std::mutex> lock(calls_mutex);
		calls.push_back({begin, Now(), offset, bytes, code});
	}
	errno = code;
	return rc;
}
struct Fixture {
	std::string path;
	std::unique_ptr<PayloadFile> payload;
	std::unique_ptr<Prewriteback> worker;
	explicit Fixture(const fs::path &path) : path(path.string()) {
		int db = open(this->path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
		Require(db >= 0, "Creating descriptor fixture");
		close(db);
		calls.clear();
		hold = entered = lease_waiting = false;
		fault = 0;
		payload = std::make_unique<PayloadFile>(this->path, true, 1000);
		payload->Initialize(std::string(32, 'a'), true);
		Resize(128);
		worker = std::make_unique<Prewriteback>(this->path, 1000);
	}
	~Fixture() {
		hold = false;
		worker->Stop();
	}
	void Resize(int mib) {
		int fd = open((path + ".payload").c_str(), O_RDWR | O_CLOEXEC);
		Require(fd >= 0, "Opening fixture payload");
		auto rc = ftruncate(fd, mib * MIB + 4096);
		close(fd);
		Require(!rc, "Resizing payload fixture");
	}
	void Drain(size_t count) {
		Wait([&] { return Calls().size() >= count; });
		worker->Stop();
		Require(Calls().size() == count, "Unexpected submission count");
		for (const auto &call : Calls())
			Require(call.bytes <= 4 * MIB && call.bytes > 0, "Submission exceeds 4 MiB budget");
	}
};
static void Rate(const fs::path &path) {
	Fixture f(path);
	f.worker->OnCommit(63 * MIB, false, {4096, 63 * MIB + 4096});
	std::this_thread::sleep_for(std::chrono::milliseconds(40));
	Require(Calls().empty(), "Sub-threshold bytes woke worker");
	f.worker->OnCommit(64 * MIB, false, {63 * MIB + 4096, 64 * MIB + 4096});
	f.Drain(16);
	auto observed = Calls();
	for (size_t i = 0; i < observed.size(); ++i) {
		Require(!observed[i].error && observed[i].offset == 4096 + int64_t(i) * 4 * MIB && observed[i].bytes == 4 * MIB,
		        "Drain omitted or repeated bytes");
		if (i)
			Require(observed[i].begin - observed[i - 1].end >= 16000000, "Completion-to-start interval below 16 ms");
	}
}
static void Pauses(const fs::path &path) {
	Fixture f(path);
	{
		Prewriteback::Pause outer(f.worker.get());
		{
			Prewriteback::Pause inner(f.worker.get());
			f.worker->OnCommit(64 * MIB, false, {4096, 64 * MIB + 4096});
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		Require(Calls().empty(), "Nested pause ended too early");
		f.worker->Retire();
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(30));
	Require(Calls().empty(), "Retired hints were submitted");
	try {
		Prewriteback::Pause unwind(f.worker.get());
		f.worker->OnCommit(128 * MIB, false, {64 * MIB + 4096, 128 * MIB + 4096});
		throw std::runtime_error("unwind");
	} catch (const std::runtime_error &) {
	}
	Prewriteback::Pause disabled(f.worker.get(), false), absent(nullptr);
	f.Drain(16);
	Require(Calls().front().offset == 64 * MIB + 4096, "Retire lost later append range");
}
static void FullRace(const fs::path &path) {
	Fixture f(path);
	hold = true;
	f.worker->OnCommit(64 * MIB, false, {4096, 64 * MIB + 4096});
	Wait([] { return entered.load(); });
	{
		Prewriteback::Pause pause(f.worker.get());
		f.worker->OnCommit(64 * MIB, true, {});
		f.worker->OnCommit(128 * MIB, false, {64 * MIB + 4096, 128 * MIB + 4096});
		hold = false;
		Wait([] { return Calls().size() == 1; });
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		Require(Calls().size() == 1, "New claim started during FULL pause");
	}
	f.Drain(17);
	auto observed = Calls();
	Require(observed[0].offset == 4096 && observed[1].offset == 64 * MIB + 4096,
	        "Late completion resurrected retired bytes");
}
static void Retry(const fs::path &path, int code) {
	Fixture f(path);
	fault = code;
	f.worker->OnCommit(64 * MIB, false, {4096, 64 * MIB + 4096});
	f.Drain(17);
	auto observed = Calls();
	Require(observed[0].error == code && observed[0].offset == observed[1].offset &&
	            observed[0].bytes == observed[1].bytes,
	        "Failed range was not retried unchanged");
	bool reported = false;
	try {
		f.worker->CheckError();
	} catch (const Error &error) {
		reported = error.code == ErrorCode::Storage;
	}
	Require(reported == (code != EINTR), "Incorrect asynchronous error latch");
	f.worker->CheckError();
}
static void Lease(const fs::path &path) {
	Fixture f(path);
	int fd = open((f.path + ".payload").c_str(), O_RDWR | O_CLOEXEC);
	Require(fd >= 0 && !flock(fd, LOCK_EX | LOCK_NB), "Acquiring GC fixture lease");
	f.worker->OnCommit(64 * MIB, false, {4096, 64 * MIB + 4096});
	Wait([] { return lease_waiting.load(); });
	{
		Prewriteback::Pause pause(f.worker.get());
		Require(!ftruncate(fd, 16 * MIB + 4096), "Clipping GC fixture");
		flock(fd, LOCK_UN);
		close(fd);
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		Require(Calls().empty(), "Worker claimed after GC wait inside FULL pause");
	}
	f.Drain(4);
	f.Resize(128);
	f.worker->Start();
	f.worker->OnCommit(192 * MIB, false, {4096, 128 * MIB + 4096});
	f.Drain(36);
	Require(Calls()[4].offset == 4096 && Calls().back().offset == 124 * MIB + 4096, "GC regrowth skipped reused bytes");
}
static void Restart(const fs::path &path) {
	Fixture f(path);
	f.worker->OnCommit(64 * MIB, false, {4096, 64 * MIB + 4096});
	Wait([] { return !Calls().empty(); });
	f.worker->Stop();
	auto before = Calls();
	Require(before.size() < 16, "No pending bytes left for restart fixture");
	f.worker->Start();
	f.Drain(16);
	auto after = Calls();
	Require(after[before.size()].begin - before.back().end >= 16000000, "Restart reset completion deadline");
}
int main(int argc, char **argv) {
	if (argc != 2)
		return 2;
	try {
		fs::path root(argv[1]);
		Rate(root / "rate.sqlite");
		std::cout << "PASS bounded complete drain, threshold and pacing\n";
		Pauses(root / "pause.sqlite");
		std::cout << "PASS nested, disabled, absent and exception-unwound pauses; retirement\n";
		FullRace(root / "race.sqlite");
		std::cout << "PASS active FULL and later append generations\n";
		for (auto code : {EIO, ENOSPC, EINTR}) {
			Retry(root / ("retry-" + std::to_string(code) + ".sqlite"), code);
			std::cout << "PASS retry/error latch " << code << '\n';
		}
		Lease(root / "lease.sqlite");
		std::cout << "PASS pause after GC wait, EOF clipping and regrowth\n";
		Restart(root / "restart.sqlite");
		std::cout << "PASS pending work and deadline survive stop/restart\n";
	} catch (const std::exception &error) {
		hold = false;
		std::cerr << error.what() << '\n';
		return 1;
	}
}
