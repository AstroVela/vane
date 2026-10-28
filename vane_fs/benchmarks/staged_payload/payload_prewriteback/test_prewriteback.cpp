// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "staging.hpp"
#include <atomic>
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <sys/wait.h>

using namespace vane_fs;
namespace fs = std::filesystem;
static constexpr int64_t MIB = 1024 * 1024;
static thread_local bool application = false, collecting = false;
static std::atomic<bool> hold {false}, entered {false}, fail_once {false}, gc_waiting {false};
static std::atomic<int> background {0}, foreground {0}, injected {0};

static void Require(bool value, const char *message) {
	if (!value)
		throw std::runtime_error(message);
}
template <class F>
static bool Until(F condition, int milliseconds = 10000) {
	auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(milliseconds);
	while (!condition()) {
		if (std::chrono::steady_clock::now() >= end)
			return false;
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	return true;
}
template <class F>
static void Wait(F condition) {
	Require(Until(condition), "Timed out waiting for payload prewriteback");
}
template <class F>
static void Storage(F operation) {
	try {
		operation();
	} catch (const Error &error) {
		Require(error.code == ErrorCode::Storage, error.what());
		return;
	}
	throw std::runtime_error("Expected storage failure");
}
static std::string Name(int fd) {
	char name[4096];
	auto link = "/proc/self/fd/" + std::to_string(fd);
	auto n = readlink(link.c_str(), name, sizeof(name));
	return n > 0 ? std::string(name, size_t(n)) : "";
}
static bool Payload(int fd) {
	auto name = Name(fd);
	return name.size() >= 8 && name.compare(name.size() - 8, 8, ".payload") == 0;
}
extern "C" int __real_fdatasync(int);
extern "C" int __real_flock(int, int);
extern "C" int __wrap_fdatasync(int fd) {
	bool payload = Payload(fd);
	if (payload && !application) {
		entered = true;
		while (hold)
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		if (fail_once.exchange(false)) {
			++injected;
			errno = EIO;
			return -1;
		}
	}
	auto code = __real_fdatasync(fd);
	if (!code && payload)
		++(application ? foreground : background);
	return code;
}
extern "C" int __wrap_flock(int fd, int mode) {
	auto code = __real_flock(fd, mode);
	auto saved = errno;
	if (collecting && code && (mode & LOCK_EX) && (saved == EWOULDBLOCK || saved == EAGAIN) && Payload(fd))
		gc_waiting = true;
	errno = saved;
	return code;
}
struct Resume {
	~Resume() {
		hold = false;
	}
};
static bool Exclusive(const std::string &path) {
	int fd = open((path + ".payload").c_str(), O_RDWR | O_CLOEXEC);
	Require(fd >= 0, "Opening lease probe");
	auto code = flock(fd, LOCK_EX | LOCK_NB);
	if (!code)
		flock(fd, LOCK_UN);
	close(fd);
	return !code;
}
static void SQL(const std::string &path, const char *text, bool small_wal = false) {
	sqlite3 *db = nullptr;
	Require(sqlite3_open(path.c_str(), &db) == SQLITE_OK, "Opening inspector");
	Staging::Configure(db);
	sqlite3_busy_timeout(db, 5000);
	auto code = sqlite3_exec(db, text, nullptr, nullptr, nullptr);
	if (small_wal) {
		int frames = -1, copied = -1;
		Require(sqlite3_wal_checkpoint_v2(db, "main", SQLITE_CHECKPOINT_NOOP, &frames, &copied) == SQLITE_OK &&
		            frames >= 0 && int64_t(frames) * 4120 < 16 * MIB,
		        "Fixture reached WAL publication threshold");
	}
	sqlite3_close(db);
	Require(code == SQLITE_OK, "Inspector/fault SQL failed");
}
static bool Pending(Staging &observer) {
	Staging::Guard held(observer);
	return observer.Pending();
}
static void Quiet(int before) {
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	Require(background == before, "Sub-threshold or aborted operation woke prewriteback");
}
static void Unlocked(const std::string &path) {
	Workspace w(path, 5000, Durability::Fsync);
	auto s = w.Checkout();
	s->WriteFile("/file", "");
	w.Sync();
	Staging observer(path, 100);
	observer.Initialize(w.Id(), false);
	observer.Ready();
	int before = background;
	entered = false;
	hold = true;
	Resume resume;
	s->Write("/file", std::string(63 * MIB, 'a'));
	Quiet(before);
	s->Write("/file", std::string(MIB, 'b'), 63 * MIB);
	SQL(path, "SELECT count(*) FROM sqlite_master", true);
	Wait([] { return entered.load(); });
	Require(Pending(observer), "Prewriteback prematurely published metadata");
	Require(!Exclusive(path), "Prewriteback omitted its shared payload lease");
	std::atomic<bool> done {false};
	std::exception_ptr error;
	std::thread writer([&] {
		application = true;
		try {
			s->Write("/file", std::string(MIB, 'c'), 64 * MIB);
		} catch (...) {
			error = std::current_exception();
		}
		done = true;
	});
	bool progressed = Until([&] { return done.load(); }, 2000);
	bool still_leased = !Exclusive(path);
	hold = false;
	writer.join();
	if (error)
		std::rethrow_exception(error);
	Require(progressed, "Prewriteback held a checkpoint/publication/database mutex across I/O");
	Require(still_leased, "Writer unlock released another thread's payload lease");
	Wait([&] { return background > before; });
	Require(Pending(observer), "Advisory flush changed OPEN to CLEAN");
	auto barriers = foreground.load();
	w.Sync();
	Require(foreground == barriers + 1 && !Pending(observer), "FULL barrier trusted advisory progress");
	Require(s->Read("/file", 64 * MIB) == std::string(MIB, 'c'), "Concurrent append differs");
	w.Close();
	std::cout << "PASS threshold, writer progress, independent lease, OPEN and authoritative FULL\n";
}
static void Reset(const std::string &path) {
	Workspace w(path, 5000, Durability::Fsync);
	auto s = w.Checkout();
	int before = background;
	s->WriteFile("/file", std::string(64 * MIB, 'a'));
	Wait([&] { return background > before; });
	s->Write("/file", std::string(32 * MIB, 'b'), 64 * MIB);
	w.Sync();
	s->Write("/file", std::string(63 * MIB, 'c'), 96 * MIB);
	Quiet(before + 1);
	s->Write("/file", std::string(MIB, 'd'), 159 * MIB);
	Wait([&] { return background > before + 1; });
	w.Close();
	Workspace reopened(path);
	Require(reopened.Checkout()->Read("/file", 159 * MIB) == std::string(MIB, 'd'), "Reset/reopen lost data");
	std::cout << "PASS repeated budget, explicit-FULL reset and reopen\n";
}
static void FullRace(const std::string &path) {
	Workspace w(path, 5000, Durability::Fsync);
	auto s = w.Checkout();
	Staging observer(path, 5000);
	observer.Initialize(w.Id(), false);
	observer.Ready();
	int before = background;
	entered = false;
	hold = true;
	Resume resume;
	s->WriteFile("/file", std::string(64 * MIB, 'a'));
	Wait([] { return entered.load(); });
	std::atomic<bool> done {false};
	std::exception_ptr error;
	std::thread barrier([&] {
		application = true;
		try {
			w.Sync();
		} catch (...) {
			error = std::current_exception();
		}
		done = true;
	});
	bool progressed = Until([&] { return done.load(); }, 2000);
	if (!progressed)
		hold = false;
	barrier.join();
	if (error)
		std::rethrow_exception(error);
	Require(progressed && !Pending(observer), "FULL barrier waited on the advisory worker's mutex");
	s->Write("/file", std::string(64 * MIB, 'b'), 64 * MIB);
	hold = false;
	Wait([&] { return background >= before + 2; });
	Require(Pending(observer), "Late commit was marked durable by an older preflush");
	w.Sync();
	Require(s->Read("/file", 127 * MIB) == std::string(MIB, 'b'), "Concurrent FULL/late append differs");
	w.Close();
	std::cout << "PASS FULL during preflush, monotonic completion and late-commit wake\n";
}
static void Failure(const std::string &path, bool closing) {
	Workspace w(path, 5000, Durability::Fsync);
	auto s = w.Checkout();
	int before = background, faults = injected;
	fail_once = true;
	s->WriteFile("/file", std::string(64 * MIB, 'f'));
	Wait([&] { return injected > faults && background > before; });
	if (closing)
		Storage([&] { w.Close(); });
	else
		Storage([&] { w.Sync(); });
	Require(s->Read("/file", 63 * MIB) == std::string(MIB, 'f'), "Failed maintenance lost visible bytes");
	w.Sync();
	w.Close();
	Workspace reopened(path);
	Require(reopened.Checkout()->Read("/file", 63 * MIB) == std::string(MIB, 'f'), "Failure retry lost durable bytes");
	std::cout << "PASS asynchronous error/retry and failed " << (closing ? "close" : "barrier") << '\n';
}
static void Rollback(const std::string &path) {
	Workspace w(path, 5000, Durability::Fsync);
	auto s = w.Checkout();
	SQL(path, "CREATE TRIGGER fail AFTER INSERT ON block_payloads WHEN new.id=256 "
	          "BEGIN SELECT RAISE(ABORT,'injected'); END");
	int before = background;
	Storage([&] { s->WriteFile("/aborted", std::string(64 * MIB, 'x')); });
	Require(fs::file_size(path + ".payload") == 64 * MIB + 4096, "Rollback did not reach external appends");
	Quiet(before);
	SQL(path, "DROP TRIGGER fail");
	s->WriteFile("/good", std::string(MIB, 'g'));
	Wait([&] { return background > before; });
	try {
		s->Stat("/aborted");
		throw std::runtime_error("Aborted metadata became visible");
	} catch (const Error &e) {
		Require(e.code == ErrorCode::NotFound, e.what());
	}
	w.CollectGarbage();
	Require(s->Read("/good") == std::string(MIB, 'g'), "GC after rollback damaged data");
	w.Close();
	std::cout << "PASS no rollback notification, conservative bytes and GC\n";
}
static void Garbage(const std::string &path) {
	Workspace w(path, 5000, Durability::Fsync), collector(path, 5000, Durability::Fsync);
	auto s = w.Checkout();
	entered = gc_waiting = false;
	hold = true;
	Resume resume;
	s->WriteFile("/file", std::string(64 * MIB, 's'));
	Wait([] { return entered.load(); });
	auto snapshot = w.Snapshot();
	auto frozen = w.OpenSnapshot(snapshot);
	std::atomic<bool> done {false};
	std::exception_ptr error;
	std::thread gc([&] {
		application = collecting = true;
		try {
			collector.CollectGarbage();
		} catch (...) {
			error = std::current_exception();
		}
		done = true;
	});
	bool waited = Until([] { return gc_waiting.load(); }, 2000);
	bool blocked = !done;
	hold = false;
	gc.join();
	if (error)
		std::rethrow_exception(error);
	Require(waited && blocked, "GC did not wait for the independent prewriteback lease");
	s->Unlink("/file");
	w.Sync();
	collector.CollectGarbage();
	Require(frozen->Read("/file", 63 * MIB) == std::string(MIB, 's'), "GC lost pinned snapshot bytes");
	frozen->Close();
	w.DropSnapshot(snapshot);
	collector.CollectGarbage();
	Require(fs::file_size(path + ".payload") == 4096, "GC did not reclaim the unused tail");
	int before = background;
	s->WriteFile("/reused", std::string(64 * MIB, 'r'));
	Wait([&] { return background > before; });
	w.Sync();
	Require(s->Read("/reused", 63 * MIB) == std::string(MIB, 'r'), "Tail reuse lost prewriteback progress");
	w.Close();
	collector.Close();
	std::cout << "PASS GC exclusion, pinned snapshots and monotonic budget after tail reuse\n";
}
static void Forked(const std::string &path) {
	Workspace w(path, 5000, Durability::Fsync);
	entered = false;
	hold = true;
	Resume resume;
	w.Checkout()->WriteFile("/file", std::string(64 * MIB, 'p'));
	Wait([] { return entered.load(); });
	auto child = fork();
	Require(child >= 0, "Forking inherited-owner fixture");
	if (!child) {
		try {
			w.Close();
			_exit(0);
		} catch (...) {
			_exit(2);
		}
	}
	int status = 0;
	bool exited = Until([&] { return waitpid(child, &status, WNOHANG) == child; }, 2000);
	if (!exited) {
		kill(child, SIGKILL);
		waitpid(child, &status, 0);
	}
	bool leased = !Exclusive(path);
	hold = false;
	Require(exited && WIFEXITED(status) && WEXITSTATUS(status) == 0,
	        "Inherited worker was joined or destroyed after fork");
	Require(leased, "Fork child unlocked the parent's prewriteback lease");
	w.Sync();
	w.Close();
	std::cout << "PASS fork abandonment preserves the parent's lease\n";
}
static void Strict(const std::string &path) {
	int before = background;
	Workspace w(path, 5000, Durability::Strict);
	w.Checkout()->WriteFile("/file", std::string(64 * MIB, 's'));
	Quiet(before);
	w.Close();
	std::cout << "PASS strict mode has no prewriteback worker\n";
}
int main(int argc, char **argv) {
	if (argc != 2)
		return 2;
	application = true;
	try {
		auto root = fs::path(argv[1]);
		Unlocked((root / "unlocked.sqlite").string());
		Reset((root / "reset.sqlite").string());
		FullRace((root / "full-race.sqlite").string());
		Failure((root / "sync-failure.sqlite").string(), false);
		Failure((root / "close-failure.sqlite").string(), true);
		Rollback((root / "rollback.sqlite").string());
		Garbage((root / "gc.sqlite").string());
		Forked((root / "fork.sqlite").string());
		Strict((root / "strict.sqlite").string());
		std::cout << "PASS nine advisory prewriteback cases\n";
	} catch (const std::exception &error) {
		hold = false;
		std::cerr << error.what() << '\n';
		return 1;
	}
}
