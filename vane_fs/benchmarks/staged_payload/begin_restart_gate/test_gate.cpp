// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

using namespace vane_fs;
static thread_local bool application = false, in_begin = false, began = false;
static thread_local int checkpoint_mode = -1;
enum class Pause { None, PassiveIO, RestartLock, PassiveDone, BeforeBegin, AfterBegin };
static std::atomic<Pause> background_pause {Pause::None}, application_pause {Pause::None};
static std::atomic<bool> background_entered {false}, application_entered {false};
static std::atomic<bool> armed {false}, fail_begin {false};
static std::atomic<int> begin_calls {0}, busy_sleeps {0}, restarts {0};

static void Require(bool value, const char *message) {
	if (!value)
		throw std::runtime_error(message);
}
template <class F>
static void Wait(F condition) {
	auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
	while (!condition()) {
		Require(std::chrono::steady_clock::now() < deadline, "Timed out at controlled checkpoint/BEGIN boundary");
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}
static void Hold(Pause point, bool foreground) {
	auto &wanted = foreground ? application_pause : background_pause;
	if (wanted != point)
		return;
	(foreground ? application_entered : background_entered) = true;
	Wait([&] { return wanted != point; });
}
extern "C" int __real_nanosleep(const timespec *, timespec *);
extern "C" int __wrap_nanosleep(const timespec *request, timespec *remaining) {
	if (in_begin)
		++busy_sleeps;
	return __real_nanosleep(request, remaining);
}
extern "C" int __real_sqlite3_step(sqlite3_stmt *);
extern "C" int __wrap_sqlite3_step(sqlite3_stmt *statement) {
	auto sql = sqlite3_sql(statement);
	auto begin = application && armed && sql && !std::strcmp(sql, "BEGIN IMMEDIATE");
	if (begin) {
		++begin_calls;
		Hold(Pause::BeforeBegin, true);
		if (fail_begin.exchange(false))
			return SQLITE_BUSY;
	} else if (application && armed && began) {
		began = false;
		Hold(Pause::AfterBegin, true);
	}
	in_begin = begin;
	auto code = __real_sqlite3_step(statement);
	in_begin = false;
	if (begin && code == SQLITE_DONE)
		began = true;
	return code;
}
extern "C" int __real_sqlite3_wal_checkpoint_v2(sqlite3 *, const char *, int, int *, int *);
extern "C" int __wrap_sqlite3_wal_checkpoint_v2(sqlite3 *db, const char *name, int mode, int *frames, int *copied) {
	if (!application && mode == SQLITE_CHECKPOINT_RESTART)
		++restarts;
	checkpoint_mode = mode;
	auto code = __real_sqlite3_wal_checkpoint_v2(db, name, mode, frames, copied);
	checkpoint_mode = -1;
	if (!application && mode == SQLITE_CHECKPOINT_PASSIVE && code == SQLITE_OK)
		Hold(Pause::PassiveDone, false);
	return code;
}

struct FileHooks {
	sqlite3_io_methods methods;
	const sqlite3_io_methods *original;
	int flags;
};
static std::map<sqlite3_file *, FileHooks> files;
static std::mutex files_mutex;
static sqlite3_vfs *original_vfs;
static FileHooks Original(sqlite3_file *file) {
	std::lock_guard<std::mutex> lock(files_mutex);
	return files.at(file);
}
static int Write(sqlite3_file *file, const void *data, int size, sqlite3_int64 offset) {
	auto hook = Original(file);
	if (!application && checkpoint_mode == SQLITE_CHECKPOINT_PASSIVE && (hook.flags & SQLITE_OPEN_MAIN_DB))
		Hold(Pause::PassiveIO, false);
	return hook.original->xWrite(file, data, size, offset);
}
static int ShmLock(sqlite3_file *file, int offset, int count, int flags) {
	auto code = Original(file).original->xShmLock(file, offset, count, flags);
	// WAL_WRITE_LOCK is byte zero. Hold AFTER the real VFS grants it, so the
	// unchanged parent genuinely enters SQLite's default busy handler here.
	if (!application && checkpoint_mode == SQLITE_CHECKPOINT_RESTART && code == SQLITE_OK && offset == 0 &&
	    count == 1 && (flags & SQLITE_SHM_LOCK) && (flags & SQLITE_SHM_EXCLUSIVE))
		Hold(Pause::RestartLock, false);
	return code;
}
static int CloseFile(sqlite3_file *file) {
	{
		std::lock_guard<std::mutex> lock(files_mutex);
		file->pMethods = files.at(file).original;
		files.erase(file);
	}
	return file->pMethods->xClose(file);
}
static int Open(sqlite3_vfs *, const char *name, sqlite3_file *file, int flags, int *output_flags) {
	auto code = original_vfs->xOpen(original_vfs, name, file, flags, output_flags);
	if (file->pMethods) {
		std::lock_guard<std::mutex> lock(files_mutex);
		auto &hook = files.emplace(file, FileHooks {*file->pMethods, file->pMethods, flags}).first->second;
		hook.methods.xWrite = Write;
		hook.methods.xShmLock = ShmLock;
		hook.methods.xClose = CloseFile;
		file->pMethods = &hook.methods;
	}
	return code;
}
struct Hooks {
	sqlite3_vfs vfs;
	Hooks() {
		original_vfs = sqlite3_vfs_find(nullptr);
		vfs = *original_vfs;
		vfs.zName = "begin-restart-gate-test";
		vfs.xOpen = Open;
		Require(sqlite3_vfs_register(&vfs, 1) == SQLITE_OK, "Installing controlled VFS");
	}
	~Hooks() {
		sqlite3_vfs_unregister(&vfs);
	}
};
struct Pager {
	sqlite3 *db = nullptr;
	explicit Pager(const std::string &path) {
		Require(sqlite3_open(path.c_str(), &db) == SQLITE_OK, "Opening fixture pager");
		sqlite3_db_config(db, SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, 1, nullptr);
		SQL("PRAGMA wal_autocheckpoint=0; PRAGMA synchronous=NORMAL");
	}
	~Pager() {
		sqlite3_close_v2(db);
	}
	void SQL(const char *sql) {
		Require(sqlite3_exec(db, sql, nullptr, nullptr, nullptr) == SQLITE_OK, sqlite3_errmsg(db));
	}
};
struct Fixture {
	std::string path;
	Workspace workspace;
	std::shared_ptr<Session> session;
	explicit Fixture(const std::string &path)
	    : path(path), workspace(path, 1000, Durability::Fsync), session(workspace.Checkout()) {
		session->WriteFile("/file", "seed");
	}
	~Fixture() {
		background_pause = application_pause = Pause::None;
		armed = false;
	}
	void Warm(Pause pause) {
		background_pause = pause;
		background_entered = false;
		Pager pager(path);
		pager.SQL("CREATE TABLE gate_padding(data BLOB); INSERT INTO gate_padding VALUES(zeroblob(18874368))");
		session->Write("/file", "warm");
		Wait([] { return background_entered.load(); });
	}
	bool LeaseAvailable() {
		int fd = open((path + ".stage-checkpoint").c_str(), O_RDWR | O_CLOEXEC);
		Require(fd >= 0, "Opening checkpoint lease");
		bool acquired = !flock(fd, LOCK_EX | LOCK_NB);
		if (acquired)
			flock(fd, LOCK_UN);
		close(fd);
		return acquired;
	}
};
struct Operation {
	std::atomic<bool> done {false};
	std::exception_ptr error;
	std::thread thread;
	template <class F>
	explicit Operation(F function)
	    : thread([this, function] {
		      application = true;
		      try {
			      function();
		      } catch (...) {
			      error = std::current_exception();
		      }
		      done = true;
	      }) {
	}
	~Operation() {
		background_pause = application_pause = Pause::None;
		if (thread.joinable())
			thread.join();
	}
	void Finish() {
		Wait([&] { return done.load(); });
		thread.join();
		if (error)
			std::rethrow_exception(error);
	}
};
static void Reset() {
	armed = fail_begin = background_entered = application_entered = false;
	background_pause = application_pause = Pause::None;
	begin_calls = busy_sleeps = restarts = 0;
	began = false;
}
static void HeldRestart(const std::string &path, bool control, bool full) {
	Reset();
	Fixture fixture(path);
	fixture.Warm(Pause::RestartLock);
	armed = true;
	Operation writer([&] {
		if (full)
			fixture.workspace.Sync();
		else
			fixture.session->Write("/file", "next");
	});
	if (control) {
		Wait([] { return begin_calls > 0 && busy_sleeps > 0; });
	} else {
		std::this_thread::sleep_for(std::chrono::milliseconds(60));
		Require(begin_calls == 0 && busy_sleeps == 0 && !writer.done,
		        "BEGIN entered busy handling under the owner's held RESTART");
	}
	background_pause = Pause::None;
	writer.Finish();
	Require(begin_calls > 0, "Writer did not enter BEGIN after RESTART release");
	if (!control)
		Require(busy_sleeps == 0, "Own RESTART caused a SQLite retry sleep");
	fixture.workspace.Sync();
	Require(fixture.session->Read("/file") == (full ? "warm" : "next"), "Held RESTART changed file bytes");
	std::cout << "PASS " << (control ? "parent reproduces busy sleeps" : "gate excludes BEGIN during real RESTART")
	          << (full ? " FULL" : " NORMAL") << " begin_calls=" << begin_calls << " busy_sleeps=" << busy_sleeps
	          << '\n';
}
static void PassiveConcurrent(const std::string &path) {
	Reset();
	Fixture fixture(path);
	fixture.Warm(Pause::PassiveIO);
	armed = true;
	Operation writer([&] { fixture.session->Write("/file", "next"); });
	writer.Finish();
	Require(background_pause == Pause::PassiveIO && busy_sleeps == 0, "Writer waited for PASSIVE database I/O");
	background_pause = Pause::None;
	fixture.workspace.Sync();
	std::cout << "PASS foreground write during real blocked PASSIVE I/O\n";
}
static void BeginBoundary(const std::string &path, bool after) {
	Reset();
	Fixture fixture(path);
	fixture.Warm(Pause::PassiveDone);
	armed = true;
	application_pause = after ? Pause::AfterBegin : Pause::BeforeBegin;
	Operation writer([&] { fixture.session->Write("/file", "next"); });
	Wait([] { return application_entered.load(); });
	background_pause = Pause::None;
	Wait([&] { return fixture.LeaseAvailable(); });
	if (after)
		Require(restarts > 0, "BEGIN gate remained held after SQL BEGIN completed");
	else
		Require(restarts == 0, "Background RESTART did not skip the held begin gate");
	application_pause = Pause::None;
	writer.Finish();
	// A fully backfilled WAL can be reused by the writer without a separate
	// RESTART call. Progress means the new committed tail is copied, not that
	// the worker must run that optional mode after every skipped attempt.
	Pager observer(path);
	observer.SQL("SELECT count(*) FROM sqlite_master");
	Wait([&] {
		int frames = -1, copied = -1;
		auto code = sqlite3_wal_checkpoint_v2(observer.db, "main", SQLITE_CHECKPOINT_NOOP, &frames, &copied);
		return code == SQLITE_OK && frames >= 0 && frames == copied;
	});
	fixture.workspace.Sync();
	Require(fixture.session->Read("/file") == "next", "Checkpoint skip/retry changed file data");
	std::cout << "PASS "
	          << (after ? "gate released immediately after BEGIN"
	                    : "try-lock skip releases checkpoint lease and next commit progresses")
	          << '\n';
}
static void FailedBegin(const std::string &path) {
	Reset();
	Fixture fixture(path);
	armed = fail_begin = true;
	bool failed = false;
	try {
		fixture.workspace.Sync();
	} catch (const Error &error) {
		Require(error.code == ErrorCode::Busy, error.what());
		failed = true;
	}
	Require(failed, "Injected BEGIN failure was hidden");
	Operation retry([&] {
		fixture.workspace.Sync();
		fixture.session->Write("/file", "next");
	});
	retry.Finish();
	fixture.Warm(Pause::RestartLock);
	background_pause = Pause::None;
	fixture.workspace.Sync();
	std::cout << "PASS failed FULL BEGIN unwinds gate/guards and permits retry and checkpoint\n";
}
static void ExternalBusy(const std::string &path) {
	Reset();
	Workspace workspace(path, 30, Durability::Fsync);
	auto session = workspace.Checkout();
	session->WriteFile("/file", "seed");
	Pager other(path);
	other.SQL("BEGIN IMMEDIATE");
	armed = true;
	bool failed = false;
	try {
		session->Write("/file", "next");
	} catch (const Error &error) {
		Require(error.code == ErrorCode::Busy, error.what());
		failed = true;
	}
	Require(failed && busy_sleeps > 0 && session->Read("/file") == "seed",
	        "External writer timeout/atomic failure changed");
	other.SQL("ROLLBACK");
	session->Write("/file", "next");
	workspace.Sync();
	std::cout << "PASS real foreign writer retains SQLite busy timeout and atomic retry\n";
}
int main(int argc, char **argv) {
	if (argc < 2)
		return 2;
	application = true;
	try {
		Hooks hooks;
		auto root = std::filesystem::path(argv[1]);
		bool control = argc == 3 && !std::strcmp(argv[2], "--parent-control");
		HeldRestart((root / "normal.sqlite").string(), control, false);
		HeldRestart((root / "full.sqlite").string(), control, true);
		if (control)
			return 0;
		PassiveConcurrent((root / "passive.sqlite").string());
		BeginBoundary((root / "skip.sqlite").string(), false);
		BeginBoundary((root / "release.sqlite").string(), true);
		FailedBegin((root / "failed.sqlite").string());
		ExternalBusy((root / "external.sqlite").string());
	} catch (const std::exception &error) {
		background_pause = application_pause = Pause::None;
		std::cerr << error.what() << '\n';
		return 1;
	}
}
