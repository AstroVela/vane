// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "checkpoint.hpp"
#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <atomic>
#include <cstring>
#include <iostream>

using namespace vane_fs;
static const auto foreground = std::this_thread::get_id();
static std::atomic<bool> hold {false}, waiting {false}, writing {false}, began_during_writes {false};
static std::atomic<bool> stop_notified {false};
static std::atomic<void *> wake_address {nullptr};
static std::atomic<int> signals {0}, hook_queries {0}, partial_passes {0}, passive_passes {0};
static thread_local bool committing = false;

static void Require(bool value, const char *message) {
	if (!value)
		throw std::runtime_error(message);
}
template <class F>
static void Wait(F condition) {
	auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
	while (!condition()) {
		Require(std::chrono::steady_clock::now() < deadline, "Timed out waiting for checkpoint progress");
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}
extern "C" int __real_sqlite3_step(sqlite3_stmt *);
extern "C" int __wrap_sqlite3_step(sqlite3_stmt *statement) {
	bool previous = committing;
	auto sql = sqlite3_sql(statement);
	committing = sql && std::strcmp(sql, "COMMIT") == 0;
	auto code = __real_sqlite3_step(statement);
	committing = previous;
	return code;
}
extern "C" void __real__ZNSt18condition_variable10notify_oneEv(void *);
extern "C" void __wrap__ZNSt18condition_variable10notify_oneEv(void *address) {
	if (committing && std::this_thread::get_id() == foreground) {
		++signals;
		wake_address = address;
	}
	__real__ZNSt18condition_variable10notify_oneEv(address);
}
extern "C" void __real__ZNSt18condition_variable10notify_allEv(void *);
extern "C" void __wrap__ZNSt18condition_variable10notify_allEv(void *address) {
	if (address == wake_address && std::this_thread::get_id() != foreground)
		stop_notified = true;
	__real__ZNSt18condition_variable10notify_allEv(address);
}
extern "C" int __real_sqlite3_wal_checkpoint_v2(sqlite3 *, const char *, int, int *, int *);
extern "C" int __wrap_sqlite3_wal_checkpoint_v2(sqlite3 *db, const char *name, int mode, int *frames, int *copied) {
	bool background = std::this_thread::get_id() != foreground;
	if (!background && committing && mode == SQLITE_CHECKPOINT_NOOP)
		++hook_queries;
	if (background && mode == SQLITE_CHECKPOINT_PASSIVE) {
		if (writing)
			began_during_writes = true;
		if (hold) {
			waiting = true;
			while (hold)
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
	auto code = __real_sqlite3_wal_checkpoint_v2(db, name, mode, frames, copied);
	if (background && mode == SQLITE_CHECKPOINT_PASSIVE) {
		++passive_passes;
		if (code == SQLITE_OK && *frames > *copied)
			++partial_passes;
	}
	return code;
}
struct SQL {
	sqlite3 *db = nullptr;
	explicit SQL(const std::string &path) {
		Require(sqlite3_open(path.c_str(), &db) == SQLITE_OK, "Opening inspector");
		sqlite3_busy_timeout(db, 1000);
		sqlite3_wal_autocheckpoint(db, 0);
	}
	~SQL() {
		sqlite3_close_v2(db);
	}
	void Exec(const char *sql) {
		Require(sqlite3_exec(db, sql, nullptr, nullptr, nullptr) == SQLITE_OK, sqlite3_errmsg(db));
	}
	bool CaughtUp() {
		int frames = -1, copied = -1;
		auto code = sqlite3_wal_checkpoint_v2(db, "main", SQLITE_CHECKPOINT_NOOP, &frames, &copied);
		Require(code == SQLITE_OK || code == SQLITE_BUSY, "Reading checkpoint progress");
		return code == SQLITE_OK && frames >= 0 && frames == copied;
	}
};
struct Resume {
	std::thread stopper;
	~Resume() {
		hold = false;
		if (stopper.joinable())
			stopper.join();
	}
};

static int Hook(void *context, sqlite3 *db, const char *name, int frames) {
	bool previous = committing;
	committing = true;
	auto code = Checkpointer::OnCommit(context, db, name, frames);
	committing = previous;
	return code;
}
static void StopRestart(const std::string &path) {
	Workspace workspace(path);
	SQL writer(path), reader(path), progress(path);
	writer.Exec("PRAGMA synchronous=NORMAL; CREATE TABLE padding(data BLOB)");
	progress.Exec("SELECT count(*) FROM padding");
	Staging staging(path, 1000);
	staging.Initialize(workspace.Id(), false);
	staging.Ready();
	Checkpointer worker(writer.db, 4096, 1000, staging);
	Resume resume;
	sqlite3_wal_hook(writer.db, Hook, &worker);
	reader.Exec("BEGIN; SELECT count(*) FROM padding");
	hold = true;
	writer.Exec("INSERT INTO padding VALUES(zeroblob(33554432))");
	Wait([] { return waiting.load(); });
	resume.stopper = std::thread([&] { worker.Stop(); });
	Wait([] { return stop_notified.load(); });
	hold = false;
	resume.stopper.join();
	Require(partial_passes > 0, "Stop fixture did not finish a reader-pinned partial pass");
	reader.Exec("ROLLBACK");
	worker.Start();
	Wait([&] { return progress.CaughtUp(); });
	sqlite3_wal_hook(writer.db, nullptr, nullptr);
	worker.Stop();
}

int main(int argc, char **argv) {
	try {
		Require(argc == 3, "Expected path and mode");
		std::string path = argv[1], mode = argv[2];
		if (mode == "stop") {
			StopRestart(path);
		} else {
			Require(mode == "notify" || mode == "clean" || mode == "continuous" || mode == "no_query", "Unknown mode");
			Workspace workspace(path, 1000, Durability::Fsync);
			auto session = workspace.Checkout();
			Resume resume;
			session->WriteFile("/file", "base");
			workspace.Sync();
			SQL inspector(path), reader(path);
			inspector.Exec("CREATE TABLE padding(id INTEGER PRIMARY KEY, data BLOB); "
			               "WITH RECURSIVE n(v) AS (VALUES(1) UNION ALL SELECT v+1 FROM n WHERE v<5000) "
			               "INSERT INTO padding SELECT v,zeroblob(4096) FROM n");
			reader.Exec("BEGIN; SELECT count(*) FROM padding");
			int frames = -1, copied = -1;
			Require(sqlite3_wal_checkpoint_v2(inspector.db, "main", SQLITE_CHECKPOINT_PASSIVE, &frames, &copied) ==
			                SQLITE_OK &&
			            frames == copied && frames > 4096,
			        "Missing retained/backfilled large WAL");
			signals = hook_queries = partial_passes = passive_passes = 0;
			if (mode == "notify") {
				hold = true;
				session->WriteFile("/file", "first");
				Wait([] { return waiting.load(); });
				signals = 0;
			}
			writing = true;
			for (int i = 0; i < (mode == "continuous" ? 256 : 32); ++i)
				session->WriteFile("/file", std::string(4096, char('a' + i % 20)));
			writing = false;
			Require(hook_queries == 0, "Commit callback performed an extra checkpoint status query");
			if (mode == "notify")
				Require(signals == 1, "Pending request repeatedly signalled the worker");
			if (mode == "continuous")
				Require(began_during_writes, "Continuous commits postponed every checkpoint until writes stopped");
			hold = false;
			if (mode != "no_query") {
				Wait([] { return partial_passes > 0; });
				Require(!inspector.CaughtUp(), "Reader did not pin the CLEAN tail");
			}
			reader.Exec("ROLLBACK");
			if (mode != "no_query")
				Wait([&] { return inspector.CaughtUp(); });
			workspace.Sync();
			auto content = session->Read("/file");
			workspace.Close();
			Workspace reopened(path);
			Require(reopened.Checkout()->Read("/file") == content, "Reopen lost acknowledged content");
			reopened.Close();
		}
		std::cout << mode << " PASS signals=" << signals << " hook_queries=" << hook_queries
		          << " passive=" << passive_passes << " partial=" << partial_passes << '\n';
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
