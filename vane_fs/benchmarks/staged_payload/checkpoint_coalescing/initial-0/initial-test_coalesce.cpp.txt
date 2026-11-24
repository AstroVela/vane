// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <thread>

using namespace vane_fs;
static constexpr int MIB = 1024 * 1024;
static auto foreground = std::this_thread::get_id();
static std::atomic<int> worker_calls {0}, worker_passes {0}, hook_queries {0};
static thread_local bool committing = false;
static std::string fault;
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
struct SQL {
	sqlite3 *db = nullptr;
	explicit SQL(const std::string &path) {
		Require(sqlite3_open(path.c_str(), &db) == SQLITE_OK, "Opening inspector");
		Require(sqlite3_wal_autocheckpoint(db, 0) == SQLITE_OK, "Disabling inspector auto-checkpoint");
	}
	~SQL() {
		sqlite3_close_v2(db);
	}
	void Exec(const char *sql) {
		Require(sqlite3_exec(db, sql, nullptr, nullptr, nullptr) == SQLITE_OK, sqlite3_errmsg(db));
	}
	std::pair<int, int> Progress(int mode = SQLITE_CHECKPOINT_NOOP) {
		int frames = -1, copied = -1;
		auto code = sqlite3_wal_checkpoint_v2(db, "main", mode, &frames, &copied);
		Require(code == SQLITE_OK || code == SQLITE_BUSY, "Reading progress");
		return code == SQLITE_OK ? std::make_pair(frames, copied) : std::make_pair(-1, -1);
	}
};
extern "C" int __real_sqlite3_step(sqlite3_stmt *);
extern "C" int __wrap_sqlite3_step(sqlite3_stmt *statement) {
	bool previous = committing;
	committing = std::strcmp(sqlite3_sql(statement), "COMMIT") == 0;
	auto code = __real_sqlite3_step(statement);
	committing = previous;
	return code;
}
extern "C" int __real_sqlite3_wal_checkpoint_v2(sqlite3 *, const char *, int, int *, int *);
extern "C" int __wrap_sqlite3_wal_checkpoint_v2(sqlite3 *db, const char *name, int mode, int *frames, int *copied) {
	bool worker = std::this_thread::get_id() != foreground;
	if (worker) {
		++worker_calls;
		if (mode == SQLITE_CHECKPOINT_PASSIVE)
			++worker_passes;
	}
	if (!worker && committing && mode == SQLITE_CHECKPOINT_NOOP) {
		++hook_queries;
		if (fault == "busy")
			return SQLITE_BUSY;
		if (fault == "io_status")
			return SQLITE_IOERR;
		if (fault == "unknown" || fault == "invalid_progress") {
			*frames = fault == "unknown" ? -1 : 1;
			*copied = fault == "unknown" ? -1 : 2;
			return SQLITE_OK;
		}
	}
	return __real_sqlite3_wal_checkpoint_v2(db, name, mode, frames, copied);
}

// Pin a WAL reader before backfilling. The log stays above 16 MiB while
// copied==frames; subsequent small commits must not restart it or dispatch work.
static void BackfilledLog(SQL &inspector, SQL &reader) {
	inspector.Exec("CREATE TABLE padding(id INTEGER PRIMARY KEY, data BLOB); "
	               "WITH RECURSIVE n(v) AS (VALUES(1) UNION ALL SELECT v+1 FROM n WHERE v<5000) "
	               "INSERT INTO padding SELECT v,zeroblob(4096) FROM n");
	reader.Exec("BEGIN; SELECT count(*) FROM padding");
	auto before = inspector.Progress();
	Require(before.first * int64_t(4120) > 16 * MIB, "Missing retained large WAL");
	auto after = inspector.Progress(SQLITE_CHECKPOINT_PASSIVE);
	Require(after.first == after.second && after.first > 0, "WAL was not fully backfilled");
}

int main(int argc, char **argv) {
	try {
		Require(argc == 3, "Expected database path and mode");
		std::string path = argv[1], mode = argv[2];
		Workspace workspace(path, 1000, Durability::Fsync);
		auto session = workspace.Checkout();
		session->WriteFile("/file", "base");
		workspace.Sync();
		SQL inspector(path), reader(path);
		BackfilledLog(inspector, reader);
		worker_calls = worker_passes = hook_queries = 0;
		if (mode == "small" || mode == "threshold") {
			for (int i = 0; i < 32; ++i)
				session->WriteFile("/file", std::string(4096, char('a' + i % 20)));
			std::this_thread::sleep_for(std::chrono::milliseconds(120));
			Require(worker_calls == 0, "Backfilled WAL dispatched a checkpoint for small CLEAN commits");
			Require(hook_queries >= 32, "Missing current-generation hook observations");
			if (mode == "threshold") {
				for (int i = 0; i < 160 && worker_passes == 0; ++i)
					session->WriteFile("/file", std::string(128 * 1024, char('a' + i % 20)));
				Wait([] { return worker_passes > 0; });
			}
		} else if (mode == "reset") {
			reader.Exec("ROLLBACK");
			auto old = inspector.Progress(SQLITE_CHECKPOINT_RESTART);
			Require(old.first == old.second && old.first > 4096, "Reset preparation failed");
			inspector.Exec("CREATE TRIGGER inflate AFTER INSERT ON block_payloads WHEN length(new.data)=8 "
			               "BEGIN INSERT INTO padding(data) VALUES(zeroblob(4096)); END");
			session->WriteFile("/file", std::string(28 * MIB, 'r'));
			Wait([] { return worker_passes > 0; });
			Require(hook_queries > 0, "New generation was not queried");
			Wait([&] {
				auto p = inspector.Progress();
				return p.first >= 0 && p.first == p.second;
			});
			Require(session->Read("/file", 27 * MIB) == std::string(MIB, 'r'), "Reset changed content");
		} else {
			Require(mode == "busy" || mode == "io_status" || mode == "unknown" || mode == "invalid_progress",
			        "Unknown mode");
			fault = mode;
			session->WriteFile("/file", "committed");
			fault.clear();
			Require(session->Read("/file") == "committed", "Hook status failure changed acknowledged data");
			Wait([] { return worker_calls > 0; });
			Require(hook_queries > 0, "Hook status fault was not exercised");
		}
		if (mode != "reset")
			reader.Exec("ROLLBACK");
		if (mode == "threshold")
			Wait([&] {
				auto p = inspector.Progress();
				return p.first >= 0 && p.first == p.second;
			});
		workspace.Sync();
		auto content = session->Read("/file");
		workspace.Close();
		Workspace reopened(path);
		Require(reopened.Checkout()->Read("/file") == content, "Reopen lost committed content");
		reopened.Close();
		std::cout << mode << " PASS hook_queries=" << hook_queries << " worker_calls=" << worker_calls
		          << " passes=" << worker_passes << '\n';
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
