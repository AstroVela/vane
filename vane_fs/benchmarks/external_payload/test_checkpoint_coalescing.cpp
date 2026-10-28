// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "checkpoint.hpp"
#include <atomic>
#include <filesystem>
#include <iostream>

static const auto foreground = std::this_thread::get_id();
static std::atomic<bool> hold_first {true}, waiting {false};
static std::atomic<int> worker_calls {0}, passive_calls {0};

static void Require(bool value, const char *message) {
	if (!value)
		throw std::runtime_error(message);
}
template <class F>
static void Wait(F condition) {
	auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
	while (!condition()) {
		Require(std::chrono::steady_clock::now() < deadline, "Timed out waiting for checkpoint");
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}
extern "C" int __real_sqlite3_wal_checkpoint_v2(sqlite3 *, const char *, int, int *, int *);
extern "C" int __wrap_sqlite3_wal_checkpoint_v2(sqlite3 *db, const char *name, int mode, int *frames, int *copied) {
	bool background = std::this_thread::get_id() != foreground;
	if (background && mode == SQLITE_CHECKPOINT_PASSIVE) {
		++passive_calls;
		if (hold_first) {
			waiting = true;
			while (hold_first)
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
	auto code = __real_sqlite3_wal_checkpoint_v2(db, name, mode, frames, copied);
	if (background)
		++worker_calls;
	return code;
}
struct Connection {
	sqlite3 *db = nullptr;
	explicit Connection(const char *path) {
		Require(sqlite3_open(path, &db) == SQLITE_OK, "Opening database");
		sqlite3_busy_timeout(db, 1000);
	}
	~Connection() {
		sqlite3_close_v2(db);
	}
	void SQL(const char *sql) {
		Require(sqlite3_exec(db, sql, nullptr, nullptr, nullptr) == SQLITE_OK, sqlite3_errmsg(db));
	}
	bool CaughtUp() {
		int frames = -1, copied = -1;
		auto code = sqlite3_wal_checkpoint_v2(db, "main", SQLITE_CHECKPOINT_NOOP, &frames, &copied);
		Require(code == SQLITE_OK || code == SQLITE_BUSY, "Inspecting WAL progress");
		return code == SQLITE_OK && frames >= 0 && frames == copied;
	}
};

int main(int argc, char **argv) {
	if (argc != 2)
		return 2;
	try {
		auto root = std::filesystem::path(argv[1]);
		Require(std::filesystem::create_directory(root), "Test directory must not exist");
		struct Cleanup {
			std::filesystem::path path;
			~Cleanup() {
				std::filesystem::remove_all(path);
				std::cout << "Removed owned test data: " << path << '\n';
			}
		} cleanup {root};
		auto path = (root / "workspace.sqlite").string();
		Connection writer(path.c_str()), reader(path.c_str()), progress(path.c_str());
		writer.SQL("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA wal_autocheckpoint=0; "
		           "PRAGMA journal_size_limit=16777216; CREATE TABLE padding(data BLOB)");
		progress.SQL("SELECT count(*) FROM padding");
		vane_fs::Checkpointer worker(writer.db, 4096, 1000);
		// Release the paused worker even if a later assertion fails.
		struct Resume {
			~Resume() {
				hold_first = false;
			}
		} resume;
		sqlite3_wal_hook(writer.db, vane_fs::Checkpointer::OnCommit, &worker);
		writer.SQL("INSERT INTO padding VALUES(zeroblob(33554432))");
		Wait([] { return waiting.load(); });
		// Establish a WAL reader before backfill completes. It keeps the old
		// WAL generation alive while we add less than the wake budget.
		reader.SQL("BEGIN; SELECT count(*) FROM padding");
		hold_first = false;
		Wait([&] { return progress.CaughtUp() && worker_calls > 0; });
		Require(passive_calls == 1, "Initial checkpoint did not finish as one batch");
		for (int i = 0; i < 32; ++i) {
			auto calls = worker_calls.load();
			worker.BeforeWrite(writer.db);
			writer.SQL("INSERT INTO padding VALUES(zeroblob(4096))");
			Wait([&] { return worker_calls > calls; });
			Require(passive_calls == 1, "Small WAL additions restarted a completed checkpoint");
		}
		worker.BeforeWrite(writer.db);
		writer.SQL("INSERT INTO padding VALUES(zeroblob(33554432))");
		Wait([] { return passive_calls > 1; });
		reader.SQL("ROLLBACK");
		// Once a budget-sized checkpoint has started, its retry must finish
		// after the reader goes away, without another foreground commit.
		Wait([&] { return progress.CaughtUp(); });
		worker.BeforeWrite(writer.db);
		writer.SQL("INSERT INTO padding VALUES(zeroblob(4096))");
		Require(std::filesystem::file_size(path + "-wal") <= 16777216,
		        "Admission restart did not retain its original WAL bound");
		sqlite3_wal_hook(writer.db, nullptr, nullptr);
		std::cout << "Checkpoint batching, reader release and WAL reuse passed\n";
		return 0;
	} catch (const std::exception &error) {
		hold_first = false;
		std::cerr << error.what() << '\n';
		return 1;
	}
}
