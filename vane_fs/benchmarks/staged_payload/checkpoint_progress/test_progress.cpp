// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "fixture.hpp"
#include <atomic>

static const auto foreground = std::this_thread::get_id();
static std::atomic<int> passes {0}, partial {0}, hook_queries {0};
static std::atomic<bool> writing {false}, began_during_writes {false};
static thread_local bool in_hook = false;
extern "C" int __real_sqlite3_wal_checkpoint_v2(sqlite3 *, const char *, int, int *, int *);
extern "C" int __wrap_sqlite3_wal_checkpoint_v2(sqlite3 *db, const char *name, int mode, int *frames, int *copied) {
	if (in_hook)
		++hook_queries;
	auto result = __real_sqlite3_wal_checkpoint_v2(db, name, mode, frames, copied);
	if (std::this_thread::get_id() != foreground && mode == SQLITE_CHECKPOINT_PASSIVE) {
		++passes;
		if (writing)
			began_during_writes = true;
		if (result == SQLITE_OK && *frames > *copied)
			++partial;
	}
	return result;
}
extern "C" int __real_fsync(int);
extern "C" int __wrap_fsync(int fd) {
	Require(!in_hook, "Commit hook called fsync");
	return __real_fsync(fd);
}
extern "C" int __real_fdatasync(int);
extern "C" int __wrap_fdatasync(int fd) {
	Require(!in_hook, "Commit hook called fdatasync");
	return __real_fdatasync(fd);
}
static int Hook(void *context, sqlite3 *db, const char *name, int frames) {
	in_hook = true;
	auto result = Checkpointer::OnCommit(context, db, name, frames);
	in_hook = false;
	return result;
}
struct Fixture {
	Workspace workspace;
	SQL writer, progress;
	Staging staging;
	std::unique_ptr<Checkpointer> worker;
	explicit Fixture(const std::string &path) : workspace(path), writer(path), progress(path), staging(path, 5000) {
		writer.Exec("PRAGMA synchronous=NORMAL; CREATE TABLE padding(id INTEGER PRIMARY KEY, data BLOB)");
		staging.Initialize(workspace.Id(), false);
		staging.Ready();
		worker = std::make_unique<Checkpointer>(writer.db, 4096, 5000, staging);
		sqlite3_wal_hook(writer.db, Hook, worker.get());
	}
	~Fixture() {
		sqlite3_wal_hook(writer.db, nullptr, nullptr);
		worker->Stop();
	}
	void WarmAndReset() {
		writer.Exec("INSERT INTO padding VALUES(1,zeroblob(20971520))");
		Wait([&] { return passes > 0 && progress.CaughtUp(); }, "Large generation did not backfill");
		worker->Stop();
		progress.Checkpoint(SQLITE_CHECKPOINT_RESTART);
		passes = partial = 0;
	}
	void Small(int id = 2) {
		writer.Exec("INSERT INTO padding VALUES(" + std::to_string(id) + ",zeroblob(4096))");
		std::pair<int, int> status;
		Wait(
		    [&] {
			    status = writer.Status();
			    return status.first >= 0;
		    },
		    "Read small WAL status");
		Require(int64_t(status.first) * 4120 + 32 < Checkpointer::START_BYTES,
		        "Small replay unexpectedly crossed the original wake threshold");
	}
};
int main(int argc, char **argv) {
	try {
		Require(argc == 3, "Expected database and case");
		std::string path = argv[1], mode = argv[2];
		Fixture fixture(path);
		if (mode == "cold") {
			fixture.Small();
			std::this_thread::sleep_for(std::chrono::milliseconds(30));
			Require(passes == 0, "Cold small commits started background I/O");
		} else {
			Require(mode == "reset" || mode == "live" || mode == "reader" || mode == "stream", "Unknown case");
			fixture.WarmAndReset();
			if (mode == "reset") {
				fixture.Small();
				Require(!fixture.progress.CaughtUp(), "Stopped worker fixture has no dirty WAL");
				fixture.worker->Start();
			} else if (mode == "live") {
				fixture.worker->Start();
				fixture.Small();
			} else if (mode == "reader") {
				fixture.Small();
				fixture.progress.Checkpoint(SQLITE_CHECKPOINT_PASSIVE);
				SQL reader(path);
				reader.Exec("BEGIN; SELECT count(*) FROM padding");
				fixture.worker->Start();
				fixture.Small(3);
				Wait([] { return partial > 0; }, "Small reader-pinned tail did not backfill partially");
				reader.Exec("ROLLBACK");
			} else {
				fixture.worker->Start();
				writing = true;
				for (int id = 2; id < 258; ++id)
					fixture.Small(id);
				writing = false;
				Require(began_during_writes, "Small continuous commits waited for the stream to end");
			}
			Wait([&] { return passes > 0 && fixture.progress.CaughtUp(); },
			     "Small post-reuse commit was not backfilled without further commits");
		}
		fixture.worker->Stop();
		Require(hook_queries == 0, "Commit hook performed a WAL status query");
		Require(fixture.writer.Text("PRAGMA quick_check") == "ok", "SQLite integrity failure");
		Require(fixture.writer.Text("SELECT hex(substr(data,1,4)) FROM padding WHERE id=2") == "00000000",
		        "Small committed bytes changed");
		std::cout << "PASS " << mode << " passive=" << passes << " partial=" << partial
		          << " hook_queries=" << hook_queries << '\n';
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
