// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "../checkpoint_progress/fixture.hpp"
#include <atomic>

static const auto foreground = std::this_thread::get_id();
static std::atomic<int> passes {0}, probes {0};
extern "C" int __real_sqlite3_wal_checkpoint_v2(sqlite3 *, const char *, int, int *, int *);
extern "C" int __wrap_sqlite3_wal_checkpoint_v2(sqlite3 *db, const char *name, int mode, int *frames, int *copied) {
	auto code = __real_sqlite3_wal_checkpoint_v2(db, name, mode, frames, copied);
	if (std::this_thread::get_id() != foreground) {
		if (mode == SQLITE_CHECKPOINT_PASSIVE)
			++passes;
		if (mode == SQLITE_CHECKPOINT_NOOP)
			++probes;
	}
	return code;
}

int main(int argc, char **argv) {
	try {
		Require(argc == 3, "Expected database and burst|stop");
		std::string path = argv[1], mode = argv[2];
		Workspace workspace(path);
		SQL writer(path), progress(path);
		writer.Exec("PRAGMA synchronous=FULL; CREATE TABLE padding(id INTEGER PRIMARY KEY, data BLOB)");
		Staging staging(path, 5000);
		staging.Initialize(workspace.Id(), false);
		staging.Ready();
		Checkpointer worker(writer.db, 4096, 5000, staging);
		sqlite3_wal_hook(writer.db, Checkpointer::OnCommit, &worker);
		writer.Exec("INSERT INTO padding VALUES(1,zeroblob(20971520))");
		Wait([&] { return passes > 0 && progress.CaughtUp(); }, "Warm generation did not backfill");
		worker.Stop();
		progress.Checkpoint(SQLITE_CHECKPOINT_RESTART);
		passes = probes = 0;
		worker.Start();
		bool live_progress = false;
		if (mode == "burst") {
			for (int i = 2; i < 82; ++i) {
				writer.Exec("INSERT INTO padding VALUES(" + std::to_string(i) + ",zeroblob(4096))");
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
				live_progress = live_progress || passes > 0;
			}
			Require(live_progress, "Continuous commits slid every checkpoint past the end of the stream");
		} else {
			Require(mode == "stop", "Unknown case");
			writer.Exec("INSERT INTO padding VALUES(2,zeroblob(4096))");
			Wait([] { return probes > 0; }, "Worker did not inspect the small CLEAN tail");
			worker.Stop();
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			worker.Start();
		}
		Wait([&] { return passes > 0 && progress.CaughtUp(); }, "Last commit never backfilled without another write");
		sqlite3_wal_hook(writer.db, nullptr, nullptr);
		worker.Stop();
		if (mode == "burst")
			Require(passes < 40, "Small FULL commits were not coalesced");
		Require(writer.Text("PRAGMA quick_check") == "ok", "Integrity check failed");
		std::cout << "PASS " << mode << " passive=" << passes << " probes=" << probes << '\n';
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
