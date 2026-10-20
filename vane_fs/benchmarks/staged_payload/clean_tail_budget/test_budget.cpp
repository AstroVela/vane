// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "../checkpoint_progress/fixture.hpp"
#include <atomic>

static const auto foreground = std::this_thread::get_id();
static std::atomic<int> probes {0}, passes {0}, first_pass_probes {-1};
extern "C" int __real_sqlite3_wal_checkpoint_v2(sqlite3 *, const char *, int, int *, int *);
extern "C" int __wrap_sqlite3_wal_checkpoint_v2(sqlite3 *db, const char *name, int mode, int *frames, int *copied) {
	if (std::this_thread::get_id() != foreground && mode == SQLITE_CHECKPOINT_PASSIVE) {
		int expected = -1;
		first_pass_probes.compare_exchange_strong(expected, probes.load());
	}
	auto code = __real_sqlite3_wal_checkpoint_v2(db, name, mode, frames, copied);
	if (std::this_thread::get_id() != foreground) {
		if (mode == SQLITE_CHECKPOINT_NOOP)
			++probes;
		if (mode == SQLITE_CHECKPOINT_PASSIVE)
			++passes;
	}
	return code;
}

int main(int argc, char **argv) {
	try {
		Require(argc == 3, "Expected database and generation|tail");
		std::string path = argv[1], mode = argv[2];
		Workspace workspace(path);
		SQL writer(path), progress(path), reader(path);
		sqlite3_wal_autocheckpoint(writer.db, 0);
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
		if (mode == "generation") {
			writer.Exec("INSERT INTO padding VALUES(2,zeroblob(20971520))");
			reader.Exec("BEGIN; SELECT count(*) FROM padding");
			progress.Checkpoint(SQLITE_CHECKPOINT_PASSIVE);
			writer.Exec("INSERT INTO padding VALUES(3,zeroblob(4096))");
			reader.Exec("COMMIT");
		} else {
			Require(mode == "tail", "Unknown case");
			writer.Exec("INSERT INTO padding VALUES(2,zeroblob(1048576))");
		}
		auto status = progress.Status();
		auto bytes = [](int frames) {
			return int64_t(frames) * 4120 + 32;
		};
		Require(status.first > status.second, "Fixture has no outstanding work");
		if (mode == "generation") {
			Require(bytes(status.first) >= Checkpointer::START_BYTES, "Fixture did not retain a large generation");
			Require(bytes(status.first - status.second) < 256 * 1024, "Fixture tail is not small");
		} else {
			Require(bytes(status.first) < Checkpointer::START_BYTES, "Fixture generation is not small");
			Require(bytes(status.first - status.second) > 256 * 1024, "Fixture prefix is too small");
		}
		passes = probes = 0;
		first_pass_probes = -1;
		worker.Start();
		Wait([&] { return passes > 0 && progress.CaughtUp(); }, "Tail did not backfill without another commit");
		worker.Stop();
		sqlite3_wal_hook(writer.db, nullptr, nullptr);
		std::cout << mode << " frames=" << status.first << " copied=" << status.second
		          << " probes_before_first_pass=" << first_pass_probes << '\n';
		Require(first_pass_probes == (mode == "tail" ? 1 : 2), "Unexpected CLEAN scheduling for prefix budget");
		Require(writer.Text("PRAGMA quick_check") == "ok", "Integrity check failed");
		std::cout << "PASS " << mode << '\n';
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
