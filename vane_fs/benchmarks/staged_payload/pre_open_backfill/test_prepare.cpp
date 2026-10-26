// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "../checkpoint_progress/fixture.hpp"
#include <atomic>
#include <type_traits>

static std::atomic<int> passes {0}, restarts {0}, injected {SQLITE_OK};
static std::atomic<int> last_frames {-1}, last_copied {-1};
extern "C" int __real_sqlite3_wal_checkpoint_v2(sqlite3 *, const char *, int, int *, int *);
extern "C" int __wrap_sqlite3_wal_checkpoint_v2(sqlite3 *db, const char *name, int mode, int *frames, int *copied) {
	if (mode == SQLITE_CHECKPOINT_RESTART)
		++restarts;
	if (mode == SQLITE_CHECKPOINT_PASSIVE) {
		++passes;
		auto error = injected.exchange(SQLITE_OK);
		if (error != SQLITE_OK)
			return error;
	}
	auto code = __real_sqlite3_wal_checkpoint_v2(db, name, mode, frames, copied);
	if (mode == SQLITE_CHECKPOINT_PASSIVE) {
		last_frames = *frames;
		last_copied = *copied;
	}
	return code;
}

// Compile the same behavioral reproduction against the preceding single-argument API.
template <class C>
void Prepare(C &worker, sqlite3 *writer, bool external) {
	if constexpr (std::is_invocable_v<decltype(&C::BeforeWrite), C *, sqlite3 *, bool>)
		worker.BeforeWrite(writer, external);
	else
		worker.BeforeWrite(writer);
}

int main(int argc, char **argv) {
	try {
		Require(argc == 3, "Expected database and case");
		std::string path = argv[1], mode = argv[2];
		Workspace workspace(path, 5000, Durability::Fsync);
		SQL writer(path), progress(path), reader(path);
		sqlite3_wal_autocheckpoint(writer.db, 0);
		writer.Exec("PRAGMA synchronous=FULL; CREATE TABLE padding(id INTEGER PRIMARY KEY, data BLOB)");
		progress.Checkpoint(SQLITE_CHECKPOINT_RESTART);
		if (mode == "reader")
			reader.Exec("BEGIN; SELECT count(*) FROM padding");
		writer.Exec(mode == "tiny" ? "INSERT INTO padding VALUES(1,zeroblob(4096))"
		                           : "INSERT INTO padding VALUES(1,zeroblob(1048576))");
		auto before = progress.Status();
		Require(before.first > before.second && int64_t(before.first) * 4120 < Checkpointer::START_BYTES,
		        "Fixture must have an unwarmed dirty prefix");
		Staging staging(path, 5000);
		staging.Initialize(workspace.Id(), false);
		staging.Ready();
		Checkpointer worker(writer.db, 4096, 5000, staging);
		worker.Stop();
		if (mode == "open") {
			Staging::Guard publication(staging);
			writer.Exec("BEGIN IMMEDIATE");
			staging.BeginBatch(writer.db);
			writer.Exec("INSERT INTO padding VALUES(2,zeroblob(4096)); COMMIT");
			before = progress.Status();
		}
		passes = restarts = 0;
		if (mode == "error" || mode == "public_error")
			injected = SQLITE_IOERR;
		if (mode == "busy")
			injected = SQLITE_BUSY;
		bool storage_error = false;
		auto payload_size = std::filesystem::file_size(path + ".payload");
		try {
			if (mode == "public" || mode == "public_error" || mode == "small")
				workspace.Checkout()->WriteFile("/incoming", std::string(mode == "small" ? 4096 : 524288, 'p'));
			else
				Prepare(worker, writer.db, true);
		} catch (const Error &error) {
			Require(error.code == ErrorCode::Storage, "Unexpected preparation error");
			storage_error = true;
		}
		Require(restarts == 0, "Preparation must not restart a sub-budget generation");
		if (mode == "large" || mode == "public") {
			Require(!storage_error && passes == 1, "Large public write skipped CLEAN backfill");
			Require(last_frames == last_copied && last_frames >= before.first, "CLEAN prefix was not copied");
			if (mode == "large")
				Require(progress.CaughtUp(), "Prepared prefix remains dirty");
			else
				Require(workspace.Checkout()->Read("/incoming") == std::string(524288, 'p'), "Public content changed");
		} else if (mode == "reader") {
			Require(!storage_error && passes == 1, "Pinned-reader preparation did not attempt PASSIVE");
			Require(!progress.CaughtUp(), "Reader did not pin the dirty prefix");
			reader.Exec("COMMIT");
		} else if (mode == "error" || mode == "public_error") {
			Require(storage_error && passes == 1, "Checkpoint failure did not abort the public write");
			Require(progress.Status() == before, "Failed preparation mutated the WAL");
			Require(std::filesystem::file_size(path + ".payload") == payload_size,
			        "Failed preparation appended payload");
			Require(workspace.Checkout()->ListDirectory("/").empty(), "Failed preparation created a file");
		} else {
			Require(mode == "small" || mode == "tiny" || mode == "open" || mode == "busy", "Unknown case");
			Require(!storage_error && passes == (mode == "busy" ? 1 : 0), "Preparation changed an excluded path");
			Require(progress.Status().second == before.second, "Excluded path unexpectedly backfilled the prefix");
		}
		if (mode == "open") {
			Staging::Guard publication(staging);
			Require(staging.Pending(), "Preparation published an OPEN batch");
			staging.Publish(writer.db);
		}
		Require(writer.Text("PRAGMA quick_check") == "ok", "Integrity check failed");
		std::cout << "PASS " << mode << " passes=" << passes << " frames=" << last_frames << " copied=" << last_copied
		          << '\n';
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
