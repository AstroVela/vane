// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "staging.hpp"
#include "wait_trace.hpp"
#include <stdexcept>
#include <sqlite3.h>

int main(int argc, char **argv) {
	if (argc != 2)
		return 2;
	fuse_waits::Install(argv[1]);
	auto path = std::string(argv[1]) + ".sqlite";
	vane_fs::Workspace workspace(path, 5000, vane_fs::Durability::Fsync);
	workspace.Checkout()->WriteFile("/file", std::string(1024 * 1024, 'p'));
	vane_fs::Staging observer(path, 5000);
	observer.Initialize(workspace.Id(), false);
	observer.Ready();
	sqlite3 *db = nullptr;
	if (sqlite3_open(path.c_str(), &db) != SQLITE_OK)
		return 3;
	vane_fs::Staging::Configure(db);
	std::mutex checkpoint;
	{
		fuse_waits::Root root("test");
		fuse_waits::HeldLock checkpoint_held(fuse_waits::CheckpointWait, fuse_waits::CheckpointHold, checkpoint);
		vane_fs::Staging::Guard held(observer);
		{
			fuse_waits::Span child(fuse_waits::CheckpointWait);
			std::this_thread::sleep_for(std::chrono::milliseconds(3));
		}
		errno = EDOM;
		if (fsync(-1) != -1 || errno != EBADF)
			return 4;
		sqlite3_stmt *statement = nullptr;
		if (sqlite3_prepare_v2(db, "SELECT 42", -1, &statement, nullptr) != SQLITE_OK ||
		    sqlite3_step(statement) != SQLITE_ROW || sqlite3_column_int(statement, 0) != 42)
			return 5;
		sqlite3_finalize(statement);
		// A real publication must nest its payload/WAL/fence operations inside
		// one complete Guard lifetime. VM steps and disjoint counters still add up.
		observer.Publish(db);
	}
	std::thread worker([&] {
		fuse_waits::Root root("worker_test");
		try {
			fuse_waits::HeldLock checkpoint_held(fuse_waits::CheckpointWait, fuse_waits::CheckpointHold, checkpoint);
			vane_fs::Staging::Guard held(observer);
			fuse_waits::Span child(fuse_waits::WriteBytes);
			throw std::runtime_error("check guard unwinding");
		} catch (const std::runtime_error &) {
		}
	});
	worker.join();
	{
		fuse_waits::Root read_root("read_test");
		if (workspace.Checkout()->Read("/file") != std::string(1024 * 1024, 'p'))
			return 6;
	}
	sqlite3_close(db);
	workspace.Close();
	return 0;
}
