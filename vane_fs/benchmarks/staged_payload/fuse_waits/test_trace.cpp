// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "wait_trace.hpp"
#include <cerrno>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <sqlite3.h>

int main(int argc, char **argv) {
	if (argc != 2)
		return 2;
	fuse_waits::Install(argv[1]);
	{
		fuse_waits::Root root("test");
		fuse_waits::Span parent(fuse_waits::Admission);
		{
			fuse_waits::Span child(fuse_waits::CheckpointWait);
			std::this_thread::sleep_for(std::chrono::milliseconds(3));
		}
		errno = EDOM;
		if (fsync(-1) != -1 || errno != EBADF)
			return 3;
		sqlite3 *db = nullptr;
		if (sqlite3_open(":memory:", &db) != SQLITE_OK)
			return 4;
		sqlite3_stmt *statement = nullptr;
		if (sqlite3_prepare_v2(db, "SELECT 42", -1, &statement, nullptr) != SQLITE_OK ||
		    sqlite3_step(statement) != SQLITE_ROW || sqlite3_column_int(statement, 0) != 42)
			return 5;
		sqlite3_finalize(statement);
		sqlite3_close(db);
	}
	std::thread worker([] {
		fuse_waits::Root root("worker_test");
		try {
			fuse_waits::Span span(fuse_waits::WriteBytes);
			throw std::runtime_error("check unwinding");
		} catch (const std::runtime_error &) {
		}
	});
	worker.join();
	return 0;
}
