// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <cstring>
#include <iostream>
#include <stdexcept>

static int writers = 0, workers = 0;
extern "C" int __real_sqlite3_exec(sqlite3 *, const char *, int (*)(void *, int, char **, char **), void *, char **);
extern "C" int __real_sqlite3_wal_autocheckpoint(sqlite3 *, int);

static void Require(bool condition, const char *message) {
	if (!condition)
		throw std::runtime_error(message);
}

static int Value(sqlite3 *db, const char *query) {
	int value = -1;
	int rc = __real_sqlite3_exec(
	    db, query,
	    [](void *context, int n, char **values, char **) {
		    if (n && values[0])
			    *static_cast<int *>(context) = std::stoi(values[0]);
		    return 0;
	    },
	    &value, nullptr);
	Require(rc == SQLITE_OK, "Reading configuration");
	return value;
}

extern "C" int __wrap_sqlite3_wal_autocheckpoint(sqlite3 *db, int frames) {
	int page = Value(db, "PRAGMA page_size");
	Require(page == VANE_FS_PAGE_BYTES, "Wrong initial page size");
	Require(Value(db, "PRAGMA cache_size") == -2048, "Writer cache budget differs");
	Require(Value(db, "PRAGMA synchronous") == 2, "Initial writer must use FULL");
	Require(int64_t(frames) * (page + 24) + 32 >= 16 * 1024 * 1024, "Checkpoint trigger too small");
	Require(int64_t(frames - 1) * (page + 24) + 32 < 16 * 1024 * 1024, "Checkpoint trigger too large");
	++writers;
	return __real_sqlite3_wal_autocheckpoint(db, frames);
}

extern "C" int __wrap_sqlite3_exec(sqlite3 *db, const char *sql, int (*callback)(void *, int, char **, char **),
                                   void *argument, char **error) {
	int rc = __real_sqlite3_exec(db, sql, callback, argument, error);
	if (rc == SQLITE_OK && std::strstr(sql, "PRAGMA synchronous=NORMAL; PRAGMA cache_size=")) {
		Require(Value(db, "PRAGMA page_size") == VANE_FS_PAGE_BYTES, "Worker page size differs");
		Require(Value(db, "PRAGMA cache_size") == -2048, "Worker cache budget differs");
		Require(Value(db, "PRAGMA synchronous") == 1, "Worker must use NORMAL");
		++workers;
	}
	return rc;
}

int main(int argc, char **argv) {
	try {
		Require(argc == 2, "Expected new database path");
		{
			vane_fs::Workspace strict(argv[1]);
			strict.Checkout()->WriteFile("/file", "strict");
		}
		{
			vane_fs::Workspace fsync(argv[1], 5000, vane_fs::Durability::Fsync);
			fsync.Checkout()->WriteFile("/file", "fsync");
			fsync.Sync();
		}
		Require(writers == 2 && workers == 1, "Missing connection checks");
		std::cout << "PASS: page size, writer/worker cache bytes, FULL/NORMAL and WAL trigger verified\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
