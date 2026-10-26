// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include <sqlite3.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

static bool allocating = false;
static unsigned writes = 0;
static size_t bytes = 0;
extern "C" ssize_t __real_pwrite64(int, const void *, size_t, off64_t);
extern "C" ssize_t __wrap_pwrite64(int fd, const void *data, size_t size, off64_t offset) {
	if (allocating) {
		++writes;
		bytes += size;
	}
	return __real_pwrite64(fd, data, size, offset);
}
static void Require(bool value, const char *message) {
	if (!value)
		throw std::runtime_error(message);
}
static void Exec(sqlite3 *db, const char *sql) {
	Require(sqlite3_exec(db, sql, nullptr, nullptr, nullptr) == SQLITE_OK, sqlite3_errmsg(db));
}
int main(int argc, char **argv) {
	sqlite3 *db = nullptr;
	try {
		Require(argc == 2, "Expected owned database path");
		Require(sqlite3_open(argv[1], &db) == SQLITE_OK, "Open probe");
		Exec(db, "PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA wal_autocheckpoint=0;"
		         "CREATE TABLE probe(id INTEGER PRIMARY KEY,value BLOB); INSERT INTO probe VALUES(1,'base');"
		         "BEGIN IMMEDIATE");
		sqlite3_file *wal = nullptr;
		Require(sqlite3_file_control(db, "main", SQLITE_FCNTL_JOURNAL_POINTER, &wal) == SQLITE_OK && wal &&
		            wal->pMethods,
		        "No opened WAL handle");
		sqlite3_int64 before = 0, after = 0, target = 16 * 1024 * 1024;
		Require(wal->pMethods->xFileSize(wal, &before) == SQLITE_OK && before > 0, "Initial WAL size");
		std::ifstream file(std::string(argv[1]) + "-wal", std::ios::binary);
		std::string prefix(size_t(before), '\0');
		Require(bool(file.read(&prefix[0], before)), "Read original WAL prefix");
		int chunk = int(target);
		Require(wal->pMethods->xFileControl(wal, SQLITE_FCNTL_CHUNK_SIZE, &chunk) == SQLITE_OK, "Set WAL chunk");
		allocating = true;
		auto code = wal->pMethods->xFileControl(wal, SQLITE_FCNTL_SIZE_HINT, &target);
		allocating = false;
		chunk = 0;
		Require(wal->pMethods->xFileControl(wal, SQLITE_FCNTL_CHUNK_SIZE, &chunk) == SQLITE_OK, "Restore WAL chunk");
		Require(code == SQLITE_OK, "WAL size hint failed");
		Require(wal->pMethods->xFileSize(wal, &after) == SQLITE_OK && after == target, "Hint did not reserve 16 MiB");
		file.clear();
		file.seekg(0);
		std::string check(size_t(before), '\0');
		Require(bool(file.read(&check[0], before)) && check == prefix, "Allocation modified committed WAL bytes");
		Exec(db, "COMMIT; INSERT INTO probe VALUES(2,'ack')");
		Require(sqlite3_close(db) == SQLITE_OK, "Close allocated WAL");
		db = nullptr;
		Require(sqlite3_open(argv[1], &db) == SQLITE_OK, "Reopen probe");
		sqlite3_stmt *row = nullptr;
		Require(sqlite3_prepare_v2(db, "SELECT group_concat(value) FROM probe ORDER BY id", -1, &row, nullptr) ==
		            SQLITE_OK,
		        "Prepare reopened content");
		Require(sqlite3_step(row) == SQLITE_ROW &&
		            std::string(reinterpret_cast<const char *>(sqlite3_column_text(row, 0))) == "base,ack",
		        "Allocated WAL changed acknowledged bytes");
		sqlite3_finalize(row);
		Require(sqlite3_close(db) == SQLITE_OK, "Close reopened probe");
		db = nullptr;
		std::cout << "{\"status\":\"PASS\",\"before_bytes\":" << before << ",\"reserved_bytes\":" << after
		          << ",\"allocation_pwrite_calls\":" << writes << ",\"allocation_requested_bytes\":" << bytes
		          << ",\"prefix_unchanged\":true,\"reopened_content\":true}\n";
	} catch (const std::exception &error) {
		if (db)
			sqlite3_close_v2(db);
		std::cerr << error.what() << '\n';
		return 1;
	}
}
