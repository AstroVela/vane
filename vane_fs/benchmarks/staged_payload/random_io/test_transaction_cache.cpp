// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <iostream>

static std::atomic<bool> count {false}, fail_commit {false};
static std::atomic<int> preparations {0}, executions {0};
static bool Control(const char *sql) {
	return sql && (!std::strcmp(sql, "BEGIN") || !std::strcmp(sql, "BEGIN IMMEDIATE") || !std::strcmp(sql, "COMMIT"));
}
static void Require(bool value, const char *message) {
	if (!value)
		throw std::runtime_error(message);
}
extern "C" int __real_sqlite3_prepare_v2(sqlite3 *, const char *, int, sqlite3_stmt **, const char **);
extern "C" int __wrap_sqlite3_prepare_v2(sqlite3 *db, const char *sql, int n, sqlite3_stmt **stmt, const char **tail) {
	if (count && Control(sql))
		++preparations;
	return __real_sqlite3_prepare_v2(db, sql, n, stmt, tail);
}
extern "C" int __real_sqlite3_exec(sqlite3 *, const char *, int (*)(void *, int, char **, char **), void *, char **);
extern "C" int __wrap_sqlite3_exec(sqlite3 *db, const char *sql, int (*callback)(void *, int, char **, char **),
                                   void *arg, char **error) {
	if (count && Control(sql))
		++executions;
	return __real_sqlite3_exec(db, sql, callback, arg, error);
}
extern "C" int __real_sqlite3_step(sqlite3_stmt *);
extern "C" int __wrap_sqlite3_step(sqlite3_stmt *stmt) {
	if (fail_commit && !std::strcmp(sqlite3_sql(stmt), "COMMIT") && fail_commit.exchange(false))
		return SQLITE_IOERR;
	return __real_sqlite3_step(stmt);
}
template <class F>
static void Fails(F operation, vane_fs::ErrorCode expected) {
	try {
		operation();
	} catch (const vane_fs::Error &error) {
		Require(error.code == expected, error.what());
		return;
	}
	throw std::runtime_error("Expected transaction failure");
}
int main(int argc, char **argv) {
	if (argc != 2)
		return 2;
	try {
		std::filesystem::path root(argv[1]);
		Require(std::filesystem::create_directory(root), "Test directory must not exist");
		struct Cleanup {
			std::filesystem::path root;
			~Cleanup() {
				std::filesystem::remove_all(root);
			}
		} cleanup {root};
		auto path = (root / "workspace.sqlite").string();
		vane_fs::Workspace workspace(path, 50, vane_fs::Durability::Fsync);
		workspace.AcquireMount("main");
		auto session = workspace.Checkout();
		auto file = session->OpenFile("/file", true, true);
		session->WriteInode(file.inode, "warm", 0);
		workspace.Sync();
		Require(session->ReadInode(file.inode, 0, 4) == "warm", "Warm read differs");
		count = true;
		for (int i = 0; i < 64; ++i) {
			auto expected = std::string(4096, char('a' + i % 26));
			session->WriteInode(file.inode, expected, 0);
			Require(session->ReadInode(file.inode, 0, 4096) == expected, "Repeated transaction differs");
		}
		count = false;
		Require(executions == 0 && preparations == 0, "Repeated transactions recompiled control statements");
		auto before = session->ReadInode(file.inode, 0, 4096);
		fail_commit = true;
		Fails([&] { session->WriteInode(file.inode, "fail", 0); }, vane_fs::ErrorCode::Storage);
		Require(!fail_commit && session->ReadInode(file.inode, 0, 4096) == before,
		        "Failed commit did not roll back the mutation");
		session->WriteInode(file.inode, "retry", 0);
		Require(session->ReadInode(file.inode, 0, 5) == "retry", "Commit retry failed");
		// This raw connection only holds a writer lock; it changes no data or
		// publication state. Exercise a real SQLITE_BUSY on cached BEGIN.
		sqlite3 *raw = nullptr;
		Require(sqlite3_open(path.c_str(), &raw) == SQLITE_OK, "Opening lock fixture");
		struct Close {
			sqlite3 *db;
			~Close() {
				sqlite3_close_v2(db);
			}
		} close {raw};
		Require(sqlite3_exec(raw, "BEGIN IMMEDIATE", nullptr, nullptr, nullptr) == SQLITE_OK, "Holding writer lock");
		Fails([&] { session->WriteInode(file.inode, "busy", 0); }, vane_fs::ErrorCode::Busy);
		Require(sqlite3_exec(raw, "ROLLBACK", nullptr, nullptr, nullptr) == SQLITE_OK, "Releasing writer lock");
		Require(session->ReadInode(file.inode, 0, 5) == "retry", "Busy BEGIN changed the file");
		session->WriteInode(file.inode, "final", 0, false, true);
		workspace.Sync();
		session->CloseFile(file.inode);
		workspace.ReleaseMount("main");
		workspace.Close();
		vane_fs::Workspace reopened(path);
		Require(reopened.Checkout()->Read("/file", 0, 5) == "final", "FULL retry did not persist");
		std::cout << "Cached control statements, failed commit rollback, busy BEGIN and FULL retry passed\n";
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
