// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <iostream>
#include <vector>

using namespace vane_fs;
static thread_local bool counting = false;
static thread_local int64_t calls = 0, steps = 0;
extern "C" int __real_sqlite3_step(sqlite3_stmt *);
extern "C" int __wrap_sqlite3_step(sqlite3_stmt *statement) {
	if (counting)
		sqlite3_stmt_status(statement, SQLITE_STMTSTATUS_VM_STEP, 1);
	auto code = __real_sqlite3_step(statement);
	if (counting) {
		++calls;
		steps += sqlite3_stmt_status(statement, SQLITE_STMTSTATUS_VM_STEP, 1);
	}
	return code;
}
static void Require(bool value, const char *message) {
	if (!value)
		throw std::runtime_error(message);
}
struct SQL {
	sqlite3 *db = nullptr;
	explicit SQL(const std::string &path) {
		Require(sqlite3_open(path.c_str(), &db) == SQLITE_OK, "Open inspector");
		sqlite3_busy_timeout(db, 5000);
	}
	~SQL() {
		sqlite3_close_v2(db);
	}
	void Exec(const char *sql) {
		Require(sqlite3_exec(db, sql, nullptr, nullptr, nullptr) == SQLITE_OK, sqlite3_errmsg(db));
	}
	int64_t Value(const char *sql) {
		sqlite3_stmt *statement = nullptr;
		Require(sqlite3_prepare_v2(db, sql, -1, &statement, nullptr) == SQLITE_OK, sqlite3_errmsg(db));
		Require(sqlite3_step(statement) == SQLITE_ROW, "Missing result");
		auto value = sqlite3_column_int64(statement, 0);
		sqlite3_finalize(statement);
		return value;
	}
	bool HasIndex() {
		return Value("SELECT count(*) FROM sqlite_schema WHERE type='index' AND name='block_versions_high'") == 1;
	}
	void Integrity() {
		Require(Value("SELECT quick_check='ok' FROM pragma_quick_check") == 1, "Quick check failed");
		Require(Value("SELECT count(*) FROM pragma_foreign_key_check") == 0, "Foreign key check failed");
	}
};
static void Upgrade(const std::string &path, bool rollback) {
	std::string snapshot;
	auto bytes = std::string(1024 * 1024, 'a');
	{
		Workspace workspace(path, 5000, Durability::Fsync);
		auto main = workspace.Checkout();
		main->WriteFile("/file", bytes);
		snapshot = workspace.Snapshot();
		main->Write("/file", std::string(4096, 'b'), 4096);
		workspace.Close();
	}
	SQL inspect(path);
	inspect.Exec("CREATE INDEX IF NOT EXISTS block_versions_high ON block_versions(inode,block,high)");
	Require(inspect.HasIndex(), "Missing old index fixture");
	if (rollback) {
		inspect.Exec("CREATE TRIGGER reject_owner BEFORE INSERT ON owners BEGIN SELECT RAISE(ABORT,'injected owner "
		             "failure'); END");
		bool failed = false;
		try {
			Workspace rejected(path, 5000, Durability::Fsync);
		} catch (const Error &) {
			failed = true;
		}
		Require(failed, "Constructor failure not injected");
		Require(inspect.HasIndex(), "Index removal escaped startup rollback");
		inspect.Exec("DROP TRIGGER reject_owner");
	}
	Workspace workspace(path, 5000, Durability::Fsync);
	Require(!inspect.HasIndex(), "Old block high index retained");
	auto frozen = workspace.OpenSnapshot(snapshot), main = workspace.Checkout();
	Require(frozen->Read("/file") == bytes, "Upgrade changed snapshot content");
	bytes.replace(4096, 4096, std::string(4096, 'b'));
	Require(main->Read("/file") == bytes, "Upgrade changed current content");
	main->Truncate("/file", 512 * 1024);
	main->Unlink("/file");
	workspace.CollectGarbage();
	Require(frozen->Read("/file") == std::string(1024 * 1024, 'a'), "GC changed retained content");
	frozen->Close();
	workspace.DropSnapshot(snapshot);
	workspace.CollectGarbage();
	inspect.Integrity();
	workspace.Close();
}
static void History(const std::string &path) {
	Workspace workspace(path, 5000, Durability::Fsync);
	auto main = workspace.Checkout();
	auto bytes = std::string(1024 * 1024, 'h');
	main->WriteFile("/file", bytes);
	std::vector<std::string> snapshots, versions;
	for (int i = 0; i < 128; ++i) {
		auto data = std::string(4096, char(1 + i));
		main->Write("/file", data);
		versions.push_back(data);
		snapshots.push_back(workspace.Snapshot());
	}
	SQL inspect(path);
#ifdef VANE_TEST_INDEX_REMOVED
	Require(!inspect.HasIndex(), "Fresh block high index retained");
#endif
	for (int i : {0, 63, 127}) {
		auto frozen = workspace.OpenSnapshot(snapshots[i]);
		Require(frozen->Read("/file", 0, 4096) == versions[i], "Historical read mismatch");
		frozen->Close();
	}
	counting = true;
	for (int i = 0; i < 100; ++i)
		Require(main->Read("/file", 0, 4096) == versions.back(), "Current read mismatch");
	counting = false;
	std::cout << "history sqlite_step_calls=" << calls << " vm_steps=" << steps << '\n';
	for (const auto &snapshot : snapshots)
		workspace.DropSnapshot(snapshot);
	workspace.CollectGarbage();
	Require(main->Read("/file", 0, 4096) == versions.back(), "GC current read mismatch");
	inspect.Integrity();
	workspace.Close();
}
int main(int argc, char **argv) {
	try {
		Require(argc == 3, "Expected path and case");
		std::string path = argv[1], mode = argv[2];
		if (mode == "upgrade" || mode == "rollback")
			Upgrade(path, mode == "rollback");
		else if (mode == "history")
			History(path);
		else
			throw std::runtime_error("Unknown case");
		std::cout << mode << " PASS\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
