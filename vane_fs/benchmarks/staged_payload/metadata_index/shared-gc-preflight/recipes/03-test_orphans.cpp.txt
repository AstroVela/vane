// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <set>

using namespace vane_fs;
using Row = std::array<std::string, 6>;
static thread_local bool counting = false;
static thread_local int64_t calls = 0, steps = 0;

extern "C" int __real_sqlite3_step(sqlite3_stmt *statement);
extern "C" int __wrap_sqlite3_step(sqlite3_stmt *statement) {
	if (counting)
		sqlite3_stmt_status(statement, SQLITE_STMTSTATUS_VM_STEP, 1);
	int result = __real_sqlite3_step(statement);
	if (counting) {
		++calls;
		steps += sqlite3_stmt_status(statement, SQLITE_STMTSTATUS_VM_STEP, 1);
	}
	return result;
}

static void Require(bool condition, const char *message) {
	if (!condition)
		throw std::runtime_error(message);
}

template <class F>
static void Expect(ErrorCode code, F operation) {
	try {
		operation();
	} catch (const Error &error) {
		Require(error.code == code, "Wrong error code");
		return;
	}
	throw std::runtime_error("Expected an exception");
}

struct Inspector {
	sqlite3 *db = nullptr;
	explicit Inspector(const std::string &path) {
		Require(sqlite3_open(path.c_str(), &db) == SQLITE_OK, "Open inspector");
	}
	~Inspector() {
		sqlite3_close(db);
	}
	void Exec(const std::string &sql) {
		Require(sqlite3_exec(db, sql.c_str(), nullptr, nullptr, nullptr) == SQLITE_OK, sqlite3_errmsg(db));
	}
	std::vector<Row> Query(const std::string &sql) {
		sqlite3_stmt *statement = nullptr;
		Require(sqlite3_prepare_v2(db, sql.c_str(), -1, &statement, nullptr) == SQLITE_OK, sqlite3_errmsg(db));
		std::vector<Row> rows;
		int result;
		while ((result = sqlite3_step(statement)) == SQLITE_ROW) {
			Row row {};
			for (int i = 0; i < sqlite3_column_count(statement); ++i) {
				auto value = sqlite3_column_text(statement, i);
				row[i] = value ? reinterpret_cast<const char *>(value) : "NULL";
			}
			rows.push_back(row);
		}
		sqlite3_finalize(statement);
		Require(result == SQLITE_DONE, "Inspector query failed");
		return rows;
	}
	std::vector<Row> Blocks(int64_t inode) {
		auto rows = Query("SELECT block,hex(low),hex(high),writer,deleted,payload FROM block_versions WHERE inode=" +
		                  std::to_string(inode));
		std::sort(rows.begin(), rows.end());
		return rows;
	}
};

// Compare every interval, writer, deletion flag and payload reference against
// the former per-key replacement semantics, including invisible future rows.
static std::vector<Row> Expected(const std::vector<Row> &before, const Row &view) {
	std::set<std::string> visible;
	for (const auto &row : before)
		if (row[1] <= view[0] && view[0] < row[2] && row[4] == "0")
			visible.insert(row[0]);
	std::vector<Row> result;
	for (auto row : before) {
		if (!visible.count(row[0]) || !(row[1] < view[1] && view[0] < row[2])) {
			result.push_back(row);
			continue;
		}
		if (row[1] < view[0]) {
			auto left = row;
			left[2] = view[0];
			result.push_back(left);
		}
		if (view[1] < row[2]) {
			row[1] = view[1];
			result.push_back(row);
		}
	}
	for (const auto &block : visible)
		result.push_back({block, view[0], view[1], view[2], "1", "NULL"});
	std::sort(result.begin(), result.end());
	return result;
}

static Row View(Inspector &inspect, const std::string &branch) {
	return inspect.Query("SELECT hex(frontier),hex(high),writer FROM branches WHERE id='" + branch + "'").at(0);
}

static void CheckGone(Inspector &inspect, const std::shared_ptr<Session> &session, int64_t inode,
                      const std::vector<Row> &expected) {
	Require(inspect.Blocks(inode) == expected, "Cleanup differs from per-key interval replacement");
	Require(inspect.Query("SELECT count(*) FROM orphans")[0][0] == "0", "Orphan retained");
	Expect(ErrorCode::Closed, [&] { session->ReadInode(inode, 0, 4096); });
	Require(inspect.Query("PRAGMA quick_check")[0][0] == "ok", "SQLite integrity failure");
}

static void Volume(const std::string &path) {
	Workspace workspace(path, 5000, Durability::Fsync);
	auto session = workspace.Checkout();
	workspace.AcquireMount("main");
	auto file = session->OpenFile("/large", true);
	auto chunk = std::string(8 * 1024 * 1024, 'v');
	for (int i = 0; i < 32; ++i)
		session->WriteInode(file.inode, chunk, int64_t(i) * chunk.size());
	workspace.Sync();
	session->Unlink("/large");
	Inspector inspect(path);
	auto expected = Expected(inspect.Blocks(file.inode), View(inspect, session->Id()));
	auto generation = workspace.GetBranch().generation;
	counting = true;
	auto start = std::chrono::steady_clock::now();
	session->CloseFile(file.inode);
	auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	counting = false;
	std::cout << "{\"sqlite_step_calls\":" << calls << ",\"vm_steps\":" << steps << ",\"seconds\":" << elapsed
	          << ",\"blocks\":65536}" << std::endl;
	CheckGone(inspect, session, file.inode, expected);
	Require(workspace.GetBranch().generation == generation + 1, "Generation increment changed");
	workspace.ReleaseMount("main");
	workspace.CollectGarbage();
	Require(inspect.Query("SELECT count(*) FROM block_payloads")[0][0] == "0", "Unreachable payloads survived GC");
	// The old implementation is an explicit negative control. This bounds SQL
	// work; wall time is diagnostic and is never a correctness assertion.
	Require(calls < 128 && steps < 6000000, "Per-block SQL publication remains");
}

static void Lifetime(const std::string &path, const std::string &scenario, Durability durability) {
	Workspace workspace(path, 5000, durability);
	auto main = workspace.Checkout();
	std::string bytes;
	for (int i = 0; i < 80; ++i)
		bytes += std::string(4096, i % 7 ? char(i + 1) : '\0');
	main->WriteFile("/file", bytes);
	std::shared_ptr<Session> frozen;
	std::string branch = "main";
	if (scenario == "snapshot" || scenario == "future") {
		frozen = workspace.OpenSnapshot(workspace.Snapshot());
		branch = workspace.Fork("main", "child").id;
		auto child = workspace.Checkout(branch);
		child->Write("/file", std::string(8192, 'c'), 4096);
		bytes.replace(4096, 8192, std::string(8192, 'c'));
	} else {
		// Old tombstones and sparse holes must not be rewritten as live blocks.
		main->Truncate("/file", 20 * 4096);
		main->Truncate("/file", bytes.size());
		bytes.replace(20 * 4096, bytes.size() - 20 * 4096, std::string(bytes.size() - 20 * 4096, '\0'));
		main->Write("/file", "tail", 75 * 4096);
		bytes.replace(75 * 4096, 4, "tail");
	}
	auto session = workspace.Checkout(branch);
	workspace.AcquireMount(branch);
	auto file = session->OpenFile("/file");
	session->OpenInode(file.inode);
	Inspector inspect(path);
	auto view = View(inspect, session->Id());
	if (scenario == "future") {
		auto future = view[0];
		// The fork point has ample room before high; this is point + 1.
		for (int i = 63; i >= 0; --i) {
			auto digit = std::string("0123456789ABCDEF").find(future[i]);
			future[i] = "0123456789ABCDEF"[(digit + 1) % 16];
			if (digit != 15)
				break;
		}
		Require(future < view[1], "No room for future fixture");
		// Add a future tombstone inside a currently visible inherited version.
		inspect.Exec(
		    "CREATE TEMP TABLE saved AS SELECT * FROM block_versions WHERE inode=" + std::to_string(file.inode) +
		    " AND block=4 AND low<=X'" + view[0] + "' AND high>X'" + view[0] + "'");
		inspect.Exec("UPDATE block_versions SET high=X'" + future + "' WHERE inode=" + std::to_string(file.inode) +
		             " AND block=4 AND low<=X'" + view[0] + "' AND high>X'" + view[0] + "'");
		inspect.Exec("INSERT INTO block_versions SELECT inode,block,X'" + future + "',X'" + view[1] +
		             "',writer,1,NULL FROM saved");
		inspect.Exec("INSERT INTO block_versions SELECT inode,block,X'" + view[1] +
		             "',high,writer,deleted,payload "
		             "FROM saved WHERE high>X'" +
		             view[1] + "'");
		// Future-only keys are invisible now and must remain untouched.
		inspect.Exec("INSERT INTO block_versions SELECT inode,999,X'" + future + "',X'" + view[1] +
		             "',writer,deleted,payload FROM saved; DROP TABLE saved");
	}
	session->Unlink("/file");
	session->CloseFile(file.inode);
	Require(session->ReadInode(file.inode, 0, bytes.size()) == bytes, "Partial close changed open-unlinked bytes");
	auto before = inspect.Blocks(file.inode);
	auto expected = Expected(before, view);
	auto generation = workspace.GetBranch(branch).generation;
	if (scenario == "rollback") {
		for (const auto &fault : {"update", "generation"}) {
			if (std::string(fault) == "update") {
				inspect.Exec("CREATE TABLE fault_counter AS SELECT 0 AS n;"
				             "CREATE TRIGGER fail_update BEFORE UPDATE ON block_versions WHEN NEW.deleted=1 BEGIN "
				             "UPDATE fault_counter SET n=n+1; SELECT CASE WHEN (SELECT n FROM fault_counter)=10 "
				             "THEN RAISE(ABORT,'partial tombstone failure') END; END;"
				             "CREATE TRIGGER fail_insert BEFORE INSERT ON block_versions WHEN NEW.deleted=1 BEGIN "
				             "UPDATE fault_counter SET n=n+1; SELECT CASE WHEN (SELECT n FROM fault_counter)=10 "
				             "THEN RAISE(ABORT,'partial tombstone failure') END; END;");
			} else
				inspect.Exec("CREATE TRIGGER fail_generation BEFORE UPDATE ON branches BEGIN "
				             "SELECT RAISE(ABORT,'generation failure'); END");
			Expect(ErrorCode::Storage, [&] { session->CloseFile(file.inode); });
			Require(inspect.Blocks(file.inode) == before, "Failed cleanup left partial block changes");
			Require(workspace.GetBranch(branch).generation == generation, "Failed cleanup changed generation");
			Require(session->ReadInode(file.inode, 0, bytes.size()) == bytes, "Failed close lost its handle");
			Require(inspect.Query("SELECT count(*) FROM orphans")[0][0] == "1", "Failed cleanup lost orphan");
			inspect.Exec(std::string(fault) == "update"
			                 ? "DROP TRIGGER fail_update; DROP TRIGGER fail_insert; DROP TABLE fault_counter"
			                 : "DROP TRIGGER fail_generation");
		}
	}
	if (scenario == "lease")
		workspace.ReleaseMount(branch);
	else {
		session->CloseFile(file.inode);
		workspace.ReleaseMount(branch);
	}
	CheckGone(inspect, session, file.inode, expected);
	Require(workspace.GetBranch(branch).generation == generation + 1, "Cleanup changed generation count");
	workspace.CollectGarbage();
	if (frozen) {
		std::string original;
		for (int i = 0; i < 80; ++i)
			original += std::string(4096, i % 7 ? char(i + 1) : '\0');
		Require(frozen->Read("/file") == original && main->Read("/file") == original,
		        "Cleanup crossed snapshot or parent");
		frozen->Close();
	}
	workspace.Sync();
	workspace.Close();
	Workspace reopened(path);
	Expect(ErrorCode::NotFound, [&] { reopened.Checkout(branch)->Read("/file"); });
}

static void Recovery(const std::string &path) {
	{
		Workspace workspace(path);
		workspace.Checkout()->WriteFile("/file", std::string(80 * 4096, 'r'));
	}
	auto pid = fork();
	Require(pid >= 0, "fork");
	if (!pid) {
		try {
			Workspace abandoned(path);
			abandoned.AcquireMount("main");
			auto session = abandoned.Checkout();
			session->OpenFile("/file");
			session->Unlink("/file");
			abandoned.Sync();
			_exit(0);
		} catch (...) {
			_exit(1);
		}
	}
	int status;
	Require(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0, "Abandoned owner failed");
	Inspector inspect(path);
	auto branch = inspect.Query("SELECT id FROM branches WHERE name='main'")[0][0];
	auto inode = std::stoll(inspect.Query("SELECT inode FROM orphans")[0][0]);
	auto expected = Expected(inspect.Blocks(inode), View(inspect, branch));
	Workspace recovered(path);
	recovered.RecoverOwners();
	Require(inspect.Blocks(inode) == expected, "Recovery changed tombstone intervals");
	Require(inspect.Query("SELECT count(*) FROM orphans")[0][0] == "0", "Dead owner orphan retained");
	recovered.CollectGarbage();
	Require(inspect.Query("SELECT count(*) FROM block_payloads")[0][0] == "0", "Recovery payloads leaked");
}

int main(int argc, char **argv) {
	try {
		Require(argc == 3, "Expected database and case");
		std::string scenario = argv[2];
		if (scenario == "volume")
			Volume(argv[1]);
		else if (scenario == "recovery")
			Recovery(argv[1]);
		else {
			Lifetime(std::string(argv[1]) + "-strict", scenario, Durability::Strict);
			Lifetime(std::string(argv[1]) + "-fsync", scenario, Durability::Fsync);
		}
		std::cout << "Orphan batching " << scenario << " PASS\n";
		return 0;
	} catch (const std::exception &error) {
		counting = false;
		std::cerr << error.what() << '\n';
		return 1;
	}
}
