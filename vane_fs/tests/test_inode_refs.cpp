// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>

using namespace vane_fs;

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
		int code = sqlite3_open(path.c_str(), &db);
		if (code != SQLITE_OK) {
			sqlite3_close(db);
			throw std::runtime_error("Opening inspection connection");
		}
	}
	~Inspector() {
		sqlite3_close(db);
	}
	void Exec(const std::string &sql) {
		int code = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, nullptr);
		Require(code == SQLITE_OK, sqlite3_errmsg(db));
	}
	int64_t Integer(const std::string &sql) {
		sqlite3_stmt *statement = nullptr;
		int code = sqlite3_prepare_v2(db, sql.c_str(), -1, &statement, nullptr);
		Require(code == SQLITE_OK, sqlite3_errmsg(db));
		code = sqlite3_step(statement);
		auto result = sqlite3_column_int64(statement, 0);
		sqlite3_finalize(statement);
		Require(code == SQLITE_ROW, "Inspection query returned no row");
		return result;
	}
};

static void CheckReferences(const std::string &path) {
	Workspace workspace(path, 20);
	auto main = workspace.Checkout();
	auto other = workspace.Checkout();
	main->WriteFile("/file", "retained bytes");
	workspace.Fork("main", "child");
	workspace.AcquireMount("main");
	workspace.AcquireMount("child");
	auto child = workspace.Checkout("child");
	auto inherited = child->OpenFile("/file");
	auto root = main->OpenDirectory("/");
	auto file = main->LookupInode(root.inode, "file");
	Inspector inspect(path);
	auto version = inspect.Integer("PRAGMA data_version");
	for (int i = 0; i < 20; ++i) {
		other->LookupInode(root.inode, "file");
		main->OpenInode(file.inode);
		other->OpenFile("/file");
		main->CloseFile(file.inode, 3);
		other->OpenDirectory("/");
		main->OpenInode(root.inode, true);
		other->CloseFile(root.inode, 2);
	}
	Require(inspect.Integer("PRAGMA data_version") == version, "Repeated references wrote the main database");
	Expect(ErrorCode::Invalid, [&] { main->CloseFile(file.inode, 2); });
	Expect(ErrorCode::NotDirectory, [&] { main->OpenInode(file.inode, true); });
	Expect(ErrorCode::Storage, [&] {
		main->CreateNode(root.inode, "file", false, 0644, false, false, std::numeric_limits<int64_t>::max());
	});
	// A later metadata failure must roll back an increment of an existing pin.
	inspect.Exec("CREATE TRIGGER fail_generation BEFORE UPDATE ON branches "
	             "BEGIN SELECT RAISE(ABORT,'injected generation failure'); END");
	Expect(ErrorCode::Storage, [&] { main->CreateNode(root.inode, "file", false, 0644, false, false, 2); });
	inspect.Exec("DROP TRIGGER fail_generation");
	Expect(ErrorCode::Invalid, [&] { main->CloseFile(file.inode, 2); });
	main->CreateNode(root.inode, "file", false, 0644, false, false, 2);

	// Failure while publishing the first pin must roll back creation and counts.
	inspect.Exec("CREATE TRIGGER fail_pin AFTER INSERT ON open_inodes "
	             "BEGIN SELECT RAISE(ABORT,'injected pin failure'); END");
	Expect(ErrorCode::Storage, [&] { main->OpenFile("/fresh", true); });
	inspect.Exec("DROP TRIGGER fail_pin");
	Expect(ErrorCode::NotFound, [&] { main->Stat("/fresh"); });
	auto fresh = main->OpenFile("/fresh", true);
	main->CloseFile(fresh.inode);
	Expect(ErrorCode::Closed, [&] { main->StatInode(fresh.inode); });

	main->Unlink("/file");
	main->CloseFile(file.inode, 2);
	workspace.CollectGarbage();
	Require(main->ReadInode(file.inode, 0, 100) == "retained bytes", "Partial close lost orphan data");
	Require(main->StatInode(file.inode).links == 0, "Unlinked inode still has links");
	// Failed final release must restore both temporary counts and the durable pin.
	inspect.Exec("CREATE TRIGGER fail_release BEFORE DELETE ON open_inodes "
	             "BEGIN SELECT RAISE(ABORT,'injected release failure'); END");
	Expect(ErrorCode::Storage, [&] { main->CloseFile(file.inode); });
	inspect.Exec("DROP TRIGGER fail_release");
	inspect.Exec("BEGIN IMMEDIATE");
	Expect(ErrorCode::Busy, [&] { main->CloseFile(file.inode); });
	inspect.Exec("ROLLBACK");
	Require(main->ReadInode(file.inode, 0, 100) == "retained bytes", "Failed close lost its reference");
	main->CloseFile(file.inode);
	Expect(ErrorCode::Closed, [&] { main->StatInode(file.inode); });
	Require(inspect.Integer("SELECT count(*) FROM orphans") == 0, "Final close retained an orphan");
	Require(child->ReadInode(inherited.inode, 0, 100) == "retained bytes", "Release crossed branch boundary");
	child->CloseFile(inherited.inode);
	workspace.ReleaseMount("child");

	// Lease release rolls back local counts too, and a new lease starts empty.
	inspect.Exec("CREATE TRIGGER fail_unmount BEFORE DELETE ON mounts "
	             "BEGIN SELECT RAISE(ABORT,'injected unmount failure'); END");
	Expect(ErrorCode::Storage, [&] { workspace.ReleaseMount("main"); });
	inspect.Exec("DROP TRIGGER fail_unmount");
	main->OpenInode(root.inode, true);
	main->CloseFile(root.inode);
	Require(main->StatInode(root.inode).is_directory, "Failed unmount lost references");
	workspace.ReleaseMount("main");
	Expect(ErrorCode::Closed, [&] { main->StatInode(root.inode); });
	workspace.AcquireMount("main");
	main->OpenDirectory("/");
	main->CloseFile(root.inode);
	Expect(ErrorCode::Closed, [&] { main->StatInode(root.inode); });
	workspace.ReleaseMount("main");
	Require(inspect.Integer("SELECT count(*) FROM open_inodes") == 0, "References leaked across leases");
}

int main() {
	auto root = std::filesystem::temp_directory_path() /
	            ("vane-fs-inode-refs-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
	try {
		std::filesystem::create_directory(root);
		CheckReferences((root / "workspace.sqlite").string());
		std::filesystem::remove_all(root);
		std::cout << "Inode reference persistence, rollback, orphan lifetime and lease isolation passed\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		std::filesystem::remove_all(root);
		return 1;
	}
}
