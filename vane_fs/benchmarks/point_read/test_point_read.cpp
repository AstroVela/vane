// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

static thread_local bool measured = false;
static thread_local int steps = 0;
extern "C" int __real_sqlite3_step(sqlite3_stmt *statement);
extern "C" int __wrap_sqlite3_step(sqlite3_stmt *statement) {
	auto sql = sqlite3_sql(statement);
	bool count = measured && sql && std::strncmp(sql, "SELECT b.block,p.data", 20) == 0;
	auto before = count ? sqlite3_stmt_status(statement, SQLITE_STMTSTATUS_VM_STEP, 0) : 0;
	auto rc = __real_sqlite3_step(statement);
	if (count)
		steps += sqlite3_stmt_status(statement, SQLITE_STMTSTATUS_VM_STEP, 0) - before;
	return rc;
}
static void Require(bool condition, const char *message) {
	if (!condition)
		throw std::runtime_error(message);
}
int main(int argc, char **argv) {
	try {
		Require(argc == 2 && !std::filesystem::exists(argv[1]), "Expected a new owned test directory");
		std::filesystem::create_directory(argv[1]);
		{
			vane_fs::Workspace workspace(std::string(argv[1]) + "/db.sqlite", 5000, vane_fs::Durability::Fsync);
			auto main = workspace.Checkout();
			const std::string original = std::string(4096, 'a') + std::string(4096, '\0') + "tail";
			main->WriteFile("/file", original);
			auto oldest = workspace.Snapshot();
			auto reader = workspace.OpenSnapshot(oldest);
			for (int i = 0; i < 16; ++i) {
				main->Write("/file", std::string(4096, char('b' + i)), 0);
				workspace.Snapshot();
			}
			measured = true;
			auto answer = reader->Read("/file", 3, 100);
			measured = false;
			Require(answer == std::string(100, 'a'), "Old snapshot bytes");
			Require(steps > 0 && steps < 64, "Single-block snapshot read scanned future versions");
			Require(reader->Read("/file", 4095, 2) == std::string("a\0", 2), "Cross-block boundary");
			Require(reader->Read("/file", 4096, 4096) == std::string(4096, '\0'), "Sparse block");
			Require(reader->Read("/file", 8193, 4096) == "ail", "Clipped EOF");
			Require(reader->Read("/file", 8196, 4096).empty(), "Empty EOF");
			Require(main->Read("/file", 0, 4096) == std::string(4096, 'q'), "Current version bytes");
			auto child = workspace.Checkout(workspace.Fork("main", "child").id);
			child->Write("/file", "child", 4090);
			Require(child->Read("/file", 4088, 8) == "qqchildq", "Inherited partial write");
			Require(main->Read("/file", 4088, 8) == std::string(8, 'q'), "Parent isolation");
			workspace.CollectGarbage();
			Require(reader->Read("/file") == original, "Pinned snapshot after GC");
			workspace.Sync();
			child->Close();
			reader->Close();
			main->Close();
			workspace.Close();
		}
		std::filesystem::remove_all(argv[1]);
		std::cout << "point-read steps=" << steps << " PASS\n";
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
