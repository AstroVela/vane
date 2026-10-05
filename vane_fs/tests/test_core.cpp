// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <sqlite3.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <thread>

using namespace vane_fs;

static void Require(bool condition, const char *message) {
	if (!condition) {
		throw std::runtime_error(message);
	}
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

static void SQL(const std::string &path, const std::string &sql) {
	sqlite3 *db = nullptr;
	Require(sqlite3_open(path.c_str(), &db) == SQLITE_OK, "Opening fault-injection connection");
	int code = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, nullptr);
	std::string error = sqlite3_errmsg(db);
	sqlite3_close(db);
	if (code != SQLITE_OK) {
		throw std::runtime_error(error);
	}
}

int main() {
	auto root = std::filesystem::temp_directory_path() /
	            ("vane-fs-native-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
	try {
		std::filesystem::create_directory(root);
		auto path = (root / "workspace.sqlite").string();
		{
			Workspace workspace(path);
			auto main = workspace.Checkout();
			main->MakeDirectory("/data");
			main->WriteFile("/data/file", std::string(8192, 'a'));
			auto child = workspace.Fork("main", "child");
			auto session = workspace.Checkout(child.id);
			session->Write("/data/file", "changed", 4094);
			Require(main->Read("/data/file") == std::string(8192, 'a'), "Parent changed after child write");
			auto frozen = workspace.Snapshot(child.id);
			auto snapshot = workspace.OpenSnapshot(frozen);
			session->Truncate("/data/file", 10);
			session->Truncate("/data/file", 9000);
			Require(session->Read("/data/file", 10) == std::string(8990, '\0'), "Truncated bytes reappeared");
			Require(snapshot->Read("/data/file", 4094, 7) == "changed", "Snapshot changed");
			Expect(ErrorCode::ReadOnly, [&] { snapshot->WriteFile("/data/new", "x"); });
			Expect(ErrorCode::Busy, [&] { workspace.DropSnapshot(frozen); });
			snapshot->Close();
			workspace.DropSnapshot(frozen);
			auto preview = workspace.PreviewMerge(child.id, "main");
			workspace.Merge(preview);
			Require(main->Read("/data/file", 10) == std::string(8990, '\0'), "Merge did not publish");
			Expect(ErrorCode::ReadOnly, [&] { session->WriteFile("/sealed", "x"); });
			workspace.DeleteBranch(child.id);
			workspace.CollectGarbage();
			Require(main->Read("/data/file").size() == 9000, "GC removed merged data");
			main->WriteFile("/rollback", "before");
			auto generation = workspace.GetBranch().generation;
			SQL(path, "CREATE TABLE fault_counter AS SELECT 0 AS writes;"
			          "CREATE TRIGGER fail_payload BEFORE INSERT ON block_payloads BEGIN "
			          "UPDATE fault_counter SET writes=writes+1;"
			          "SELECT CASE WHEN (SELECT writes FROM fault_counter)=2 "
			          "THEN RAISE(ABORT,'injected block failure') END; END");
			Expect(ErrorCode::Storage, [&] { main->WriteFile("/rollback", std::string(16384, 'z')); });
			SQL(path, "DROP TRIGGER fail_payload; DROP TABLE fault_counter");
			Require(main->Read("/rollback") == "before", "Write failure did not roll back");
			Require(workspace.GetBranch().generation == generation, "Failed write advanced branch generation");
			main->WriteFile("/retry", std::string(16384, 'r'));
			Require(main->Read("/retry") == std::string(16384, 'r'), "Failed statement poisoned later writes");
			main->MakeDirectory("/nested");
			main->MakeDirectory("/nested/child");
			Expect(ErrorCode::Invalid, [&] { main->Rename("/nested", "/nested/child/cycle"); });
			Expect(ErrorCode::NotEmpty, [&] { main->RemoveDirectory("/nested"); });
			workspace.AcquireMount("main");
			auto handle = main->OpenFile("/inode-ranges", true, true);
			main->WriteInode(handle.inode, std::string(4096, 'b'), 4096);
			main->TruncateInode(handle.inode, 12288);
			Require(main->ReadInode(handle.inode, 4094, 4100) ==
			            std::string(2, '\0') + std::string(4096, 'b') + std::string(2, '\0'),
			        "Inode range read lost a partial block or sparse boundary");
			main->CloseFile(handle.inode);
			workspace.ReleaseMount("main");
			workspace.Close();
			Expect(ErrorCode::Closed, [&] { main->Stat("/"); });
		}
		{
			Workspace reopened(path);
			Require(reopened.Checkout()->Read("/rollback") == "before", "Reopen lost data");
			reopened.Checkout()->WriteFile("/parallel", std::string(8192, '\0'));
			std::exception_ptr failures[2];
			std::thread writers[2];
			for (int i = 0; i < 2; ++i) {
				writers[i] = std::thread([&, i] {
					try {
						Workspace connection(path);
						auto session = connection.Checkout();
						for (int j = 0; j < 20; ++j) {
							session->Write("/parallel", std::string(4096, char('a' + i)), i * 4096);
						}
					} catch (...) {
						failures[i] = std::current_exception();
					}
				});
			}
			for (auto &writer : writers) {
				writer.join();
			}
			for (const auto &failure : failures) {
				if (failure) {
					std::rethrow_exception(failure);
				}
			}
			Require(reopened.Checkout()->Read("/parallel") == std::string(4096, 'a') + std::string(4096, 'b'),
			        "Concurrent write lost data");
		}
		std::filesystem::remove_all(root);
		std::cout << "VaneFS native isolation, transactions, reopen and concurrency checks passed\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		std::filesystem::remove_all(root);
		return 1;
	}
}
