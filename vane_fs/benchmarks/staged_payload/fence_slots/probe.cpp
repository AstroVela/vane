// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include "wait_trace.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sqlite3.h>
#include <time.h>

using namespace vane_fs;
static constexpr int64_t MIB = 1024 * 1024;
static void Require(bool value, const char *message) {
	if (!value)
		throw std::runtime_error(message);
}
static int64_t Now() {
	return fuse_waits::Now(CLOCK_MONOTONIC);
}
int main(int argc, char **argv) {
	try {
		Require(argc == 5, "Expected database, fresh|reused|gc, original|materialized, result JSON");
		std::string path = argv[1], state = argv[2], variant = argv[3];
		Require(state == "fresh" || state == "reused" || state == "gc", "Invalid preparation");
		Require(variant == "original" || variant == "materialized", "Invalid variant");
		std::string block;
		for (int i = 0; i < 2048; ++i)
			block += std::string(4096, char(i % 251 + 1));
		fuse_waits::Install(path.c_str());
		auto preparation_begin = Now();
		std::string snapshot;
		CollectionResult gc;
		if (state != "fresh") {
			fuse_waits::Root root("prepare");
			Workspace setup(path, 5000, Durability::Fsync);
			auto session = setup.Checkout();
			session->WriteFile("/retained", std::string(MIB, 'g'));
			setup.Sync();
			if (state == "gc") {
				snapshot = setup.Snapshot();
				session->WriteFile("/retained", std::string(MIB, 'n'));
				setup.Fork("main", "discarded");
				auto discarded = setup.Checkout("discarded");
				for (int i = 0; i < 8; ++i)
					discarded->WriteFile("/junk-" + std::to_string(i), std::string(MIB, char('a' + i)));
				discarded->Close();
				setup.DeleteBranch("discarded");
				gc = setup.CollectGarbage();
				Require(gc.versions && gc.payloads, "Preparation did not reclaim payloads/versions");
			}
			session->Close();
			setup.Close();
		}
		auto begin = Now();
		std::unique_ptr<Workspace> workspace;
		std::shared_ptr<Session> session;
		int64_t parent, file;
		{
			fuse_waits::Root root("open_workspace");
			workspace = std::make_unique<Workspace>(path, 5000, Durability::Fsync);
			session = workspace->Checkout();
			workspace->AcquireMount("main");
			parent = session->OpenDirectory("/").inode;
			file = session->CreateNode(parent, "fixture", false, 0600, true, false, 2).inode;
		}
		auto fence_before = std::filesystem::file_size(path + ".stage-fence");
		Require(fence_before == ((state == "fresh" && variant == "original") ? 4096 : 8192),
		        "Unexpected initial fence size");
		auto writes_begin = Now();
		int64_t offset = 0;
		int requests = 0;
		while (offset < 8 * MIB) {
			auto size = std::min(offset == 0 ? MIB - 48 : MIB, 8 * MIB - offset);
			fuse_waits::Root root("write", offset, size, file);
			session->WriteInode(file, block.substr(size_t(offset), size_t(size)), offset);
			offset += size;
			++requests;
		}
		auto writes_end = Now();
		Require(requests == 9, "Original buffer-offset request split differs");
		{
			fuse_waits::Root root("fsync", -1, -1, file);
			session->StatInode(file);
			workspace->Sync();
		}
		auto sync_end = Now();
		{
			fuse_waits::Root root("close");
			session->CloseFile(file, 2);
			session->CloseFile(parent);
			workspace->ReleaseMount("main");
			session->Close();
			workspace->Close();
		}
		auto end = Now();
		auto fence_after = std::filesystem::file_size(path + ".stage-fence");
		Require(fence_after == 8192, "Final fence size differs");
		{
			fuse_waits::Root root("validate");
			Workspace reopened(path);
			auto read = reopened.Checkout();
			Require(read->Read("/fixture") == block, "First 8 MiB content differs after close");
			if (state != "fresh")
				Require(read->Read("/retained") == std::string(MIB, state == "gc" ? 'n' : 'g'),
				        "Retained live content differs");
			if (state == "gc") {
				auto pinned = reopened.OpenSnapshot(snapshot);
				Require(pinned->Read("/retained") == std::string(MIB, 'g'), "Retained snapshot differs");
				pinned->Close();
			}
			read->Close();
			reopened.Close();
			sqlite3 *db = nullptr;
			Require(sqlite3_open(path.c_str(), &db) == SQLITE_OK, "Open integrity inspector");
			sqlite3_stmt *statement = nullptr;
			Require(sqlite3_prepare_v2(db, "PRAGMA quick_check", -1, &statement, nullptr) == SQLITE_OK,
			        "Prepare integrity check");
			Require(sqlite3_step(statement) == SQLITE_ROW &&
			            std::string(reinterpret_cast<const char *>(sqlite3_column_text(statement, 0))) == "ok",
			        "SQLite integrity differs");
			sqlite3_finalize(statement);
			sqlite3_close(db);
		}
		std::ofstream result(argv[4]);
		result << "{\"status\":\"PASS\",\"state\":\"" << state << "\",\"variant\":\"" << variant
		       << "\",\"preparation_begin_ns\":" << preparation_begin << ",\"begin_ns\":" << begin
		       << ",\"writes_begin_ns\":" << writes_begin << ",\"writes_end_ns\":" << writes_end
		       << ",\"sync_end_ns\":" << sync_end << ",\"end_ns\":" << end << ",\"fence_before\":" << fence_before
		       << ",\"fence_after\":" << fence_after << ",\"requests\":" << requests << ",\"gc\":[" << gc.versions
		       << ',' << gc.payloads << ',' << gc.snapshots << "],\"content_and_integrity\":true}\n";
		Require(bool(result), "Write result");
		std::cout << "PASS " << state << ' ' << variant << '\n';
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
