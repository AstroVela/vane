// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "support.hpp"
#include "offsets.hpp"
#include <fstream>
#include <iomanip>

static constexpr int64_t MIB = 1024 * 1024;

int main(int argc, char **argv) {
	try {
		Require(argc == 5, "Expected database, warm|gc, active|paused, output JSON");
		const std::string path = argv[1], state = argv[2], arm = argv[3];
		Require(state == "warm" || state == "gc", "Unknown preparation state");
		Require(arm == "active" || arm == "paused", "Unknown checkpoint arm");
		fuse_waits::Install(path.c_str());
		auto preparation_begin = Now();
		Workspace workspace(path, 5000, Durability::Fsync);
		auto session = workspace.Checkout();
		Require(writer && worker, "Missing original WAL hook context");
		std::string snapshot;
		CollectionResult gc;
		if (state == "gc") {
			fuse_waits::Root root("setup_gc");
			session->WriteFile("/retained", std::string(MIB / 2, 'k'));
			snapshot = workspace.Snapshot();
			session->WriteFile("/retained", std::string(MIB / 2, 'n'));
			workspace.Fork("main", "discarded");
			auto discarded = workspace.Checkout("discarded");
			for (int i = 0; i < 8; ++i)
				discarded->WriteFile("/junk-" + std::to_string(i), std::string(MIB, char('a' + i)));
			discarded->Close();
			workspace.DeleteBranch("discarded");
			gc = workspace.CollectGarbage();
			Require(gc.versions > 0 && gc.payloads > 0, "GC fixture reclaimed no versions/payloads");
		}
		workspace.AcquireMount("main");
		auto parent = session->OpenDirectory("/").inode;
		auto file = session->CreateNode(parent, "fixture", false, 0600, true, false, 2).inode;
		std::string block;
		for (int i = 0; i < 256; ++i)
			block += std::string(4096, char(i % 251 + 1));
		std::string expected;
		for (int i = 0; i < 64; ++i) {
			fuse_waits::Root root("setup_write", int64_t(i) * MIB, MIB, file);
			session->WriteInode(file, block, int64_t(i) * MIB);
			expected += block;
		}
		{
			fuse_waits::Root root("setup_drain");
			Require(session->ReadInode(file, 0, 64 * MIB) == expected, "Prepared fixture differs");
			workspace.Sync();
			// FULL retires all pending payload ranges. A root counter also covers
			// a worker that claimed a range but has not entered its syscall yet.
			Await(fuse_waits::PayloadWorkerIdle);
			Warm();
		}
		auto before = Status();
		Require(before.first == before.second, "Preparation backfill was not drained");
		auto highwater = std::filesystem::file_size(path + "-wal");
		auto sample_size = [&] {
			highwater = std::max(highwater, std::filesystem::file_size(path + "-wal"));
			Require(highwater < uint64_t(Checkpointer::LIMIT_BYTES), "Phase reached WAL admission limit");
		};
		std::vector<std::pair<int64_t, int64_t>> random_operations, small_operations;
		std::vector<int64_t> inodes;
		auto begin = Now();
		if (arm == "active")
			worker->Start();
		auto random_begin = Now();
		for (auto offset : OFFSETS) {
			auto start = Now();
			{
				fuse_waits::Root root("write", offset, 4096, file);
				session->WriteInode(file, std::string(4096, 'z'), offset);
			}
			random_operations.emplace_back(start, Now());
			expected.replace(size_t(offset), 4096, 4096, 'z');
			sample_size();
		}
		auto random_sync_begin = Now();
		{
			fuse_waits::Root root("fsync", -1, -1, file);
			session->StatInode(file);
			workspace.Sync();
		}
		auto random_end = Now();
		auto small_begin = Now();
		for (int i = 0; i < 128; ++i) {
			auto start = Now();
			auto name = "small-" + std::to_string(i);
			{
				fuse_waits::Root root("lookup");
				try {
					session->LookupInode(parent, name);
					throw std::runtime_error("Small file already exists");
				} catch (const Error &error) {
					Require(error.code == ErrorCode::NotFound, "Unexpected lookup failure");
				}
			}
			int64_t inode;
			{
				fuse_waits::Root root("create");
				inode = session->CreateNode(parent, name, false, 0600, true, false, 2).inode;
			}
			{
				fuse_waits::Root root("write", 0, 4096, inode);
				session->WriteInode(inode, std::string(4096, 's'), 0);
			}
			{
				fuse_waits::Root root("fsync", -1, -1, inode);
				session->StatInode(inode);
				workspace.Sync();
			}
			{
				fuse_waits::Root root("flush");
				session->StatInode(inode);
			}
			{
				fuse_waits::Root root("release");
				session->CloseFile(inode);
			}
			small_operations.emplace_back(start, Now());
			inodes.push_back(inode);
			sample_size();
		}
		auto foreground_end = Now();
		worker->Stop();
		worker->CheckError();
		auto after = Status();
		Require(after.first >= 0, "Missing phase WAL status");
		if (arm == "paused")
			Require(after.first > after.second, "Paused arm has no deferred maintenance");
		auto resume_begin = Now();
		{
			fuse_waits::Root root("resume");
			worker->Start();
			Await(CaughtUp); // No additional commit is allowed to wake the worker.
			worker->Stop();
			worker->CheckError();
		}
		auto resume_end = Now();
		auto resumed = Status();
		Require(resumed.first == resumed.second, "Deferred backfill did not complete");
		auto close_begin = Now();
		{
			fuse_waits::Root root("close");
			for (auto inode : inodes)
				session->CloseFile(inode);
			session->CloseFile(file, 2);
			session->CloseFile(parent);
			workspace.ReleaseMount("main");
			session->Close();
			workspace.Close();
		}
		auto end = Now();
		Require(writer == nullptr && worker == nullptr, "Close retained its WAL callback");
		{
			fuse_waits::Root root("validation");
			Workspace reopened(path);
			auto read = reopened.Checkout();
			Require(read->Read("/fixture") == expected, "Random-overwrite content differs after close");
			for (int i = 0; i < 128; ++i)
				Require(read->Read("/small-" + std::to_string(i)) == std::string(4096, 's'),
				        "Small-file content differs");
			if (state == "gc") {
				Require(read->Read("/retained") == std::string(MIB / 2, 'n'), "Retained live data differs");
				auto pinned = reopened.OpenSnapshot(snapshot);
				Require(pinned->Read("/retained") == std::string(MIB / 2, 'k'), "Retained snapshot differs");
				pinned->Close();
			}
			read->Close();
			reopened.Close();
			SQL inspect(path);
			Require(inspect.Text("PRAGMA quick_check") == "ok", "SQLite integrity failed");
		}
		std::ofstream result(argv[4]);
		result << std::setprecision(17) << "{\"status\":\"PASS\",\"state\":\"" << state << "\",\"arm\":\"" << arm
		       << "\",\"preparation_begin_ns\":" << preparation_begin << ",\"begin_ns\":" << begin
		       << ",\"random_begin_ns\":" << random_begin << ",\"random_sync_begin_ns\":" << random_sync_begin
		       << ",\"random_end_ns\":" << random_end << ",\"small_begin_ns\":" << small_begin
		       << ",\"foreground_end_ns\":" << foreground_end << ",\"resume_begin_ns\":" << resume_begin
		       << ",\"resume_end_ns\":" << resume_end << ",\"close_begin_ns\":" << close_begin << ",\"end_ns\":" << end
		       << ",\"wal_before\":[" << before.first << ',' << before.second << "],\"wal_after\":[" << after.first
		       << ',' << after.second << "],\"wal_resumed\":[" << resumed.first << ',' << resumed.second
		       << "],\"wal_highwater_bytes\":" << highwater << ",\"gc\":[" << gc.versions << ',' << gc.payloads << ','
		       << gc.snapshots << "],\"content_and_integrity\":true";
		auto intervals = [&](const char *name, const auto &values) {
			result << ",\"" << name << "\":[";
			for (size_t i = 0; i < values.size(); ++i)
				result << (i ? "," : "") << '[' << values[i].first << ',' << values[i].second << ']';
			result << ']';
		};
		intervals("random_operations", random_operations);
		intervals("small_operations", small_operations);
		result << "}\n";
		Require(bool(result), "Writing probe result");
		std::cout << "PASS " << state << ' ' << arm << '\n';
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
