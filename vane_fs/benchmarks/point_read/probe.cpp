// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <sys/resource.h>

static constexpr int64_t MIB = 1024 * 1024;
static void Require(bool ok, const char *message) {
	if (!ok)
		throw std::runtime_error(message);
}
static double Cpu() {
	struct rusage usage {};
	Require(getrusage(RUSAGE_THREAD, &usage) == 0, "Thread CPU sampling");
	return usage.ru_utime.tv_sec + usage.ru_stime.tv_sec + (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1e6;
}
int main(int argc, char **argv) {
	try {
		Require(argc == 3, "Expected database path and state");
		std::string state(argv[2]), block(MIB, '\0');
		Require(state == "fresh" || state == "aged" || state == "history", "Unknown state");
		for (int i = 0; i < MIB; ++i)
			block[size_t(i)] = char((i / 4096) % 251 + 1);
		vane_fs::Workspace workspace(argv[1], 5000, vane_fs::Durability::Fsync);
		workspace.AcquireMount("main");
		auto session = workspace.Checkout();
		if (state == "aged") {
			auto guard = session->OpenFile("/guard", true, true);
			for (int i = 0; i < 32; ++i)
				session->WriteInode(guard.inode, block, i * MIB);
			workspace.Sync();
			for (int i = 0; i < 8192; i += 2)
				session->WriteInode(guard.inode, std::string(4096, '\0'), int64_t(i) * 4096);
			workspace.Sync();
			session->CloseFile(guard.inode);
			workspace.CollectGarbage();
		}
		auto file = session->OpenFile("/fixture", true, true);
		std::string expected;
		expected.reserve(64 * MIB);
		for (int i = 0; i < 64; ++i) {
			session->WriteInode(file.inode, block, i * MIB);
			expected += block;
		}
		workspace.Sync();
		std::shared_ptr<vane_fs::Session> oldest;
		if (state == "history") {
			session->CloseFile(file.inode);
			workspace.ReleaseMount("main");
			oldest = workspace.OpenSnapshot(workspace.Snapshot());
			for (int i = 0; i < 16; ++i) {
				session->Write("/fixture", std::string(MIB, char('B' + i)), 0);
				workspace.Snapshot();
			}
			expected.replace(0, MIB, std::string(MIB, 'Q'));
			workspace.Sync();
			workspace.AcquireMount("main");
			Require(session->OpenFile("/fixture").inode == file.inode, "Reopen fixture inode");
		}
		for (int i = 0; i < 64; ++i)
			Require(session->ReadInode(file.inode, i * MIB, MIB) == expected.substr(size_t(i * MIB), MIB),
			        "Warm fixture");
		if (oldest)
			Require(oldest->Read("/fixture", 0, MIB) == block, "Warm old snapshot");
		bool comma = false;
		std::cout << std::setprecision(12) << "{\"state\":" << std::quoted(state) << ",\"samples\":[";
		auto measure = [&](const std::string &name, size_t size, int count, int64_t limit, bool old) {
			std::mt19937 rng(4319);
			std::vector<int64_t> offsets;
			for (int i = 0; i < count; ++i) {
				auto offset = int64_t(rng() % uint32_t(limit / 4096 - 1)) * 4096;
				if (name == "partial")
					offset += 51;
				else if (name == "cross")
					offset += 13;
				else if (name == "sequential")
					offset = int64_t(i) * int64_t(size);
				offsets.push_back(offset);
			}
			auto cpu = Cpu();
			auto start = std::chrono::steady_clock::now();
			for (auto offset : offsets) {
				auto data = old ? oldest->Read("/fixture", offset, int64_t(size))
				                : session->ReadInode(file.inode, offset, int64_t(size));
				const auto &reference = old ? block : expected;
				Require(data.size() == size && std::memcmp(data.data(), reference.data() + offset, size) == 0,
				        "Read content");
			}
			auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
			auto cpu_seconds = Cpu() - cpu;
			if (comma)
				std::cout << ',';
			comma = true;
			std::cout << "{\"workload\":" << std::quoted(old ? "old_snapshot" : name) << ",\"requests\":" << count
			          << ",\"bytes\":" << count * size << ",\"seconds\":" << seconds
			          << ",\"cpu_seconds\":" << cpu_seconds << '}';
		};
		measure("sequential", 512 * 1024, 128, 64 * MIB, false);
		for (const auto &name : {"point", "partial", "cross"})
			measure(name, std::string(name) == "partial" ? 128 : 4096, 8192, state == "history" ? MIB : 64 * MIB,
			        false);
		if (oldest)
			measure("point", 4096, 8192, MIB, true);
		std::cout << "],\"content_verified\":true}\n";
		if (oldest)
			oldest->Close();
		session->CloseFile(file.inode);
		session->Close();
		workspace.ReleaseMount("main");
		workspace.Close();
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
