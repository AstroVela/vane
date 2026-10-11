// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Uninstrumented library calls. Run on tmpfs to isolate software costs; this
// probe does not model durable storage performance or power-loss behavior.
#include "vane_fs/workspace.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

using Clock = std::chrono::steady_clock;
constexpr size_t MIB = 1024 * 1024;

static double CPU() {
	rusage usage {};
	if (getrusage(RUSAGE_SELF, &usage))
		throw std::runtime_error("getrusage failed");
	return usage.ru_utime.tv_sec + usage.ru_stime.tv_sec + (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1e6;
}

static double Seconds(Clock::time_point start) {
	return std::chrono::duration<double>(Clock::now() - start).count();
}

template <class F>
static void Phase(const char *name, F run) {
	auto cpu = CPU();
	auto start = Clock::now();
	run();
	auto wall = Seconds(start);
	cpu = CPU() - cpu;
	std::cout << "{\"phase\":\"" << name << "\",\"seconds\":" << wall << ",\"cpu_seconds\":" << cpu << "}\n";
}

int main(int argc, char **argv) {
	try {
		if (argc != 4)
			throw std::runtime_error("usage: native-probe NEW_DATABASE OFFSET MIB");
		const auto shift = std::stoll(argv[2]);
		const auto count = std::stoll(argv[3]);
		if (shift < 0 || shift >= 4096 || count <= 0 || count > 256)
			throw std::runtime_error("invalid size or offset");
		std::cout.precision(12);
		vane_fs::Workspace workspace(argv[1]);
		workspace.AcquireMount("main");
		auto session = workspace.Checkout();
		auto file = session->OpenFile("/bulk", true, true);
		std::string data(MIB, '\0');
		for (size_t block = 0; block < MIB / 4096; ++block)
			std::fill(data.begin() + block * 4096, data.begin() + (block + 1) * 4096, char(block % 251 + 1));
		if (shift)
			session->WriteInode(file.inode, std::string(shift, 'p'), 0);
		std::vector<double> requests;
		requests.reserve(count);
		Phase("write", [&] {
			for (int64_t i = 0; i < count; ++i) {
				auto start = Clock::now();
				session->WriteInode(file.inode, data, shift + i * MIB);
				requests.push_back(Seconds(start));
			}
		});
		Phase("sync", [&] { workspace.Sync(); });
		for (int64_t i = 0; i < count; ++i)
			if (session->ReadInode(file.inode, shift + i * MIB, MIB) != data)
				throw std::runtime_error("content mismatch");
		if (shift && session->ReadInode(file.inode, 0, shift) != std::string(shift, 'p'))
			throw std::runtime_error("prefix mismatch");
		if (session->ReadInode(file.inode, shift + count * MIB, 1) != "")
			throw std::runtime_error("unexpected suffix");
		Phase("close", [&] {
			session->CloseFile(file.inode);
			session->Close();
			workspace.ReleaseMount("main");
			workspace.Close();
		});
		std::cout << "{\"status\":\"PASS\",\"validated_bytes\":" << shift + count * MIB << ",\"requests\":[";
		for (size_t i = 0; i < requests.size(); ++i) {
			if (i)
				std::cout << ',';
			std::cout << requests[i];
		}
		std::cout << "]}\n";
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
