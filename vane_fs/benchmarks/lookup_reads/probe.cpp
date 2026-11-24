// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

using namespace vane_fs;
using Clock = std::chrono::steady_clock;

int main(int argc, char **argv) {
	try {
		if (argc != 3)
			throw std::runtime_error("usage: lookup-probe database strict|fsync");
		Workspace workspace(argv[1], 10, std::string(argv[2]) == "strict" ? Durability::Strict : Durability::Fsync);
		auto session = workspace.Checkout();
		session->WriteFile("/file", "lookup data");
		workspace.AcquireMount("main");
		auto root = session->OpenDirectory("/");
		for (const auto *kind : {"missing", "retained", "first"}) {
			int iterations = std::string(kind) == "first" ? 512 : 4096;
			int64_t retained = 0;
			if (std::string(kind) == "retained")
				retained = session->LookupInode(root.inode, "file").inode;
			std::vector<int64_t> samples;
			for (int i = 0; i < iterations; ++i) {
				auto start = Clock::now();
				FileStat file;
				try {
					file = session->LookupInode(root.inode, std::string(kind) == "missing" ? "missing" : "file");
					if (std::string(kind) == "missing")
						throw std::runtime_error("Missing lookup returned a file");
				} catch (const Error &error) {
					if (std::string(kind) != "missing" || error.code != ErrorCode::NotFound)
						throw;
				}
				samples.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
				if (std::string(kind) == "first")
					session->CloseFile(file.inode);
			}
			if (retained)
				session->CloseFile(retained, iterations + 1);
			std::cout << "{\"case\":\"" << kind << "\",\"nanoseconds\":[";
			for (size_t i = 0; i < samples.size(); ++i)
				std::cout << (i ? "," : "") << samples[i];
			std::cout << "]}\n";
		}
		session->CloseFile(root.inode);
		workspace.ReleaseMount("main");
		workspace.Close();
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
