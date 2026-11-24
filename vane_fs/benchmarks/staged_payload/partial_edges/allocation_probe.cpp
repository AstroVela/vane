// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Counts write syscalls, not physical device writes. FD classification perturbs
// this diagnostic; use separate plain binaries for throughput comparisons.
#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <atomic>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

static std::atomic<bool> measuring {false};
static std::atomic<uint64_t> bytes[4] {};
static void Require(bool value, const char *message) {
	if (!value)
		throw std::runtime_error(message);
}
static bool Ends(const std::string &name, const char *suffix) {
	auto n = std::strlen(suffix);
	return name.size() >= n && name.compare(name.size() - n, n, suffix) == 0;
}
static void Count(int fd, ssize_t written) {
	if (written <= 0 || !measuring.load())
		return;
	char target[4096];
	auto link = "/proc/self/fd/" + std::to_string(fd);
	auto n = readlink(link.c_str(), target, sizeof(target));
	Require(n > 0, "Classify owned write descriptor");
	std::string name(target, size_t(n));
	int kind = Ends(name, "-wal") ? 0 : Ends(name, ".sqlite") ? 1 : Ends(name, ".payload") ? 2 : 3;
	bytes[kind].fetch_add(uint64_t(written));
}
extern "C" ssize_t __real_pwrite(int, const void *, size_t, off_t);
extern "C" ssize_t __real_pwrite64(int, const void *, size_t, off64_t);
extern "C" ssize_t __wrap_pwrite(int fd, const void *data, size_t size, off_t offset) {
	auto written = __real_pwrite(fd, data, size, offset);
	Count(fd, written);
	return written;
}
extern "C" ssize_t __wrap_pwrite64(int fd, const void *data, size_t size, off64_t offset) {
	auto written = __real_pwrite64(fd, data, size, offset);
	Count(fd, written);
	return written;
}
int main(int argc, char **argv) {
	try {
		Require(argc == 3, "Expected new database path and aligned/split mode");
		std::string mode(argv[2]);
		Require(mode == "aligned" || mode == "split", "Unknown mode");
		constexpr size_t MIB = 1024 * 1024;
		std::string data(8 * MIB, '\0');
		for (size_t i = 0; i < data.size(); ++i)
			data[i] = char((i / 4096) % 251 + 1);
		{
			vane_fs::Workspace workspace(argv[1], 5000, vane_fs::Durability::Fsync);
			workspace.AcquireMount("main");
			auto session = workspace.Checkout();
			auto file = session->OpenFile("/file", true, true);
			measuring = true;
			for (int application = 0; application < 4; ++application) {
				size_t used = 0;
				while (used < data.size()) {
					size_t amount = std::min(MIB, data.size() - used);
					if (mode == "split" && !used)
						amount -= 48;
					session->WriteInode(file.inode, data.substr(used, amount),
					                    int64_t(application * data.size() + used));
					used += amount;
				}
			}
			workspace.Sync();
			for (int i = 0; i < 4; ++i)
				Require(session->ReadInode(file.inode, int64_t(i * data.size()), int64_t(data.size())) == data,
				        "Content bytes");
			session->CloseFile(file.inode);
			session->Close();
			workspace.ReleaseMount("main");
			workspace.Close();
			measuring = false;
		}
		sqlite3 *db = nullptr;
		Require(sqlite3_open_v2(argv[1], &db, SQLITE_OPEN_READONLY, nullptr) == SQLITE_OK, "Inspect database");
		sqlite3_stmt *statement = nullptr;
		const char *sql =
		    "SELECT length(p.data),count(*) FROM block_versions v JOIN block_payloads p ON p.id=v.payload "
		    "JOIN branches b ON b.name='main' AND v.low<=b.frontier AND v.high>b.frontier "
		    "WHERE v.deleted=0 GROUP BY length(p.data)";
		Require(sqlite3_prepare_v2(db, sql, -1, &statement, nullptr) == SQLITE_OK, "Inspect live payloads");
		std::cout << "{\"mode\":\"" << mode << "\",\"logical_bytes\":" << 32 * MIB << ",\"live_payload_lengths\":[";
		bool comma = false;
		int rc;
		while ((rc = sqlite3_step(statement)) == SQLITE_ROW) {
			if (comma)
				std::cout << ',';
			comma = true;
			std::cout << '[' << sqlite3_column_int64(statement, 0) << ',' << sqlite3_column_int64(statement, 1) << ']';
		}
		Require(rc == SQLITE_DONE, "Read payload counts");
		sqlite3_finalize(statement);
		sqlite3_close(db);
		std::cout << "],\"wal_write_bytes\":" << bytes[0] << ",\"database_write_bytes\":" << bytes[1]
		          << ",\"payload_write_bytes\":" << bytes[2] << ",\"other_write_bytes\":" << bytes[3]
		          << ",\"content_verified\":true}\n";
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
