// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0
#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <atomic>
#include <cerrno>
#include <filesystem>
#include <iostream>
#include <unistd.h>
using namespace vane_fs;
static std::atomic<int> wal_fsync {0}, wal_fdatasync {0}, injected {0};
static std::atomic<bool> fail_wal {false};
static bool Wal(int fd) {
	char text[4096];
	auto path = "/proc/self/fd/" + std::to_string(fd);
	auto n = readlink(path.c_str(), text, sizeof(text));
	return n >= 4 && std::string(text, n).substr(n - 4) == "-wal";
}
extern "C" int __real_fsync(int);
extern "C" int __real_fdatasync(int);
static int Sync(int fd, bool data) {
	auto saved = errno;
	bool wal = Wal(fd);
	errno = saved;
	if (wal) {
		(data ? wal_fdatasync : wal_fsync).fetch_add(1);
		if (fail_wal.exchange(false)) {
			++injected;
			errno = EIO;
			return -1;
		}
	}
	return data ? __real_fdatasync(fd) : __real_fsync(fd);
}
extern "C" int __wrap_fsync(int fd) {
	return Sync(fd, false);
}
extern "C" int __wrap_fdatasync(int fd) {
	return Sync(fd, true);
}
static void Require(bool ok, const char *message) {
	if (!ok)
		throw std::runtime_error(message);
}
template <class F>
static void ExpectStorage(F operation) {
	auto before = injected.load();
	fail_wal = true;
	bool failed = false;
	try {
		operation();
	} catch (const Error &e) {
		Require(e.code == ErrorCode::Storage, "Wrong failure code");
		failed = true;
	}
	fail_wal = false;
	Require(failed, "OS sync error was swallowed");
	Require(injected == before + 1, "WAL syscall fault not reached");
}
static int Count() {
	return wal_fsync + wal_fdatasync;
}
static void Check(const std::string &path, Durability mode) {
	Workspace w(path, 100, mode);
	auto s = w.Checkout();
	s->WriteFile("/file", "initial");
	w.Sync();
	w.AcquireMount("main");
	sqlite3 *pin = nullptr;
	Require(sqlite3_open(path.c_str(), &pin) == SQLITE_OK, "Opening pin");
	Require(sqlite3_exec(pin, "BEGIN; SELECT * FROM format", nullptr, nullptr, nullptr) == SQLITE_OK, "Pinning WAL");
	try {
		auto before = Count();
		s->WriteFile("/file", "pending");
		Require((Count() > before) == (mode == Durability::Strict), "Wrong ordinary commit barrier");
		ExpectStorage([&] { w.Sync(); });
		before = Count();
		w.Sync();
		Require(Count() > before, "Retry missed barrier");
		auto inode = s->OpenFile("/file");
		ExpectStorage([&] { s->WriteInode(inode.inode, "failed!", 0, false, true); });
		Require(s->Read("/file") == "pending", "Failed synchronous write became visible");
		s->WriteInode(inode.inode, "success", 0, false, true);
		ExpectStorage([&] { w.Close(); });
		before = Count();
		w.Close();
		Require(Count() > before, "Close retry missed barrier");
		sqlite3_close(pin);
		pin = nullptr;
	} catch (...) {
		fail_wal = false;
		if (pin)
			sqlite3_close(pin);
		throw;
	}
	Workspace reopened(path);
	Require(reopened.Checkout()->Read("/file") == "success", "Reopen lost acknowledged bytes");
	reopened.Close();
}
int main(int argc, char **argv) {
	if (argc != 3 ||
	    (std::string(argv[2]) != "auto" && std::string(argv[2]) != "fdatasync" && std::string(argv[2]) != "fsync"))
		return 2;
	auto root = std::filesystem::path(argv[1]);
	const std::string expected = argv[2];
	try {
		Require(!std::filesystem::exists(root), "Test root already exists");
		std::filesystem::create_directory(root);
		Check((root / "strict.sqlite").string(), Durability::Strict);
		Check((root / "fsync.sqlite").string(), Durability::Fsync);
		Require((wal_fdatasync > 0) != (wal_fsync > 0), "SQLite WAL must use one observable sync syscall");
		Require(expected == "auto" || (expected == "fdatasync" ? wal_fsync == 0 : wal_fdatasync == 0),
		        "SQLite WAL used unexpected syscall");
		Require(injected == 6, "Missing failure/retry cases");
		std::cout << "{\"wal_fsync\":" << wal_fsync << ",\"wal_fdatasync\":" << wal_fdatasync
		          << ",\"injected_failures\":" << injected << "}\n";
		std::filesystem::remove_all(root);
		return 0;
	} catch (const std::exception &e) {
		fail_wal = false;
		std::cerr << e.what() << '\n';
		return 1;
	}
}
