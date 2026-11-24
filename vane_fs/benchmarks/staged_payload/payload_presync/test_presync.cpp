// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace vane_fs;
namespace fs = std::filesystem;
static constexpr size_t MIB = 1024 * 1024;
static const auto foreground = std::this_thread::get_id();
static std::mutex probe_mutex;
static std::thread::id worker_id;
static std::atomic<bool> armed {false}, first_entered {false}, release_first {false};
static std::atomic<bool> retry_entered {false}, release_retry {false};
static std::atomic<int> payload_calls {0}, passive_calls {0}, fail_call {0}, hold_call {0};
static bool crash_case = false;
static int notify_fd = -1;

static void Require(bool value, const char *message) {
	if (!value)
		throw std::runtime_error(message);
}
template <class F>
static bool Wait(F condition, int milliseconds = 10000) {
	auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(milliseconds);
	while (!condition()) {
		if (std::chrono::steady_clock::now() >= end)
			return false;
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	return true;
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
static std::string Name(int fd) {
	char target[4096];
	auto n = readlink(("/proc/self/fd/" + std::to_string(fd)).c_str(), target, sizeof(target));
	return n > 0 ? std::string(target, size_t(n)) : "";
}
static bool Ends(const std::string &name, const char *suffix) {
	auto length = std::strlen(suffix);
	return name.size() >= length && name.compare(name.size() - length, length, suffix) == 0;
}
extern "C" int __real_fdatasync(int fd);
extern "C" int __wrap_fdatasync(int fd) {
	auto name = Name(fd);
	bool target = false;
	if (armed && Ends(name, ".payload")) {
		std::lock_guard<std::mutex> lock(probe_mutex);
		if (worker_id == std::thread::id() && std::this_thread::get_id() != foreground)
			worker_id = std::this_thread::get_id();
		target = std::this_thread::get_id() == worker_id;
	}
	if (!target)
		return __real_fdatasync(fd);
	int call = ++payload_calls;
	if (crash_case && call == 1) {
		auto rc = __real_fdatasync(fd);
		if (rc)
			return rc;
		fs::copy_file(name, name + "-durable", fs::copy_options::overwrite_existing);
		first_entered = true;
		while (!release_first)
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		return 0;
	}
	if (crash_case && call == 2) {
		char ready = 'x';
		if (write(notify_fd, &ready, 1) != 1)
			_exit(80);
		raise(SIGSTOP);
	}
	if (call == 1) {
		first_entered = true;
		while (!release_first)
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	if (call == hold_call) {
		retry_entered = true;
		while (!release_retry)
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	if (call == fail_call) {
		errno = EIO;
		return -1;
	}
	return __real_fdatasync(fd);
}
extern "C" int __real_sqlite3_wal_checkpoint_v2(sqlite3 *, const char *, int, int *, int *);
extern "C" int __wrap_sqlite3_wal_checkpoint_v2(sqlite3 *db, const char *name, int mode, int *frames, int *copied) {
	auto code = __real_sqlite3_wal_checkpoint_v2(db, name, mode, frames, copied);
	if (armed && std::this_thread::get_id() != foreground && mode == SQLITE_CHECKPOINT_PASSIVE)
		++passive_calls;
	return code;
}

struct SQL {
	sqlite3 *db = nullptr;
	explicit SQL(const std::string &path) {
		Require(sqlite3_open(path.c_str(), &db) == SQLITE_OK, "Open inspector");
		sqlite3_busy_timeout(db, 1000);
		sqlite3_wal_autocheckpoint(db, 0);
	}
	~SQL() {
		sqlite3_close_v2(db);
	}
	void Exec(const char *sql) {
		Require(sqlite3_exec(db, sql, nullptr, nullptr, nullptr) == SQLITE_OK, sqlite3_errmsg(db));
	}
	bool CaughtUp() {
		int frames = -1, copied = -1;
		auto rc = sqlite3_wal_checkpoint_v2(db, "main", SQLITE_CHECKPOINT_NOOP, &frames, &copied);
		Require(rc == SQLITE_OK || rc == SQLITE_BUSY, "Checkpoint status");
		return rc == SQLITE_OK && frames >= 0 && frames == copied;
	}
};
static void Padding(SQL &sql) {
	sql.Exec("PRAGMA synchronous=NORMAL; CREATE TABLE padding(data BLOB);"
	         "INSERT INTO padding VALUES(zeroblob(20971520))");
}
static uint64_t Decode(const unsigned char *data) {
	uint64_t result = 0;
	for (int i = 0; i < 8; ++i)
		result = result * 256 + data[i];
	return result;
}
static bool Pending(const std::string &path) {
	std::array<unsigned char, 8192> bytes {};
	std::ifstream input(path + ".stage-fence", std::ios::binary);
	input.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
	Require(input.gcount() == int64_t(bytes.size()), "Incomplete fence fixture");
	auto slot = Decode(bytes.data() + 32) > Decode(bytes.data() + 4096 + 32) ? 0 : 4096;
	return Decode(bytes.data() + slot + 40) != 0;
}
struct Release {
	std::thread operation;
	~Release() {
		release_first = release_retry = true;
		if (operation.joinable())
			operation.join();
	}
};

static void Concurrent(const std::string &path, const std::string &mode) {
	Workspace workspace(path, 1000, Durability::Fsync);
	auto session = workspace.Checkout();
	if (mode == "gc") {
		session->WriteFile("/dead", std::string(MIB, 'd'));
		session->Unlink("/dead");
	}
	session->WriteFile("/stable", std::string(MIB, 's'));
	workspace.Sync();
	auto snapshot = workspace.Snapshot();
	Release release;
	SQL sql(path);
	session->WriteFile("/pending", std::string(MIB, 'p'));
	Padding(sql);
	if (mode == "presync_error") {
		fail_call = 1;
		hold_call = 2;
	} else if (mode == "final_error") {
		fail_call = 2;
		hold_call = 3;
	}
	armed = true;
	session->WriteFile("/trigger", "inline");
	Require(Wait([] { return first_entered.load(); }), "Background payload sync never started");
	Require(Pending(path), "Background sync cleared OPEN before it completed");
	if (mode == "presync_error" || mode == "final_error") {
		release_first = true;
		Require(Wait([] { return retry_entered.load(); }), "Failed sync lost its retry");
		Require(Pending(path), "Failed sync published CLEAN");
		Expect(ErrorCode::Storage, [&] { session->WriteFile("/rejected", "no"); });
		Expect(ErrorCode::NotFound, [&] { session->Stat("/rejected"); });
		release_retry = true;
	} else {
		std::atomic<bool> done {false};
		std::exception_ptr failure;
		release.operation = std::thread([&] {
			try {
				if (mode == "barrier")
					workspace.Sync();
				else if (mode == "gc")
					workspace.CollectGarbage();
				else
					session->WriteFile("/append", std::string(MIB, 'a'));
			} catch (...) {
				failure = std::current_exception();
			}
			done = true;
		});
		bool progressed = Wait([&] { return done.load(); }, 1500);
		if (progressed) {
			release.operation.join();
			if (failure)
				std::rethrow_exception(failure);
			Require(payload_calls == 1, "Unexpected background sync while presync was blocked");
			Require(Pending(path) == (mode == "append"), "Foreground operation changed the wrong fence state");
		}
		release_first = true;
		if (release.operation.joinable())
			release.operation.join();
		Require(progressed, "Foreground operation waited for blocked presync");
	}
	Require(Wait([&] { return sql.CaughtUp(); }), "Worker failed to finish backfill");
	Require(!Pending(path), "Successful publication remained OPEN");
	if (mode == "append")
		Require(payload_calls >= 2, "Final protected sync missing after concurrent append");
	if (mode == "barrier" || mode == "gc")
		Require(payload_calls == 1, "Worker ignored a foreground CLEAN fence");
	if (mode == "presync_error" || mode == "final_error")
		Require(payload_calls >= fail_call + 2, "Retry omitted preparatory or authoritative sync");
	Require(session->Read("/pending") == std::string(MIB, 'p'), "Pending content changed");
	if (mode == "append")
		Require(session->Read("/append") == std::string(MIB, 'a'), "Concurrent append changed");
	Require(workspace.OpenSnapshot(snapshot)->Read("/stable") == std::string(MIB, 's'), "Retained snapshot changed");
	workspace.Sync();
	workspace.Close();
	Workspace reopened(path);
	Require(reopened.Checkout()->Read("/pending") == std::string(MIB, 'p'), "Reopen lost acknowledged content");
	std::cout << mode << " payload_calls=" << payload_calls << " passive_calls=" << passive_calls << '\n';
}

static void Clean(const std::string &path) {
	Workspace workspace(path, 1000, Durability::Fsync);
	auto session = workspace.Checkout();
	session->WriteFile("/stable", "inline");
	workspace.Sync();
	SQL sql(path), reader(path);
	Padding(sql);
	reader.Exec("BEGIN; SELECT count(*) FROM padding");
	int frames = -1, copied = -1;
	Require(sqlite3_wal_checkpoint_v2(sql.db, "main", SQLITE_CHECKPOINT_PASSIVE, &frames, &copied) == SQLITE_OK &&
	            frames == copied && frames > 4096,
	        "No retained/backfilled large WAL");
	Release release;
	armed = true;
	session->WriteFile("/inline", "small tail");
	Require(Wait([] { return passive_calls > 0; }), "CLEAN tail did not get prompt checkpoint progress");
	reader.Exec("ROLLBACK");
	Require(Wait([&] { return sql.CaughtUp(); }), "CLEAN tail did not finish");
	Require(payload_calls == 0, "CLEAN tail performed unnecessary payload presync");
	workspace.Close();
}

static void Crash(const std::string &path) {
	std::string snapshot;
	{
		Workspace initial(path);
		initial.Checkout()->WriteFile("/stable", std::string(MIB, 's'));
		snapshot = initial.Snapshot();
		initial.Close();
	}
	int pipes[2];
	Require(pipe(pipes) == 0, "Pipe");
	auto pid = fork();
	Require(pid >= 0, "Fork");
	if (!pid) {
		close(pipes[0]);
		notify_fd = pipes[1];
		crash_case = true;
		try {
			Workspace workspace(path, 1000, Durability::Fsync);
			auto session = workspace.Checkout();
			SQL sql(path);
			session->WriteFile("/pending", std::string(MIB, 'p'));
			Padding(sql);
			armed = true;
			session->WriteFile("/trigger", "inline");
			Require(Wait([] { return first_entered.load(); }), "No first sync");
			session->WriteFile("/append", std::string(MIB, 'a'));
			release_first = true;
			std::this_thread::sleep_for(std::chrono::seconds(20));
			_exit(83);
		} catch (...) {
			_exit(84);
		}
	}
	close(pipes[1]);
	pollfd ready {pipes[0], POLLIN, 0};
	bool reached = poll(&ready, 1, 10000) == 1 && (ready.revents & POLLIN);
	kill(pid, SIGKILL);
	int status;
	Require(waitpid(pid, &status, 0) == pid, "Wait for crash child");
	close(pipes[0]);
	Require(reached && WIFSIGNALED(status), "Final protected sync crash boundary not reached");
	Require(Pending(path), "Presync incorrectly published concurrent appends");
	// Restore only payload bytes captured after the preparatory sync. The
	// later NORMAL append is lost, and OPEN must discard its WAL references.
	fs::copy_file(path + ".payload-durable", path + ".payload", fs::copy_options::overwrite_existing);
	fs::remove(path + "-shm");
	for (int i = 0; i < 2; ++i) {
		Workspace recovered(path);
		auto session = recovered.Checkout();
		Require(session->Read("/stable") == std::string(MIB, 's'), "Recovery lost earlier durable data");
		Require(recovered.OpenSnapshot(snapshot)->Read("/stable") == std::string(MIB, 's'),
		        "Recovery changed snapshot");
		Expect(ErrorCode::NotFound, [&] { session->Read("/pending"); });
		Expect(ErrorCode::NotFound, [&] { session->Read("/append"); });
		recovered.CollectGarbage();
		recovered.Close();
	}
}

int main(int argc, char **argv) {
	try {
		Require(argc == 3, "Expected database and case");
		std::string mode = argv[2];
		if (mode == "clean")
			Clean(argv[1]);
		else if (mode == "crash")
			Crash(argv[1]);
		else {
			Require(mode == "append" || mode == "barrier" || mode == "gc" || mode == "presync_error" ||
			            mode == "final_error",
			        "Unknown case");
			Concurrent(argv[1], mode);
		}
		std::cout << mode << " PASS\n";
		return 0;
	} catch (const std::exception &error) {
		release_first = release_retry = true;
		std::cerr << error.what() << '\n';
		return 1;
	}
}
