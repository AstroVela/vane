// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include "staging.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>

using namespace vane_fs;
namespace fs = std::filesystem;
static constexpr int64_t MIB = 1024 * 1024;
static const auto foreground = std::this_thread::get_id();
static std::atomic<bool> armed {false}, entered {false}, release_sync {false}, fail_sync {false};
static std::atomic<bool> fail_statement {false}, ready_to_fail {false};
static std::atomic<int> foreground_syncs {0}, background_syncs {0};
static bool suppress_callbacks = false;
static int notify_fd = -1;
static std::string payload_path;

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
static void Expect(ErrorCode code, F function) {
	try {
		function();
	} catch (const Error &error) {
		Require(error.code == code, "Wrong storage error");
		return;
	}
	throw std::runtime_error("Missing expected error");
}
static bool Payload(int fd) {
	std::array<char, 4096> name {};
	auto length = readlink(("/proc/self/fd/" + std::to_string(fd)).c_str(), name.data(), name.size());
	return length > 0 && std::string(name.data(), size_t(length)) == payload_path;
}
extern "C" int __real_fdatasync(int);
extern "C" int __wrap_fdatasync(int fd) {
	if (armed && Payload(fd)) {
		if (std::this_thread::get_id() == foreground) {
			++foreground_syncs;
			if (fail_sync.exchange(false)) {
				errno = EIO;
				return -1;
			}
		} else {
			++background_syncs;
			entered = true;
			if (notify_fd >= 0) {
				char ready = 'x';
				if (write(notify_fd, &ready, 1) != 1)
					_exit(90);
			}
			while (!release_sync)
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
	return __real_fdatasync(fd);
}
extern "C" ssize_t __real_pwrite(int, const void *, size_t, off_t);
extern "C" ssize_t __wrap_pwrite(int fd, const void *bytes, size_t size, off_t offset) {
	auto result = __real_pwrite(fd, bytes, size, offset);
	if (fail_statement && result > 0 && Payload(fd) && offset + result >= 64 * MIB + 4096)
		ready_to_fail = true;
	return result;
}
extern "C" int __real_sqlite3_step(sqlite3_stmt *);
extern "C" int __wrap_sqlite3_step(sqlite3_stmt *statement) {
	auto sql = sqlite3_sql(statement);
	if (ready_to_fail && sql && std::strstr(sql, "INSERT INTO block_payloads")) {
		ready_to_fail = fail_statement = false;
		return SQLITE_IOERR;
	}
	return __real_sqlite3_step(statement);
}
using Hook = int (*)(void *, sqlite3 *, const char *, int);
extern "C" void *__real_sqlite3_wal_hook(sqlite3 *, Hook, void *);
extern "C" void *__wrap_sqlite3_wal_hook(sqlite3 *db, Hook hook, void *context) {
	return __real_sqlite3_wal_hook(db, suppress_callbacks ? nullptr : hook, context);
}
extern "C" int __real_sqlite3_wal_checkpoint_v2(sqlite3 *, const char *, int, int *, int *);
extern "C" int __wrap_sqlite3_wal_checkpoint_v2(sqlite3 *db, const char *name, int mode, int *frames, int *copied) {
	return __real_sqlite3_wal_checkpoint_v2(db, name, mode, frames, copied);
}
struct Release {
	~Release() {
		release_sync = true;
	}
};
struct SQL {
	sqlite3 *db = nullptr;
	explicit SQL(const std::string &path) {
		Require(sqlite3_open(path.c_str(), &db) == SQLITE_OK, "Opening inspector");
		Staging::Configure(db);
		sqlite3_busy_timeout(db, 1000);
		Require(sqlite3_exec(db, "SELECT count(*) FROM sqlite_master", nullptr, nullptr, nullptr) == SQLITE_OK,
		        "Opening inspection pager");
	}
	~SQL() {
		sqlite3_close_v2(db);
	}
	int64_t WalBytes() {
		int frames = -1, copied = -1;
		Require(sqlite3_wal_checkpoint_v2(db, "main", SQLITE_CHECKPOINT_NOOP, &frames, &copied) == SQLITE_OK &&
		            frames >= 0,
		        "Inspecting committed WAL");
		return 32 + int64_t(frames) * 4120;
	}
	bool CaughtUp() {
		int frames = -1, copied = -1;
		auto rc = sqlite3_wal_checkpoint_v2(db, "main", SQLITE_CHECKPOINT_NOOP, &frames, &copied);
		Require(rc == SQLITE_OK || rc == SQLITE_BUSY, "Inspecting checkpoint progress");
		return rc == SQLITE_OK && frames >= 0 && frames == copied;
	}
};
using Record = std::array<unsigned char, 4096>;
static uint64_t Decode(const unsigned char *p) {
	uint64_t value = 0;
	for (int i = 0; i < 8; ++i)
		value = (value << 8) | p[i];
	return value;
}
static void Encode(unsigned char *p, uint64_t value) {
	for (int i = 7; i >= 0; --i) {
		p[i] = value & 255;
		value >>= 8;
	}
}
static std::pair<Record, int> Fence(const std::string &path) {
	std::ifstream input(path + ".stage-fence", std::ios::binary);
	Record result {};
	int selected = -1;
	for (int i = 0; i < 2; ++i) {
		Record record {};
		input.read(reinterpret_cast<char *>(record.data()), record.size());
		if (input.gcount() == int64_t(record.size()) && Decode(record.data() + 32) > Decode(result.data() + 32)) {
			result = record;
			selected = i;
		}
	}
	Require(selected >= 0, "No publication fence");
	return {result, selected};
}
static bool Pending(const std::string &path) {
	return Decode(Fence(path).first.data() + 40) == 1;
}
static uint64_t Origin(const std::string &path) {
	return Decode(Fence(path).first.data() + 80);
}
static void AwaitPublication(const std::string &path) {
	Require(Wait([] { return entered.load(); }, 2500), "No payload publication below WAL threshold");
	Require(Pending(path), "Publication became CLEAN before its payload sync");
	release_sync = true;
	Require(Wait([&] { return !Pending(path); }), "Publication failed to finish");
}
static void Wake(const std::string &path) {
	Workspace workspace(path, 1000, Durability::Fsync);
	auto session = workspace.Checkout();
	Release release;
	SQL sql(path);
	auto begin = fs::file_size(payload_path);
	armed = true;
	session->WriteFile("/file", std::string(63 * MIB, 'a'));
	Require(sql.WalBytes() < 16 * MIB, "Fixture WAL reached the former publication threshold");
	Require(!entered, "Under-budget payload published early");
	session->Write("/file", std::string(MIB, 'b'), 63 * MIB);
	Require(sql.WalBytes() < 16 * MIB, "Fixture did not isolate the payload threshold");
	AwaitPublication(path);
#ifdef VANE_TEST_PAYLOAD_BUDGET
	Require(Origin(path) == begin, "Wrong shared append origin");
#endif
	Require(session->Read("/file", 0, 63 * MIB) == std::string(63 * MIB, 'a'), "First content changed");
	Require(session->Read("/file", 63 * MIB) == std::string(MIB, 'b'), "Last content changed");
	armed = false;
	workspace.Close();
}
#ifdef VANE_TEST_PAYLOAD_BUDGET
static void Shared(const std::string &path) {
	Workspace first(path, 1000, Durability::Fsync);
	auto a = first.Checkout();
	a->WriteFile("/stable", std::string(128 * MIB, 's'));
	first.Sync();
	Workspace second(path, 1000, Durability::Fsync);
	auto b = second.Checkout();
	Release release;
	SQL sql(path);
	auto old_wal = sql.WalBytes();
	std::cout << "shared fixture WAL bytes=" << old_wal << " CLEAN=" << !Pending(path) << std::endl;
	// Constructor bookkeeping can restart a small CLEAN WAL without waking
	// the worker. Such a tail is deliberately below the existing WAL budget.
	if (old_wal >= 16 * MIB)
		Require(Wait([&] { return sql.CaughtUp(); }), "Old WAL failed to backfill");
	auto begin = fs::file_size(payload_path);
	armed = true;
	a->WriteFile("/new", std::string(32 * MIB, 'a'));
	Require(!entered, "Historical file size counted as dirty payload");
	b->Write("/new", std::string(32 * MIB, 'b'), 32 * MIB);
	AwaitPublication(path);
	Require(Origin(path) == begin, "Second connection reset the OPEN origin");
	Require(Wait([&] { return sql.CaughtUp(); }), "Shared publication failed to backfill");
	auto calls = background_syncs.load();
	begin = fs::file_size(payload_path);
	a->WriteFile("/tail", std::string(8 * MIB, 't'));
	Require(Origin(path) == begin, "Stale connection retained a previous batch origin");
	Require(background_syncs == calls, "Small new batch inherited previous budget");
	Require(a->Read("/new", 0, 32 * MIB) == std::string(32 * MIB, 'a'), "First writer content changed");
	Require(b->Read("/new", 32 * MIB) == std::string(32 * MIB, 'b'), "Second writer content changed");
	Require(a->Read("/stable") == std::string(128 * MIB, 's'), "Historical payload changed");
	armed = false;
	first.Sync();
	second.Close();
	first.Close();
}
static void Garbage(const std::string &path) {
	Workspace workspace(path, 1000, Durability::Fsync);
	auto session = workspace.Checkout();
	session->WriteFile("/guard", std::string(32 * MIB, 'g'));
	auto snapshot = workspace.Snapshot();
	auto pinned = workspace.OpenSnapshot(snapshot);
	session->WriteFile("/dead", std::string(64 * MIB, 'd'));
	workspace.Sync();
	auto before = fs::file_size(payload_path);
	session->Unlink("/dead");
	workspace.CollectGarbage();
	auto begin = fs::file_size(payload_path);
	Require(begin < before, "GC fixture did not shrink payload");
	Release release;
	SQL sql(path);
	if (sql.WalBytes() >= 16 * MIB)
		Require(Wait([&] { return sql.CaughtUp(); }), "GC WAL failed to backfill");
	armed = true;
	session->WriteFile("/new", std::string(16 * MIB, 'a'));
	Require(Origin(path) == begin, "GC did not reset the next OPEN origin");
	Require(!entered, "Reclaimed space counted as new dirty payload");
	session->Write("/new", std::string(48 * MIB, 'b'), 16 * MIB);
	AwaitPublication(path);
	Require(pinned->Read("/guard") == std::string(32 * MIB, 'g'), "GC lost pinned historical data");
	Require(session->Read("/new", 16 * MIB) == std::string(48 * MIB, 'b'), "Post-GC payload changed");
	armed = false;
	workspace.Close();
}
static void Admission(const std::string &path) {
	Workspace workspace(path, 1000, Durability::Fsync);
	auto session = workspace.Checkout();
	armed = true;
	session->WriteFile("/full", std::string(64 * MIB, 'a'));
	Require(Pending(path) && foreground_syncs == 0, "Suppressed-worker fixture published early");
	auto begin = fs::file_size(payload_path);
	fail_sync = true;
	Expect(ErrorCode::Storage, [&] { session->WriteFile("/new", std::string(MIB, 'b')); });
	Require(Pending(path), "Failed admission sync cleared OPEN");
	Expect(ErrorCode::NotFound, [&] { session->Stat("/new"); });
	session->WriteFile("/new", std::string(MIB, 'b'));
	Require(foreground_syncs == 2, "Admission did not retry the original protected sync");
	Require(Origin(path) == begin && Pending(path), "Admission failed to open a fresh batch");
	Require(session->Read("/full") == std::string(64 * MIB, 'a'), "Admission changed earlier content");
	armed = false;
	workspace.Close();
	Workspace reopened(path);
	Require(reopened.Checkout()->Read("/new") == std::string(MIB, 'b'), "Reopen lost admitted content");
}
static void Rollback(const std::string &path) {
	Workspace workspace(path, 1000, Durability::Fsync);
	auto session = workspace.Checkout();
	armed = true;
	fail_statement = true;
	Expect(ErrorCode::Storage, [&] { session->WriteFile("/failed", std::string(64 * MIB, 'x')); });
	Require(!fail_statement && Pending(path), "SQL failure fixture did not leave an OPEN append tail");
	Expect(ErrorCode::NotFound, [&] { session->Stat("/failed"); });
	session->WriteFile("/new", std::string(MIB, 'n'));
	Require(foreground_syncs == 1, "Rolled-back physical bytes were omitted from the sync budget");
	armed = false;
	workspace.CollectGarbage();
	Require(session->Read("/new") == std::string(MIB, 'n'), "GC lost post-rollback content");
	workspace.Close();
}
static void RewriteOrigin(const std::string &path, uint64_t origin) {
	auto [record, slot] = Fence(path);
	Encode(record.data() + 80, origin);
	Encode(record.data() + 32, Decode(record.data() + 32) + 1);
	uint64_t crc = 0;
	for (size_t i = 0; i < 152; ++i) {
		crc ^= uint64_t(record[i]) << 56;
		for (int bit = 0; bit < 8; ++bit)
			crc = (crc << 1) ^ ((crc >> 63) ? UINT64_C(0x42f0e1eba9ea3693) : 0);
	}
	Encode(record.data() + 4088, crc);
	int fd = open((path + ".stage-fence").c_str(), O_RDWR);
	Require(fd >= 0 && pwrite(fd, record.data(), record.size(), slot * 4096) == int64_t(record.size()) &&
	            fdatasync(fd) == 0,
	        "Rewriting advisory origin");
	close(fd);
}
static void Legacy(const std::string &path) {
	Workspace workspace(path, 1000, Durability::Fsync);
	auto session = workspace.Checkout();
	session->WriteFile("/file", std::string(MIB, 'a'));
	for (auto origin : {UINT64_C(0), uint64_t(fs::file_size(payload_path) + 4096)}) {
		RewriteOrigin(path, origin);
		Staging observer(path, 1000);
		observer.Initialize(workspace.Id(), false);
		observer.Ready();
		Staging::Guard guard(observer);
		Require(observer.PendingPayloadBytes() == fs::file_size(payload_path),
		        "Invalid advisory origin underestimated");
	}
	workspace.Sync();
	Require(!Pending(path), "Advisory origin interfered with FULL publication");
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
	fs::copy_file(payload_path, payload_path + "-durable");
	int pipes[2];
	Require(pipe(pipes) == 0, "Pipe");
	auto pid = fork();
	Require(pid >= 0, "Fork");
	if (!pid) {
		close(pipes[0]);
		notify_fd = pipes[1];
		try {
			Workspace workspace(path, 1000, Durability::Fsync);
			auto session = workspace.Checkout();
			armed = true;
			session->WriteFile("/pending", std::string(64 * MIB, 'p'));
			while (true)
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
		} catch (...) {
			_exit(91);
		}
	}
	close(pipes[1]);
	pollfd ready {pipes[0], POLLIN, 0};
	bool reached = poll(&ready, 1, 10000) == 1 && (ready.revents & POLLIN);
	kill(pid, SIGKILL);
	int status = 0;
	Require(waitpid(pid, &status, 0) == pid, "Waiting for child");
	close(pipes[0]);
	Require(reached && WIFSIGNALED(status), "Payload-budget crash boundary not reached");
	Require(Pending(path), "Unsynced payload became CLEAN");
	fs::copy_file(payload_path + "-durable", payload_path, fs::copy_options::overwrite_existing);
	fs::remove(path + "-shm");
	for (int i = 0; i < 2; ++i) {
		Workspace recovered(path);
		auto session = recovered.Checkout();
		Require(session->Read("/stable") == std::string(MIB, 's'), "Recovery lost stable content");
		Require(recovered.OpenSnapshot(snapshot)->Read("/stable") == std::string(MIB, 's'),
		        "Recovery changed snapshot");
		Expect(ErrorCode::NotFound, [&] { session->Stat("/pending"); });
		recovered.RecoverOwners();
		recovered.CollectGarbage();
		recovered.Close();
	}
}
#endif
int main(int argc, char **argv) {
	try {
		Require(argc == 3, "Expected database and case");
		std::string path = argv[1], mode = argv[2];
		payload_path = fs::weakly_canonical(path).string() + ".payload";
		suppress_callbacks = mode == "admission" || mode == "rollback" || mode == "legacy";
		if (mode == "wake")
			Wake(path);
#ifdef VANE_TEST_PAYLOAD_BUDGET
		else if (mode == "shared")
			Shared(path);
		else if (mode == "gc")
			Garbage(path);
		else if (mode == "admission")
			Admission(path);
		else if (mode == "rollback")
			Rollback(path);
		else if (mode == "legacy")
			Legacy(path);
		else if (mode == "crash")
			Crash(path);
#endif
		else
			throw std::runtime_error("Unknown case");
		std::cout << mode << " PASS foreground_syncs=" << foreground_syncs << " background_syncs=" << background_syncs
		          << '\n';
		return 0;
	} catch (const std::exception &error) {
		release_sync = true;
		std::cerr << error.what() << '\n';
		return 1;
	}
}
