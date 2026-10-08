// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <atomic>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <fcntl.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace vane_fs;
namespace fs = std::filesystem;
static constexpr size_t MIB = 1024 * 1024;
static std::atomic<int> payload_syncs {0}, fence_syncs {0};
static bool fail_payload = false, fail_fence = false, pending_record = false;
static int partial_fence = 0;
static int boundary = 0, notify_fd = -1;
static fs::path durable_images;

static void Require(bool value, const char *message) {
	if (!value)
		throw std::runtime_error(message);
}
template <class F>
static void Expect(ErrorCode code, F operation) {
	try {
		operation();
	} catch (const Error &error) {
		Require(error.code == code, error.what());
		return;
	}
	throw std::runtime_error("Expected an error");
}
static std::string Name(int fd) {
	char target[4096];
	auto link = "/proc/self/fd/" + std::to_string(fd);
	auto n = readlink(link.c_str(), target, sizeof(target));
	return n > 0 ? std::string(target, size_t(n)) : "";
}
static bool Ends(const std::string &s, const std::string &suffix) {
	return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}
static void Crash(int at) {
	if (boundary != at)
		return;
	char ready = 'x';
	if (write(notify_fd, &ready, 1) != 1)
		_exit(81);
	raise(SIGSTOP);
}
extern "C" ssize_t __real_pwrite(int, const void *, size_t, off_t);
extern "C" int __real_fdatasync(int);
extern "C" ssize_t __wrap_pwrite(int fd, const void *data, size_t size, off_t offset) {
	auto name = Name(fd);
	if (Ends(name, ".stage-fence") && partial_fence) {
		if (partial_fence == 1) {
			partial_fence = 2;
			return __real_pwrite(fd, data, std::min<size_t>(73, size), offset);
		}
		partial_fence = 0;
		errno = EIO;
		return -1;
	}
	auto result = __real_pwrite(fd, data, size, offset);
	if (result == 4096 && Ends(name, ".stage-fence")) {
		pending_record = static_cast<const unsigned char *>(data)[47] != 0;
		Crash(pending_record ? 1 : 7);
	}
	if (result > 0 && offset >= 4096 && Ends(name, ".payload"))
		Crash(3);
	return result;
}
extern "C" int __wrap_fdatasync(int fd) {
	auto name = Name(fd);
	bool payload = Ends(name, ".payload"), fence = Ends(name, ".stage-fence");
	if (payload) {
		++payload_syncs;
		if (fail_payload) {
			fail_payload = false;
			errno = EIO;
			return -1;
		}
		Crash(5);
	}
	if (fence) {
		++fence_syncs;
		if (fail_fence) {
			fail_fence = false;
			errno = EIO;
			return -1;
		}
	}
	auto result = __real_fdatasync(fd);
	if (!result && (payload || fence) && !durable_images.empty()) {
		std::error_code error;
		fs::copy_file(name, durable_images / fs::path(name).filename(), fs::copy_options::overwrite_existing, error);
		Require(!error, "Capturing durable sidecar");
	}
	if (!result && payload)
		Crash(6);
	if (!result && fence)
		Crash(pending_record ? 2 : 8);
	return result;
}
static void SQL(const std::string &path, const char *sql) {
	sqlite3 *db = nullptr;
	Require(sqlite3_open(path.c_str(), &db) == SQLITE_OK, "Opening test inspector");
	auto code = sqlite3_exec(db, sql, nullptr, nullptr, nullptr);
	std::string error = sqlite3_errmsg(db);
	sqlite3_close(db);
	Require(code == SQLITE_OK, error.c_str());
}
static void Batching(const std::string &path) {
	Workspace first(path, 100, Durability::Fsync), second(path, 100, Durability::Fsync);
	auto a = first.Checkout(), b = second.Checkout();
	a->WriteFile("/file", "");
	first.Sync();
	auto data_before = payload_syncs.load(), fence_before = fence_syncs.load();
	for (int i = 0; i < 8; ++i)
		a->Write("/file", std::string(MIB, char('a' + i)), int64_t(i) * MIB);
	Require(payload_syncs == data_before, "Ordinary append synchronized payload per operation");
	Require(fence_syncs == fence_before + 1, "Batch did not share one OPEN fence");
	Require(b->Read("/file", 7 * MIB) == std::string(MIB, 'h'), "Pending bytes invisible to another connection");
	// Another connection has no local dirty payload flag, but must flush all.
	second.Sync();
	Require(payload_syncs == data_before + 1, "Cross-connection barrier omitted payload sync");
	Require(fence_syncs == fence_before + 3, "Observer did not adopt OPEN and publish CLEAN");
	first.Sync(); // Adopt the other connection's CLEAN record once.
	data_before = payload_syncs;
	fence_before = fence_syncs;
	a->WriteFile("/inline", "small");
	first.Sync();
	Require(payload_syncs == data_before && fence_syncs == fence_before, "Clean inline barrier added sidecar syncs");
	a->WriteFile("/closing", std::string(MIB, 'c'));
	first.Close();
	second.Close();
	Workspace reopened(path);
	Require(reopened.Checkout()->Read("/closing") == std::string(MIB, 'c'), "Explicit close lost pending bytes");
}
static void Failures(const std::string &path) {
	Workspace workspace(path, 20, Durability::Fsync);
	auto session = workspace.Checkout();
	session->WriteFile("/stable", std::string(MIB, 's'));
	workspace.Sync();
	partial_fence = 1;
	Expect(ErrorCode::Storage, [&] { session->WriteFile("/torn", std::string(MIB, 't')); });
	Expect(ErrorCode::NotFound, [&] { session->Stat("/torn"); });
	fail_fence = true;
	Expect(ErrorCode::Storage, [&] { session->WriteFile("/failed", std::string(MIB, 'f')); });
	Expect(ErrorCode::NotFound, [&] { session->Stat("/failed"); });
	// A valid OPEN record whose sync failed must be synced on adoption.
	fail_fence = true;
	Expect(ErrorCode::Storage, [&] { session->WriteFile("/untrusted", std::string(MIB, 'u')); });
	Require(!fail_fence, "Retry trusted a cached but unsynced OPEN record");
	Expect(ErrorCode::NotFound, [&] { session->Stat("/untrusted"); });
	session->WriteFile("/pending", std::string(MIB, 'p'));
	fail_payload = true;
	Expect(ErrorCode::Storage, [&] { workspace.Sync(); });
	Require(session->Read("/pending") == std::string(MIB, 'p'), "Failed barrier lost visible bytes");
	workspace.Sync();
	session->WriteFile("/late", std::string(MIB, 'l'));
	fail_fence = true;
	Expect(ErrorCode::Storage, [&] { workspace.Sync(); });
	// CLEAN is also untrusted after a failed sync: another connection must
	// not acknowledge a barrier until the record is durable.
	fail_fence = true;
	Expect(ErrorCode::Storage, [&] { Workspace observer(path, 20); });
	Require(!fail_fence, "Observer trusted a cached but unsynced CLEAN record");
	{
		Workspace observer(path, 20);
		observer.Sync();
	}
	session->WriteFile("/closing", std::string(MIB, 'c'));
	fail_payload = true;
	Expect(ErrorCode::Storage, [&] { workspace.Close(); });
	Require(session->Read("/closing") == std::string(MIB, 'c'), "Failed close was not retryable");
	workspace.Close();
	Workspace reopened(path);
	Require(reopened.Checkout()->Read("/closing") == std::string(MIB, 'c'), "Close retry lost data");
}
static void SynchronousHandles(const std::string &path) {
	Workspace workspace(path, 100, Durability::Fsync);
	workspace.AcquireMount("main");
	auto session = workspace.Checkout();
	auto file = session->OpenFile("/file", true);
	auto before = payload_syncs.load();
	session->WriteInode(file.inode, std::string(MIB, 's'), 0, false, true);
	Require(payload_syncs == before + 1, "Synchronous inode write omitted payload barrier");
	workspace.Close();
	Workspace reopened(path);
	Require(reopened.Checkout()->Read("/file") == std::string(MIB, 's'), "Synchronous inode write differs");
}
static void Recovery(const std::string &path, int point, bool lose_unsynced, bool restart, bool spill = false) {
	{
		Workspace workspace(path);
		workspace.Checkout()->WriteFile("/stable", std::string(MIB, 's'));
	}
	auto images = fs::path(path + ".images");
	fs::create_directory(images);
	for (const auto &suffix : {".payload", ".stage-fence"})
		fs::copy_file(path + suffix, images / fs::path(path + suffix).filename());
	int channel[2];
	Require(pipe(channel) == 0, "Creating crash pipe");
	auto child = fork();
	Require(child >= 0, "Forking crash writer");
	if (!child) {
		close(channel[0]);
		notify_fd = channel[1];
		durable_images = images;
		try {
			Workspace workspace(path, 100, Durability::Fsync);
			if (spill)
				SQL(path, "CREATE TABLE spill_padding(data BLOB); CREATE TRIGGER spill BEFORE INSERT ON inode_versions "
				          "BEGIN INSERT INTO spill_padding VALUES(zeroblob(2097152)); END");
			if (restart)
				SQL(path, "PRAGMA wal_checkpoint(TRUNCATE)"); // CLEAN, test-only forced reuse.
			auto session = workspace.Checkout();
			boundary = point;
			session->WriteFile("/pending", std::string(MIB, 'p'));
			if (spill)
				Crash(4);
			session->Rename("/stable", "/renamed");
			Crash(4);
			workspace.Sync();
		} catch (const std::exception &error) {
			std::cerr << error.what() << '\n';
		}
		_exit(82);
	}
	close(channel[1]);
	pollfd wait {channel[0], POLLIN, 0};
	char ready = 0;
	bool reached = poll(&wait, 1, 20000) == 1 && read(channel[0], &ready, 1) == 1 && ready == 'x';
	close(channel[0]);
	kill(child, SIGKILL);
	int status = 0;
	waitpid(child, &status, 0);
	Require(reached && WIFSIGNALED(status), "Crash boundary not reached");
	if (lose_unsynced) {
		// Worst permitted ordering: retain ALL latest SQL bytes (including the
		// NORMAL commit), but only the last successfully synced sidecar images.
		for (const auto &suffix : {".payload", ".stage-fence"})
			fs::copy_file(images / fs::path(path + suffix).filename(), path + suffix,
			              fs::copy_options::overwrite_existing);
	}
	Workspace reopened(path);
	reopened.RecoverOwners();
	auto session = reopened.Checkout();
	bool published = point == 8 || (point == 7 && !lose_unsynced);
	if (published) {
		Require(session->Read("/renamed") == std::string(MIB, 's'), "Published namespace lost");
		Require(session->Read("/pending") == std::string(MIB, 'p'), "Published bytes lost");
	} else {
		Require(session->Read("/stable") == std::string(MIB, 's'), "Recovery lost earlier durable bytes");
		Expect(ErrorCode::NotFound, [&] { session->Stat("/pending"); });
		Expect(ErrorCode::NotFound, [&] { session->Stat("/renamed"); });
	}
	reopened.CollectGarbage();
	Require(fs::file_size(path + ".payload") == (published ? 2 : 1) * MIB + 4096, "Recovery/GC leaked external bytes");
	reopened.Close();
	Workspace again(path);
	Require(again.Checkout()->Read(published ? "/renamed" : "/stable") == std::string(MIB, 's'),
	        "Recovery not idempotent");
	std::cout << "boundary=" << point << " loss=" << lose_unsynced << " restart=" << restart << " spill=" << spill
	          << " PASS\n";
}
static void ActiveConnection(const std::string &path) {
	Workspace keeper(path, 100, Durability::Fsync);
	keeper.Checkout()->WriteFile("/stable", "s");
	keeper.Sync();
	auto child = fork();
	Require(child >= 0, "Forking live writer");
	if (!child) {
		try {
			Workspace writer(path, 100, Durability::Fsync);
			writer.Checkout()->WriteFile("/pending", std::string(MIB, 'p'));
			_exit(0);
		} catch (...) {
			_exit(1);
		}
	}
	int status = 0;
	waitpid(child, &status, 0);
	Require(status == 0, "Live writer failed");
	Require(keeper.Checkout()->Read("/pending") == std::string(MIB, 'p'), "Other active connection lost visible data");
	Workspace joined(path);
	Require(joined.Checkout()->Read("/pending") == std::string(MIB, 'p'), "Joining connection truncated active WAL");
	joined.Sync();
	keeper.Close();
	joined.Close();
	Workspace reopened(path);
	Require(reopened.Checkout()->Read("/pending") == std::string(MIB, 'p'), "Joined barrier lost data");
}
static void GateAndSpill(const std::string &path) {
	Workspace workspace(path, 15, Durability::Fsync);
	auto session = workspace.Checkout();
	session->WriteFile("/stable", "s");
	workspace.Sync();
	int held = open((path + ".stage-publish").c_str(), O_RDWR);
	Require(held >= 0 && flock(held, LOCK_EX) == 0, "Holding publication gate");
	Expect(ErrorCode::Busy, [&] { session->WriteFile("/blocked", std::string(MIB, 'b')); });
	close(held);
	Expect(ErrorCode::NotFound, [&] { session->Stat("/blocked"); });
	// Uncommitted spill precedes BeginBatch. Its frames must not become the
	// recovery prefix. Abort after the first external batch, then reuse WAL.
	SQL(path, "CREATE TABLE padding(data BLOB); CREATE TRIGGER spill BEFORE INSERT ON inode_versions BEGIN INSERT INTO "
	          "padding VALUES(zeroblob(4194304)); END; CREATE TRIGGER fail AFTER INSERT ON block_payloads WHEN "
	          "new.id%64=0 BEGIN SELECT RAISE(ABORT,'injected'); END");
	SQL(path, "PRAGMA wal_checkpoint(RESTART)");
	auto before = fs::file_size(path + ".payload");
	Expect(ErrorCode::Storage, [&] { session->WriteFile("/failed", std::string(MIB, 'f')); });
	Require(fs::file_size(path + ".payload") >= before + 262144, "Spill test did not reach external append");
	Expect(ErrorCode::NotFound, [&] { session->Stat("/failed"); });
	SQL(path, "DROP TRIGGER fail; DROP TRIGGER spill");
	session->WriteFile("/retry", std::string(MIB, 'r'));
	workspace.Sync();
	Require(session->Read("/retry") == std::string(MIB, 'r'), "Spilled rollback damaged later append");
}
int main() {
	auto root = fs::temp_directory_path() / ("vane-stage-tests-" + std::to_string(getpid()));
	try {
		fs::create_directory(root);
		Batching((root / "batch.sqlite").string());
		Failures((root / "failure.sqlite").string());
		SynchronousHandles((root / "synchronous.sqlite").string());
		ActiveConnection((root / "active.sqlite").string());
		GateAndSpill((root / "spill.sqlite").string());
		for (bool lose : {false, true})
			for (bool restart : {false, true})
				for (int point = 1; point <= 8; ++point)
					Recovery((root / ("recovery-" + std::to_string(lose) + "-" + std::to_string(restart) + "-" +
					                  std::to_string(point) + ".sqlite"))
					             .string(),
					         point, lose, restart);
		Recovery((root / "spill-recovery.sqlite").string(), 4, true, true, true);
		fs::remove_all(root);
		std::cout << "PASS: batching, visibility, retry, gate/spill and 33 crash/loss cases\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		fs::remove_all(root);
		return 1;
	}
}
