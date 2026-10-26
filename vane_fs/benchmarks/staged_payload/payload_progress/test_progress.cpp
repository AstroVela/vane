// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "checkpoint.hpp"
#include <atomic>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

using namespace vane_fs;
namespace fs = std::filesystem;
static constexpr int64_t MIB = 1024 * 1024;
static const auto foreground = std::this_thread::get_id();
static std::atomic<int> background_payload {0}, injected {0}, fail_kind {0};

static void Require(bool value, const char *message) {
	if (!value)
		throw std::runtime_error(message);
}
template <class F>
static void Wait(F condition) {
	auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
	while (!condition()) {
		Require(std::chrono::steady_clock::now() < deadline, "Timed out waiting for payload progress");
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}
template <class F>
static void StorageError(F operation) {
	try {
		operation();
	} catch (const Error &error) {
		Require(error.code == ErrorCode::Storage, error.what());
		return;
	}
	throw std::runtime_error("Expected storage error");
}
static bool Ends(const std::string &text, const char *suffix) {
	auto length = std::strlen(suffix);
	return text.size() >= length && text.compare(text.size() - length, length, suffix) == 0;
}
static std::string Name(int fd) {
	char name[4096];
	auto link = "/proc/self/fd/" + std::to_string(fd);
	auto size = readlink(link.c_str(), name, sizeof(name));
	return size > 0 ? std::string(name, size_t(size)) : "";
}
static bool Fail(int fd, bool wal) {
	if (std::this_thread::get_id() == foreground)
		return false;
	auto name = Name(fd);
	int expected = wal && Ends(name, "-wal") ? 2 : Ends(name, ".payload") ? 1 : Ends(name, ".stage-fence") ? 3 : 0;
	if (!expected || !fail_kind.compare_exchange_strong(expected, 0))
		return false;
	++injected;
	errno = EIO;
	return true;
}
extern "C" int __real_fdatasync(int);
extern "C" int __real_fsync(int);
extern "C" int __wrap_fdatasync(int fd) {
	if (Fail(fd, false))
		return -1;
	auto result = __real_fdatasync(fd);
	if (!result && std::this_thread::get_id() != foreground && Ends(Name(fd), ".payload"))
		++background_payload;
	return result;
}
extern "C" int __wrap_fsync(int fd) {
	return Fail(fd, true) ? -1 : __real_fsync(fd);
}
struct SQL {
	sqlite3 *db = nullptr;
	explicit SQL(const std::string &path) {
		Require(sqlite3_open(path.c_str(), &db) == SQLITE_OK, "Opening inspector");
		sqlite3_busy_timeout(db, 5000);
		Staging::Configure(db);
		Exec("SELECT count(*) FROM sqlite_master");
	}
	~SQL() {
		sqlite3_close_v2(db);
	}
	void Exec(const char *text) {
		Require(sqlite3_exec(db, text, nullptr, nullptr, nullptr) == SQLITE_OK, sqlite3_errmsg(db));
	}
	void SmallWAL() {
		int frames = -1, copied = -1;
		auto code = sqlite3_wal_checkpoint_v2(db, "main", SQLITE_CHECKPOINT_NOOP, &frames, &copied);
		Require(code == SQLITE_OK && frames >= 0 && int64_t(frames) * 4120 < Checkpointer::START_BYTES,
		        "Fixture reached the WAL-based wake threshold");
	}
};
static bool Pending(Staging &observer) {
	Staging::Guard held(observer);
	return observer.Pending();
}
static void Quiet(int count) {
	// Negative scheduling assertions need an opportunity for an erroneously
	// notified worker to run; positive assertions use bounded condition waits.
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	Require(background_payload == count, "Sub-threshold or aborted transaction published payload");
}
static void Reset(const std::string &path) {
	Workspace workspace(path, 5000, Durability::Fsync);
	auto session = workspace.Checkout();
	session->WriteFile("/file", "");
	workspace.Sync();
	Staging observer(path, 5000);
	observer.Initialize(workspace.Id(), false);
	observer.Ready();
	int before = background_payload;
	session->Write("/file", std::string(63 * MIB, 'a'));
	Quiet(before);
	Require(Pending(observer), "Sub-threshold batch was not OPEN");
	SQL(path).SmallWAL();
	session->Write("/file", std::string(MIB, 'b'), 63 * MIB);
	Wait([&] { return !Pending(observer); });
	Require(background_payload == before + 1, "64 MiB did not cause exactly one background publication");
	Require(session->Read("/file", 63 * MIB) == std::string(MIB, 'b'), "Threshold commit is not visible");
	session->Write("/file", std::string(MIB, 'c'), 64 * MIB);
	Quiet(before + 1);
	workspace.Sync();
	session->Write("/file", std::string(63 * MIB, 'd'), 65 * MIB);
	Quiet(before + 1);
	Require(Pending(observer), "Explicit barrier did not reset the payload origin");
	session->Write("/file", std::string(MIB, 'e'), 128 * MIB);
	Wait([&] { return !Pending(observer); });
	Require(background_payload == before + 2, "Second cycle did not reset its independent budget");
	workspace.Close();
	Workspace reopened(path);
	Require(reopened.Checkout()->Read("/file", 128 * MIB) == std::string(MIB, 'e'), "Reopen lost threshold data");
	std::cout << "PASS threshold below WAL budget, background/explicit resets and reopen\n";
}
static void Owners(const std::string &path) {
	Workspace first(path, 5000, Durability::Fsync), second(path, 5000, Durability::Fsync);
	auto a = first.Checkout(), b = second.Checkout();
	a->WriteFile("/file", "");
	first.Sync();
	Staging observer(path, 5000);
	observer.Initialize(first.Id(), false);
	observer.Ready();
	int before = background_payload;
	a->Write("/file", std::string(32 * MIB, 'a'));
	b->Write("/file", std::string(31 * MIB, 'b'), 32 * MIB);
	Quiet(before);
	b->Write("/file", std::string(MIB, 'c'), 63 * MIB);
	Wait([&] { return !Pending(observer); });
	Require(background_payload == before + 1, "Owners did not share the same OPEN origin");
	a->Write("/file", std::string(MIB, 'd'), 64 * MIB);
	Quiet(before + 1);
	Require(Pending(observer), "Owner did not adopt another owner's CLEAN reset");
	second.Sync();
	Require(b->Read("/file", 31 * MIB, 3 * MIB) == std::string(MIB, 'a') + std::string(2 * MIB, 'b'),
	        "Publication changed multi-owner data");
	first.Close();
	second.Close();
	std::cout << "PASS shared owner budget and cross-connection reset\n";
}
static void Rollback(const std::string &path) {
	Workspace workspace(path, 5000, Durability::Fsync);
	auto session = workspace.Checkout();
	SQL sql(path);
	sql.Exec("CREATE TRIGGER fail AFTER INSERT ON block_payloads WHEN new.id=256 "
	         "BEGIN SELECT RAISE(ABORT,'injected'); END");
	workspace.Sync();
	int before = background_payload;
	StorageError([&] { session->WriteFile("/aborted", std::string(64 * MIB, 'x')); });
	Require(fs::file_size(path + ".payload") == 64 * MIB + 4096, "Abort did not reach threshold append");
	Quiet(before);
	Staging observer(path, 5000);
	observer.Initialize(workspace.Id(), false);
	observer.Ready();
	Require(Pending(observer), "Rollback fixture lost its OPEN fence");
	sql.SmallWAL();
	sql.Exec("DROP TRIGGER fail");
	session->WriteFile("/committed", std::string(MIB, 'c'));
	Wait([&] { return !Pending(observer); });
	Require(background_payload == before + 1, "Conservative rollback-byte budget did not progress after commit");
	try {
		session->Stat("/aborted");
		throw std::runtime_error("Aborted metadata became visible");
	} catch (const Error &error) {
		Require(error.code == ErrorCode::NotFound, error.what());
	}
	workspace.CollectGarbage();
	Require(session->Read("/committed") == std::string(MIB, 'c'), "GC damaged committed data after rollback");
	workspace.Close();
	std::cout << "PASS no rollback wake; abandoned-byte conservative accounting and GC\n";
}
static void Failure(const std::string &path, int kind) {
	Workspace workspace(path, 5000, Durability::Fsync);
	auto session = workspace.Checkout();
	session->WriteFile("/file", "");
	workspace.Sync();
	Staging observer(path, 5000);
	observer.Initialize(workspace.Id(), false);
	observer.Ready();
	int before = injected;
	fail_kind = kind;
	session->Write("/file", std::string(64 * MIB, 'f'));
	Wait([&] { return injected == before + 1; });
	Wait([&] { return !Pending(observer); });
	StorageError([&] { workspace.Sync(); }); // Deferred error is delivered once.
	workspace.Sync();
	Require(session->Read("/file", 63 * MIB) == std::string(MIB, 'f'), "Background retry lost committed data");
	workspace.Close();
	Workspace reopened(path);
	Require(reopened.Checkout()->Read("/file", 63 * MIB) == std::string(MIB, 'f'),
	        "Retry publication lost durable data");
	std::cout << "PASS asynchronous failure kind=" << kind << ", latch, retry and reopen\n";
}
int main(int argc, char **argv) {
	if (argc != 2)
		return 2;
	try {
		auto root = fs::path(argv[1]);
		Reset((root / "reset.sqlite").string());
		Owners((root / "owners.sqlite").string());
		Rollback((root / "rollback.sqlite").string());
		for (int kind : {1, 2, 3})
			Failure((root / ("failure-" + std::to_string(kind) + ".sqlite")).string(), kind);
		std::cout << "PASS six payload scheduling cases\n";
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
