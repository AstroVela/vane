// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>
#include <poll.h>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>

using namespace vane_fs;
static constexpr int64_t MIB = 1024 * 1024;

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
	throw std::runtime_error("Expected an exception");
}
template <class F>
static void Wait(F condition) {
	auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
	while (!condition()) {
		Require(std::chrono::steady_clock::now() < deadline, "Timed out waiting for checkpoint");
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}
struct Inspector {
	sqlite3 *db = nullptr;
	explicit Inspector(const std::string &path) {
		Require(sqlite3_open(path.c_str(), &db) == SQLITE_OK, "Opening inspector");
	}
	~Inspector() {
		sqlite3_close_v2(db);
	}
	std::string SQL(const char *sql) {
		std::string result;
		auto code = sqlite3_exec(
		    db, sql,
		    [](void *context, int count, char **values, char **) {
			    for (int i = 0; i < count; ++i)
				    *static_cast<std::string *>(context) += values[i] ? values[i] : "NULL";
			    return 0;
		    },
		    &result, nullptr);
		Require(code == SQLITE_OK, sqlite3_errmsg(db));
		return result;
	}
	bool CaughtUp() {
		int frames = -1, copied = -1;
		auto code = sqlite3_wal_checkpoint_v2(db, "main", SQLITE_CHECKPOINT_NOOP, &frames, &copied);
		Require(code == SQLITE_OK || code == SQLITE_BUSY, "Reading WAL progress");
		return code == SQLITE_OK && frames >= 0 && frames == copied;
	}
};

// Faults apply only to the worker's main-database writes. WAL commits and
// foreground FULL barriers remain real, successful storage operations.
static auto foreground = std::this_thread::get_id();
static std::atomic<int> background_writes {0}, faults {0};
static std::atomic<int> main_opens {0}, fail_open_at {-1};
static std::atomic<bool> fail_background {false}, pause_background {false}, paused {false};
struct FileHooks {
	sqlite3_io_methods methods;
	const sqlite3_io_methods *original;
	int flags;
};
static std::map<sqlite3_file *, FileHooks> files;
static std::mutex files_mutex;
static sqlite3_vfs *original_vfs;
static int Write(sqlite3_file *file, const void *data, int size, sqlite3_int64 offset) {
	const sqlite3_io_methods *original;
	bool background;
	{
		std::lock_guard<std::mutex> lock(files_mutex);
		auto &hook = files.at(file);
		original = hook.original;
		background = (hook.flags & SQLITE_OPEN_MAIN_DB) && std::this_thread::get_id() != foreground;
	}
	if (background) {
		++background_writes;
		if (fail_background) {
			++faults;
			return SQLITE_IOERR_WRITE;
		}
		if (pause_background) {
			paused = true;
			while (pause_background)
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
	return original->xWrite(file, data, size, offset);
}
static int CloseFile(sqlite3_file *file) {
	{
		std::lock_guard<std::mutex> lock(files_mutex);
		file->pMethods = files.at(file).original;
		files.erase(file);
	}
	return file->pMethods->xClose(file);
}
static int Open(sqlite3_vfs *, const char *name, sqlite3_file *file, int flags, int *output_flags) {
	if ((flags & SQLITE_OPEN_MAIN_DB) && ++main_opens == fail_open_at)
		return SQLITE_CANTOPEN;
	auto code = original_vfs->xOpen(original_vfs, name, file, flags, output_flags);
	if (file->pMethods) {
		std::lock_guard<std::mutex> lock(files_mutex);
		auto &hook = files.emplace(file, FileHooks {*file->pMethods, file->pMethods, flags}).first->second;
		hook.methods.xWrite = Write;
		hook.methods.xClose = CloseFile;
		file->pMethods = &hook.methods;
	}
	return code;
}
struct Hooks {
	sqlite3_vfs vfs;
	Hooks() {
		original_vfs = sqlite3_vfs_find(nullptr);
		vfs = *original_vfs;
		vfs.zName = "vane-fs-checkpoint-test";
		vfs.xOpen = Open;
		Require(sqlite3_vfs_register(&vfs, 1) == SQLITE_OK, "Installing checkpoint VFS");
	}
	~Hooks() {
		sqlite3_vfs_unregister(&vfs);
	}
};

static void CheckStartupFailure(const std::string &path) {
	fail_open_at = main_opens + 2; // Allow writer registration, reject worker connection.
	Expect(ErrorCode::Storage, [&] { Workspace workspace(path, 30, Durability::Fsync); });
	fail_open_at = -1;
	Workspace recovered(path);
	Require(recovered.RecoverOwners().owners == 1, "Failed worker startup left an unrecoverable owner");
	recovered.Checkout()->WriteFile("/retry", "ok");
	recovered.Close();
}

static void CheckProgress(const std::string &path) {
	Workspace workspace(path, 1000, Durability::Fsync);
	auto session = workspace.Checkout();
	session->WriteFile("/file", "base");
	auto snapshot = workspace.OpenSnapshot(workspace.Snapshot());
	auto before = background_writes.load();
	session->WriteFile("/file", std::string(32 * MIB, 'a'));
	Inspector inspector(path);
	inspector.SQL("SELECT * FROM format");
	Wait([&] { return background_writes > before && inspector.CaughtUp(); });
	Require(snapshot->Read("/file") == "base", "Checkpoint changed retained snapshot");
	Require(session->Read("/file", 31 * MIB) == std::string(MIB, 'a'), "Checkpoint changed live data");
	workspace.Sync();
	// Restart/reset must continue working over several admission windows.
	for (int i = 0; i < 96; ++i)
		session->Write("/file", std::string(MIB, char('a' + i % 20)), int64_t(i) * MIB);
	workspace.Sync();
	// A final log below the 16 MiB start threshold may remain in WAL. It
	// must be durable and bounded, but need not have been checkpointed yet.
	for (int i = 0; i < 96; ++i)
		Require(session->Read("/file", int64_t(i) * MIB, MIB) == std::string(MIB, char('a' + i % 20)),
		        "Sustained write changed bytes");
	Require(std::filesystem::file_size(path + "-wal") < 67 * MIB, "WAL budget not enforced");
	workspace.Close();
}

static void CheckPinnedReader(const std::string &path, int page_size) {
	{
		Inspector inspector(path);
		inspector.SQL(page_size == 8192 ? "PRAGMA page_size=8192; VACUUM" : "PRAGMA page_size=4096; VACUUM");
	}
	Workspace workspace(path, 30, Durability::Fsync);
	Workspace other(path, 30, Durability::Fsync);
	auto session = workspace.Checkout();
	auto second = other.Checkout();
	session->WriteFile("/file", "");
	Inspector inspector(path);
	inspector.SQL("BEGIN; SELECT * FROM format");
	int writes = 0;
	for (; writes < 80; ++writes) {
		try {
			(writes % 2 ? second : session)->Write("/file", std::string(MIB, 'p'), int64_t(writes) * MIB);
		} catch (const Error &error) {
			Require(error.code == ErrorCode::Busy, error.what());
			break;
		}
	}
	Require(writes > 0 && writes < 80, "Pinned reader did not apply backpressure");
	Require(session->Stat("/file").size == int64_t(writes) * MIB, "Rejected write partially committed");
	auto bytes = std::filesystem::file_size(path + "-wal");
	Require(bytes >= 64 * MIB && bytes < 68 * MIB, "Wrong WAL byte budget for page size");
	Expect(ErrorCode::Busy, [&] { session->Write("/file", "no", 0); });
	// A full WAL must not prevent synchronizing already acknowledged writes.
	workspace.Sync();
	Require(session->Read("/file", 0, 2) == "pp", "Backpressure changed earlier data");
	inspector.SQL("ROLLBACK");
	Wait([&] {
		try {
			session->Write("/file", "ok", 0);
			return true;
		} catch (const Error &error) {
			Require(error.code == ErrorCode::Busy, error.what());
			return false;
		}
	});
	Require(second->Read("/file", 0, 2) == "ok", "Retry not visible across connections");
	Require(std::filesystem::file_size(path + "-wal") <= 16 * MIB, "Restart retained oversized WAL allocation");
	workspace.Close();
	other.Close();
}

static void CheckErrors(const std::string &path) {
	Workspace workspace(path, 30, Durability::Fsync);
	auto session = workspace.Checkout();
	faults = 0;
	fail_background = true;
	session->WriteFile("/file", std::string(32 * MIB, 'e'));
	Wait([] { return faults > 0; });
	fail_background = false;
	// Close joins the worker before checking its error, including an error
	// concurrently arriving after the final user operation.
	Expect(ErrorCode::Storage, [&] { workspace.Close(); });
	Require(session->Read("/file", 0, MIB) == std::string(MIB, 'e'), "Checkpoint error undid a committed write");
	Inspector inspector(path);
	inspector.SQL("SELECT * FROM format");
	// A retrying worker must resume without needing another commit to wake it.
	Wait([&] { return inspector.CaughtUp(); });
	workspace.Sync();
	inspector.SQL("BEGIN IMMEDIATE");
	Expect(ErrorCode::Busy, [&] { workspace.Close(); });
	inspector.SQL("ROLLBACK");
	auto before = background_writes.load();
	session->WriteFile("/retry", std::string(32 * MIB, 'r'));
	Wait([&] { return background_writes > before && inspector.CaughtUp(); });
	workspace.Close();
	Workspace reopened(path);
	Require(reopened.Checkout()->Read("/retry", 31 * MIB) == std::string(MIB, 'r'), "Close retry lost bytes");
}

static void CheckBackfilledReader(const std::string &path) {
	Workspace workspace(path, 30, Durability::Fsync);
	struct Pause {
		Pause() {
			paused = false;
			pause_background = true;
		}
		~Pause() {
			pause_background = false;
		}
	} pause_worker;
	auto session = workspace.Checkout();
	// A single atomic operation may exceed the admission budget.
	session->WriteFile("/file", std::string(64 * MIB, 'b'));
	Wait([] { return paused.load(); });
	Inspector inspector(path);
	inspector.SQL("BEGIN; SELECT * FROM format");
	pause_background = false;
	Inspector progress(path);
	progress.SQL("SELECT * FROM format");
	Wait([&] { return progress.CaughtUp(); });
	Require(std::filesystem::file_size(path + "-wal") > 64 * MIB, "Missing oversized transaction");
	// Backlog is zero, but this reader still prevents reuse of the WAL.
	Expect(ErrorCode::Busy, [&] { session->Write("/file", "no", 0); });
	workspace.Sync();
	Require(session->Read("/file", 0, 2) == "bb", "Rejected write changed bytes");
	inspector.SQL("ROLLBACK");
	session->Write("/file", "ok", 0);
	workspace.Close();
}

static void CheckConcurrentWriters(const std::string &path) {
	Workspace first(path, 2000, Durability::Fsync), second(path, 2000, Durability::Fsync);
	auto a = first.Checkout(), b = second.Checkout();
	a->WriteFile("/a", "");
	b->WriteFile("/b", "");
	std::exception_ptr errors[2];
	auto write = [&](int index, const std::shared_ptr<Session> &session, const char *name) {
		try {
			for (int i = 0; i < 48; ++i) {
				// Independent connections can legitimately exceed the SQLite
				// lock timeout. Retry the whole rejected operation, then verify
				// both streams rather than assuming a contention-free schedule.
				Wait([&] {
					try {
						session->Write(name, std::string(MIB, char('a' + index)), int64_t(i) * MIB);
						return true;
					} catch (const Error &error) {
						Require(error.code == ErrorCode::Busy, error.what());
						return false;
					}
				});
			}
		} catch (...) {
			errors[index] = std::current_exception();
		}
	};
	std::thread left(write, 0, a, "/a"), right(write, 1, b, "/b");
	left.join();
	right.join();
	for (const auto &error : errors)
		if (error)
			std::rethrow_exception(error);
	first.Sync();
	Require(a->Read("/b") == std::string(48 * MIB, 'b'), "Other writer's bytes differ");
	Require(b->Read("/a") == std::string(48 * MIB, 'a'), "First writer's bytes differ");
	first.Close();
	second.Close();
}

static void CheckSlowCheckpoint(const std::string &path) {
	Workspace workspace(path, 10, Durability::Fsync);
	struct Pause {
		Pause() {
			paused = false;
			pause_background = true;
		}
		~Pause() {
			pause_background = false;
		}
	} pause_worker;
	auto session = workspace.Checkout();
	session->WriteFile("/file", std::string(64 * MIB, 's'));
	Wait([] { return paused.load(); });
	std::thread resume([] {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		pause_background = false;
	});
	std::exception_ptr error;
	try {
		// Waiting for this connection's slow storage must not become a Busy
		// error after 10 ms. There are no competing writers or pinned readers.
		session->Write("/file", "ok", 0);
	} catch (...) {
		error = std::current_exception();
	}
	resume.join();
	if (error)
		std::rethrow_exception(error);
	workspace.Sync();
	Require(session->Read("/file", 0, 2) == "ok", "Slow checkpoint lost the following write");
	workspace.Close();
}

static void CheckFork(const std::string &path) {
	auto workspace = std::make_unique<Workspace>(path, 100, Durability::Fsync);
	auto session = workspace->Checkout();
	auto child = fork();
	Require(child >= 0, "Forking inherited workspace");
	if (child == 0) {
		try {
			Expect(ErrorCode::Closed, [&] { session->Stat("/"); });
			session.reset();
			workspace.reset(); // Must not join or destroy an inherited worker.
			_exit(0);
		} catch (...) {
			_exit(1);
		}
	}
	int status = 0;
	Require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
	        "Inherited worker shutdown failed");
	session->WriteFile("/parent", "alive");
	workspace->Close();
}

static void CheckCrash(const std::string &path) {
	int descriptors[2];
	Require(pipe(descriptors) == 0, "Creating crash pipe");
	auto child = fork();
	Require(child >= 0, "Forking checkpoint writer");
	if (child == 0) {
		close(descriptors[0]);
		try {
			Workspace workspace(path, 1000, Durability::Fsync);
			paused = false;
			pause_background = true;
			auto session = workspace.Checkout();
			session->WriteFile("/file", std::string(32 * MIB, 'c'));
			Wait([] { return paused.load(); });
			workspace.Sync();
			char ready = 'x';
			if (write(descriptors[1], &ready, 1) != 1)
				_exit(2);
			for (;;)
				pause();
		} catch (...) {
			_exit(3);
		}
	}
	close(descriptors[1]);
	pollfd waiting {descriptors[0], POLLIN, 0};
	char ready = 0;
	bool synced = poll(&waiting, 1, 45000) == 1 && read(descriptors[0], &ready, 1) == 1 && ready == 'x';
	close(descriptors[0]);
	kill(child, SIGKILL);
	int status = 0;
	waitpid(child, &status, 0);
	Require(synced && WIFSIGNALED(status), "Writer failed to sync with a paused checkpoint");
	Workspace recovered(path);
	Require(recovered.RecoverOwners().owners == 1, "Checkpoint crash owner not recovered");
	Require(recovered.Checkout()->Read("/file") == std::string(32 * MIB, 'c'), "Checkpoint crash lost synced bytes");
	Inspector inspector(path);
	Require(inspector.SQL("PRAGMA integrity_check") == "ok", "Integrity failure after checkpoint crash");
}

int main() {
	auto root = std::filesystem::temp_directory_path() / ("vane-fs-checkpoint-" + std::to_string(getpid()));
	try {
		std::filesystem::create_directory(root);
		Hooks hooks;
		std::cout << "CheckStartupFailure" << std::endl;
		CheckStartupFailure((root / "startup.sqlite").string());
		std::cout << "CheckProgress" << std::endl;
		CheckProgress((root / "progress.sqlite").string());
		std::cout << "CheckPinnedReader" << std::endl;
		CheckPinnedReader((root / "pinned.sqlite").string(), 4096);
		std::cout << "CheckPinnedReader" << std::endl;
		CheckPinnedReader((root / "pinned-8k.sqlite").string(), 8192);
		std::cout << "CheckErrors" << std::endl;
		CheckErrors((root / "errors.sqlite").string());
		std::cout << "CheckBackfilledReader" << std::endl;
		CheckBackfilledReader((root / "backfilled.sqlite").string());
		std::cout << "CheckConcurrentWriters" << std::endl;
		CheckConcurrentWriters((root / "concurrent.sqlite").string());
		std::cout << "CheckSlowCheckpoint" << std::endl;
		CheckSlowCheckpoint((root / "slow.sqlite").string());
		std::cout << "CheckFork" << std::endl;
		CheckFork((root / "fork.sqlite").string());
		std::cout << "CheckCrash" << std::endl;
		CheckCrash((root / "crash.sqlite").string());
		std::filesystem::remove_all(root);
		std::cout << "Background progress, budgets, errors, retry, fork and crash recovery passed\n";
		return 0;
	} catch (const std::exception &error) {
		fail_background = false;
		pause_background = false;
		std::cerr << error.what() << '\n';
		std::filesystem::remove_all(root);
		return 1;
	}
}
