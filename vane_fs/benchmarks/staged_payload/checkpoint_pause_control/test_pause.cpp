// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "support.hpp"
#include <atomic>
#include <map>

static const auto foreground = std::this_thread::get_id();
static std::atomic<bool> fail_foreground_sync {false}, fail_background_write {false};
static std::atomic<int> foreground_failures {0}, background_failures {0};
struct FileHooks {
	sqlite3_io_methods methods;
	const sqlite3_io_methods *original;
	int flags;
};
static std::map<sqlite3_file *, FileHooks> files;
static std::mutex files_mutex;
static sqlite3_vfs *original_vfs;

static int Write(sqlite3_file *file, const void *data, int bytes, sqlite3_int64 offset) {
	FileHooks hook;
	{
		std::lock_guard<std::mutex> lock(files_mutex);
		hook = files.at(file);
	}
	if (std::this_thread::get_id() != foreground && (hook.flags & SQLITE_OPEN_MAIN_DB) && fail_background_write) {
		++background_failures;
		return SQLITE_IOERR_WRITE;
	}
	return hook.original->xWrite(file, data, bytes, offset);
}
static int Sync(sqlite3_file *file, int flags) {
	FileHooks hook;
	{
		std::lock_guard<std::mutex> lock(files_mutex);
		hook = files.at(file);
	}
	if (std::this_thread::get_id() == foreground && (hook.flags & SQLITE_OPEN_WAL) && fail_foreground_sync) {
		++foreground_failures;
		return SQLITE_IOERR_FSYNC;
	}
	return hook.original->xSync(file, flags);
}
static int CloseFile(sqlite3_file *file) {
	{
		std::lock_guard<std::mutex> lock(files_mutex);
		file->pMethods = files.at(file).original;
		files.erase(file);
	}
	return file->pMethods->xClose(file);
}
static int Open(sqlite3_vfs *, const char *name, sqlite3_file *file, int flags, int *out) {
	auto rc = original_vfs->xOpen(original_vfs, name, file, flags, out);
	if (file->pMethods) {
		std::lock_guard<std::mutex> lock(files_mutex);
		auto &hook = files.emplace(file, FileHooks {*file->pMethods, file->pMethods, flags}).first->second;
		hook.methods.xWrite = Write;
		hook.methods.xSync = Sync;
		hook.methods.xClose = CloseFile;
		file->pMethods = &hook.methods;
	}
	return rc;
}
struct Hooks {
	sqlite3_vfs vfs;
	Hooks() {
		original_vfs = sqlite3_vfs_find(nullptr);
		Require(original_vfs != nullptr, "Missing SQLite VFS");
		vfs = *original_vfs;
		vfs.zName = "vane-pause-fault-test";
		vfs.xOpen = Open;
		Require(sqlite3_vfs_register(&vfs, 1) == SQLITE_OK, "Installing test VFS");
	}
	~Hooks() {
		fail_foreground_sync = fail_background_write = false;
		sqlite3_vfs_unregister(&vfs);
	}
};
template <class F>
static void Expect(ErrorCode code, F operation) {
	try {
		operation();
	} catch (const Error &error) {
		Require(error.code == code, error.what());
		return;
	}
	throw std::runtime_error("Expected injected failure");
}

int main(int argc, char **argv) {
	try {
		Require(argc == 2, "Expected test database");
		Hooks hooks;
		fuse_waits::Install(argv[1]);
		{
			fuse_waits::Root root("prewriteback_worker");
			Require(!fuse_waits::PayloadWorkerIdle(), "Payload-root counter missed an active root");
		}
		Require(fuse_waits::PayloadWorkerIdle(), "Payload-root counter did not retire");
		Workspace workspace(argv[1], 30, Durability::Fsync);
		auto session = workspace.Checkout();
		session->WriteFile("/file", "before");
		Warm();
		{
			fuse_waits::Root root("paused_full_failure");
			fail_foreground_sync = true;
			Expect(ErrorCode::Storage, [&] { workspace.Sync(); });
			fail_foreground_sync = false;
			workspace.Sync();
			Require(foreground_failures > 0 && session->Read("/file") == "before", "Paused FULL failure was lost");
		}
		fail_background_write = true;
		worker->Start();
		Require(sqlite3_exec(writer, "INSERT INTO diagnostic_padding VALUES(zeroblob(1048576))", nullptr, nullptr,
		                     nullptr) == SQLITE_OK,
		        "Creating pending backfill");
		Await([] { return background_failures > 0; });
		worker->Stop();
		fail_background_write = false;
		Expect(ErrorCode::Storage, [] { worker->CheckError(); });
		worker->Start();
		Await(CaughtUp); // A preserved retry must progress without another commit.
		worker->Stop();
		worker->CheckError();
		{
			SQL lock(argv[1]);
			lock.Exec("BEGIN IMMEDIATE");
			Expect(ErrorCode::Busy, [&] { workspace.Close(); });
			lock.Exec("ROLLBACK");
		}
		session->WriteFile("/file", "closing");
		fail_foreground_sync = true;
		Expect(ErrorCode::Storage, [&] { workspace.Close(); });
		fail_foreground_sync = false;
		Require(session->Read("/file") == "closing", "Failed close lost committed data");
		workspace.Close();
		Require(writer == nullptr && worker == nullptr, "Close did not clear its WAL callback");
		{
			Workspace reopened(argv[1]);
			Require(reopened.Checkout()->Read("/file") == "closing", "Close retry changed durable content");
			reopened.Close();
		}
		Require(files.empty(), "Test VFS retained an open file");
		std::cout << "PASS payload-root counter, paused FULL retry, stopped worker error, resume without commit, "
		             "busy/IO close retry\n";
		return 0;
	} catch (const std::exception &error) {
		fail_foreground_sync = fail_background_write = false;
		std::cerr << error.what() << '\n';
		return 1;
	}
}
