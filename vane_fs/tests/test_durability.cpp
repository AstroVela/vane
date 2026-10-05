// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <filesystem>
#include <iostream>
#include <map>
#include <sys/wait.h>
#include <unistd.h>

using namespace vane_fs;

static void Require(bool condition, const char *message) {
	if (!condition)
		throw std::runtime_error(message);
}
template <class F>
static void Expect(ErrorCode code, F operation) {
	try {
		operation();
	} catch (const Error &error) {
		Require(error.code == code, "Unexpected error code");
		return;
	}
	throw std::runtime_error("Expected an exception");
}

// Intercept the VFS sync method, so failures and counts exercise actual storage
// barriers for either a static or shared SQLite build.
static int syncs = 0;
static bool fail_sync = false;
struct FileHooks {
	sqlite3_io_methods methods;
	const sqlite3_io_methods *original;
	std::string path;
	int flags;
};
static std::map<sqlite3_file *, FileHooks> files;
static sqlite3_vfs *original_vfs = nullptr;
static std::filesystem::path synced_images;
static int Sync(sqlite3_file *file, int flags) {
	++syncs;
	if (fail_sync)
		return SQLITE_IOERR_FSYNC;
	auto &hook = files.at(file);
	auto result = hook.original->xSync(file, flags);
	if (result == SQLITE_OK && !synced_images.empty() && (hook.flags & (SQLITE_OPEN_MAIN_DB | SQLITE_OPEN_WAL))) {
		std::error_code error;
		std::filesystem::copy_file(hook.path, synced_images / std::filesystem::path(hook.path).filename(),
		                           std::filesystem::copy_options::overwrite_existing, error);
		if (error)
			return SQLITE_IOERR_FSYNC;
	}
	return result;
}
static int Close(sqlite3_file *file) {
	file->pMethods = files.at(file).original;
	files.erase(file);
	return file->pMethods->xClose(file);
}
static int Open(sqlite3_vfs *, const char *name, sqlite3_file *file, int flags, int *output_flags) {
	auto result = original_vfs->xOpen(original_vfs, name, file, flags, output_flags);
	if (file->pMethods) {
		auto &hook =
		    files.emplace(file, FileHooks {*file->pMethods, file->pMethods, name ? name : "", flags}).first->second;
		hook.methods.xSync = Sync;
		hook.methods.xClose = Close;
		file->pMethods = &hook.methods;
	}
	return result;
}
struct SyncHooks {
	sqlite3_vfs vfs;
	SyncHooks() {
		original_vfs = sqlite3_vfs_find(nullptr);
		Require(original_vfs != nullptr, "Missing SQLite VFS");
		vfs = *original_vfs;
		vfs.zName = "vane-fs-test-sync";
		vfs.xOpen = Open;
		Require(sqlite3_vfs_register(&vfs, 1) == SQLITE_OK, "Installing sync VFS");
	}
	~SyncHooks() {
		fail_sync = false;
		sqlite3_vfs_unregister(&vfs);
	}
};

static void Check(const std::string &path, Durability durability) {
	Workspace workspace(path, 10, durability);
	auto session = workspace.Checkout();
	session->WriteFile("/file", "before");
	workspace.AcquireMount("main");
	auto file = session->OpenFile("/file");
	Workspace observer(path, 10);
	auto reader = observer.Checkout();
	// Keep an old read transaction open so WAL reset/checkpoint does not add
	// incidental syncs to a NORMAL write. It must not prevent a WAL barrier.
	sqlite3 *inspection = nullptr;
	Require(sqlite3_open(path.c_str(), &inspection) == SQLITE_OK, "Opening inspector");
	Require(sqlite3_exec(inspection, "BEGIN; SELECT * FROM format", nullptr, nullptr, nullptr) == SQLITE_OK,
	        "Pinning WAL reader");
	try {
		syncs = 0;
		session->WriteInode(file.inode, "visible", 0);
		Require((syncs > 0) == (durability == Durability::Strict), "Wrong ordinary write durability");
		Require(reader->Read("/file") == "visible", "Write not visible across connections before fsync");
		syncs = 0;
		workspace.Sync();
		Require(syncs > 0, "Sync did not reach storage with a pinned WAL reader");
		syncs = 0;
		workspace.Sync();
		Require(syncs > 0, "Repeated barrier became a no-op transaction");
		if (durability == Durability::Fsync) {
			syncs = 0;
			session->WriteInode(file.inode, "pending", 0);
			Require(syncs == 0, "Sync failed to restore NORMAL");
			syncs = 0;
			observer.Sync();
			Require(syncs > 0, "Strict connection did not synchronize another connection's NORMAL writes");
		}
		syncs = 0;
		session->WriteInode(file.inode, "synced!", 0, false, true);
		Require(syncs > 0, "Synchronous handle write did not sync");
		fail_sync = true;
		Expect(ErrorCode::Storage, [&] { workspace.Sync(); });
		Expect(ErrorCode::Storage, [&] { session->WriteInode(file.inode, "failed!", 0, false, true); });
		fail_sync = false;
		Require(reader->Read("/file") == "synced!", "Failed synchronous write published data");
		syncs = 0;
		workspace.Sync();
		Require(syncs > 0, "Failed barrier was not retryable");
		session->WriteInode(file.inode, "retry!!", 0, false, true);
		Require(reader->Read("/file") == "retry!!", "Synchronous write retry failed");
		// A busy barrier must fail visibly and remain retryable as well.
		Require(sqlite3_exec(inspection, "ROLLBACK; BEGIN IMMEDIATE", nullptr, nullptr, nullptr) == SQLITE_OK,
		        "Holding write lock");
		Expect(ErrorCode::Busy, [&] { workspace.Sync(); });
		Require(sqlite3_exec(inspection, "ROLLBACK", nullptr, nullptr, nullptr) == SQLITE_OK, "Releasing write lock");
		workspace.Sync();
		// Close must synchronize outstanding commits even without file fsync.
		session->WriteInode(file.inode, "closing", 0);
		fail_sync = true;
		Expect(ErrorCode::Storage, [&] { workspace.Close(); });
		fail_sync = false;
		Require(reader->Read("/file") == "closing", "Failed close lost committed data");
		syncs = 0;
		workspace.Close();
		Require(syncs > 0, "Close retry did not sync");
		Expect(ErrorCode::Closed, [&] { workspace.Sync(); });
		observer.Close();
		sqlite3_close(inspection);
	} catch (...) {
		fail_sync = false;
		sqlite3_close(inspection);
		throw;
	}
	Workspace reopened(path);
	Require(reopened.Checkout()->Read("/file") == "closing", "Reopened data differs");
}

static void CheckSyncedImages(const std::filesystem::path &root) {
	// A deterministic loss model: only bytes observed at successful VFS xSync
	// calls survive. Process-kill alone cannot exercise loss of OS page cache.
	auto images = root / "synced";
	std::filesystem::create_directory(images);
	auto child = fork();
	Require(child >= 0, "Forking durable image writer");
	if (child == 0) {
		try {
			synced_images = images;
			Workspace workspace((root / "model.sqlite").string(), 10, Durability::Fsync);
			auto session = workspace.Checkout();
			session->WriteFile("/before", "initial");
			workspace.Sync();
			session->Rename("/before", "/renamed");
			session->WriteFile("/renamed", "durable content");
			workspace.Sync();
			syncs = 0;
			session->WriteFile("/renamed", "unsynchronized overwrite");
			Require(syncs == 0, "Loss model unexpectedly synchronized later writes");
			_exit(0); // No destructor, close, or checkpoint may help the barrier.
		} catch (const std::exception &error) {
			std::cerr << error.what() << '\n';
			_exit(1);
		}
	}
	int status = 0;
	Require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
	        "Durable image writer failed");
	Workspace recovered((images / "model.sqlite").string());
	recovered.RecoverOwners();
	Require(recovered.Checkout()->Read("/renamed") == "durable content", "Barrier did not persist file content");
	Expect(ErrorCode::NotFound, [&] { recovered.Checkout()->Stat("/before"); });
	recovered.Close();
}

int main() {
	auto root = std::filesystem::temp_directory_path() / ("vane-fs-durability-" + std::to_string(getpid()));
	try {
		std::filesystem::create_directory(root);
		SyncHooks hooks;
		Check((root / "strict.sqlite").string(), Durability::Strict);
		Check((root / "fsync.sqlite").string(), Durability::Fsync);
		CheckSyncedImages(root);
		std::filesystem::remove_all(root);
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		std::filesystem::remove_all(root);
		return 1;
	}
}
