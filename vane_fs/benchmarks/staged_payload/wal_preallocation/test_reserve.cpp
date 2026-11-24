// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "../checkpoint_progress/fixture.hpp"
#include <csignal>
#include <map>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

static std::string mode;
static int hints = 0, resets = 0, allocation_writes = 0, notification = -1;
static thread_local bool allocating = false;
static constexpr int64_t RESERVE = 16 * 1024 * 1024;
extern "C" ssize_t __real_pwrite64(int, const void *, size_t, off64_t);
extern "C" ssize_t __wrap_pwrite64(int fd, const void *data, size_t size, off64_t offset) {
	if (allocating && mode == "partial" && ++allocation_writes == 2) {
		errno = EIO;
		return -1;
	}
	return __real_pwrite64(fd, data, size, offset);
}
static void Pause() {
	char ready = 'r';
	Require(write(notification, &ready, 1) == 1, "Notify crash parent");
	raise(SIGSTOP);
}
struct FileHooks {
	sqlite3_io_methods methods;
	const sqlite3_io_methods *original;
	int flags;
};
static std::map<sqlite3_file *, FileHooks> files;
static std::mutex files_mutex;
static sqlite3_vfs *original_vfs;
static int Control(sqlite3_file *file, int op, void *argument) {
	const sqlite3_io_methods *original;
	bool wal;
	{
		std::lock_guard<std::mutex> lock(files_mutex);
		original = files.at(file).original;
		wal = (files.at(file).flags & SQLITE_OPEN_WAL) != 0;
	}
	if (wal && op == SQLITE_FCNTL_CHUNK_SIZE) {
		if (*static_cast<int *>(argument) > 0 && mode == "unsupported")
			return SQLITE_NOTFOUND;
		if (*static_cast<int *>(argument) == 0)
			++resets;
	}
	if (wal && op == SQLITE_FCNTL_SIZE_HINT) {
		++hints;
		Require(*static_cast<sqlite3_int64 *>(argument) == RESERVE, "Unbounded WAL reservation");
		if (mode == "hint_unsupported")
			return SQLITE_NOTFOUND;
		if (mode == "error")
			return SQLITE_IOERR_WRITE;
		allocating = true;
		auto code = original->xFileControl(file, op, argument);
		allocating = false;
		if (mode == "crash_prepare" && code == SQLITE_OK)
			Pause();
		return code;
	}
	return original->xFileControl(file, op, argument);
}
static int Close(sqlite3_file *file) {
	{
		std::lock_guard<std::mutex> lock(files_mutex);
		file->pMethods = files.at(file).original;
		files.erase(file);
	}
	return file->pMethods->xClose(file);
}
static int Open(sqlite3_vfs *, const char *name, sqlite3_file *file, int flags, int *output) {
	auto code = original_vfs->xOpen(original_vfs, name, file, flags, output);
	if (file->pMethods) {
		std::lock_guard<std::mutex> lock(files_mutex);
		auto &hook = files.emplace(file, FileHooks {*file->pMethods, file->pMethods, flags}).first->second;
		hook.methods.xFileControl = Control;
		hook.methods.xClose = Close;
		file->pMethods = &hook.methods;
	}
	return code;
}
struct Hooks {
	sqlite3_vfs vfs;
	Hooks() {
		original_vfs = sqlite3_vfs_find(nullptr);
		vfs = *original_vfs;
		vfs.zName = "vane-wal-reservation-test";
		vfs.xOpen = Open;
		Require(sqlite3_vfs_register(&vfs, 1) == SQLITE_OK, "Install reserve VFS");
	}
	~Hooks() {
		sqlite3_vfs_unregister(&vfs);
	}
};
static void Content(Workspace &workspace) {
	auto session = workspace.Checkout();
	session->WriteFile("/ack", std::string(4096, 'a'));
	workspace.Sync();
	Require(session->Read("/ack") == std::string(4096, 'a'), "Acknowledged content changed");
}
static void Crash(const std::string &path) {
	int pipefd[2];
	Require(pipe(pipefd) == 0, "Create crash pipe");
	auto child = fork();
	Require(child >= 0, "Fork reserve crash");
	if (!child) {
		close(pipefd[0]);
		notification = pipefd[1];
		try {
			Workspace workspace(path, 5000, Durability::Fsync);
			Content(workspace);
			Pause();
		} catch (const std::exception &error) {
			std::cerr << error.what() << '\n';
			_exit(1);
		}
		_exit(2);
	}
	close(pipefd[1]);
	pollfd waiting {pipefd[0], POLLIN, 0};
	char ready = 0;
	auto stopped = poll(&waiting, 1, 10000) == 1 && read(pipefd[0], &ready, 1) == 1 && ready == 'r';
	close(pipefd[0]);
	kill(child, SIGKILL);
	int status = 0;
	Require(waitpid(child, &status, 0) == child && stopped && WIFSIGNALED(status), "Crash did not reach target");
	auto prepared_only = mode == "crash_prepare";
	mode = "off";
	Workspace recovered(path, 5000, Durability::Fsync);
	Require(recovered.RecoverOwners().owners == 1, "Crash owner was not recoverable");
	auto session = recovered.Checkout();
	if (prepared_only)
		Require(session->ListDirectory("/").empty(), "Reservation created phantom directory entries");
	else
		Require(session->Read("/ack") == std::string(4096, 'a'), "Crash lost first FULL write");
	Content(recovered);
	recovered.Close();
	SQL inspect(path);
	Require(inspect.Text("PRAGMA quick_check") == "ok", "Crash recovery integrity failed");
}
int main(int argc, char **argv) {
	try {
		Require(argc == 3, "Expected database and case");
		std::string path = argv[1];
		mode = argv[2];
		Hooks hooks;
		if (mode == "crash_prepare" || mode == "crash_ack") {
			Crash(path);
		} else if (mode == "error" || mode == "partial") {
			bool failed = false;
			try {
				Workspace workspace(path, 5000, Durability::Fsync);
			} catch (const Error &error) {
				Require(error.code == ErrorCode::Storage, "Wrong allocation error code");
				Require(std::string(error.what()).find("Preparing WAL allocation") != std::string::npos,
				        "Missing allocation failure context");
				failed = true;
			}
			Require(failed && hints == 1 && resets == 1, "Allocation error or reset was lost");
			if (mode == "partial")
				Require(allocation_writes == 2, "Partial allocation fault did not fire");
			mode = "off";
			Workspace recovered(path, 5000, Durability::Fsync);
			Require(hints == 1 && recovered.RecoverOwners().owners == 1, "Failed allocation owner leaked");
			Content(recovered);
			recovered.Close();
		} else {
			Require(mode == "normal" || mode == "strict" || mode == "reopen" || mode == "unsupported" ||
			            mode == "hint_unsupported",
			        "Unknown case");
			Workspace workspace(path, 5000, mode == "strict" ? Durability::Strict : Durability::Fsync);
			auto expected = mode == "strict" || mode == "unsupported" ? 0 : 1;
			Require(hints == expected, "New filesystem omitted WAL reservation");
			if (mode == "normal" || mode == "reopen") {
				Require(resets == 1 && std::filesystem::file_size(path + "-wal") == RESERVE,
				        "Reservation size or reset differs");
			}
			Content(workspace);
			workspace.Close();
			Workspace reopened(path, 5000, Durability::Fsync);
			Require(hints == expected, "Reopened workspace reserved again");
			Require(reopened.Checkout()->Read("/ack") == std::string(4096, 'a'), "Reopen lost acknowledged bytes");
			reopened.Close();
		}
		std::cout << "PASS " << argv[2] << " hints=" << hints << " resets=" << resets << '\n';
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
