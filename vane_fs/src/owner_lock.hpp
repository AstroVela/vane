// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vane_fs/workspace.hpp"
#include <filesystem>
#include <cstring>
#include <string>

#ifndef _WIN32
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace vane_fs {

// A separate open-file-description lock per connection also distinguishes
// owners in the same process. Never unlink a lock before its SQL owner retires.
class OwnerLock {
public:
	std::string path;
	int64_t device = 0, inode = 0;
	OwnerLock() = default;
	OwnerLock(const OwnerLock &) = delete;
	OwnerLock &operator=(const OwnerLock &) = delete;
	~OwnerLock() {
		Close();
	}
	void Create(const std::string &database, const std::string &owner) {
#ifndef _WIN32
		auto directory = std::filesystem::canonical(database).string() + ".vane_fs-locks";
		std::filesystem::create_directory(directory);
		path = directory + "/" + owner;
		fd = open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
		owns_path = fd >= 0;
		struct stat metadata {};
		if (fd < 0 || flock(fd, LOCK_EX | LOCK_NB) != 0 || fstat(fd, &metadata) != 0) {
			throw Error(ErrorCode::Storage, "Could not acquire VaneFS owner lock: " +
			                                    std::string(std::strerror(errno)) + " (" + path + ")");
		}
		device = metadata.st_dev;
		inode = metadata.st_ino;
#endif
	}
	bool TryRecover(const std::string &filename, int64_t expected_device, int64_t expected_inode) {
#ifndef _WIN32
		if (filename.empty()) {
			return false;
		}
		fd = open(filename.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
		struct stat metadata {};
		if (fd < 0 || fstat(fd, &metadata) != 0 || !S_ISREG(metadata.st_mode) ||
		    int64_t(metadata.st_dev) != expected_device || int64_t(metadata.st_ino) != expected_inode ||
		    flock(fd, LOCK_EX | LOCK_NB) != 0) {
			Close();
			return false;
		}
		path = filename;
		device = expected_device;
		inode = expected_inode;
		owns_path = true;
		return true;
#else
		return false;
#endif
	}
	void Remove() {
		if (owns_path && !path.empty()) {
			bool same_file = false;
#ifndef _WIN32
			struct stat metadata {};
			same_file = lstat(path.c_str(), &metadata) == 0 && int64_t(metadata.st_dev) == device &&
			            int64_t(metadata.st_ino) == inode;
#endif
			std::error_code error;
			if (same_file)
				std::filesystem::remove(path, error);
			owns_path = false;
		}
		Close();
	}
	void Close() {
#ifndef _WIN32
		if (fd >= 0) {
			// close, not LOCK_UN: forked children may share this description.
			close(fd);
			fd = -1;
		}
#endif
	}
	static int64_t Process() {
#ifndef _WIN32
		return getpid();
#else
		return 0;
#endif
	}

private:
	bool owns_path = false;
#ifndef _WIN32
	int fd = -1;
#endif
};

} // namespace vane_fs
