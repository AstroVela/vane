// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Linux-only experiment. Published extents are immutable; SQLite stores their
// offsets. Data is synchronized before references can enter a committed WAL.
#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <thread>
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vane_fs {
class PayloadFile {
public:
	static constexpr int64_t BLOCK = 4096;
	static constexpr const char *CLIENT_DATA = "vane_fs.external_payload_experiment";
	PayloadFile(const std::string &database, bool create, int timeout) : timeout(timeout) {
		path = std::filesystem::canonical(database).string() + ".payload";
		fd = open(path.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW | (create ? O_CREAT : 0), 0600);
		if (fd < 0)
			Failure("Opening payload file");
		struct stat st {};
		if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
			close(fd);
			fd = -1;
			throw Error(ErrorCode::Storage, "Payload store is not a regular file");
		}
	}
	~PayloadFile() {
		if (fd >= 0)
			close(fd);
	}
	PayloadFile(const PayloadFile &) = delete;
	PayloadFile &operator=(const PayloadFile &) = delete;
	static PayloadFile &Get(sqlite3 *db) {
		auto *file = static_cast<PayloadFile *>(sqlite3_get_clientdata(db, CLIENT_DATA));
		if (!file)
			throw Error(ErrorCode::Storage, "Missing payload store");
		return *file;
	}
	void Initialize(const std::string &uuid, bool fresh) {
		std::string header = "VANE-EXTERNAL-1003:" + uuid;
		header.resize(BLOCK, '\0');
		if (fresh) {
			// A failed first initialization may leave a sidecar. Refuse it rather
			// than guessing ownership or overwriting bytes from another database.
			if (Size() != 0)
				throw Error(ErrorCode::Storage, "Existing payload file for an empty database");
			Write(header.data(), header.size(), 0);
			Flush();
			int directory = open(std::filesystem::path(path).parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
			if (directory < 0)
				Failure("Opening payload directory");
			int rc = fsync(directory), saved = errno;
			close(directory);
			if (rc != 0) {
				errno = saved;
				Failure("Synchronizing payload directory");
			}
		} else {
			std::string actual(BLOCK, '\0');
			Read(actual.data(), actual.size(), 0);
			if (actual != header)
				throw Error(ErrorCode::Storage, "Payload file belongs to a different workspace");
		}
	}
	class Guard {
	public:
		Guard(PayloadFile &file, bool exclusive) : fd(file.fd), process(getpid()) {
			auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(file.timeout);
			for (;;) {
				if (flock(fd, (exclusive ? LOCK_EX : LOCK_SH) | LOCK_NB) == 0)
					return;
				if (errno != EWOULDBLOCK && errno != EAGAIN && errno != EINTR)
					Failure("Locking payload file");
				if (std::chrono::steady_clock::now() >= end)
					throw Error(ErrorCode::Busy, "Payload file is busy");
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
		}
		~Guard() {
			// A fork child must not unlock its parent's open-file description.
			if (getpid() == process)
				flock(fd, LOCK_UN);
		}
		Guard(const Guard &) = delete;
		Guard &operator=(const Guard &) = delete;

	private:
		int fd;
		pid_t process;
	};
	int64_t Append(const std::string &data) {
		// The caller holds SQLite's IMMEDIATE writer transaction, serializing
		// append allocation across connections and processes.
		auto end = Size();
		if (end > std::numeric_limits<int64_t>::max() - BLOCK - int64_t(data.size()))
			throw Error(ErrorCode::Capacity, "Payload file is too large");
		auto offset = (end + BLOCK - 1) / BLOCK * BLOCK;
		Write(data.data(), data.size(), offset);
		return offset;
	}
	static std::string Token(int64_t offset) {
		std::string token(8, '\0');
		for (int i = 7; i >= 0; --i) {
			token[size_t(i)] = char(offset & 255);
			offset >>= 8;
		}
		return token;
	}
	static int64_t Offset(const std::string &token) {
		if (token.size() != 8)
			throw Error(ErrorCode::Storage, "Invalid external payload token");
		uint64_t value = 0;
		for (unsigned char byte : token)
			value = (value << 8) | byte;
		if (value < BLOCK || value % BLOCK || value > uint64_t(std::numeric_limits<int64_t>::max() - BLOCK))
			throw Error(ErrorCode::Storage, "Invalid external payload offset");
		return int64_t(value);
	}
	void Read(char *data, size_t size, int64_t offset) {
		while (size) {
			auto n = pread(fd, data, size, offset);
			if (n < 0 && errno == EINTR)
				continue;
			if (n < 0)
				Failure("Reading payload file");
			if (!n)
				throw Error(ErrorCode::Storage, "Truncated payload file");
			data += n;
			size -= size_t(n);
			offset += n;
		}
	}
	void Flush() {
		if (dirty) {
			int rc;
			do {
				rc = fdatasync(fd);
			} while (rc != 0 && errno == EINTR);
			if (rc != 0)
				Failure("Synchronizing payload file");
			dirty = false;
		}
	}
	void Reclaim(std::vector<int64_t> live) {
		// Caller holds the exclusive payload guard across a FULL metadata
		// commit and this sweep. Old native readers cannot retain dead offsets.
		std::sort(live.begin(), live.end());
		live.erase(std::unique(live.begin(), live.end()), live.end());
		auto size = Size();
		for (auto offset : live)
			if (offset < BLOCK || offset > size - BLOCK)
				throw Error(ErrorCode::Storage, "Live payload exceeds file size");
		int64_t end = BLOCK;
		for (auto offset : live) {
			if (offset > end && fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, end, offset - end) != 0)
				Failure("Reclaiming unused payload range");
			end = offset + BLOCK;
		}
		if (size != end && ftruncate(fd, end) != 0)
			Failure("Truncating unused payload tail");
		dirty = true;
		Flush();
	}

private:
	[[noreturn]] static void Failure(const char *operation) {
		throw Error(ErrorCode::Storage, std::string(operation) + ": " + std::strerror(errno));
	}
	int64_t Size() {
		struct stat st {};
		if (fstat(fd, &st) != 0)
			Failure("Inspecting payload file");
		return st.st_size;
	}
	void Write(const char *data, size_t size, int64_t offset) {
		dirty = true;
		while (size) {
			auto n = pwrite(fd, data, size, offset);
			if (n < 0 && errno == EINTR)
				continue;
			if (n <= 0)
				Failure("Writing payload file");
			data += n;
			size -= size_t(n);
			offset += n;
		}
	}
	std::string path;
	int fd = -1;
	int timeout;
	bool dirty = false;
};
} // namespace vane_fs
