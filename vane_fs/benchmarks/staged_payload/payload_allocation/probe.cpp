// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Ordinary-file hypothesis probe: no SQLite, staging fence or FUSE adapter.
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <linux/fiemap.h>
#include <linux/fs.h>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
constexpr int64_t MIB = 1024 * 1024, BLOCK = 4096, APPEND = MIB / 4;
using Clock = std::chrono::steady_clock;
int64_t Now() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
void Require(bool value, const char *message) {
	if (!value)
		throw std::runtime_error(std::string(message) + ": " + std::strerror(errno));
}
struct File {
	int fd;
	explicit File(const std::string &path, int flags) : fd(open(path.c_str(), flags | O_CLOEXEC, 0600)) {
		Require(fd >= 0, "open");
	}
	~File() {
		close(fd);
	}
};
int64_t Size(int fd) {
	struct stat st {};
	Require(!fstat(fd, &st), "fstat");
	return st.st_size;
}
void Write(int fd, const char *data, int64_t size, int64_t offset) {
	while (size) {
		auto n = pwrite(fd, data, size_t(size), offset);
		if (n < 0 && errno == EINTR)
			continue;
		Require(n > 0, "pwrite");
		data += n;
		offset += n;
		size -= n;
	}
}
void Sync(int fd) {
	int code;
	do {
		code = fdatasync(fd);
	} while (code && errno == EINTR);
	Require(!code, "fdatasync");
}
void Reserve(int fd, int64_t offset, int64_t size) {
	int code;
	do {
		code = fallocate(fd, FALLOC_FL_KEEP_SIZE, offset, size);
	} while (code && errno == EINTR);
	Require(!code, "fallocate KEEP_SIZE");
}
struct Call {
	const char *kind;
	int64_t begin, end, offset, bytes;
};
struct Recorder {
	std::vector<Call> calls;
	template <class F>
	void Run(const char *kind, int64_t offset, int64_t bytes, F operation) {
#ifdef VANE_ALLOCATION_TRACE
		auto begin = Now();
#endif
		operation();
#ifdef VANE_ALLOCATION_TRACE
		calls.push_back({kind, begin, Now(), offset, bytes});
#else
		(void)kind;
		(void)offset;
		(void)bytes;
#endif
	}
};
void Validate(const std::string &path, bool aged, int64_t total) {
	File file(path, O_RDONLY);
	Require(Size(file.fd) == BLOCK + (aged ? 48 * MIB : 0) + total, "logical EOF changed");
	std::vector<char> bytes(size_t(APPEND), '\0');
	int64_t offset = 0, end = Size(file.fd);
	while (offset < end) {
		auto count = std::min(APPEND, end - offset);
		int64_t done = 0;
		while (done < count) {
			auto n = pread(file.fd, bytes.data() + done, size_t(count - done), offset + done);
			if (n < 0 && errno == EINTR)
				continue;
			Require(n > 0, "validation pread");
			done += n;
		}
		for (int64_t i = 0; i < count; ++i) {
			auto at = offset + i - BLOCK;
			char expected = at < 0 ? 'H' : aged && at < 48 * MIB ? (at / (16 * MIB) == 1 ? '\0' : 'g') : 'w';
			Require(bytes[size_t(i)] == expected, "content or retained guard differs");
		}
		offset += count;
	}
}
void PrintMap(int fd) {
	// Inspect only after the measured barrier; FIEMAP_FLAG_SYNC is unnecessary.
	constexpr uint32_t capacity = 4096;
	std::vector<unsigned long long> storage((sizeof(fiemap) + capacity * sizeof(fiemap_extent) + 7) / 8, 0);
	auto *map = reinterpret_cast<fiemap *>(storage.data());
	map->fm_length = FIEMAP_MAX_OFFSET;
	map->fm_extent_count = capacity;
	auto code = ioctl(fd, FS_IOC_FIEMAP, map);
	std::cout << ",\"fiemap_code\":" << code << ",\"fiemap_errno\":" << (code ? errno : 0) << ",\"extents\":[";
	if (!code) {
		Require(map->fm_mapped_extents < capacity, "FIEMAP buffer exhausted");
		for (uint32_t i = 0; i < map->fm_mapped_extents; ++i) {
			auto &e = map->fm_extents[i];
			std::cout << (i ? "," : "") << '[' << e.fe_logical << ',' << e.fe_physical << ',' << e.fe_length << ','
			          << e.fe_flags << ']';
		}
	}
	std::cout << ']';
}
} // namespace

int main(int argc, char **argv) {
	try {
		Require(argc == 5, "expected directory, delayed|reserve256k|reserve1m, fresh|after_gc, MiB");
		std::string root = argv[1], variant = argv[2], state = argv[3], path = root + "/payload";
		Require(variant == "delayed" || variant == "reserve256k" || variant == "reserve1m", "unknown variant");
		Require(state == "fresh" || state == "after_gc", "unknown state");
		int64_t total = std::stoll(argv[4]) * MIB;
		Require(total > 0 && total <= 256 * MIB, "invalid length");
		bool aged = state == "after_gc";
		std::vector<char> allocation(size_t(APPEND + BLOCK), 'w');
		auto *data = allocation.data();
		// Match the application buffer's 48-byte page offset without copying in timing.
		data += (48 + BLOCK - reinterpret_cast<uintptr_t>(data) % BLOCK) % BLOCK;
		std::string header(size_t(BLOCK), 'H'), guard(size_t(APPEND), 'g');
		Recorder recorder;
		recorder.calls.reserve(2100);
		int64_t preparation_begin = Now();
		if (aged) {
			File file(path, O_CREAT | O_EXCL | O_RDWR), directory(root, O_RDONLY | O_DIRECTORY);
			Write(file.fd, header.data(), BLOCK, 0);
			for (int64_t offset = BLOCK; offset < BLOCK + 64 * MIB; offset += APPEND)
				Write(file.fd, guard.data(), APPEND, offset);
			Sync(file.fd);
			Require(!fallocate(file.fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, BLOCK + 16 * MIB, 16 * MIB),
			        "hole punch");
			Require(!ftruncate(file.fd, BLOCK + 48 * MIB), "GC tail truncate");
			Sync(file.fd);
			Require(!fsync(directory.fd), "preparation directory fsync");
		}
		int64_t preparation_end = Now(), begin = Now(), initialized, write_end, sync_end, end;
		int64_t reserves = 0, initial_size = BLOCK + (aged ? 48 * MIB : 0);
		{
			File file(path, O_RDWR | (aged ? 0 : O_CREAT | O_EXCL)), directory(root, O_RDONLY | O_DIRECTORY);
			if (!aged) {
				recorder.Run("header_write", 0, BLOCK, [&] { Write(file.fd, header.data(), BLOCK, 0); });
				recorder.Run("header_sync", 0, 0, [&] { Sync(file.fd); });
				recorder.Run("directory_sync", 0, 0, [&] { Require(!fsync(directory.fd), "directory fsync"); });
			}
			initialized = Now();
			for (int64_t done = 0; done < total; done += APPEND) {
				// Same per-append logical EOF lookup as PayloadFile::Append().
				auto offset = Size(file.fd);
				Require(offset == initial_size + done, "append position differs");
				int64_t budget = variant == "reserve256k" ? APPEND : MIB;
				if (variant != "delayed" && done % budget == 0) {
					auto length = std::min(budget, total - done);
					recorder.Run("reserve", offset, length, [&] { Reserve(file.fd, offset, length); });
					++reserves;
#ifdef VANE_ALLOCATION_TRACE
					Require(Size(file.fd) == offset, "KEEP_SIZE advanced logical EOF");
#endif
				}
				recorder.Run("append", offset, APPEND, [&] { Write(file.fd, data, APPEND, offset); });
			}
			write_end = Now();
			recorder.Run("final_sync", 0, total, [&] { Sync(file.fd); });
			sync_end = Now();
		}
		end = Now();
		Validate(path, aged, total);
		File inspect(path, O_RDONLY);
		struct stat st {};
		Require(!fstat(inspect.fd, &st), "inspect stat");
		std::cout << std::setprecision(17) << "{\"status\":\"PASS\",\"variant\":\"" << variant << "\",\"state\":\""
		          << state << "\",\"bytes\":" << total << ",\"preparation_ns\":" << preparation_end - preparation_begin
		          << ",\"initialization_ns\":" << initialized - begin << ",\"write_ns\":" << write_end - initialized
		          << ",\"final_sync_ns\":" << sync_end - write_end << ",\"close_ns\":" << end - sync_end
		          << ",\"total_ns\":" << end - begin << ",\"phase_ns\":" << sync_end - initialized
		          << ",\"logical_bytes\":" << st.st_size << ",\"allocated_bytes\":" << st.st_blocks * 512
		          << ",\"reserve_calls\":" << reserves << ",\"content_and_guard_valid\":true";
		PrintMap(inspect.fd);
		std::cout << ",\"call_columns\":[\"kind\",\"begin_ns\",\"end_ns\",\"offset\",\"bytes\"],\"calls\":[";
		for (size_t i = 0; i < recorder.calls.size(); ++i) {
			auto &call = recorder.calls[i];
			std::cout << (i ? "," : "") << "[\"" << call.kind << "\"," << call.begin - begin << ',' << call.end - begin
			          << ',' << call.offset << ',' << call.bytes << ']';
		}
		std::cout << "]}\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
