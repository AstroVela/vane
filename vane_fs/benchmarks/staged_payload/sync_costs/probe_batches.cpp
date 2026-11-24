// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Raw-file hypothesis probe, not VaneFS or its publication protocol.
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <vector>
#include <linux/falloc.h>
#include <unistd.h>

namespace {
constexpr int64_t MIB = 1024 * 1024;
using Clock = std::chrono::steady_clock;
int64_t Now() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
int64_t Epoch() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
	    .count();
}
void Require(bool condition) {
	if (!condition) {
		std::perror("raw batch probe");
		std::exit(1);
	}
}
int Open(const std::string &path) {
	auto fd = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
	Require(fd >= 0);
	return fd;
}
void Write(int fd, const char *data, int64_t size, int64_t offset) {
	while (size) {
		auto result = pwrite(fd, data, size_t(size), offset);
		if (result < 0 && errno == EINTR)
			continue;
		Require(result > 0);
		data += result;
		offset += result;
		size -= result;
	}
}
void Sync(int fd) {
	int result;
	do {
		result = fdatasync(fd);
	} while (result && errno == EINTR);
	Require(!result);
}
struct Call {
	const char *kind;
	int part;
	int64_t begin, end, dirty, file_size;
};
} // namespace
int main(int argc, char **argv) {
	Require(argc == 4);
	std::string root = argv[1], mode = argv[2], state = argv[3];
	Require(mode == "grow128" || mode == "grow16" || mode == "sealed16");
	Require(state == "fresh" || state == "after_reclamation");
	bool sealed = mode == "sealed16", aged = state == "after_reclamation";
	int64_t budget = (mode == "grow128" ? 128 : 16) * MIB;
	void *allocation = nullptr;
	Require(!posix_memalign(&allocation, 4096, size_t(8 * MIB + 4096)));
	auto buffer = static_cast<char *>(allocation) + 48;
	std::memset(buffer, 0x5a, size_t(8 * MIB));
	int directory = open(root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
	Require(directory >= 0);
	if (aged) {
		if (sealed) {
			for (int i = 0; i < 12; ++i) {
				auto path = root + "/old-" + std::to_string(i);
				int fd = Open(path);
				Write(fd, buffer, 8 * MIB, 0);
				Write(fd, buffer, 8 * MIB, 8 * MIB);
				Sync(fd);
				Require(!close(fd));
				if (i % 2)
					Require(!unlink(path.c_str()));
			}
		} else {
			int fd = Open(root + "/payload");
			for (int i = 0; i < 24; ++i)
				Write(fd, buffer, 8 * MIB, i * 8 * MIB);
			Sync(fd);
			for (int i = 1; i < 12; i += 2)
				Require(!fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, i * 16 * MIB, 16 * MIB));
			Sync(fd);
			Require(!close(fd));
		}
		Require(!fsync(directory));
	}
	std::memset(buffer, 0x37, size_t(8 * MIB));
	std::vector<Call> calls;
	calls.reserve(128);
	int64_t phase_begin[2], phase_end[2], phase_epoch[2], phase_epoch_end[2];
	int64_t written = 0, dirty = 0, offset = sealed || !aged ? 0 : 192 * MIB;
	int part = 0, fd = -1;
	auto complete_epoch = Epoch(), complete_begin = Now();
	for (int phase = 0; phase < 2; ++phase) {
		phase_epoch[phase] = Epoch();
		phase_begin[phase] = Now();
		int64_t total = (phase == 0 ? 256 : 64) * MIB, chunk = (phase == 0 ? 8 : 1) * MIB;
		for (int64_t done = 0; done < total; done += chunk) {
			if (fd < 0)
				fd = Open(root + (sealed ? "/part-" + std::to_string(part) : "/payload"));
			auto begin = Now();
			Write(fd, buffer, chunk, offset);
			calls.push_back({"write", part, begin, Now(), chunk, offset + chunk});
			offset += chunk;
			written += chunk;
			dirty += chunk;
			if (dirty >= budget || done + chunk == total) {
				begin = Now();
				Sync(fd);
				calls.push_back({"payload_sync", part, begin, Now(), dirty, offset});
				dirty = 0;
				if (sealed) {
					Require(!close(fd));
					fd = -1;
					++part;
					offset = 0;
				}
			}
		}
		phase_end[phase] = Now();
		phase_epoch_end[phase] = Epoch();
	}
	if (fd >= 0)
		Require(!close(fd));
	int marker = Open(root + "/marker");
	Write(marker, buffer, 4096, 0);
	auto begin = Now();
	Sync(marker);
	calls.push_back({"small_sync", -1, begin, Now(), 4096, 4096});
	Require(!close(marker));
	begin = Now();
	Require(!fsync(directory));
	calls.push_back({"directory_sync", -1, begin, Now(), 0, 0});
	Require(!close(directory));
	auto complete_end = Now(), complete_epoch_end = Epoch();
	Require(written == 320 * MIB && !dirty);
	std::free(allocation);
	std::cout << "{\"status\":\"PASS\",\"bytes\":" << written << ",\"begin_epoch_ns\":" << complete_epoch
	          << ",\"end_epoch_ns\":" << complete_epoch_end << ",\"complete_ns\":" << complete_end - complete_begin
	          << ",\"phases\":[";
	for (int i = 0; i < 2; ++i) {
		if (i)
			std::cout << ',';
		std::cout << "{\"workload\":\"" << (i == 0 ? "stream_256" : "write_64")
		          << "\",\"duration_ns\":" << phase_end[i] - phase_begin[i] << ",\"begin_epoch_ns\":" << phase_epoch[i]
		          << ",\"end_epoch_ns\":" << phase_epoch_end[i] << '}';
	}
	std::cout
	    << "],\"call_columns\":[\"kind\",\"part\",\"begin_ns\",\"end_ns\",\"dirty_bytes\",\"file_size\"],\"calls\":[";
	bool first = true;
	for (auto &call : calls) {
		if (!first)
			std::cout << ',';
		first = false;
		std::cout << "[\"" << call.kind << "\"," << call.part << ',' << call.begin - complete_begin << ','
		          << call.end - complete_begin << ',' << call.dirty << ',' << call.file_size << ']';
	}
	std::cout << "]}\n";
}
