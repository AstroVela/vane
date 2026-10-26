// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Owned-file diagnostic only: this is not a SQLite VFS or a database format.
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

void timeline_install(const char *);
using Clock = std::chrono::steady_clock;
constexpr size_t MiB = 1024 * 1024;
constexpr size_t PAGE = 4096;
constexpr size_t TOTAL = 256 * MiB;

static double Seconds(Clock::time_point start) {
	return std::chrono::duration<double>(Clock::now() - start).count();
}
static void Check(bool ok) {
	if (!ok)
		throw std::runtime_error("I/O or content validation failed");
}
static void Write(int fd, const void *data, size_t count, size_t offset) {
	Check(pwrite(fd, data, count, off_t(offset)) == ssize_t(count));
}
static void Read(int fd, void *data, size_t count, size_t offset) {
	Check(pread(fd, data, count, off_t(offset)) == ssize_t(count));
}
struct File {
	int fd;
	explicit File(const std::string &path) : fd(open(path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600)) {
		Check(fd >= 0);
	}
	~File() {
		close(fd);
	}
};
struct Sample {
	double first_write, first_sync, first_copy, remaining_write, remaining_sync, remaining_copy, db_sync, seconds;
	double overlap_seconds;
};

static int Run(const std::string &directory, bool framed, bool concurrent, int pace, int group_mib) {
	Check(group_mib == 16 || group_mib == 64);
	const size_t GROUP = group_mib * MiB, FIRST = GROUP / 4;
	std::vector<char> bytes(TOTAL);
	for (size_t i = 0; i < PAGE; ++i)
		bytes[i] = char((i * 37 + 17) % 251);
	for (size_t i = PAGE; i < TOTAL; i += PAGE)
		std::memcpy(bytes.data() + i, bytes.data(), PAGE);
	for (size_t i = 0; i < TOTAL; i += PAGE) {
		uint64_t page = i / PAGE;
		std::memcpy(bytes.data() + i, &page, sizeof(page));
	}
	File wal(directory + "/wal.raw"), db(directory + "/db.raw");
	timeline_install((directory + "/io").c_str());
	std::vector<Sample> samples;
	std::vector<double> requests;
	const auto start = Clock::now();
	for (size_t group = 0; group < TOTAL; group += GROUP) {
		Sample s {};
		const auto cycle = Clock::now();
		auto write = [&](size_t begin, size_t end) {
			const auto before = Clock::now();
			if (begin == 0 && framed) {
				std::array<char, 32> header {};
				Write(wal.fd, header.data(), header.size(), 0);
			}
			for (size_t m = begin; m < end; m += MiB) {
				const auto request = Clock::now();
				if (framed) {
					for (size_t i = m; i < m + MiB; i += PAGE) {
						std::array<char, 24> header {};
						uint64_t page = (group + i) / PAGE;
						std::memcpy(header.data(), &page, sizeof(page));
						const auto offset = 32 + i / PAGE * (PAGE + header.size());
						Write(wal.fd, header.data(), header.size(), offset);
						Write(wal.fd, bytes.data() + group + i, PAGE, offset + header.size());
					}
				} else {
					Write(wal.fd, bytes.data() + group + m, MiB, m);
				}
				requests.push_back(Seconds(request));
				if (pace > 0)
					std::this_thread::sleep_until(request + std::chrono::microseconds(1000000 / pace));
			}
			return Seconds(before);
		};
		auto copy = [&](size_t begin, size_t end) {
			const auto before = Clock::now();
			const size_t step = framed ? PAGE : MiB;
			std::vector<char> buffer(step);
			for (size_t i = begin; i < end; i += step) {
				auto offset = framed ? 32 + i / PAGE * (PAGE + 24) + 24 : i;
				Read(wal.fd, buffer.data(), step, offset);
				Write(db.fd, buffer.data(), step, group + i);
			}
			return Seconds(before);
		};
		auto sync = [&](int fd) {
			auto before = Clock::now();
			Check(fsync(fd) == 0);
			return Seconds(before);
		};
		s.first_write = write(0, FIRST);
		Clock::time_point checkpoint_begin, checkpoint_end, append_begin, append_end;
		if (concurrent) {
			std::mutex mutex;
			std::condition_variable wake;
			bool entered = false;
			std::exception_ptr error;
			std::jthread checkpoint([&] {
				try {
					checkpoint_begin = Clock::now();
					{
						std::lock_guard<std::mutex> lock(mutex);
						entered = true;
					}
					wake.notify_one();
					s.first_sync = sync(wal.fd);
					s.first_copy = copy(0, FIRST);
					checkpoint_end = Clock::now();
				} catch (...) {
					error = std::current_exception();
				}
			});
			{
				std::unique_lock<std::mutex> lock(mutex);
				wake.wait(lock, [&] { return entered; });
			}
			append_begin = Clock::now();
			s.remaining_write = write(FIRST, GROUP);
			append_end = Clock::now();
			checkpoint.join();
			if (error)
				std::rethrow_exception(error);
			s.overlap_seconds = std::max(0.0, std::chrono::duration<double>(std::min(checkpoint_end, append_end) -
			                                                                std::max(checkpoint_begin, append_begin))
			                                      .count());
		} else {
			s.first_sync = sync(wal.fd);
			s.first_copy = copy(0, FIRST);
			s.remaining_write = write(FIRST, GROUP);
		}
		s.remaining_sync = sync(wal.fd);
		s.remaining_copy = copy(FIRST, GROUP);
		s.db_sync = sync(db.fd);
		s.seconds = Seconds(cycle);
		samples.push_back(s);
	}
	const double seconds = Seconds(start);
	// Validate both files after the timer, including data copied across threads.
	std::vector<char> buffer(MiB);
	for (size_t i = 0; i < TOTAL; i += MiB) {
		Read(db.fd, buffer.data(), MiB, i);
		Check(std::memcmp(buffer.data(), bytes.data() + i, MiB) == 0);
	}
	for (size_t i = 0; i < GROUP; i += PAGE) {
		auto offset = framed ? 32 + i / PAGE * (PAGE + 24) + 24 : i;
		Read(wal.fd, buffer.data(), PAGE, offset);
		Check(std::memcmp(buffer.data(), bytes.data() + TOTAL - GROUP + i, PAGE) == 0);
	}
	std::cout.precision(12);
	std::cout << "{\"status\":\"PASS\",\"logical_bytes\":" << TOTAL << ",\"framed\":" << framed
	          << ",\"group_mib\":" << group_mib << ",\"concurrent\":" << concurrent << ",\"pace_mib_s\":" << pace
	          << ",\"seconds\":" << seconds << ",\"groups\":[";
	for (size_t i = 0; i < samples.size(); ++i) {
		if (i)
			std::cout << ',';
		auto &s = samples[i];
		std::cout << "{\"seconds\":" << s.seconds << ",\"first_write\":" << s.first_write
		          << ",\"first_sync\":" << s.first_sync << ",\"first_copy\":" << s.first_copy
		          << ",\"remaining_write\":" << s.remaining_write << ",\"remaining_sync\":" << s.remaining_sync
		          << ",\"remaining_copy\":" << s.remaining_copy << ",\"db_sync\":" << s.db_sync
		          << ",\"overlap_seconds\":" << s.overlap_seconds << '}';
	}
	std::cout << "],\"write_requests\":[";
	for (size_t i = 0; i < requests.size(); ++i) {
		if (i)
			std::cout << ',';
		std::cout << requests[i];
	}
	std::cout << "],\"full_content_validation\":true}\n";
	return 0;
}
int main(int argc, char **argv) {
	try {
		if (argc != 5 && argc != 6)
			throw std::runtime_error("usage: io-probe DIRECTORY FRAMED CONCURRENT PACE_MIB_S [GROUP_MIB=64]");
		return Run(argv[1], std::stoi(argv[2]) != 0, std::stoi(argv[3]) != 0, std::stoi(argv[4]),
		           argc == 6 ? std::stoi(argv[5]) : 64);
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return 1;
	}
}
