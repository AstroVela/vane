// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "wait_trace.hpp"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <thread>

int main(int argc, char **argv) {
	if (argc != 2)
		return 2;
	fuse_waits::Install(argv[1]);
	std::mutex mutex;
	std::atomic<bool> ready {false};
	std::thread holder([&] {
		fuse_waits::Root root("gate_holder");
		std::lock_guard<std::mutex> lock(mutex);
		fuse_waits::Span held(fuse_waits::BeginRestartHold);
		ready = true;
		std::this_thread::sleep_for(std::chrono::milliseconds(80));
	});
	while (!ready)
		std::this_thread::yield();
	{
		fuse_waits::Root root("gate_waiter");
		errno = EDOM;
		auto lock = fuse_waits::Lock(fuse_waits::BeginRestartWait, mutex);
		if (!lock.owns_lock() || errno != EDOM)
			std::abort();
	}
	holder.join();
	{
		fuse_waits::Root root("gate_unwind");
		try {
			auto lock = fuse_waits::Lock(fuse_waits::BeginRestartWait, mutex);
			fuse_waits::Span held(fuse_waits::BeginRestartHold);
			throw std::runtime_error("unwind");
		} catch (const std::runtime_error &) {
		}
		if (!mutex.try_lock())
			return 3;
		mutex.unlock();
	}
	return 0;
}
