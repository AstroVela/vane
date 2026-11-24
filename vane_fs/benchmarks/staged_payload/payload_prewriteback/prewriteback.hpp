// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "payload_file.hpp"
#include <atomic>
#include <condition_variable>
#include <mutex>

namespace vane_fs {
// Advisory data writeback only. This worker owns no SQLite connection and
// never advances publication, WAL checkpoint state, or a durability promise.
class Prewriteback {
public:
	static constexpr uint64_t START_BYTES = 64 * 1024 * 1024;
	Prewriteback(const std::string &database, int timeout)
	    : file(std::make_unique<PayloadFile>(database, false, timeout)) {
		Start();
	}
	~Prewriteback() {
		Stop();
	}
	void Start() {
		std::lock_guard<std::mutex> lock(mutex);
		stopped = false;
		worker = std::thread([this] { Run(); });
	}
	void Stop() {
		{
			std::lock_guard<std::mutex> lock(mutex);
			stopped = true;
		}
		wake.notify_all();
		if (worker.joinable())
			worker.join();
	}
	void AbandonInherited() {
		// Close the child's duplicate, without unlocking the parent's lease or
		// touching a mutex/thread object inherited from a vanished thread.
		file.reset();
	}
	void CheckError() {
		if (error.exchange(false))
			throw Error(ErrorCode::Storage, "Background payload prewriteback failed");
	}
	// The caller holds its Database mutex. Notify only after successful SQL
	// commit; an authoritative FULL barrier may also retire earlier requests.
	// Neither path may report failure after the commit has already succeeded.
	void OnCommit(uint64_t appended, bool full) noexcept {
		try {
			bool notify;
			{
				std::lock_guard<std::mutex> lock(mutex);
				requested = std::max(requested, appended);
				if (full)
					completed = std::max(completed, appended);
				notify = full || requested - completed >= START_BYTES;
			}
			if (notify)
				wake.notify_one();
		} catch (...) {
			error.store(true);
		}
	}

private:
	void Run() {
		std::unique_lock<std::mutex> lock(mutex);
		while (true) {
			wake.wait(lock, [this] { return stopped || requested - completed >= START_BYTES; });
			if (stopped)
				return;
			auto target = requested;
			lock.unlock();
			bool success = false;
			try {
				// This separate open-file description must not share flock state
				// with writer transactions. SH permits appends and excludes GC.
				PayloadFile::Guard lease(*file, false);
				{
					std::lock_guard<std::mutex> state(mutex);
					// GC or another FULL barrier may have retired this request
					// while the worker waited for the payload lease.
					if (stopped)
						return;
					if (completed >= target) {
						success = true;
					} else {
						target = requested;
					}
				}
				if (!success) {
					file->Flush(true);
					success = true;
				}
			} catch (const Error &failure) {
				if (failure.code != ErrorCode::Busy)
					error.store(true);
			} catch (...) {
				error.store(true);
			}
			lock.lock();
			if (success)
				completed = std::max(completed, target);
			else
				wake.wait_for(lock, std::chrono::milliseconds(20), [this] { return stopped; });
		}
	}

	std::unique_ptr<PayloadFile> file;
	std::mutex mutex;
	std::condition_variable wake;
	std::thread worker;
	std::atomic<bool> error {false};
	uint64_t requested = 0, completed = 0;
	bool stopped = false;
};
} // namespace vane_fs
