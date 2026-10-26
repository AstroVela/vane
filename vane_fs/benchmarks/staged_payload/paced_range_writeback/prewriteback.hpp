// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "payload_file.hpp"
#include <atomic>
#include <condition_variable>
#include <mutex>

namespace vane_fs {
// Optional finite data submission. Neither completion nor a FULL pause makes
// this worker a durability authority; publication retains its own barriers.
class Prewriteback {
public:
	static constexpr uint64_t START_BYTES = 64 * 1024 * 1024;
	static constexpr int64_t BATCH_BYTES = 4 * 1024 * 1024;
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
		file.reset();
	}
	void CheckError() {
		if (error.exchange(false))
			throw Error(ErrorCode::Storage, "Background payload prewriteback failed");
	}
	// Stop new claims, without waiting for an active or already claimed call.
	// A nested pause remains effective until its last owner releases it.
	class Pause {
	public:
		explicit Pause(Prewriteback *worker, bool enabled = true) : worker(enabled ? worker : nullptr) {
			if (this->worker) {
				std::lock_guard<std::mutex> lock(worker->mutex);
				++worker->pauses;
				worker->wake.notify_all();
			}
		}
		~Pause() {
			if (worker) {
				std::lock_guard<std::mutex> lock(worker->mutex);
				--worker->pauses;
				worker->wake.notify_all();
			}
		}
		Pause(const Pause &) = delete;
		Pause &operator=(const Pause &) = delete;

	private:
		Prewriteback *worker;
	};
	void Retire() noexcept {
		OnCommit(0, true, {});
	}
	void OnCommit(uint64_t appended, bool full, std::pair<int64_t, int64_t> range) noexcept {
		try {
			bool notify;
			{
				std::lock_guard<std::mutex> lock(mutex);
				requested = std::max(requested, appended);
				if (full) {
					completed = requested;
					pending = {};
					draining = false;
				} else {
					Merge(range);
				}
				notify = full || draining || requested - completed >= START_BYTES;
			}
			if (notify)
				wake.notify_one();
		} catch (...) {
			error.store(true);
		}
	}

private:
	void Merge(std::pair<int64_t, int64_t> range) {
		if (range.second <= range.first)
			return;
		pending.first = pending.second ? std::min(pending.first, range.first) : range.first;
		pending.second = std::max(pending.second, range.second);
	}
	void Run() {
		std::unique_lock<std::mutex> lock(mutex);
		for (;;) {
			wake.wait(lock,
			          [this] { return stopped || (!pauses && (draining || requested - completed >= START_BYTES)); });
			if (stopped)
				return;
			if (std::chrono::steady_clock::now() < next) {
				wake.wait_until(lock, next, [this] {
					return stopped || pauses || (!draining && requested - completed < START_BYTES);
				});
				continue;
			}
			draining = true;
			auto target = requested;
			lock.unlock();
			bool success = false, last = false, skipped = false;
			std::pair<int64_t, int64_t> range {};
			try {
				// Independent SH lease allows appends and excludes GC. Recheck
				// pause/retirement after waiting, before claiming any queued bytes.
				PayloadFile::Guard lease(*file, false);
				{
					std::lock_guard<std::mutex> state(mutex);
					if (stopped)
						return;
					if (pauses || completed >= target) {
						skipped = true;
					} else {
						target = requested;
						auto end = std::min(pending.second, file->WritebackSize());
						auto begin = std::min(pending.first, end);
						range = {begin, begin + std::min(end - begin, BATCH_BYTES)};
						pending =
						    range.second < end ? std::make_pair(range.second, end) : std::pair<int64_t, int64_t> {};
						last = !pending.second;
					}
				}
				if (!skipped) {
					file->WritebackRange(range);
					success = true;
				}
			} catch (const Error &failure) {
				if (failure.code != ErrorCode::Busy)
					error.store(true);
			} catch (...) {
				error.store(true);
			}
			lock.lock();
			if (skipped)
				continue;
			// Keep the completion-to-next-start deadline across Stop/Start.
			next = std::chrono::steady_clock::now() + std::chrono::milliseconds(16);
			if (success) {
				if (last)
					completed = std::max(completed, target);
				if (!pending.second)
					draining = false;
			} else {
				if (completed < target)
					Merge(range);
				wake.wait_for(lock, std::chrono::milliseconds(20), [this] { return stopped || pauses; });
			}
		}
	}
	std::unique_ptr<PayloadFile> file;
	std::mutex mutex;
	std::condition_variable wake;
	std::thread worker;
	std::atomic<bool> error {false};
	uint64_t requested = 0, completed = 0;
	std::pair<int64_t, int64_t> pending {};
	std::chrono::steady_clock::time_point next {};
	unsigned pauses = 0;
	bool stopped = false, draining = false;
};
} // namespace vane_fs
