// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <thread>

namespace vane_fs {

// Only fsync-mode connections use this worker. The writer still commits each
// operation, and FULL barriers remain on that connection. Checkpoint errors
// are reported before a subsequent mutation/barrier, never from the WAL hook:
// returning an error from that hook would report failure AFTER committing.
class Checkpointer {
public:
	static constexpr int64_t START_BYTES = 16 * 1024 * 1024;
	static constexpr int64_t LIMIT_BYTES = 64 * 1024 * 1024;

	Checkpointer(sqlite3 *writer, int page_size, int timeout_ms)
	    : page_size(page_size), timeout_ms(timeout_ms),
	      wal_path(std::string(sqlite3_db_filename(writer, "main")) + "-wal") {
		try {
			Check(sqlite3_open_v2(sqlite3_db_filename(writer, "main"), &db,
			                      SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX | SQLITE_OPEN_PRIVATECACHE, nullptr));
			Check(sqlite3_exec(db, "PRAGMA synchronous=NORMAL; SELECT count(*) FROM sqlite_master", nullptr, nullptr,
			                   nullptr));
			Start();
		} catch (...) {
			if (db)
				sqlite3_close_v2(db);
			throw;
		}
	}
	~Checkpointer() {
		Stop();
		sqlite3_close_v2(db);
	}
	void Start() {
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
	static int OnCommit(void *context, sqlite3 *, const char *, int frames) {
		auto &self = *static_cast<Checkpointer *>(context);
		if (self.Bytes(frames) >= START_BYTES) {
			{
				std::lock_guard<std::mutex> lock(self.mutex);
				self.requested = true;
			}
			self.wake.notify_one();
		}
		return SQLITE_OK;
	}
	void CheckError() {
		auto code = error.exchange(SQLITE_OK);
		if (code != SQLITE_OK)
			throw Error(ErrorCode::Storage, std::string("Background WAL checkpoint: ") + sqlite3_errstr(code));
	}
	void BeforeWrite(sqlite3 *writer) {
		CheckError();
		int frames = -1, copied = -1;
		auto code = sqlite3_wal_checkpoint_v2(writer, "main", SQLITE_CHECKPOINT_NOOP, &frames, &copied);
		if (code == SQLITE_BUSY) {
			// A running checkpoint owns the status lock. File length is only a
			// conservative upper bound here, not our usual progress measure.
			std::error_code failure;
			auto bytes = std::filesystem::file_size(wal_path, failure);
			if (!failure && bytes < LIMIT_BYTES)
				return;
		} else {
			Check(code);
			if (Bytes(frames) < LIMIT_BYTES)
				return;
		}
		// Stop admitting writes before opening their SQL transaction. Waiting
		// with a writer lock would prevent the worker from reclaiming the WAL.
		// One atomic operation can exceed this budget; it is not a file quota.
		// Our worker performs real I/O here, just as an inline checkpoint did.
		// Busy timeouts bound SQLite lock contention, not disk write/sync time.
		// Turning a slow local sync into Busy can cause otherwise valid FUSE
		// writes to fail halfway through their requests.
		std::lock_guard<std::mutex> checkpoint_lock(checkpoint_mutex);
		auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
		for (;;) {
			code = sqlite3_wal_checkpoint_v2(writer, "main", SQLITE_CHECKPOINT_NOOP, &frames, &copied);
			if (code == SQLITE_OK && Bytes(frames) < LIMIT_BYTES)
				return;
			if (code != SQLITE_BUSY) {
				Check(code);
				auto remaining =
				    std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
				Check(sqlite3_busy_timeout(writer, int(std::max<int64_t>(0, remaining.count()))));
				code = sqlite3_wal_checkpoint_v2(writer, "main", SQLITE_CHECKPOINT_RESTART, &frames, &copied);
				sqlite3_busy_timeout(writer, timeout_ms);
				if (code != SQLITE_BUSY) {
					Check(code);
					return;
				}
			}
			if (std::chrono::steady_clock::now() >= deadline)
				Check(SQLITE_BUSY);
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}

private:
	static void Check(int code) {
		if (code != SQLITE_OK)
			throw Error((code & 0xff) == SQLITE_BUSY ? ErrorCode::Busy : ErrorCode::Storage,
			            std::string("WAL checkpoint: ") + sqlite3_errstr(code));
	}
	int64_t Bytes(int frames) const {
		return frames < 0 ? 0 : int64_t(frames) * (page_size + 24) + 32;
	}
	void Run() {
		bool retry = false;
		std::unique_lock<std::mutex> lock(mutex);
		for (;;) {
			if (retry)
				wake.wait_for(lock, std::chrono::milliseconds(20), [this] { return stopped; });
			else
				wake.wait(lock, [this] { return stopped || requested; });
			if (stopped) {
				// A failed explicit close may restart this worker without any
				// further commits. Preserve unfinished maintenance across Stop.
				requested = requested || retry;
				return;
			}
			requested = false;
			lock.unlock();
			int frames = -1, copied = -1, code;
			{
				std::lock_guard<std::mutex> checkpoint_lock(checkpoint_mutex);
				code = sqlite3_wal_checkpoint_v2(db, "main", SQLITE_CHECKPOINT_PASSIVE, &frames, &copied);
			}
			if (code != SQLITE_OK && code != SQLITE_BUSY) {
				int expected = SQLITE_OK;
				error.compare_exchange_strong(expected, code);
			}
			retry = code != SQLITE_OK || frames > copied;
			lock.lock();
		}
	}

	sqlite3 *db = nullptr;
	const int page_size, timeout_ms;
	const std::string wal_path;
	std::atomic<int> error {SQLITE_OK};
	std::mutex mutex;
	std::mutex checkpoint_mutex;
	std::condition_variable wake;
	bool stopped = false, requested = false;
	std::thread worker;
};
} // namespace vane_fs
