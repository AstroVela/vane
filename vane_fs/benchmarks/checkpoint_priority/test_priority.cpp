// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Scheduling probe: prepared header copies add two observation hooks. SQLite
// checkpoints are mocked here; the component suites separately test real I/O.
#include "checkpoint.hpp"
#include <condition_variable>
#include <filesystem>
#include <iostream>
#include <string>
#include <unistd.h>

static std::mutex state_mutex;
static std::condition_variable changed;
static bool first_entered = false, release_first = false, announced = false, release_writer = false;
static bool yielded = false, restarted = false, second_early = false, busy = false;
static int passive_calls = 0;

static void Require(bool value, const char *message) {
	if (!value)
		throw std::runtime_error(message);
}
template <class F>
static void Wait(std::unique_lock<std::mutex> &lock, F predicate) {
	Require(changed.wait_for(lock, std::chrono::seconds(5), predicate), "Scheduling probe timed out");
}

void checkpoint_priority_before_lock() {
	std::unique_lock<std::mutex> lock(state_mutex);
	announced = true;
	changed.notify_all();
	Wait(lock, [] { return release_writer; });
}
void checkpoint_priority_yield() {
	std::lock_guard<std::mutex> lock(state_mutex);
	yielded = true;
	changed.notify_all();
}

extern "C" int __wrap_sqlite3_wal_checkpoint_v2(sqlite3 *, const char *, int mode, int *frames, int *copied) {
	std::unique_lock<std::mutex> lock(state_mutex);
	*frames = *copied = 17000; // Exceed the unchanged 64 MiB admission limit.
	if (mode == SQLITE_CHECKPOINT_PASSIVE) {
		if (++passive_calls == 1) {
			first_entered = true;
			changed.notify_all();
			Wait(lock, [] { return release_first; });
		} else {
			second_early = second_early || !restarted;
			changed.notify_all();
		}
	} else if (mode == SQLITE_CHECKPOINT_RESTART) {
		restarted = true;
		changed.notify_all();
		if (busy)
			return SQLITE_BUSY;
		*frames = *copied = 0;
	}
	return SQLITE_OK;
}

int main(int argc, char **argv) {
	std::string scenario = argc > 1 ? argv[1] : "success";
	busy = scenario == "busy";
	auto path = std::filesystem::temp_directory_path() / ("vane-priority-probe-" + std::to_string(getpid()));
	sqlite3 *writer = nullptr;
	std::unique_ptr<vane_fs::Checkpointer> checkpoint;
	std::thread foreground;
	std::exception_ptr writer_error;
	bool writer_busy = false;
	int result = 1;
	try {
		std::filesystem::create_directory(path);
		Require(sqlite3_open((path / "db.sqlite").c_str(), &writer) == SQLITE_OK, "Opening probe database");
		Require(sqlite3_exec(writer, "PRAGMA journal_mode=WAL; CREATE TABLE fixture(value)", nullptr, nullptr,
		                     nullptr) == SQLITE_OK,
		        "Initializing probe database");
		checkpoint = std::make_unique<vane_fs::Checkpointer>(writer, 4096, 0);
		vane_fs::Checkpointer::OnCommit(checkpoint.get(), writer, "main", 17000);
		{
			std::unique_lock<std::mutex> lock(state_mutex);
			Wait(lock, [] { return first_entered; });
		}
		// A commit arriving during the first checkpoint requests another pass.
		vane_fs::Checkpointer::OnCommit(checkpoint.get(), writer, "main", 17000);
		foreground = std::thread([&] {
			try {
				checkpoint->BeforeWrite(writer);
			} catch (const vane_fs::Error &error) {
				if (busy && error.code == vane_fs::ErrorCode::Busy)
					writer_busy = true;
				else
					writer_error = std::current_exception();
			} catch (...) {
				writer_error = std::current_exception();
			}
		});
		bool priority_observed;
		{
			std::unique_lock<std::mutex> lock(state_mutex);
			Wait(lock, [] { return announced; });
			release_first = true;
			changed.notify_all();
			Wait(lock, [] { return yielded || passive_calls >= 2; });
			priority_observed = yielded && passive_calls == 1;
		}
		if (scenario == "restart")
			checkpoint->Stop();
		{
			std::lock_guard<std::mutex> lock(state_mutex);
			release_writer = true;
			changed.notify_all();
		}
		foreground.join();
		if (writer_error)
			std::rethrow_exception(writer_error);
		Require(writer_busy == busy, "Unexpected admission error");
		if (scenario == "restart")
			checkpoint->Start();
		{
			std::unique_lock<std::mutex> lock(state_mutex);
			Wait(lock, [] { return passive_calls >= 2; });
			Require(priority_observed && !second_early && restarted,
			        "Background checkpoint preceded pending foreground admission");
		}
		result = 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
	}
	{
		std::lock_guard<std::mutex> lock(state_mutex);
		release_first = release_writer = true;
		changed.notify_all();
	}
	if (foreground.joinable())
		foreground.join();
	checkpoint.reset();
	if (writer)
		sqlite3_close_v2(writer);
	std::filesystem::remove_all(path);
	std::cout << "{\"scenario\":\"" << scenario << "\",\"passed\":" << (result == 0 ? "true" : "false")
	          << ",\"yielded\":" << (yielded ? "true" : "false")
	          << ",\"second_early\":" << (second_early ? "true" : "false")
	          << ",\"restarted\":" << (restarted ? "true" : "false") << ",\"busy\":" << (writer_busy ? "true" : "false")
	          << "}\n";
	return result;
}
