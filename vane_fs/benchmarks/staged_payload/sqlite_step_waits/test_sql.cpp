// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "wait_trace.hpp"
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <sqlite3.h>

static void Require(bool condition) {
	if (!condition)
		std::abort();
}
static void Exec(sqlite3 *db, const char *text) {
	Require(sqlite3_exec(db, text, nullptr, nullptr, nullptr) == SQLITE_OK);
}
static void Pause(long nanoseconds) {
	timespec delay {0, nanoseconds};
	while (nanosleep(&delay, &delay) && errno == EINTR) {
	}
}
static void CustomSleep(sqlite3_context *context, int, sqlite3_value **) {
	Pause(40000000);
	sqlite3_result_int(context, 42);
}
static volatile sig_atomic_t interrupted = 0;
static void Signal(int) {
	interrupted = 1;
}

int main(int argc, char **argv) {
	if (argc < 2)
		return 2;
	fuse_waits::Install(argv[1]);
	if (argc == 3 && std::string(argv[2]) == "--observe") {
		std::puts("ready");
		std::fflush(stdout);
		fuse_waits::Root root("sampler_test");
		Pause(200000000);
		return 0;
	}
	sqlite3 *holder = nullptr, *contender = nullptr;
	std::string path = std::string(argv[1]) + ".sqlite";
	Require(sqlite3_open(path.c_str(), &holder) == SQLITE_OK);
	Exec(holder, "PRAGMA journal_mode=WAL; CREATE TABLE trial(value)");
	Require(sqlite3_open(path.c_str(), &contender) == SQLITE_OK);
	Require(sqlite3_busy_timeout(contender, 1000) == SQLITE_OK);
	Exec(holder, "BEGIN IMMEDIATE");
	std::atomic<bool> entered {false};
	std::thread waiter([&] {
		fuse_waits::Root root("busy_test");
		sqlite3_stmt *statement = nullptr;
		Require(sqlite3_prepare_v2(contender, "INSERT INTO trial VALUES(1)", -1, &statement, nullptr) == SQLITE_OK);
		entered = true;
		Require(sqlite3_step(statement) == SQLITE_DONE);
		Require(sqlite3_finalize(statement) == SQLITE_OK);
	});
	while (!entered)
		std::this_thread::yield();
	Pause(60000000);
	Exec(holder, "ROLLBACK");
	waiter.join();
	{
		fuse_waits::Root root("busy_timeout_test");
		Exec(holder, "BEGIN IMMEDIATE");
		Require(sqlite3_busy_timeout(contender, 3) == SQLITE_OK);
		sqlite3_stmt *statement = nullptr;
		Require(sqlite3_prepare_v2(contender, "INSERT INTO trial VALUES(2)", -1, &statement, nullptr) == SQLITE_OK);
		Require(sqlite3_step(statement) == SQLITE_BUSY);
		Require(sqlite3_finalize(statement) == SQLITE_BUSY);
		Exec(holder, "ROLLBACK");
	}
	{
		fuse_waits::Root root("custom_sleep_test");
		Require(sqlite3_create_function(contender, "diagnostic_sleep", 0, SQLITE_UTF8, nullptr, CustomSleep, nullptr,
		                                nullptr) == SQLITE_OK);
		sqlite3_stmt *statement = nullptr;
		Require(sqlite3_prepare_v2(contender, "SELECT diagnostic_sleep(), '\n\"\\\t'", -1, &statement, nullptr) ==
		        SQLITE_OK);
		Require(sqlite3_step(statement) == SQLITE_ROW && sqlite3_column_int(statement, 0) == 42);
		Require(sqlite3_finalize(statement) == SQLITE_OK);
	}
	{
		fuse_waits::Root root("interrupt_test");
		struct sigaction action {};
		action.sa_handler = Signal;
		sigemptyset(&action.sa_mask);
		Require(sigaction(SIGALRM, &action, nullptr) == 0);
		ualarm(10000, 0);
		timespec request {0, 200000000}, remaining {};
		errno = EDOM;
		Require(nanosleep(&request, &remaining) == -1 && errno == EINTR);
		Require(interrupted && remaining.tv_sec == 0 && remaining.tv_nsec > 0 && remaining.tv_nsec < request.tv_nsec);
		ualarm(0, 0);
	}
	{
		fuse_waits::Root root("invalid_sleep_test");
		timespec invalid {0, 1000000000};
		errno = EDOM;
		Require(nanosleep(&invalid, nullptr) == -1 && errno == EINVAL);
	}
	Require(sqlite3_close(contender) == SQLITE_OK && sqlite3_close(holder) == SQLITE_OK);
	std::puts("PASS default busy handler, timeout, non-busy SQL sleep, SQL escaping, EINTR and EINVAL");
	return 0;
}
