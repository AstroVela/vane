// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../checkpoint_progress/fixture.hpp"
#include "wait_trace.hpp"
#include <ctime>

namespace fuse_waits {
bool PayloadWorkerIdle();
}

static sqlite3 *writer = nullptr;
static Checkpointer *worker = nullptr;
using WalHook = int (*)(void *, sqlite3 *, const char *, int);
extern "C" void *__real_sqlite3_wal_hook(sqlite3 *, WalHook, void *);
extern "C" void *__wrap_sqlite3_wal_hook(sqlite3 *db, WalHook hook, void *context) {
	if (hook == Checkpointer::OnCommit) {
		Require(writer == nullptr, "Expected one workspace writer");
		writer = db;
		worker = static_cast<Checkpointer *>(context);
	}
	if (!hook && db == writer) {
		writer = nullptr;
		worker = nullptr;
	}
	return __real_sqlite3_wal_hook(db, hook, context);
}

static int64_t Now() {
	return fuse_waits::Now(CLOCK_MONOTONIC);
}
template <class F>
static void Await(F condition) {
	auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
	while (!condition()) {
		Require(std::chrono::steady_clock::now() < deadline, "Timed out waiting for diagnostic progress");
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}
static std::pair<int, int> Status() {
	int frames = -1, copied = -1;
	auto rc = sqlite3_wal_checkpoint_v2(writer, "main", SQLITE_CHECKPOINT_NOOP, &frames, &copied);
	if (rc == SQLITE_BUSY)
		return {-1, -1};
	Require(rc == SQLITE_OK && frames >= copied && copied >= 0, "Invalid WAL status");
	return {frames, copied};
}
static bool CaughtUp() {
	auto status = Status();
	return status.first >= 0 && status.first == status.second;
}
static void Warm() {
	Require(sqlite3_exec(writer,
	                     "PRAGMA synchronous=FULL; CREATE TABLE diagnostic_padding(data BLOB);"
	                     "INSERT INTO diagnostic_padding VALUES(zeroblob(20971520)); PRAGMA synchronous=NORMAL",
	                     nullptr, nullptr, nullptr) == SQLITE_OK,
	        "Warm committed CLEAN WAL");
	Await(CaughtUp);
	worker->Stop();
	worker->CheckError();
}
