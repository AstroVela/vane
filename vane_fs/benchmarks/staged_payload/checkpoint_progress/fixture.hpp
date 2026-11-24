// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "checkpoint.hpp"
#include "vane_fs/workspace.hpp"
#include <iostream>

using namespace vane_fs;

inline void Require(bool condition, const char *message) {
	if (!condition)
		throw std::runtime_error(message);
}
template <class F>
void Wait(F condition, const char *message) {
	auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
	while (!condition()) {
		Require(std::chrono::steady_clock::now() < deadline, message);
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}
struct SQL {
	sqlite3 *db = nullptr;
	explicit SQL(const std::string &path) {
		Require(sqlite3_open(path.c_str(), &db) == SQLITE_OK, "Open SQL connection");
		sqlite3_busy_timeout(db, 5000);
		Staging::Configure(db);
		Exec("SELECT count(*) FROM sqlite_master");
	}
	~SQL() {
		sqlite3_close_v2(db);
	}
	void Exec(const std::string &sql) {
		auto code = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, nullptr);
		Require(code == SQLITE_OK, sqlite3_errmsg(db));
	}
	std::string Text(const std::string &sql) {
		sqlite3_stmt *raw = nullptr;
		Require(sqlite3_prepare_v2(db, sql.c_str(), -1, &raw, nullptr) == SQLITE_OK, "Prepare inspection");
		std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> statement(raw, sqlite3_finalize);
		Require(sqlite3_step(raw) == SQLITE_ROW, "Missing inspection row");
		auto *text = sqlite3_column_text(raw, 0);
		Require(text != nullptr, "Null inspection result");
		return reinterpret_cast<const char *>(text);
	}
	std::pair<int, int> Status() {
		int frames = -1, copied = -1;
		auto code = sqlite3_wal_checkpoint_v2(db, "main", SQLITE_CHECKPOINT_NOOP, &frames, &copied);
		if (code == SQLITE_BUSY)
			return {-1, -1};
		Require(code == SQLITE_OK && frames >= 0 && copied >= 0, "Missing initialized WAL status");
		return {frames, copied};
	}
	bool CaughtUp() {
		auto status = Status();
		return status.first >= 0 && status.first == status.second;
	}
	void Checkpoint(int mode) {
		int frames = -1, copied = -1;
		Require(sqlite3_wal_checkpoint_v2(db, "main", mode, &frames, &copied) == SQLITE_OK,
		        "Foreground fixture checkpoint failed");
		Require(frames >= 0 && frames == copied, "Incomplete fixture checkpoint");
	}
};
