// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0
#include <sqlite3.h>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <random>
static void ck(int c) {
	if (c != SQLITE_OK && c != SQLITE_DONE && c != SQLITE_ROW)
		throw std::runtime_error(sqlite3_errstr(c));
}
static void ex(sqlite3 *d, const char *s) {
	ck(sqlite3_exec(d, s, nullptr, nullptr, nullptr));
}
static sqlite3_stmt *prep(sqlite3 *d, const char *s) {
	sqlite3_stmt *p = nullptr;
	ck(sqlite3_prepare_v2(d, s, -1, &p, nullptr));
	return p;
}
int main() {
	try {
		std::cout << "{\"samples\":[";
		bool comma = false;
		for (int history : {1, 16}) {
			sqlite3 *d = nullptr;
			ck(sqlite3_open(":memory:", &d));
			ex(d, "PRAGMA cache_size=-2048; CREATE TABLE block_payloads(id INTEGER PRIMARY KEY,data BLOB NOT NULL) "
			      "STRICT; CREATE TABLE block_versions(inode INTEGER NOT NULL,block INTEGER NOT NULL,low BLOB NOT "
			      "NULL,high BLOB NOT NULL,writer TEXT NOT NULL,deleted INTEGER NOT NULL,payload INTEGER,PRIMARY "
			      "KEY(inode,block,low)) STRICT,WITHOUT ROWID; CREATE INDEX block_versions_high ON "
			      "block_versions(inode,block,high); BEGIN IMMEDIATE;");
			auto *p = prep(d, "INSERT INTO block_payloads VALUES(?,?)");
			auto *v = prep(d, "INSERT INTO block_versions VALUES(1,?,?,?,'w',0,?)");
			std::string data(8, 'a');
			for (int b = 0; b < 16384; ++b) {
				sqlite3_bind_int(p, 1, b + 1);
				sqlite3_bind_blob(p, 2, data.data(), 8, SQLITE_STATIC);
				ck(sqlite3_step(p));
				sqlite3_reset(p);
				for (int j = 0; j < history; ++j) {
					std::string lo(32, 0), hi(32, 0);
					lo[31] = char(j);
					hi[31] = char(j + 1);
					sqlite3_bind_int(v, 1, b);
					sqlite3_bind_blob(v, 2, lo.data(), 32, SQLITE_TRANSIENT);
					sqlite3_bind_blob(v, 3, hi.data(), 32, SQLITE_TRANSIENT);
					sqlite3_bind_int(v, 4, b + 1);
					ck(sqlite3_step(v));
					sqlite3_reset(v);
				}
			}
			sqlite3_finalize(p);
			sqlite3_finalize(v);
			ex(d, "COMMIT");
			const char *queries[] = {
			    "SELECT b.block,p.data FROM block_versions b LEFT JOIN block_payloads p ON p.id=b.payload WHERE "
			    "b.inode=? AND b.block>=? AND b.block<=? AND b.low<=? AND b.high>? AND b.deleted=0 ORDER BY b.block",
			    "SELECT b.block,p.data FROM block_versions b LEFT JOIN block_payloads p ON p.id=b.payload WHERE "
			    "b.inode=?1 AND b.block=?2 AND b.low<=?4 AND b.high>?5 AND b.deleted=0 ORDER BY b.block"};
			for (int r = 0; r < 6; ++r)
				for (int pos = 0; pos < 2; ++pos) {
					int q = (pos + r) % 2;
					auto *s = prep(d, queries[q]);
					std::string point(32, 0);
					point[31] = char(history - 1);
					sqlite3_bind_int(s, 1, 1);
					sqlite3_bind_blob(s, 4, point.data(), 32, SQLITE_STATIC);
					sqlite3_bind_blob(s, 5, point.data(), 32, SQLITE_STATIC);
					std::mt19937 rng(4319);
					long long steps = 0;
					auto t = std::chrono::steady_clock::now();
					for (int i = 0; i < 32768; ++i) {
						int b = int(rng() % 16384);
						sqlite3_bind_int(s, 2, b);
						sqlite3_bind_int(s, 3, b);
						int rows = 0, c;
						while ((c = sqlite3_step(s)) == SQLITE_ROW) {
							if (sqlite3_column_int(s, 0) != b || sqlite3_column_bytes(s, 1) != 8)
								throw std::runtime_error("content");
							++rows;
						}
						ck(c);
						if (rows != 1)
							throw std::runtime_error("visibility");
						steps += sqlite3_stmt_status(s, SQLITE_STMTSTATUS_VM_STEP, 1);
						sqlite3_reset(s);
					}
					auto sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
					auto plan = prep(d, (std::string("EXPLAIN QUERY PLAN ") + queries[q]).c_str());
					std::string plans;
					while (sqlite3_step(plan) == SQLITE_ROW) {
						if (!plans.empty())
							plans += ';';
						plans += reinterpret_cast<const char *>(sqlite3_column_text(plan, 3));
					}
					sqlite3_finalize(plan);
					sqlite3_finalize(s);
					if (comma)
						std::cout << ',';
					comma = true;
					std::cout << "{\"history\":" << history << ",\"round\":" << r << ",\"query\":\""
					          << (q ? "point" : "range") << "\",\"vm_steps\":" << steps << ",\"seconds\":" << sec
					          << ",\"plan\":\"" << plans << "\"}";
				}
			sqlite3_close(d);
		}
		std::cout << "],\"content_verified\":true}\n";
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return 1;
	}
}
