// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <unordered_map>
#include <vector>
#include <filesystem>
#include <memory>

using Clock = std::chrono::steady_clock;
static std::string json_string(const std::string &value) {
	std::string result = "\"";
	const char *digits = "0123456789abcdef";
	for (unsigned char byte : value) {
		switch (byte) {
		case '"':
			result += "\\\"";
			break;
		case '\\':
			result += "\\\\";
			break;
		case '\n':
			result += "\\n";
			break;
		case '\r':
			result += "\\r";
			break;
		case '\t':
			result += "\\t";
			break;
		default:
			if (byte < 0x20) {
				result += "\\u00";
				result += digits[byte >> 4];
				result += digits[byte & 15];
			} else
				result += char(byte);
		}
	}
	return result + '"';
}

static bool measuring = false;
static sqlite3 *current_db = nullptr;
static uint64_t sync_calls = 0, sync_ns = 0;
struct SQLStat {
	uint64_t prepares = 0, steps = 0, vm_steps = 0, fullscan_steps = 0, prepare_ns = 0, step_ns = 0, finalize_ns = 0;
};
static std::map<std::string, SQLStat> queries;
static std::unordered_map<sqlite3_stmt *, std::string> live;
static std::map<std::string, std::pair<uint64_t, uint64_t>> executions;
static uint64_t elapsed_ns(Clock::time_point start) {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}

struct IOStat {
	uint64_t writes = 0, write_bytes = 0, write_ns = 0, reads = 0, read_bytes = 0, read_ns = 0, syncs = 0, sync_ns = 0;
};
static std::map<std::string, IOStat> io;
struct FileHook {
	sqlite3_io_methods methods;
	const sqlite3_io_methods *original;
	std::string kind;
};
static std::map<sqlite3_file *, FileHook> files;
static sqlite3_vfs *original_vfs = nullptr;
static int IORead(sqlite3_file *file, void *data, int amount, sqlite3_int64 offset) {
	auto &hook = files.at(file);
	auto start = Clock::now();
	int rc = hook.original->xRead(file, data, amount, offset);
	auto ns = elapsed_ns(start);
	if (measuring) {
		auto &s = io[hook.kind];
		++s.reads;
		s.read_bytes += amount;
		s.read_ns += ns;
	}
	return rc;
}
static int IOWrite(sqlite3_file *file, const void *data, int amount, sqlite3_int64 offset) {
	auto &hook = files.at(file);
	auto start = Clock::now();
	int rc = hook.original->xWrite(file, data, amount, offset);
	auto ns = elapsed_ns(start);
	if (measuring) {
		auto &s = io[hook.kind];
		++s.writes;
		s.write_bytes += amount;
		s.write_ns += ns;
	}
	return rc;
}
static int IOSync(sqlite3_file *file, int flags) {
	auto &hook = files.at(file);
	auto start = Clock::now();
	int rc = hook.original->xSync(file, flags);
	auto ns = elapsed_ns(start);
	if (measuring) {
		auto &s = io[hook.kind];
		++s.syncs;
		s.sync_ns += ns;
	}
	return rc;
}
static int IOClose(sqlite3_file *file) {
	file->pMethods = files.at(file).original;
	files.erase(file);
	return file->pMethods->xClose(file);
}
static int IOOpen(sqlite3_vfs *, const char *name, sqlite3_file *file, int flags, int *output) {
	int rc = original_vfs->xOpen(original_vfs, name, file, flags, output);
	if (file->pMethods) {
		std::string kind = (flags & SQLITE_OPEN_MAIN_DB)        ? "database"
		                   : (flags & SQLITE_OPEN_WAL)          ? "wal"
		                   : (flags & SQLITE_OPEN_MAIN_JOURNAL) ? "journal"
		                                                        : "other";
		auto &h = files.emplace(file, FileHook {*file->pMethods, file->pMethods, kind}).first->second;
		h.methods.xRead = IORead;
		h.methods.xWrite = IOWrite;
		h.methods.xSync = IOSync;
		h.methods.xClose = IOClose;
		file->pMethods = &h.methods;
	}
	return rc;
}
struct IOHooks {
	sqlite3_vfs vfs;
	IOHooks() {
		original_vfs = sqlite3_vfs_find(nullptr);
		vfs = *original_vfs;
		vfs.zName = "vane-gap-probe";
		vfs.xOpen = IOOpen;
		if (sqlite3_vfs_register(&vfs, 1) != SQLITE_OK)
			throw std::runtime_error("register VFS");
	}
	~IOHooks() {
		sqlite3_vfs_unregister(&vfs);
	}
};
struct CommitStat {
	uint64_t ns, main_bytes, wal_bytes, sync_ns;
};
static std::vector<CommitStat> commits;
extern "C" {
int __real_fsync(int);
int __real_fdatasync(int);
int __wrap_fsync(int fd) {
	auto start = Clock::now();
	int rc = __real_fsync(fd);
	if (measuring) {
		++sync_calls;
		sync_ns += elapsed_ns(start);
	}
	return rc;
}
int __wrap_fdatasync(int fd) {
	auto start = Clock::now();
	int rc = __real_fdatasync(fd);
	if (measuring) {
		++sync_calls;
		sync_ns += elapsed_ns(start);
	}
	return rc;
}

int __real_sqlite3_prepare_v2(sqlite3 *, const char *, int, sqlite3_stmt **, const char **);
int __real_sqlite3_step(sqlite3_stmt *);
int __real_sqlite3_finalize(sqlite3_stmt *);
int __real_sqlite3_exec(sqlite3 *, const char *, int (*)(void *, int, char **, char **), void *, char **);
int __wrap_sqlite3_prepare_v2(sqlite3 *db, const char *sql, int n, sqlite3_stmt **stmt, const char **tail) {
	current_db = db;
	auto start = Clock::now();
	int rc = __real_sqlite3_prepare_v2(db, sql, n, stmt, tail);
	auto duration = elapsed_ns(start);
	if (measuring) {
		auto &stat = queries[sql];
		++stat.prepares;
		stat.prepare_ns += duration;
	}
	if (rc == SQLITE_OK && *stmt)
		live[*stmt] = sql;
	return rc;
}
int __wrap_sqlite3_step(sqlite3_stmt *stmt) {
	auto start = Clock::now();
	int rc = __real_sqlite3_step(stmt);
	auto duration = elapsed_ns(start);
	auto vm = sqlite3_stmt_status(stmt, SQLITE_STMTSTATUS_VM_STEP, 1);
	auto fullscan = sqlite3_stmt_status(stmt, SQLITE_STMTSTATUS_FULLSCAN_STEP, 1);
	auto found = live.find(stmt);
	if (measuring && found != live.end()) {
		auto &stat = queries[found->second];
		++stat.steps;
		stat.step_ns += duration;
		stat.vm_steps += vm;
		stat.fullscan_steps += fullscan;
	}
	return rc;
}
int __wrap_sqlite3_finalize(sqlite3_stmt *stmt) {
	auto found = live.find(stmt);
	std::string sql = found == live.end() ? "" : found->second;
	live.erase(stmt);
	auto start = Clock::now();
	int rc = __real_sqlite3_finalize(stmt);
	auto duration = elapsed_ns(start);
	if (measuring && !sql.empty())
		queries[sql].finalize_ns += duration;
	return rc;
}
int __wrap_sqlite3_exec(sqlite3 *db, const char *sql, int (*cb)(void *, int, char **, char **), void *arg, char **err) {
	current_db = db;
	if (!measuring)
		return __real_sqlite3_exec(db, sql, cb, arg, err);
	auto main_before = io["database"].write_bytes, wal_before = io["wal"].write_bytes, sync_before = sync_ns;
	auto start = Clock::now();
	int rc = __real_sqlite3_exec(db, sql, cb, arg, err);
	auto duration = elapsed_ns(start);
	if (std::string(sql) == "COMMIT")
		commits.push_back({duration, io["database"].write_bytes - main_before, io["wal"].write_bytes - wal_before,
		                   sync_ns - sync_before});
	auto &stat = executions[sql];
	++stat.first;
	stat.second += duration;
	return rc;
}
}
static int cache_count(int kind, bool reset) {
	if (!current_db)
		return 0;
	int value = 0, high = 0;
	sqlite3_db_status(current_db, kind, &value, &high, reset);
	return value;
}
static double seconds(timeval v) {
	return v.tv_sec + v.tv_usec / 1e6;
}
template <class F>
static void phase(const std::string &name, F action) {
	queries.clear();
	executions.clear();
	io.clear();
	commits.clear();
	sync_calls = sync_ns = 0;
	cache_count(SQLITE_DBSTATUS_CACHE_HIT, true);
	cache_count(SQLITE_DBSTATUS_CACHE_MISS, true);
	cache_count(SQLITE_DBSTATUS_CACHE_WRITE, true);
	cache_count(SQLITE_DBSTATUS_CACHE_SPILL, true);
	rusage before {}, after {};
	getrusage(RUSAGE_SELF, &before);
	measuring = true;
	auto start = Clock::now();
	action();
	auto ns = elapsed_ns(start);
	measuring = false;
	getrusage(RUSAGE_SELF, &after);
	std::cout << "{\"phase\":" << json_string(name) << ",\"seconds\":" << ns / 1e9
	          << ",\"user_seconds\":" << seconds(after.ru_utime) - seconds(before.ru_utime)
	          << ",\"system_seconds\":" << seconds(after.ru_stime) - seconds(before.ru_stime)
	          << ",\"sync_calls\":" << sync_calls << ",\"sync_seconds\":" << sync_ns / 1e9
	          << ",\"cache_hits\":" << cache_count(SQLITE_DBSTATUS_CACHE_HIT, false)
	          << ",\"cache_misses\":" << cache_count(SQLITE_DBSTATUS_CACHE_MISS, false)
	          << ",\"cache_spills\":" << cache_count(SQLITE_DBSTATUS_CACHE_SPILL, false)
	          << ",\"cache_writes\":" << cache_count(SQLITE_DBSTATUS_CACHE_WRITE, false) << ",\"sql\":[";
	bool first = true;
	for (auto &item : queries) {
		auto &s = item.second;
		if (!first)
			std::cout << ',';
		first = false;
		std::cout << "{\"text\":" << json_string(item.first) << ",\"prepares\":" << s.prepares
		          << ",\"steps\":" << s.steps << ",\"vm_steps\":" << s.vm_steps
		          << ",\"fullscan_steps\":" << s.fullscan_steps << ",\"prepare_seconds\":" << s.prepare_ns / 1e9
		          << ",\"step_seconds\":" << s.step_ns / 1e9 << ",\"finalize_seconds\":" << s.finalize_ns / 1e9 << '}';
	}
	std::cout << "],\"exec\":[";
	first = true;
	for (auto &item : executions) {
		if (!first)
			std::cout << ',';
		first = false;
		std::cout << "{\"text\":" << json_string(item.first) << ",\"count\":" << item.second.first
		          << ",\"seconds\":" << item.second.second / 1e9 << '}';
	}
	std::cout << "],\"io\":[";
	first = true;
	for (auto &item : io) {
		auto &s = item.second;
		if (!first)
			std::cout << ',';
		first = false;
		std::cout << "{\"file\":" << json_string(item.first) << ",\"writes\":" << s.writes
		          << ",\"write_bytes\":" << s.write_bytes << ",\"write_seconds\":" << s.write_ns / 1e9
		          << ",\"reads\":" << s.reads << ",\"read_bytes\":" << s.read_bytes
		          << ",\"read_seconds\":" << s.read_ns / 1e9 << ",\"syncs\":" << s.syncs
		          << ",\"sync_seconds\":" << s.sync_ns / 1e9 << '}';
	}
	std::cout << "],\"commits\":[";
	first = true;
	for (auto &c : commits) {
		if (!first)
			std::cout << ',';
		first = false;
		std::cout << "{\"seconds\":" << c.ns / 1e9 << ",\"main_bytes\":" << c.main_bytes
		          << ",\"wal_bytes\":" << c.wal_bytes << ",\"sync_seconds\":" << c.sync_ns / 1e9 << '}';
	}
	std::cout << "]}" << std::endl;
}

static std::string block_data(size_t first = 0) {
	std::string result(1048576, '\0');
	for (size_t b = 0; b < 256; ++b)
		std::fill(result.begin() + b * 4096, result.begin() + (b + 1) * 4096, char((first + b) % 251 + 1));
	return result;
}
int main(int argc, char **argv) {
	try {
		if (argc != 3)
			return 2;
		const int64_t shift = std::stoll(argv[2]);
		IOHooks hooks;
		vane_fs::Workspace workspace(argv[1]);
		workspace.AcquireMount("main");
		auto session = workspace.Checkout();
		auto file = session->OpenFile("/bulk", true, true);
		auto data = block_data();
		if (shift)
			session->WriteInode(file.inode, std::string(shift, 'p'), 0);
		phase("write", [&] {
			for (size_t i = 0; i < 16; ++i)
				session->WriteInode(file.inode, data, shift + i * 1048576);
		});
		phase("sync", [&] { workspace.Sync(); });
		for (size_t i = 0; i < 16; ++i)
			if (session->ReadInode(file.inode, shift + i * 1048576, 1048576) != data)
				throw std::runtime_error("bytes mismatch");
		if (shift && session->ReadInode(file.inode, 0, shift) != std::string(shift, 'p'))
			throw std::runtime_error("prefix mismatch");
		phase("close", [&] {
			session->CloseFile(file.inode);
			session->Close();
			workspace.ReleaseMount("main");
			workspace.Close();
			current_db = nullptr;
		});
		std::cout << "{\"status\":\"PASS\"}" << std::endl;
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return 1;
	}
}
