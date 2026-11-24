// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

static bool crash_remap = false;
static int remaps = 0;
extern "C" int __real_sqlite3_step(sqlite3_stmt *);
extern "C" int __wrap_sqlite3_step(sqlite3_stmt *statement) {
	int result = __real_sqlite3_step(statement);
	const char *sql = sqlite3_sql(statement);
	const char *prefix = "UPDATE block_versions SET payload=";
	if (crash_remap && result == SQLITE_DONE && sql && std::strncmp(sql, prefix, std::strlen(prefix)) == 0 &&
	    ++remaps == 2) {
		_exit(86);
	}
	return result;
}

static void Require(bool condition, const char *message) {
	if (!condition)
		throw std::runtime_error(message);
}

static std::string Inspect(const char *path) {
	sqlite3 *db = nullptr;
	Require(sqlite3_open(path, &db) == SQLITE_OK, "Open inspector");
	std::string result;
	int rc = sqlite3_exec(
	    db,
	    "SELECT quote(payload),quote(payload_offset),hex(low),hex(high) FROM block_versions ORDER BY inode,block,low;"
	    "SELECT id,hex(data) FROM block_payloads ORDER BY id; PRAGMA integrity_check;",
	    [](void *context, int n, char **values, char **) {
		    auto &text = *static_cast<std::string *>(context);
		    for (int i = 0; i < n; i++)
			    text += std::string(values[i] ? values[i] : "NULL") + "|";
		    text += "\n";
		    return 0;
	    },
	    &result, nullptr);
	sqlite3_close(db);
	Require(rc == SQLITE_OK, "Inspect rows");
	Require(result.size() >= 4 && result.substr(result.size() - 4) == "ok|\n", "Integrity check");
	return result;
}

int main(int argc, char **argv) {
	try {
		Require(argc == 2, "Expected database path");
		const std::string expected = std::string(4096, '\0') + std::string(VANE_FS_EXTENT_BYTES - 4096, 'a');
		{
			vane_fs::Workspace workspace(argv[1]);
			auto main = workspace.Checkout();
			main->WriteFile("/file", std::string(VANE_FS_EXTENT_BYTES, 'a'));
			main->Write("/file", std::string(4096, '\0'));
		}
		auto before = Inspect(argv[1]);
		pid_t child = fork();
		Require(child >= 0, "fork");
		if (!child) {
			vane_fs::Workspace workspace(argv[1]);
			crash_remap = true;
			workspace.CollectGarbage();
			_exit(1);
		}
		int status;
		Require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 86,
		        "Child must crash after its second remap");
		Require(Inspect(argv[1]) == before, "Crash exposed partial GC");
		{
			vane_fs::Workspace workspace(argv[1]);
			Require(workspace.Checkout()->Read("/file") == expected, "Crash changed file contents");
			workspace.RecoverOwners();
			workspace.CollectGarbage();
			Require(workspace.Checkout()->Read("/file") == expected, "Retry changed file contents");
		}
		Require(Inspect(argv[1]) != before, "Retry did not compact");
		std::cout << "PASS: crash after two slice remaps rolls back and GC retries\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
