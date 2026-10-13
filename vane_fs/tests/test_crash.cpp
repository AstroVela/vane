// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace vane_fs;
static int notification = -1;

static void Require(bool condition, const std::string &message) {
	if (!condition)
		throw std::runtime_error(message);
}

static int PauseExtension(sqlite3 *db, char **, const sqlite3_api_routines *) {
	return sqlite3_create_function(
	    db, "test_pause", 0, SQLITE_UTF8, nullptr,
	    [](sqlite3_context *, int, sqlite3_value **) {
		    char ready = 'x';
		    if (write(notification, &ready, 1) != 1)
			    _exit(71);
		    raise(SIGSTOP);
	    },
	    nullptr, nullptr);
}

static std::string SQL(const std::string &path, const std::string &sql) {
	sqlite3 *db = nullptr;
	Require(sqlite3_open(path.c_str(), &db) == SQLITE_OK, "Opening inspection connection");
	std::string result;
	int code = sqlite3_exec(
	    db, sql.c_str(),
	    [](void *context, int count, char **values, char **) {
		    for (int i = 0; i < count; ++i)
			    *static_cast<std::string *>(context) += std::string(values[i] ? values[i] : "NULL") + ";";
		    return 0;
	    },
	    &result, nullptr);
	std::string message = sqlite3_errmsg(db);
	sqlite3_close(db);
	Require(code == SQLITE_OK, message);
	return result;
}

static std::string Counts(const std::string &path) {
	std::string result;
	for (auto *table :
	     {"branches", "snapshots", "inode_versions", "dirent_versions", "block_versions", "block_payloads"}) {
		result += SQL(path, "SELECT count(*) FROM " + std::string(table));
	}
	return result;
}

static void CheckCrash(const std::filesystem::path &root, const std::string &operation) {
	auto path = (root / (operation + ".sqlite")).string();
	{
		Workspace workspace(path);
		workspace.Checkout()->WriteFile("/file", "base");
		workspace.Fork("main", "source");
		workspace.Checkout("source")->WriteFile("/file", std::string(8192, 's'));
		workspace.Checkout("source")->WriteFile("/new", "source only");
		if (operation == "gc") {
			workspace.DeleteBranch("source");
			workspace.Checkout()->WriteFile("/file", "base");
		}
	}
	auto before = Counts(path);
	auto trigger = operation == "fork"    ? "AFTER INSERT ON branches"
	               : operation == "merge" ? "AFTER INSERT ON inode_versions"
	                                      : "AFTER DELETE ON block_payloads";
	SQL(path, "CREATE TRIGGER stop_inside_transaction " + std::string(trigger) + " BEGIN SELECT test_pause(); END");
	int descriptors[2];
	Require(pipe(descriptors) == 0, "Creating crash-test pipe");
	auto child = fork();
	Require(child >= 0, "Forking crash-test child");
	if (child == 0) {
		close(descriptors[0]);
		notification = descriptors[1];
		sqlite3_auto_extension(reinterpret_cast<void (*)()>(PauseExtension));
		try {
			Workspace workspace(path);
			if (operation == "fork")
				workspace.Fork("main", "interrupted");
			else if (operation == "merge")
				workspace.Merge(workspace.PreviewMerge("source", "main"));
			else
				workspace.CollectGarbage();
		} catch (const std::exception &error) {
			std::cerr << error.what() << '\n';
		}
		_exit(72);
	}
	close(descriptors[1]);
	pollfd waiting {descriptors[0], POLLIN, 0};
	char ready = 0;
	bool paused = poll(&waiting, 1, 10000) == 1 && read(descriptors[0], &ready, 1) == 1 && ready == 'x';
	close(descriptors[0]);
	kill(child, SIGKILL);
	int status = 0;
	waitpid(child, &status, 0);
	Require(paused, "Operation did not stop inside its transaction: " + operation);
	Require(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL, "Child was not killed");
	SQL(path, "DROP TRIGGER stop_inside_transaction");
	Require(SQL(path, "PRAGMA integrity_check") == "ok;", "SQLite integrity after crash");
	Require(Counts(path) == before, "Partially committed " + operation);
	{
		Workspace reopened(path);
		Require(reopened.RecoverOwners().owners == 1, "Crash owner was not recovered");
		Require(reopened.Checkout()->Read("/file") == "base", "Crash changed target bytes");
		Require(reopened.Checkout()->ListDirectory("/").size() == 1, "Crash changed target namespace");
		if (operation != "gc") {
			reopened.Merge(reopened.PreviewMerge("source", "main"));
			Require(reopened.Checkout()->Read("/file") == std::string(8192, 's'), "Retry after crash failed");
		}
		reopened.CollectGarbage();
	}
}

int main() {
	auto root = std::filesystem::temp_directory_path() /
	            ("vane-fs-crash-" + std::to_string(getpid()) + "-" +
	             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
	try {
		std::filesystem::create_directory(root);
		for (const auto *operation : {"fork", "merge", "gc"})
			CheckCrash(root, operation);
		std::filesystem::remove_all(root);
		std::cout << "Deterministic SIGKILL rollback and recovery for fork, merge and GC passed\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		std::filesystem::remove_all(root);
		return 1;
	}
}
