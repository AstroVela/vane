// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "costs.hpp"
#include <cerrno>
#include <iostream>
#include <new>
#include <sqlite3.h>
#include <stdexcept>
#include <unistd.h>

static void Require(bool condition, const char *message) {
	if (!condition)
		throw std::runtime_error(message);
}
int main() {
	try {
		using namespace write_costs;
		Begin(0);
		{
			Scope root(Root);
			{
				Scope batch(Batch);
				auto *memory = ::operator new(4096);
				::operator delete(memory);
				Require(pwrite(-1, "x", 1, 0) == -1 && errno == EBADF, "Wrapped pwrite lost errno");
				Require(fdatasync(-1) == -1 && errno == EBADF, "Wrapped fdatasync lost errno");
				batch.Finish();
			}
			sqlite3 *db = nullptr;
			Require(sqlite3_open(":memory:", &db) == SQLITE_OK, "Open SQLite counter control");
			sqlite3_stmt *statement = nullptr;
			Require(sqlite3_prepare_v2(db, "SELECT 1", -1, &statement, nullptr) == SQLITE_OK,
			        "Prepare counter control");
			Require(sqlite3_step(statement) == SQLITE_ROW && sqlite3_column_int(statement, 0) == 1,
			        "Wrapped SQLite step changed result");
			sqlite3_finalize(statement);
			Require(sqlite3_close(db) == SQLITE_OK, "Close counter control");
		}
		auto values = End();
		Require(values[Batch].calls == 1 && values[Batch].cpp_allocations == 1 &&
		            values[Batch].cpp_allocated_bytes == 4096,
		        "Allocation attribution or explicit Finish changed");
		Require(values[Pwrite].calls == 1 && values[StepOther].calls == 1 && values[StepOther].vm_steps > 0,
		        "Counter did not observe actual wrapped calls");
		int64_t wall = 0, cpu = 0;
		for (auto &value : values) {
			Require(value.exclusive_wall_ns >= 0 && value.exclusive_cpu_ns >= 0, "Negative exclusive interval");
			wall += value.exclusive_wall_ns;
			cpu += value.exclusive_cpu_ns;
		}
		Require(wall == values[Root].inclusive_wall_ns && cpu == values[Root].inclusive_cpu_ns,
		        "Nested stages double counted time");
		std::cout << "PASS nested accounting, allocation ownership, errno and SQLite VM control\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
