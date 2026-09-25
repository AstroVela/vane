// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"
#include "duckdb/common/types.hpp"

using namespace duckdb;

TEST_CASE("Logical type constants link C++17 consumers to the C++11 engine", "[types][cxx17]") {
	constexpr LogicalTypeId expected[] = {LogicalType::BOOLEAN, LogicalType::BIGINT, LogicalType::DOUBLE};
	// Volatile pointers retain the symbol references even in optimized builds.
	const LogicalTypeId *volatile identifiers[] = {&LogicalType::BOOLEAN, &LogicalType::BIGINT, &LogicalType::DOUBLE};
	for (idx_t i = 0; i < 3; i++) {
		REQUIRE(*identifiers[i] == expected[i]);
		REQUIRE(LogicalType(*identifiers[i]) == LogicalType(expected[i]));
	}
}
