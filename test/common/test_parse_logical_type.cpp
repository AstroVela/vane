// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "catch.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/main/config.hpp"

using namespace duckdb;

TEST_CASE("Test parse logical type", "[parse_logical_type]") {
	SECTION("simple types") {
		REQUIRE(DBConfig::ParseLogicalType("integer") == LogicalType::INTEGER);
		REQUIRE(DBConfig::ParseLogicalType("any") == LogicalType::ANY);
	}

	SECTION("nested types") {
		// list
		REQUIRE(DBConfig::ParseLogicalType("ANY[]") == LogicalType::LIST(LogicalType::ANY));
		REQUIRE(DBConfig::ParseLogicalType("VARCHAR[]") == LogicalType::LIST(LogicalType::VARCHAR));

		// array
		REQUIRE(DBConfig::ParseLogicalType("ANY[3]") == LogicalType::ARRAY(LogicalType::ANY, 3));
		REQUIRE(DBConfig::ParseLogicalType("FLOAT[42]") == LogicalType::ARRAY(LogicalType::FLOAT, 42));
		REQUIRE(DBConfig::ParseLogicalType("VARCHAR[100000]") ==
		        LogicalType::ARRAY(LogicalType::VARCHAR, ArrayType::MAX_ARRAY_SIZE));

		// map
		REQUIRE(DBConfig::ParseLogicalType("MAP(VARCHAR, VARCHAR)") ==
		        LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR));
		REQUIRE(DBConfig::ParseLogicalType("MAP(ANY,ANY)") == LogicalType::MAP(LogicalType::ANY, LogicalType::ANY));
		REQUIRE(DBConfig::ParseLogicalType("MAP(INTEGER,ANY)") ==
		        LogicalType::MAP(LogicalType::INTEGER, LogicalType::ANY));
		REQUIRE(DBConfig::ParseLogicalType("MAP(ANY, DOUBLE)") ==
		        LogicalType::MAP(LogicalType::ANY, LogicalType::DOUBLE));

		// union
		child_list_t<LogicalType> union_members;
		union_members.emplace_back(make_pair("num", LogicalTypeId::INTEGER));
		union_members.emplace_back(make_pair("v", LogicalTypeId::VARCHAR));
		union_members.emplace_back(make_pair("f", LogicalTypeId::FLOAT));
		REQUIRE(DBConfig::ParseLogicalType("UNION(num INTEGER, v VARCHAR, f FLOAT)") ==
		        LogicalType::UNION(union_members));

		// struct
		child_list_t<LogicalType> struct_children;
		struct_children.emplace_back(make_pair("year", LogicalTypeId::BIGINT));
		struct_children.emplace_back(make_pair("month", LogicalTypeId::BIGINT));
		struct_children.emplace_back(make_pair("day", LogicalTypeId::BIGINT));
		REQUIRE(DBConfig::ParseLogicalType("STRUCT(year BIGINT, month BIGINT, day BIGINT)") ==
		        LogicalType::STRUCT(struct_children));
	}

	SECTION("deeper nested types") {
		// list of lists
		REQUIRE(DBConfig::ParseLogicalType("VARCHAR[][]") ==
		        LogicalType::LIST(LogicalType::LIST(LogicalType::VARCHAR)));

		// array of lists
		REQUIRE(DBConfig::ParseLogicalType("VARCHAR[][3]") ==
		        LogicalType::ARRAY(LogicalType::LIST(LogicalType::VARCHAR), 3));

		// list of structs
		child_list_t<LogicalType> date_struct_children;
		date_struct_children.emplace_back(make_pair("year", LogicalTypeId::BIGINT));
		date_struct_children.emplace_back(make_pair("month", LogicalTypeId::BIGINT));
		date_struct_children.emplace_back(make_pair("day", LogicalTypeId::BIGINT));
		REQUIRE(DBConfig::ParseLogicalType("STRUCT(year BIGINT, month BIGINT, day BIGINT)[]") ==
		        LogicalType::LIST(LogicalType::STRUCT(date_struct_children)));

		// map with list as key
		REQUIRE(DBConfig::ParseLogicalType("MAP(VARCHAR[],FLOAT)") ==
		        LogicalType::MAP(LogicalType::LIST(LogicalType::VARCHAR), LogicalType::FLOAT));

		// struct with list, array and map
		child_list_t<LogicalType> mix_struct_children;
		mix_struct_children.emplace_back(make_pair("my_list", LogicalType::LIST(LogicalType::ANY)));
		mix_struct_children.emplace_back(make_pair("my_array", LogicalType::ARRAY(LogicalType::VARCHAR, 2)));
		mix_struct_children.emplace_back(
		    make_pair("my_map", LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR)));
		REQUIRE(DBConfig::ParseLogicalType("STRUCT(my_list ANY[], my_array VARCHAR[2], my_map MAP(VARCHAR,VARCHAR))") ==
		        LogicalType::STRUCT(mix_struct_children));
	}

	SECTION("quoted nested member names round trip") {
		child_list_t<LogicalType> struct_children;
		struct_children.emplace_back(make_pair("label", LogicalType::BIGINT));
		struct_children.emplace_back(make_pair("display name", LogicalType::VARCHAR));
		struct_children.emplace_back(make_pair("quoted\"name", LogicalType::DOUBLE));
		struct_children.emplace_back(make_pair("comma,name", LogicalType::LIST(LogicalType::INTEGER)));
		auto struct_type = LogicalType::STRUCT(std::move(struct_children));

		REQUIRE(DBConfig::ParseLogicalType(struct_type.ToString()) == struct_type);
		REQUIRE(DBConfig::ParseLogicalType(LogicalType::LIST(struct_type).ToString()) ==
		        LogicalType::LIST(struct_type));

		child_list_t<LogicalType> union_members;
		union_members.emplace_back(make_pair("label", LogicalType::BIGINT));
		union_members.emplace_back(make_pair("value with space", struct_type));
		auto union_type = LogicalType::UNION(std::move(union_members));
		REQUIRE(DBConfig::ParseLogicalType(union_type.ToString()) == union_type);
	}
}
