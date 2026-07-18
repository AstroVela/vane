// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb/execution/operator/exchange/repartition.hpp"

using namespace duckdb;

TEST_CASE("ClusteringSpec: basic properties and factories", "[execution][repartition]") {
	// Range variant
	{
		RangeClusteringConfig cfg(3, std::vector<ExprRef> {nullptr}, std::vector<bool> {false});
		auto spec = ClusteringSpec::from_range_config(cfg);
		REQUIRE(spec->type() == ClusteringSpec::Type::Range);
		REQUIRE(spec->var_name() == "Range");
		REQUIRE(spec->num_partitions() == 3);
		auto by = spec->partition_by();
		REQUIRE(by.size() == 1);
		auto md = spec->multiline_display();
		REQUIRE(!md.empty());
	}

	// Hash variant
	{
		HashClusteringConfig cfg(4, std::vector<ExprRef> {nullptr, nullptr});
		auto spec = ClusteringSpec::from_hash_config(cfg);
		REQUIRE(spec->type() == ClusteringSpec::Type::Hash);
		REQUIRE(spec->var_name() == "Hash");
		REQUIRE(spec->num_partitions() == 4);
		auto by = spec->partition_by();
		REQUIRE(by.size() == 2);
		auto md = spec->multiline_display();
		REQUIRE(!md.empty());
	}

	// Random variant
	{
		RandomClusteringConfig cfg(7);
		auto spec = ClusteringSpec::from_random_config(cfg);
		REQUIRE(spec->type() == ClusteringSpec::Type::Random);
		REQUIRE(spec->var_name() == "Random");
		REQUIRE(spec->num_partitions() == 7);
		auto by = spec->partition_by();
		REQUIRE(by.empty());
		auto md = spec->multiline_display();
		REQUIRE(!md.empty());
	}

	// Unknown variant and factory helpers
	{
		auto spec = ClusteringSpec::unknown();
		REQUIRE(spec->type() == ClusteringSpec::Type::Unknown);
		REQUIRE(spec->var_name() == "Unknown");
		// unknown() was implemented to create UnknownClusteringConfig(0)
		REQUIRE(spec->num_partitions() == 0);

		auto spec2 = ClusteringSpec::unknown_with_num_partitions(5);
		REQUIRE(spec2->num_partitions() == 5);
	}
}
