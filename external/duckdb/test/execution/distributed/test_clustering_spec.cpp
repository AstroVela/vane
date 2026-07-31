// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb/execution/distributed/exchange/flight_exchange_manager.hpp"
#include "duckdb/execution/distributed/pipeline_node/shuffles/repartition.hpp"
#include "duckdb/execution/operator/exchange/physical_remote_exchange_sink.hpp"
#include "duckdb/execution/operator/exchange/repartition.hpp"
#include "duckdb/execution/operator/scan/physical_dummy_scan.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

using namespace duckdb;
using namespace duckdb::distributed;

namespace {

vector<BoundOrderByNode> TestRangeOrders() {
	vector<BoundOrderByNode> orders;
	orders.emplace_back(OrderType::ASCENDING, OrderByNullType::NULLS_LAST,
	                    make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0));
	orders.emplace_back(OrderType::DESCENDING, OrderByNullType::NULLS_FIRST,
	                    make_uniq<BoundReferenceExpression>(LogicalType::VARCHAR, 1));
	return orders;
}

} // namespace

TEST_CASE("ClusteringSpec: basic properties and factories", "[execution][repartition]") {
	// Range variant
	{
		RangeClusteringConfig cfg(3, TestRangeOrders());
		auto spec = ClusteringSpec::from_range_config(cfg);
		REQUIRE(spec->type() == ClusteringSpec::Type::Range);
		REQUIRE(spec->var_name() == "Range");
		REQUIRE(spec->num_partitions() == 3);
		auto by = spec->partition_by();
		REQUIRE(by.size() == 2);
		REQUIRE(by[0] != nullptr);
		REQUIRE(by[1] != nullptr);
		auto *range = dynamic_cast<RangeClusteringSpec *>(spec.get());
		REQUIRE(range != nullptr);
		REQUIRE(range->config().orders()[0].type == OrderType::ASCENDING);
		REQUIRE(range->config().orders()[0].null_order == OrderByNullType::NULLS_LAST);
		REQUIRE(range->config().orders()[1].type == OrderType::DESCENDING);
		REQUIRE(range->config().orders()[1].null_order == OrderByNullType::NULLS_FIRST);
		auto md = spec->multiline_display();
		REQUIRE(md.size() == 2);
		REQUIRE(md[1].find("ASC NULLS LAST") != string::npos);
		REQUIRE(md[1].find("DESC NULLS FIRST") != string::npos);
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

TEST_CASE("Range repartition rejects incomplete metadata", "[execution][repartition]") {
	REQUIRE_THROWS_WITH(RepartitionSpec::create_range(3, {}, {}),
	                    Catch::Matchers::Contains("at least one order expression"));

	vector<BoundOrderByNode> null_order;
	null_order.emplace_back(OrderType::ASCENDING, OrderByNullType::NULLS_LAST, nullptr);
	REQUIRE_THROWS_WITH(RepartitionSpec::create_range(3, std::move(null_order), {}),
	                    Catch::Matchers::Contains("cannot be null"));

	auto too_many_boundaries = TestRangeOrders();
	REQUIRE_THROWS_WITH(RepartitionSpec::create_range(2, std::move(too_many_boundaries), {"a", "b"}),
	                    Catch::Matchers::Contains("fewer boundary keys"));

	auto unsorted_boundaries = TestRangeOrders();
	REQUIRE_THROWS_WITH(RepartitionSpec::create_range(3, std::move(unsorted_boundaries), {"z", "a"}),
	                    Catch::Matchers::Contains("sorted"));
	REQUIRE_NOTHROW(RepartitionSpec::create_range(3, TestRangeOrders(), {"a", "a"}));

	vector<BoundOrderByNode> unresolved_order;
	unresolved_order.emplace_back(OrderType::ORDER_DEFAULT, OrderByNullType::ORDER_DEFAULT,
	                              make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0));
	REQUIRE_THROWS_WITH(RepartitionSpec::create_range(3, std::move(unresolved_order), {}),
	                    Catch::Matchers::Contains("explicit ASC or DESC"));

	auto automatic = RepartitionSpec::create_range(0, TestRangeOrders(), {"boundary-0"});
	REQUIRE(automatic->to_clustering_spec(2)->num_partitions() == 2);
	REQUIRE_THROWS_WITH(automatic->to_clustering_spec(1), Catch::Matchers::Contains("fewer boundary keys"));
}

TEST_CASE("Generic exchange builder preserves range partition metadata", "[distributed][repartition]") {
	Allocator allocator;
	auto plan = std::make_shared<PhysicalPlan>(allocator);
	vector<LogicalType> types {LogicalType::INTEGER, LogicalType::VARCHAR};
	auto &scan = plan->Make<PhysicalDummyScan>(types, 10);
	plan->SetRoot(scan);

	auto spec = RepartitionSpec::create_range(3, TestRangeOrders(), {"boundary-0", "boundary-1"});
	auto clustering = spec->to_clustering_spec(1);
	REQUIRE(clustering->type() == ClusteringSpec::Type::Range);
	REQUIRE(clustering->num_partitions() == 3);
	auto clustering_by = clustering->partition_by();
	REQUIRE(clustering_by.size() == 2);
	REQUIRE(clustering_by[0] != nullptr);
	REQUIRE(clustering_by[1] != nullptr);

	ExchangeSinkInstanceHandle sink_handle;
	sink_handle.sink_handle.task_partition_id = 0;
	sink_handle.output_partition_count = 3;
	FlightExchangeConfig flight_config;
	flight_config.node_id = "node-1";
	auto exchange_mgr = std::make_shared<FlightExchangeManager>(std::move(flight_config));

	REQUIRE_THROWS_WITH(AddRemoteExchangeSinkPlan(plan, spec, 4, "range-stage", sink_handle, exchange_mgr),
	                    Catch::Matchers::Contains("does not match exchange partition count"));
	auto result = AddRemoteExchangeSinkPlan(plan, spec, 3, "range-stage", sink_handle, std::move(exchange_mgr));
	REQUIRE(result.get() == plan.get());
	auto *sink = dynamic_cast<PhysicalRemoteExchangeSink *>(&result->Root());
	REQUIRE(sink != nullptr);
	REQUIRE(sink->RepartitionType() == RepartitionSpec::Type::Range);
	REQUIRE(sink->PartitionBy().size() == 2);
	REQUIRE(sink->PartitionBy()[0] != nullptr);
	REQUIRE(sink->PartitionBy()[1] != nullptr);
	const vector<string> expected_boundaries {"boundary-0", "boundary-1"};
	const vector<string> expected_modifiers {"ASC NULLS LAST", "DESC NULLS FIRST"};
	REQUIRE(sink->RangeBoundaries() == expected_boundaries);
	REQUIRE(sink->RangeOrderModifiers() == expected_modifiers);
	REQUIRE(sink->children.size() == 1);
}
