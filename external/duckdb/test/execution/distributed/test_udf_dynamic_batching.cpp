// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb/execution/operator/projection/udf_dynamic_batching.hpp"

using namespace duckdb;

namespace {

UDFDynamicBatchingConfig TestConfig() {
	UDFDynamicBatchingConfig config;
	config.enabled = true;
	config.min_batch_rows = 1;
	config.max_batch_rows = 512;
	config.initial_batch_rows = 256;
	config.target_batch_latency = std::chrono::milliseconds(100);
	config.latency_tolerance = std::chrono::milliseconds(10);
	config.step_size_alpha = 20;
	config.correction_delta = 5;
	config.history_size = 16;
	return config;
}

} // namespace

TEST_CASE("Dynamic UDF batching starts with a bounded exploratory batch", "[execution][udf][dynamic-batching]") {
	auto config = TestConfig();
	UDFDynamicBatchSizer sizer(config);

	REQUIRE(sizer.Enabled());
	REQUIRE(sizer.CurrentBatchRows() == 256);
	REQUIRE(sizer.SearchLow() == 1);
	REQUIRE(sizer.SearchHigh() == 256);
	REQUIRE(sizer.ObservationCount() == 0);
}

TEST_CASE("Dynamic UDF batching expands after low latency", "[execution][udf][dynamic-batching]") {
	auto config = TestConfig();
	UDFDynamicBatchSizer sizer(config);

	sizer.Record(256, std::chrono::milliseconds(50));

	REQUIRE(sizer.SearchLow() == 256);
	REQUIRE(sizer.SearchHigh() == 260);
	REQUIRE(sizer.CurrentBatchRows() == 258);
}

TEST_CASE("Dynamic UDF batching contracts after high latency", "[execution][udf][dynamic-batching]") {
	auto config = TestConfig();
	UDFDynamicBatchSizer sizer(config);

	sizer.Record(256, std::chrono::milliseconds(150));

	REQUIRE(sizer.SearchLow() == 1);
	REQUIRE(sizer.SearchHigh() == 256);
	REQUIRE(sizer.CurrentBatchRows() == 128);
}

TEST_CASE("Dynamic UDF batching tightens around target latency", "[execution][udf][dynamic-batching]") {
	auto config = TestConfig();
	UDFDynamicBatchSizer sizer(config);

	sizer.Record(80, std::chrono::milliseconds(100));

	REQUIRE(sizer.SearchLow() == 70);
	REQUIRE(sizer.SearchHigh() == 90);
	REQUIRE(sizer.CurrentBatchRows() == 80);
}

TEST_CASE("Dynamic UDF batching respects its maximum and rolling window", "[execution][udf][dynamic-batching]") {
	auto config = TestConfig();
	config.max_batch_rows = 100;
	config.initial_batch_rows = 100;
	UDFDynamicBatchSizer sizer(config);

	for (idx_t observation = 0; observation < 32; observation++) {
		sizer.Record(100, std::chrono::milliseconds(50));
	}

	REQUIRE(sizer.CurrentBatchRows() == 100);
	REQUIRE(sizer.SearchLow() == 100);
	REQUIRE(sizer.SearchHigh() == 100);
	REQUIRE(sizer.ObservationCount() == 16);
}

TEST_CASE("Dynamic UDF batching handles crossed search endpoints from out-of-order completions",
          "[execution][udf][dynamic-batching]") {
	auto config = TestConfig();
	config.history_size = 1;
	UDFDynamicBatchSizer sizer(config);

	// A short, slow batch can complete before an earlier full, fast batch.
	sizer.Record(1, std::chrono::milliseconds(150));
	sizer.Record(256, std::chrono::milliseconds(50));

	REQUIRE(sizer.SearchLow() == 256);
	REQUIRE(sizer.SearchHigh() == 24);
	REQUIRE(sizer.CurrentBatchRows() == 140);
}

TEST_CASE("Dynamic UDF batching rejects invalid bounds", "[execution][udf][dynamic-batching]") {
	auto config = TestConfig();
	config.min_batch_rows = 20;
	config.max_batch_rows = 10;

	REQUIRE_THROWS_WITH(UDFDynamicBatchSizer(config),
	                    Catch::Matchers::Contains("min batch rows must not exceed max batch rows"));
}

TEST_CASE("Dynamic UDF batching rejects latency-bound overflow", "[execution][udf][dynamic-batching]") {
	auto config = TestConfig();
	config.target_batch_latency =
	    std::chrono::microseconds(std::numeric_limits<std::chrono::microseconds::rep>::max() - 5);
	config.latency_tolerance = std::chrono::microseconds(10);

	REQUIRE_THROWS_WITH(UDFDynamicBatchSizer(config),
	                    Catch::Matchers::Contains("target latency plus tolerance is too large"));
}
