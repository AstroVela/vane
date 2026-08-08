// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"
#include "duckdb/execution/udf_executor.hpp"

#include <deque>
#include <stdexcept>
#include <type_traits>

using namespace duckdb;

static_assert(!std::is_copy_constructible<UDFOutputLease>::value,
              "UDF output leases must never duplicate physical ownership");
static_assert(!std::is_copy_assignable<UDFOutputLease>::value,
              "UDF output leases must never duplicate physical ownership");
static_assert(std::is_nothrow_move_constructible<UDFOutputLease>::value,
              "UDF output leases must transfer ownership without throwing");
static_assert(std::is_nothrow_move_assignable<UDFOutputLease>::value,
              "UDF output leases must transfer ownership without throwing");

TEST_CASE("UDF output lease transfers ownership exactly once", "[udf][output-lease]") {
	idx_t handoffs = 0;
	idx_t releases = 0;
	{
		UDFOutputEvent producer_event;
		producer_event.output_lease = UDFOutputLease([&]() { handoffs++; }, [&]() { releases++; });

		UDFOutputEvent consumer_event(std::move(producer_event));
		CHECK_FALSE(static_cast<bool>(producer_event.output_lease));
		CHECK(static_cast<bool>(consumer_event.output_lease));
		consumer_event.output_lease.Handoff();
		consumer_event.output_lease.Handoff();
		CHECK(handoffs == 1);
		CHECK(releases == 0);
	}
	CHECK(releases == 1);
}

TEST_CASE("UDF output event consumer failure releases the current owner", "[udf][output-lease]") {
	idx_t releases_before_move = 0;
	UDFOutputEvent retained_event;
	retained_event.output_lease = UDFOutputLease({}, [&]() { releases_before_move++; });
	try {
		auto reject_before_move = [](UDFOutputEvent &&) {
			throw std::runtime_error("rejected before move");
		};
		reject_before_move(std::move(retained_event));
	} catch (const std::runtime_error &) {
		retained_event.output_lease.Release();
	}
	CHECK(releases_before_move == 1);

	idx_t releases_after_move = 0;
	std::deque<UDFOutputEvent> accepted_events;
	UDFOutputEvent transferred_event;
	transferred_event.output_lease = UDFOutputLease({}, [&]() { releases_after_move++; });
	try {
		auto accept_then_fail = [&](UDFOutputEvent &&event) {
			accepted_events.push_back(std::move(event));
			throw std::runtime_error("failed after move");
		};
		accept_then_fail(std::move(transferred_event));
	} catch (const std::runtime_error &) {
		transferred_event.output_lease.Release();
	}
	CHECK(releases_after_move == 0);
	accepted_events.clear();
	CHECK(releases_after_move == 1);
}

TEST_CASE("UDF output lease move assignment releases replaced ownership", "[udf][output-lease]") {
	idx_t replaced_releases = 0;
	idx_t incoming_releases = 0;
	{
		UDFOutputLease destination({}, [&]() { replaced_releases++; });
		UDFOutputLease source({}, [&]() { incoming_releases++; });
		destination = std::move(source);
		CHECK(replaced_releases == 1);
		CHECK(incoming_releases == 0);
		CHECK_FALSE(static_cast<bool>(source));
	}
	CHECK(incoming_releases == 1);
}

TEST_CASE("UDF output lease destructor contains release exceptions", "[udf][output-lease]") {
	idx_t release_attempts = 0;
	{
		UDFOutputLease lease({}, [&]() {
			release_attempts++;
			throw std::runtime_error("planned release failure");
		});
	}
	CHECK(release_attempts == 1);
}
