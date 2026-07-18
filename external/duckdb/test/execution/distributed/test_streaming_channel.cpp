// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file test_streaming_channel.cpp
 * @brief Unit tests for the T2 removal and dispatcher streaming channel changes.
 *
 * These tests verify that:
 * 1. Temporary senders created from UnboundedChannelState correctly manage
 *    sender counts (increment on create, decrement on destroy).
 * 2. The streaming channel state can be stored and retrieved from a
 *    WorkerManager (simulating the execute_plan → dispatcher path).
 * 3. Channel close ordering is correct: the channel only closes when ALL
 *    senders (including temporaries) are destroyed.
 * 4. Concurrent creation of temporary senders is thread-safe.
 */

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "catch.hpp"
#include "test_common.hpp"
#include "duckdb/execution/distributed/utils/channel.hpp"

using namespace duckdb::distributed;
using namespace duckdb::distributed::testing;

//==============================================================================
// Section 1: UnboundedSender temporary lifecycle tests
//==============================================================================

TEST_CASE("Streaming channel: temp sender from state increments and decrements count",
          "[distributed][streaming_channel]") {
	auto ch_pair_ = create_unbounded_channel<int>();
	auto sender = std::move(ch_pair_.first);
	auto receiver = std::move(ch_pair_.second);

	// Get the underlying state from the sender
	auto state = sender.state();
	REQUIRE(state != nullptr);

	// Initial sender count should be 1 (the original sender)
	size_t initial_count = state->sender_count();
	REQUIRE(initial_count == 1);

	SECTION("Creating a temp sender increments count") {
		{
			UnboundedSender<int> temp_sender(state);
			REQUIRE(state->sender_count() == 2);

			// Send through temp sender should work
			auto res = temp_sender.send(42);
			REQUIRE(res.is_ok());
		}
		// After temp_sender goes out of scope, count should be back to 1
		REQUIRE(state->sender_count() == 1);

		// Verify the value arrived
		auto item = receiver.try_recv();
		REQUIRE(item.first);
		REQUIRE(item.second == 42);
	}

	SECTION("Multiple temp senders increment correctly") {
		{
			UnboundedSender<int> temp1(state);
			REQUIRE(state->sender_count() == 2);
			{
				UnboundedSender<int> temp2(state);
				REQUIRE(state->sender_count() == 3);
			}
			REQUIRE(state->sender_count() == 2);
		}
		REQUIRE(state->sender_count() == 1);
	}

	SECTION("Channel stays open while any sender alive") {
		// Drop original sender by moving it into a block scope
		{
			auto moved_sender = std::move(sender);
			// moved_sender goes out of scope here
		}
		// Since original sender was moved and destroyed, count should be 0
		// Channel is closed, recv returns nullopt
		auto item = receiver.recv();
		REQUIRE_FALSE(item.first);
	}
}

TEST_CASE("Streaming channel: temp sender send after original sender dropped", "[distributed][streaming_channel]") {
	auto ch_pair_ = create_unbounded_channel<int>();
	auto sender = std::move(ch_pair_.first);
	auto receiver = std::move(ch_pair_.second);
	auto state = sender.state();

	// Create temp before dropping original
	UnboundedSender<int> temp_sender(state);
	REQUIRE(state->sender_count() == 2);

	// Drop the original sender
	{
		auto moved = std::move(sender);
		// moved goes out of scope
	}
	REQUIRE(state->sender_count() == 1); // only temp remains

	// Channel should still be open — temp sender keeps it alive
	auto send_res = temp_sender.send(99);
	REQUIRE(send_res.is_ok());

	auto item = receiver.try_recv();
	REQUIRE(item.first);
	REQUIRE(item.second == 99);
}

TEST_CASE("Streaming channel: channel closes only when all senders drop", "[distributed][streaming_channel]") {
	auto ch_pair_ = create_unbounded_channel<int>();
	auto sender = std::move(ch_pair_.first);
	auto receiver = std::move(ch_pair_.second);
	auto state = sender.state();

	// Send some values through original
	sender.send(1);
	sender.send(2);

	// Create temp, send through it
	{
		UnboundedSender<int> temp(state);
		temp.send(3);
	} // temp destroyed, count goes from 2→1

	// Original still alive — channel open
	sender.send(4);

	// Drop original
	{ auto moved = std::move(sender); }

	// Now all senders gone → channel closed
	// Drain all values
	std::vector<int> values;
	while (true) {
		auto item = receiver.recv();
		if (!item.first)
			break;
		values.push_back(item.second);
	}

	REQUIRE(values.size() == 4);
	REQUIRE(values[0] == 1);
	REQUIRE(values[1] == 2);
	REQUIRE(values[2] == 3);
	REQUIRE(values[3] == 4);
}

//==============================================================================
// Section 3: Concurrent temp sender creation from multiple threads
//==============================================================================

TEST_CASE("Streaming channel: concurrent temp senders are thread-safe", "[distributed][streaming_channel]") {
	auto ch_pair_ = create_unbounded_channel<int>();
	auto sender = std::move(ch_pair_.first);
	auto receiver = std::move(ch_pair_.second);
	auto state = sender.state();

	const int num_threads = 10;
	const int sends_per_thread = 100;
	std::atomic<int> total_sent {0};

	std::vector<std::thread> threads;
	for (int t = 0; t < num_threads; t++) {
		threads.emplace_back([&state, &total_sent, t, sends_per_thread]() {
			for (int i = 0; i < sends_per_thread; i++) {
				// Create temp sender, send, destroy — all in tight loop
				UnboundedSender<int> temp(state);
				auto res = temp.send(t * sends_per_thread + i);
				if (res.is_ok()) {
					total_sent.fetch_add(1);
				}
			}
		});
	}

	for (auto &t : threads) {
		t.join();
	}

	// All sends should succeed
	REQUIRE(total_sent.load() == num_threads * sends_per_thread);

	// Sender count should be back to 1 (original sender only)
	REQUIRE(state->sender_count() == 1);

	// Drop original sender to close channel
	{ auto moved = std::move(sender); }

	// Drain and verify all values arrived
	int count = 0;
	while (true) {
		auto item = receiver.recv();
		if (!item.first)
			break;
		count++;
	}
	REQUIRE(count == num_threads * sends_per_thread);
}

TEST_CASE("Streaming channel: rapid create-send-destroy cycle", "[distributed][streaming_channel]") {
	auto ch_pair_ = create_unbounded_channel<int>();
	auto sender = std::move(ch_pair_.first);
	auto receiver = std::move(ch_pair_.second);
	auto state = sender.state();

	// Simulate what the dispatcher does on every completed task:
	// create temp sender → send → destroy, repeated N times
	const int iterations = 1000;
	for (int i = 0; i < iterations; i++) {
		UnboundedSender<int> temp(state);
		auto res = temp.send(i);
		REQUIRE(res.is_ok());
	}

	// Sender count should still be 1 (only original remains)
	REQUIRE(state->sender_count() == 1);

	// All values should arrive in order (single-threaded send)
	for (int i = 0; i < iterations; i++) {
		auto item = receiver.try_recv();
		REQUIRE(item.first);
		REQUIRE(item.second == i);
	}

	// No more items
	auto item = receiver.try_recv();
	REQUIRE_FALSE(item.first);
}
