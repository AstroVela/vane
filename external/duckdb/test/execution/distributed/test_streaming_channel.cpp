// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file test_streaming_channel.cpp
 * @brief Unit tests for distributed streaming channel lifecycle behavior.
 *
 * These tests verify that:
 * 1. Temporary senders created from UnboundedChannelState correctly manage
 *    sender counts (increment on create, decrement on destroy).
 * 2. Sender completion preserves queued values for the receiver to drain.
 * 3. Receiver abandonment closes the channel and releases queued values.
 * 4. Moving a receiver transfers its single-consumer ownership.
 * 5. Concurrent sender activity and receiver closure are thread-safe.
 */

#include <atomic>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "catch.hpp"
#include "test_common.hpp"
#include "duckdb/execution/distributed/plan/distributed_physical_plan.hpp"
#include "duckdb/execution/distributed/utils/channel.hpp"

using namespace duckdb::distributed;
using namespace duckdb::distributed::testing;

static_assert(!std::is_copy_constructible<UnboundedReceiver<int>>::value,
              "unbounded channels must have exactly one receiver owner");
static_assert(!std::is_copy_assignable<UnboundedReceiver<int>>::value,
              "unbounded channels must have exactly one receiver owner");
static_assert(std::is_nothrow_move_assignable<UnboundedReceiver<int>>::value,
              "replacing an unbounded receiver must not throw");
static_assert(std::is_nothrow_destructible<UnboundedReceiver<int>>::value,
              "destroying an unbounded receiver must not throw");
static_assert(noexcept(std::declval<UnboundedReceiver<int> &>().close()),
              "closing an unbounded receiver must not throw");

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
// Section 2: UnboundedReceiver lifecycle tests
//==============================================================================

TEST_CASE("Streaming channel: dropping receiver closes channel and rejects sends", "[distributed][streaming_channel]") {
	auto ch_pair_ = create_unbounded_channel<int>();
	auto sender = std::move(ch_pair_.first);
	auto receiver = std::move(ch_pair_.second);
	auto state = sender.state();

	{ auto dropped_receiver = std::move(receiver); }

	REQUIRE(state->is_closed());
	REQUIRE(state->is_empty());
	REQUIRE(sender.send(42).is_err());
}

TEST_CASE("Streaming channel: closing receiver releases queued values", "[distributed][streaming_channel]") {
	auto ch_pair_ = create_unbounded_channel<std::shared_ptr<int>>();
	auto sender = std::move(ch_pair_.first);
	auto receiver = std::move(ch_pair_.second);
	auto state = sender.state();
	auto payload = std::make_shared<int>(42);
	std::weak_ptr<int> payload_lifetime = payload;

	REQUIRE(sender.send(std::move(payload)).is_ok());
	REQUIRE_FALSE(payload_lifetime.expired());

	receiver.close();
	receiver.close();

	REQUIRE(state->is_closed());
	REQUIRE(state->is_empty());
	REQUIRE(payload_lifetime.expired());
	REQUIRE(sender.send(std::make_shared<int>(43)).is_err());
}

TEST_CASE("Streaming channel: moving receiver transfers channel ownership", "[distributed][streaming_channel]") {
	auto ch_pair_ = create_unbounded_channel<int>();
	auto sender = std::move(ch_pair_.first);
	auto receiver = std::move(ch_pair_.second);
	auto moved_receiver = std::move(receiver);

	REQUIRE(sender.send(42).is_ok());
	auto item = moved_receiver.recv();
	REQUIRE(item.first);
	REQUIRE(item.second == 42);
}

TEST_CASE("Streaming channel: receiver move assignment closes replaced channel", "[distributed][streaming_channel]") {
	auto old_pair = create_unbounded_channel<int>();
	auto old_sender = std::move(old_pair.first);
	auto old_receiver = std::move(old_pair.second);
	auto replacement_pair = create_unbounded_channel<int>();
	auto replacement_sender = std::move(replacement_pair.first);
	auto replacement_receiver = std::move(replacement_pair.second);

	old_receiver = std::move(replacement_receiver);

	REQUIRE(old_sender.send(1).is_err());
	REQUIRE(replacement_sender.send(2).is_ok());
	auto item = old_receiver.recv();
	REQUIRE(item.first);
	REQUIRE(item.second == 2);
}

TEST_CASE("Streaming channel: dropping plan result stream disconnects result receiver",
          "[distributed][streaming_channel]") {
	auto ch_pair_ = create_unbounded_channel<MaterializedOutput>();
	auto sender = std::move(ch_pair_.first);
	auto receiver = std::move(ch_pair_.second);

	{ PlanResultStream stream(nullptr, std::move(receiver)); }

	REQUIRE(sender.send(MaterializedOutput()).is_err());
}

TEST_CASE("Streaming channel: receiver close races safely with active sender", "[distributed][streaming_channel]") {
	for (idx_t iteration = 0; iteration < 64; iteration++) {
		auto ch_pair_ = create_unbounded_channel<int>();
		auto sender = std::move(ch_pair_.first);
		auto receiver = std::move(ch_pair_.second);
		auto state = sender.state();
		std::atomic<bool> start {false};

		std::thread producer([concurrent_sender = sender.clone(), &start]() mutable {
			while (!start.load(std::memory_order_acquire)) {
				std::this_thread::yield();
			}
			for (idx_t value = 0; value < 128; value++) {
				if (concurrent_sender.send(static_cast<int>(value)).is_err()) {
					break;
				}
			}
		});

		start.store(true, std::memory_order_release);
		receiver.close();
		producer.join();

		REQUIRE(state->is_closed());
		REQUIRE(state->is_empty());
		REQUIRE(sender.send(42).is_err());
	}
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
