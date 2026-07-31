// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/execution/vllm_inflight_counter.hpp"

#include <atomic>
#include <thread>
#include <vector>

using namespace duckdb;

TEST_CASE("vLLM inflight counter tracks submissions and completions", "[execution][vllm][concurrency]") {
	VLLMInflightCounter counter;

	REQUIRE(counter.Count() == 0);
	counter.RecordSubmission(5);
	REQUIRE(counter.Count() == 5);
	counter.RecordCompletion(2);
	REQUIRE(counter.Count() == 3);
	counter.RecordCompletion(3);
	REQUIRE(counter.Count() == 0);
}

TEST_CASE("vLLM inflight counter rejects invalid transitions", "[execution][vllm][concurrency]") {
	VLLMInflightCounter counter;

	counter.RecordSubmission(2);
	REQUIRE_THROWS_AS(counter.RecordCompletion(3), InternalException);
	REQUIRE(counter.Count() == 2);

	counter.RecordCompletion(2);
	counter.RecordSubmission(NumericLimits<idx_t>::Maximum());
	REQUIRE_THROWS_AS(counter.RecordSubmission(1), InternalException);
	REQUIRE(counter.Count() == NumericLimits<idx_t>::Maximum());
	counter.RecordCompletion(NumericLimits<idx_t>::Maximum());
	REQUIRE(counter.Count() == 0);
}

TEST_CASE("vLLM inflight counter atomically rejects concurrent over-completion", "[execution][vllm][concurrency]") {
	VLLMInflightCounter counter;
	counter.RecordSubmission(1);

	std::atomic<idx_t> successful_completions {0};
	std::atomic<idx_t> rejected_completions {0};
	auto complete = [&]() {
		try {
			counter.RecordCompletion(1);
			successful_completions.fetch_add(1);
		} catch (const InternalException &) {
			rejected_completions.fetch_add(1);
		}
	};

	std::thread first(complete);
	std::thread second(complete);
	first.join();
	second.join();

	REQUIRE(successful_completions.load() == 1);
	REQUIRE(rejected_completions.load() == 1);
	REQUIRE(counter.Count() == 0);
}

TEST_CASE("vLLM inflight counter remains consistent under concurrent updates", "[execution][vllm][concurrency]") {
	VLLMInflightCounter counter;
	constexpr idx_t thread_count = 8;
	constexpr idx_t iterations = 10000;
	std::atomic<bool> start {false};
	std::atomic<bool> first_observation_done {false};
	std::atomic<idx_t> first_submissions {0};
	std::atomic<idx_t> active_workers {thread_count};
	std::atomic<idx_t> failures {0};
	std::atomic<idx_t> observations {0};
	std::vector<std::thread> threads;

	for (idx_t thread_idx = 0; thread_idx < thread_count; thread_idx++) {
		threads.emplace_back([&]() {
			while (!start.load()) {
			}
			try {
				counter.RecordSubmission(1);
			} catch (const InternalException &) {
				failures.fetch_add(1);
			}
			first_submissions.fetch_add(1);
			while (!first_observation_done.load()) {
			}
			try {
				counter.RecordCompletion(1);
			} catch (const InternalException &) {
				failures.fetch_add(1);
			}
			for (idx_t iteration = 1; iteration < iterations; iteration++) {
				try {
					counter.RecordSubmission(1);
					counter.RecordCompletion(1);
				} catch (const InternalException &) {
					failures.fetch_add(1);
				}
			}
			active_workers.fetch_sub(1);
		});
	}

	std::thread observer([&]() {
		while (!start.load()) {
		}
		while (first_submissions.load() < thread_count) {
		}
		try {
			if (counter.Count() != thread_count) {
				failures.fetch_add(1);
			}
		} catch (const InternalException &) {
			failures.fetch_add(1);
		}
		observations.fetch_add(1);
		first_observation_done.store(true);
		while (active_workers.load() > 0) {
			try {
				if (counter.Count() > thread_count) {
					failures.fetch_add(1);
				}
			} catch (const InternalException &) {
				failures.fetch_add(1);
			}
			observations.fetch_add(1);
		}
	});

	start.store(true);
	for (auto &thread : threads) {
		thread.join();
	}
	observer.join();

	REQUIRE(failures.load() == 0);
	REQUIRE(observations.load() > 0);
	REQUIRE(counter.Count() == 0);
}
