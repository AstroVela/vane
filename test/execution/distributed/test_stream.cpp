// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file test_stream.cpp
 * @brief Unit tests for distributed stream utilities
 *
 * Translated from DuckDB's duckdb-distributed/src/utils/stream.rs tests to C++20.
 */

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "test_common.hpp"
#include "duckdb/execution/distributed/utils/stream.hpp"

namespace duckdb {
namespace distributed {
namespace testing {

//------------------------------------------------------------------------------
// Test Helpers - Concrete Stream Implementations
//------------------------------------------------------------------------------

/**
 * @brief VectorStream - a concrete stream that yields values from a vector
 */
template <typename T>
class VectorStream : public Stream<T> {
public:
	explicit VectorStream(std::vector<T> values) : values_(std::move(values)), index_(0) {
	}

	std::pair<bool, T> poll_next() override {
		if (index_ < values_.size()) {
			return std::make_pair(true, std::move(values_[index_++]));
		}
		return std::pair<bool, T>(false, T());
	}

	bool is_exhausted() const override {
		return index_ >= values_.size();
	}

private:
	std::vector<T> values_;
	size_t index_;
};

/**
 * @brief DelayedStream - a stream that yields values with delays
 */
template <typename T>
class DelayedStream : public Stream<T> {
public:
	DelayedStream(std::vector<std::pair<T, std::chrono::milliseconds>> values_with_delays)
	    : values_(std::move(values_with_delays)), index_(0) {
	}

	std::pair<bool, T> poll_next() override {
		if (index_ < values_.size()) {
			auto &vd = values_[index_];
			std::this_thread::sleep_for(vd.second);
			return std::make_pair(true, std::move(values_[index_++].first));
		}
		return std::pair<bool, T>(false, T());
	}

	bool is_exhausted() const override {
		return index_ >= values_.size();
	}

private:
	std::vector<std::pair<T, std::chrono::milliseconds>> values_;
	size_t index_;
};

/**
 * @brief ErrorStream - a stream that throws an error after yielding some values
 */
template <typename T>
class ErrorStream : public Stream<T> {
public:
	ErrorStream(std::vector<T> values, const std::string &error_msg)
	    : values_(std::move(values)), error_msg_(error_msg), index_(0) {
	}

	std::pair<bool, T> poll_next() override {
		if (index_ < values_.size()) {
			return std::make_pair(true, std::move(values_[index_++]));
		}
		if (!error_thrown_) {
			error_thrown_ = true;
			throw std::runtime_error(error_msg_);
		}
		return std::pair<bool, T>(false, T());
	}

	bool is_exhausted() const override {
		return index_ >= values_.size() && error_thrown_;
	}

private:
	std::vector<T> values_;
	std::string error_msg_;
	size_t index_;
	bool error_thrown_ = false;
};

/// Helper to create streams
template <typename T>
std::shared_ptr<Stream<T>> make_vector_stream(std::vector<T> values) {
	return std::make_shared<VectorStream<T>>(std::move(values));
}

template <typename T>
std::shared_ptr<Stream<T>>
make_delayed_stream(std::vector<std::pair<T, std::chrono::milliseconds>> values_with_delays) {
	return std::make_shared<DelayedStream<T>>(std::move(values_with_delays));
}

template <typename T>
std::shared_ptr<Stream<T>> make_error_stream(std::vector<T> values, const std::string &error_msg) {
	return std::make_shared<ErrorStream<T>>(std::move(values), error_msg);
}

TEST_CASE("VectorStream: basic", "[distributed][stream]") {
	using namespace duckdb::distributed::testing;
	std::vector<int> values = {1, 2, 3, 4, 5};
	auto stream = make_vector_stream(values);

	std::vector<int> received;
	while (true) {
		auto item = stream->poll_next();
		if (!item.first)
			break;
		received.push_back(item.second);
	}

	REQUIRE(received == values);
	REQUIRE(stream->is_exhausted());
}

TEST_CASE("VectorStream: empty", "[distributed][stream]") {
	using namespace duckdb::distributed::testing;
	std::vector<int> values;
	auto stream = make_vector_stream(values);

	REQUIRE(stream->is_exhausted());
	auto item = stream->poll_next();
	REQUIRE(!item.first);
}

TEST_CASE("ErrorStream: throws after values", "[distributed][stream]") {
	using namespace duckdb::distributed::testing;
	std::vector<int> values = {1, 2, 3};
	auto stream = make_error_stream(values, "test_error");

	std::vector<int> received;
	bool got_error = false;

	try {
		while (true) {
			auto item_opt = stream->poll_next();
			if (!item_opt.first)
				break;
			received.push_back(item_opt.second);
		}
	} catch (const std::runtime_error &e) {
		got_error = true;
		REQUIRE(std::string(e.what()).find("test_error") != std::string::npos);
	}

	REQUIRE(received == values);
	REQUIRE(got_error);
}

TEST_CASE("DelayedStream: respects delays", "[distributed][stream][slow]") {
	using namespace duckdb::distributed::testing;
	std::vector<std::pair<int, std::chrono::milliseconds>> values_with_delays = {
	    {1, std::chrono::milliseconds(10)},
	    {2, std::chrono::milliseconds(20)},
	    {3, std::chrono::milliseconds(10)},
	};
	auto stream = make_delayed_stream(values_with_delays);

	auto start = std::chrono::steady_clock::now();
	std::vector<int> received;
	while (true) {
		auto item = stream->poll_next();
		if (!item.first)
			break;
		received.push_back(item.second);
	}
	auto elapsed =
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

	REQUIRE(received == std::vector<int>({1, 2, 3}));
	REQUIRE(elapsed >= 35);
}

} // namespace testing
} // namespace distributed
} // namespace duckdb
