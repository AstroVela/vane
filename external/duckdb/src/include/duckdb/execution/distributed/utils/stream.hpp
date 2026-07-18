// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file stream.hpp
 * @brief Stream utilities for the distributed framework
 *
 * Translated from DuckDB's duckdb-distributed/src/utils/stream.rs.
 * Provides Stream utilities for the distributed framework.
 */

#pragma once

#include <utility>

#include "duckdb/execution/distributed/utils/channel.hpp"

namespace duckdb {
namespace distributed {

//------------------------------------------------------------------------------
// Stream Trait Interface
//------------------------------------------------------------------------------

/// Stream interface for async iteration
template <typename T>
class Stream {
public:
	virtual ~Stream() = default;

	/// Poll for next item, blocking until an item or end-of-stream is available.
	virtual std::pair<bool, T> poll_next() = 0;

	/// Poll for an item that is already available. The default implementation
	/// reports no ready item; stream types with non-blocking receivers override it.
	virtual std::pair<bool, T> try_poll_next() {
		return std::make_pair(false, T());
	}

	/// Check if stream is exhausted
	virtual bool is_exhausted() const = 0;
};

/// ChannelStream - Stream backed by a channel receiver
template <typename T>
class ChannelStream : public Stream<T> {
public:
	explicit ChannelStream(Receiver<T> receiver) : receiver_(std::move(receiver)), exhausted_(false) {
	}

	std::pair<bool, T> poll_next() override {
		auto result = receiver_.recv();
		if (result.first) {
			return result;
		}
		// Channel closed and empty
		exhausted_ = true;
		return std::make_pair(false, T());
	}

	std::pair<bool, T> try_poll_next() override {
		auto result = receiver_.try_recv();
		if (result.first) {
			return result;
		}
		auto state = receiver_.state();
		if (state && state->is_closed() && state->len() == 0) {
			exhausted_ = true;
		}
		return std::make_pair(false, T());
	}

	bool is_exhausted() const override {
		return exhausted_;
	}

private:
	Receiver<T> receiver_;
	bool exhausted_;
};

} // namespace distributed
} // namespace duckdb
