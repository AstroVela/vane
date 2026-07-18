// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file channel.hpp
 * @brief Channel utilities for async communication in the distributed framework
 *
 * Translated from DuckDB's duckdb-distributed/src/utils/channel.rs to C++20.
 * Provides wrappers around async channels for task communication.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <type_traits>
#include <utility>

#include "duckdb/execution/distributed/common_types.hpp"

namespace duckdb {
namespace distributed {

//------------------------------------------------------------------------------
// Forward Declarations
//------------------------------------------------------------------------------

template <typename T>
class Sender;
template <typename T>
class Receiver;
template <typename T>
class UnboundedSender;
template <typename T>
class UnboundedReceiver;

//------------------------------------------------------------------------------
// Channel State
//------------------------------------------------------------------------------

/// Shared state for bounded channels
template <typename T>
class ChannelState : public std::enable_shared_from_this<ChannelState<T>> {
public:
	explicit ChannelState(size_t capacity) : capacity_(capacity), closed_(false) {
	}

	/// Send a value into the channel
	bool send(T value) {
		std::unique_lock<std::mutex> lock(mutex_);

		// Wait until there's room or channel is closed.
		not_full_.wait(lock, [this] {
			return queue_.size() < capacity_ || closed_ || receiver_count_.load(std::memory_order_relaxed) == 0;
		});

		if (closed_ || receiver_count_.load(std::memory_order_relaxed) == 0) {
			return false;
		}

		queue_.push(std::move(value));
		not_empty_.notify_one();
		return true;
	}

	// Sender count management (track number of Sender<> instances)
	void increment_sender_count() {
		sender_count_.fetch_add(1, std::memory_order_relaxed);
	}

	size_t decrement_sender_count() {
		auto val = sender_count_.fetch_sub(1, std::memory_order_acq_rel);
		// fetch_sub returns previous value
		auto remaining = (val > 0) ? val - 1 : 0;
		return remaining;
	}
	// Receiver count management (track number of Receiver<> instances)
	void increment_receiver_count() {
		receiver_count_.fetch_add(1, std::memory_order_relaxed);
	}

	size_t decrement_receiver_count() {
		auto val = receiver_count_.fetch_sub(1, std::memory_order_acq_rel);
		auto remaining = (val > 0) ? val - 1 : 0;
		return remaining;
	}

	/// Receive a value from the channel
	std::pair<bool, T> recv() {
		std::unique_lock<std::mutex> lock(mutex_);

		// Wait until there's data or channel is closed and empty
		not_empty_.wait(lock, [this] { return !queue_.empty() || closed_; });

		if (queue_.empty()) {
			return std::make_pair(false, T());
		}

		T value = std::move(queue_.front());
		queue_.pop();
		not_full_.notify_one();
		return std::make_pair(true, std::move(value));
	}

	/// Try to receive without blocking
	std::pair<bool, T> try_recv() {
		std::lock_guard<std::mutex> lock(mutex_);
		if (queue_.empty()) {
			return std::make_pair(false, T());
		}
		T value = std::move(queue_.front());
		queue_.pop();
		not_full_.notify_one();
		return std::make_pair(true, std::move(value));
	}

	/// Close the channel
	void close() {
		std::lock_guard<std::mutex> lock(mutex_);
		closed_ = true;
		not_empty_.notify_all();
		not_full_.notify_all();
	}

	/// Check if channel is closed
	bool is_closed() const {
		std::lock_guard<std::mutex> lock(mutex_);
		return closed_;
	}

	/// Get number of items in queue
	size_t len() const {
		std::lock_guard<std::mutex> lock(mutex_);
		return queue_.size();
	}

private:
	mutable std::mutex mutex_;
	std::condition_variable not_empty_;
	std::condition_variable not_full_;
	std::queue<T> queue_;
	size_t capacity_;
	bool closed_;
	// Count of active Sender<> objects (only senders increment this).
	std::atomic<size_t> sender_count_ {0};
	// Count of active Receiver<> objects. When this reaches zero, send() fails.
	std::atomic<size_t> receiver_count_ {0};
};

//------------------------------------------------------------------------------
// Bounded Channel Sender
//------------------------------------------------------------------------------

/// Bounded channel sender (Rust: Sender<T> = tokio::sync::mpsc::Sender<T>)
template <typename T>
class Sender {
public:
	Sender() = default;

	// Public constructor used by callers: increment sender counter on creation
	explicit Sender(std::shared_ptr<ChannelState<T>> state, bool increment_count) : state_(std::move(state)) {
		if (state_ && increment_count) {
			state_->increment_sender_count();
		}
	}
	// Move constructor - transfer ownership without losing sender accounting
	Sender(Sender &&other) noexcept : state_(std::move(other.state_)) {
		// Moving a Sender transfers ownership and should NOT change the
		// logical sender count.
	}

	// Disable copy operations to keep sender-count management explicit.
	Sender(const Sender &) = delete;
	Sender &operator=(const Sender &) = delete;
	Sender &operator=(Sender &&) = delete;

	~Sender() noexcept {
		// Decrement sender counter and close the channel if this was the
		// last Sender object. Use the sender-specific counter to avoid the
		// ambiguity caused by Receiver holding a shared_ptr to the state.
		try {
			if (state_) {
				auto remaining = state_->decrement_sender_count();
				if (remaining == 0) {
					state_->close();
				}
			}
		} catch (...) {
			// Never propagate exceptions from destructors.
		}
	}

	/// Send a value (blocking)
	DuckDBResult<void> send(T value) {
		if (!state_ || !state_->send(std::move(value))) {
			return DuckDBResult<void>::err(DuckDBError::internal_error("Channel closed or send failed"));
		}
		return DuckDBResult<void>::ok();
	}

	/// Explicitly close the channel
	void close() {
		if (state_)
			state_->close();
	}

	/// Get the underlying state (for coroutine integration)
	std::shared_ptr<ChannelState<T>> state() const {
		return state_;
	}

private:
	std::shared_ptr<ChannelState<T>> state_;
};

//------------------------------------------------------------------------------
// Bounded Channel Receiver
//------------------------------------------------------------------------------

/// Bounded channel receiver (Rust: Receiver<T> = tokio::sync::mpsc::Receiver<T>)
template <typename T>
class Receiver {
public:
	Receiver() = default;

	explicit Receiver(std::shared_ptr<ChannelState<T>> state) : state_(std::move(state)) {
		if (state_) {
			state_->increment_receiver_count();
		}
	}

	// Move-only receiver semantics (tokio::mpsc-style single receiver).
	Receiver(const Receiver &) = delete;
	Receiver &operator=(const Receiver &) = delete;

	Receiver(Receiver &&other) noexcept : state_(std::move(other.state_)) {
	}

	Receiver &operator=(Receiver &&other) noexcept {
		if (this != &other) {
			release_state();
			state_ = std::move(other.state_);
		}
		return *this;
	}

	~Receiver() {
		release_state();
	}

	/// Receive a value (blocking)
	std::pair<bool, T> recv() {
		if (!state_) {
			return std::make_pair(false, T());
		}
		auto res = state_->recv();
		return res;
	}

	/// Try to receive without blocking
	std::pair<bool, T> try_recv() {
		if (!state_) {
			return std::make_pair(false, T());
		}
		return state_->try_recv();
	}

	/// Get the underlying state
	std::shared_ptr<ChannelState<T>> state() const {
		return state_;
	}

private:
	void release_state() {
		if (!state_) {
			return;
		}
		auto remaining = state_->decrement_receiver_count();
		if (remaining == 0) {
			state_->close();
		}
		state_.reset();
	}

	std::shared_ptr<ChannelState<T>> state_;
};

/// Create a bounded channel (Rust: create_channel<T>(capacity))
template <typename T>
std::pair<Sender<T>, Receiver<T>> create_channel(size_t capacity) {
	auto state = std::make_shared<ChannelState<T>>(capacity);
	return {Sender<T>(state, /*increment_count=*/true), Receiver<T>(state)};
}

//------------------------------------------------------------------------------
// Unbounded Channel State
//------------------------------------------------------------------------------

/// Shared state for unbounded channels
template <typename T>
class UnboundedChannelState : public std::enable_shared_from_this<UnboundedChannelState<T>> {
public:
	UnboundedChannelState() : closed_(false) {
	}

	/// Send a value into the channel (always succeeds unless closed)
	bool send(T value) {
		std::lock_guard<std::mutex> lock(mutex_);
		if (closed_) {
			return false;
		}
		queue_.push(std::move(value));
		not_empty_.notify_one();
		return true;
	}

	// Sender count management (track number of UnboundedSender<> instances)
	void increment_sender_count() {
		sender_count_.fetch_add(1, std::memory_order_relaxed);
	}

	size_t decrement_sender_count() {
		auto val = sender_count_.fetch_sub(1, std::memory_order_acq_rel);
		auto remaining = (val > 0) ? val - 1 : 0;
		return remaining;
	}

	size_t sender_count() const {
		return sender_count_.load(std::memory_order_relaxed);
	}

	/// Receive a value from the channel
	std::pair<bool, T> recv() {
		std::unique_lock<std::mutex> lock(mutex_);

		not_empty_.wait(lock, [this] { return !queue_.empty() || closed_; });

		if (queue_.empty()) {
			return std::make_pair(false, T());
		}

		T value = std::move(queue_.front());
		queue_.pop();
		return std::make_pair(true, std::move(value));
	}

	/// Try to receive without blocking
	std::pair<bool, T> try_recv() {
		std::lock_guard<std::mutex> lock(mutex_);
		if (queue_.empty()) {
			return std::make_pair(false, T());
		}
		T value = std::move(queue_.front());
		queue_.pop();
		return std::make_pair(true, std::move(value));
	}

	/// Close the channel
	void close() {
		std::lock_guard<std::mutex> lock(mutex_);
		closed_ = true;
		not_empty_.notify_all();
	}

	/// Check if closed
	bool is_closed() const {
		std::lock_guard<std::mutex> lock(mutex_);
		return closed_;
	}

	/// Check if empty
	bool is_empty() const {
		std::lock_guard<std::mutex> lock(mutex_);
		return queue_.empty();
	}

private:
	mutable std::mutex mutex_;
	std::condition_variable not_empty_;
	std::queue<T> queue_;
	bool closed_;
	// Track number of UnboundedSender holders so we can close on last sender
	std::atomic<size_t> sender_count_ {0};
};

//------------------------------------------------------------------------------
// Unbounded Channel Sender
//------------------------------------------------------------------------------

/// Unbounded channel sender (Rust: UnboundedSender<T>)
template <typename T>
class UnboundedSender {
public:
	UnboundedSender() = default;

	explicit UnboundedSender(std::shared_ptr<UnboundedChannelState<T>> state) : state_(std::move(state)) {
		if (state_) {
			state_->increment_sender_count();
		}
	}
	UnboundedSender(std::shared_ptr<UnboundedChannelState<T>> state, bool increment_count) : state_(std::move(state)) {
		if (state_ && increment_count) {
			state_->increment_sender_count();
		}
	}
	// Move constructor - transfer ownership without changing sender count.
	UnboundedSender(UnboundedSender &&other) noexcept : state_(std::move(other.state_)) {
	}
	UnboundedSender(const UnboundedSender &) = delete;
	UnboundedSender &operator=(const UnboundedSender &) = delete;
	UnboundedSender &operator=(UnboundedSender &&) = delete;

	~UnboundedSender() noexcept {
		try {
			if (state_) {
				auto remaining = state_->decrement_sender_count();
				if (remaining == 0) {
					state_->close();
				}
			}
		} catch (...) {
			// Never propagate exceptions from destructors — especially during
			// stack unwinding this would call std::terminate().
		}
	}

	/// Send a value
	DuckDBResult<void> send(T value) {
		if (!state_ || !state_->send(std::move(value))) {
			return DuckDBResult<void>::err(DuckDBError::internal_error("Failed to send task to scheduler"));
		}
		return DuckDBResult<void>::ok();
	}

	/// Clone the sender
	UnboundedSender<T> clone() const {
		// Use constructor variant to increment sender count for the clone.
		return UnboundedSender<T>(state_, /*increment_count=*/true);
	}

	std::shared_ptr<UnboundedChannelState<T>> state() const {
		return state_;
	}

private:
	std::shared_ptr<UnboundedChannelState<T>> state_;
};

//------------------------------------------------------------------------------
// Unbounded Channel Receiver
//------------------------------------------------------------------------------

/// Unbounded channel receiver (Rust: UnboundedReceiver<T>)
template <typename T>
class UnboundedReceiver {
public:
	UnboundedReceiver() = default;

	explicit UnboundedReceiver(std::shared_ptr<UnboundedChannelState<T>> state) : state_(std::move(state)) {
	}

	/// Receive a value (blocking)
	std::pair<bool, T> recv() {
		if (!state_) {
			return std::make_pair(false, T());
		}
		return state_->recv();
	}

	/// Try to receive without blocking
	std::pair<bool, T> try_recv() {
		if (!state_) {
			return std::make_pair(false, T());
		}
		return state_->try_recv();
	}

	/// Check if channel is closed and empty (disconnected)
	bool is_disconnected() const {
		if (!state_) {
			return true;
		}
		return state_->is_closed() && state_->is_empty();
	}

private:
	std::shared_ptr<UnboundedChannelState<T>> state_;
};

/// Create an unbounded channel (Rust: create_unbounded_channel<T>())
template <typename T>
std::pair<UnboundedSender<T>, UnboundedReceiver<T>> create_unbounded_channel() {
	auto state = std::make_shared<UnboundedChannelState<T>>();
	return {UnboundedSender<T>(state, /*increment_count=*/true), UnboundedReceiver<T>(state)};
}

// Stream concept (C++20 concept removed for C++11 compatibility)
// Types used as streams should provide:
//   - std::pair<bool, value_type> poll_next()

// 简化的流类型别名（对应 Rust 的 BoxStream）
template <typename T>
class BoxStream {
public:
	using value_type = T;

	template <typename StreamType>
	BoxStream(StreamType &&stream)
	    : stream_(
	          std::unique_ptr<StreamModel<StreamType>>(new StreamModel<StreamType>(std::forward<StreamType>(stream)))) {
	}

	std::pair<bool, T> poll_next() {
		return stream_->poll_next();
	}

	std::pair<bool, T> try_poll_next() {
		return stream_->try_poll_next();
	}

	bool is_exhausted() const {
		return stream_->is_exhausted();
	}

private:
	struct StreamConcept {
		virtual ~StreamConcept() {
		}
		virtual std::pair<bool, T> poll_next() = 0;
		virtual std::pair<bool, T> try_poll_next() = 0;
		virtual bool is_exhausted() const = 0;
	};

	template <typename StreamType>
	struct StreamModel : StreamConcept {
		StreamModel(StreamType stream) : stream_(std::move(stream)) {
		}

		std::pair<bool, T> poll_next() override {
			return stream_.poll_next();
		}

		std::pair<bool, T> try_poll_next() override {
			return try_poll_next_impl(static_cast<StreamType *>(nullptr));
		}

		bool is_exhausted() const override {
			return is_exhausted_impl(static_cast<const StreamType *>(nullptr));
		}

	private:
		template <typename S>
		typename std::enable_if<std::is_same<decltype(std::declval<S &>().try_poll_next()), std::pair<bool, T>>::value,
		                        std::pair<bool, T>>::type
		try_poll_next_impl(S *) {
			return stream_.try_poll_next();
		}

		std::pair<bool, T> try_poll_next_impl(...) {
			return std::make_pair(false, T());
		}

		template <typename S>
		typename std::enable_if<std::is_same<decltype(std::declval<const S>().is_exhausted()), bool>::value, bool>::type
		is_exhausted_impl(const S *) const {
			return stream_.is_exhausted();
		}
		bool is_exhausted_impl(...) const {
			return false;
		}

		StreamType stream_;
	};

	std::unique_ptr<StreamConcept> stream_;
};

// 工具函数：将流装箱（对应 .boxed() 方法）
template <typename T, typename StreamType>
std::unique_ptr<BoxStream<T>> boxed(StreamType &&stream) {
	return std::unique_ptr<BoxStream<T>>(new BoxStream<T>(std::forward<StreamType>(stream)));
}

} // namespace distributed
} // namespace duckdb
