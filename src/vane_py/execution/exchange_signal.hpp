// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "direct_exchange.hpp"

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>

namespace duckdb {
namespace vane_execution {

// One waiter per signal. Retain a notification that arrives between checking
// channel state and entering Wait; condition-variable notifications alone do not.
class ExchangeSignal : public std::enable_shared_from_this<ExchangeSignal> {
public:
	void Notify() {
		{
			std::lock_guard<std::mutex> guard(lock);
			notified = true;
		}
		changed.notify_one();
	}
	void Wait(std::chrono::milliseconds timeout = std::chrono::milliseconds(10)) {
		std::unique_lock<std::mutex> guard(lock);
		changed.wait_for(guard, timeout, [&]() { return notified; });
		notified = false;
	}
	ExchangeWakeup Wakeup() {
		std::weak_ptr<ExchangeSignal> weak = shared_from_this();
		return [weak]() {
			if (auto signal = weak.lock()) {
				signal->Notify();
			}
		};
	}

private:
	std::mutex lock;
	std::condition_variable changed;
	bool notified = false;
};

} // namespace vane_execution
} // namespace duckdb
