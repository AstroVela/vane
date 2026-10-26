// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>

// Linked only into diagnostic copies, never a plain timing/control binary.
namespace payload_trace {
long long Now();
void Record(const char *name, long long begin);
class Span {
public:
	explicit Span(const char *name) : name(name), begin(Now()) {
	}
	~Span() {
		Record(name, begin);
	}

private:
	const char *name;
	long long begin;
};
inline std::unique_lock<std::mutex> Lock(const char *name, std::mutex &mutex) {
	Span span(name);
	return std::unique_lock<std::mutex>(mutex);
}
} // namespace payload_trace
