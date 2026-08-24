// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace vane {

class BoundedErrorDetails {
public:
	static constexpr size_t MAX_DETAILS = 16;
	static constexpr size_t MAX_DETAIL_BYTES = 4096;

	void Add(std::string label, const char *detail) {
		count_++;
		if (details_.size() >= MAX_DETAILS) {
			return;
		}
		if (!detail) {
			details_.push_back(std::move(label) + ": unknown error");
			return;
		}
		size_t length = 0;
		while (length <= MAX_DETAIL_BYTES && detail[length] != '\0') {
			length++;
		}
		if (length > MAX_DETAIL_BYTES) {
			details_.push_back(std::move(label) + ": error detail exceeds 4096 bytes and was omitted");
			return;
		}
		details_.push_back(std::move(label) + ": " + std::string(detail, length));
	}

	explicit operator bool() const {
		return count_ != 0;
	}

	size_t Count() const {
		return count_;
	}

	std::string AppendTo(std::string message) const {
		for (const auto &detail : details_) {
			message += "; " + detail;
		}
		if (count_ > details_.size()) {
			message += "; additional " + std::to_string(count_ - details_.size()) + " error(s) omitted";
		}
		return message;
	}

private:
	size_t count_ = 0;
	std::vector<std::string> details_;
};

} // namespace vane
