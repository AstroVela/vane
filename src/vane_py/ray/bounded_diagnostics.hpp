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
	static constexpr size_t MAX_LABEL_BYTES = 256;

	static std::string BoundDetailText(const std::string &text) {
		return BoundText(text, MAX_DETAIL_BYTES);
	}

	void Add(const std::string &label, const char *detail) {
		count_++;
		if (details_.size() >= MAX_DETAILS) {
			return;
		}
		auto bounded_label = BoundText(label, MAX_LABEL_BYTES);
		if (!detail) {
			details_.push_back(std::move(bounded_label) + ": unknown error");
			return;
		}
		size_t length = 0;
		while (length <= MAX_DETAIL_BYTES && detail[length] != '\0') {
			length++;
		}
		if (length > MAX_DETAIL_BYTES) {
			details_.push_back(std::move(bounded_label) + ": error detail exceeds 4096 bytes and was omitted");
			return;
		}
		details_.push_back(std::move(bounded_label) + ": " + std::string(detail, length));
	}

	static std::string FormatDetail(const std::string &label, const char *detail) {
		BoundedErrorDetails errors;
		errors.Add(label, detail);
		return errors.details_.front();
	}

	explicit operator bool() const {
		return count_ != 0;
	}

	size_t Count() const {
		return count_;
	}

	std::string AppendTo(std::string message) const {
		message = BoundText(message, MAX_DETAIL_BYTES);
		for (const auto &detail : details_) {
			message += "; " + detail;
		}
		if (count_ > details_.size()) {
			message += "; additional " + std::to_string(count_ - details_.size()) + " error(s) omitted";
		}
		return message;
	}

private:
	static std::string BoundText(const std::string &text, size_t max_bytes) {
		if (text.size() <= max_bytes) {
			return text;
		}
		static constexpr const char *OMISSION = "...";
		static constexpr size_t OMISSION_BYTES = 3;
		const auto remaining = max_bytes - OMISSION_BYTES;
		auto prefix_bytes = remaining / 2;
		auto suffix_offset = text.size() - (remaining - prefix_bytes);
		// Diagnostic strings are valid UTF-8. Keep both retained edges on
		// code-point boundaries so the bounded value remains safe to marshal
		// back through pybind11.
		while (prefix_bytes > 0 && prefix_bytes < text.size() &&
		       (static_cast<unsigned char>(text[prefix_bytes]) & 0xC0U) == 0x80U) {
			prefix_bytes--;
		}
		while (suffix_offset < text.size() && (static_cast<unsigned char>(text[suffix_offset]) & 0xC0U) == 0x80U) {
			suffix_offset++;
		}
		return text.substr(0, prefix_bytes) + OMISSION + text.substr(suffix_offset);
	}

	size_t count_ = 0;
	std::vector<std::string> details_;
};

} // namespace vane
