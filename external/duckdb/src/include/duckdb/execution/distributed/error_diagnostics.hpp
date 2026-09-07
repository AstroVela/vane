// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace duckdb {
namespace distributed {

static constexpr size_t MAX_DIAGNOSTIC_ESCAPE_BYTES = 10;

inline size_t DiagnosticEscapeSize(std::string_view text, size_t offset) {
	if (offset >= text.size() || text[offset] != '\\' || text.size() - offset < 2) {
		return 0;
	}
	const size_t size = text[offset + 1] == 'x' ? 4 : text[offset + 1] == 'u' ? 6 : text[offset + 1] == 'U' ? 10 : 0;
	if (!size || text.size() - offset < size) {
		return 0;
	}
	for (size_t i = 2; i < size; i++) {
		const auto byte = text[offset + i];
		if (!((byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f') || (byte >= 'A' && byte <= 'F'))) {
			return 0;
		}
	}
	return size;
}

inline size_t DiagnosticTextBoundary(std::string_view text, size_t offset, bool keep_suffix) {
	offset = std::min(offset, text.size());
	while (offset && offset < text.size() && (static_cast<unsigned char>(text[offset]) & 0xC0U) == 0x80U) {
		if (keep_suffix) {
			offset++;
		} else {
			offset--;
		}
	}
	// Normalization emits ASCII escape tokens for NULs and Python surrogates.
	// Recognize them again when already-normalized text is bounded a second
	// time. At most nine bytes of lookbehind locate any crossing token.
	const auto first = offset > MAX_DIAGNOSTIC_ESCAPE_BYTES - 1 ? offset - (MAX_DIAGNOSTIC_ESCAPE_BYTES - 1) : 0;
	for (auto start = first; start < offset; start++) {
		const auto size = DiagnosticEscapeSize(text, start);
		if (size && size > offset - start) {
			return keep_suffix ? start + size : start;
		}
	}
	return offset;
}

// Diagnostic formatting must never make retaining the primary error depend on
// the size of a traceback, a cleanup failure, or an enclosing error message.
inline std::string BoundDiagnosticText(std::string_view text, size_t max_bytes) {
	std::string result;
	result.reserve(std::min(text.size(), max_bytes));
	size_t offset = 0;
	while (offset < text.size()) {
		const auto first = static_cast<unsigned char>(text[offset]);
		if (first == 0) {
			if (result.size() + 4 > max_bytes) {
				break;
			}
			result += "\\x00";
			offset++;
			continue;
		}
		const auto escape_size = DiagnosticEscapeSize(text, offset);
		if (escape_size) {
			if (escape_size > max_bytes - result.size()) {
				break;
			}
			result.append(text.data() + offset, escape_size);
			offset += escape_size;
			continue;
		}
		size_t width = first < 0x80                     ? 1
		               : first >= 0xC2 && first <= 0xDF ? 2
		               : first >= 0xE0 && first <= 0xEF ? 3
		               : first >= 0xF0 && first <= 0xF4 ? 4
		                                                : 0;
		bool valid = width && offset + width <= text.size();
		for (size_t i = 1; valid && i < width; i++) {
			const auto byte = static_cast<unsigned char>(text[offset + i]);
			valid = (byte & 0xC0U) == 0x80U;
		}
		if (valid && width >= 3) {
			const auto second = static_cast<unsigned char>(text[offset + 1]);
			valid = !(first == 0xE0 && second < 0xA0) && !(first == 0xED && second >= 0xA0) &&
			        !(first == 0xF0 && second < 0x90) && !(first == 0xF4 && second >= 0x90);
		}
		const size_t output_width = valid ? width : 3;
		if (result.size() + output_width > max_bytes) {
			break;
		}
		if (valid) {
			result.append(text.data() + offset, width);
			offset += width;
		} else {
			result += "\xEF\xBF\xBD";
			offset++;
		}
	}
	if (offset < text.size() && max_bytes >= 3) {
		const auto length = DiagnosticTextBoundary(result, std::min(result.size(), max_bytes - 3), false);
		result.resize(length);
		result += "...";
	}
	return result;
}

inline std::string BoundDiagnosticCString(const char *text, size_t max_bytes) {
	if (!text) {
		return BoundDiagnosticText("unknown error", max_bytes);
	}
	size_t length = 0;
	while (length < max_bytes + MAX_DIAGNOSTIC_ESCAPE_BYTES && text[length]) {
		length++;
	}
	return BoundDiagnosticText(std::string_view(text, length), max_bytes);
}

class ErrorDiagnostic {
public:
	static constexpr size_t MAX_TYPE_BYTES = 128;
	static constexpr size_t MAX_MESSAGE_BYTES = 1918;
	static constexpr size_t MAX_TRACEBACK_BYTES = 768;
	static constexpr size_t MAX_CAUSE_BYTES = 512;
	static constexpr size_t MAX_TRACEBACK_FRAMES = 8;
	static constexpr size_t MAX_CAUSES = 4;

	ErrorDiagnostic(std::string_view type, std::string_view message, std::string_view traceback = {},
	                std::string_view causes = {})
	    : type_(BoundDiagnosticText(type, MAX_TYPE_BYTES)), message_(BoundDiagnosticText(message, MAX_MESSAGE_BYTES)),
	      traceback_(BoundDiagnosticText(traceback, MAX_TRACEBACK_BYTES)),
	      causes_(BoundDiagnosticText(causes, MAX_CAUSE_BYTES)) {
	}

	std::string Summary() const {
		return type_.empty() ? message_ : type_ + ": " + message_;
	}
	const std::string &Traceback() const {
		return traceback_;
	}
	const std::string &Causes() const {
		return causes_;
	}

private:
	std::string type_;
	std::string message_;
	std::string traceback_;
	std::string causes_;
};

class ErrorDiagnostics {
public:
	static constexpr size_t MAX_DETAILS = 16;
	static constexpr size_t MAX_DETAIL_BYTES = 4096;
	static constexpr size_t MAX_LABEL_BYTES = 256;
	static constexpr size_t MAX_TOTAL_BYTES = 65536;

	static std::string BoundDetailText(std::string_view text, size_t max_bytes = MAX_DETAIL_BYTES) {
		if (max_bytes < 3) {
			return BoundDiagnosticText(text, max_bytes);
		}
		// NUL escaping and invalid UTF-8 replacement can expand raw input.
		// Normalize at most one budget of input from each end, then choose the
		// edges by their output-byte sizes. Never materialize the full input.
		const auto slice_size = std::min(text.size(), max_bytes);
		auto prefix =
		    BoundDiagnosticText(text.substr(0, DiagnosticTextBoundary(text, slice_size, false)), slice_size * 4);
		if (slice_size == text.size() && prefix.size() <= max_bytes) {
			return prefix;
		}
		const auto suffix_start = DiagnosticTextBoundary(text, text.size() - slice_size, true);
		auto suffix = BoundDiagnosticText(text.substr(suffix_start), slice_size * 4);
		const auto head_budget = (max_bytes - 3) / 2;
		const auto head_end = DiagnosticTextBoundary(prefix, std::min(prefix.size(), head_budget), false);
		// Space left by a wide character or escape at the head is available to
		// the tail, including when only one complete token fits the budget.
		const auto tail_budget = max_bytes - 3 - head_end;
		const auto tail_start =
		    DiagnosticTextBoundary(suffix, suffix.size() > tail_budget ? suffix.size() - tail_budget : 0, true);
		prefix.resize(head_end);
		return prefix + "..." + suffix.substr(tail_start);
	}

	static ErrorDiagnostics FromDiagnostic(ErrorDiagnostic diagnostic) {
		ErrorDiagnostics result;
		result.entries_.push_back({{}, std::move(diagnostic)});
		result.count_ = 1;
		return result;
	}

	static ErrorDiagnostics FromText(std::string_view message, std::string_view type = {}) {
		// Opaque native/status text can contain a formatted traceback before the
		// actual failure. Preserve both ends when it enters structured storage.
		return FromDiagnostic(ErrorDiagnostic(type, BoundDetailText(message, ErrorDiagnostic::MAX_MESSAGE_BYTES)));
	}

	void Add(std::string_view label, const ErrorDiagnostics &error) {
		Merge(label, error, false);
	}
	void Add(std::string_view label, std::string_view message) {
		Add(label, FromText(message));
	}
	void AddPrimary(std::string_view label, const ErrorDiagnostics &error) {
		Merge(label, error, true);
	}

	ErrorDiagnostics WithContext(std::string_view context) const {
		ErrorDiagnostics result;
		result.Add(context, *this);
		return result;
	}

	static ErrorDiagnostics FormatDetail(std::string_view label, const ErrorDiagnostics &error) {
		return error.WithContext(label);
	}
	static ErrorDiagnostics FormatDetail(std::string_view label, std::string_view message) {
		return FromText(message).WithContext(label);
	}

	explicit operator bool() const {
		return count_ != 0;
	}
	size_t Count() const {
		return count_;
	}

	std::string AppendTo(std::string_view context = {}) const {
		std::string result = BoundDiagnosticText(context, MAX_DETAIL_BYTES);
		// Render every retained summary before optional tracebacks. Rewrapping
		// uses the entries themselves, never this rendered representation.
		for (const auto &entry : entries_) {
			if (!result.empty()) {
				result += "; ";
			}
			if (!entry.label.empty()) {
				result += entry.label + ": ";
			}
			result += entry.diagnostic.Summary();
			if (!entry.diagnostic.Causes().empty()) {
				result += " [" + entry.diagnostic.Causes() + "]";
			}
		}
		if (count_ > entries_.size()) {
			result += "; additional " + std::to_string(count_ - entries_.size()) + " error(s) omitted";
		}
		for (const auto &entry : entries_) {
			if (!entry.diagnostic.Traceback().empty()) {
				result += "\nTraceback [" + entry.label + "]:\n" + entry.diagnostic.Traceback();
			}
		}
		return BoundDiagnosticText(result, MAX_TOTAL_BYTES);
	}

private:
	struct Entry {
		std::string label;
		ErrorDiagnostic diagnostic;
	};

	void Merge(std::string_view label, const ErrorDiagnostics &error, bool primary) {
		std::vector<Entry> entries;
		entries.reserve(MAX_DETAILS);
		auto append = [&](const ErrorDiagnostics &source, std::string_view context) {
			for (const auto &entry : source.entries_) {
				if (entries.size() == MAX_DETAILS) {
					break;
				}
				auto combined = BoundDiagnosticText(context, MAX_LABEL_BYTES);
				if (!entry.label.empty()) {
					if (!combined.empty()) {
						combined += ": ";
					}
					combined += entry.label;
				}
				entries.push_back({BoundDiagnosticText(combined, MAX_LABEL_BYTES), entry.diagnostic});
			}
		};
		if (primary) {
			append(error, label);
			append(*this, {});
		} else {
			append(*this, {});
			append(error, label);
		}
		count_ += std::min(error.count_, std::numeric_limits<size_t>::max() - count_);
		entries_ = std::move(entries);
	}

	size_t count_ = 0;
	std::vector<Entry> entries_;
};

} // namespace distributed
} // namespace duckdb
