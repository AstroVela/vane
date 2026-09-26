// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/exception.hpp"

#include <algorithm>
#include <chrono>
#include <deque>
#include <limits>

namespace duckdb {

struct UDFDynamicBatchingConfig {
	bool enabled = false;
	idx_t min_batch_rows = 1;
	idx_t max_batch_rows = 1;
	idx_t initial_batch_rows = 1;
	std::chrono::microseconds target_batch_latency {0};
	std::chrono::microseconds latency_tolerance {0};
	idx_t step_size_alpha = 1;
	idx_t correction_delta = 1;
	idx_t history_size = 1;

	void Validate() const {
		if (!enabled) {
			return;
		}
		if (min_batch_rows == 0 || max_batch_rows == 0 || initial_batch_rows == 0) {
			throw InvalidInputException("dynamic UDF batch sizes must be positive");
		}
		if (min_batch_rows > max_batch_rows) {
			throw InvalidInputException("dynamic UDF min batch rows must not exceed max batch rows");
		}
		if (initial_batch_rows < min_batch_rows || initial_batch_rows > max_batch_rows) {
			throw InvalidInputException("dynamic UDF initial batch rows must be within the configured bounds");
		}
		if (target_batch_latency.count() <= 0) {
			throw InvalidInputException("dynamic UDF target batch latency must be positive");
		}
		if (latency_tolerance.count() < 0 || latency_tolerance >= target_batch_latency) {
			throw InvalidInputException(
			    "dynamic UDF latency tolerance must be non-negative and below the target latency");
		}
		using latency_rep = std::chrono::microseconds::rep;
		if (target_batch_latency.count() > std::numeric_limits<latency_rep>::max() - latency_tolerance.count()) {
			throw InvalidInputException("dynamic UDF target latency plus tolerance is too large");
		}
		if (step_size_alpha == 0 || correction_delta == 0 || history_size == 0) {
			throw InvalidInputException("dynamic UDF adjustment and history sizes must be positive");
		}
	}
};

// Latency-constrained batching follows Algorithm 2 in "Optimizing LLM Inference
// Throughput via Memory-aware and SLA-constrained Dynamic Batching". Its
// parameter defaults and flexible upper-bound semantics match Daft. The current
// row limit is the upper bound for the next UDF submission; byte pressure and
// end-of-stream may still produce a shorter batch.
class UDFDynamicBatchSizer {
public:
	UDFDynamicBatchSizer() = default;

	explicit UDFDynamicBatchSizer(const UDFDynamicBatchingConfig &config) {
		Reset(config);
	}

	void Reset(const UDFDynamicBatchingConfig &config) {
		config.Validate();
		config_ = config;
		recent_batch_rows_.clear();
		recent_latencies_.clear();
		if (!config_.enabled) {
			current_batch_rows_ = 0;
			search_low_ = 0;
			search_high_ = 0;
			return;
		}
		current_batch_rows_ = config_.initial_batch_rows;
		search_low_ = config_.min_batch_rows;
		search_high_ = config_.initial_batch_rows;
	}

	bool Enabled() const {
		return config_.enabled;
	}

	idx_t CurrentBatchRows() const {
		return current_batch_rows_;
	}

	idx_t SearchLow() const {
		return search_low_;
	}

	idx_t SearchHigh() const {
		return search_high_;
	}

	idx_t ObservationCount() const {
		return recent_batch_rows_.size();
	}

	void Record(idx_t batch_rows, std::chrono::microseconds duration) {
		if (!config_.enabled) {
			return;
		}
		if (batch_rows == 0) {
			throw InternalException("cannot record an empty dynamic UDF batch");
		}
		if (duration.count() < 0) {
			throw InternalException("cannot record a negative dynamic UDF batch latency");
		}

		recent_batch_rows_.push_back(batch_rows);
		recent_latencies_.push_back(duration);
		while (recent_batch_rows_.size() > config_.history_size) {
			recent_batch_rows_.pop_front();
			recent_latencies_.pop_front();
		}
		Recalculate();
	}

private:
	static idx_t SaturatingAdd(idx_t left, idx_t right) {
		if (right > std::numeric_limits<idx_t>::max() - left) {
			return std::numeric_limits<idx_t>::max();
		}
		return left + right;
	}

	static idx_t SaturatingSubtract(idx_t left, idx_t right) {
		return left > right ? left - right : 0;
	}

	static idx_t Midpoint(idx_t left, idx_t right) {
		return left / 2 + right / 2 + (left % 2 + right % 2) / 2;
	}

	void Recalculate() {
		if (recent_batch_rows_.empty()) {
			return;
		}

		idx_t total_rows = 0;
		int64_t total_latency_us = 0;
		for (auto rows : recent_batch_rows_) {
			total_rows = SaturatingAdd(total_rows, rows);
		}
		for (auto latency : recent_latencies_) {
			if (latency.count() > std::numeric_limits<int64_t>::max() - total_latency_us) {
				total_latency_us = std::numeric_limits<int64_t>::max();
				break;
			}
			total_latency_us += latency.count();
		}

		const auto observation_count = recent_batch_rows_.size();
		const auto average_rows = std::max<idx_t>(1, total_rows / observation_count);
		const auto average_latency = std::chrono::microseconds(total_latency_us / observation_count);
		const auto upper_latency = config_.target_batch_latency + config_.latency_tolerance;
		const auto lower_latency = config_.target_batch_latency - config_.latency_tolerance;

		if (average_latency > upper_latency) {
			search_high_ =
			    std::max(average_rows, SaturatingAdd(SaturatingSubtract(search_low_, 1), config_.step_size_alpha));
			search_low_ = std::max(SaturatingSubtract(SaturatingSubtract(search_low_, 1), config_.correction_delta),
			                       config_.min_batch_rows);
		} else if (average_latency < lower_latency) {
			search_low_ = std::max(average_rows,
			                       SaturatingSubtract(SaturatingSubtract(search_high_, 1), config_.step_size_alpha));
			search_high_ = std::min(config_.max_batch_rows,
			                        SaturatingAdd(SaturatingSubtract(search_high_, 1), config_.correction_delta));
		} else {
			const auto tighten_amount = config_.step_size_alpha / 2;
			search_high_ = std::min(SaturatingAdd(average_rows, tighten_amount), config_.max_batch_rows);
			search_low_ = std::max(SaturatingSubtract(average_rows, tighten_amount), config_.min_batch_rows);
		}

		// Out-of-order completions can temporarily cross the two search
		// endpoints. Daft still takes their midpoint instead of collapsing the
		// search to the latest observed batch size.
		current_batch_rows_ = Midpoint(search_low_, search_high_);
		current_batch_rows_ = std::min(std::max(current_batch_rows_, config_.min_batch_rows), config_.max_batch_rows);
	}

	UDFDynamicBatchingConfig config_;
	idx_t current_batch_rows_ = 0;
	idx_t search_low_ = 0;
	idx_t search_high_ = 0;
	std::deque<idx_t> recent_batch_rows_;
	std::deque<std::chrono::microseconds> recent_latencies_;
};

} // namespace duckdb
