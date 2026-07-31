// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/vllm_inflight_counter.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/atomic.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/limits.hpp"

namespace duckdb {

//! Tracks the number of submitted vLLM prompts whose results have not been consumed.
//! A single atomic value makes each observation a consistent snapshot.
class VLLMInflightCounter {
public:
	idx_t Count() const {
		return count.load();
	}

	void RecordSubmission(idx_t amount) {
		if (amount == 0) {
			return;
		}

		auto current = count.load();
		do {
			if (amount > NumericLimits<idx_t>::Maximum() - current) {
				throw InternalException("vllm inflight prompt count overflow");
			}
		} while (!count.compare_exchange_weak(current, current + amount));
	}

	void RecordCompletion(idx_t amount) {
		if (amount == 0) {
			return;
		}

		auto current = count.load();
		do {
			if (amount > current) {
				throw InternalException("vllm completed prompt count exceeded submitted prompt count");
			}
		} while (!count.compare_exchange_weak(current, current - amount));
	}

private:
	atomic<idx_t> count {0};
};

} // namespace duckdb
