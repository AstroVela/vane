// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/types/hugeint.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include <cmath>

namespace duckdb {

// Signed 192-bit accumulation covers uint64_t rows of signed 128-bit input.
// Keep addition exact until the complete group has arrived, including combine.
struct WideInteger {
	uint64_t lower = 0, middle = 0, upper = 0;

	void Add(uint64_t low, uint64_t mid, uint64_t high) {
		auto old_lower = lower;
		lower += low;
		auto carry = uint64_t(lower < old_lower);
		auto old_middle = middle;
		middle += mid;
		upper += high + uint64_t(middle < old_middle);
		old_middle = middle;
		middle += carry;
		upper += uint64_t(middle < old_middle);
	}
	void Add(hugeint_t value) {
		Add(value.lower, uint64_t(value.upper), value.upper < 0 ? NumericLimits<uint64_t>::Maximum() : 0);
	}
	bool TryGetHugeint(hugeint_t &result) const {
		const auto sign_extension = middle >> 63 ? NumericLimits<uint64_t>::Maximum() : 0;
		if (upper != sign_extension) {
			return false;
		}
		result = hugeint_t(int64_t(middle), lower);
		return true;
	}
	long double ToLongDouble() const {
		hugeint_t narrow;
		if (TryGetHugeint(narrow)) {
			// Preserve native AVG's conversion/rounding when its sum fits.
			return Hugeint::Cast<long double>(narrow);
		}
		auto magnitude = *this;
		const bool negative = upper >> 63;
		if (negative) {
			magnitude.lower = ~lower;
			magnitude.middle = ~middle;
			magnitude.upper = ~upper;
			magnitude.Add(1, 0, 0);
		}
		auto result = std::ldexp(static_cast<long double>(magnitude.upper), 128) +
		              std::ldexp(static_cast<long double>(magnitude.middle), 64) +
		              static_cast<long double>(magnitude.lower);
		return negative ? -result : result;
	}
};

struct WideIntegerState {
	WideInteger value;
	uint64_t count;

	void Initialize() {
		value = WideInteger();
		count = 0;
	}
	void AddCount(uint64_t increment) {
		if (increment > NumericLimits<uint64_t>::Maximum() - count) {
			throw OutOfRangeException("Overflow in wide aggregate row count");
		}
		count += increment;
	}
	void Combine(const WideIntegerState &other) {
		AddCount(other.count);
		value.Add(other.value.lower, other.value.middle, other.value.upper);
	}
};

struct WideIntegerOperation {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.Initialize();
	}
	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &) {
		target.Combine(source);
	}
	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE &state, const INPUT_TYPE &value, AggregateUnaryInput &) {
		state.AddCount(1);
		state.value.Add(value);
	}
	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE &state, const INPUT_TYPE &value, AggregateUnaryInput &input, idx_t count) {
		for (idx_t row = 0; row < count; row++) {
			Operation<INPUT_TYPE, STATE, OP>(state, value, input);
		}
	}
	static bool IgnoreNull() {
		return true;
	}
};

inline void ValidateWideIntegerInput(const LogicalType &input_type) {
	if (input_type.InternalType() != PhysicalType::INT128 ||
	    (input_type.id() != LogicalTypeId::HUGEINT && input_type.id() != LogicalTypeId::DECIMAL)) {
		throw SerializationException("invalid wide integer aggregate input type");
	}
}

} // namespace duckdb
