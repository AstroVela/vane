// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "media_reader.hpp"

#include <boost/multiprecision/cpp_int.hpp>
#include <charconv>

namespace duckdb {

//! Same exact rational arithmetic as Python's Fraction. In particular, public
//! DOUBLE options use their shortest decimal representation, not an epsilon.
using VideoRational = boost::multiprecision::cpp_rational;
using VideoInteger = boost::multiprecision::cpp_int;

inline VideoRational VideoTimeOption(double value) {
	char buffer[64];
	auto converted = std::to_chars(buffer, buffer + sizeof(buffer), value);
	if (converted.ec != std::errc()) {
		throw InternalException("cannot represent video time option");
	}
	VideoInteger numerator = 0;
	int decimal_places = 0;
	bool decimal = false;
	auto cursor = buffer;
	for (; cursor != converted.ptr && *cursor != 'e' && *cursor != 'E'; cursor++) {
		if (*cursor == '.') {
			decimal = true;
		} else if (*cursor != '-') {
			numerator = numerator * 10 + (*cursor - '0');
			decimal_places += decimal ? 1 : 0;
		}
	}
	int exponent = 0;
	if (cursor != converted.ptr) {
		cursor++;
		bool negative = *cursor == '-';
		if (*cursor == '+' || *cursor == '-') {
			cursor++;
		}
		for (; cursor != converted.ptr; cursor++) {
			exponent = exponent * 10 + (*cursor - '0');
		}
		exponent *= negative ? -1 : 1;
	}
	exponent -= decimal_places;
	VideoInteger power = 1;
	for (int i = 0; i < std::abs(exponent); i++) {
		power *= 10;
	}
	if (exponent >= 0) {
		return VideoRational(numerator * power);
	}
	return VideoRational(numerator) / power;
}

inline VideoRational VideoFrameTime(int64_t pts, AVRational base, int64_t origin = 0) {
	return VideoRational((VideoInteger(pts) - origin) * base.num) / base.den;
}

inline double VideoTimeDouble(const VideoRational &value) {
	return value.convert_to<double>();
}

} // namespace duckdb
