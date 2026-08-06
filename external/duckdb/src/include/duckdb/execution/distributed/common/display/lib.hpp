// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once
#include <string>

namespace duckdb {
namespace distributed {

// Display detail levels
enum class DisplayLevel {
	Compact, // Show only the most important information
	Default, // Show commonly useful information
	Verbose  // Show all available information
};

} // namespace distributed
} // namespace duckdb
