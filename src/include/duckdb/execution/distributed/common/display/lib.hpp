// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once
#include <string>

namespace duckdb {
namespace distributed {

// 显示级别枚举
enum class DisplayLevel {
	Compact, // 紧凑显示，仅展示最重要信息
	Default, // 默认显示，展示常用信息
	Verbose  // 详细显示，展示所有可用信息
};

} // namespace distributed
} // namespace duckdb
