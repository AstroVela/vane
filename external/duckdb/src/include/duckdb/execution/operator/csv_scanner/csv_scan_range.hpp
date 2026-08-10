// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/exception.hpp"
#include "duckdb/common/open_file_info.hpp"

namespace duckdb {

struct CSVScanRange {
	static constexpr const char *START_OPTION = "__vane_csv_scan_range_start";
	static constexpr const char *END_OPTION = "__vane_csv_scan_range_end";

	idx_t start;
	idx_t end;

	idx_t Size() const {
		return end - start;
	}

	static bool TryGet(const OpenFileInfo &file, CSVScanRange &result) {
		if (!file.extended_info) {
			return false;
		}
		auto start_entry = file.extended_info->options.find(START_OPTION);
		auto end_entry = file.extended_info->options.find(END_OPTION);
		if (start_entry == file.extended_info->options.end() && end_entry == file.extended_info->options.end()) {
			return false;
		}
		if (start_entry == file.extended_info->options.end() || end_entry == file.extended_info->options.end()) {
			throw InvalidInputException("CSV scan task for \"%s\" has an incomplete byte range", file.path);
		}
		result.start = start_entry->second.GetValue<idx_t>();
		result.end = end_entry->second.GetValue<idx_t>();
		if (result.start >= result.end) {
			throw InvalidInputException("CSV scan task for \"%s\" has invalid byte range [%llu, %llu)", file.path,
			                            result.start, result.end);
		}
		return true;
	}

	static OpenFileInfo Set(const OpenFileInfo &file, idx_t start, idx_t end) {
		OpenFileInfo result = file;
		result.extended_info = make_shared_ptr<ExtendedOpenFileInfo>();
		if (file.extended_info) {
			result.extended_info->options = file.extended_info->options;
		}
		result.extended_info->options[START_OPTION] = Value::UBIGINT(start);
		result.extended_info->options[END_OPTION] = Value::UBIGINT(end);
		return result;
	}

	static OpenFileInfo Strip(const OpenFileInfo &file) {
		if (!file.extended_info) {
			return file;
		}
		OpenFileInfo result = file;
		result.extended_info = make_shared_ptr<ExtendedOpenFileInfo>();
		result.extended_info->options = file.extended_info->options;
		result.extended_info->options.erase(START_OPTION);
		result.extended_info->options.erase(END_OPTION);
		if (result.extended_info->options.empty()) {
			result.extended_info.reset();
		}
		return result;
	}
};

} // namespace duckdb
