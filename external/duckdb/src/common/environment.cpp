// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/common/environment.hpp"

#ifdef _WIN32
#include "duckdb/common/windows_util.hpp"
#else
#include <cstdlib>
#endif

namespace duckdb {

EnvironmentVariable::EnvironmentVariable(const char *name) {
#ifdef _WIN32
	// A statically linked CRT has its own environment table. Read the process
	// environment so changes made by Python or another DLL remain visible.
	auto wide_name = WindowsUtil::UTF8ToUnicode(name);
	vector<wchar_t> buffer(256);
	while (true) {
		SetLastError(ERROR_SUCCESS);
		auto length = GetEnvironmentVariableW(wide_name.c_str(), buffer.data(), static_cast<DWORD>(buffer.size()));
		if (length == 0) {
			auto error = GetLastError();
			if (error == ERROR_ENVVAR_NOT_FOUND) {
				return;
			}
			if (error != ERROR_SUCCESS) {
				throw IOException("Could not read environment variable '%s': Windows error %lu", name, error);
			}
			is_set = true;
			return;
		}
		if (length >= buffer.size()) {
			// The required size includes the terminator. Retry if the value grew
			// between reads, instead of using a truncated value.
			buffer.resize(length);
			continue;
		}
		value = WindowsUtil::UnicodeToUTF8(buffer.data());
		is_set = true;
		return;
	}
#else
	auto raw = std::getenv(name);
	if (raw) {
		value = raw;
		is_set = true;
	}
#endif
}

} // namespace duckdb
