#include "duckdb/common/windows_util.hpp"
#include <algorithm>

namespace duckdb {

#ifdef DUCKDB_WINDOWS

std::wstring WindowsUtil::UTF8ToUnicode(const char *input) {
	idx_t result_size;

	result_size = MultiByteToWideChar(CP_UTF8, 0, input, -1, nullptr, 0);
	if (result_size == 0) {
		throw IOException("Failure in MultiByteToWideChar");
	}
	auto buffer = make_unsafe_uniq_array<wchar_t>(result_size);
	result_size = MultiByteToWideChar(CP_UTF8, 0, input, -1, buffer.get(), result_size);
	if (result_size == 0) {
		throw IOException("Failure in MultiByteToWideChar");
	}
	return std::wstring(buffer.get(), result_size);
}

std::wstring WindowsUtil::UTF8ToUnicodePath(const char *input) {
	auto path = UTF8ToUnicode(input);
	// CreateDirectory reserves twelve characters below MAX_PATH. Explicit
	// extended paths work without a machine-wide LongPathsEnabled setting.
	if (path.size() < MAX_PATH - 12 || path.compare(0, 4, L"\\\\?\\") == 0 || path.compare(0, 4, L"\\\\.\\") == 0) {
		return path;
	}
	auto required = GetFullPathNameW(path.c_str(), 0, nullptr, nullptr);
	if (!required) {
		throw IOException("Could not normalize long Windows path");
	}
	vector<wchar_t> buffer(required);
	while (true) {
		auto length = GetFullPathNameW(path.c_str(), static_cast<DWORD>(buffer.size()), buffer.data(), nullptr);
		if (!length) {
			throw IOException("Could not normalize long Windows path");
		}
		if (length >= buffer.size()) {
			buffer.resize(length + 1);
			continue;
		}
		std::wstring absolute(buffer.data(), length);
		std::replace(absolute.begin(), absolute.end(), L'/', L'\\');
		if (absolute.compare(0, 2, L"\\\\") == 0) {
			return L"\\\\?\\UNC\\" + absolute.substr(2);
		}
		return L"\\\\?\\" + absolute;
	}
}

static string WideCharToMultiByteWrapper(LPCWSTR input, uint32_t code_page) {
	idx_t result_size;

	result_size = WideCharToMultiByte(code_page, 0, input, -1, 0, 0, 0, 0);
	if (result_size == 0) {
		throw IOException("Failure in WideCharToMultiByte");
	}
	auto buffer = make_unsafe_uniq_array<char>(result_size);
	result_size = WideCharToMultiByte(code_page, 0, input, -1, buffer.get(), result_size, 0, 0);
	if (result_size == 0) {
		throw IOException("Failure in WideCharToMultiByte");
	}
	return string(buffer.get(), result_size - 1);
}

string WindowsUtil::UnicodeToUTF8(LPCWSTR input) {
	return WideCharToMultiByteWrapper(input, CP_UTF8);
}

static string WindowsUnicodeToMBCS(LPCWSTR unicode_text, int use_ansi) {
	uint32_t code_page = use_ansi ? CP_ACP : CP_OEMCP;
	return WideCharToMultiByteWrapper(unicode_text, code_page);
}

string WindowsUtil::UTF8ToMBCS(const char *input, bool use_ansi) {
	auto unicode = WindowsUtil::UTF8ToUnicode(input);
	return WindowsUnicodeToMBCS(unicode.c_str(), use_ansi);
}

#endif

} // namespace duckdb
