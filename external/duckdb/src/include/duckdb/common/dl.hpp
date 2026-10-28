//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/dl.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/windows.hpp"
#include "duckdb/common/local_file_system.hpp"
#include "duckdb/common/windows_util.hpp"

#ifndef _WIN32
#include <dlfcn.h>
#else
#define RTLD_NOW   0
#define RTLD_LOCAL 0
#endif

namespace duckdb {

#ifdef _WIN32

inline void *dlopen(const char *file, int mode) {
	if (!file) {
		return (void *)GetModuleHandle(nullptr);
	}
	auto fpath = WindowsUtil::UTF8ToUnicode(file);
	return (void *)LoadLibraryW(fpath.c_str());
}

inline void *dlsym(void *handle, const char *name) {
	D_ASSERT(handle);
	return (void *)GetProcAddress((HINSTANCE)handle, name);
}

inline std::string GetDLError(void) {
	return LocalFileSystem::GetLastErrorAsString();
}

inline void dlclose(void *handle) {
	D_ASSERT(handle);
	FreeLibrary((HINSTANCE)handle);
}

#else

inline std::string GetDLError(void) {
	return dlerror();
}

#endif

//! Media extensions keep their verified codec dependencies next to the artifact
//! in .libs. Scope that directory to this load, without changing the process's
//! default DLL search policy or the behavior of unrelated extensions.
inline void *dlopen_with_runtime(const char *file, int mode) {
#ifdef _WIN32
	auto path = WindowsUtil::UTF8ToUnicode(file);
	auto required = GetFullPathNameW(path.c_str(), 0, nullptr, nullptr);
	if (!required) {
		return nullptr;
	}
	std::wstring absolute(required, L'\0');
	auto written = GetFullPathNameW(path.c_str(), required, &absolute[0], nullptr);
	if (!written || written >= required) {
		if (written >= required) {
			SetLastError(ERROR_INSUFFICIENT_BUFFER);
		}
		return nullptr;
	}
	absolute.resize(written);
	auto separator = absolute.find_last_of(L"\\/");
	if (separator == std::wstring::npos) {
		SetLastError(ERROR_PATH_NOT_FOUND);
		return nullptr;
	}
	auto libraries = absolute.substr(0, separator + 1) + L".libs";
	using AddDirectory = void *(WINAPI *)(PCWSTR);
	using RemoveDllDirectoryFunction = BOOL(WINAPI *)(void *);
	auto kernel = GetModuleHandleW(L"kernel32.dll");
	auto add_directory = reinterpret_cast<AddDirectory>(GetProcAddress(kernel, "AddDllDirectory"));
	auto remove_directory = reinterpret_cast<RemoveDllDirectoryFunction>(GetProcAddress(kernel, "RemoveDllDirectory"));
	if (!add_directory || !remove_directory) {
		SetLastError(ERROR_CALL_NOT_IMPLEMENTED);
		return nullptr;
	}
	auto cookie = add_directory(libraries.c_str());
	if (!cookie) {
		return nullptr;
	}
	auto handle =
	    LoadLibraryExW(absolute.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
	auto error = GetLastError();
	remove_directory(cookie);
	SetLastError(error);
	return reinterpret_cast<void *>(handle);
#else
	return dlopen(file, mode);
#endif
}

} // namespace duckdb
