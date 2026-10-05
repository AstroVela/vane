// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "file_snapshot.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/local_file_system.hpp"
#include "duckdb/main/client_context.hpp"
#include "mbedtls_wrapper.hpp"

#include <filesystem>
#include <mutex>
#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include <cerrno>
#endif

namespace duckdb {
namespace vane_execution {
namespace {
using Hash = duckdb_mbedtls::MbedTlsWrapper::SHA256State;

string UTF8Path(const std::filesystem::path &path) {
	auto encoded = path.u8string();
	return string(reinterpret_cast<const char *>(encoded.data()), encoded.size());
}

string Finish(Hash &hash) {
	char result[64];
	hash.FinishHex(result);
	return string(result, sizeof(result));
}

void CheckFile(FileHandle &file) {
	if (!file.file_system.IsLocalFileSystem() || file.GetType() != FileType::FILE_TYPE_REGULAR) {
		throw NotImplementedException("FTE snapshots require regular local files");
	}
}

string Fingerprint(ClientContext &context, FileHandle &file, FileHandle *target, idx_t maximum) {
	CheckFile(file);
	auto before = file.Stats();
	if (before.file_size < 0 || idx_t(before.file_size) > maximum) {
		throw OutOfMemoryException("file snapshots exceed reserved storage capacity");
	}
	Hash hash;
	vector<char> buffer(65536);
	for (idx_t offset = 0; offset < idx_t(before.file_size);) {
		if (context.IsInterrupted()) {
			throw InterruptException();
		}
		auto count = MinValue<idx_t>(buffer.size(), before.file_size - offset);
		file.Read(buffer.data(), count, offset);
		if (target) {
			target->file_system.Write(*target, buffer.data(), count, offset);
		}
		hash.AddBytes(reinterpret_cast<const_data_ptr_t>(buffer.data()), count);
		offset += count;
	}
	auto after = file.Stats();
	if (before.file_size != after.file_size || before.last_modification_time != after.last_modification_time ||
	    before.extended_file_info != after.extended_file_info) {
		throw IOException("file changed while freezing query input");
	}
	return Finish(hash);
}
} // namespace

string FileFingerprint(ClientContext &context, const OpenFileInfo &source) {
	auto &fs = FileSystem::GetFileSystem(context);
	if (!fs.IsPathAbsolute(source.path)) {
		throw NotImplementedException("FTE snapshots require absolute local paths");
	}
	auto file = fs.OpenFile(source, FileFlags::FILE_FLAGS_READ | FileLockType::READ_LOCK);
	return Fingerprint(context, *file, nullptr, 1ULL << 40);
}

string FrozenFilePath(ClientContext &context, const string &source, const string &directory) {
	namespace paths = std::filesystem;
	auto &fs = FileSystem::GetFileSystem(context);
	if (!fs.IsPathAbsolute(source) || !fs.IsPathAbsolute(directory) || directory.find('=') != string::npos) {
		throw InvalidInputException("FTE snapshot paths must be absolute; the store prefix cannot contain '='");
	}
	auto original = paths::u8path(source).lexically_normal();
	auto root = paths::u8path(directory).lexically_normal();
	// Resolving before normalization distinguishes symlink/../file from a
	// different file with the same lexical path. Keep the reference's directory
	// layout below this namespace so Hive keys retain their original meaning.
	Hash source_hash;
	auto identity = UTF8Path(paths::weakly_canonical(paths::u8path(source)));
	source_hash.AddBytes(reinterpret_cast<const_data_ptr_t>(identity.data()), identity.size());
	return UTF8Path(root / Finish(source_hash) / original.relative_path());
}

FrozenFile FreezeFile(ClientContext &context, const string &source, const string &target, idx_t remaining) {
	namespace paths = std::filesystem;
	auto &fs = FileSystem::GetFileSystem(context);
	auto input = fs.OpenFile(source, FileFlags::FILE_FLAGS_READ | FileLockType::READ_LOCK);
	CheckFile(*input);
	auto destination = paths::u8path(target);
	// Preserve original hive key=value directories, without introducing new
	// partition keys. Filename virtual columns are rejected during validation.
	if (paths::weakly_canonical(destination) != destination) {
		throw InvalidInputException("FTE snapshot path contains a symlink");
	}
	paths::create_directories(destination.parent_path());
	LocalFileSystem local;
	auto output = local.OpenFile(UTF8Path(destination),
	                             FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE |
	                                 FileFlags::FILE_FLAGS_EXCLUSIVE_CREATE | FileFlags::FILE_FLAGS_PRIVATE);
	auto size = input->GetFileSize();
	auto sha = Fingerprint(context, *input, output.get(), remaining);
	output->Sync();
	output->Close();
	// A second pass also catches ordinary concurrent rewrites of an equal-size
	// file. Subsequent attempts read only the frozen copy, never the original.
	if (sha != Fingerprint(context, *input, nullptr, remaining) || input->GetFileSize() != size) {
		throw IOException("file changed while freezing query input");
	}
	return {UTF8Path(destination), idx_t(size), sha};
}

struct StoreGuard::Impl {
	std::mutex mutex;
#ifdef _WIN32
	HANDLE file = INVALID_HANDLE_VALUE;
#else
	int file = -1;
#endif
};

StoreGuard::StoreGuard() : impl(make_uniq<Impl>()) {
}
StoreGuard::~StoreGuard() {
	Close();
}

shared_ptr<StoreGuard> StoreGuard::Acquire(const string &path, bool exclusive, bool create) {
	if (path.empty() || path.find('\0') != string::npos) {
		throw InvalidInputException("invalid store lock path");
	}
	auto result = shared_ptr<StoreGuard>(new StoreGuard());
#ifdef _WIN32
	auto name = std::filesystem::u8path(path).wstring();
	result->impl->file =
	    CreateFileW(name.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
	                nullptr, create ? OPEN_ALWAYS : OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (result->impl->file == INVALID_HANDLE_VALUE) {
		throw IOException("cannot open store lock: Windows error %lu", GetLastError());
	}
	OVERLAPPED overlap = {};
	if (!LockFileEx(result->impl->file, LOCKFILE_FAIL_IMMEDIATELY | (exclusive ? LOCKFILE_EXCLUSIVE_LOCK : 0), 0, 1, 0,
	                &overlap)) {
		if (GetLastError() == ERROR_LOCK_VIOLATION) {
			return nullptr;
		}
		throw IOException("cannot acquire store lock: Windows error %lu", GetLastError());
	}
#else
	result->impl->file = open(path.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW | (create ? O_CREAT : 0), 0600);
	if (result->impl->file < 0) {
		throw IOException("cannot open store lock: errno %d", errno);
	}
	if (flock(result->impl->file, (exclusive ? LOCK_EX : LOCK_SH) | LOCK_NB) != 0) {
		if (errno == EWOULDBLOCK || errno == EAGAIN) {
			return nullptr;
		}
		throw IOException("cannot acquire store lock: errno %d", errno);
	}
#endif
	return result;
}

void StoreGuard::Close() {
	std::lock_guard<std::mutex> lock(impl->mutex);
#ifdef _WIN32
	if (impl->file != INVALID_HANDLE_VALUE) {
		CloseHandle(impl->file);
		impl->file = INVALID_HANDLE_VALUE;
	}
#else
	if (impl->file >= 0) {
		close(impl->file);
		impl->file = -1;
	}
#endif
}

} // namespace vane_execution
} // namespace duckdb
