// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "file_snapshot.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/hive_partitioning.hpp"
#include "duckdb/common/local_file_system.hpp"
#include "duckdb/main/client_context.hpp"
#include "sha256.hpp"

#include <filesystem>
#include <mutex>
#include <set>
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
using Hash = SHA256State;

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

string FileIdentity(ClientContext &context, const OpenFileInfo &source) {
	auto &fs = FileSystem::GetFileSystem(context);
	if (!fs.IsPathAbsolute(source.path)) {
		throw NotImplementedException("submission file sources require absolute local paths");
	}
	auto file = fs.OpenFile(source, FileOpenFlags(FileOpenFlags::FILE_FLAGS_READ));
	CheckFile(*file);
	auto stats = file->Stats();
	string result = std::to_string(stats.file_size) + ":" + std::to_string(stats.last_modification_time.value);
	for (auto &key : {"mtime_nsec", "file_identity", "ctime", "ctime_nsec"}) {
		auto found = stats.extended_file_info.find(key);
		if (found == stats.extended_file_info.end()) {
			throw NotImplementedException("submission file sources require precise local file identities");
		}
		result += ":" + found->second.ToString();
	}
	return result;
}

string FrozenFilePath(ClientContext &context, const string &source, const string &directory) {
	namespace paths = std::filesystem;
	auto &fs = FileSystem::GetFileSystem(context);
	if (!fs.IsPathAbsolute(source) || !fs.IsPathAbsolute(directory) || directory.find('=') != string::npos) {
		throw InvalidInputException("FTE snapshot paths must be absolute; the store prefix cannot contain '='");
	}
	auto root = paths::u8path(directory).lexically_normal();
	// Physical identity and Hive values are distinct: parent traversal may
	// remove a directory from the physical path while Hive still reads its key.
	Hash source_hash;
	auto identity = UTF8Path(paths::weakly_canonical(paths::u8path(source)));
	source_hash.AddBytes(reinterpret_cast<const_data_ptr_t>(identity.data()), identity.size());
	auto target = root / Finish(source_hash);
	// Parse before normalization, using the same first-key-wins and escaping
	// rules as binding, pruning and execution. Parse only returns directory
	// components, so each key=value segment is safe to append without traversal.
	for (auto &part : HivePartitioning::Parse(source)) {
		target /= paths::u8path(part.first + "=" + part.second);
	}
	return UTF8Path(target / "data.parquet");
}

FrozenFile FreezeFile(ClientContext &context, const string &source, const string &target, idx_t remaining,
                      bool *created) {
	namespace paths = std::filesystem;
	auto &fs = FileSystem::GetFileSystem(context);
	auto input = fs.OpenFile(source, FileFlags::FILE_FLAGS_READ | FileLockType::READ_LOCK);
	CheckFile(*input);
	auto destination = paths::u8path(target);
	// Filename virtual columns are rejected during validation. Only the Hive
	// values extracted from the original reference are encoded in this path.
	if (paths::weakly_canonical(destination) != destination) {
		throw InvalidInputException("FTE snapshot path contains a symlink");
	}
	paths::create_directories(destination.parent_path());
	LocalFileSystem local;
	auto output = local.OpenFile(UTF8Path(destination),
	                             FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE |
	                                 FileFlags::FILE_FLAGS_EXCLUSIVE_CREATE | FileFlags::FILE_FLAGS_PRIVATE);
	if (created) {
		*created = true;
	}
	auto size = input->GetFileSize();
	auto sha = Fingerprint(context, *input, output.get(), remaining);
	if (input->GetFileSize() != size) {
		throw IOException("file changed while freezing query input");
	}
	output->Sync();
	output->Close();
	// Fingerprint checked size, inode, mtime and ctime around the copy. The
	// query-owned frozen inode now has one verified content identity. Attempts
	// validate that same inode/version instead of rehashing its entire contents.
	auto path = UTF8Path(destination);
	return {path, idx_t(size), sha, FileIdentity(context, OpenFileInfo(path))};
}

struct SourceSnapshotCache::Impl {
	Impl(string directory_p, idx_t capacity_p) : directory(std::move(directory_p)), capacity(capacity_p) {
	}
	const string directory;
	const idx_t capacity;
	mutable std::mutex lock;
	std::map<string, FrozenFile> files;
	std::set<string> failed_cleanup;
	idx_t used = 0;
	idx_t hits = 0;
	idx_t misses = 0;
	idx_t bypasses = 0;
};

SourceSnapshotCache::SourceSnapshotCache(const string &directory, idx_t capacity)
    : impl(make_uniq<Impl>(directory, capacity)) {
	namespace paths = std::filesystem;
	LocalFileSystem fs;
	if (!capacity || capacity > (1ULL << 40) || !fs.IsPathAbsolute(directory) || directory.find('=') != string::npos ||
	    directory.find('\0') != string::npos ||
	    paths::weakly_canonical(paths::u8path(directory)) != paths::u8path(directory)) {
		throw InvalidInputException("source snapshot cache requires a canonical directory and finite capacity");
	}
	paths::create_directories(paths::u8path(directory));
}

SourceSnapshotCache::~SourceSnapshotCache() {
}

FrozenFile SourceSnapshotCache::Freeze(ClientContext &context, const string &source, const string &destination,
                                       idx_t remaining) {
	namespace paths = std::filesystem;
	std::unique_lock<std::mutex> guard(impl->lock);
	if (context.IsInterrupted()) {
		throw InterruptException();
	}
	auto relative = paths::weakly_canonical(paths::u8path(source)).lexically_relative(paths::u8path(impl->directory));
	if (!relative.empty() && *relative.begin() != paths::u8path("..")) {
		throw InvalidInputException(
		    "source patterns include snapshot cache files; use a store outside the input dataset");
	}
	auto version = FileIdentity(context, OpenFileInfo(source));
	// FileIdentity starts with the exact byte size, then the inode and precise
	// modification/change timestamps. A restored mtime cannot reuse a rewrite.
	auto bytes = idx_t(std::stoull(version.substr(0, version.find(':'))));
	if (bytes > remaining) {
		throw OutOfMemoryException("file snapshots exceed reserved storage capacity");
	}
	Hash hash;
	hash.AddBytes(reinterpret_cast<const_data_ptr_t>(version.data()), version.size());
	auto directory = UTF8Path(paths::u8path(impl->directory) / Finish(hash));
	auto target = FrozenFilePath(context, source, directory);
	auto check_source = [&]() {
		if (FileIdentity(context, OpenFileInfo(source)) != version ||
		    FrozenFilePath(context, source, directory) != target) {
			throw IOException("file changed while freezing query input");
		}
	};
	auto found = impl->files.find(target);
	if (found != impl->files.end()) {
		check_source();
		if (FileIdentity(context, OpenFileInfo(found->second.path)) != found->second.identity) {
			throw IOException("cached source snapshot changed");
		}
		impl->hits++;
		return found->second;
	}
	// Bound both data and metadata, including failed fills. Full caches use the
	// ordinary query-owned path without evicting files borrowed by live readers.
	if (bytes > impl->capacity - impl->used || impl->misses >= 4096 || impl->failed_cleanup.count(target)) {
		impl->bypasses++;
		guard.unlock();
		return FreezeFile(context, source, destination, remaining);
	}
	impl->misses++;
	impl->used += bytes;
	bool created = false;
	try {
		auto frozen = FreezeFile(context, source, target, bytes, &created);
		check_source();
		impl->files.emplace(target, frozen);
		return frozen;
	} catch (...) {
		std::error_code error;
		if (created) {
			paths::remove(paths::u8path(target), error);
		}
		if (error) {
			// Failed deletion keeps its complete charge until the cache lease
			// is retired. Never offer these partial bytes to another query.
			impl->failed_cleanup.insert(target);
		} else {
			impl->used -= bytes;
		}
		throw;
	}
}

std::map<string, idx_t> SourceSnapshotCache::Snapshot() const {
	std::lock_guard<std::mutex> guard(impl->lock);
	return {{"capacity_bytes", impl->capacity}, {"used_bytes", impl->used},
	        {"entries", impl->files.size()},    {"hits", impl->hits},
	        {"misses", impl->misses},           {"bypasses", impl->bypasses}};
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
