// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/vane_fs.hpp"
#include "vane_fs/reader_abi.h"
#include "duckdb/common/file_opener.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"

#include <algorithm>
#include <cerrno>
#include <limits>

namespace duckdb {
namespace {
constexpr const char *STATE_KEY = "vane.fs.snapshots.v1";
constexpr const char *FILESYSTEM_NAME = "vanefs_native";
thread_local ClientContext *standalone_context = nullptr;

bool ValidID(const string &id) {
	return id.size() == 32 && id.find_first_not_of("0123456789abcdef") == string::npos;
}

struct SnapshotPath {
	string workspace;
	string snapshot;
	string path;

	explicit SnapshotPath(const string &url) {
		const string prefix = "vanefs://";
		const string middle = "/snapshots/";
		const auto snapshot_start = prefix.size() + 32 + middle.size();
		const auto path_start = snapshot_start + 32;
		if (url.compare(0, prefix.size(), prefix) != 0 || url.size() < path_start ||
		    url.compare(prefix.size() + 32, middle.size(), middle) != 0 ||
		    (url.size() > path_start && url[path_start] != '/') || url.find('\0') != string::npos) {
			throw InvalidInputException("Invalid VaneFS snapshot URL: %s", url);
		}
		workspace = url.substr(prefix.size(), 32);
		snapshot = url.substr(snapshot_start, 32);
		path = url.size() == path_start ? "/" : url.substr(path_start);
		if (!ValidID(workspace) || !ValidID(snapshot)) {
			throw InvalidInputException("Invalid VaneFS workspace or snapshot ID");
		}
		for (auto &part : StringUtil::Split(path, '/')) {
			if (part == "..") {
				throw InvalidInputException("VaneFS paths cannot contain '..'");
			}
		}
	}
};

void CheckResult(int32_t code, const VaneFSReaderError &error) {
	if (code == VANE_FS_READER_OK) {
		return;
	}
	const string message(error.message, std::find(error.message, error.message + sizeof(error.message), '\0'));
	if (code == VANE_FS_READER_INVALID) {
		throw InvalidInputException("VaneFS: %s", message);
	}
	if (code == VANE_FS_READER_NOT_FOUND) {
		throw IOException({{"errno", std::to_string(ENOENT)}}, "VaneFS: %s", message);
	}
	throw IOException("VaneFS: %s", message);
}

struct SnapshotProvider {
	explicit SnapshotProvider(py::object capsule_p) : capsule(std::move(capsule_p)) {
		if (!PyCapsule_IsValid(capsule.ptr(), VANE_FS_READER_CAPSULE)) {
			throw InvalidInputException("Expected a VaneFS snapshot reader v1 capsule");
		}
		api = static_cast<VaneFSReaderV1 *>(PyCapsule_GetPointer(capsule.ptr(), VANE_FS_READER_CAPSULE));
		if (api->version != VANE_FS_READER_VERSION || api->struct_size < sizeof(VaneFSReaderV1) || !api->context ||
		    !api->workspace_id || !api->open || !api->read || !api->close || !ValidID(api->workspace_id)) {
			throw InvalidInputException("Incompatible VaneFS snapshot reader ABI");
		}
	}
	~SnapshotProvider() {
		PythonGILWrapper gil;
		capsule = py::object();
	}
	py::object capsule;
	VaneFSReaderV1 *api;
};

// All native allocations are released by the module that created them. The
// capsule keeps that provider alive independently of connection registration.
struct SnapshotInode {
	explicit SnapshotInode(shared_ptr<SnapshotProvider> provider_p) : provider(std::move(provider_p)) {
	}
	~SnapshotInode() {
		if (handle) {
			provider->api->close(handle);
		}
	}
	shared_ptr<SnapshotProvider> provider;
	void *handle = nullptr;
	VaneFSReaderStat stat {};
};

unique_ptr<SnapshotInode> OpenInode(shared_ptr<SnapshotProvider> provider, const SnapshotPath &path,
                                    bool return_missing = false) {
	auto inode = make_uniq<SnapshotInode>(std::move(provider));
	VaneFSReaderError error {};
	auto &api = *inode->provider->api;
	auto result = api.open(api.context, path.snapshot.c_str(), path.path.c_str(), &inode->handle, &inode->stat, &error);
	if (return_missing && result == VANE_FS_READER_NOT_FOUND) {
		return nullptr;
	}
	CheckResult(result, error);
	return inode;
}

struct SnapshotState : ClientContextState {
	void QueryBegin(ClientContext &) override {
		lock_guard<mutex> guard(lock);
		query_active = true;
	}
	void QueryEnd(ClientContext &, optional_ptr<ErrorData>) override {
		lock_guard<mutex> guard(lock);
		query_pins.clear();
		query_active = false;
	}
	unique_ptr<SnapshotInode> Open(const SnapshotPath &path, bool return_missing, bool pin_for_query) {
		lock_guard<mutex> guard(lock);
		auto entry = providers.find(path.workspace);
		if (entry == providers.end()) {
			throw IOException("VaneFS workspace %s is not registered on this connection", path.workspace);
		}
		const auto key = path.workspace + path.snapshot;
		// Pin before opening individual files so GC cannot invalidate subsequent
		// rows in this query. QueryEnd runs after executor tasks have quiesced.
		if (query_active && pin_for_query && query_pins.find(key) == query_pins.end()) {
			auto root = path;
			root.path = "/";
			auto pin = OpenInode(entry->second, root, return_missing);
			if (!pin) {
				return nullptr;
			}
			query_pins.emplace(key, std::move(pin));
		}
		return OpenInode(entry->second, path, return_missing);
	}
	mutex lock;
	bool query_active = false;
	unordered_map<string, shared_ptr<SnapshotProvider>> providers;
	unordered_map<string, unique_ptr<SnapshotInode>> query_pins;
};

struct SnapshotFileHandle : FileHandle {
	SnapshotFileHandle(unique_ptr<FileSystem> filesystem_p, const string &path, FileOpenFlags flags,
	                   unique_ptr<SnapshotInode> inode_p, weak_ptr<ClientContext> context_p)
	    : FileHandle(*filesystem_p, path, flags), filesystem(std::move(filesystem_p)), inode(std::move(inode_p)),
	      context(std::move(context_p)) {
	}
	void Close() override {
		lock_guard<mutex> guard(lock);
		inode.reset();
	}
	// The router may be removed while readers are alive. Each handle owns its
	// stateless implementation rather than referencing the registry's lifetime.
	unique_ptr<FileSystem> filesystem;
	unique_ptr<SnapshotInode> inode;
	weak_ptr<ClientContext> context;
	mutex lock;
	idx_t position = 0;
};

class SnapshotFileSystem : public FileSystem {
public:
	string GetName() const override {
		return FILESYSTEM_NAME;
	}
	bool CanHandleFile(const string &path) override {
		return StringUtil::StartsWith(path, "vanefs://");
	}
	bool IsManuallySet() override {
		return true;
	}
	bool CanSeek() override {
		return true;
	}
	bool OnDiskFile(FileHandle &) override {
		return false;
	}
	bool HasDirectorySemantics(const string &, optional_ptr<FileOpener>) override {
		return true;
	}
	unique_ptr<FileHandle> OpenFile(const string &url, FileOpenFlags flags, optional_ptr<FileOpener> opener) override {
		if (!flags.OpenForReading() || flags.OpenForWriting() || flags.OpenForAppending() ||
		    flags.CreateFileIfNotExists() || flags.OverwriteExistingFile() || flags.ExclusiveCreate() ||
		    flags.ReturnNullIfExists() || flags.Lock() != FileLockType::NO_LOCK) {
			throw PermissionException("VaneFS snapshots are read-only");
		}
		if (flags.Compression() != FileCompressionType::UNCOMPRESSED || flags.DirectIO()) {
			throw NotImplementedException("VaneFS does not support compressed or direct native opens");
		}
		auto context = FileOpener::TryGetClientContext(opener);
		if (!context) {
			throw IOException("VaneFS reads require a registered Vane connection");
		}
		auto state = context->registered_state->Get<SnapshotState>(STATE_KEY);
		if (!state) {
			throw IOException("VaneFS workspace is not registered on this connection");
		}
		if (context->IsInterrupted()) {
			throw InterruptException();
		}
		auto inode = state->Open(SnapshotPath(url), flags.ReturnNullIfNotExists(), standalone_context != context.get());
		if (!inode) {
			return nullptr;
		}
		return make_uniq<SnapshotFileHandle>(make_uniq<SnapshotFileSystem>(), url, flags, std::move(inode),
		                                     context->shared_from_this());
	}
	int64_t Read(FileHandle &file, void *buffer, int64_t size) override {
		auto &handle = file.Cast<SnapshotFileHandle>();
		lock_guard<mutex> guard(handle.lock);
		auto count = ReadAt(handle, buffer, size, handle.position);
		handle.position += count;
		return count;
	}
	void Read(FileHandle &file, void *buffer, int64_t size, idx_t offset) override {
		auto &handle = file.Cast<SnapshotFileHandle>();
		lock_guard<mutex> guard(handle.lock);
		if (ReadAt(handle, buffer, size, offset) != size) {
			throw IOException("VaneFS read extends beyond end of file");
		}
	}
	void Seek(FileHandle &file, idx_t position) override {
		auto &handle = file.Cast<SnapshotFileHandle>();
		lock_guard<mutex> guard(handle.lock);
		handle.position = position;
	}
	idx_t SeekPosition(FileHandle &file) override {
		auto &handle = file.Cast<SnapshotFileHandle>();
		lock_guard<mutex> guard(handle.lock);
		return handle.position;
	}
	int64_t GetFileSize(FileHandle &file) override {
		return GetStat(file).size;
	}
	timestamp_t GetLastModifiedTime(FileHandle &file) override {
		return timestamp_t(GetStat(file).mtime_ns / 1000);
	}
	string GetVersionTag(FileHandle &file) override {
		return SnapshotPath(file.path).snapshot + ":" + std::to_string(GetStat(file).inode);
	}
	FileType GetFileType(FileHandle &file) override {
		return GetStat(file).directory ? FileType::FILE_TYPE_DIR : FileType::FILE_TYPE_REGULAR;
	}
	bool FileExists(const string &path, optional_ptr<FileOpener> opener) override {
		auto handle = OpenFile(path, FileFlags::FILE_FLAGS_READ | FileFlags::FILE_FLAGS_NULL_IF_NOT_EXISTS, opener);
		return handle && GetFileType(*handle) == FileType::FILE_TYPE_REGULAR;
	}
	bool DirectoryExists(const string &path, optional_ptr<FileOpener> opener) override {
		auto handle = OpenFile(path, FileFlags::FILE_FLAGS_READ | FileFlags::FILE_FLAGS_NULL_IF_NOT_EXISTS, opener);
		return handle && GetFileType(*handle) == FileType::FILE_TYPE_DIR;
	}
	bool IsPipe(const string &, optional_ptr<FileOpener>) override {
		return false;
	}
	vector<OpenFileInfo> Glob(const string &path, FileOpener *opener) override {
		if (HasGlob(path)) {
			throw NotImplementedException(
			    "Native VaneFS reads require explicit snapshot file paths; globbing is unsupported");
		}
		return FileExists(path, opener) ? vector<OpenFileInfo> {OpenFileInfo(path)} : vector<OpenFileInfo> {};
	}

private:
	static VaneFSReaderStat GetStat(FileHandle &file) {
		auto &handle = file.Cast<SnapshotFileHandle>();
		lock_guard<mutex> guard(handle.lock);
		if (!handle.inode) {
			throw IOException("VaneFS file is closed");
		}
		return handle.inode->stat;
	}
	static int64_t ReadAt(SnapshotFileHandle &handle, void *buffer, int64_t size, idx_t offset) {
		if (!handle.inode) {
			throw IOException("VaneFS file is closed");
		}
		if (size < 0 || offset > idx_t(std::numeric_limits<int64_t>::max()) ||
		    idx_t(size) > idx_t(std::numeric_limits<int64_t>::max()) - offset) {
			throw InvalidInputException("Invalid VaneFS read range");
		}
		auto context = handle.context.lock();
		idx_t done = 0;
		while (done < idx_t(size)) {
			if (context && context->IsInterrupted()) {
				throw InterruptException();
			}
			const auto count = MinValue<idx_t>(idx_t(size) - done, 1024 * 1024);
			uint64_t read_size = 0;
			VaneFSReaderError error {};
			CheckResult(handle.inode->provider->api->read(handle.inode->handle, offset + done, count,
			                                              static_cast<char *>(buffer) + done, &read_size, &error),
			            error);
			done += read_size;
			if (read_size < count) {
				break;
			}
		}
		return int64_t(done);
	}
};
} // namespace

VaneFSStandaloneOpenScope::VaneFSStandaloneOpenScope(ClientContext &context) : previous(standalone_context) {
	standalone_context = &context;
}

VaneFSStandaloneOpenScope::~VaneFSStandaloneOpenScope() {
	standalone_context = previous;
}

void InitializeVaneFS(py::class_<DuckDBPyConnection, shared_ptr<DuckDBPyConnection>> &connection) {
	connection.def("_register_vane_fs", [](DuckDBPyConnection &owner, py::object capsule) {
		auto provider = make_shared_ptr<SnapshotProvider>(std::move(capsule));
		const string id = provider->api->workspace_id;
		py::gil_scoped_release release;
		std::lock_guard<std::recursive_mutex> connection_guard(owner.py_connection_lock);
		if (owner.GetRunnerType() != "local-fast") {
			throw InvalidInputException("Native VaneFS registration requires runner='local-fast'");
		}
		auto &context = *owner.con.GetConnection().context;
		// Registration is infrequent. Serialize the list/register pair across
		// connections sharing one database; file reads do not use this lock.
		static mutex registration_lock;
		lock_guard<mutex> registration_guard(registration_lock);
		auto &fs = owner.con.GetDatabase().GetFileSystem();
		auto names = fs.ListSubSystems();
		if (std::find(names.begin(), names.end(), "vanefs") != names.end()) {
			throw InvalidInputException("Unregister the Python vanefs filesystem before registering native VaneFS");
		}
		if (std::find(names.begin(), names.end(), FILESYSTEM_NAME) == names.end()) {
			fs.RegisterSubSystem(make_uniq<SnapshotFileSystem>());
		}
		auto state = context.registered_state->GetOrCreate<SnapshotState>(STATE_KEY);
		lock_guard<mutex> guard(state->lock);
		state->providers[id] = std::move(provider);
		return id;
	});
	connection.def("_unregister_vane_fs", [](DuckDBPyConnection &owner, const string &id) {
		py::gil_scoped_release release;
		std::lock_guard<std::recursive_mutex> connection_guard(owner.py_connection_lock);
		auto &context = *owner.con.GetConnection().context;
		auto state = context.registered_state->Get<SnapshotState>(STATE_KEY);
		if (state) {
			lock_guard<mutex> guard(state->lock);
			state->providers.erase(id);
		}
	});
}
} // namespace duckdb
