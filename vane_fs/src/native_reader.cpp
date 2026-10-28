// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "native_reader.hpp"
#include "vane_fs/reader_abi.h"
#include "vane_fs/workspace.hpp"
#include <pybind11/stl/filesystem.h>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace {
using namespace vane_fs;
namespace py = pybind11;

struct Provider {
	explicit Provider(const std::string &path, int timeout) : workspace(path, timeout), id(workspace.Id()) {
	}
	Workspace workspace;
	std::string id;
	VaneFSReaderV1 api {};
	std::mutex mutex;
	std::unordered_map<std::string, std::weak_ptr<Session>> snapshots;
};

struct Handle {
	Handle(Provider &provider, const std::string &id) : provider(provider), id(id) {
		std::lock_guard<std::mutex> lock(provider.mutex);
		snapshot = provider.snapshots[id].lock();
		if (!snapshot) {
			try {
				snapshot = provider.workspace.OpenSnapshot(id);
				provider.snapshots[id] = snapshot;
			} catch (...) {
				provider.snapshots.erase(id);
				throw;
			}
		}
	}
	~Handle() {
		std::lock_guard<std::mutex> lock(provider.mutex);
		snapshot.reset();
		if (provider.snapshots.at(id).expired())
			provider.snapshots.erase(id);
	}
	Provider &provider;
	std::string id;
	std::shared_ptr<Session> snapshot;
	int64_t inode;
};

template <class F>
int32_t Guard(VaneFSReaderError *error, F operation) noexcept {
	try {
		operation();
		return VANE_FS_READER_OK;
	} catch (const std::exception &exception) {
		std::strncpy(error->message, exception.what(), sizeof(error->message) - 1);
		error->message[sizeof(error->message) - 1] = '\0';
		if (auto native = dynamic_cast<const Error *>(&exception)) {
			if (native->code == ErrorCode::NotFound || native->code == ErrorCode::NotDirectory)
				return VANE_FS_READER_NOT_FOUND;
			if (native->code == ErrorCode::Invalid || native->code == ErrorCode::Closed)
				return VANE_FS_READER_INVALID;
		}
	} catch (...) {
		std::strcpy(error->message, "Unknown native VaneFS read failure");
	}
	return VANE_FS_READER_IO;
}

int32_t Open(void *context, const char *snapshot, const char *path, void **out, VaneFSReaderStat *stat,
             VaneFSReaderError *error) noexcept {
	*out = nullptr;
	return Guard(error, [&] {
		// Reuse a live immutable session across query and file handles. Its
		// final release retires the pin, avoiding a durable SQLite write pair
		// for every file opened within one query.
		auto handle = std::make_unique<Handle>(*static_cast<Provider *>(context), snapshot);
		auto file = handle->snapshot->Stat(path);
		handle->inode = file.inode;
		*stat = {file.inode, file.size, file.mtime_ns, file.is_directory ? 1 : 0};
		*out = handle.release();
	});
}

int32_t Read(void *opaque, uint64_t offset, uint64_t size, void *buffer, uint64_t *read_size,
             VaneFSReaderError *error) noexcept {
	*read_size = 0;
	return Guard(error, [&] {
		if (offset > uint64_t(std::numeric_limits<int64_t>::max()) ||
		    size > uint64_t(std::numeric_limits<int64_t>::max()) || (!buffer && size))
			throw Error(ErrorCode::Invalid, "Invalid native read range");
		auto &handle = *static_cast<Handle *>(opaque);
		auto data = handle.snapshot->ReadInode(handle.inode, int64_t(offset), int64_t(size));
		if (!data.empty())
			std::memcpy(buffer, data.data(), data.size());
		*read_size = data.size();
	});
}

void Close(void *handle) noexcept {
	delete static_cast<Handle *>(handle);
}
} // namespace

void RegisterNativeReader(pybind11::module_ &module) {
	module.def(
	    "_reader_capsule",
	    [](const std::filesystem::path &path, int timeout_ms) {
		    std::unique_ptr<Provider> provider;
		    {
			    py::gil_scoped_release release;
			    if (!std::filesystem::is_regular_file(path))
				    throw Error(ErrorCode::NotFound, "Native reader requires an existing VaneFS database");
			    provider = std::make_unique<Provider>(path.u8string(), timeout_ms);
		    }
		    provider->api = {VANE_FS_READER_VERSION,
		                     sizeof(VaneFSReaderV1),
		                     provider.get(),
		                     provider->id.c_str(),
		                     Open,
		                     Read,
		                     Close};
		    auto capsule = py::capsule(&provider->api, VANE_FS_READER_CAPSULE, [](PyObject *object) {
			    auto api = static_cast<VaneFSReaderV1 *>(PyCapsule_GetPointer(object, VANE_FS_READER_CAPSULE));
			    delete static_cast<Provider *>(api->context);
		    });
		    provider.release();
		    return capsule;
	    },
	    py::arg("path"), py::arg("timeout_ms") = 5000);
}
