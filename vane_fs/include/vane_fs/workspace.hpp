// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace vane_fs {

enum class ErrorCode {
	Invalid,
	NotFound,
	Exists,
	NotDirectory,
	IsDirectory,
	NotEmpty,
	ReadOnly,
	Busy,
	Capacity,
	Conflict,
	Stale,
	Closed,
	Storage
};

class Error : public std::runtime_error {
public:
	Error(ErrorCode code, const std::string &message) : std::runtime_error(message), code(code) {
	}
	ErrorCode code;
};

struct FileStat {
	int64_t inode = 0;
	bool is_directory = false;
	int64_t size = 0;
	int64_t mode = 0;
	int64_t mtime_ns = 0;
	int64_t links = 1;
};

struct BranchInfo {
	std::string id, name, parent_id, fork_base, state;
	int64_t generation = 0;
};

struct Change {
	std::string path, kind;
};

struct MergePreview {
	std::string workspace_id, source, target;
	int64_t source_generation = 0, target_generation = 0;
	std::vector<Change> changes;
	std::vector<std::string> conflicts;
};

struct CollectionResult {
	int64_t versions = 0, payloads = 0, snapshots = 0;
};

struct RecoveryResult {
	int64_t owners = 0, pins = 0, mounts = 0;
};

class Database;

// Sessions are thread-safe. Every method is one transaction. Live branch
// sessions refresh their writable interval under the database writer lock.
class Session {
public:
	~Session();
	Session(const Session &) = delete;
	Session &operator=(const Session &) = delete;
	void Close();
	FileStat Stat(const std::string &path);
	std::vector<std::string> ListDirectory(const std::string &path);
	std::string Read(const std::string &path, int64_t offset = 0, int64_t size = -1);
	void MakeDirectory(const std::string &path, int64_t mode = 0755);
	void WriteFile(const std::string &path, const std::string &data);
	void Write(const std::string &path, const std::string &data, int64_t offset = 0);
	void Truncate(const std::string &path, int64_t size);
	void Rename(const std::string &source, const std::string &target, bool no_replace = false);
	void Unlink(const std::string &path);
	void RemoveDirectory(const std::string &path);
	// Mount adapters hold an exclusive branch lease. Inode handles survive
	// rename/unlink; the last close reclaims an unlinked inode.
	FileStat OpenFile(const std::string &path, bool create = false, bool exclusive = false, bool truncate = false,
	                  int64_t mode = 0644);
	FileStat OpenDirectory(const std::string &path);
	void CloseFile(int64_t inode, int64_t references = 1);
	FileStat OpenInode(int64_t inode, bool directory = false, bool truncate = false);
	// Low-level FUSE lookup references are retained until FORGET. Create can
	// atomically retain both a lookup and an open reference.
	FileStat LookupInode(int64_t parent, const std::string &name);
	FileStat CreateNode(int64_t parent, const std::string &name, bool directory, int64_t mode, bool exclusive = true,
	                    bool truncate = false, int64_t references = 1);
	void RemoveNode(int64_t parent, const std::string &name, bool directory);
	void RenameNode(int64_t parent, const std::string &name, int64_t new_parent, const std::string &new_name,
	                bool no_replace = false);
	std::vector<std::pair<std::string, FileStat>> DirectoryEntries(int64_t inode);
	FileStat StatInode(int64_t inode);
	std::vector<std::string> ListDirectoryInode(int64_t inode);
	std::string ReadInode(int64_t inode, int64_t offset, int64_t size);
	void WriteInode(int64_t inode, const std::string &data, int64_t offset, bool append = false);
	void TruncateInode(int64_t inode, int64_t size);
	void SetAttributes(const std::string &path, std::optional<int64_t> mode = {}, std::optional<int64_t> mtime_ns = {},
	                   int64_t inode = 0, std::optional<int64_t> size = {});
	const std::string &Id() const {
		return id;
	}
	bool IsSnapshot() const {
		return snapshot;
	}

private:
	friend class Workspace;
	Session(std::shared_ptr<Database> database, std::string id, bool snapshot, std::string pin = {});
	std::shared_ptr<Database> database;
	std::string id, pin;
	bool snapshot, closed = false;
};

class Workspace {
public:
	explicit Workspace(const std::string &path, int timeout_ms = 5000);
	~Workspace();
	Workspace(const Workspace &) = delete;
	Workspace &operator=(const Workspace &) = delete;
	void Close();
	std::string Id() const;
	static std::string SQLiteVersion();
	BranchInfo GetBranch(const std::string &branch = "main");
	std::vector<BranchInfo> ListBranches();
	BranchInfo Fork(const std::string &source, const std::string &name, bool terminal = false);
	std::shared_ptr<Session> Checkout(const std::string &branch = "main");
	std::string Snapshot(const std::string &branch = "main");
	std::shared_ptr<Session> OpenSnapshot(const std::string &snapshot);
	void DropSnapshot(const std::string &snapshot);
	std::vector<Change> Diff(const std::string &source_snapshot, const std::string &target_snapshot);
	MergePreview PreviewMerge(const std::string &source, const std::string &target);
	// Conflict resolutions map paths to "source" or "target". A successful
	// direct-child merge seals the source. Stale previews never publish.
	void Merge(const MergePreview &preview, const std::map<std::string, std::string> &resolutions = {});
	void DeleteBranch(const std::string &branch, bool recursive = false);
	CollectionResult CollectGarbage();
	// Recovery only removes owners whose OS lock can be acquired. Unknown
	// legacy owners and unavailable lock files remain conservatively retained.
	RecoveryResult RecoverOwners();
	BranchInfo AcquireMount(const std::string &branch);
	void ReleaseMount(const std::string &branch);

private:
	std::shared_ptr<Database> database;
};

} // namespace vane_fs
