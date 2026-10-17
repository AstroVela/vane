// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/common/file_system.hpp"

#include <map>

namespace duckdb {
class ClientContext;
namespace vane_execution {

struct FrozenFile {
	string path;
	idx_t bytes;
	string sha256;
	string identity;
};

// Copy before binding/optimization. Files remain owned by the query store,
// including partial copies on failure, until its cleanup succeeds.
string FrozenFilePath(ClientContext &context, const string &source, const string &directory);
FrozenFile FreezeFile(ClientContext &context, const string &source, const string &destination, idx_t remaining,
                      bool *created = nullptr);
// Stable across reads, sensitive to rewrites (even restored mtime) and replacement.
string FileIdentity(ClientContext &context, const OpenFileInfo &source);

// A bounded, process-local index of immutable source versions. The caller owns
// the directory's durable lease and pins it for every query/worker that borrows
// a file. Full caches fall back to query-owned copies; files are never evicted
// while another query could still reference them.
class SourceSnapshotCache {
public:
	SourceSnapshotCache(const string &directory, idx_t capacity);
	~SourceSnapshotCache();
	FrozenFile Freeze(ClientContext &context, const string &source, const string &destination, idx_t remaining);
	std::map<string, idx_t> Snapshot() const;

private:
	struct Impl;
	unique_ptr<Impl> impl;
};

// Open-description locks, not POSIX process-associated record locks: closing
// another lease in the same process must not unlock this owner's reservation.
class StoreGuard {
public:
	~StoreGuard();
	static shared_ptr<StoreGuard> Acquire(const string &path, bool exclusive, bool create);
	void Close();

private:
	struct Impl;
	StoreGuard();
	unique_ptr<Impl> impl;
};

} // namespace vane_execution
} // namespace duckdb
