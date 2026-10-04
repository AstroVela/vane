// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/common/file_system.hpp"

namespace duckdb {
class ClientContext;
namespace vane_execution {

struct FrozenFile {
	string path;
	idx_t bytes;
	string sha256;
};

// Copy before binding/optimization. Files remain owned by the query store,
// including partial copies on failure, until its cleanup succeeds.
FrozenFile FreezeFile(ClientContext &context, const string &source, const string &directory, idx_t remaining);
string FileFingerprint(ClientContext &context, const OpenFileInfo &source);

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
