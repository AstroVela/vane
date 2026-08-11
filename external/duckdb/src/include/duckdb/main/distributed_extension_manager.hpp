// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/map.hpp"
#include "duckdb/common/mutex.hpp"

namespace duckdb {

class ClientContext;
class DatabaseInstance;

//! Coarse capability groups exposed by an extension to the distributed runner.
//! Ordinary DuckDB registration remains authoritative for local execution; these
//! entries describe only the additional protocol implemented by an extension.
enum class DistributedExtensionCapabilityKind : uint8_t {
	TABLE_FUNCTION = 0,
	AGGREGATE_FUNCTION = 1,
	COPY_FUNCTION = 2,
	OPERATOR = 3,
	STORAGE = 4,
	CONTEXT = 5
};

DUCKDB_API string DistributedExtensionCapabilityKindToString(DistributedExtensionCapabilityKind kind);
DUCKDB_API DistributedExtensionCapabilityKind DistributedExtensionCapabilityKindFromString(const string &value);

struct DistributedExtensionCapability {
	DistributedExtensionCapabilityKind kind = DistributedExtensionCapabilityKind::TABLE_FUNCTION;
	string name;
	idx_t protocol_version = 0;

	DUCKDB_API string CanonicalIdentity() const;
	DUCKDB_API bool operator==(const DistributedExtensionCapability &other) const;
	DUCKDB_API bool operator<(const DistributedExtensionCapability &other) const;
};

//! Stable capability identity embedded by a per-plan distributed provider.
//! The runner resolves it against the database-local manifest before work is
//! scheduled; the provider object continues to own the actual bind/execute hook.
struct DistributedExtensionCapabilityReference {
	string extension_name;
	idx_t extension_protocol_version = 0;
	DistributedExtensionCapability capability;

	DUCKDB_API string CanonicalIdentity() const;
};

//! A manifest is registered once by an extension during Extension::Load.
//! The extension's ordinary DuckDB version is transported separately by the
//! connection snapshot; protocol_version versions the distributed manifest.
struct DistributedExtensionManifest {
	string extension_name;
	idx_t protocol_version = 0;
	vector<DistributedExtensionCapability> capabilities;

	DUCKDB_API string CanonicalIdentity() const;
	DUCKDB_API bool operator==(const DistributedExtensionManifest &other) const;
	DUCKDB_API bool operator<(const DistributedExtensionManifest &other) const;
};

//! Database-local registry for explicit distributed extension contracts.
//!
//! It deliberately stores protocol declarations rather than changing DuckDB's
//! normal function/type/catalog registrations. A native runner therefore keeps
//! executing the original extension path, while a distributed runner can demand
//! an exact coordinator/worker manifest match before scheduling any work.
class DistributedExtensionManager {
public:
	explicit DistributedExtensionManager(DatabaseInstance &db);

	DUCKDB_API void RegisterManifest(const DistributedExtensionManifest &manifest);

	DUCKDB_API bool TryGetExtension(const string &extension_name, DistributedExtensionManifest &result) const;
	DUCKDB_API vector<DistributedExtensionManifest> GetExtensions() const;
	DUCKDB_API void RequireCapability(const string &extension_name, DistributedExtensionCapabilityKind kind,
	                                  const string &capability_name, idx_t protocol_version) const;
	DUCKDB_API void RequireCapability(const DistributedExtensionCapabilityReference &capability) const;
	DUCKDB_API void ValidateExact(const vector<DistributedExtensionManifest> &expected) const;

	DUCKDB_API static DistributedExtensionManager &Get(DatabaseInstance &db);
	DUCKDB_API static DistributedExtensionManager &Get(ClientContext &context);
	DUCKDB_API static void ValidateManifest(const DistributedExtensionManifest &manifest);

private:
	mutable mutex lock;
	map<string, DistributedExtensionManifest> extensions;
};

} // namespace duckdb
