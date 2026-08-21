// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/map.hpp"
#include "duckdb/common/mutex.hpp"

namespace duckdb {

class ClientContext;
class DatabaseInstance;
class Deserializer;
class Serializer;
struct DistributedWriteOperatorExtension;

//! Coarse capability groups exposed by an extension to the distributed runner.
//! Ordinary DuckDB registration remains authoritative for local execution; these
//! entries describe only the additional protocol implemented by an extension.
enum class DistributedExtensionCapabilityKind : uint8_t { TABLE_FUNCTION = 0, WRITE_OPERATOR = 1 };

struct DistributedExtensionCapability {
	DistributedExtensionCapabilityKind kind = DistributedExtensionCapabilityKind::TABLE_FUNCTION;
	string name;
	idx_t protocol_version = 0;
	//! Canonical catalog overload signature. Required for table-function
	//! capabilities and empty for capability kinds that are not overloaded.
	string function_signature;

	DUCKDB_API string CanonicalIdentity() const;
	DUCKDB_API bool operator==(const DistributedExtensionCapability &other) const;
	DUCKDB_API bool operator<(const DistributedExtensionCapability &other) const;
};

//! Stable capability identity embedded by a distributed execution contract.
//! The runner resolves it against the database-local manifest before work is
//! scheduled; the extension continues to own the actual execution hooks.
struct DistributedExtensionCapabilityReference {
	string extension_name;
	DistributedExtensionCapability capability;

	DUCKDB_API void Validate() const;
	DUCKDB_API void Serialize(Serializer &serializer) const;
	DUCKDB_API static DistributedExtensionCapabilityReference Deserialize(Deserializer &deserializer);
	DUCKDB_API string CanonicalIdentity() const;
	DUCKDB_API bool operator==(const DistributedExtensionCapabilityReference &other) const;
	DUCKDB_API bool operator!=(const DistributedExtensionCapabilityReference &other) const;
};

//! Stable identity of an extension-owned opaque payload codec.
struct DistributedPayloadCodec {
	string name;
	idx_t version = 0;

	DUCKDB_API void Validate(const string &description) const;
	DUCKDB_API void Serialize(Serializer &serializer) const;
	DUCKDB_API static DistributedPayloadCodec Deserialize(Deserializer &deserializer);
	DUCKDB_API string CanonicalIdentity() const;
	DUCKDB_API bool operator==(const DistributedPayloadCodec &other) const;
	DUCKDB_API bool operator!=(const DistributedPayloadCodec &other) const;
};

//! A manifest is registered once by an extension during Extension::Load.
//! The extension's ordinary DuckDB version is transported separately by the
//! connection snapshot. Every concrete capability owns its protocol version.
struct DistributedExtensionManifest {
	string extension_name;
	vector<DistributedExtensionCapability> capabilities;

	DUCKDB_API string CanonicalIdentity() const;
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

	//! Atomically publish every concrete contract contributed by one extension.
	DUCKDB_API void RegisterExtension(const DistributedExtensionManifest &manifest,
	                                  vector<shared_ptr<const DistributedWriteOperatorExtension>> write_operators = {});

	DUCKDB_API vector<string> GetContractIdentities() const;
	DUCKDB_API void RequireCapability(const DistributedExtensionCapabilityReference &capability) const;
	DUCKDB_API shared_ptr<const DistributedWriteOperatorExtension>
	GetWriteOperator(const DistributedExtensionCapabilityReference &capability) const;
	DUCKDB_API shared_ptr<const DistributedWriteOperatorExtension> GetWriteOperator(const string &extension_name,
	                                                                                const string &operator_name) const;
	DUCKDB_API void ValidateExact(const vector<string> &expected_contract_identities) const;

	DUCKDB_API static DistributedExtensionManager &Get(DatabaseInstance &db);
	DUCKDB_API static DistributedExtensionManager &Get(ClientContext &context);
	DUCKDB_API static void ValidateManifest(const DistributedExtensionManifest &manifest);

private:
	mutable mutex lock;
	map<string, DistributedExtensionManifest> extensions;
	map<string, shared_ptr<const DistributedWriteOperatorExtension>> write_operators;
};

} // namespace duckdb
