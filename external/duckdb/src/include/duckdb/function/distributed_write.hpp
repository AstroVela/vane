// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/main/distributed_extension_manager.hpp"

namespace duckdb {

class ClientContext;
class DataChunk;
class Deserializer;
class ExecutionContext;
class ExtensionLoader;
class Serializer;

//! Selects the worker-side implementation used by an extension write.
//! FILE_ARTIFACT consumes DuckDB WRITTEN_FILE_STATISTICS. CALLBACK_SINK replaces
//! the coordinator-only extension root with the registered streaming callback
//! sink on every worker.
enum class DistributedWriteMode : uint8_t { FILE_ARTIFACT = 0, CALLBACK_SINK = 1 };

//! An immutable object created by a worker write. The extension owns the
//! artifact codec and payload. URI is optional for non-file artifacts. Worker
//! attempts must use immutable, non-overwriting objects.
struct DistributedWriteArtifact {
	string artifact_id;
	string uri;
	DistributedPayloadCodec codec;
	string payload;

	DUCKDB_API void Validate() const;
	DUCKDB_API void Serialize(Serializer &serializer) const;
	DUCKDB_API static DistributedWriteArtifact Deserialize(Deserializer &deserializer);
};

//! One portable extension-owned commit fragment. Payload and artifacts are
//! deliberately opaque to Vane; row/byte counts are protocol metadata used for
//! validation and reporting.
struct DistributedWriteFragment {
	string fragment_id;
	string payload;
	vector<DistributedWriteArtifact> artifacts;
	idx_t row_count = 0;
	idx_t byte_count = 0;

	DUCKDB_API void Validate() const;
	DUCKDB_API void Serialize(Serializer &serializer) const;
	DUCKDB_API static DistributedWriteFragment Deserialize(Deserializer &deserializer);
};

//! Runtime identity supplied explicitly to every worker-side callback. query_id
//! groups all attempts from one execution; task_attempt_id distinguishes
//! speculative and retried worker attempts. Neither value is a durable commit
//! idempotency key.
struct DistributedWriteTaskContext {
	string query_id;
	string task_attempt_id;

	DUCKDB_API void Validate() const;
};

//! Portable outer envelope returned by a selected callback worker task. The
//! fixed file adapter uses the same shape with one deterministic synthetic
//! task_attempt_id per selected file. Query, capability, and codec identities
//! are repeated in every envelope so a coordinator never mixes worker output
//! from different executions or extension interfaces.
struct DistributedWriteTaskResult {
	DistributedExtensionCapabilityReference capability;
	DistributedPayloadCodec fragment_codec;
	string query_id;
	string task_attempt_id;
	vector<DistributedWriteFragment> fragments;

	DUCKDB_API void Validate() const;
	DUCKDB_API idx_t RowCount() const;
	DUCKDB_API idx_t ByteCount() const;
	DUCKDB_API void Serialize(Serializer &serializer) const;
	DUCKDB_API static DistributedWriteTaskResult Deserialize(Deserializer &deserializer);
	DUCKDB_API string SerializeToBytes() const;
	DUCKDB_API static DistributedWriteTaskResult DeserializeFromBytes(const string &bytes);
};

//! Immutable coordinator plan embedded in the distributed physical write.
//! worker_bind_data is extension-owned binary state, not JSON and not a
//! process-local pointer.
struct DistributedExtensionWriteInfo {
	DistributedExtensionCapabilityReference capability;
	DistributedWriteMode mode = DistributedWriteMode::FILE_ARTIFACT;
	DistributedPayloadCodec fragment_codec;
	string worker_bind_data;

	DUCKDB_API const string &Name() const;
	DUCKDB_API void Validate() const;
	DUCKDB_API void Serialize(Serializer &serializer) const;
	DUCKDB_API static DistributedExtensionWriteInfo Deserialize(Deserializer &deserializer);
};

//! Extension-owned state used only by worker callbacks.
class DistributedWriteGlobalState {
public:
	virtual ~DistributedWriteGlobalState() = default;

	template <class TARGET>
	TARGET &Cast() {
		DynamicCastCheck<TARGET>(this);
		return reinterpret_cast<TARGET &>(*this);
	}
	template <class TARGET>
	const TARGET &Cast() const {
		DynamicCastCheck<TARGET>(this);
		return reinterpret_cast<const TARGET &>(*this);
	}
};

class DistributedWriteLocalState {
public:
	virtual ~DistributedWriteLocalState() = default;

	template <class TARGET>
	TARGET &Cast() {
		DynamicCastCheck<TARGET>(this);
		return reinterpret_cast<TARGET &>(*this);
	}
	template <class TARGET>
	const TARGET &Cast() const {
		DynamicCastCheck<TARGET>(this);
		return reinterpret_cast<const TARGET &>(*this);
	}
};

using distributed_write_initialize_global_t = unique_ptr<DistributedWriteGlobalState> (*)(
    ClientContext &context, const DistributedExtensionWriteInfo &info, const DistributedWriteTaskContext &task);
using distributed_write_initialize_local_t = unique_ptr<DistributedWriteLocalState> (*)(
    ExecutionContext &context, const DistributedExtensionWriteInfo &info, const DistributedWriteTaskContext &task,
    DistributedWriteGlobalState &global_state);
using distributed_write_sink_t = void (*)(ExecutionContext &context, const DistributedExtensionWriteInfo &info,
                                          const DistributedWriteTaskContext &task,
                                          DistributedWriteGlobalState &global_state,
                                          DistributedWriteLocalState &local_state, DataChunk &input);
using distributed_write_combine_t = void (*)(ExecutionContext &context, const DistributedExtensionWriteInfo &info,
                                             const DistributedWriteTaskContext &task,
                                             DistributedWriteGlobalState &global_state,
                                             DistributedWriteLocalState &local_state);
using distributed_write_finalize_t = vector<DistributedWriteFragment> (*)(ClientContext &context,
                                                                          const DistributedExtensionWriteInfo &info,
                                                                          const DistributedWriteTaskContext &task,
                                                                          DistributedWriteGlobalState &global_state);

//! Complete worker-side streaming write implementation registered by a static
//! extension. Every callback is mandatory; missing callbacks are hard errors.
struct DistributedExtensionWriteCallbacks {
	distributed_write_initialize_global_t initialize_global = nullptr;
	distributed_write_initialize_local_t initialize_local = nullptr;
	distributed_write_sink_t sink = nullptr;
	distributed_write_combine_t combine = nullptr;
	distributed_write_finalize_t finalize = nullptr;

	DUCKDB_API void Validate(const string &capability_identity) const;
	DUCKDB_API bool Empty() const;
};

//! One worker-side distributed write hook. Like DuckDB's OperatorExtension,
//! this is not a loadable Extension subclass. A top-level Extension registers
//! the hook independently from catalog functions. ExtensionLoader supplies the
//! current extension identity and publishes the hook atomically with the
//! derived extension manifest.
struct DistributedWriteOperatorExtension {
	string name;
	idx_t protocol_version = 0;
	DistributedWriteMode mode = DistributedWriteMode::FILE_ARTIFACT;
	DistributedPayloadCodec fragment_codec;
	DistributedExtensionWriteCallbacks callbacks;

	DUCKDB_API void Validate(const string &capability_identity) const;
	DUCKDB_API static void Register(ExtensionLoader &loader, DistributedWriteOperatorExtension extension);
};

} // namespace duckdb
