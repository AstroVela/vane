// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/function/distributed_write.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/set.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/execution/distributed/extension_write_task_provider.hpp"

namespace duckdb {

namespace {

static const string &WriteName(const DistributedExtensionWriteInfo &info) {
	return info.capability.capability.name;
}

static idx_t AddCount(idx_t total, idx_t value, const char *name) {
	if (value > NumericLimits<idx_t>::Maximum() - total) {
		throw SerializationException("distributed write %s total exceeds idx_t", name);
	}
	return total + value;
}

static string SerializeCopyFileInfo(const distributed::DistributedCopyFileInfo &file) {
	MemoryStream stream(Allocator::DefaultAllocator());
	BinarySerializer serializer(stream);
	serializer.Begin();
	serializer.WriteProperty(1, "staging_path", file.staging_path);
	serializer.WriteProperty(2, "final_path", file.final_path);
	serializer.WriteProperty(3, "row_count", file.row_count);
	serializer.WriteProperty(4, "file_size_bytes", file.file_size_bytes);
	serializer.WriteProperty(5, "footer_size_bytes", file.footer_size_bytes);
	serializer.WriteProperty(6, "column_statistics", file.column_statistics);
	serializer.WriteProperty(7, "partition_keys", file.partition_keys);
	serializer.End();
	return string(reinterpret_cast<const char *>(stream.GetData()), stream.GetPosition());
}

static distributed::DistributedCopyFileInfo DeserializeCopyFileInfo(const string &bytes) {
	if (bytes.empty()) {
		throw SerializationException("distributed file write fragment has an empty payload");
	}
	auto *data = reinterpret_cast<data_ptr_t>(const_cast<char *>(bytes.data()));
	MemoryStream stream(data, bytes.size());
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	distributed::DistributedCopyFileInfo result;
	result.staging_path = deserializer.ReadProperty<string>(1, "staging_path");
	result.final_path = deserializer.ReadProperty<string>(2, "final_path");
	result.row_count = deserializer.ReadProperty<idx_t>(3, "row_count");
	result.file_size_bytes = deserializer.ReadProperty<idx_t>(4, "file_size_bytes");
	result.footer_size_bytes = deserializer.ReadProperty<Value>(5, "footer_size_bytes");
	result.column_statistics = deserializer.ReadProperty<Value>(6, "column_statistics");
	result.partition_keys = deserializer.ReadProperty<Value>(7, "partition_keys");
	deserializer.End();
	return result;
}

} // namespace

void DistributedWriteArtifact::Validate() const {
	if (artifact_id.empty()) {
		throw SerializationException("distributed write artifact has an empty artifact_id");
	}
	codec.Validate("Distributed write artifact '" + artifact_id + "'");
}

void DistributedWriteArtifact::Serialize(Serializer &serializer) const {
	Validate();
	serializer.WriteProperty(1, "artifact_id", artifact_id);
	serializer.WriteProperty(2, "uri", uri);
	serializer.WriteObject(3, "codec", [&](Serializer &object) { codec.Serialize(object); });
	serializer.WriteProperty(4, "payload", payload);
}

DistributedWriteArtifact DistributedWriteArtifact::Deserialize(Deserializer &deserializer) {
	DistributedWriteArtifact result;
	result.artifact_id = deserializer.ReadProperty<string>(1, "artifact_id");
	result.uri = deserializer.ReadProperty<string>(2, "uri");
	deserializer.ReadObject(3, "codec",
	                        [&](Deserializer &object) { result.codec = DistributedPayloadCodec::Deserialize(object); });
	result.payload = deserializer.ReadProperty<string>(4, "payload");
	result.Validate();
	return result;
}

void DistributedWriteFragment::Validate() const {
	if (fragment_id.empty()) {
		throw SerializationException("distributed write fragment has an empty fragment_id");
	}
	set<string> artifact_ids;
	for (const auto &artifact : artifacts) {
		artifact.Validate();
		if (!artifact_ids.insert(artifact.artifact_id).second) {
			throw SerializationException("distributed write fragment '%s' has duplicate artifact '%s'", fragment_id,
			                             artifact.artifact_id);
		}
	}
}

void DistributedWriteFragment::Serialize(Serializer &serializer) const {
	Validate();
	serializer.WriteProperty(1, "fragment_id", fragment_id);
	serializer.WriteProperty(2, "payload", payload);
	serializer.WriteList(3, "artifacts", artifacts.size(), [&](Serializer::List &list, idx_t index) {
		list.WriteObject([&](Serializer &object) { artifacts[index].Serialize(object); });
	});
	serializer.WriteProperty(4, "row_count", row_count);
	serializer.WriteProperty(5, "byte_count", byte_count);
}

DistributedWriteFragment DistributedWriteFragment::Deserialize(Deserializer &deserializer) {
	DistributedWriteFragment result;
	result.fragment_id = deserializer.ReadProperty<string>(1, "fragment_id");
	result.payload = deserializer.ReadProperty<string>(2, "payload");
	deserializer.ReadList(3, "artifacts", [&](Deserializer::List &list, idx_t) {
		list.ReadObject(
		    [&](Deserializer &object) { result.artifacts.push_back(DistributedWriteArtifact::Deserialize(object)); });
	});
	result.row_count = deserializer.ReadProperty<idx_t>(4, "row_count");
	result.byte_count = deserializer.ReadProperty<idx_t>(5, "byte_count");
	result.Validate();
	return result;
}

void DistributedWriteTaskContext::Validate() const {
	if (query_id.empty()) {
		throw InvalidInputException("distributed extension write task requires a non-empty query identity");
	}
	if (task_attempt_id.empty()) {
		throw InvalidInputException("distributed extension write task requires a non-empty task-attempt identity");
	}
}

void DistributedWriteTaskResult::Validate() const {
	capability.Validate();
	if (capability.capability.kind != DistributedExtensionCapabilityKind::WRITE_OPERATOR) {
		throw SerializationException("distributed write task result capability is not a write operator");
	}
	fragment_codec.Validate("Distributed write task result");
	if (query_id.empty()) {
		throw SerializationException("distributed write task result has an empty query_id");
	}
	if (task_attempt_id.empty()) {
		throw SerializationException("distributed write task result has an empty task_attempt_id");
	}
	set<string> fragment_ids;
	for (const auto &fragment : fragments) {
		fragment.Validate();
		if (!fragment_ids.insert(fragment.fragment_id).second) {
			throw SerializationException("distributed write task result '%s' has duplicate fragment '%s'",
			                             task_attempt_id, fragment.fragment_id);
		}
	}
	(void)RowCount();
	(void)ByteCount();
}

idx_t DistributedWriteTaskResult::RowCount() const {
	idx_t result = 0;
	for (const auto &fragment : fragments) {
		result = AddCount(result, fragment.row_count, "row count");
	}
	return result;
}

idx_t DistributedWriteTaskResult::ByteCount() const {
	idx_t result = 0;
	for (const auto &fragment : fragments) {
		result = AddCount(result, fragment.byte_count, "byte count");
	}
	return result;
}

void DistributedWriteTaskResult::Serialize(Serializer &serializer) const {
	Validate();
	serializer.WriteObject(1, "capability", [&](Serializer &object) { capability.Serialize(object); });
	serializer.WriteObject(2, "fragment_codec", [&](Serializer &object) { fragment_codec.Serialize(object); });
	serializer.WriteProperty(3, "query_id", query_id);
	serializer.WriteProperty(4, "task_attempt_id", task_attempt_id);
	serializer.WriteList(5, "fragments", fragments.size(), [&](Serializer::List &list, idx_t index) {
		list.WriteObject([&](Serializer &object) { fragments[index].Serialize(object); });
	});
}

DistributedWriteTaskResult DistributedWriteTaskResult::Deserialize(Deserializer &deserializer) {
	DistributedWriteTaskResult result;
	deserializer.ReadObject(1, "capability", [&](Deserializer &object) {
		result.capability = DistributedExtensionCapabilityReference::Deserialize(object);
	});
	deserializer.ReadObject(2, "fragment_codec", [&](Deserializer &object) {
		result.fragment_codec = DistributedPayloadCodec::Deserialize(object);
	});
	result.query_id = deserializer.ReadProperty<string>(3, "query_id");
	result.task_attempt_id = deserializer.ReadProperty<string>(4, "task_attempt_id");
	deserializer.ReadList(5, "fragments", [&](Deserializer::List &list, idx_t) {
		list.ReadObject(
		    [&](Deserializer &object) { result.fragments.push_back(DistributedWriteFragment::Deserialize(object)); });
	});
	result.Validate();
	return result;
}

string DistributedWriteTaskResult::SerializeToBytes() const {
	MemoryStream stream(Allocator::DefaultAllocator());
	BinarySerializer serializer(stream);
	serializer.Begin();
	Serialize(serializer);
	serializer.End();
	return string(reinterpret_cast<const char *>(stream.GetData()), stream.GetPosition());
}

DistributedWriteTaskResult DistributedWriteTaskResult::DeserializeFromBytes(const string &bytes) {
	if (bytes.empty()) {
		throw SerializationException("cannot deserialize an empty distributed write task result");
	}
	auto *data = reinterpret_cast<data_ptr_t>(const_cast<char *>(bytes.data()));
	MemoryStream stream(data, bytes.size());
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto result = Deserialize(deserializer);
	deserializer.End();
	return result;
}

void DistributedExtensionWriteInfo::Validate() const {
	capability.Validate();
	if (capability.capability.kind != DistributedExtensionCapabilityKind::WRITE_OPERATOR) {
		throw InvalidInputException("distributed extension write capability must be a write operator");
	}
	fragment_codec.Validate("Distributed extension write '" + WriteName(*this) + "'");
	switch (mode) {
	case DistributedWriteMode::FILE_ARTIFACT:
		if (fragment_codec.name != distributed::DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC ||
		    fragment_codec.version != distributed::DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC_VERSION) {
			throw InvalidInputException(
			    "distributed file write '%s' must use %s@%llu", WriteName(*this),
			    distributed::DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC,
			    static_cast<unsigned long long>(distributed::DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC_VERSION));
		}
		if (!worker_bind_data.empty()) {
			throw InvalidInputException("distributed file write '%s' cannot carry callback worker bind data",
			                            WriteName(*this));
		}
		break;
	case DistributedWriteMode::CALLBACK_SINK:
		break;
	default:
		throw InvalidInputException("distributed extension write '%s' has an unknown mode", WriteName(*this));
	}
}

const string &DistributedExtensionWriteInfo::Name() const {
	return WriteName(*this);
}

void DistributedExtensionWriteInfo::Serialize(Serializer &serializer) const {
	Validate();
	serializer.WriteObject(1, "capability", [&](Serializer &object) { capability.Serialize(object); });
	serializer.WriteProperty(2, "mode", static_cast<uint8_t>(mode));
	serializer.WriteObject(3, "fragment_codec", [&](Serializer &object) { fragment_codec.Serialize(object); });
	serializer.WriteProperty(4, "worker_bind_data", worker_bind_data);
}

DistributedExtensionWriteInfo DistributedExtensionWriteInfo::Deserialize(Deserializer &deserializer) {
	DistributedExtensionWriteInfo result;
	deserializer.ReadObject(1, "capability", [&](Deserializer &object) {
		result.capability = DistributedExtensionCapabilityReference::Deserialize(object);
	});
	result.mode = static_cast<DistributedWriteMode>(deserializer.ReadProperty<uint8_t>(2, "mode"));
	deserializer.ReadObject(3, "fragment_codec", [&](Deserializer &object) {
		result.fragment_codec = DistributedPayloadCodec::Deserialize(object);
	});
	result.worker_bind_data = deserializer.ReadProperty<string>(4, "worker_bind_data");
	result.Validate();
	return result;
}

void DistributedExtensionWriteCallbacks::Validate(const string &capability_identity) const {
	if (!initialize_global || !initialize_local || !sink || !combine || !finalize) {
		throw InvalidInputException("distributed write callbacks for %s are incomplete", capability_identity);
	}
}

bool DistributedExtensionWriteCallbacks::Empty() const {
	return !initialize_global && !initialize_local && !sink && !combine && !finalize;
}

void DistributedWriteOperatorExtension::Validate(const string &capability_identity) const {
	fragment_codec.Validate("Distributed write operator '" + capability_identity + "'");
	switch (mode) {
	case DistributedWriteMode::FILE_ARTIFACT:
		if (fragment_codec.name != distributed::DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC ||
		    fragment_codec.version != distributed::DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC_VERSION) {
			throw InvalidInputException(
			    "Distributed file write operator '%s' must use %s@%llu", capability_identity,
			    distributed::DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC,
			    static_cast<unsigned long long>(distributed::DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC_VERSION));
		}
		if (!callbacks.Empty()) {
			throw InvalidInputException("Distributed file write operator '%s' cannot register worker callbacks",
			                            capability_identity);
		}
		return;
	case DistributedWriteMode::CALLBACK_SINK:
		callbacks.Validate(capability_identity);
		return;
	default:
		throw InvalidInputException("Distributed write operator '%s' has an unknown mode", capability_identity);
	}
}

namespace distributed {

void DistributedExtensionWritePlan::Validate() const {
	if (extension_name.empty() || operator_name.empty()) {
		throw InvalidInputException("distributed extension write plan requires extension and operator names");
	}
}

DistributedExtensionWriteInfo ResolveDistributedExtensionWriteInfo(ClientContext &context,
                                                                   const DistributedExtensionWritePlan &plan) {
	plan.Validate();
	auto write_operator =
	    DistributedExtensionManager::Get(context).GetWriteOperator(plan.extension_name, plan.operator_name);
	DistributedExtensionWriteInfo result;
	result.capability.extension_name = plan.extension_name;
	result.capability.capability = {DistributedExtensionCapabilityKind::WRITE_OPERATOR, write_operator->name,
	                                write_operator->protocol_version};
	result.mode = write_operator->mode;
	result.fragment_codec = write_operator->fragment_codec;
	result.worker_bind_data = plan.worker_bind_data;
	result.Validate();
	return result;
}

vector<DistributedWriteTaskResult> EncodeDistributedFileWriteResults(const DistributedExtensionWriteInfo &info,
                                                                     const string &query_id,
                                                                     const vector<DistributedCopyFileInfo> &files) {
	info.Validate();
	if (query_id.empty()) {
		throw InvalidInputException("distributed extension write requires a non-empty query identity");
	}
	if (info.mode != DistributedWriteMode::FILE_ARTIFACT) {
		throw InvalidInputException("distributed extension write '%s' is not a file-artifact write", WriteName(info));
	}
	vector<DistributedWriteTaskResult> results;
	results.reserve(files.size());
	set<string> fragment_ids;
	for (idx_t index = 0; index < files.size(); index++) {
		const auto &file = files[index];
		const auto &path = file.final_path.empty() ? file.staging_path : file.final_path;
		if (path.empty()) {
			throw InvalidInputException("distributed file write '%s' returned an empty file path", WriteName(info));
		}
		if (!fragment_ids.insert(path).second) {
			throw InvalidInputException("distributed file write '%s' selected file '%s' more than once",
			                            WriteName(info), path);
		}
		DistributedWriteArtifact artifact;
		artifact.artifact_id = "data_file";
		artifact.uri = path;
		artifact.codec = {"duckdb.file", 1};

		DistributedWriteFragment fragment;
		fragment.fragment_id = path;
		fragment.payload = SerializeCopyFileInfo(file);
		fragment.artifacts.push_back(std::move(artifact));
		fragment.row_count = file.row_count;
		fragment.byte_count = file.file_size_bytes;

		DistributedWriteTaskResult result;
		result.capability = info.capability;
		result.fragment_codec = info.fragment_codec;
		result.query_id = query_id;
		result.task_attempt_id = "file:" + path;
		result.fragments.push_back(std::move(fragment));
		result.Validate();
		results.push_back(std::move(result));
	}
	return results;
}

vector<DistributedCopyFileInfo> DecodeDistributedFileWriteResults(const DistributedExtensionWriteInfo &info,
                                                                  const vector<DistributedWriteTaskResult> &results) {
	info.Validate();
	if (info.mode != DistributedWriteMode::FILE_ARTIFACT) {
		throw InvalidInputException("distributed extension write '%s' is not a file-artifact write", WriteName(info));
	}
	vector<DistributedCopyFileInfo> files;
	set<string> task_attempt_ids;
	set<string> fragment_ids;
	string query_id;
	for (const auto &result : results) {
		result.Validate();
		if (result.capability != info.capability || result.fragment_codec != info.fragment_codec) {
			throw InvalidInputException("distributed file write '%s' received a mismatched task result protocol",
			                            WriteName(info));
		}
		if (query_id.empty()) {
			query_id = result.query_id;
		} else if (result.query_id != query_id) {
			throw InvalidInputException("distributed file write '%s' received results from multiple query identities",
			                            WriteName(info));
		}
		if (!task_attempt_ids.insert(result.task_attempt_id).second) {
			throw InvalidInputException("distributed file write '%s' selected task attempt '%s' more than once",
			                            WriteName(info), result.task_attempt_id);
		}
		for (const auto &fragment : result.fragments) {
			if (!fragment_ids.insert(fragment.fragment_id).second) {
				throw InvalidInputException("distributed file write '%s' selected fragment '%s' more than once",
				                            WriteName(info), fragment.fragment_id);
			}
			if (fragment.artifacts.size() != 1 || fragment.artifacts[0].artifact_id != "data_file" ||
			    fragment.artifacts[0].codec != DistributedPayloadCodec {"duckdb.file", 1} ||
			    !fragment.artifacts[0].payload.empty()) {
				throw InvalidInputException("distributed file write '%s' received an invalid file artifact",
				                            WriteName(info));
			}
			auto file = DeserializeCopyFileInfo(fragment.payload);
			const auto &path = file.final_path.empty() ? file.staging_path : file.final_path;
			if (path != fragment.fragment_id || path != fragment.artifacts[0].uri ||
			    file.row_count != fragment.row_count || file.file_size_bytes != fragment.byte_count) {
				throw InvalidInputException("distributed file write '%s' fragment metadata does not match its file DTO",
				                            WriteName(info));
			}
			files.push_back(std::move(file));
		}
	}
	return files;
}

vector<DistributedWriteTaskResult> ParseDistributedWriteTaskResults(const DistributedExtensionWriteInfo &info,
                                                                    const string &query_id,
                                                                    const vector<ResultPartitionRef> &partitions) {
	info.Validate();
	if (query_id.empty()) {
		throw InvalidInputException("distributed extension write requires a non-empty query identity");
	}
	if (info.mode != DistributedWriteMode::CALLBACK_SINK) {
		throw InvalidInputException("distributed extension write '%s' is not a callback write", WriteName(info));
	}
	vector<DistributedWriteTaskResult> results;
	set<string> task_attempt_ids;
	set<string> fragment_ids;
	for (idx_t partition_index = 0; partition_index < partitions.size(); partition_index++) {
		const auto &partition = partitions[partition_index];
		auto collection = partition ? partition->to_column_data() : nullptr;
		if (!collection) {
			throw InvalidInputException("distributed extension write '%s' partition %llu is not tabular",
			                            WriteName(info), static_cast<unsigned long long>(partition_index));
		}
		if (collection->Types().size() != 1 || collection->Types()[0].id() != LogicalTypeId::BLOB) {
			throw InvalidInputException(
			    "distributed extension write '%s' partition %llu must contain exactly one BLOB column", WriteName(info),
			    static_cast<unsigned long long>(partition_index));
		}
		if (collection->Count() != 1) {
			throw InvalidInputException(
			    "distributed extension write '%s' partition %llu must contain exactly one task envelope",
			    WriteName(info), static_cast<unsigned long long>(partition_index));
		}
		ColumnDataScanState scan_state;
		collection->InitializeScan(scan_state);
		DataChunk chunk;
		collection->InitializeScanChunk(chunk);
		while (collection->Scan(scan_state, chunk)) {
			if (chunk.ColumnCount() != 1) {
				throw InvalidInputException("distributed extension write '%s' result schema changed while scanning",
				                            WriteName(info));
			}
			for (idx_t row = 0; row < chunk.size(); row++) {
				auto value = chunk.GetValue(0, row);
				if (value.IsNull()) {
					throw InvalidInputException("distributed extension write '%s' returned a NULL task envelope",
					                            WriteName(info));
				}
				auto result = DistributedWriteTaskResult::DeserializeFromBytes(StringValue::Get(value));
				if (result.query_id != query_id || result.capability != info.capability ||
				    result.fragment_codec != info.fragment_codec) {
					throw InvalidInputException(
					    "distributed extension write '%s' received a mismatched task result protocol", WriteName(info));
				}
				if (!task_attempt_ids.insert(result.task_attempt_id).second) {
					throw InvalidInputException(
					    "distributed extension write '%s' selected task attempt '%s' more than once", WriteName(info),
					    result.task_attempt_id);
				}
				for (const auto &fragment : result.fragments) {
					if (!fragment_ids.insert(fragment.fragment_id).second) {
						throw InvalidInputException(
						    "distributed extension write '%s' selected fragment '%s' more than once", WriteName(info),
						    fragment.fragment_id);
					}
				}
				results.push_back(std::move(result));
			}
		}
	}
	return results;
}

} // namespace distributed
} // namespace duckdb
