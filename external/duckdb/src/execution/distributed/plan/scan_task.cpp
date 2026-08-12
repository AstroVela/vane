// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/distributed/plan/scan_task.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/execution/distributed/plan/scan_task.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/multi_file/multi_file_list.hpp"
#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/types/blob.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "duckdb/common/set.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/execution/distributed/plan/fte_split_queue.hpp"

#include <mutex>
#include <limits>

namespace duckdb {
namespace distributed {

namespace {

struct ApplyScanTasksStats {
	idx_t table_scans = 0;
	idx_t applied = 0;
	idx_t missing_node_id = 0;
	idx_t missing_task = 0;
	idx_t missing_bind = 0;
	idx_t non_multi_bind = 0;
	idx_t invalid_assignment = 0;
	idx_t duplicate_node_id = 0;
	idx_t copied_tasks = 0;
	idx_t missing_group_id = 0;
};

class FteDynamicScanFileList : public MultiFileList {
private:
	struct State {
		explicit State(std::shared_ptr<FteSplitQueue> queue_p) : queue(std::move(queue_p)) {
		}

		std::shared_ptr<FteSplitQueue> queue;
		mutable std::mutex mutex;
		mutable std::mutex load_mutex;
		vector<OpenFileInfo> files;
		bool finished = false;
	};

public:
	explicit FteDynamicScanFileList(std::shared_ptr<FteSplitQueue> queue_p)
	    : state(std::make_shared<State>(std::move(queue_p))) {
	}

	explicit FteDynamicScanFileList(std::shared_ptr<State> state_p) : state(std::move(state_p)) {
	}

	vector<OpenFileInfo> GetAllFiles() const override {
		LoadUntilFinished();
		std::lock_guard<std::mutex> lock(state->mutex);
		return state->files;
	}

	FileExpandResult GetExpandResult() const override {
		std::lock_guard<std::mutex> lock(state->mutex);
		if (state->files.size() > 1) {
			return FileExpandResult::MULTIPLE_FILES;
		}
		if (state->files.size() == 1) {
			return FileExpandResult::SINGLE_FILE;
		}
		return state->finished ? FileExpandResult::NO_FILES : FileExpandResult::MULTIPLE_FILES;
	}

	idx_t GetTotalFileCount() const override {
		LoadUntilFinished();
		std::lock_guard<std::mutex> lock(state->mutex);
		return state->files.size();
	}

	MultiFileCount GetFileCount(idx_t min_exact_count = 0) const override {
		{
			std::lock_guard<std::mutex> lock(state->mutex);
			if (state->finished || state->files.size() >= min_exact_count) {
				return MultiFileCount(state->files.size(), state->finished ? FileExpansionType::ALL_FILES_EXPANDED
				                                                           : FileExpansionType::NOT_ALL_FILES_KNOWN);
			}
		}
		LoadUntilAtLeast(min_exact_count);
		std::lock_guard<std::mutex> lock(state->mutex);
		return MultiFileCount(state->files.size(), state->finished ? FileExpansionType::ALL_FILES_EXPANDED
		                                                           : FileExpansionType::NOT_ALL_FILES_KNOWN);
	}

	vector<OpenFileInfo> GetDisplayFileList(optional_idx max_files = optional_idx()) const override {
		if (max_files.IsValid()) {
			LoadUntilAtLeast(max_files.GetIndex());
		} else {
			LoadUntilFinished();
		}
		std::lock_guard<std::mutex> lock(state->mutex);
		vector<OpenFileInfo> result;
		idx_t limit = state->files.size();
		if (max_files.IsValid()) {
			limit = MinValue<idx_t>(limit, max_files.GetIndex());
		}
		result.reserve(limit);
		for (idx_t i = 0; i < limit; i++) {
			result.push_back(state->files[i]);
		}
		return result;
	}

	unique_ptr<MultiFileList> Copy() const override {
		return make_uniq<FteDynamicScanFileList>(state);
	}

protected:
	bool FileIsAvailable(idx_t i) const override {
		std::lock_guard<std::mutex> lock(state->mutex);
		return i < state->files.size() || state->finished;
	}

	OpenFileInfo GetFile(idx_t i) const override {
		LoadUntilAtLeast(i + 1);
		std::lock_guard<std::mutex> lock(state->mutex);
		if (i < state->files.size()) {
			return state->files[i];
		}
		return OpenFileInfo();
	}

private:
	void LoadUntilAtLeast(idx_t count) const {
		while (true) {
			{
				std::lock_guard<std::mutex> lock(state->mutex);
				if (state->finished || state->files.size() >= count) {
					return;
				}
			}
			if (!LoadNextSplit()) {
				return;
			}
		}
	}

	void LoadUntilFinished() const {
		while (true) {
			{
				std::lock_guard<std::mutex> lock(state->mutex);
				if (state->finished) {
					return;
				}
			}
			if (!LoadNextSplit()) {
				return;
			}
		}
	}

	bool LoadNextSplit() const {
		std::lock_guard<std::mutex> load_lock(state->load_mutex);
		{
			std::lock_guard<std::mutex> lock(state->mutex);
			if (state->finished) {
				return false;
			}
		}
		auto next = state->queue->WaitForNext();
		std::lock_guard<std::mutex> lock(state->mutex);
		if (next.state == FteSplitQueue::GetResult::CANCELED || next.state == FteSplitQueue::GetResult::FINISHED) {
			state->finished = true;
			return false;
		}
		if (next.state != FteSplitQueue::GetResult::SPLIT) {
			return false;
		}
		if (next.input.kind != TaskInput::Kind::ScanTask) {
			throw InvalidInputException("dynamic scan source queue received non-scan split");
		}
		auto descriptor = ScanTaskDescriptor::DeserializeFromBytes(next.input.scan_task_bytes);
		if (descriptor.kind != ScanTaskKind::FILES) {
			throw InvalidInputException("dynamic MultiFile scan source received an extension scan task");
		}
		if (descriptor.files.empty()) {
			return true;
		}
		for (auto &file : descriptor.files) {
			state->files.push_back(std::move(file));
		}
		return true;
	}

	std::shared_ptr<State> state;
};

static idx_t MaxScanNodeId(const PhysicalOperator &op, idx_t max_id) {
	if (op.type == PhysicalOperatorType::TABLE_SCAN) {
		auto &scan = op.Cast<PhysicalTableScan>();
		if (scan.extra_info.scan_node_id.IsValid()) {
			const auto id = scan.extra_info.scan_node_id.GetIndex();
			if (id > max_id) {
				max_id = id;
			}
		}
	}
	for (auto &child : op.children) {
		max_id = MaxScanNodeId(child.get(), max_id);
	}
	return max_id;
}

static void CollectScanNodeIds(const PhysicalOperator &op, set<idx_t> &node_ids) {
	if (op.type == PhysicalOperatorType::TABLE_SCAN) {
		auto &scan = op.Cast<PhysicalTableScan>();
		if (scan.extra_info.scan_node_id.IsValid()) {
			node_ids.insert(scan.extra_info.scan_node_id.GetIndex());
		}
	}
	for (const auto &child : op.children) {
		CollectScanNodeIds(child.get(), node_ids);
	}
}

static void SetApplyError(string *error, const string &message) {
	if (error && error->empty()) {
		*error = message;
	}
}

static bool CollectRequiredScanNodeIds(const PhysicalOperator &op, set<idx_t> &node_ids, string *error) {
	if (op.type == PhysicalOperatorType::TABLE_SCAN) {
		auto &scan = op.Cast<PhysicalTableScan>();
		if (!scan.extra_info.scan_node_id.IsValid()) {
			SetApplyError(error,
			              "distributed table scan '" + scan.function.name + "' has no runtime scan node identity");
			return false;
		}
		node_ids.insert(scan.extra_info.scan_node_id.GetIndex());
	}
	for (const auto &child : op.children) {
		if (!CollectRequiredScanNodeIds(child.get(), node_ids, error)) {
			return false;
		}
	}
	return true;
}

static void NormalizeScanNodeIdsByGroup(PhysicalOperator &op, std::unordered_map<idx_t, idx_t> &base_for_group,
                                        std::unordered_map<idx_t, idx_t> &dup_to_base, idx_t &next_id,
                                        ApplyScanTasksStats &stats) {
	if (op.type == PhysicalOperatorType::TABLE_SCAN) {
		auto &scan = op.Cast<PhysicalTableScan>();
		if (!scan.extra_info.scan_group_id.IsValid()) {
			stats.missing_group_id++;
			if (scan.extra_info.scan_node_id.IsValid()) {
				scan.extra_info.scan_group_id = scan.extra_info.scan_node_id;
			} else {
				scan.extra_info.scan_group_id = optional_idx(next_id++);
			}
		}
		if (!scan.extra_info.scan_node_id.IsValid()) {
			scan.extra_info.scan_node_id = optional_idx(next_id++);
		}

		const idx_t group_id = scan.extra_info.scan_group_id.GetIndex();
		idx_t node_id = scan.extra_info.scan_node_id.GetIndex();
		auto it = base_for_group.find(group_id);
		if (it == base_for_group.end()) {
			base_for_group[group_id] = node_id;
		} else {
			const idx_t base_id = it->second;
			if (node_id == base_id) {
				const idx_t new_id = next_id++;
				scan.extra_info.scan_node_id = optional_idx(new_id);
				node_id = new_id;
				stats.duplicate_node_id++;
			}
			dup_to_base[node_id] = base_id;
		}
	}
	for (auto &child : op.children) {
		NormalizeScanNodeIdsByGroup(child.get(), base_for_group, dup_to_base, next_id, stats);
	}
}
static bool ApplyExtensionScanTasks(PhysicalTableScan &scan, const ScanTaskDescriptor &descriptor, string *error) {
	if (!scan.function.HasDistributedScanCallbacks()) {
		SetApplyError(error, "extension scan task assigned to table function without distributed callbacks: " +
		                         scan.function.name);
		return false;
	}
	const auto &callbacks = scan.function.GetDistributedScanCallbacks();
	callbacks.Validate(scan.function.name);
	if (descriptor.extension_capability != callbacks.capability) {
		SetApplyError(error, "distributed scan capability mismatch for table function '" + scan.function.name +
		                         "': task=" + descriptor.extension_capability.CanonicalIdentity() +
		                         ", worker=" + callbacks.capability.CanonicalIdentity());
		return false;
	}
	if (descriptor.task_codec != callbacks.task_codec ||
	    descriptor.task_codec_version != callbacks.task_codec_version) {
		SetApplyError(error, "distributed scan task codec mismatch for table function '" + scan.function.name +
		                         "': task=" + descriptor.task_codec + "@" +
		                         std::to_string(descriptor.task_codec_version) + ", worker=" + callbacks.task_codec +
		                         "@" + std::to_string(callbacks.task_codec_version));
		return false;
	}
	callbacks.apply_tasks(*scan.bind_data, descriptor.extension_tasks);
	scan.extra_info.total_files = optional_idx(descriptor.task_count());
	scan.extra_info.filtered_files = optional_idx(descriptor.task_count());
	scan.distributed_scan_tasks_applied = true;
	return true;
}

using ScanTaskReferenceMap = std::unordered_map<idx_t, const ScanTaskDescriptor *>;

static bool ApplyScanTasksToOperator(PhysicalOperator &op, const ScanTaskReferenceMap &tasks, set<idx_t> &matched_tasks,
                                     ApplyScanTasksStats &stats, string *error) {
	bool applied_any = false;
	if (op.type == PhysicalOperatorType::TABLE_SCAN) {
		stats.table_scans++;
		auto &scan = op.Cast<PhysicalTableScan>();
		if (!scan.extra_info.scan_node_id.IsValid()) {
			stats.missing_node_id++;
		} else {
			const idx_t node_id = scan.extra_info.scan_node_id.GetIndex();
			auto it = tasks.find(node_id);
			if (it == tasks.end()) {
				stats.missing_task++;
			} else if (!scan.bind_data) {
				matched_tasks.insert(node_id);
				stats.missing_bind++;
				stats.invalid_assignment++;
				SetApplyError(error, "scan task assigned to table function with null bind data: " + scan.function.name);
			} else {
				matched_tasks.insert(node_id);
				const auto &descriptor = *it->second;
				if (descriptor.kind == ScanTaskKind::EXTENSION) {
					if (ApplyExtensionScanTasks(scan, descriptor, error)) {
						stats.applied++;
						applied_any = true;
					} else {
						stats.non_multi_bind++;
						stats.invalid_assignment++;
					}
				} else if (scan.function.HasDistributedScanCallbacks()) {
					SetApplyError(error,
					              "file scan task assigned to extension table function '" + scan.function.name + "'");
					stats.non_multi_bind++;
					stats.invalid_assignment++;
				} else if (auto *multi_bind = dynamic_cast<MultiFileBindData *>(scan.bind_data.get())) {
					multi_bind->file_list = duckdb::make_shared_ptr<SimpleMultiFileList>(descriptor.files);
					const idx_t file_count = descriptor.file_count();
					scan.extra_info.total_files = optional_idx(file_count);
					scan.extra_info.filtered_files = optional_idx(file_count);
					scan.distributed_scan_tasks_applied = true;
					stats.applied++;
					applied_any = true;
				} else {
					stats.non_multi_bind++;
					stats.invalid_assignment++;
					SetApplyError(error,
					              "file scan task assigned to non-MultiFile table function: " + scan.function.name);
				}
			}
		}
	}

	for (auto &child : op.children) {
		if (ApplyScanTasksToOperator(child.get(), tasks, matched_tasks, stats, error)) {
			applied_any = true;
		}
	}
	return applied_any;
}

} // namespace

static idx_t SaturatingAddScanTaskEstimate(idx_t left, idx_t right) {
	const auto maximum = std::numeric_limits<idx_t>::max();
	return right > maximum - left ? maximum : left + right;
}

static void ComputeExtensionTaskEstimates(const vector<DistributedScanTask> &tasks, idx_t &cardinality, idx_t &bytes) {
	cardinality = 0;
	bytes = 0;
	bool complete_cardinality = true;
	bool complete_bytes = true;
	for (const auto &task : tasks) {
		if (task.estimated_cardinality.IsValid()) {
			cardinality = SaturatingAddScanTaskEstimate(cardinality, task.estimated_cardinality.GetIndex());
		} else {
			complete_cardinality = false;
		}
		if (task.estimated_bytes.IsValid()) {
			bytes = SaturatingAddScanTaskEstimate(bytes, task.estimated_bytes.GetIndex());
		} else {
			complete_bytes = false;
		}
	}
	if (!complete_cardinality) {
		cardinality = 0;
	}
	if (!complete_bytes) {
		bytes = 0;
	}
}

void ScanTaskDescriptor::Validate() const {
	switch (kind) {
	case ScanTaskKind::FILES:
		if (!extension_tasks.empty() || !extension_capability.extension_name.empty() || !task_codec.empty() ||
		    task_codec_version != 0) {
			throw SerializationException("file scan task descriptor contains extension task state");
		}
		break;
	case ScanTaskKind::EXTENSION: {
		if (!files.empty()) {
			throw SerializationException("extension scan task descriptor contains OpenFileInfo entries");
		}
		if (task_codec.empty() || task_codec_version == 0) {
			throw SerializationException("extension scan task descriptor has an invalid task codec identity");
		}
		if (extension_capability.capability.kind != DistributedExtensionCapabilityKind::TABLE_FUNCTION) {
			throw SerializationException("extension scan task descriptor capability is not a table function");
		}
		DistributedExtensionManifest manifest;
		manifest.extension_name = extension_capability.extension_name;
		manifest.protocol_version = extension_capability.extension_protocol_version;
		manifest.capabilities.push_back(extension_capability.capability);
		DistributedExtensionManager::ValidateManifest(manifest);

		set<string> task_ids;
		for (const auto &task : extension_tasks) {
			if (task.task_id.empty()) {
				throw SerializationException("extension scan task has an empty task_id");
			}
			if (!task_ids.insert(task.task_id).second) {
				throw SerializationException("extension scan task_id '%s' appears more than once", task.task_id);
			}
			set<string> artifact_names;
			for (const auto &artifact : task.artifacts) {
				if (artifact.name.empty()) {
					throw SerializationException("extension scan task '%s' has an artifact with an empty name",
					                             task.task_id);
				}
				if (!artifact_names.insert(artifact.name).second) {
					throw SerializationException("extension scan task '%s' has duplicate artifact '%s'", task.task_id,
					                             artifact.name);
				}
				if (artifact.codec.empty() || artifact.codec_version == 0) {
					throw SerializationException("extension scan task '%s' artifact '%s' has an invalid codec identity",
					                             task.task_id, artifact.name);
				}
			}
		}
		idx_t expected_cardinality;
		idx_t expected_bytes;
		ComputeExtensionTaskEstimates(extension_tasks, expected_cardinality, expected_bytes);
		if (estimated_cardinality != expected_cardinality || estimated_bytes != expected_bytes) {
			throw SerializationException("extension scan task descriptor estimates do not match its opaque tasks");
		}
		break;
	}
	default:
		throw SerializationException("unknown scan task descriptor kind: %u", static_cast<unsigned int>(kind));
	}
}

void ScanTaskDescriptor::Merge(ScanTaskDescriptor other) {
	Validate();
	other.Validate();
	if (kind != other.kind) {
		throw InvalidInputException("cannot merge file and extension scan task descriptors");
	}
	if (kind == ScanTaskKind::EXTENSION &&
	    (extension_capability != other.extension_capability || task_codec != other.task_codec ||
	     task_codec_version != other.task_codec_version)) {
		throw InvalidInputException("cannot merge extension scan task descriptors with different protocol identities");
	}
	if (kind == ScanTaskKind::EXTENSION) {
		set<string> task_ids;
		for (const auto &task : extension_tasks) {
			task_ids.insert(task.task_id);
		}
		for (const auto &task : other.extension_tasks) {
			if (!task_ids.insert(task.task_id).second) {
				throw InvalidInputException("cannot merge extension scan task descriptors with duplicate task_id '%s'",
				                            task.task_id);
			}
		}
	}
	if (kind == ScanTaskKind::FILES) {
		estimated_cardinality = SaturatingAddScanTaskEstimate(estimated_cardinality, other.estimated_cardinality);
		estimated_bytes = SaturatingAddScanTaskEstimate(estimated_bytes, other.estimated_bytes);
		files.insert(files.end(), std::make_move_iterator(other.files.begin()),
		             std::make_move_iterator(other.files.end()));
	} else {
		extension_tasks.insert(extension_tasks.end(), std::make_move_iterator(other.extension_tasks.begin()),
		                       std::make_move_iterator(other.extension_tasks.end()));
		ComputeExtensionTaskEstimates(extension_tasks, estimated_cardinality, estimated_bytes);
	}
	source_task_partition_id = DConstants::INVALID_INDEX;
	Validate();
}

void ScanTaskDescriptor::Serialize(Serializer &serializer) const {
	Validate();
	serializer.WriteProperty(1, "kind", static_cast<uint8_t>(kind));
	if (kind == ScanTaskKind::FILES) {
		serializer.WriteList(2, "files", files.size(), [&](Serializer::List &list, idx_t i) {
			list.WriteObject([&](Serializer &obj) {
				obj.WriteProperty(1, "path", files[i].path);
				unordered_map<string, Value> options;
				if (files[i].extended_info) {
					options = files[i].extended_info->options;
				}
				obj.WriteProperty(2, "options", options);
			});
		});
	} else {
		serializer.WriteObject(10, "extension_capability", [&](Serializer &obj) {
			obj.WriteProperty(1, "extension_name", extension_capability.extension_name);
			obj.WriteProperty(2, "extension_protocol_version", extension_capability.extension_protocol_version);
			obj.WriteProperty(3, "capability_kind", static_cast<uint8_t>(extension_capability.capability.kind));
			obj.WriteProperty(4, "capability_name", extension_capability.capability.name);
			obj.WriteProperty(5, "capability_protocol_version", extension_capability.capability.protocol_version);
		});
		serializer.WriteProperty(11, "task_codec", task_codec);
		serializer.WriteProperty(12, "task_codec_version", task_codec_version);
		serializer.WriteList(13, "extension_tasks", extension_tasks.size(), [&](Serializer::List &list, idx_t i) {
			const auto &task = extension_tasks[i];
			list.WriteObject([&](Serializer &obj) {
				obj.WriteProperty(1, "task_id", task.task_id);
				obj.WriteProperty(2, "payload", task.payload);
				obj.WriteProperty(3, "estimated_cardinality", task.estimated_cardinality);
				obj.WriteProperty(4, "estimated_bytes", task.estimated_bytes);
				obj.WriteList(5, "artifacts", task.artifacts.size(), [&](Serializer::List &artifacts, idx_t j) {
					const auto &artifact = task.artifacts[j];
					artifacts.WriteObject([&](Serializer &artifact_obj) {
						artifact_obj.WriteProperty(1, "name", artifact.name);
						artifact_obj.WriteProperty(2, "codec", artifact.codec);
						artifact_obj.WriteProperty(3, "codec_version", artifact.codec_version);
						artifact_obj.WriteProperty(4, "data", artifact.data);
					});
				});
			});
		});
	}
	serializer.WriteProperty(20, "estimated_cardinality", estimated_cardinality);
	serializer.WriteProperty(21, "estimated_bytes", estimated_bytes);
	serializer.WriteProperty(22, "source_task_partition_id", source_task_partition_id);
}

ScanTaskDescriptor ScanTaskDescriptor::Deserialize(Deserializer &deserializer) {
	ScanTaskDescriptor desc;
	desc.kind = static_cast<ScanTaskKind>(deserializer.ReadProperty<uint8_t>(1, "kind"));
	if (desc.kind == ScanTaskKind::FILES) {
		deserializer.ReadList(2, "files", [&](Deserializer::List &list, idx_t) {
			list.ReadObject([&](Deserializer &obj) {
				OpenFileInfo info;
				info.path = obj.ReadProperty<string>(1, "path");
				auto options = obj.ReadProperty<unordered_map<string, Value>>(2, "options");
				if (!options.empty()) {
					auto ext = make_shared_ptr<ExtendedOpenFileInfo>();
					ext->options = std::move(options);
					info.extended_info = std::move(ext);
				}
				desc.files.push_back(std::move(info));
			});
		});
	} else if (desc.kind == ScanTaskKind::EXTENSION) {
		deserializer.ReadObject(10, "extension_capability", [&](Deserializer &obj) {
			desc.extension_capability.extension_name = obj.ReadProperty<string>(1, "extension_name");
			desc.extension_capability.extension_protocol_version =
			    obj.ReadProperty<idx_t>(2, "extension_protocol_version");
			desc.extension_capability.capability.kind =
			    static_cast<DistributedExtensionCapabilityKind>(obj.ReadProperty<uint8_t>(3, "capability_kind"));
			desc.extension_capability.capability.name = obj.ReadProperty<string>(4, "capability_name");
			desc.extension_capability.capability.protocol_version =
			    obj.ReadProperty<idx_t>(5, "capability_protocol_version");
		});
		desc.task_codec = deserializer.ReadProperty<string>(11, "task_codec");
		desc.task_codec_version = deserializer.ReadProperty<idx_t>(12, "task_codec_version");
		deserializer.ReadList(13, "extension_tasks", [&](Deserializer::List &list, idx_t) {
			DistributedScanTask task;
			list.ReadObject([&](Deserializer &obj) {
				task.task_id = obj.ReadProperty<string>(1, "task_id");
				task.payload = obj.ReadProperty<string>(2, "payload");
				task.estimated_cardinality = obj.ReadProperty<optional_idx>(3, "estimated_cardinality");
				task.estimated_bytes = obj.ReadProperty<optional_idx>(4, "estimated_bytes");
				obj.ReadList(5, "artifacts", [&](Deserializer::List &artifacts, idx_t) {
					DistributedScanTaskArtifact artifact;
					artifacts.ReadObject([&](Deserializer &artifact_obj) {
						artifact.name = artifact_obj.ReadProperty<string>(1, "name");
						artifact.codec = artifact_obj.ReadProperty<string>(2, "codec");
						artifact.codec_version = artifact_obj.ReadProperty<idx_t>(3, "codec_version");
						artifact.data = artifact_obj.ReadProperty<string>(4, "data");
					});
					task.artifacts.push_back(std::move(artifact));
				});
			});
			desc.extension_tasks.push_back(std::move(task));
		});
	} else {
		throw SerializationException("unknown scan task descriptor kind: %u", static_cast<unsigned int>(desc.kind));
	}
	desc.estimated_cardinality = deserializer.ReadProperty<idx_t>(20, "estimated_cardinality");
	desc.estimated_bytes = deserializer.ReadProperty<idx_t>(21, "estimated_bytes");
	desc.source_task_partition_id = deserializer.ReadProperty<idx_t>(22, "source_task_partition_id");
	desc.Validate();
	return desc;
}

std::string ScanTaskDescriptor::SerializeToBytes() const {
	MemoryStream stream(Allocator::DefaultAllocator());
	BinarySerializer serializer(stream);
	serializer.Begin();
	Serialize(serializer);
	serializer.End();
	return std::string(reinterpret_cast<const char *>(stream.GetData()), stream.GetPosition());
}

std::string ScanTaskDescriptor::SerializeToBase64() const {
	auto bytes = SerializeToBytes();
	if (bytes.empty()) {
		return std::string();
	}
	return Blob::ToBase64(string_t(bytes.data(), bytes.size()));
}

ScanTaskDescriptor ScanTaskDescriptor::DeserializeFromBytes(const std::string &bytes) {
	if (bytes.empty()) {
		throw SerializationException("cannot deserialize an empty scan task descriptor");
	}
	auto *data_ptr = reinterpret_cast<data_ptr_t>(const_cast<char *>(bytes.data()));
	MemoryStream stream(data_ptr, bytes.size());
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto desc = Deserialize(deserializer);
	deserializer.End();
	return desc;
}

ScanTaskDescriptor ScanTaskDescriptor::DeserializeFromBase64(const std::string &base64) {
	if (base64.empty()) {
		throw SerializationException("cannot deserialize an empty base64 scan task descriptor");
	}
	auto raw = Blob::FromBase64(string_t(base64.data(), base64.size()));
	return DeserializeFromBytes(raw);
}

bool ApplyScanTasksToPlan(duckdb::PhysicalPlan &plan, const std::unordered_map<idx_t, ScanTaskDescriptor> &tasks,
                          std::string *error) {
	if (!plan.HasRoot()) {
		if (error) {
			*error = "plan has no root";
		}
		return false;
	}
	if (tasks.empty()) {
		if (error) {
			*error = "scan task map is empty";
		}
		return false;
	}
	for (const auto &entry : tasks) {
		entry.second.Validate();
	}
	ScanTaskReferenceMap task_references;
	task_references.reserve(tasks.size());
	for (const auto &entry : tasks) {
		task_references.emplace(entry.first, &entry.second);
	}
	ApplyScanTasksStats stats;
	idx_t max_id = MaxScanNodeId(plan.Root(), 0);
	for (const auto &kv : tasks) {
		if (kv.first > max_id) {
			max_id = kv.first;
		}
	}
	idx_t next_id = max_id + 1;
	std::unordered_map<idx_t, idx_t> base_for_group;
	std::unordered_map<idx_t, idx_t> dup_to_base;
	NormalizeScanNodeIdsByGroup(plan.Root(), base_for_group, dup_to_base, next_id, stats);
	for (const auto &kv : dup_to_base) {
		if (task_references.find(kv.first) != task_references.end()) {
			continue;
		}
		auto base_it = task_references.find(kv.second);
		if (base_it != task_references.end()) {
			task_references.emplace(kv.first, base_it->second);
			stats.copied_tasks++;
		}
	}
	set<idx_t> plan_scan_node_ids;
	CollectScanNodeIds(plan.Root(), plan_scan_node_ids);
	for (const auto &entry : tasks) {
		if (plan_scan_node_ids.find(entry.first) == plan_scan_node_ids.end()) {
			if (error) {
				*error = "scan task node_id=" + std::to_string(entry.first) + " is not present in the worker plan";
			}
			return false;
		}
	}
	set<idx_t> matched_tasks;
	ApplyScanTasksToOperator(plan.Root(), task_references, matched_tasks, stats, error);
	if (stats.invalid_assignment != 0) {
		if (error && error->empty()) {
			*error = "one or more scan tasks had invalid assignments";
		}
		return false;
	}
	for (const auto &entry : tasks) {
		if (matched_tasks.find(entry.first) == matched_tasks.end()) {
			if (error && error->empty()) {
				*error = "scan task node_id=" + std::to_string(entry.first) + " is not present in the worker plan";
			}
			return false;
		}
	}
	if (stats.applied == 0) {
		if (error && error->empty()) {
			*error = "no scan tasks applied";
		}
		return false;
	}
	return true;
}

namespace {

static bool ApplyFteExtensionScanTasks(PhysicalTableScan &scan, const std::shared_ptr<FteSplitQueue> &queue,
                                       string *error) {
	ScanTaskDescriptor merged;
	bool has_descriptor = false;
	while (true) {
		auto next = queue->WaitForNext();
		if (next.state == FteSplitQueue::GetResult::CANCELED) {
			SetApplyError(error, "FTE extension scan source queue was canceled");
			return false;
		}
		if (next.state == FteSplitQueue::GetResult::FINISHED) {
			break;
		}
		if (next.state != FteSplitQueue::GetResult::SPLIT) {
			continue;
		}
		if (next.input.kind != TaskInput::Kind::ScanTask) {
			SetApplyError(error, "FTE extension scan source queue received a non-scan split");
			return false;
		}
		auto descriptor = ScanTaskDescriptor::DeserializeFromBytes(next.input.scan_task_bytes);
		if (descriptor.kind != ScanTaskKind::EXTENSION) {
			SetApplyError(error, "FTE extension scan source queue received a file task descriptor");
			return false;
		}
		if (!has_descriptor) {
			merged = std::move(descriptor);
			has_descriptor = true;
		} else {
			merged.Merge(std::move(descriptor));
		}
	}
	if (has_descriptor) {
		return ApplyExtensionScanTasks(scan, merged, error);
	}
	SetApplyError(error, "FTE extension scan source queue finished without an explicit task descriptor");
	return false;
}

bool ApplyFteScanSourceQueuesToOperator(PhysicalOperator &op,
                                        const std::unordered_map<idx_t, std::shared_ptr<FteSplitQueue>> &queues,
                                        set<idx_t> &matched_queues, std::string *error, idx_t &applied) {
	bool ok = true;
	if (op.type == PhysicalOperatorType::TABLE_SCAN) {
		auto &scan = op.Cast<PhysicalTableScan>();
		if (scan.extra_info.scan_node_id.IsValid()) {
			const auto node_id = scan.extra_info.scan_node_id.GetIndex();
			auto entry = queues.find(node_id);
			if (entry != queues.end()) {
				matched_queues.insert(node_id);
				if (!entry->second) {
					if (error) {
						*error = "null FTE scan source split queue for scan_node_id=" + std::to_string(node_id);
					}
					return false;
				}
				if (!scan.bind_data) {
					if (error) {
						*error = "FTE scan source queue target has null bind_data for scan_node_id=" +
						         std::to_string(node_id);
					}
					return false;
				}
				if (scan.function.HasDistributedScanCallbacks()) {
					if (!ApplyFteExtensionScanTasks(scan, entry->second, error)) {
						return false;
					}
					applied++;
				} else if (auto *multi_bind = dynamic_cast<MultiFileBindData *>(scan.bind_data.get())) {
					multi_bind->file_list = make_shared_ptr<FteDynamicScanFileList>(entry->second);
					scan.extra_info.total_files = optional_idx();
					scan.extra_info.filtered_files = optional_idx();
					scan.distributed_scan_tasks_applied = true;
					applied++;
				} else {
					if (error) {
						*error =
						    "FTE dynamic scan source requires MultiFileBindData or explicit distributed table-function "
						    "callbacks for scan_node_id=" +
						    std::to_string(node_id);
					}
					return false;
				}
			}
		}
	}
	for (auto &child : op.children) {
		if (!ApplyFteScanSourceQueuesToOperator(child.get(), queues, matched_queues, error, applied)) {
			ok = false;
		}
	}
	return ok;
}

} // namespace

bool ApplyFteScanSourceQueuesToPlan(duckdb::PhysicalPlan &plan,
                                    const std::unordered_map<idx_t, std::shared_ptr<FteSplitQueue>> &queues,
                                    std::string *error) {
	if (!plan.HasRoot()) {
		if (error) {
			*error = "plan has no root";
		}
		return false;
	}
	if (queues.empty()) {
		if (error) {
			*error = "FTE scan source queue map is empty";
		}
		return false;
	}
	set<idx_t> plan_scan_node_ids;
	CollectScanNodeIds(plan.Root(), plan_scan_node_ids);
	for (const auto &entry : queues) {
		if (!entry.second) {
			if (error) {
				*error = "null FTE scan source split queue for scan_node_id=" + std::to_string(entry.first);
			}
			return false;
		}
		if (plan_scan_node_ids.find(entry.first) == plan_scan_node_ids.end()) {
			if (error) {
				*error = "FTE scan source queue node_id=" + std::to_string(entry.first) +
				         " is not present in the worker plan";
			}
			return false;
		}
	}
	idx_t applied = 0;
	set<idx_t> matched_queues;
	if (!ApplyFteScanSourceQueuesToOperator(plan.Root(), queues, matched_queues, error, applied)) {
		return false;
	}
	for (const auto &entry : queues) {
		if (matched_queues.find(entry.first) == matched_queues.end()) {
			if (error) {
				*error = "FTE scan source queue node_id=" + std::to_string(entry.first) +
				         " is not present in the worker plan";
			}
			return false;
		}
	}
	if (applied == 0) {
		if (error) {
			*error = "no FTE scan source queues applied";
		}
		return false;
	}
	return true;
}

bool ValidateScanTaskAssignments(const duckdb::PhysicalPlan &plan, const set<idx_t> &assigned_node_ids,
                                 std::string *error) {
	if (!plan.HasRoot()) {
		SetApplyError(error, "plan has no root");
		return false;
	}
	set<idx_t> scan_node_ids;
	if (!CollectRequiredScanNodeIds(plan.Root(), scan_node_ids, error)) {
		return false;
	}
	for (auto node_id : scan_node_ids) {
		if (assigned_node_ids.find(node_id) == assigned_node_ids.end()) {
			SetApplyError(error, "distributed table scan has no explicit worker task assignment for scan_node_id=" +
			                         std::to_string(node_id));
			return false;
		}
	}
	for (auto node_id : assigned_node_ids) {
		if (scan_node_ids.find(node_id) == scan_node_ids.end()) {
			SetApplyError(error, "scan task assignment node_id=" + std::to_string(node_id) +
			                         " is not present in the worker plan");
			return false;
		}
	}
	return true;
}

namespace {

static bool ValidateDistributedScanTasksAppliedToOperator(const PhysicalOperator &op, string *error) {
	if (op.type == PhysicalOperatorType::TABLE_SCAN) {
		auto &scan = op.Cast<PhysicalTableScan>();
		if (!scan.extra_info.scan_node_id.IsValid()) {
			SetApplyError(error,
			              "distributed table scan '" + scan.function.name + "' has no runtime scan node identity");
			return false;
		}
		if (!scan.distributed_scan_tasks_applied) {
			SetApplyError(error,
			              "distributed table scan '" + scan.function.name + "' has no explicit worker task assignment");
			return false;
		}
	}
	for (const auto &child : op.children) {
		if (!ValidateDistributedScanTasksAppliedToOperator(child.get(), error)) {
			return false;
		}
	}
	return true;
}

} // namespace

bool ValidateDistributedScanTasksApplied(const duckdb::PhysicalPlan &plan, std::string *error) {
	if (!plan.HasRoot()) {
		SetApplyError(error, "plan has no root");
		return false;
	}
	return ValidateDistributedScanTasksAppliedToOperator(plan.Root(), error);
}

bool HasDistributedScanTaskTargets(const duckdb::PhysicalPlan &plan) {
	if (!plan.HasRoot()) {
		return false;
	}
	set<idx_t> node_ids;
	CollectScanNodeIds(plan.Root(), node_ids);
	return !node_ids.empty();
}

} // namespace distributed
} // namespace duckdb
