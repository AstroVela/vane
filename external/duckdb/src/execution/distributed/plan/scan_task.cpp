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
#include "duckdb/common/limits.hpp"
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
		bool received_descriptor = false;
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
		if (next.state == FteSplitQueue::GetResult::CANCELED) {
			state->finished = true;
			throw InvalidInputException("FTE MultiFile scan source queue was canceled");
		}
		if (next.state == FteSplitQueue::GetResult::FINISHED) {
			state->finished = true;
			if (!state->received_descriptor) {
				throw InvalidInputException(
				    "FTE MultiFile scan source queue finished without an explicit task descriptor");
			}
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
		state->received_descriptor = true;
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

static idx_t AllocateScanNodeId(idx_t &last_id) {
	if (last_id == NumericLimits<idx_t>::Maximum()) {
		throw InvalidInputException("cannot allocate a unique distributed scan node identity");
	}
	return ++last_id;
}

static void NormalizeScanNodeIdsByGroup(PhysicalOperator &op, unordered_map<idx_t, idx_t> &base_node_for_group,
                                        unordered_map<idx_t, idx_t> &group_for_node,
                                        unordered_map<idx_t, idx_t> &alias_to_parent, idx_t &last_id,
                                        ApplyScanTasksStats &stats) {
	if (op.type == PhysicalOperatorType::TABLE_SCAN) {
		auto &scan = op.Cast<PhysicalTableScan>();
		if (!scan.extra_info.scan_group_id.IsValid()) {
			stats.missing_group_id++;
			if (scan.extra_info.scan_node_id.IsValid()) {
				scan.extra_info.scan_group_id = scan.extra_info.scan_node_id;
			} else {
				scan.extra_info.scan_group_id = optional_idx(AllocateScanNodeId(last_id));
			}
		}
		if (!scan.extra_info.scan_node_id.IsValid()) {
			scan.extra_info.scan_node_id = optional_idx(AllocateScanNodeId(last_id));
		}

		const idx_t group_id = scan.extra_info.scan_group_id.GetIndex();
		idx_t node_id = scan.extra_info.scan_node_id.GetIndex();
		const idx_t original_node_id = node_id;
		auto existing_node = group_for_node.find(node_id);
		if (existing_node != group_for_node.end()) {
			if (existing_node->second != group_id) {
				throw InvalidInputException(
				    "distributed scan node identity %llu is reused by scan groups %llu and %llu",
				    static_cast<unsigned long long>(node_id), static_cast<unsigned long long>(existing_node->second),
				    static_cast<unsigned long long>(group_id));
			}
			node_id = AllocateScanNodeId(last_id);
			scan.extra_info.scan_node_id = optional_idx(node_id);
			group_for_node.emplace(node_id, group_id);
			alias_to_parent.emplace(node_id, original_node_id);
			stats.duplicate_node_id++;
		} else {
			group_for_node.emplace(node_id, group_id);
			auto group_entry = base_node_for_group.find(group_id);
			if (group_entry == base_node_for_group.end()) {
				base_node_for_group.emplace(group_id, node_id);
			} else if (node_id != group_entry->second) {
				alias_to_parent.emplace(node_id, group_entry->second);
			}
		}
	}
	for (auto &child : op.children) {
		NormalizeScanNodeIdsByGroup(child.get(), base_node_for_group, group_for_node, alias_to_parent, last_id, stats);
	}
}

template <class MAP>
static typename MAP::const_iterator
ResolveScanNodeAlias(const MAP &assignments, const unordered_map<idx_t, idx_t> &alias_to_parent, idx_t node_id) {
	set<idx_t> visited;
	while (true) {
		auto assignment = assignments.find(node_id);
		if (assignment != assignments.end()) {
			return assignment;
		}
		if (!visited.insert(node_id).second) {
			throw InternalException("distributed scan node alias cycle detected at node %llu",
			                        static_cast<unsigned long long>(node_id));
		}
		auto alias = alias_to_parent.find(node_id);
		if (alias == alias_to_parent.end()) {
			return assignments.end();
		}
		node_id = alias->second;
	}
}
static bool ApplyExtensionScanTasks(PhysicalTableScan &scan, const ScanTaskDescriptor &descriptor, string *error) {
	if (!scan.function.HasDistributedScanCallbacks()) {
		SetApplyError(error, "extension scan task assigned to table function without distributed callbacks: " +
		                         scan.function.name);
		return false;
	}
	const auto &callbacks = scan.function.GetDistributedScanCallbacks();
	callbacks.Validate(scan.function);
	const auto &capability = callbacks.GetCapability();
	if (descriptor.extension_capability != capability) {
		SetApplyError(error, "distributed scan capability mismatch for table function '" + scan.function.name +
		                         "': task=" + descriptor.extension_capability.CanonicalIdentity() +
		                         ", worker=" + capability.CanonicalIdentity());
		return false;
	}
	if (descriptor.task_codec != callbacks.task_codec) {
		SetApplyError(error, "distributed scan task codec mismatch for table function '" + scan.function.name +
		                         "': task=" + descriptor.task_codec.CanonicalIdentity() +
		                         ", worker=" + callbacks.task_codec.CanonicalIdentity());
		return false;
	}
	callbacks.apply_tasks(*scan.bind_data, descriptor.extension_tasks);
	scan.extra_info.total_files = optional_idx(descriptor.task_count());
	scan.extra_info.filtered_files = optional_idx(descriptor.task_count());
	scan.distributed_scan_tasks_applied = true;
	return true;
}

using ScanTaskReferenceMap = unordered_map<idx_t, const ScanTaskDescriptor *>;

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
	const auto maximum = NumericLimits<idx_t>::Maximum();
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
		if (!extension_tasks.empty() || !extension_capability.extension_name.empty() || !task_codec.name.empty() ||
		    task_codec.version != 0) {
			throw SerializationException("file scan task descriptor contains extension task state");
		}
		break;
	case ScanTaskKind::EXTENSION: {
		if (!files.empty()) {
			throw SerializationException("extension scan task descriptor contains OpenFileInfo entries");
		}
		task_codec.Validate("Extension scan task descriptor");
		extension_capability.Validate();
		if (extension_capability.capability.kind != DistributedExtensionCapabilityKind::TABLE_FUNCTION) {
			throw SerializationException("extension scan task descriptor capability is not a table function");
		}

		set<string> task_ids;
		for (const auto &task : extension_tasks) {
			task.Validate();
			if (!task_ids.insert(task.task_id).second) {
				throw SerializationException("extension scan task_id '%s' appears more than once", task.task_id);
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
	    (extension_capability != other.extension_capability || task_codec != other.task_codec)) {
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
		serializer.WriteObject(10, "extension_capability",
		                       [&](Serializer &object) { extension_capability.Serialize(object); });
		serializer.WriteObject(11, "task_codec", [&](Serializer &object) { task_codec.Serialize(object); });
		serializer.WriteList(12, "extension_tasks", extension_tasks.size(), [&](Serializer::List &list, idx_t i) {
			list.WriteObject([&](Serializer &object) { extension_tasks[i].Serialize(object); });
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
		deserializer.ReadObject(10, "extension_capability", [&](Deserializer &object) {
			desc.extension_capability = DistributedExtensionCapabilityReference::Deserialize(object);
		});
		deserializer.ReadObject(11, "task_codec", [&](Deserializer &object) {
			desc.task_codec = DistributedPayloadCodec::Deserialize(object);
		});
		deserializer.ReadList(12, "extension_tasks", [&](Deserializer::List &list, idx_t) {
			list.ReadObject([&](Deserializer &object) {
				desc.extension_tasks.push_back(DistributedScanTask::Deserialize(object));
			});
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
	idx_t last_id = max_id;
	unordered_map<idx_t, idx_t> base_node_for_group;
	unordered_map<idx_t, idx_t> group_for_node;
	unordered_map<idx_t, idx_t> alias_to_parent;
	NormalizeScanNodeIdsByGroup(plan.Root(), base_node_for_group, group_for_node, alias_to_parent, last_id, stats);
	for (const auto &kv : alias_to_parent) {
		if (task_references.find(kv.first) != task_references.end()) {
			continue;
		}
		auto assignment = ResolveScanNodeAlias(task_references, alias_to_parent, kv.first);
		if (assignment != task_references.end()) {
			task_references.emplace(kv.first, assignment->second);
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

using FteExtensionDescriptorCache = unordered_map<const FteSplitQueue *, ScanTaskDescriptor>;
using FteDynamicFileListCache = unordered_map<const FteSplitQueue *, shared_ptr<MultiFileList>>;

static bool ApplyFteExtensionScanTasks(PhysicalTableScan &scan, const std::shared_ptr<FteSplitQueue> &queue,
                                       FteExtensionDescriptorCache &descriptor_cache, string *error) {
	auto cached_descriptor = descriptor_cache.find(queue.get());
	if (cached_descriptor != descriptor_cache.end()) {
		return ApplyExtensionScanTasks(scan, cached_descriptor->second, error);
	}

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
	if (!has_descriptor) {
		SetApplyError(error, "FTE extension scan source queue finished without an explicit task descriptor");
		return false;
	}
	auto inserted = descriptor_cache.emplace(queue.get(), std::move(merged));
	return ApplyExtensionScanTasks(scan, inserted.first->second, error);
}

bool ApplyFteScanSourceQueuesToOperator(PhysicalOperator &op,
                                        const std::unordered_map<idx_t, std::shared_ptr<FteSplitQueue>> &queues,
                                        FteExtensionDescriptorCache &extension_descriptor_cache,
                                        FteDynamicFileListCache &dynamic_file_list_cache, set<idx_t> &matched_queues,
                                        string *error, idx_t &applied) {
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
					if (!ApplyFteExtensionScanTasks(scan, entry->second, extension_descriptor_cache, error)) {
						return false;
					}
					applied++;
				} else if (auto *multi_bind = dynamic_cast<MultiFileBindData *>(scan.bind_data.get())) {
					auto cached_file_list = dynamic_file_list_cache.find(entry->second.get());
					if (cached_file_list == dynamic_file_list_cache.end()) {
						auto file_list = make_shared_ptr<FteDynamicScanFileList>(entry->second);
						cached_file_list =
						    dynamic_file_list_cache.emplace(entry->second.get(), std::move(file_list)).first;
					}
					multi_bind->file_list = cached_file_list->second;
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
		if (!ApplyFteScanSourceQueuesToOperator(child.get(), queues, extension_descriptor_cache,
		                                        dynamic_file_list_cache, matched_queues, error, applied)) {
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
	ApplyScanTasksStats stats;
	idx_t max_id = MaxScanNodeId(plan.Root(), 0);
	for (const auto &entry : queues) {
		if (entry.first > max_id) {
			max_id = entry.first;
		}
	}
	idx_t last_id = max_id;
	unordered_map<idx_t, idx_t> base_node_for_group;
	unordered_map<idx_t, idx_t> group_for_node;
	unordered_map<idx_t, idx_t> alias_to_parent;
	NormalizeScanNodeIdsByGroup(plan.Root(), base_node_for_group, group_for_node, alias_to_parent, last_id, stats);
	auto queue_references = queues;
	for (const auto &entry : alias_to_parent) {
		if (queue_references.find(entry.first) != queue_references.end()) {
			continue;
		}
		auto assignment = ResolveScanNodeAlias(queue_references, alias_to_parent, entry.first);
		if (assignment != queue_references.end()) {
			queue_references.emplace(entry.first, assignment->second);
		}
	}

	idx_t applied = 0;
	set<idx_t> matched_queues;
	FteExtensionDescriptorCache extension_descriptor_cache;
	FteDynamicFileListCache dynamic_file_list_cache;
	if (!ApplyFteScanSourceQueuesToOperator(plan.Root(), queue_references, extension_descriptor_cache,
	                                        dynamic_file_list_cache, matched_queues, error, applied)) {
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

bool ValidateScanTaskAssignments(const duckdb::PhysicalPlan &plan, const set<idx_t> &assigned_node_ids, string *error) {
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

bool ValidateDistributedScanTasksApplied(const duckdb::PhysicalPlan &plan, string *error) {
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
