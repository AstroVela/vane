// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/distributed/plan/scan_split.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/execution/distributed/plan/scan_split.hpp"

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

#include <iterator>
#include <mutex>

namespace duckdb {
namespace distributed {

namespace {

struct ApplyScanSplitsStats {
	idx_t table_scans = 0;
	idx_t applied = 0;
	idx_t missing_node_id = 0;
	idx_t missing_batch = 0;
	idx_t missing_bind = 0;
	idx_t non_multi_bind = 0;
	idx_t invalid_assignment = 0;
	idx_t duplicate_node_id = 0;
	idx_t copied_batches = 0;
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
		set<string> split_ids;
		bool received_batch = false;
		bool received_empty_split = false;
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
			if (!state->received_batch) {
				throw InvalidInputException("FTE MultiFile scan source queue finished without an explicit split batch");
			}
			return false;
		}
		if (next.state != FteSplitQueue::GetResult::SPLIT) {
			return false;
		}
		if (next.input.kind != TaskInput::Kind::ScanSplitBatch) {
			throw InvalidInputException("dynamic scan source queue received non-scan split");
		}
		auto batch = ScanSplitBatch::DeserializeFromBytes(next.input.scan_split_batch_bytes);
		if (batch.IsExtension()) {
			throw InvalidInputException("dynamic MultiFile scan source received an extension scan split");
		}
		if (state->received_batch && (state->received_empty_split || batch.splits[0].empty)) {
			throw InvalidInputException("an explicit empty scan split cannot be combined with another split batch");
		}
		state->received_batch = true;
		for (auto &split : batch.splits) {
			if (!state->split_ids.insert(split.split_id).second) {
				throw InvalidInputException("dynamic MultiFile scan source received duplicate split_id '%s'",
				                            split.split_id);
			}
			state->received_empty_split = split.empty;
			if (!split.empty) {
				state->files.push_back(std::move(split.file));
			}
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
                                        ApplyScanSplitsStats &stats) {
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
static bool ApplyExtensionScanSplits(PhysicalTableScan &scan, const ScanSplitBatch &batch, string *error) {
	if (!scan.function.HasDistributedScanCallbacks()) {
		SetApplyError(error, "extension scan split assigned to table function without distributed callbacks: " +
		                         scan.function.name);
		return false;
	}
	batch.Validate();
	if (!batch.IsExtension()) {
		SetApplyError(error, "file scan split batch assigned to extension table function: " + scan.function.name);
		return false;
	}
	const auto &callbacks = scan.function.GetDistributedScanCallbacks();
	callbacks.Validate(scan.function);
	if (!scan.bind_data && callbacks.bind_data_mode == TableFunctionDistributedBindDataMode::REQUIRED) {
		SetApplyError(error, "distributed table function requires worker bind data: " + scan.function.name);
		return false;
	}
	const auto &capability = callbacks.GetCapability();
	const auto &first_split = batch.splits[0];
	if (first_split.extension_capability != capability) {
		SetApplyError(error, "distributed scan capability mismatch for table function '" + scan.function.name +
		                         "': split=" + first_split.extension_capability.CanonicalIdentity() +
		                         ", worker=" + capability.CanonicalIdentity());
		return false;
	}
	if (first_split.split_codec != callbacks.split_codec) {
		SetApplyError(error, "distributed scan split codec mismatch for table function '" + scan.function.name +
		                         "': split=" + first_split.split_codec.CanonicalIdentity() +
		                         ", worker=" + callbacks.split_codec.CanonicalIdentity());
		return false;
	}
	vector<DistributedScanSplit> assigned_splits;
	assigned_splits.reserve(batch.splits.size());
	for (const auto &split : batch.splits) {
		if (split.empty) {
			continue;
		}
		DistributedScanSplit assigned;
		assigned.split_id = split.split_id;
		assigned.payload = split.extension_payload;
		assigned.estimated_cardinality = split.estimated_cardinality;
		assigned.estimated_bytes = split.estimated_bytes;
		assigned_splits.push_back(std::move(assigned));
	}
	callbacks.apply_splits(scan.bind_data.get(), assigned_splits);
	scan.extra_info.total_files = optional_idx(assigned_splits.size());
	scan.extra_info.filtered_files = optional_idx(assigned_splits.size());
	scan.distributed_scan_splits_applied = true;
	return true;
}

using ScanSplitBatchReferenceMap = unordered_map<idx_t, const ScanSplitBatch *>;

static bool ApplyScanSplitBatchesToOperator(PhysicalOperator &op, const ScanSplitBatchReferenceMap &batches,
                                            set<idx_t> &matched_batches, ApplyScanSplitsStats &stats, string *error) {
	bool applied_any = false;
	if (op.type == PhysicalOperatorType::TABLE_SCAN) {
		stats.table_scans++;
		auto &scan = op.Cast<PhysicalTableScan>();
		if (!scan.extra_info.scan_node_id.IsValid()) {
			stats.missing_node_id++;
		} else {
			const idx_t node_id = scan.extra_info.scan_node_id.GetIndex();
			auto it = batches.find(node_id);
			if (it == batches.end()) {
				stats.missing_batch++;
			} else {
				matched_batches.insert(node_id);
				const auto &batch = *it->second;
				if (batch.IsExtension()) {
					if (ApplyExtensionScanSplits(scan, batch, error)) {
						stats.applied++;
						applied_any = true;
					} else {
						stats.non_multi_bind++;
						stats.invalid_assignment++;
					}
				} else if (!scan.bind_data) {
					stats.missing_bind++;
					stats.invalid_assignment++;
					SetApplyError(error, "scan split batch assigned to table function with null bind data: " +
					                         scan.function.name);
				} else if (scan.function.HasDistributedScanCallbacks()) {
					SetApplyError(error,
					              "file scan split assigned to extension table function '" + scan.function.name + "'");
					stats.non_multi_bind++;
					stats.invalid_assignment++;
				} else if (auto *multi_bind = dynamic_cast<MultiFileBindData *>(scan.bind_data.get())) {
					vector<OpenFileInfo> files;
					files.reserve(batch.splits.size());
					for (const auto &split : batch.splits) {
						if (!split.empty) {
							files.push_back(split.file);
						}
					}
					const idx_t file_count = files.size();
					multi_bind->file_list = duckdb::make_shared_ptr<SimpleMultiFileList>(std::move(files));
					scan.extra_info.total_files = optional_idx(file_count);
					scan.extra_info.filtered_files = optional_idx(file_count);
					scan.distributed_scan_splits_applied = true;
					stats.applied++;
					applied_any = true;
				} else {
					stats.non_multi_bind++;
					stats.invalid_assignment++;
					SetApplyError(error,
					              "file scan split assigned to non-MultiFile table function: " + scan.function.name);
				}
			}
		}
	}

	for (auto &child : op.children) {
		if (ApplyScanSplitBatchesToOperator(child.get(), batches, matched_batches, stats, error)) {
			applied_any = true;
		}
	}
	return applied_any;
}

} // namespace

static idx_t SaturatingAddScanSplitEstimate(idx_t left, idx_t right) {
	const auto maximum = NumericLimits<idx_t>::Maximum();
	return right > maximum - left ? maximum : left + right;
}

ScanSplit ScanSplit::File(string split_id, OpenFileInfo file, optional_idx estimated_cardinality,
                          optional_idx estimated_bytes) {
	ScanSplit result;
	result.kind = ScanSplitKind::FILE;
	result.split_id = std::move(split_id);
	result.file = std::move(file);
	result.estimated_cardinality = estimated_cardinality;
	result.estimated_bytes = estimated_bytes;
	result.Validate();
	return result;
}

ScanSplit ScanSplit::Extension(string split_id, string payload, DistributedExtensionCapabilityReference capability,
                               DistributedPayloadCodec split_codec, optional_idx estimated_cardinality,
                               optional_idx estimated_bytes) {
	ScanSplit result;
	result.kind = ScanSplitKind::EXTENSION;
	result.split_id = std::move(split_id);
	result.extension_capability = std::move(capability);
	result.split_codec = std::move(split_codec);
	result.extension_payload = std::move(payload);
	result.estimated_cardinality = estimated_cardinality;
	result.estimated_bytes = estimated_bytes;
	result.Validate();
	return result;
}

ScanSplit ScanSplit::EmptyFile() {
	ScanSplit result;
	result.kind = ScanSplitKind::FILE;
	result.split_id = "empty";
	result.empty = true;
	result.estimated_cardinality = optional_idx(0);
	result.estimated_bytes = optional_idx(0);
	result.Validate();
	return result;
}

ScanSplit ScanSplit::EmptyExtension(DistributedExtensionCapabilityReference capability,
                                    DistributedPayloadCodec split_codec) {
	ScanSplit result;
	result.kind = ScanSplitKind::EXTENSION;
	result.split_id = "empty";
	result.empty = true;
	result.extension_capability = std::move(capability);
	result.split_codec = std::move(split_codec);
	result.estimated_cardinality = optional_idx(0);
	result.estimated_bytes = optional_idx(0);
	result.Validate();
	return result;
}

void ScanSplit::Validate() const {
	if (split_id.empty()) {
		throw SerializationException("scan split has an empty split_id");
	}
	switch (kind) {
	case ScanSplitKind::FILE:
		if (!extension_capability.extension_name.empty() || !split_codec.name.empty() || split_codec.version != 0 ||
		    !extension_payload.empty()) {
			throw SerializationException("file scan split contains extension state");
		}
		if (empty) {
			if (!file.path.empty() || file.extended_info) {
				throw SerializationException("empty file scan split contains file state");
			}
		} else if (file.path.empty()) {
			throw SerializationException("file scan split has an empty path");
		}
		break;
	case ScanSplitKind::EXTENSION:
		if (!file.path.empty() || file.extended_info) {
			throw SerializationException("extension scan split contains file state");
		}
		split_codec.Validate("Extension scan split");
		extension_capability.Validate();
		if (extension_capability.capability.kind != DistributedExtensionCapabilityKind::TABLE_FUNCTION) {
			throw SerializationException("extension scan split capability is not a table function");
		}
		if (empty && !extension_payload.empty()) {
			throw SerializationException("empty extension scan split contains a payload");
		}
		break;
	default:
		throw SerializationException("unknown scan split kind: %u", static_cast<unsigned int>(kind));
	}
}

void ScanSplit::Serialize(Serializer &serializer) const {
	Validate();
	serializer.WriteProperty(1, "kind", static_cast<uint8_t>(kind));
	serializer.WriteProperty(2, "split_id", split_id);
	serializer.WriteProperty(3, "empty", empty);
	if (kind == ScanSplitKind::FILE) {
		serializer.WriteObject(10, "file", [&](Serializer &object) {
			object.WriteProperty(1, "path", file.path);
			unordered_map<string, Value> options;
			if (file.extended_info) {
				options = file.extended_info->options;
			}
			object.WriteProperty(2, "options", options);
		});
	} else {
		serializer.WriteObject(20, "extension_capability",
		                       [&](Serializer &object) { extension_capability.Serialize(object); });
		serializer.WriteObject(21, "split_codec", [&](Serializer &object) { split_codec.Serialize(object); });
		serializer.WriteProperty(22, "extension_payload", extension_payload);
	}
	serializer.WriteProperty(30, "estimated_cardinality", estimated_cardinality);
	serializer.WriteProperty(31, "estimated_bytes", estimated_bytes);
}

ScanSplit ScanSplit::Deserialize(Deserializer &deserializer) {
	ScanSplit split;
	split.kind = static_cast<ScanSplitKind>(deserializer.ReadProperty<uint8_t>(1, "kind"));
	split.split_id = deserializer.ReadProperty<string>(2, "split_id");
	split.empty = deserializer.ReadProperty<bool>(3, "empty");
	if (split.kind == ScanSplitKind::FILE) {
		deserializer.ReadObject(10, "file", [&](Deserializer &object) {
			split.file.path = object.ReadProperty<string>(1, "path");
			auto options = object.ReadProperty<unordered_map<string, Value>>(2, "options");
			if (!options.empty()) {
				auto extended = make_shared_ptr<ExtendedOpenFileInfo>();
				extended->options = std::move(options);
				split.file.extended_info = std::move(extended);
			}
		});
	} else if (split.kind == ScanSplitKind::EXTENSION) {
		deserializer.ReadObject(20, "extension_capability", [&](Deserializer &object) {
			split.extension_capability = DistributedExtensionCapabilityReference::Deserialize(object);
		});
		deserializer.ReadObject(21, "split_codec", [&](Deserializer &object) {
			split.split_codec = DistributedPayloadCodec::Deserialize(object);
		});
		split.extension_payload = deserializer.ReadProperty<string>(22, "extension_payload");
	} else {
		throw SerializationException("unknown scan split kind: %u", static_cast<unsigned int>(split.kind));
	}
	split.estimated_cardinality = deserializer.ReadProperty<optional_idx>(30, "estimated_cardinality");
	split.estimated_bytes = deserializer.ReadProperty<optional_idx>(31, "estimated_bytes");
	split.Validate();
	return split;
}

idx_t ScanSplitBatch::file_count() const {
	idx_t result = 0;
	for (const auto &split : splits) {
		if (!split.IsExtension() && !split.empty) {
			result++;
		}
	}
	return result;
}

bool ScanSplitBatch::IsExtension() const {
	Validate();
	return splits[0].IsExtension();
}

static idx_t AggregateScanSplitEstimate(const vector<ScanSplit> &splits, bool cardinality) {
	idx_t result = 0;
	for (const auto &split : splits) {
		const auto &estimate = cardinality ? split.estimated_cardinality : split.estimated_bytes;
		if (!estimate.IsValid()) {
			return 0;
		}
		result = SaturatingAddScanSplitEstimate(result, estimate.GetIndex());
	}
	return result;
}

idx_t ScanSplitBatch::EstimatedCardinality() const {
	return AggregateScanSplitEstimate(splits, true);
}

idx_t ScanSplitBatch::EstimatedBytes() const {
	return AggregateScanSplitEstimate(splits, false);
}

void ScanSplitBatch::Validate() const {
	if (splits.empty()) {
		throw SerializationException("scan split batch is empty");
	}
	set<string> split_ids;
	const auto expected_kind = splits[0].kind;
	const auto expected_capability = splits[0].extension_capability;
	const auto expected_codec = splits[0].split_codec;
	for (const auto &split : splits) {
		split.Validate();
		if (split.kind != expected_kind) {
			throw SerializationException("scan split batch mixes file and extension splits");
		}
		if (!split_ids.insert(split.split_id).second) {
			throw SerializationException("scan split_id '%s' appears more than once in a batch", split.split_id);
		}
		if (split.empty && splits.size() != 1) {
			throw SerializationException("an explicit empty scan split must be the only split in its batch");
		}
		if (expected_kind == ScanSplitKind::EXTENSION &&
		    (split.extension_capability != expected_capability || split.split_codec != expected_codec)) {
			throw SerializationException("extension scan split batch contains different protocol identities");
		}
	}
}

void ScanSplitBatch::Merge(ScanSplitBatch other) {
	Validate();
	other.Validate();
	if (splits[0].empty || other.splits[0].empty) {
		throw InvalidInputException("an explicit empty scan split cannot be merged with another batch");
	}
	if (splits[0].kind != other.splits[0].kind) {
		throw InvalidInputException("cannot merge file and extension scan split batches");
	}
	if (splits[0].kind == ScanSplitKind::EXTENSION &&
	    (splits[0].extension_capability != other.splits[0].extension_capability ||
	     splits[0].split_codec != other.splits[0].split_codec)) {
		throw InvalidInputException("cannot merge extension scan split batches with different protocol identities");
	}
	set<string> split_ids;
	for (const auto &split : splits) {
		split_ids.insert(split.split_id);
	}
	for (const auto &split : other.splits) {
		if (!split_ids.insert(split.split_id).second) {
			throw InvalidInputException("cannot merge scan split batches with duplicate split_id '%s'", split.split_id);
		}
	}
	splits.insert(splits.end(), std::make_move_iterator(other.splits.begin()),
	              std::make_move_iterator(other.splits.end()));
	Validate();
}

vector<ScanSplitBatch> ScanSplitBatch::Explode() const {
	Validate();
	vector<ScanSplitBatch> result;
	result.reserve(splits.size());
	for (const auto &split : splits) {
		ScanSplitBatch batch;
		batch.splits.push_back(split);
		result.push_back(std::move(batch));
	}
	return result;
}

void ScanSplitBatch::Serialize(Serializer &serializer) const {
	Validate();
	serializer.WriteList(1, "splits", splits.size(), [&](Serializer::List &list, idx_t index) {
		list.WriteObject([&](Serializer &object) { splits[index].Serialize(object); });
	});
}

ScanSplitBatch ScanSplitBatch::Deserialize(Deserializer &deserializer) {
	ScanSplitBatch batch;
	deserializer.ReadList(1, "splits", [&](Deserializer::List &list, idx_t) {
		list.ReadObject([&](Deserializer &object) { batch.splits.push_back(ScanSplit::Deserialize(object)); });
	});
	batch.Validate();
	return batch;
}

std::string ScanSplitBatch::SerializeToBytes() const {
	MemoryStream stream(Allocator::DefaultAllocator());
	BinarySerializer serializer(stream);
	serializer.Begin();
	Serialize(serializer);
	serializer.End();
	return std::string(reinterpret_cast<const char *>(stream.GetData()), stream.GetPosition());
}

std::string ScanSplitBatch::SerializeToBase64() const {
	auto bytes = SerializeToBytes();
	if (bytes.empty()) {
		return std::string();
	}
	return Blob::ToBase64(string_t(bytes.data(), bytes.size()));
}

ScanSplitBatch ScanSplitBatch::DeserializeFromBytes(const std::string &bytes) {
	if (bytes.empty()) {
		throw SerializationException("cannot deserialize an empty scan split batch");
	}
	auto *data_ptr = reinterpret_cast<data_ptr_t>(const_cast<char *>(bytes.data()));
	MemoryStream stream(data_ptr, bytes.size());
	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	auto desc = Deserialize(deserializer);
	deserializer.End();
	return desc;
}

ScanSplitBatch ScanSplitBatch::DeserializeFromBase64(const std::string &base64) {
	if (base64.empty()) {
		throw SerializationException("cannot deserialize an empty base64 scan split batch");
	}
	auto raw = Blob::FromBase64(string_t(base64.data(), base64.size()));
	return DeserializeFromBytes(raw);
}

bool ApplyScanSplitBatchesToPlan(duckdb::PhysicalPlan &plan, const std::unordered_map<idx_t, ScanSplitBatch> &batches,
                                 std::string *error) {
	if (!plan.HasRoot()) {
		if (error) {
			*error = "plan has no root";
		}
		return false;
	}
	if (batches.empty()) {
		if (error) {
			*error = "scan split batch map is empty";
		}
		return false;
	}
	for (const auto &entry : batches) {
		entry.second.Validate();
	}
	ScanSplitBatchReferenceMap batch_references;
	batch_references.reserve(batches.size());
	for (const auto &entry : batches) {
		batch_references.emplace(entry.first, &entry.second);
	}
	ApplyScanSplitsStats stats;
	idx_t max_id = MaxScanNodeId(plan.Root(), 0);
	for (const auto &kv : batches) {
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
		if (batch_references.find(kv.first) != batch_references.end()) {
			continue;
		}
		auto assignment = ResolveScanNodeAlias(batch_references, alias_to_parent, kv.first);
		if (assignment != batch_references.end()) {
			batch_references.emplace(kv.first, assignment->second);
			stats.copied_batches++;
		}
	}
	set<idx_t> plan_scan_node_ids;
	CollectScanNodeIds(plan.Root(), plan_scan_node_ids);
	for (const auto &entry : batches) {
		if (plan_scan_node_ids.find(entry.first) == plan_scan_node_ids.end()) {
			if (error) {
				*error =
				    "scan split batch node_id=" + std::to_string(entry.first) + " is not present in the worker plan";
			}
			return false;
		}
	}
	set<idx_t> matched_batches;
	ApplyScanSplitBatchesToOperator(plan.Root(), batch_references, matched_batches, stats, error);
	if (stats.invalid_assignment != 0) {
		if (error && error->empty()) {
			*error = "one or more scan split batches had invalid assignments";
		}
		return false;
	}
	for (const auto &entry : batches) {
		if (matched_batches.find(entry.first) == matched_batches.end()) {
			if (error && error->empty()) {
				*error =
				    "scan split batch node_id=" + std::to_string(entry.first) + " is not present in the worker plan";
			}
			return false;
		}
	}
	if (stats.applied == 0) {
		if (error && error->empty()) {
			*error = "no scan splits applied";
		}
		return false;
	}
	return true;
}

namespace {

using FteExtensionSplitBatchCache = unordered_map<const FteSplitQueue *, ScanSplitBatch>;
using FteDynamicFileListCache = unordered_map<const FteSplitQueue *, shared_ptr<MultiFileList>>;

static bool ApplyFteExtensionScanSplits(PhysicalTableScan &scan, const std::shared_ptr<FteSplitQueue> &queue,
                                        FteExtensionSplitBatchCache &batch_cache, string *error) {
	auto cached_batch = batch_cache.find(queue.get());
	if (cached_batch != batch_cache.end()) {
		return ApplyExtensionScanSplits(scan, cached_batch->second, error);
	}

	ScanSplitBatch merged;
	bool has_batch = false;
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
		if (next.input.kind != TaskInput::Kind::ScanSplitBatch) {
			SetApplyError(error, "FTE extension scan source queue received a non-scan split");
			return false;
		}
		auto batch = ScanSplitBatch::DeserializeFromBytes(next.input.scan_split_batch_bytes);
		if (!batch.IsExtension()) {
			SetApplyError(error, "FTE extension scan source queue received a file scan split");
			return false;
		}
		if (!has_batch) {
			merged = std::move(batch);
			has_batch = true;
		} else {
			merged.Merge(std::move(batch));
		}
	}
	if (!has_batch) {
		SetApplyError(error, "FTE extension scan source queue finished without an explicit split batch");
		return false;
	}
	auto inserted = batch_cache.emplace(queue.get(), std::move(merged));
	return ApplyExtensionScanSplits(scan, inserted.first->second, error);
}

bool ApplyFteScanSourceQueuesToOperator(PhysicalOperator &op,
                                        const std::unordered_map<idx_t, std::shared_ptr<FteSplitQueue>> &queues,
                                        FteExtensionSplitBatchCache &extension_batch_cache,
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
				if (scan.function.HasDistributedScanCallbacks()) {
					if (!ApplyFteExtensionScanSplits(scan, entry->second, extension_batch_cache, error)) {
						return false;
					}
					applied++;
				} else if (!scan.bind_data) {
					if (error) {
						*error = "FTE scan source queue target has null bind_data for scan_node_id=" +
						         std::to_string(node_id);
					}
					return false;
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
					scan.distributed_scan_splits_applied = true;
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
		if (!ApplyFteScanSourceQueuesToOperator(child.get(), queues, extension_batch_cache, dynamic_file_list_cache,
		                                        matched_queues, error, applied)) {
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
	ApplyScanSplitsStats stats;
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
	FteExtensionSplitBatchCache extension_batch_cache;
	FteDynamicFileListCache dynamic_file_list_cache;
	if (!ApplyFteScanSourceQueuesToOperator(plan.Root(), queue_references, extension_batch_cache,
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

bool ValidateScanSplitAssignments(const duckdb::PhysicalPlan &plan, const set<idx_t> &assigned_node_ids,
                                  string *error) {
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
			SetApplyError(error, "distributed table scan has no explicit worker split assignment for scan_node_id=" +
			                         std::to_string(node_id));
			return false;
		}
	}
	for (auto node_id : assigned_node_ids) {
		if (scan_node_ids.find(node_id) == scan_node_ids.end()) {
			SetApplyError(error, "scan split assignment node_id=" + std::to_string(node_id) +
			                         " is not present in the worker plan");
			return false;
		}
	}
	return true;
}

namespace {

static bool ValidateDistributedScanSplitsAppliedToOperator(const PhysicalOperator &op, string *error) {
	if (op.type == PhysicalOperatorType::TABLE_SCAN) {
		auto &scan = op.Cast<PhysicalTableScan>();
		if (!scan.extra_info.scan_node_id.IsValid()) {
			SetApplyError(error,
			              "distributed table scan '" + scan.function.name + "' has no runtime scan node identity");
			return false;
		}
		if (!scan.distributed_scan_splits_applied) {
			SetApplyError(error, "distributed table scan '" + scan.function.name +
			                         "' has no explicit worker split assignment");
			return false;
		}
	}
	for (const auto &child : op.children) {
		if (!ValidateDistributedScanSplitsAppliedToOperator(child.get(), error)) {
			return false;
		}
	}
	return true;
}

} // namespace

bool ValidateDistributedScanSplitsApplied(const duckdb::PhysicalPlan &plan, string *error) {
	if (!plan.HasRoot()) {
		SetApplyError(error, "plan has no root");
		return false;
	}
	return ValidateDistributedScanSplitsAppliedToOperator(plan.Root(), error);
}

bool HasDistributedScanSplitTargets(const duckdb::PhysicalPlan &plan) {
	if (!plan.HasRoot()) {
		return false;
	}
	set<idx_t> node_ids;
	CollectScanNodeIds(plan.Root(), node_ids);
	return !node_ids.empty();
}

} // namespace distributed
} // namespace duckdb
