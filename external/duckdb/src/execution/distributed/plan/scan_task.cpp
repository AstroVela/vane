// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/distributed/plan/scan_task.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/execution/distributed/plan/scan_task.hpp"
#include "duckdb/function/extension_file_list_provider.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/multi_file/multi_file_list.hpp"
#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/types/blob.hpp"
#include "duckdb/common/types/string_type.hpp"
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

static bool ApplyScanTasksToOperator(PhysicalOperator &op, const std::unordered_map<idx_t, ScanTaskDescriptor> &tasks,
                                     ApplyScanTasksStats &stats) {
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
				stats.missing_bind++;
			} else {
				auto *multi_bind = dynamic_cast<MultiFileBindData *>(scan.bind_data.get());
				if (!multi_bind) {
					auto *ext_provider = dynamic_cast<ExtensionFileListProvider *>(scan.bind_data.get());
					if (ext_provider) {
						vector<string> paths;
						paths.reserve(it->second.files.size());
						for (auto &f : it->second.files) {
							paths.push_back(f.path);
						}
						ext_provider->SetFileList(paths);
						const idx_t file_count = it->second.file_count();
						scan.extra_info.total_files = optional_idx(file_count);
						scan.extra_info.filtered_files = optional_idx(file_count);
						stats.applied++;
						applied_any = true;
					} else {
						stats.non_multi_bind++;
					}
				} else {
					multi_bind->file_list = duckdb::make_shared_ptr<SimpleMultiFileList>(it->second.files);
					const idx_t file_count = it->second.file_count();
					scan.extra_info.total_files = optional_idx(file_count);
					scan.extra_info.filtered_files = optional_idx(file_count);
					stats.applied++;
					applied_any = true;
				}
			}
		}
	}

	for (auto &child : op.children) {
		if (ApplyScanTasksToOperator(child.get(), tasks, stats)) {
			applied_any = true;
		}
	}
	return applied_any;
}

} // namespace

void ScanTaskDescriptor::Serialize(Serializer &serializer) const {
	serializer.WriteList(1, "files", files.size(), [&](Serializer::List &list, idx_t i) {
		list.WriteObject([&](Serializer &obj) {
			obj.WriteProperty(1, "path", files[i].path);
			unordered_map<string, Value> options;
			if (files[i].extended_info) {
				options = files[i].extended_info->options;
			}
			obj.WriteProperty(2, "options", options);
		});
	});
	serializer.WriteProperty(3, "estimated_cardinality", estimated_cardinality);
	serializer.WriteProperty(4, "estimated_bytes", estimated_bytes);
}

ScanTaskDescriptor ScanTaskDescriptor::Deserialize(Deserializer &deserializer) {
	ScanTaskDescriptor desc;
	deserializer.ReadList(1, "files", [&](Deserializer::List &list, idx_t) {
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
	deserializer.ReadPropertyWithDefault<idx_t>(3, "estimated_cardinality", desc.estimated_cardinality);
	deserializer.ReadPropertyWithDefault<idx_t>(4, "estimated_bytes", desc.estimated_bytes);
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
		return ScanTaskDescriptor();
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
		return ScanTaskDescriptor();
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
	auto expanded_tasks = tasks;
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
		if (expanded_tasks.find(kv.first) != expanded_tasks.end()) {
			continue;
		}
		auto base_it = expanded_tasks.find(kv.second);
		if (base_it != expanded_tasks.end()) {
			expanded_tasks.emplace(kv.first, base_it->second);
			stats.copied_tasks++;
		}
	}
	ApplyScanTasksToOperator(plan.Root(), expanded_tasks, stats);
	if (stats.applied == 0) {
		if (error) {
			*error = "no scan tasks applied";
		}
		return false;
	}
	return true;
}

namespace {

bool ApplyFteScanSourceQueuesToOperator(PhysicalOperator &op,
                                        const std::unordered_map<idx_t, std::shared_ptr<FteSplitQueue>> &queues,
                                        std::string *error, idx_t &applied) {
	bool ok = true;
	if (op.type == PhysicalOperatorType::TABLE_SCAN) {
		auto &scan = op.Cast<PhysicalTableScan>();
		if (scan.extra_info.scan_node_id.IsValid()) {
			const auto node_id = scan.extra_info.scan_node_id.GetIndex();
			auto entry = queues.find(node_id);
			if (entry != queues.end()) {
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
				auto *multi_bind = dynamic_cast<MultiFileBindData *>(scan.bind_data.get());
				if (!multi_bind) {
					auto *ext_provider = dynamic_cast<ExtensionFileListProvider *>(scan.bind_data.get());
					if (!ext_provider) {
						if (error) {
							*error = "FTE dynamic scan source currently requires MultiFileBindData or "
							         "ExtensionFileListProvider for scan_node_id=" +
							         std::to_string(node_id);
						}
						return false;
					}
					FteDynamicScanFileList dynamic_files(entry->second);
					auto files = dynamic_files.GetAllFiles();
					vector<string> paths;
					paths.reserve(files.size());
					for (auto &file : files) {
						paths.push_back(file.path);
					}
					ext_provider->SetFileList(paths);
					const idx_t file_count = files.size();
					scan.extra_info.total_files = optional_idx(file_count);
					scan.extra_info.filtered_files = optional_idx(file_count);
					applied++;
				} else {
					multi_bind->file_list = make_shared_ptr<FteDynamicScanFileList>(entry->second);
					scan.extra_info.total_files = optional_idx();
					scan.extra_info.filtered_files = optional_idx();
					applied++;
				}
			}
		}
	}
	for (auto &child : op.children) {
		if (!ApplyFteScanSourceQueuesToOperator(child.get(), queues, error, applied)) {
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
	idx_t applied = 0;
	if (!ApplyFteScanSourceQueuesToOperator(plan.Root(), queues, error, applied)) {
		return false;
	}
	if (applied == 0) {
		if (error) {
			*error = "no FTE scan source queues applied";
		}
		return false;
	}
	return true;
}

} // namespace distributed
} // namespace duckdb
