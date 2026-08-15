// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/translator_scan.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/common/multi_file/multi_file_list.hpp"
#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/execution/physical_plan.hpp"
#include "duckdb/main/database.hpp"

#include <algorithm>

namespace duckdb {
namespace distributed {
namespace {

ExtraOperatorInfo CopyExtraOperatorInfo(const ExtraOperatorInfo &info) {
	ExtraOperatorInfo copy;
	copy.file_filters = info.file_filters;
	copy.total_files = info.total_files;
	copy.filtered_files = info.filtered_files;
	copy.scan_node_id = info.scan_node_id;
	copy.scan_group_id = info.scan_group_id;
	if (info.sample_options) {
		copy.sample_options = info.sample_options->Copy();
	}
	return copy;
}

std::vector<std::vector<OpenFileInfo>> GroupFilesByCount(const std::vector<OpenFileInfo> &files, size_t max_tasks) {
	std::vector<std::vector<OpenFileInfo>> groups;
	if (files.empty()) {
		return groups;
	}
	if (max_tasks == 0 || max_tasks > files.size()) {
		max_tasks = files.size();
	}
	const size_t files_per_task = (files.size() + max_tasks - 1) / max_tasks;
	for (size_t start = 0; start < files.size(); start += files_per_task) {
		const size_t end = std::min(files.size(), start + files_per_task);
		std::vector<OpenFileInfo> group;
		group.reserve(end - start);
		for (size_t idx = start; idx < end; ++idx) {
			group.push_back(files[idx]);
		}
		groups.push_back(std::move(group));
	}
	return groups;
}

uint64_t ComputeMaxSplitBytes(uint64_t total_bytes, size_t file_count, uint64_t max_partition_bytes,
                              uint64_t open_cost_bytes, size_t min_partition_num) {
	if (min_partition_num == 0) {
		min_partition_num = 1;
	}
	uint64_t total_with_open_cost = total_bytes + file_count * open_cost_bytes;
	uint64_t bytes_per_partition = total_with_open_cost / min_partition_num;
	return std::min(max_partition_bytes, std::max(open_cost_bytes, bytes_per_partition));
}

std::vector<std::vector<OpenFileInfo>> PackFilesByMaxSplitBytes(const std::vector<OpenFileInfo> &files,
                                                                const std::vector<uint64_t> &sizes,
                                                                uint64_t max_split_bytes, uint64_t open_cost_bytes) {
	std::vector<std::vector<OpenFileInfo>> partitions;
	if (files.empty()) {
		return partitions;
	}

	std::vector<OpenFileInfo> current;
	uint64_t current_size = 0;

	for (size_t i = 0; i < files.size(); ++i) {
		uint64_t effective_size = sizes[i] + open_cost_bytes;
		if (!current.empty() && current_size + effective_size > max_split_bytes) {
			partitions.push_back(std::move(current));
			current = {};
			current_size = 0;
		}
		current.push_back(files[i]);
		current_size += effective_size;
	}
	if (!current.empty()) {
		partitions.push_back(std::move(current));
	}
	return partitions;
}

bool HasPositiveSizes(const std::vector<uint64_t> &sizes) {
	for (auto size : sizes) {
		if (size > 0) {
			return true;
		}
	}
	return false;
}

uint64_t SaturatingAddExtensionScanEstimate(uint64_t left, uint64_t right) {
	const auto maximum = NumericLimits<uint64_t>::Maximum();
	return right > maximum - left ? maximum : left + right;
}

size_t ResolveScanTaskTargetCount(size_t source_count, const DuckDBExecutionConfig &exec_cfg) {
	if (source_count == 0) {
		return 0;
	}

	size_t target = 0;
	auto worker_slots = exec_cfg.distributed_worker_slots();
	if (worker_slots > 0) {
		target = worker_slots;
	} else {
		auto node_count = exec_cfg.distributed_node_count();
		if (node_count > 0) {
			target = node_count;
		}
	}
	if (target == 0) {
		target = source_count;
	}
	auto min_partitions = exec_cfg.scan_task_min_partition_num();
	if (min_partitions > 0) {
		target = std::max(target, min_partitions);
	}
	return target;
}

vector<vector<idx_t>> GroupIndexesByCount(idx_t count, size_t max_tasks) {
	vector<vector<idx_t>> groups;
	if (count == 0) {
		return groups;
	}
	size_t tasks = max_tasks == 0 ? static_cast<size_t>(count) : std::min(max_tasks, static_cast<size_t>(count));
	const idx_t per_task = (count + tasks - 1) / tasks;
	for (idx_t start = 0; start < count; start += per_task) {
		const idx_t end = std::min(count, start + per_task);
		vector<idx_t> group;
		group.reserve(end - start);
		for (idx_t idx = start; idx < end; ++idx) {
			group.push_back(idx);
		}
		groups.push_back(std::move(group));
	}
	return groups;
}

vector<vector<idx_t>> GroupWeightedIndexesByTargetCount(const vector<uint64_t> &weights, size_t target_groups) {
	vector<vector<idx_t>> groups;
	const auto item_count = static_cast<idx_t>(weights.size());
	if (item_count == 0) {
		return groups;
	}
	if (target_groups == 0 || target_groups >= static_cast<size_t>(item_count)) {
		return GroupIndexesByCount(item_count, target_groups);
	}
	if (!HasPositiveSizes(weights)) {
		return GroupIndexesByCount(item_count, target_groups);
	}

	long double remaining_weight = 0;
	for (auto weight : weights) {
		remaining_weight += static_cast<long double>(weight);
	}

	size_t remaining_groups = target_groups;
	vector<idx_t> current;
	current.reserve(static_cast<size_t>(item_count + target_groups - 1) / target_groups);
	long double current_weight = 0;

	for (idx_t i = 0; i < item_count; ++i) {
		const auto weight = weights[static_cast<size_t>(i)];
		current.push_back(i);
		current_weight += static_cast<long double>(weight);
		remaining_weight -= static_cast<long double>(weight);

		if (remaining_groups <= 1) {
			continue;
		}

		const auto remaining_items = static_cast<size_t>(item_count - i - 1);
		const auto target_weight = (current_weight + remaining_weight) / static_cast<long double>(remaining_groups);
		const bool reached_target = current_weight >= target_weight;
		const bool must_split = remaining_items == (remaining_groups - 1);
		if (reached_target || must_split) {
			groups.push_back(std::move(current));
			current = {};
			current_weight = 0;
			--remaining_groups;
		}
	}

	if (!current.empty()) {
		groups.push_back(std::move(current));
	}
	return groups;
}

std::vector<uint64_t> GetFileSizesFromDB(const std::vector<OpenFileInfo> &files,
                                         const shared_ptr<DatabaseInstance> &db) {
	std::vector<uint64_t> sizes;
	sizes.reserve(files.size());
	if (!db) {
		return sizes;
	}

	auto &fs = FileSystem::GetFileSystem(*db);
	for (const auto &file : files) {
		try {
			auto handle = fs.OpenFile(file, FileOpenFlags::FILE_FLAGS_READ);
			if (!handle) {
				sizes.push_back(0);
				continue;
			}
			auto size = fs.GetFileSize(*handle);
			sizes.push_back(size >= 0 ? static_cast<uint64_t>(size) : 0);
		} catch (...) {
			sizes.push_back(0);
		}
	}
	return sizes;
}

TableFunctionDistributedScanInput MakeDistributedScanInput(const PhysicalTableScan &scan) {
	return TableFunctionDistributedScanInput(*scan.bind_data, scan.column_ids, scan.projection_ids,
	                                         scan.table_filters.get(), scan.estimated_cardinality);
}

vector<ScanTaskDescriptor> MakeExtensionScanTasks(const PhysicalTableScan &scan, const DuckDBExecutionConfig &exec_cfg,
                                                  const shared_ptr<DatabaseInstance> &db) {
	if (!scan.function.HasSerializationCallbacks()) {
		throw SerializationException("Distributed table function '%s' requires complete serialize and deserialize "
		                             "callbacks; worker rebind is not supported",
		                             scan.function.name);
	}
	const auto &callbacks = scan.function.GetDistributedScanCallbacks();
	callbacks.Validate(scan.function.name);
	const auto &capability = callbacks.GetCapability();
	if (!db) {
		throw InvalidInputException("Distributed extension scan '%s' requires a DatabaseInstance for capability "
		                            "validation",
		                            scan.function.name);
	}
	DistributedExtensionManager::Get(*db).RequireCapability(capability);

	auto planned_tasks = callbacks.plan(MakeDistributedScanInput(scan));
	if (planned_tasks.empty()) {
		ScanTaskDescriptor empty_task;
		empty_task.kind = ScanTaskKind::EXTENSION;
		empty_task.extension_capability = capability;
		empty_task.task_codec = callbacks.task_codec;
		empty_task.source_task_partition_id = 0;
		empty_task.Validate();
		return {std::move(empty_task)};
	}
	ScanTaskDescriptor validation_descriptor;
	validation_descriptor.kind = ScanTaskKind::EXTENSION;
	validation_descriptor.extension_capability = capability;
	validation_descriptor.task_codec = callbacks.task_codec;
	validation_descriptor.extension_tasks = std::move(planned_tasks);
	bool validation_complete_cardinality = true;
	bool validation_complete_bytes = true;
	for (const auto &task : validation_descriptor.extension_tasks) {
		if (task.estimated_cardinality.IsValid()) {
			validation_descriptor.estimated_cardinality = SaturatingAddExtensionScanEstimate(
			    validation_descriptor.estimated_cardinality, task.estimated_cardinality.GetIndex());
		} else {
			validation_complete_cardinality = false;
		}
		if (task.estimated_bytes.IsValid()) {
			validation_descriptor.estimated_bytes = SaturatingAddExtensionScanEstimate(
			    validation_descriptor.estimated_bytes, task.estimated_bytes.GetIndex());
		} else {
			validation_complete_bytes = false;
		}
	}
	if (!validation_complete_cardinality) {
		validation_descriptor.estimated_cardinality = 0;
	}
	if (!validation_complete_bytes) {
		validation_descriptor.estimated_bytes = 0;
	}
	validation_descriptor.Validate();
	auto &elementary_tasks = validation_descriptor.extension_tasks;

	auto target_count = ResolveScanTaskTargetCount(elementary_tasks.size(), exec_cfg);
	vector<vector<idx_t>> groups;
	vector<uint64_t> weights;
	weights.reserve(elementary_tasks.size());
	bool complete_weights = true;
	for (const auto &task : elementary_tasks) {
		if (!task.estimated_bytes.IsValid()) {
			complete_weights = false;
			break;
		}
		weights.push_back(task.estimated_bytes.GetIndex());
	}
	if (complete_weights && HasPositiveSizes(weights)) {
		groups = GroupWeightedIndexesByTargetCount(weights, target_count);
	} else {
		groups = GroupIndexesByCount(elementary_tasks.size(), target_count);
	}

	vector<ScanTaskDescriptor> result;
	result.reserve(groups.size());
	for (auto &group : groups) {
		ScanTaskDescriptor descriptor;
		descriptor.kind = ScanTaskKind::EXTENSION;
		descriptor.extension_capability = capability;
		descriptor.task_codec = callbacks.task_codec;
		descriptor.source_task_partition_id = result.size();
		bool complete_cardinality = true;
		bool complete_bytes = true;
		for (auto task_index : group) {
			auto &task = elementary_tasks[task_index];
			if (task.estimated_cardinality.IsValid()) {
				descriptor.estimated_cardinality = SaturatingAddExtensionScanEstimate(
				    descriptor.estimated_cardinality, task.estimated_cardinality.GetIndex());
			} else {
				complete_cardinality = false;
			}
			if (task.estimated_bytes.IsValid()) {
				descriptor.estimated_bytes =
				    SaturatingAddExtensionScanEstimate(descriptor.estimated_bytes, task.estimated_bytes.GetIndex());
			} else {
				complete_bytes = false;
			}
			descriptor.extension_tasks.push_back(std::move(task));
		}
		if (!complete_cardinality) {
			descriptor.estimated_cardinality = 0;
		}
		if (!complete_bytes) {
			descriptor.estimated_bytes = 0;
		}
		descriptor.Validate();
		result.push_back(std::move(descriptor));
	}
	return result;
}

} // namespace

DuckPhysicalPlanRef MakeTableScanPlan(const PhysicalTableScan &scan) {
	Allocator &alloc = Allocator::DefaultAllocator();
	auto plan = std::make_shared<PhysicalPlan>(alloc);

	unique_ptr<FunctionData> bind_data;
	TableFunction function = scan.function;
	if (scan.function.HasDistributedScanCallbacks()) {
		if (!scan.bind_data) {
			throw SerializationException("Distributed table function '%s' requires bind data", scan.function.name);
		}
		if (!scan.function.HasSerializationCallbacks()) {
			throw SerializationException("Distributed table function '%s' requires complete serialize and deserialize "
			                             "callbacks; worker rebind is not supported",
			                             scan.function.name);
		}
		const auto &callbacks = scan.function.GetDistributedScanCallbacks();
		callbacks.Validate(scan.function.name);
		bind_data = callbacks.create_worker_bind(MakeDistributedScanInput(scan));
		if (!bind_data) {
			throw InvalidInputException("Distributed table function '%s' returned null from create_worker_bind",
			                            scan.function.name);
		}
	} else {
		if (!scan.bind_data) {
			throw NotImplementedException("Distributed execution does not support table function \"%s\": bind data is "
			                              "missing",
			                              scan.function.name);
		}
		try {
			bind_data = scan.bind_data->Copy();
		} catch (const NotImplementedException &ex) {
			ErrorData error(ex);
			throw NotImplementedException("Distributed execution does not support table function \"%s\": %s",
			                              scan.function.name, error.RawMessage());
		}
		auto *multi_bind = dynamic_cast<MultiFileBindData *>(bind_data.get());
		if (!multi_bind) {
			throw NotImplementedException("Distributed execution does not support table function \"%s\": its bind data "
			                              "does not provide a distributable file list",
			                              scan.function.name);
		}
		// The coordinator file list is never a worker fallback. Every worker must
		// receive an explicit static descriptor or FTE queue, including an empty
		// descriptor for a legal zero-file scan.
		multi_bind->file_list = make_shared_ptr<SimpleMultiFileList>(vector<OpenFileInfo> {});
	}
	auto table_filters = scan.table_filters ? scan.table_filters->Copy() : nullptr;
	auto extra_info = CopyExtraOperatorInfo(scan.extra_info);

	auto &scan_op = plan->Make<PhysicalTableScan>(scan.GetTypes(), std::move(function), std::move(bind_data),
	                                              scan.returned_types, scan.column_ids, scan.projection_ids, scan.names,
	                                              std::move(table_filters), scan.estimated_cardinality,
	                                              std::move(extra_info), scan.parameters, scan.virtual_columns);
	plan->SetRoot(scan_op);
	return plan;
}

TableScanTaskSet MakeTableScanTasks(const PhysicalTableScan &scan, const DuckDBExecutionConfig &exec_cfg,
                                    const shared_ptr<DatabaseInstance> &db) {
	vector<ScanTaskDescriptor> tasks;

	if (!scan.bind_data) {
		throw NotImplementedException("Distributed execution does not support table function \"%s\": bind data is "
		                              "missing",
		                              scan.function.name);
	}
	if (scan.function.HasDistributedScanCallbacks()) {
		return {MakeExtensionScanTasks(scan, exec_cfg, db), false};
	}

	vector<OpenFileInfo> files;
	auto *multi_bind = dynamic_cast<MultiFileBindData *>(scan.bind_data.get());
	if (multi_bind && multi_bind->file_list) {
		files = multi_bind->file_list->GetAllFiles();
	} else {
		throw NotImplementedException("Distributed execution does not support table function \"%s\": its bind data "
		                              "does not provide a distributable file list",
		                              scan.function.name);
	}
	if (files.empty()) {
		ScanTaskDescriptor empty_task;
		empty_task.source_task_partition_id = 0;
		empty_task.Validate();
		tasks.push_back(std::move(empty_task));
		return {std::move(tasks), false};
	}
	const idx_t estimated_scan_rows =
	    scan.estimated_cardinality == DConstants::INVALID_INDEX ? 0 : scan.estimated_cardinality;
	std::vector<uint64_t> file_sizes;
	bool file_sizes_loaded = false;
	bool file_sizes_complete = false;
	uint64_t total_file_bytes = 0;
	auto ensure_file_sizes = [&]() {
		if (file_sizes_loaded) {
			return;
		}
		file_sizes_loaded = true;
		if (!db) {
			return;
		}
		file_sizes = GetFileSizesFromDB(files, db);
		file_sizes_complete = file_sizes.size() == files.size();
		total_file_bytes = 0;
		for (auto size : file_sizes) {
			total_file_bytes += size;
		}
	};
	auto estimate_rows_for_task = [&](const auto &task_files, uint64_t task_bytes) -> idx_t {
		if (estimated_scan_rows == 0) {
			return 0;
		}
		if (file_sizes_complete && total_file_bytes > 0 && task_bytes > 0) {
			auto scaled = static_cast<long double>(estimated_scan_rows) * static_cast<long double>(task_bytes) /
			              static_cast<long double>(total_file_bytes);
			return static_cast<idx_t>(scaled);
		}
		if (!files.empty() && !task_files.empty()) {
			auto scaled = static_cast<long double>(estimated_scan_rows) * static_cast<long double>(task_files.size()) /
			              static_cast<long double>(files.size());
			return static_cast<idx_t>(scaled);
		}
		return 0;
	};
	auto estimate_bytes_for_group = [&](const std::vector<OpenFileInfo> &group) -> uint64_t {
		ensure_file_sizes();
		if (file_sizes.size() != files.size()) {
			return 0;
		}
		uint64_t bytes = 0;
		for (const auto &group_file : group) {
			for (size_t fi = 0; fi < files.size(); ++fi) {
				if (files[fi].path == group_file.path) {
					bytes += file_sizes[fi];
					break;
				}
			}
		}
		return bytes;
	};
	if (files.size() == 1) {
		ensure_file_sizes();
		ScanTaskDescriptor task;
		task.estimated_cardinality = estimate_rows_for_task(files, total_file_bytes);
		task.estimated_bytes = static_cast<idx_t>(total_file_bytes);
		task.source_task_partition_id = 0;
		task.files = std::move(files);
		tasks.push_back(std::move(task));
		return {std::move(tasks), false};
	}

	size_t target_task_count = ResolveScanTaskTargetCount(files.size(), exec_cfg);
	const bool use_size_thresholds = exec_cfg.scan_task_size_grouping_enabled() &&
	                                 (exec_cfg.scan_task_min_bytes() > 0 || exec_cfg.scan_task_max_bytes() > 0);
	std::vector<std::vector<OpenFileInfo>> groups;
	if (use_size_thresholds && db) {
		ensure_file_sizes();
		auto sizes = file_sizes;
		if (!sizes.empty() && file_sizes_complete && HasPositiveSizes(sizes)) {
			uint64_t total_bytes = 0;
			for (auto size : sizes) {
				total_bytes += size;
			}
			size_t min_partitions = exec_cfg.scan_task_min_partition_num();
			if (min_partitions == 0) {
				min_partitions = target_task_count;
			}
			uint64_t max_split = ComputeMaxSplitBytes(total_bytes, files.size(), exec_cfg.scan_task_max_bytes(),
			                                          exec_cfg.scan_task_open_cost_bytes(), min_partitions);
			groups = PackFilesByMaxSplitBytes(files, sizes, max_split, exec_cfg.scan_task_open_cost_bytes());
			if (target_task_count > 0 && groups.size() > target_task_count) {
				vector<uint64_t> group_weights;
				group_weights.reserve(groups.size());
				for (const auto &group : groups) {
					group_weights.push_back(estimate_bytes_for_group(group));
				}
				auto merged_indexes = GroupWeightedIndexesByTargetCount(group_weights, target_task_count);
				std::vector<std::vector<OpenFileInfo>> merged_groups;
				merged_groups.reserve(merged_indexes.size());
				for (const auto &idx_group : merged_indexes) {
					std::vector<OpenFileInfo> merged;
					for (auto idx : idx_group) {
						const auto &src = groups[static_cast<size_t>(idx)];
						merged.insert(merged.end(), src.begin(), src.end());
					}
					merged_groups.push_back(std::move(merged));
				}
				groups = std::move(merged_groups);
			}
		}
	}
	if (groups.empty()) {
		groups = GroupFilesByCount(files, target_task_count);
	}

	for (auto &group : groups) {
		if (group.empty()) {
			continue;
		}
		ScanTaskDescriptor task;
		vector<OpenFileInfo> task_files;
		task_files.reserve(group.size());
		for (const auto &info : group) {
			task_files.push_back(info);
		}
		const auto task_bytes = estimate_bytes_for_group(group);
		task.files = std::move(task_files);
		task.estimated_cardinality = estimate_rows_for_task(group, task_bytes);
		task.estimated_bytes = static_cast<idx_t>(task_bytes);
		task.source_task_partition_id = tasks.size();
		tasks.push_back(std::move(task));
	}

	return {std::move(tasks), false};
}

SchemaRef MakeTableScanSchema(const PhysicalTableScan &scan, const vector<LogicalType> &output_types) {
	if (output_types.empty()) {
		return nullptr;
	}

	std::vector<std::string> scan_names;
	if (!scan.names.empty()) {
		if (scan.names.size() == output_types.size()) {
			scan_names = scan.names;
		} else if (scan.projection_ids.size() == output_types.size()) {
			scan_names.reserve(scan.projection_ids.size());
			for (auto proj_idx : scan.projection_ids) {
				if (proj_idx < scan.column_ids.size()) {
					auto col_idx = scan.column_ids[proj_idx].GetPrimaryIndex();
					if (col_idx < scan.names.size()) {
						scan_names.push_back(scan.names[col_idx]);
					} else {
						scan_names.push_back("c" + std::to_string(scan_names.size()));
					}
				} else {
					scan_names.push_back("c" + std::to_string(scan_names.size()));
				}
			}
		} else {
			scan_names = scan.names;
		}
		if (scan_names.size() < output_types.size()) {
			while (scan_names.size() < output_types.size()) {
				scan_names.push_back("c" + std::to_string(scan_names.size()));
			}
		} else if (scan_names.size() > output_types.size()) {
			scan_names.resize(output_types.size());
		}
	}
	if (!scan_names.empty() && scan_names.size() == output_types.size()) {
		return MakeSchemaRef(output_types, scan_names);
	}
	return MakeSchemaRef(output_types);
}

} // namespace distributed
} // namespace duckdb
