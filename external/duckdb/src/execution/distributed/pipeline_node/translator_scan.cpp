// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/translator_scan.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/enums/file_compression_type.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/multi_file/multi_file_list.hpp"
#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/physical_plan.hpp"
#include "duckdb/execution/operator/csv_scanner/csv_scan_range.hpp"
#include "duckdb/execution/operator/csv_scanner/scanner_boundary.hpp"
#include "duckdb/function/extension_file_list_provider.hpp"
#include "duckdb/function/table/read_csv.hpp"
#include "duckdb/main/database.hpp"

#include <algorithm>
#include <typeinfo>

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

size_t ResolveSingleCSVTaskCount(uint64_t file_size, const DuckDBExecutionConfig &exec_cfg,
                                 const CSVReaderOptions &options) {
	auto target = ResolveScanTaskTargetCount(1, exec_cfg);
	if (target <= 1) {
		return 1;
	}
	if (exec_cfg.scan_task_size_grouping_enabled() && exec_cfg.scan_task_min_partition_num() == 0 &&
	    exec_cfg.scan_task_min_bytes() > 0) {
		const auto size_limited_tasks = MaxValue<uint64_t>(1, file_size / exec_cfg.scan_task_min_bytes());
		target = MinValue<uint64_t>(target, size_limited_tasks);
	}
	const auto minimum_range_size = MaxValue<idx_t>(2, CSVIterator::BytesPerThread(options));
	const auto safe_range_count = MaxValue<uint64_t>(1, file_size / minimum_range_size);
	return static_cast<size_t>(MinValue<uint64_t>(target, safe_range_count));
}

void ValidateSingleCSVSplit(const OpenFileInfo &file, const ReadCSVData &csv_data) {
	if (!csv_data.options.parallel) {
		throw InvalidInputException("Distributed byte-range scanning of CSV file \"%s\" requires parallel=true",
		                            file.path);
	}
	if (csv_data.options.GetSkipRows() > 0) {
		throw InvalidInputException("Distributed byte-range scanning of CSV file \"%s\" does not support skip_rows",
		                            file.path);
	}
	if (StringUtil::Lower(csv_data.options.encoding) != "utf-8") {
		throw InvalidInputException("Distributed byte-range scanning of CSV file \"%s\" requires UTF-8 input",
		                            file.path);
	}
	const auto compression = csv_data.options.compression;
	const bool compressed =
	    compression == FileCompressionType::GZIP || compression == FileCompressionType::ZSTD ||
	    (compression == FileCompressionType::AUTO_DETECT && (IsFileCompressed(file.path, FileCompressionType::GZIP) ||
	                                                         IsFileCompressed(file.path, FileCompressionType::ZSTD)));
	if (compressed) {
		throw InvalidInputException("Distributed byte-range scanning requires an uncompressed CSV file: \"%s\"",
		                            file.path);
	}
}

std::vector<std::vector<idx_t>> GroupIndexesByCount(idx_t count, size_t max_tasks) {
	std::vector<std::vector<idx_t>> groups;
	if (count == 0) {
		return groups;
	}
	size_t tasks = max_tasks == 0 ? static_cast<size_t>(count) : std::min(max_tasks, static_cast<size_t>(count));
	const idx_t per_task = (count + tasks - 1) / tasks;
	for (idx_t start = 0; start < count; start += per_task) {
		const idx_t end = std::min(count, start + per_task);
		std::vector<idx_t> group;
		group.reserve(end - start);
		for (idx_t idx = start; idx < end; ++idx) {
			group.push_back(idx);
		}
		groups.push_back(std::move(group));
	}
	return groups;
}

std::vector<std::vector<idx_t>> GroupWeightedIndexesByTargetCount(const std::vector<uint64_t> &weights,
                                                                  size_t target_groups) {
	std::vector<std::vector<idx_t>> groups;
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

	uint64_t remaining_weight = 0;
	for (auto weight : weights) {
		remaining_weight += weight;
	}

	size_t remaining_groups = target_groups;
	std::vector<idx_t> current;
	current.reserve(static_cast<size_t>(item_count + target_groups - 1) / target_groups);
	uint64_t current_weight = 0;

	for (idx_t i = 0; i < item_count; ++i) {
		const auto weight = weights[static_cast<size_t>(i)];
		current.push_back(i);
		current_weight += weight;
		remaining_weight -= weight;

		if (remaining_groups <= 1) {
			continue;
		}

		const auto remaining_items = static_cast<size_t>(item_count - i - 1);
		const auto target_weight = (current_weight + remaining_weight + remaining_groups - 1) / remaining_groups;
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

uint64_t GetRequiredFileSizeFromDB(const OpenFileInfo &file, const shared_ptr<DatabaseInstance> &db) {
	if (!db) {
		throw InvalidInputException("Distributed byte-range scanning requires database file-system access for \"%s\"",
		                            file.path);
	}
	auto &fs = FileSystem::GetFileSystem(*db);
	auto handle = fs.OpenFile(file, FileOpenFlags::FILE_FLAGS_READ);
	if (!handle) {
		throw IOException("Could not open CSV file \"%s\" while creating byte-range scan tasks", file.path);
	}
	const auto size = fs.GetFileSize(*handle);
	if (size <= 0) {
		throw InvalidInputException(
		    "Distributed byte-range scanning requires a non-empty CSV file with a known size: \"%s\"", file.path);
	}
	return static_cast<uint64_t>(size);
}

} // namespace

DuckPhysicalPlanRef MakeTableScanPlan(const PhysicalTableScan &scan) {
	Allocator &alloc = Allocator::DefaultAllocator();
	auto plan = std::make_shared<PhysicalPlan>(alloc);

	auto bind_data = scan.bind_data ? scan.bind_data->Copy() : nullptr;
	auto table_filters = scan.table_filters ? scan.table_filters->Copy() : nullptr;
	auto extra_info = CopyExtraOperatorInfo(scan.extra_info);

	auto &scan_op = plan->Make<PhysicalTableScan>(scan.GetTypes(), scan.function, std::move(bind_data),
	                                              scan.returned_types, scan.column_ids, scan.projection_ids, scan.names,
	                                              std::move(table_filters), scan.estimated_cardinality,
	                                              std::move(extra_info), scan.parameters, scan.virtual_columns);
	plan->SetRoot(scan_op);
	return plan;
}

std::vector<ScanTaskDescriptor> MakeTableScanTasks(const PhysicalTableScan &scan, const DuckDBExecutionConfig &exec_cfg,
                                                   const shared_ptr<DatabaseInstance> &db) {
	std::vector<ScanTaskDescriptor> tasks;

	if (!scan.bind_data) {
		throw BinderException("MakeTableScanTasks: bind_data is nullptr for function '%s'", scan.function.name);
	}

	vector<OpenFileInfo> files;
	auto *multi_bind = dynamic_cast<MultiFileBindData *>(scan.bind_data.get());
	auto *csv_data =
	    multi_bind && multi_bind->bind_data ? dynamic_cast<ReadCSVData *>(multi_bind->bind_data.get()) : nullptr;
	if (csv_data && csv_data->options.store_rejects.GetValue()) {
		throw InvalidInputException("Distributed CSV scanning does not support store_rejects");
	}
	if (multi_bind && multi_bind->file_list) {
		files = multi_bind->file_list->GetAllFiles();
	} else {
		auto *ext_provider = dynamic_cast<ExtensionFileListProvider *>(scan.bind_data.get());
		if (!ext_provider) {
			throw BinderException("MakeTableScanTasks: bind_data for '%s' is not MultiFileBindData or "
			                      "ExtensionFileListProvider (type: %s)",
			                      scan.function.name, typeid(*scan.bind_data).name());
		}
		for (auto &path : ext_provider->GetFileList()) {
			files.emplace_back(path);
		}
	}
	if (files.empty()) {
		return tasks;
	}
	const idx_t estimated_scan_rows =
	    scan.estimated_cardinality == DConstants::INVALID_INDEX ? 0 : scan.estimated_cardinality;
	std::vector<uint64_t> file_sizes;
	bool file_sizes_loaded = false;
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
		total_file_bytes = 0;
		for (auto size : file_sizes) {
			total_file_bytes += size;
		}
	};
	auto estimate_rows_for_task = [&](uint64_t task_bytes, size_t task_file_count) -> idx_t {
		if (estimated_scan_rows == 0) {
			return 0;
		}
		if (total_file_bytes > 0 && task_bytes > 0) {
			auto scaled = static_cast<long double>(estimated_scan_rows) * static_cast<long double>(task_bytes) /
			              static_cast<long double>(total_file_bytes);
			return static_cast<idx_t>(scaled);
		}
		if (!files.empty() && task_file_count > 0) {
			auto scaled = static_cast<long double>(estimated_scan_rows) * static_cast<long double>(task_file_count) /
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
		const auto requested_task_count = ResolveScanTaskTargetCount(1, exec_cfg);
		if (csv_data && requested_task_count > 1) {
			ValidateSingleCSVSplit(files.front(), *csv_data);
			if (file_sizes.size() != 1 || total_file_bytes == 0) {
				total_file_bytes = GetRequiredFileSizeFromDB(files.front(), db);
				file_sizes = {total_file_bytes};
			}
			const auto task_count = ResolveSingleCSVTaskCount(total_file_bytes, exec_cfg, csv_data->options);
			if (task_count > 1) {
				const auto scan_unit_bytes = MaxValue<idx_t>(2, CSVIterator::BytesPerThread(csv_data->options));
				const auto scan_unit_count = total_file_bytes / scan_unit_bytes;
				const auto units_per_task = scan_unit_count / task_count;
				const auto extra_units = scan_unit_count % task_count;
				const auto rows_per_task = estimated_scan_rows / task_count;
				const auto extra_rows = estimated_scan_rows % task_count;
				uint64_t start = 0;
				for (size_t task_idx = 0; task_idx < task_count; ++task_idx) {
					const auto task_units = units_per_task + (task_idx < extra_units ? 1 : 0);
					const auto end =
					    task_idx + 1 == task_count ? total_file_bytes : start + task_units * scan_unit_bytes;
					const auto task_bytes = end - start;
					ScanTaskDescriptor task;
					task.files.push_back(CSVScanRange::Set(files.front(), start, end));
					task.estimated_cardinality = rows_per_task + (task_idx < extra_rows ? 1 : 0);
					task.estimated_bytes = static_cast<idx_t>(task_bytes);
					task.source_task_partition_id = task_idx;
					tasks.push_back(std::move(task));
					start = end;
				}
				return tasks;
			}
		}
		ScanTaskDescriptor task;
		task.files = std::move(files);
		task.estimated_cardinality = estimated_scan_rows;
		task.estimated_bytes = static_cast<idx_t>(total_file_bytes);
		task.source_task_partition_id = 0;
		tasks.push_back(std::move(task));
		return tasks;
	}

	size_t target_task_count = ResolveScanTaskTargetCount(files.size(), exec_cfg);
	const bool use_size_thresholds = exec_cfg.scan_task_size_grouping_enabled() &&
	                                 (exec_cfg.scan_task_min_bytes() > 0 || exec_cfg.scan_task_max_bytes() > 0);
	std::vector<std::vector<OpenFileInfo>> groups;
	if (use_size_thresholds && db) {
		ensure_file_sizes();
		auto sizes = file_sizes;
		if (!sizes.empty() && HasPositiveSizes(sizes)) {
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
				std::vector<uint64_t> group_weights;
				group_weights.reserve(groups.size());
				for (const auto &group : groups) {
					uint64_t weight = 0;
					for (const auto &file_info : group) {
						for (size_t fi = 0; fi < files.size(); ++fi) {
							if (files[fi].path == file_info.path) {
								weight += sizes[fi];
								break;
							}
						}
					}
					group_weights.push_back(weight);
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
		task.estimated_cardinality = estimate_rows_for_task(task_bytes, group.size());
		task.estimated_bytes = static_cast<idx_t>(task_bytes);
		task.source_task_partition_id = tasks.size();
		tasks.push_back(std::move(task));
	}

	return tasks;
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
