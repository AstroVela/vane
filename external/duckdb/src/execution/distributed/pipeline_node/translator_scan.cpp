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
#include "duckdb/common/set.hpp"
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

idx_t ResolveScanSplitTargetCount(const DuckDBExecutionConfig &exec_cfg) {
	size_t target = 1;
	if (exec_cfg.distributed_worker_slots() > 0) {
		target = exec_cfg.distributed_worker_slots();
	} else if (exec_cfg.distributed_node_count() > 0) {
		target = exec_cfg.distributed_node_count();
	}
	target = std::max(target, exec_cfg.scan_split_min_count());
	return target > NumericLimits<idx_t>::Maximum() ? NumericLimits<idx_t>::Maximum() : static_cast<idx_t>(target);
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
				return {};
			}
			auto size = fs.GetFileSize(*handle);
			if (size < 0) {
				return {};
			}
			sizes.push_back(static_cast<uint64_t>(size));
		} catch (...) {
			return {};
		}
	}
	return sizes;
}

TableFunctionDistributedScanInput MakeDistributedScanInput(const PhysicalTableScan &scan) {
	return TableFunctionDistributedScanInput(*scan.bind_data, scan.column_ids, scan.projection_ids,
	                                         scan.table_filters.get(), scan.estimated_cardinality);
}

vector<ScanSplit> MakeExtensionScanSplits(const PhysicalTableScan &scan, const DuckDBExecutionConfig &exec_cfg,
                                          const shared_ptr<DatabaseInstance> &db) {
	if (!scan.function.HasSerializationCallbacks()) {
		throw SerializationException("Distributed table function '%s' requires complete serialize and deserialize "
		                             "callbacks; worker rebind is not supported",
		                             scan.function.name);
	}
	const auto &callbacks = scan.function.GetDistributedScanCallbacks();
	callbacks.Validate(scan.function);
	const auto &capability = callbacks.GetCapability();
	if (!db) {
		throw InvalidInputException("Distributed extension scan '%s' requires a DatabaseInstance for capability "
		                            "validation",
		                            scan.function.name);
	}
	DistributedExtensionManager::Get(*db).RequireCapability(capability);

	auto scan_input = MakeDistributedScanInput(scan);
	TableFunctionDistributedScanPlanningInput planning_input(scan_input, ResolveScanSplitTargetCount(exec_cfg));
	auto planned_splits = callbacks.plan_splits(planning_input);
	if (planned_splits.empty()) {
		return {ScanSplit::EmptyExtension(capability, callbacks.split_codec)};
	}

	set<string> split_ids;
	vector<ScanSplit> result;
	result.reserve(planned_splits.size());
	for (auto &planned_split : planned_splits) {
		planned_split.Validate();
		if (!split_ids.insert(planned_split.split_id).second) {
			throw InvalidInputException("Distributed extension scan '%s' planned duplicate split_id '%s'",
			                            scan.function.name, planned_split.split_id);
		}
		result.push_back(ScanSplit::Extension(std::move(planned_split.split_id), std::move(planned_split.payload),
		                                      capability, callbacks.split_codec, planned_split.estimated_cardinality,
		                                      planned_split.estimated_bytes));
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
		callbacks.Validate(scan.function);
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
		// The coordinator file list is never a worker fallback. Every worker
		// receives an explicit static split batch or an FTE split queue,
		// including one explicit empty split for a legal zero-file scan.
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

vector<ScanSplit> MakeTableScanSplits(const PhysicalTableScan &scan, const DuckDBExecutionConfig &exec_cfg,
                                      const shared_ptr<DatabaseInstance> &db) {
	if (!scan.bind_data) {
		throw NotImplementedException("Distributed execution does not support table function \"%s\": bind data is "
		                              "missing",
		                              scan.function.name);
	}
	if (scan.function.HasDistributedScanCallbacks()) {
		return MakeExtensionScanSplits(scan, exec_cfg, db);
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
		return {ScanSplit::EmptyFile()};
	}

	auto file_sizes = GetFileSizesFromDB(files, db);
	const bool file_sizes_complete = file_sizes.size() == files.size();
	long double total_file_bytes = 0;
	for (auto size : file_sizes) {
		total_file_bytes += static_cast<long double>(size);
	}
	const bool cardinality_known = scan.estimated_cardinality != DConstants::INVALID_INDEX;

	vector<ScanSplit> splits;
	splits.reserve(files.size());
	for (idx_t index = 0; index < files.size(); index++) {
		optional_idx estimated_bytes;
		if (file_sizes_complete) {
			estimated_bytes = optional_idx(static_cast<idx_t>(file_sizes[index]));
		}
		optional_idx estimated_cardinality;
		if (cardinality_known) {
			idx_t rows;
			if (file_sizes_complete && total_file_bytes > 0) {
				auto scaled = static_cast<long double>(scan.estimated_cardinality) *
				              static_cast<long double>(file_sizes[index]) / total_file_bytes;
				rows = static_cast<idx_t>(scaled);
			} else {
				rows = scan.estimated_cardinality / files.size();
			}
			estimated_cardinality = optional_idx(rows);
		}
		splits.push_back(ScanSplit::File("file-" + std::to_string(index), std::move(files[index]),
		                                 estimated_cardinality, estimated_bytes));
	}
	return splits;
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
