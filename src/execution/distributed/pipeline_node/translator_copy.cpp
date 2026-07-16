// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/translator.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/distributed/pipeline_node/copy_finish.hpp"
#include "duckdb/execution/distributed/pipeline_node/sink.hpp"
#include "duckdb/execution/operator/persistent/physical_batch_copy_to_file.hpp"
#include "duckdb/execution/operator/persistent/physical_copy_to_file.hpp"

namespace duckdb {
namespace distributed {

namespace {

DistributedCopySpec BuildCopyToFileSpecForTranslator(const duckdb::PhysicalCopyToFile &op) {
	if (!op.function.copy_to_get_written_statistics) {
		throw NotImplementedException("Distributed COPY requires copy_to_get_written_statistics for format \"%s\"",
		                              op.function.name);
	}
	if (!op.bind_data) {
		throw NotImplementedException("Distributed COPY requires bind_data for format \"%s\"", op.function.name);
	}

	DistributedCopySpec spec;
	spec.type = DistributedCopyType::COPY_TO_FILE;
	spec.function = op.function;
	spec.bind_data = op.bind_data->Copy();
	spec.file_path = op.file_path;
	spec.use_tmp_file = op.use_tmp_file;
	spec.filename_pattern = op.filename_pattern;
	spec.file_extension = op.file_extension;
	spec.overwrite_mode = op.overwrite_mode;
	spec.parallel = op.parallel;
	spec.per_thread_output = op.per_thread_output;
	spec.file_size_bytes = op.file_size_bytes;
	spec.rotate = op.rotate;
	spec.return_type = op.return_type;
	spec.partition_output = op.partition_output;
	spec.write_partition_columns = op.write_partition_columns;
	spec.write_empty_file = op.write_empty_file;
	spec.hive_file_pattern = op.hive_file_pattern;
	spec.partition_columns = op.partition_columns;
	spec.names = op.names;
	spec.expected_types = op.expected_types;
	return spec;
}

DistributedCopySpec BuildBatchCopyToFileSpecForTranslator(const duckdb::PhysicalBatchCopyToFile &op) {
	if (!op.function.copy_to_get_written_statistics) {
		throw NotImplementedException("Distributed COPY requires copy_to_get_written_statistics for format \"%s\"",
		                              op.function.name);
	}
	if (!op.bind_data) {
		throw NotImplementedException("Distributed COPY requires bind_data for format \"%s\"", op.function.name);
	}

	DistributedCopySpec spec;
	spec.type = DistributedCopyType::BATCH_COPY_TO_FILE;
	spec.function = op.function;
	spec.bind_data = op.bind_data->Copy();
	spec.file_path = op.file_path;
	spec.use_tmp_file = op.use_tmp_file;
	spec.return_type = op.return_type;
	spec.write_empty_file = op.write_empty_file;
	if (!op.function.extension.empty()) {
		spec.file_extension = op.function.extension;
	}
	return spec;
}

PipelineNodeRef RequiredCopyChildImpl(const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	if (children.empty() || !children[0]) {
		throw NotImplementedException("Distributed COPY requires a child operator");
	}
	return children[0]->inner();
}

} // namespace

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateCopyToFile(
    const PhysicalCopyToFile &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	auto child_impl = RequiredCopyChildImpl(children);
	auto copy_sink =
	    std::make_shared<CopySinkNode>(get_next_pipeline_node_id(), child_impl, BuildCopyToFileSpecForTranslator(op));
	return std::make_shared<CopyFinishNode>(get_next_pipeline_node_id(), copy_sink);
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateBatchCopyToFile(
    const PhysicalBatchCopyToFile &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	auto child_impl = RequiredCopyChildImpl(children);
	auto copy_sink = std::make_shared<CopySinkNode>(get_next_pipeline_node_id(), child_impl,
	                                                BuildBatchCopyToFileSpecForTranslator(op));
	return std::make_shared<CopyFinishNode>(get_next_pipeline_node_id(), copy_sink);
}

} // namespace distributed
} // namespace duckdb
