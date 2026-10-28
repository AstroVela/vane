// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/common/environment.hpp"
#include "vane_python/native_resource_graph.hpp"
#include "vane_python/vane_fs.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "task.hpp"
#include "worker.hpp"
#include "worker_manager.hpp"
#include "python_bounded_diagnostics.hpp"
#include "safe_pyobject.hpp"
#include "datasource_function.hpp"

#include "vane_python/pyrelation.hpp"
#include "vane_python/merge_relation.hpp"
#include "vane_python/pyconnection/pyconnection.hpp"
#include "vane_python/python_objects.hpp"
#include "vane_python/python_udf_actor_resources.hpp"
#include "vane_python/physical_plan_traversal.hpp"
#include "vane_python/arrow/arrow_array_stream.hpp"
#include "vane_python/arrow/arrow_export_utils.hpp"
#include "vane_python/pandas/pandas_scan.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"

#include <duckdb/execution/distributed/plan/distributed_physical_plan.hpp>
#include <duckdb/execution/distributed/plan/plan_config.hpp>
#include <duckdb/execution/distributed/plan/runner.hpp>
#include <duckdb/execution/distributed/common_types.hpp>
#include <duckdb/execution/distributed/copy_to_file.hpp>
#include <duckdb/execution/distributed/copy_finalize.hpp>
#include <duckdb/execution/distributed/extension_write_task_provider.hpp>
#include <duckdb/execution/distributed/plan/exchange_sink_instance_task.hpp>
#include <duckdb/execution/distributed/plan/exchange_source_task.hpp>
#include <duckdb/execution/distributed/plan/fte_split_queue.hpp>
#include <duckdb/execution/distributed/plan/scan_split.hpp>
#include <duckdb/execution/distributed/pipeline_node/pipeline_node.hpp>
#include <duckdb/execution/distributed/pipeline_node/sink.hpp>
#include <duckdb/execution/distributed/pipeline_node/streaming_udf_passthrough.hpp>
#include <duckdb/execution/distributed/pipeline_node/table_inout.hpp>
#include <duckdb/execution/distributed/pipeline_node/translator.hpp>
#include <duckdb/execution/distributed/pipeline_node/translator_api.hpp>
#include <duckdb/execution/distributed/pipeline_node/vllm.hpp>
#include <duckdb/execution/distributed/utils/channel.hpp>
#include <duckdb/planner/planner.hpp>
#include <duckdb/common/file_system.hpp>
#include <duckdb/common/limits.hpp>
#include <duckdb/common/local_file_system.hpp>
#include <duckdb/common/map.hpp>
#include <duckdb/common/set.hpp>
#include <duckdb/common/types/uuid.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/optimizer/remove_unused_columns.hpp>

#include <exception>
#include <tuple>
#include <utility>

static inline int DuckdbGetEnvIntMs(const char *name) {
	const duckdb::EnvironmentVariable val_env(name);
	const char *val = val_env.GetValue();
	if (!val || !*val) {
		return 0;
	}
	char *end = nullptr;
	long parsed = std::strtol(val, &end, 10);
	if (end == val || parsed <= 0) {
		return 0;
	}
	if (parsed > std::numeric_limits<int>::max()) {
		return std::numeric_limits<int>::max();
	}
	return static_cast<int>(parsed);
}

#include <duckdb/execution/physical_plan_generator.hpp>
#include <duckdb/execution/executor.hpp>
#include <duckdb/parser/statement/logical_plan_statement.hpp>
#include <duckdb/parser/statement/relation_statement.hpp>
#include <duckdb/common/serializer/memory_stream.hpp>
#include <duckdb/common/serializer/binary_serializer.hpp>
#include <duckdb/common/serializer/binary_deserializer.hpp>
#include <duckdb/main/client_context.hpp>
#include "vane_python/bound_plan.hpp"
#include <duckdb/main/client_data.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/main/database.hpp>
#include <duckdb/main/database_manager.hpp>
#include <duckdb/main/attached_database.hpp>
#include <duckdb/main/distributed_extension_manager.hpp>
#include <duckdb/main/extension_helper.hpp>
#include <duckdb/parser/keyword_helper.hpp>
#include <duckdb/common/types/data_chunk.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/common/vector_size.hpp>
#include <duckdb/common/algorithm.hpp>
#include <duckdb/parallel/interrupt.hpp>
#include <duckdb/parallel/thread_context.hpp>
#include <duckdb/parallel/task_scheduler.hpp>
#include <duckdb/main/prepared_statement_data.hpp>
#include <duckdb/main/relation/data_sink_relation.hpp>
#include <duckdb/execution/operator/helper/physical_materialized_collector.hpp>
#include <duckdb/execution/operator/helper/physical_data_sink.hpp>
#include <duckdb/execution/operator/exchange/physical_remote_exchange_sink.hpp>
#include <duckdb/execution/operator/exchange/physical_remote_exchange_source.hpp>
#include <duckdb/execution/operator/persistent/physical_batch_copy_to_file.hpp>
#include <duckdb/execution/operator/persistent/physical_copy_to_file.hpp>
#include <duckdb/execution/operator/persistent/physical_distributed_extension_write.hpp>
#include <duckdb/execution/operator/scan/physical_column_data_scan.hpp>
#include <duckdb/execution/operator/scan/physical_dummy_scan.hpp>
#include <duckdb/execution/operator/scan/physical_table_scan.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/filter/in_filter.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/execution/operator/projection/physical_tableinout_function.hpp>
#include <duckdb/execution/operator/projection/physical_udf_inout.hpp>
#include <duckdb/function/scalar/udf_functions.hpp>
#include <duckdb/function/table/datasource_scan.hpp>
#include <duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp>
#include <duckdb/execution/operator/helper/physical_result_collector.hpp>
#include <duckdb/execution/operator/projection/physical_projection.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/common/arrow/arrow_converter.hpp>
#include <duckdb/common/arrow/arrow.hpp>
#include <duckdb/common/arrow/arrow_type_extension.hpp>
#include <duckdb/common/string_util.hpp>
#include <duckdb/execution/distributed/exchange/flight_exchange_manager.hpp>
#include <duckdb/execution/distributed/exchange/shuffle_cache_registry.hpp>

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <limits>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;
