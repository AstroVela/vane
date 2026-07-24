// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Included by ray_module.cpp before namespace duckdb.

namespace {

struct DuckDBPyTransactionGuard {
	duckdb::ClientContext &context;
	bool started = false;
	bool finished = false;

	explicit DuckDBPyTransactionGuard(duckdb::ClientContext &ctx) : context(ctx) {
		if (!context.transaction.HasActiveTransaction()) {
			context.transaction.BeginTransaction();
			started = true;
		}
	}

	void Commit() {
		if (started && !finished) {
			context.transaction.Commit();
			finished = true;
		}
	}

	void Rollback() {
		if (started && !finished) {
			try {
				context.transaction.Rollback(nullptr);
			} catch (...) {
			}
			finished = true;
		}
	}

	~DuckDBPyTransactionGuard() {
		if (started && !finished) {
			try {
				context.transaction.Rollback(nullptr);
			} catch (...) {
			}
		}
	}
};

} // namespace

struct NativePartitionMetadata {
	size_t num_rows;
	size_t size_bytes;

	NativePartitionMetadata(size_t num_rows_p = 0, size_t size_bytes_p = 0)
	    : num_rows(num_rows_p), size_bytes(size_bytes_p) {
	}
};

struct NativeDistributedTaskResult {
	duckdb::distributed::python::ray::SafePyObject partition_payloads;
	duckdb::distributed::python::ray::SafePyObject partition_metadatas;
	duckdb::distributed::python::ray::SafePyObject result_schema;
	duckdb::distributed::python::ray::SafePyObject stats;
	duckdb::distributed::python::ray::SafePyObject task_stats;
	std::string completion_status;
	int flight_port;
	duckdb::distributed::python::ray::SafePyObject exchange_sink_instance;

	NativeDistributedTaskResult(pybind11::iterable payloads, pybind11::iterable metadatas,
	                            pybind11::object result_schema_p, pybind11::object stats_p,
	                            std::string completion_status_p, int flight_port_p = 0,
	                            pybind11::object exchange_sink_instance_p = pybind11::none(),
	                            pybind11::object task_stats_p = pybind11::none())
	    : partition_payloads(duckdb::distributed::python::ray::SafePyObject(pybind11::list(payloads))),
	      partition_metadatas(duckdb::distributed::python::ray::SafePyObject(pybind11::list(metadatas))),
	      result_schema(duckdb::distributed::python::ray::SafePyObject(std::move(result_schema_p))),
	      stats(duckdb::distributed::python::ray::SafePyObject(std::move(stats_p))),
	      task_stats(duckdb::distributed::python::ray::SafePyObject(std::move(task_stats_p))),
	      completion_status(std::move(completion_status_p)), flight_port(flight_port_p),
	      exchange_sink_instance(duckdb::distributed::python::ray::SafePyObject(std::move(exchange_sink_instance_p))) {
	}

	pybind11::object PartitionPayloads() const {
		return partition_payloads.get();
	}

	pybind11::object PartitionMetadatas() const {
		return partition_metadatas.get();
	}

	pybind11::object ResultSchema() const {
		return result_schema.get();
	}

	pybind11::object Stats() const {
		return stats.get();
	}

	pybind11::object TaskStats() const {
		return task_stats.get();
	}

	pybind11::object ExchangeSinkInstance() const {
		return exchange_sink_instance.get();
	}
};

static pybind11::dict BuildNativeResultSchema(const duckdb::vector<string> &names,
                                              const duckdb::vector<duckdb::LogicalType> &types) {
	pybind11::list py_names;
	pybind11::list py_types;
	for (auto &name : names) {
		py_names.append(name);
	}
	for (auto &type : types) {
		py_types.append(type.ToString());
	}
	pybind11::dict schema;
	schema["names"] = py_names;
	schema["types"] = py_types;
	return schema;
}

static size_t GetPyPayloadSizeBytes(const pybind11::object &payload) {
	try {
		if (pybind11::hasattr(payload, "nbytes")) {
			return payload.attr("nbytes").cast<size_t>();
		}
	} catch (...) {
	}
	return 0;
}

static pybind11::object BuildNativeTaskResult(pybind11::iterable payloads, pybind11::iterable metadatas,
                                              pybind11::object result_schema, pybind11::object stats,
                                              pybind11::object task_stats, const std::string &completion_status,
                                              int flight_port = 0,
                                              pybind11::object exchange_sink_instance = pybind11::none()) {
	return pybind11::cast(NativeDistributedTaskResult(payloads, metadatas, std::move(result_schema), std::move(stats),
	                                                  completion_status, flight_port, std::move(exchange_sink_instance),
	                                                  std::move(task_stats)));
}

static idx_t SaturatingAddIdx(idx_t lhs, idx_t rhs) {
	if (std::numeric_limits<idx_t>::max() - lhs < rhs) {
		return std::numeric_limits<idx_t>::max();
	}
	return lhs + rhs;
}

static bool PipelineSnapshotSourceIs(const duckdb::PipelineProgressSnapshot &snapshot, const char *needle) {
	return !snapshot.operators.empty() && snapshot.operators.front() == needle;
}

static bool IsCompletionOnlyRemoteExchangeSinkPipeline(const duckdb::PipelineProgressSnapshot &snapshot) {
	// PhysicalRemoteExchangeSink implements IsSource() only to let DuckDB close
	// the sink pipeline. That singleton source pipeline always returns zero rows
	// and is not user-visible work. Keep the child -> EXCHANGE_SINK data pipeline.
	return snapshot.operators.size() == 1 && snapshot.operators.front() == "EXCHANGE_SINK";
}

static std::string PipelineSnapshotName(const duckdb::PipelineProgressSnapshot &snapshot) {
	std::string result;
	for (idx_t i = 0; i < snapshot.operators.size(); i++) {
		if (i > 0) {
			result += "->";
		}
		result += snapshot.operators[i];
	}
	return result.empty() ? "Pipeline" : result;
}

static py::list StageIdsToPyList(const duckdb::vector<duckdb::idx_t> &stage_ids) {
	py::list out;
	for (auto stage_id : stage_ids) {
		out.append(py::int_(stage_id));
	}
	return out;
}

static py::list OperatorsToPyList(const duckdb::vector<std::string> &operators) {
	py::list out;
	for (const auto &op : operators) {
		out.append(py::str(op));
	}
	return out;
}

static py::list
OperatorDetailsToPyList(const duckdb::vector<duckdb::InsertionOrderPreservingMap<std::string>> &operator_details) {
	py::list out;
	for (const auto &details : operator_details) {
		py::dict item;
		for (const auto &entry : details) {
			item[py::str(entry.first)] = py::str(entry.second);
		}
		out.append(std::move(item));
	}
	return out;
}

static std::string OperatorListName(const duckdb::vector<std::string> &operators) {
	std::string result;
	for (idx_t i = 0; i < operators.size(); i++) {
		if (i > 0) {
			result += "->";
		}
		result += operators[i];
	}
	return result.empty() ? "Result" : result;
}

static duckdb::InsertionOrderPreservingMap<std::string> ProgressOperatorDetails(const duckdb::PhysicalOperator &op) {
	duckdb::InsertionOrderPreservingMap<std::string> details;
	if (op.type != duckdb::PhysicalOperatorType::STREAMING_UDF &&
	    op.type != duckdb::PhysicalOperatorType::INOUT_FUNCTION) {
		return details;
	}
	auto params = op.ParamsToString();
	auto udf_name = params.find("udf_name");
	if (udf_name != params.end()) {
		details["udf_name"] = udf_name->second;
	}
	for (const auto &entry : params) {
		if (entry.first.rfind("udf_", 0) == 0) {
			details[entry.first] = entry.second;
		}
	}
	return details;
}

static void CollectPrimaryPipelineOperators(
    const duckdb::PhysicalOperator &op, duckdb::vector<std::string> &operators,
    duckdb::vector<duckdb::InsertionOrderPreservingMap<std::string>> *operator_details = nullptr) {
	auto children = op.GetChildren();
	if (!children.empty()) {
		CollectPrimaryPipelineOperators(children[0].get(), operators, operator_details);
	}
	operators.push_back(duckdb::EnumUtil::ToString(op.type));
	if (operator_details) {
		operator_details->push_back(ProgressOperatorDetails(op));
	}
}

static idx_t OperatorDetailIdxValue(const duckdb::InsertionOrderPreservingMap<std::string> &details,
                                    const std::string &key) {
	auto entry = details.find(key);
	if (entry == details.end() || entry->second.empty()) {
		return 0;
	}
	try {
		auto value = std::stoull(entry->second);
		if (value > static_cast<unsigned long long>(std::numeric_limits<idx_t>::max())) {
			return std::numeric_limits<idx_t>::max();
		}
		return static_cast<idx_t>(value);
	} catch (...) {
		return 0;
	}
}

static idx_t SnapshotUDFCounterMax(const duckdb::PipelineProgressSnapshot &snapshot, const std::string &key) {
	idx_t value = 0;
	for (const auto &details : snapshot.operator_details) {
		value = duckdb::MaxValue<idx_t>(value, OperatorDetailIdxValue(details, key));
	}
	return value;
}

static idx_t
OperatorDetailsCounterMax(const duckdb::vector<duckdb::InsertionOrderPreservingMap<std::string>> &operator_details,
                          const std::string &key) {
	idx_t value = 0;
	for (const auto &details : operator_details) {
		value = duckdb::MaxValue<idx_t>(value, OperatorDetailIdxValue(details, key));
	}
	return value;
}

static idx_t
ScanTaskInputRows(const std::unordered_map<idx_t, duckdb::distributed::ScanTaskDescriptor> *scan_task_map) {
	idx_t rows = 0;
	if (!scan_task_map) {
		return rows;
	}
	for (const auto &entry : *scan_task_map) {
		rows = SaturatingAddIdx(rows, entry.second.estimated_cardinality);
	}
	return rows;
}

static idx_t
ScanTaskInputBytes(const std::unordered_map<idx_t, duckdb::distributed::ScanTaskDescriptor> *scan_task_map) {
	idx_t bytes = 0;
	if (!scan_task_map) {
		return bytes;
	}
	for (const auto &entry : *scan_task_map) {
		bytes = SaturatingAddIdx(bytes, entry.second.estimated_bytes);
	}
	return bytes;
}

static idx_t ExchangeSourceInputBytes(
    const std::unordered_map<idx_t, duckdb::distributed::ExchangeSourceTaskDescriptor> *exchange_source_task_map) {
	idx_t bytes = 0;
	if (!exchange_source_task_map) {
		return bytes;
	}
	for (const auto &entry : *exchange_source_task_map) {
		for (const auto &handle : entry.second.source_handles) {
			for (const auto &file : handle.files) {
				bytes = SaturatingAddIdx(bytes, static_cast<idx_t>(file.file_size));
			}
		}
	}
	return bytes;
}

static idx_t ExchangeSourceInputRows(
    const std::unordered_map<idx_t, duckdb::distributed::ExchangeSourceTaskDescriptor> *exchange_source_task_map) {
	idx_t rows = 0;
	if (!exchange_source_task_map) {
		return rows;
	}
	for (const auto &entry : *exchange_source_task_map) {
		for (const auto &handle : entry.second.source_handles) {
			for (const auto &file : handle.files) {
				rows = SaturatingAddIdx(rows, file.rows);
			}
		}
	}
	return rows;
}

static idx_t
FteQueueConsumedRows(const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>> *queue_map) {
	idx_t rows = 0;
	if (!queue_map) {
		return rows;
	}
	for (const auto &entry : *queue_map) {
		if (entry.second) {
			rows = SaturatingAddIdx(rows, entry.second->ConsumedRows());
		}
	}
	return rows;
}

static idx_t
FteQueueConsumedBytes(const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>> *queue_map) {
	idx_t bytes = 0;
	if (!queue_map) {
		return bytes;
	}
	for (const auto &entry : *queue_map) {
		if (entry.second) {
			bytes = SaturatingAddIdx(bytes, entry.second->ConsumedInputBytes());
		}
	}
	return bytes;
}

struct FteQueueProgressStats {
	idx_t submitted_split_count = 0;
	idx_t consumed_split_count = 0;
	idx_t completed_split_count = 0;
	idx_t submitted_split_bytes = 0;
	idx_t consumed_split_bytes = 0;
	idx_t completed_split_bytes = 0;
	idx_t queue_wait_ms = 0;
	py::dict submitted_split_count_by_source;
	py::dict consumed_split_count_by_source;
	py::dict completed_split_count_by_source;
	py::dict submitted_split_bytes_by_source;
	py::dict consumed_split_bytes_by_source;
	py::dict completed_split_bytes_by_source;
	py::dict queue_wait_ms_by_source;
};

static void AddFteQueueProgressStats(
    FteQueueProgressStats &stats,
    const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>> *queue_map) {
	if (!queue_map) {
		return;
	}
	for (const auto &entry : *queue_map) {
		if (!entry.second) {
			continue;
		}
		const auto source_id = py::str(std::to_string(entry.first));
		const auto submitted_count = entry.second->SubmittedSplits();
		const auto consumed_count = entry.second->ConsumedSplits();
		const auto completed_count = entry.second->CompletedSplits();
		const auto submitted_bytes = entry.second->SubmittedInputBytes();
		const auto consumed_bytes = entry.second->ConsumedInputBytes();
		const auto completed_bytes = entry.second->CompletedInputBytes();
		const auto wait_ms = entry.second->QueueWaitMillis();
		stats.submitted_split_count = SaturatingAddIdx(stats.submitted_split_count, submitted_count);
		stats.consumed_split_count = SaturatingAddIdx(stats.consumed_split_count, consumed_count);
		stats.completed_split_count = SaturatingAddIdx(stats.completed_split_count, completed_count);
		stats.submitted_split_bytes = SaturatingAddIdx(stats.submitted_split_bytes, submitted_bytes);
		stats.consumed_split_bytes = SaturatingAddIdx(stats.consumed_split_bytes, consumed_bytes);
		stats.completed_split_bytes = SaturatingAddIdx(stats.completed_split_bytes, completed_bytes);
		stats.queue_wait_ms = SaturatingAddIdx(stats.queue_wait_ms, wait_ms);
		stats.submitted_split_count_by_source[source_id] = py::int_(submitted_count);
		stats.consumed_split_count_by_source[source_id] = py::int_(consumed_count);
		stats.completed_split_count_by_source[source_id] = py::int_(completed_count);
		stats.submitted_split_bytes_by_source[source_id] = py::int_(submitted_bytes);
		stats.consumed_split_bytes_by_source[source_id] = py::int_(consumed_bytes);
		stats.completed_split_bytes_by_source[source_id] = py::int_(completed_bytes);
		stats.queue_wait_ms_by_source[source_id] = py::int_(wait_ms);
	}
}

static void AppendFteQueueProgressStats(
    py::dict &out, const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>> *scan_queue_map,
    const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>> *exchange_queue_map) {
	FteQueueProgressStats stats;
	AddFteQueueProgressStats(stats, scan_queue_map);
	AddFteQueueProgressStats(stats, exchange_queue_map);
	out["submitted_split_count"] = py::int_(stats.submitted_split_count);
	out["consumed_split_count"] = py::int_(stats.consumed_split_count);
	out["completed_split_count"] = py::int_(stats.completed_split_count);
	out["submitted_split_count_by_source"] = std::move(stats.submitted_split_count_by_source);
	out["consumed_split_count_by_source"] = std::move(stats.consumed_split_count_by_source);
	out["completed_split_count_by_source"] = std::move(stats.completed_split_count_by_source);
	out["submitted_split_bytes"] = py::int_(stats.submitted_split_bytes);
	out["consumed_split_bytes"] = py::int_(stats.consumed_split_bytes);
	out["completed_split_bytes"] = py::int_(stats.completed_split_bytes);
	out["submitted_split_bytes_by_source"] = std::move(stats.submitted_split_bytes_by_source);
	out["consumed_split_bytes_by_source"] = std::move(stats.consumed_split_bytes_by_source);
	out["completed_split_bytes_by_source"] = std::move(stats.completed_split_bytes_by_source);
	out["queue_wait_ms"] = py::int_(stats.queue_wait_ms);
	out["queue_wait_ms_by_source"] = std::move(stats.queue_wait_ms_by_source);
}

static py::dict BuildNativeTaskStatsDict(
    const duckdb::vector<duckdb::PipelineProgressSnapshot> &snapshots,
    const std::unordered_map<idx_t, duckdb::distributed::ScanTaskDescriptor> *scan_task_map,
    const std::unordered_map<idx_t, duckdb::distributed::ExchangeSourceTaskDescriptor> *exchange_source_task_map,
    const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>> *fte_scan_source_queue_map,
    const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>>
        *fte_exchange_source_queue_map) {
	idx_t scan_rows =
	    SaturatingAddIdx(ScanTaskInputRows(scan_task_map), FteQueueConsumedRows(fte_scan_source_queue_map));
	idx_t scan_bytes =
	    SaturatingAddIdx(ScanTaskInputBytes(scan_task_map), FteQueueConsumedBytes(fte_scan_source_queue_map));
	idx_t exchange_rows = SaturatingAddIdx(ExchangeSourceInputRows(exchange_source_task_map),
	                                       FteQueueConsumedRows(fte_exchange_source_queue_map));
	idx_t exchange_bytes = SaturatingAddIdx(ExchangeSourceInputBytes(exchange_source_task_map),
	                                        FteQueueConsumedBytes(fte_exchange_source_queue_map));
	idx_t memory_rows = 0;
	idx_t memory_bytes = 0;

	py::list pipelines;
	idx_t processed_rows = 0;
	idx_t processed_bytes = 0;
	idx_t physical_input_rows = 0;
	idx_t physical_input_bytes = 0;
	idx_t internal_network_input_rows = 0;
	idx_t internal_network_input_bytes = 0;
	idx_t memory_input_rows = 0;
	idx_t memory_input_bytes = 0;
	idx_t udf_completed_rows = 0;
	idx_t udf_completed_bytes = 0;
	idx_t udf_emitted_rows = 0;
	idx_t udf_emitted_bytes = 0;
	idx_t udf_running_task_count = 0;
	idx_t udf_queued_task_count = 0;
	idx_t udf_max_running_tasks = 0;
	idx_t total_pipeline_tasks = 0;
	idx_t queued_pipeline_tasks = 0;
	idx_t running_pipeline_tasks = 0;
	idx_t completed_pipeline_tasks = 0;
	for (const auto &snapshot : snapshots) {
		if (IsCompletionOnlyRemoteExchangeSinkPipeline(snapshot)) {
			continue;
		}
		const auto pipeline_rows = snapshot.input_rows;
		const auto pipeline_bytes = snapshot.input_bytes;
		idx_t physical_rows = 0;
		idx_t physical_bytes = 0;
		idx_t network_rows = 0;
		idx_t network_bytes = 0;

		if (PipelineSnapshotSourceIs(snapshot, "TABLE_SCAN")) {
			physical_rows = snapshot.input_rows;
			physical_bytes = snapshot.input_bytes;
			physical_input_rows = SaturatingAddIdx(physical_input_rows, physical_rows);
			physical_input_bytes = SaturatingAddIdx(physical_input_bytes, physical_bytes);
		}
		if (PipelineSnapshotSourceIs(snapshot, "EXCHANGE_SOURCE")) {
			network_rows = snapshot.input_rows;
			network_bytes = snapshot.input_bytes;
			internal_network_input_rows = SaturatingAddIdx(internal_network_input_rows, network_rows);
			internal_network_input_bytes = SaturatingAddIdx(internal_network_input_bytes, network_bytes);
		}
		if (PipelineSnapshotSourceIs(snapshot, "COLUMN_DATA_SCAN")) {
			memory_input_rows = SaturatingAddIdx(memory_input_rows, snapshot.input_rows);
			memory_input_bytes = SaturatingAddIdx(memory_input_bytes, snapshot.input_bytes);
		}
		auto snapshot_udf_completed_rows = SnapshotUDFCounterMax(snapshot, "udf_completed_input_rows");
		auto snapshot_udf_completed_bytes = SnapshotUDFCounterMax(snapshot, "udf_completed_input_bytes");
		auto snapshot_udf_emitted_rows = SnapshotUDFCounterMax(snapshot, "udf_emitted_output_rows");
		auto snapshot_udf_emitted_bytes = SnapshotUDFCounterMax(snapshot, "udf_emitted_output_bytes");
		udf_running_task_count =
		    duckdb::MaxValue<idx_t>(udf_running_task_count, SnapshotUDFCounterMax(snapshot, "udf_running_task_count"));
		udf_queued_task_count =
		    duckdb::MaxValue<idx_t>(udf_queued_task_count, SnapshotUDFCounterMax(snapshot, "udf_queued_task_count"));
		udf_max_running_tasks =
		    duckdb::MaxValue<idx_t>(udf_max_running_tasks, SnapshotUDFCounterMax(snapshot, "udf_max_running_tasks"));
		udf_completed_rows = duckdb::MaxValue<idx_t>(udf_completed_rows, snapshot_udf_completed_rows);
		udf_completed_bytes = duckdb::MaxValue<idx_t>(udf_completed_bytes, snapshot_udf_completed_bytes);
		udf_emitted_rows = duckdb::MaxValue<idx_t>(udf_emitted_rows, snapshot_udf_emitted_rows);
		udf_emitted_bytes = duckdb::MaxValue<idx_t>(udf_emitted_bytes, snapshot_udf_emitted_bytes);
		processed_rows = duckdb::MaxValue<idx_t>(processed_rows, pipeline_rows);
		processed_bytes = duckdb::MaxValue<idx_t>(processed_bytes, pipeline_bytes);
		total_pipeline_tasks = SaturatingAddIdx(total_pipeline_tasks, snapshot.total_pipeline_tasks);
		queued_pipeline_tasks = SaturatingAddIdx(queued_pipeline_tasks, snapshot.queued_pipeline_tasks);
		running_pipeline_tasks = SaturatingAddIdx(running_pipeline_tasks, snapshot.running_pipeline_tasks);
		completed_pipeline_tasks = SaturatingAddIdx(completed_pipeline_tasks, snapshot.completed_pipeline_tasks);

		py::dict pipeline;
		pipeline["pipeline_id"] = py::int_(snapshot.pipeline_index);
		pipeline["name"] = PipelineSnapshotName(snapshot);
		pipeline["operators"] = OperatorsToPyList(snapshot.operators);
		pipeline["operator_details"] = OperatorDetailsToPyList(snapshot.operator_details);
		pipeline["stage_ids"] = StageIdsToPyList(snapshot.stage_ids);
		pipeline["input_rows"] = py::int_(snapshot.input_rows);
		pipeline["input_bytes"] = py::int_(snapshot.input_bytes);
		pipeline["output_rows"] = py::int_(snapshot.output_rows);
		pipeline["output_bytes"] = py::int_(snapshot.output_bytes);
		pipeline["total_pipeline_tasks"] = py::int_(snapshot.total_pipeline_tasks);
		pipeline["queued_pipeline_tasks"] = py::int_(snapshot.queued_pipeline_tasks);
		pipeline["running_pipeline_tasks"] = py::int_(snapshot.running_pipeline_tasks);
		pipeline["completed_pipeline_tasks"] = py::int_(snapshot.completed_pipeline_tasks);
		pipelines.append(std::move(pipeline));
	}

	py::dict out;
	out["schema"] = "task_stats";
	out["processed_input_rows"] = py::int_(processed_rows);
	out["processed_input_bytes"] = py::int_(processed_bytes);
	out["processed_rows"] = py::int_(processed_rows);
	out["processed_bytes"] = py::int_(processed_bytes);
	out["physical_input_rows"] = py::int_(physical_input_rows);
	out["physical_input_bytes"] = py::int_(physical_input_bytes);
	out["internal_network_input_rows"] = py::int_(internal_network_input_rows);
	out["internal_network_input_bytes"] = py::int_(internal_network_input_bytes);
	out["memory_input_rows"] = py::int_(memory_input_rows);
	out["memory_input_bytes"] = py::int_(memory_input_bytes);
	out["estimated_physical_input_rows"] = py::int_(scan_rows);
	out["estimated_physical_input_bytes"] = py::int_(scan_bytes);
	out["estimated_internal_network_input_rows"] = py::int_(exchange_rows);
	out["estimated_internal_network_input_bytes"] = py::int_(exchange_bytes);
	out["estimated_memory_input_rows"] = py::int_(memory_rows);
	out["estimated_memory_input_bytes"] = py::int_(memory_bytes);
	out["total_pipeline_tasks"] = py::int_(total_pipeline_tasks);
	out["queued_pipeline_tasks"] = py::int_(queued_pipeline_tasks);
	out["running_pipeline_tasks"] = py::int_(running_pipeline_tasks);
	out["completed_pipeline_tasks"] = py::int_(completed_pipeline_tasks);
	out["udf_completed_rows"] = py::int_(udf_completed_rows);
	out["udf_completed_bytes"] = py::int_(udf_completed_bytes);
	out["udf_emitted_rows"] = py::int_(udf_emitted_rows);
	out["udf_emitted_bytes"] = py::int_(udf_emitted_bytes);
	out["udf_running_task_count"] = py::int_(udf_running_task_count);
	out["udf_queued_task_count"] = py::int_(udf_queued_task_count);
	out["udf_max_running_tasks"] = py::int_(udf_max_running_tasks);
	AppendFteQueueProgressStats(out, fte_scan_source_queue_map, fte_exchange_source_queue_map);
	out["pipelines"] = std::move(pipelines);
	return out;
}

static py::dict BuildOutputOnlyTaskStatsDict(
    idx_t rows, idx_t bytes, const duckdb::vector<std::string> *operators = nullptr,
    const duckdb::vector<duckdb::InsertionOrderPreservingMap<std::string>> *operator_details = nullptr,
    idx_t physical_rows = 0, idx_t physical_bytes = 0, idx_t network_rows = 0, idx_t network_bytes = 0,
    idx_t memory_rows = 0, idx_t memory_bytes = 0) {
	py::dict out;
	out["schema"] = "task_stats";
	out["processed_input_rows"] = py::int_(rows);
	out["processed_input_bytes"] = py::int_(bytes);
	out["processed_rows"] = py::int_(rows);
	out["processed_bytes"] = py::int_(bytes);
	out["physical_input_rows"] = py::int_(physical_rows);
	out["physical_input_bytes"] = py::int_(physical_bytes);
	out["internal_network_input_rows"] = py::int_(network_rows);
	out["internal_network_input_bytes"] = py::int_(network_bytes);
	out["memory_input_rows"] = py::int_(memory_rows);
	out["memory_input_bytes"] = py::int_(memory_bytes);
	out["total_pipeline_tasks"] = py::int_(1);
	out["queued_pipeline_tasks"] = py::int_(0);
	out["running_pipeline_tasks"] = py::int_(1);
	out["completed_pipeline_tasks"] = py::int_(0);
	if (operator_details) {
		out["udf_completed_rows"] = py::int_(OperatorDetailsCounterMax(*operator_details, "udf_completed_input_rows"));
		out["udf_completed_bytes"] =
		    py::int_(OperatorDetailsCounterMax(*operator_details, "udf_completed_input_bytes"));
		out["udf_emitted_rows"] = py::int_(OperatorDetailsCounterMax(*operator_details, "udf_emitted_output_rows"));
		out["udf_emitted_bytes"] = py::int_(OperatorDetailsCounterMax(*operator_details, "udf_emitted_output_bytes"));
		out["udf_running_task_count"] =
		    py::int_(OperatorDetailsCounterMax(*operator_details, "udf_running_task_count"));
		out["udf_queued_task_count"] = py::int_(OperatorDetailsCounterMax(*operator_details, "udf_queued_task_count"));
		out["udf_max_running_tasks"] = py::int_(OperatorDetailsCounterMax(*operator_details, "udf_max_running_tasks"));
	}
	py::list pipelines;
	py::dict pipeline;
	pipeline["pipeline_id"] = py::int_(1);
	pipeline["name"] = operators ? OperatorListName(*operators) : "Result";
	pipeline["operators"] = operators ? OperatorsToPyList(*operators) : py::list();
	pipeline["operator_details"] = operator_details ? OperatorDetailsToPyList(*operator_details) : py::list();
	pipeline["stage_ids"] = py::list();
	pipeline["input_rows"] = py::int_(rows);
	pipeline["input_bytes"] = py::int_(bytes);
	pipeline["output_rows"] = py::int_(rows);
	pipeline["output_bytes"] = py::int_(bytes);
	pipeline["total_pipeline_tasks"] = py::int_(1);
	pipeline["queued_pipeline_tasks"] = py::int_(0);
	pipeline["running_pipeline_tasks"] = py::int_(1);
	pipeline["completed_pipeline_tasks"] = py::int_(0);
	pipelines.append(std::move(pipeline));
	out["pipelines"] = std::move(pipelines);
	return out;
}

static bool TryValueToIdx(const duckdb::Value &value, idx_t &out) {
	if (value.IsNull()) {
		return false;
	}
	duckdb::Value cast_value;
	string error;
	if (!value.DefaultTryCastAs(duckdb::LogicalType::UBIGINT, cast_value, &error, false)) {
		return false;
	}
	const auto raw = cast_value.GetValue<uint64_t>();
	if (raw > static_cast<uint64_t>(std::numeric_limits<idx_t>::max())) {
		out = std::numeric_limits<idx_t>::max();
	} else {
		out = static_cast<idx_t>(raw);
	}
	return true;
}

static bool ExtractCopyResultProgressStats(duckdb::MaterializedQueryResult &result, idx_t fallback_bytes, idx_t &rows,
                                           idx_t &bytes) {
	const auto row_count = result.RowCount();
	if (row_count == 0 || result.types.empty()) {
		rows = 0;
		bytes = 0;
		return true;
	}

	// COPY ... RETURN_STATS emits one row per written file:
	// file_path, row_count, file_size_bytes, ...
	if (result.types.size() >= 3 && result.types[0].id() == duckdb::LogicalTypeId::VARCHAR) {
		idx_t copied_rows = 0;
		idx_t written_bytes = 0;
		bool has_rows = false;
		bool has_bytes = false;
		for (idx_t row = 0; row < row_count; row++) {
			idx_t value = 0;
			if (TryValueToIdx(result.GetValue(1, row), value)) {
				copied_rows = SaturatingAddIdx(copied_rows, value);
				has_rows = true;
			}
			if (TryValueToIdx(result.GetValue(2, row), value)) {
				written_bytes = SaturatingAddIdx(written_bytes, value);
				has_bytes = true;
			}
		}
		if (has_rows) {
			rows = copied_rows;
			bytes = has_bytes ? written_bytes : fallback_bytes;
			return true;
		}
	}

	// COPY ... RETURN_CHANGED_ROWS and RETURN_CHANGED_ROWS_AND_FILE_LIST emit
	// one summary row whose first column is rows_copied.
	idx_t changed_rows = 0;
	if (TryValueToIdx(result.GetValue(0, 0), changed_rows)) {
		rows = changed_rows;
		bytes = fallback_bytes;
		return true;
	}
	return false;
}

static py::dict BuildMaterializedInputTaskStats(
    const duckdb::PhysicalOperator &root_op,
    const std::unordered_map<idx_t, duckdb::distributed::ScanTaskDescriptor> *scan_task_map,
    const std::unordered_map<idx_t, duckdb::distributed::ExchangeSourceTaskDescriptor> *exchange_source_task_map,
    const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>> *fte_scan_source_queue_map,
    const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>> *fte_exchange_source_queue_map,
    idx_t fallback_rows, idx_t fallback_bytes, duckdb::MaterializedQueryResult *copy_result = nullptr) {
	duckdb::vector<std::string> progress_operators;
	duckdb::vector<duckdb::InsertionOrderPreservingMap<std::string>> progress_operator_details;
	CollectPrimaryPipelineOperators(root_op, progress_operators, &progress_operator_details);

	idx_t physical_rows =
	    SaturatingAddIdx(ScanTaskInputRows(scan_task_map), FteQueueConsumedRows(fte_scan_source_queue_map));
	idx_t physical_bytes =
	    SaturatingAddIdx(ScanTaskInputBytes(scan_task_map), FteQueueConsumedBytes(fte_scan_source_queue_map));
	idx_t network_rows = SaturatingAddIdx(ExchangeSourceInputRows(exchange_source_task_map),
	                                      FteQueueConsumedRows(fte_exchange_source_queue_map));
	idx_t network_bytes = SaturatingAddIdx(ExchangeSourceInputBytes(exchange_source_task_map),
	                                       FteQueueConsumedBytes(fte_exchange_source_queue_map));
	idx_t memory_rows = 0;
	idx_t memory_bytes = 0;
	idx_t progress_rows = duckdb::MaxValue<idx_t>(physical_rows, duckdb::MaxValue<idx_t>(network_rows, memory_rows));
	idx_t progress_bytes =
	    duckdb::MaxValue<idx_t>(physical_bytes, duckdb::MaxValue<idx_t>(network_bytes, memory_bytes));
	if (progress_rows == 0 && progress_bytes == 0) {
		progress_rows = fallback_rows;
		progress_bytes = fallback_bytes;
	}
	if (copy_result && (root_op.type == duckdb::PhysicalOperatorType::COPY_TO_FILE ||
	                    root_op.type == duckdb::PhysicalOperatorType::BATCH_COPY_TO_FILE)) {
		idx_t copy_rows = 0;
		idx_t copy_bytes = 0;
		if (ExtractCopyResultProgressStats(*copy_result, fallback_bytes, copy_rows, copy_bytes)) {
			progress_rows = copy_rows;
			if (progress_bytes == 0) {
				progress_bytes = copy_bytes;
			}
		}
	}

	auto out = BuildOutputOnlyTaskStatsDict(progress_rows, progress_bytes, &progress_operators,
	                                        &progress_operator_details, physical_rows, physical_bytes, network_rows,
	                                        network_bytes, memory_rows, memory_bytes);
	AppendFteQueueProgressStats(out, fte_scan_source_queue_map, fte_exchange_source_queue_map);
	return out;
}

static void MarkTaskStatsCompleted(py::dict &stats) {
	idx_t total_pipeline_tasks = 1;
	if (stats.contains("total_pipeline_tasks")) {
		total_pipeline_tasks = duckdb::MaxValue<idx_t>(1, stats["total_pipeline_tasks"].cast<idx_t>());
	}
	stats["total_pipeline_tasks"] = py::int_(total_pipeline_tasks);
	stats["queued_pipeline_tasks"] = py::int_(0);
	stats["running_pipeline_tasks"] = py::int_(0);
	stats["completed_pipeline_tasks"] = py::int_(total_pipeline_tasks);
	if (stats.contains("pipelines")) {
		for (auto item : stats["pipelines"].cast<py::list>()) {
			auto pipeline = py::reinterpret_borrow<py::dict>(item);
			idx_t pipeline_total = 1;
			if (pipeline.contains("total_pipeline_tasks")) {
				pipeline_total = duckdb::MaxValue<idx_t>(1, pipeline["total_pipeline_tasks"].cast<idx_t>());
			}
			pipeline["total_pipeline_tasks"] = py::int_(pipeline_total);
			pipeline["queued_pipeline_tasks"] = py::int_(0);
			pipeline["running_pipeline_tasks"] = py::int_(0);
			pipeline["completed_pipeline_tasks"] = py::int_(pipeline_total);
		}
	}
}

static bool NativePipelineMatchesMaterializedOperators(const py::handle &native_obj,
                                                       const py::handle &materialized_obj) {
	if (!py::isinstance<py::list>(native_obj) || !py::isinstance<py::list>(materialized_obj)) {
		return false;
	}
	auto native = py::reinterpret_borrow<py::list>(native_obj);
	auto materialized = py::reinterpret_borrow<py::list>(materialized_obj);
	const bool has_appended_result_collector =
	    native.size() == materialized.size() + 1 &&
	    py::str(native[native.size() - 1]).cast<std::string>() == "RESULT_COLLECTOR";
	if (native.size() != materialized.size() && !has_appended_result_collector) {
		return false;
	}
	for (py::ssize_t index = 0; index < materialized.size(); index++) {
		if (py::str(native[index]).cast<std::string>() != py::str(materialized[index]).cast<std::string>()) {
			return false;
		}
	}
	return true;
}

static void OverlayMaterializedProgressStats(py::dict &native_stats, const py::dict &materialized_stats) {
	// Materialized stats remain authoritative for source rows/bytes and queue
	// counters. Native pipeline identity and task lifecycle remain authoritative
	// in native_stats, so fast tasks cannot collapse back to one synthetic row.
	for (auto item : materialized_stats) {
		auto key = py::str(item.first).cast<std::string>();
		if (key == "schema" || key == "pipelines" || key == "total_pipeline_tasks" || key == "queued_pipeline_tasks" ||
		    key == "running_pipeline_tasks" || key == "completed_pipeline_tasks") {
			continue;
		}
		native_stats[item.first] = item.second;
	}

	if (!native_stats.contains("pipelines") || !materialized_stats.contains("pipelines")) {
		return;
	}
	auto materialized_pipelines = materialized_stats["pipelines"].cast<py::list>();
	if (materialized_pipelines.size() == 0) {
		return;
	}
	auto materialized_pipeline = py::reinterpret_borrow<py::dict>(materialized_pipelines[0]);
	if (!materialized_pipeline.contains("operators")) {
		return;
	}
	for (auto item : native_stats["pipelines"].cast<py::list>()) {
		auto native_pipeline = py::reinterpret_borrow<py::dict>(item);
		if (!native_pipeline.contains("operators") ||
		    !NativePipelineMatchesMaterializedOperators(native_pipeline["operators"],
		                                                materialized_pipeline["operators"])) {
			continue;
		}
		for (auto key : {"input_rows", "input_bytes", "output_rows", "output_bytes"}) {
			if (materialized_pipeline.contains(key)) {
				native_pipeline[py::str(key)] = materialized_pipeline[py::str(key)];
			}
		}
		break;
	}
}

static py::dict BuildNativePipelineTopology(const duckdb::vector<duckdb::PipelineProgressSnapshot> &snapshots) {
	py::list pipelines;
	for (const auto &snapshot : snapshots) {
		if (IsCompletionOnlyRemoteExchangeSinkPipeline(snapshot)) {
			continue;
		}
		py::dict pipeline;
		pipeline["pipeline_id"] = py::int_(snapshot.pipeline_index);
		pipeline["operators"] = OperatorsToPyList(snapshot.operators);
		pipeline["operator_details"] = OperatorDetailsToPyList(snapshot.operator_details);
		pipeline["stage_ids"] = StageIdsToPyList(snapshot.stage_ids);
		pipelines.append(std::move(pipeline));
	}
	py::dict topology;
	topology["schema"] = "pipeline_topology";
	topology["pipelines"] = std::move(pipelines);
	return topology;
}

static bool NativePlanNeedsResultCollector(const duckdb::PhysicalOperator &root_op) {
	using duckdb::PhysicalOperatorType;

	// A remote exchange sink exposes a source interface only so DuckDB can mark
	// the sink pipeline complete. Its source side never produces rows; the real
	// output has already been sent through Arrow Flight. Materializing that
	// zero-row source adds two meaningless RESULT_COLLECTOR pipelines.
	if (root_op.type == PhysicalOperatorType::EXCHANGE_SINK) {
		return false;
	}
	return !root_op.IsSink() || root_op.IsSource() || root_op.type == PhysicalOperatorType::CTE ||
	       root_op.type == PhysicalOperatorType::RECURSIVE_CTE;
}

static py::dict BuildNativeProgressTopology(duckdb::ClientContext &context,
                                            const std::shared_ptr<duckdb::PhysicalPlan> &physical_plan) {
	using namespace duckdb;
	if (!physical_plan || !physical_plan->HasRoot()) {
		throw pybind11::value_error("Physical plan is missing or has no root operator");
	}

	auto &root_op = physical_plan->Root();
	const bool needs_result_collector = NativePlanNeedsResultCollector(root_op);

	auto build_topology = [&](PhysicalOperator &topology_root) {
		Executor executor(context);
		executor.InitializeProgressTopology(topology_root);
		return BuildNativePipelineTopology(executor.GetPipelinesProgressSnapshots());
	};

	if (!needs_result_collector) {
		return build_topology(root_op);
	}

	auto prepared_data = make_shared_ptr<PreparedStatementData>(StatementType::SELECT_STATEMENT);
	prepared_data->types = root_op.types;
	for (idx_t i = 0; i < root_op.types.size(); i++) {
		prepared_data->names.push_back("col_" + std::to_string(i));
	}
	prepared_data->properties.return_type = StatementReturnType::QUERY_RESULT;
	prepared_data->output_type = QueryResultOutputType::FORCE_MATERIALIZED;
	prepared_data->memory_type = QueryResultMemoryType::IN_MEMORY;
	prepared_data->physical_plan = unique_ptr<PhysicalPlan>(physical_plan.get());
	struct PhysicalPlanReleaseGuard {
		unique_ptr<PhysicalPlan> &plan;
		~PhysicalPlanReleaseGuard() {
			plan.release();
		}
	} plan_guard {prepared_data->physical_plan};

	auto &collector = physical_plan->Make<PhysicalMaterializedCollector>(*prepared_data, true);
	return build_topology(collector);
}

static void AppendDistributedCopyResultMetadata(pybind11::dict &out,
                                                const duckdb::distributed::DistributedCopyResult &result) {
	out["copy_output_base_path"] = result.output_base_path;
	out["copy_output_run_id"] = result.output_run_id;
	out["copy_output_commit_dir"] = result.output_commit_dir;
	out["copy_output_manifest_path"] = result.output_manifest_path;
	out["copy_output_committed_marker_path"] = result.output_committed_marker_path;
	out["copy_output_lifecycle_path"] = result.output_lifecycle_path;
	out["copy_output_direct_write"] = result.output_direct_write;
	out["copy_output_committed"] = result.output_committed;
	out["copy_staging_write_ms"] = result.staging_write_ms;
	out["copy_finalize_ms"] = result.finalize_ms;
	out["copy_cleanup_ms"] = result.cleanup_ms;
}

static pybind11::object CollectionToArrowTable(duckdb::ColumnDataCollection &collection,
                                               duckdb::ClientContext &context) {
	auto &types = collection.Types();
	duckdb::vector<string> names;
	names.reserve(types.size());
	for (duckdb::idx_t i = 0; i < types.size(); i++) {
		names.push_back(duckdb::StringUtil::Format("c%d", i));
	}

	auto options = context.GetClientProperties();
	auto extension_type_cast = duckdb::ArrowTypeExtensionData::GetExtensionTypes(context, types);
	pybind11::list batches;

	duckdb::ColumnDataScanState scan_state;
	collection.InitializeScan(scan_state);
	duckdb::DataChunk scan_chunk;
	collection.InitializeScanChunk(scan_chunk);

	while (collection.Scan(scan_state, scan_chunk)) {
		ArrowSchema schema;
		duckdb::ArrowConverter::ToArrowSchema(&schema, types, names, options);
		ArrowArray array;
		duckdb::ArrowConverter::ToArrowArray(scan_chunk, &array, options, extension_type_cast);
		duckdb::TransformDuckToArrowChunk(schema, array, batches);
	}

	return duckdb::pyarrow::ToArrowTable(types, names, batches, options);
}

using namespace duckdb::distributed::python::ray;

static py::dict FteSplitQueueResultToDict(const duckdb::distributed::FteSplitQueue::GetNextResult &result) {
	py::dict out;
	out["state"] = duckdb::distributed::FteSplitQueueGetResultName(result.state);
	if (result.HasSplit()) {
		switch (result.input.kind) {
		case duckdb::distributed::TaskInput::Kind::ScanTask:
			out["kind"] = "scan_task";
			out["data"] = py::bytes(result.input.scan_task_bytes);
			break;
		case duckdb::distributed::TaskInput::Kind::ExchangeSourceTask:
			out["kind"] = "exchange_source_task";
			out["data"] = py::bytes(result.input.exchange_source_task_bytes);
			break;
		}
	}
	return out;
}

static py::dict BuildFragmentStatsSummary(
    const std::unordered_map<std::string, std::unordered_map<std::string, duckdb::idx_t>> &stats_by_worker) {
	py::dict workers;
	std::unordered_map<std::string, duckdb::idx_t> totals;
	for (const auto &worker_entry : stats_by_worker) {
		py::dict worker_stats;
		for (const auto &stat_entry : worker_entry.second) {
			worker_stats[py::str(stat_entry.first)] = py::int_(stat_entry.second);
			totals[stat_entry.first] += stat_entry.second;
		}
		workers[py::str(worker_entry.first)] = std::move(worker_stats);
	}

	py::dict totals_dict;
	for (const auto &entry : totals) {
		totals_dict[py::str(entry.first)] = py::int_(entry.second);
	}

	py::dict result;
	result["workers"] = std::move(workers);
	result["totals"] = std::move(totals_dict);
	return result;
}

namespace {
struct CopyOutputInfo {
	std::string base;        // staging_root_base (empty for direct-write COPY)
	std::string run_id;      // staging_run_id (UUID)
	std::string remote_base; // spec.file_path (for direct write)
};

/// Worker-side: generate a unique output directory and set copy.file_path.
/// Called once per task execution. Each call generates a fresh UUID-based dir
/// so merged tasks (multiple scan files) still produce a unique output location.
static void ApplyTaskLocalCopyOutput(duckdb::PhysicalPlan &plan, const CopyOutputInfo *info,
                                     duckdb::ClientContext *client_context) {
	if (!plan.HasRoot()) {
		return;
	}
	auto &root = plan.Root();

	// Determine which operator to modify
	duckdb::PhysicalCopyToFile *copy_to_file = nullptr;
	duckdb::PhysicalBatchCopyToFile *batch_copy = nullptr;
	std::string *file_path_ptr = nullptr;

	if (root.type == duckdb::PhysicalOperatorType::COPY_TO_FILE) {
		copy_to_file = &root.Cast<duckdb::PhysicalCopyToFile>();
		file_path_ptr = &copy_to_file->file_path;
	} else if (root.type == duckdb::PhysicalOperatorType::BATCH_COPY_TO_FILE) {
		batch_copy = &root.Cast<duckdb::PhysicalBatchCopyToFile>();
		file_path_ptr = &batch_copy->file_path;
	} else {
		return; // Not a COPY operator
	}

	if (!duckdb::distributed::IsDistributedCopyOutputPlaceholder(*file_path_ptr)) {
		return; // Not a distributed placeholder — nothing to do
	}

	if (!info) {
		throw duckdb::InvalidInputException("Distributed COPY task is missing copy output info");
	}

	// Generate unique task directory using UUID
	auto worker_uuid = duckdb::UUID::ToString(duckdb::UUID::GenerateRandomUUID());
	auto worker_dir_name = "w_" + worker_uuid;
	const bool direct_target_output = info->base.empty();
	const bool remote_direct_target_output =
	    direct_target_output && duckdb::FileSystem::IsRemoteFile(info->remote_base);

	std::string task_dir;
	if (direct_target_output) {
		task_dir = info->remote_base;
	} else {
		// Local FS: use staging directory
		task_dir = info->base + "/" + info->run_id + "/" + worker_dir_name;
	}

	// Create the output directory
	if (client_context && !remote_direct_target_output) {
		auto &fs = duckdb::FileSystem::GetFileSystem(*client_context);
		if (!fs.DirectoryExists(task_dir)) {
			fs.CreateDirectoriesRecursive(task_dir);
		}
	}

	// Build the actual output path (file or directory depending on spec)
	// For COPY_TO_FILE: check spec flags to determine if needs_directory
	if (copy_to_file) {
		bool needs_directory =
		    copy_to_file->per_thread_output || copy_to_file->partition_output || copy_to_file->rotate;
		if (needs_directory) {
			if (direct_target_output) {
				copy_to_file->filename_pattern.SetFilenamePattern(
				    duckdb::distributed::BuildCopyDirectTargetFilenamePattern(info->run_id, worker_dir_name));
				// Direct-target distributed COPY writes unique run/task-prefixed files into
				// the final directory before finalize. Let task-local writers share that
				// directory; the coordinator still owns user-visible conflict semantics.
				copy_to_file->overwrite_mode = duckdb::CopyOverwriteMode::COPY_OVERWRITE_OR_IGNORE;
			}
			*file_path_ptr = task_dir;
		} else {
			// Single-file mode: append the base filename
			auto base_name = duckdb::StringUtil::GetFileName(info->remote_base);
			if (base_name.empty()) {
				base_name = "data";
			}
			if (!copy_to_file->file_extension.empty() && base_name.find('.') == std::string::npos) {
				if (copy_to_file->file_extension[0] != '.') {
					base_name += ".";
				}
				base_name += copy_to_file->file_extension;
			}
			if (direct_target_output) {
				auto separator = std::string("/");
				if (client_context) {
					auto &fs = duckdb::FileSystem::GetFileSystem(*client_context);
					separator = fs.PathSeparator(task_dir);
				}
				*file_path_ptr = duckdb::distributed::BuildCopyDirectTargetFilePath(
				    task_dir, info->run_id, worker_dir_name, base_name, separator);
			} else {
				*file_path_ptr = task_dir + "/" + base_name;
			}
		}
	} else {
		// BATCH_COPY_TO_FILE: always single-file
		auto base_name = duckdb::StringUtil::GetFileName(info->remote_base);
		if (base_name.empty()) {
			base_name = "data";
		}
		if (direct_target_output) {
			auto separator = std::string("/");
			if (client_context) {
				auto &fs = duckdb::FileSystem::GetFileSystem(*client_context);
				separator = fs.PathSeparator(task_dir);
			}
			*file_path_ptr = duckdb::distributed::BuildCopyDirectTargetFilePath(task_dir, info->run_id, worker_dir_name,
			                                                                    base_name, separator);
		} else {
			*file_path_ptr = task_dir + "/" + base_name;
		}
	}
}

static bool PyDictContains(const py::dict &dict, const char *key) {
	return dict.contains(py::str(key));
}

static py::object PyDictGet(const py::dict &dict, const char *key) {
	if (!PyDictContains(dict, key)) {
		return py::none();
	}
	return py::reinterpret_borrow<py::object>(dict[py::str(key)]);
}

static bool PyObjectTruthy(const py::object &obj) {
	if (obj.is_none()) {
		return false;
	}
	const auto truth = PyObject_IsTrue(obj.ptr());
	if (truth < 0) {
		PyErr_Clear();
		return false;
	}
	return truth != 0;
}

static bool IsDynamicFilterSpecDict(const py::dict &dict) {
	static constexpr const char *keys[] = {
	    "single_value", "value", "values", "range",       "min",          "max",
	    "lower",        "upper", "column", "column_name", "column_index", "column_id",
	};
	for (auto key : keys) {
		if (PyDictContains(dict, key)) {
			return true;
		}
	}
	return false;
}

static py::dict CopyDynamicFilterSpecWithColumn(const py::dict &domain, py::handle column) {
	py::dict copy;
	for (auto item : domain) {
		copy[item.first] = item.second;
	}
	copy["column"] = py::reinterpret_borrow<py::object>(column);
	return copy;
}

static duckdb::Value PyObjectToDuckValue(const py::object &obj) {
	if (obj.is_none()) {
		return duckdb::Value();
	}
	if (py::isinstance<py::bool_>(obj)) {
		return duckdb::Value::BOOLEAN(obj.cast<bool>());
	}
	if (py::isinstance<py::int_>(obj)) {
		return duckdb::Value::BIGINT(obj.cast<int64_t>());
	}
	if (py::isinstance<py::float_>(obj)) {
		return duckdb::Value::DOUBLE(obj.cast<double>());
	}
	if (py::isinstance<py::bytes>(obj)) {
		return duckdb::Value::BLOB(obj.cast<std::string>());
	}
	return duckdb::Value(py::str(obj).cast<std::string>());
}

static bool TryCastDynamicFilterValue(const py::object &obj, const duckdb::LogicalType &target_type,
                                      duckdb::Value &out) {
	if (obj.is_none()) {
		return false;
	}
	auto value = PyObjectToDuckValue(obj);
	if (value.IsNull()) {
		return false;
	}
	std::string error;
	return value.DefaultTryCastAs(target_type, out, &error, false);
}

static bool TryResolveScanFilterIndexForColumn(const duckdb::PhysicalTableScan &scan,
                                               duckdb::idx_t requested_column_index, duckdb::idx_t &filter_column_index,
                                               duckdb::idx_t &base_column_index) {
	for (duckdb::idx_t i = 0; i < scan.column_ids.size(); i++) {
		if (!scan.column_ids[i].IsRowIdColumn() && scan.column_ids[i].GetPrimaryIndex() == requested_column_index) {
			filter_column_index = i;
			base_column_index = requested_column_index;
			return true;
		}
	}
	if (requested_column_index < scan.column_ids.size()) {
		filter_column_index = requested_column_index;
		if (!scan.column_ids[filter_column_index].IsRowIdColumn()) {
			base_column_index = scan.column_ids[filter_column_index].GetPrimaryIndex();
		} else {
			base_column_index = requested_column_index;
		}
		return true;
	}
	return false;
}

static bool TryGetScanColumnType(const duckdb::PhysicalTableScan &scan, duckdb::idx_t base_column_index,
                                 duckdb::idx_t filter_column_index, duckdb::LogicalType &target_type) {
	if (base_column_index < scan.returned_types.size()) {
		target_type = scan.returned_types[base_column_index];
		return true;
	}
	if (filter_column_index < scan.types.size()) {
		target_type = scan.types[filter_column_index];
		return true;
	}
	return false;
}

static bool TryResolveDynamicFilterColumn(const duckdb::PhysicalTableScan &scan, const py::dict &domain,
                                          duckdb::idx_t &filter_column_index, duckdb::LogicalType &target_type) {
	py::object column_index_obj = PyDictGet(domain, "column_index");
	if (column_index_obj.is_none()) {
		column_index_obj = PyDictGet(domain, "column_id");
	}
	if (!column_index_obj.is_none()) {
		auto idx = column_index_obj.cast<int64_t>();
		if (idx < 0) {
			return false;
		}
		auto requested_column_index = static_cast<duckdb::idx_t>(idx);
		duckdb::idx_t base_column_index = requested_column_index;
		if (!TryResolveScanFilterIndexForColumn(scan, requested_column_index, filter_column_index, base_column_index)) {
			return false;
		}
		return TryGetScanColumnType(scan, base_column_index, filter_column_index, target_type);
	}

	py::object column_obj = PyDictGet(domain, "column");
	if (column_obj.is_none()) {
		column_obj = PyDictGet(domain, "column_name");
	}
	if (column_obj.is_none()) {
		column_obj = PyDictGet(domain, "name");
	}
	if (column_obj.is_none()) {
		return false;
	}
	auto column_name = py::str(column_obj).cast<std::string>();
	for (duckdb::idx_t base_column_index = 0; base_column_index < scan.names.size(); base_column_index++) {
		if (scan.names[base_column_index] != column_name) {
			continue;
		}
		duckdb::idx_t resolved_base_column_index = base_column_index;
		if (!TryResolveScanFilterIndexForColumn(scan, base_column_index, filter_column_index,
		                                        resolved_base_column_index)) {
			return false;
		}
		return TryGetScanColumnType(scan, resolved_base_column_index, filter_column_index, target_type);
	}
	return false;
}

static bool DynamicFilterSourceMatches(const duckdb::PhysicalTableScan &scan, const std::string &source_node_id) {
	if (source_node_id.empty()) {
		return true;
	}
	if (!scan.extra_info.scan_node_id.IsValid()) {
		return false;
	}
	return std::to_string(scan.extra_info.scan_node_id.GetIndex()) == source_node_id;
}

static std::string DynamicFilterSourceFromSpec(const std::string &fallback_source_id, const py::dict &domain) {
	for (auto key : {"source_node_id", "plan_node_id", "node_id", "source"}) {
		auto value = PyDictGet(domain, key);
		if (!value.is_none()) {
			return py::str(value).cast<std::string>();
		}
	}
	return fallback_source_id;
}

static void PushDynamicTableFilter(duckdb::PhysicalTableScan &scan, duckdb::idx_t filter_column_index,
                                   duckdb::unique_ptr<duckdb::TableFilter> filter) {
	if (!filter) {
		return;
	}
	if (!scan.table_filters) {
		scan.table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
	}
	scan.table_filters->PushFilter(duckdb::ColumnIndex(filter_column_index), std::move(filter));
}

static duckdb::idx_t ApplyDynamicFilterSpecToScan(duckdb::PhysicalTableScan &scan,
                                                  const std::string &fallback_source_id, const py::dict &domain) {
	auto source_node_id = DynamicFilterSourceFromSpec(fallback_source_id, domain);
	if (!DynamicFilterSourceMatches(scan, source_node_id)) {
		return 0;
	}

	duckdb::idx_t filter_column_index = 0;
	duckdb::LogicalType target_type;
	if (!TryResolveDynamicFilterColumn(scan, domain, filter_column_index, target_type)) {
		return 0;
	}
	duckdb::idx_t applied = 0;

	auto single_value = PyDictGet(domain, "single_value");
	if (single_value.is_none()) {
		single_value = PyDictGet(domain, "value");
	}
	duckdb::Value value;
	if (TryCastDynamicFilterValue(single_value, target_type, value)) {
		PushDynamicTableFilter(
		    scan, filter_column_index,
		    duckdb::make_uniq<duckdb::ConstantFilter>(duckdb::ExpressionType::COMPARE_EQUAL, std::move(value)));
		applied++;
	}

	auto values_obj = PyDictGet(domain, "values");
	if (PyObjectTruthy(values_obj) && py::isinstance<py::sequence>(values_obj) &&
	    !py::isinstance<py::str>(values_obj) && !py::isinstance<py::bytes>(values_obj)) {
		std::vector<duckdb::Value> values;
		for (py::handle item : values_obj.cast<py::sequence>()) {
			duckdb::Value item_value;
			if (TryCastDynamicFilterValue(py::reinterpret_borrow<py::object>(item), target_type, item_value)) {
				values.push_back(std::move(item_value));
			}
		}
		if (values.size() == 1) {
			auto only_value = std::move(values[0]);
			PushDynamicTableFilter(scan, filter_column_index,
			                       duckdb::make_uniq<duckdb::ConstantFilter>(duckdb::ExpressionType::COMPARE_EQUAL,
			                                                                 std::move(only_value)));
			applied++;
		} else if (!values.empty()) {
			PushDynamicTableFilter(scan, filter_column_index, duckdb::make_uniq<duckdb::InFilter>(std::move(values)));
			applied++;
		}
	}

	py::object lower = PyDictGet(domain, "min");
	if (lower.is_none()) {
		lower = PyDictGet(domain, "lower");
	}
	py::object upper = PyDictGet(domain, "max");
	if (upper.is_none()) {
		upper = PyDictGet(domain, "upper");
	}
	auto range_obj = PyDictGet(domain, "range");
	if (PyObjectTruthy(range_obj) && py::isinstance<py::sequence>(range_obj) && py::len(range_obj) >= 2) {
		auto range = range_obj.cast<py::sequence>();
		lower = py::reinterpret_borrow<py::object>(range[0]);
		upper = py::reinterpret_borrow<py::object>(range[1]);
	}

	duckdb::Value lower_value;
	if (TryCastDynamicFilterValue(lower, target_type, lower_value)) {
		PushDynamicTableFilter(scan, filter_column_index,
		                       duckdb::make_uniq<duckdb::ConstantFilter>(
		                           duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, std::move(lower_value)));
		applied++;
	}
	duckdb::Value upper_value;
	if (TryCastDynamicFilterValue(upper, target_type, upper_value)) {
		PushDynamicTableFilter(scan, filter_column_index,
		                       duckdb::make_uniq<duckdb::ConstantFilter>(
		                           duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO, std::move(upper_value)));
		applied++;
	}

	return applied;
}

static duckdb::idx_t ApplyDynamicFilterDomainsToScan(duckdb::PhysicalTableScan &scan, const py::dict &domains) {
	duckdb::idx_t applied = 0;
	for (auto item : domains) {
		auto key = py::str(item.first).cast<std::string>();
		auto value_obj = py::reinterpret_borrow<py::object>(item.second);
		if (!py::isinstance<py::dict>(value_obj)) {
			continue;
		}
		auto domain = value_obj.cast<py::dict>();
		if (IsDynamicFilterSpecDict(domain)) {
			applied += ApplyDynamicFilterSpecToScan(scan, std::string(), domain);
			continue;
		}

		auto columns_obj = PyDictGet(domain, "columns");
		if (!columns_obj.is_none() && py::isinstance<py::dict>(columns_obj)) {
			auto columns = columns_obj.cast<py::dict>();
			for (auto column_item : columns) {
				auto column_domain_obj = py::reinterpret_borrow<py::object>(column_item.second);
				if (!py::isinstance<py::dict>(column_domain_obj)) {
					continue;
				}
				auto nested = CopyDynamicFilterSpecWithColumn(column_domain_obj.cast<py::dict>(), column_item.first);
				applied += ApplyDynamicFilterSpecToScan(scan, key, nested);
			}
			continue;
		}

		for (auto column_item : domain) {
			auto nested_obj = py::reinterpret_borrow<py::object>(column_item.second);
			if (!py::isinstance<py::dict>(nested_obj)) {
				continue;
			}
			auto nested = nested_obj.cast<py::dict>();
			if (!IsDynamicFilterSpecDict(nested)) {
				continue;
			}
			if (!PyDictContains(nested, "column") && !PyDictContains(nested, "column_name") &&
			    !PyDictContains(nested, "column_index") && !PyDictContains(nested, "column_id")) {
				nested = CopyDynamicFilterSpecWithColumn(nested, column_item.first);
			}
			applied += ApplyDynamicFilterSpecToScan(scan, key, nested);
		}
	}
	return applied;
}

static duckdb::idx_t ApplyDynamicFilterDomainsToOperator(duckdb::PhysicalOperator &op, const py::dict &domains) {
	duckdb::idx_t applied = 0;
	if (op.type == duckdb::PhysicalOperatorType::TABLE_SCAN) {
		applied += ApplyDynamicFilterDomainsToScan(op.Cast<duckdb::PhysicalTableScan>(), domains);
	}
	auto children = op.GetChildren();
	if (!children.empty()) {
		for (auto &child : children) {
			applied +=
			    ApplyDynamicFilterDomainsToOperator(const_cast<duckdb::PhysicalOperator &>(child.get()), domains);
		}
		return applied;
	}
	for (auto &child : op.children) {
		applied += ApplyDynamicFilterDomainsToOperator(child.get(), domains);
	}
	return applied;
}

static duckdb::idx_t ApplyDynamicFilterDomainsToPlan(duckdb::PhysicalPlan &plan,
                                                     py::object dynamic_filter_domains_obj) {
	if (dynamic_filter_domains_obj.is_none() || !plan.HasRoot()) {
		return 0;
	}
	if (!py::isinstance<py::dict>(dynamic_filter_domains_obj)) {
		throw py::value_error("dynamic_filter_domains must be a dict");
	}
	auto domains = dynamic_filter_domains_obj.cast<py::dict>();
	if (py::len(domains) == 0) {
		return 0;
	}
	return ApplyDynamicFilterDomainsToOperator(plan.Root(), domains);
}

} // namespace
