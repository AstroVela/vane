// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/function/table/datasource_scan.hpp"

#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/type_visitor.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/function/cast/cast_function_set.hpp"

namespace duckdb {

static const string DATASOURCE_SPLIT_CODEC = "vane.datasource-split.python-pickle";

// Global produce_stream callback — set once from Python module init,
// used to restore the callback on workers after deserialization.
static std::atomic<datasource_produce_stream_t> g_global_produce_stream {nullptr};
static std::atomic<datasource_acquire_source_t> g_global_acquire_source {nullptr};
static std::atomic<datasource_release_source_t> g_global_release_source {nullptr};
static std::atomic<datasource_get_schema_t> g_global_get_schema {nullptr};

static datasource_produce_stream_t RequireProduceStream(datasource_produce_stream_t callback) {
	if (!callback) {
		throw InvalidInputException(
		    "Python datasource runtime is not initialized in this process; import duckdb before executing "
		    "datasource_scan on Ray workers");
	}
	return callback;
}

static vector<DistributedScanSplit>
DataSourcePlanDistributedScanSplits(const TableFunctionDistributedScanPlanningInput &input) {
	if (!input.bind_data) {
		throw InvalidInputException("distributed datasource scan requires bind data");
	}
	auto &bind_data = input.bind_data->Cast<DataSourceScanBindData>();
	vector<DistributedScanSplit> splits;
	splits.reserve(bind_data.pickled_tasks.size());
	for (idx_t task_index = 0; task_index < bind_data.pickled_tasks.size(); task_index++) {
		DistributedScanSplit split;
		split.split_id = std::to_string(task_index);
		split.payload = bind_data.pickled_tasks[task_index];
		splits.push_back(std::move(split));
	}
	return splits;
}

static unique_ptr<FunctionData> DataSourceCreateDistributedWorkerBind(const TableFunctionDistributedScanInput &input) {
	if (!input.bind_data) {
		throw InvalidInputException("distributed datasource scan requires bind data");
	}
	auto &source_bind = input.bind_data->Cast<DataSourceScanBindData>();
	auto worker_bind = make_uniq<DataSourceScanBindData>();
	worker_bind->pickled_source = source_bind.pickled_source;
	worker_bind->query_id = source_bind.query_id;
	worker_bind->estimated_cardinality = source_bind.estimated_cardinality;
	worker_bind->snapshot_types = source_bind.snapshot_types;
	worker_bind->produce_stream = nullptr;
	return std::move(worker_bind);
}

static bool IsCanonicalDataSourceSplitId(const string &split_id) {
	if (split_id.empty() || (split_id.size() > 1 && split_id[0] == '0')) {
		return false;
	}
	for (auto character : split_id) {
		if (character < '0' || character > '9') {
			return false;
		}
	}
	return true;
}

static void DataSourceApplyDistributedSplits(optional_ptr<FunctionData> worker_bind_data,
                                             const vector<DistributedScanSplit> &splits) {
	if (!worker_bind_data) {
		throw InvalidInputException("distributed datasource scan requires worker bind data");
	}
	auto &bind_data = worker_bind_data->Cast<DataSourceScanBindData>();
	vector<string> validated_tasks;
	validated_tasks.reserve(splits.size());
	for (const auto &split : splits) {
		if (!IsCanonicalDataSourceSplitId(split.split_id) || split.payload.empty()) {
			throw InvalidInputException("invalid distributed DataSource split '%s'", split.split_id);
		}
		validated_tasks.push_back(split.payload);
	}
	bind_data.pickled_tasks = std::move(validated_tasks);
}

// ── Bind ───────────────────────────────────────────────────────────
// Args: produce_stream_ptr (POINTER), get_schema_ptr (POINTER),
//       pickled_source (BLOB), pickled_tasks (LIST<BLOB>)

static unique_ptr<FunctionData> DataSourceScanBind(ClientContext &context, TableFunctionBindInput &input,
                                                   vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_uniq<DataSourceScanBindData>();

	auto produce_stream_ptr = input.inputs[0].GetPointer();
	auto get_schema_ptr = input.inputs[1].GetPointer();
	auto &pickled_source = StringValue::Get(input.inputs[2]);

	result->produce_stream = reinterpret_cast<datasource_produce_stream_t>(produce_stream_ptr);
	RequireProduceStream(result->produce_stream);
	result->pickled_source = pickled_source;

	// Extract pickled tasks from the LIST<BLOB>
	auto &task_list = input.inputs[3];
	auto &task_children = ListValue::GetChildren(task_list);
	for (auto &child : task_children) {
		result->pickled_tasks.push_back(StringValue::Get(child));
	}

	// Get schema via callback
	auto get_schema = reinterpret_cast<datasource_get_schema_t>(get_schema_ptr);
	if (!get_schema) {
		throw InvalidInputException(
		    "Python datasource runtime is not initialized in this process; missing datasource schema callback");
	}
	ArrowSchemaWrapper arrow_schema;
	get_schema(pickled_source.c_str(), pickled_source.size(), &arrow_schema.arrow_schema);

	// Parse Arrow schema into DuckDB types
	ArrowTableFunction::PopulateArrowTableSchema(context, result->arrow_table, arrow_schema.arrow_schema);
	names = result->arrow_table.GetNames();
	return_types = result->arrow_table.GetTypes();

	return std::move(result);
}

// ── Init Global ────────────────────────────────────────────────────

void DataSourceScanFunction::SetSnapshotTypes(DataSourceScanBindData &bind_data, const vector<LogicalType> &types) {
	auto &arrow_types = bind_data.arrow_table.GetTypes();
	if (arrow_types.size() != types.size()) {
		throw InvalidInputException("Native memory snapshot column count does not match its bound schema");
	}
	for (idx_t i = 0; i < types.size(); i++) {
		if (arrow_types[i] == types[i] ||
		    (types[i].id() == LogicalTypeId::ENUM && arrow_types[i].id() == LogicalTypeId::VARCHAR) ||
		    (TypeVisitor::Contains(types[i], GovernedLogicalType::IsGoverned) &&
		     GovernedLogicalType::IsCanonicalStorageType(arrow_types[i], types[i]))) {
			continue;
		}
		throw InvalidInputException("Native memory snapshot changed column '%s' from %s to %s",
		                            bind_data.arrow_table.GetNames()[i], types[i], arrow_types[i]);
	}
	bind_data.snapshot_types = arrow_types == types ? vector<LogicalType>() : types;
}

DataSourceScanGlobalState::~DataSourceScanGlobalState() {
	if (!release_source_on_destroy || !release_source) {
		return;
	}
	try {
		release_source(pickled_source.c_str(), pickled_source.size());
	} catch (...) { // Destructors must not propagate callback failures.
	}
}

static unique_ptr<GlobalTableFunctionState> DataSourceScanInitGlobal(ClientContext &context,
                                                                     TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<DataSourceScanBindData>();
	auto result = make_uniq<DataSourceScanGlobalState>();
	result->total_tasks = bind_data.pickled_tasks.size();
	result->next_task_idx = 0;

	// Resolve ownership callbacks before restoring schema, but do not acquire
	// until schema restoration has passed every fallible initialization step.
	auto acquire_source = g_global_acquire_source.load();
	auto release_source = g_global_release_source.load();
	if (!bind_data.pickled_source.empty() && (!acquire_source || !release_source)) {
		throw InvalidInputException(
		    "Python datasource runtime is not initialized on this Ray worker; missing datasource source callbacks");
	}

	// Restore arrow_table on worker nodes (type_info is not picklable).
	if (!bind_data.pickled_source.empty()) {
		auto get_schema_cb = g_global_get_schema.load();
		if (!get_schema_cb) {
			throw InvalidInputException(
			    "Python datasource runtime is not initialized on this Ray worker; missing datasource schema callback");
		}
		ArrowSchemaWrapper arrow_schema;
		get_schema_cb(bind_data.pickled_source.c_str(), bind_data.pickled_source.size(), &arrow_schema.arrow_schema);
		// Reset to empty so AddColumn's emplace() succeeds
		const_cast<DataSourceScanBindData &>(bind_data).arrow_table = ArrowTableSchema();
		ArrowTableFunction::PopulateArrowTableSchema(
		    context, const_cast<DataSourceScanBindData &>(bind_data).arrow_table, arrow_schema.arrow_schema);
	}
	if (!bind_data.snapshot_types.empty()) {
		// Repeat schema admission on workers before acquiring a source or
		// interpreting Arrow buffers as the coordinator's bound logical types.
		DataSourceScanFunction::SetSnapshotTypes(const_cast<DataSourceScanBindData &>(bind_data),
		                                         bind_data.snapshot_types);
		result->snapshot_storage_types = const_cast<DataSourceScanBindData &>(bind_data).arrow_table.GetTypes();
	}

	// Acquire one process-local factory owner for this execution. Local scan
	// ownership follows the global state; distributed ownership follows the
	// logical query and is released only after worker executions are drained.
	if (acquire_source && release_source && !bind_data.pickled_source.empty()) {
		if (bind_data.query_id.empty()) {
			result->release_source = release_source;
			result->pickled_source = bind_data.pickled_source;
		}
		acquire_source(bind_data.pickled_source.c_str(), bind_data.pickled_source.size(), bind_data.query_id.c_str(),
		               bind_data.query_id.size());
		if (bind_data.query_id.empty()) {
			result->release_source_on_destroy = true;
		}
	}

	return std::move(result);
}

// ── Init Local ─────────────────────────────────────────────────────
// Each pipeline thread gets its own local state. On init, grab first task.

static void DataSourceScanStartNextTask(ClientContext &context, const DataSourceScanBindData &bind_data,
                                        DataSourceScanGlobalState &gstate, DataSourceScanLocalState &lstate) {
	D_ASSERT(lstate.state == DataSourceScanLocalState::ScanState::NEED_TASK);
	D_ASSERT(!lstate.stream);

	auto idx = gstate.next_task_idx.fetch_add(1);
	if (idx >= gstate.total_tasks) {
		lstate.state = DataSourceScanLocalState::ScanState::EXHAUSTED;
		return;
	}

	auto &pickled = bind_data.pickled_tasks[idx];
	auto stream_wrapper = make_uniq<ArrowArrayStreamWrapper>();
	RequireProduceStream(bind_data.produce_stream)(pickled.c_str(), pickled.size(), &stream_wrapper->arrow_array_stream,
	                                               &context);
	lstate.stream = std::move(stream_wrapper);
	lstate.state = DataSourceScanLocalState::ScanState::NEED_BATCH;
}

static unique_ptr<LocalTableFunctionState> DataSourceScanInitLocal(ExecutionContext &context,
                                                                   TableFunctionInitInput &input,
                                                                   GlobalTableFunctionState *global_state) {
	auto &bind_data = input.bind_data->Cast<DataSourceScanBindData>();
	auto &gstate = global_state->Cast<DataSourceScanGlobalState>();
	auto result = make_uniq<DataSourceScanLocalState>(context.client);
	for (idx_t i = 0; i < bind_data.arrow_table.GetColumns().size(); i++) {
		result->scan_state.column_ids.push_back(i);
	}
	if (!bind_data.snapshot_types.empty()) {
		result->snapshot_chunk.Initialize(context.client, gstate.snapshot_storage_types);
	}
	DataSourceScanStartNextTask(context.client, bind_data, gstate, *result);
	return std::move(result);
}

// ── GetData ────────────────────────────────────────────────────────
// Each pipeline thread pulls chunks from its current ArrowArrayStream.
// When exhausted, grabs the next task.

static void DataSourceScanGetData(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<DataSourceScanBindData>();
	auto &gstate = data.global_state->Cast<DataSourceScanGlobalState>();
	auto &lstate = data.local_state->Cast<DataSourceScanLocalState>();

	while (true) {
		switch (lstate.state) {
		case DataSourceScanLocalState::ScanState::NEED_TASK:
			DataSourceScanStartNextTask(context, bind_data, gstate, lstate);
			break;
		case DataSourceScanLocalState::ScanState::NEED_BATCH: {
			D_ASSERT(lstate.stream);
			auto &scan_state = lstate.scan_state;
			scan_state.Reset();
			auto chunk = lstate.stream->GetNextChunk();
			while (chunk->arrow_array.release && chunk->arrow_array.length == 0) {
				chunk = lstate.stream->GetNextChunk();
			}
			scan_state.chunk = std::move(chunk);
			if (scan_state.chunk->arrow_array.release) {
				lstate.state = DataSourceScanLocalState::ScanState::SCANNING;
			} else {
				lstate.stream.reset();
				lstate.state = DataSourceScanLocalState::ScanState::NEED_TASK;
			}
			break;
		}
		case DataSourceScanLocalState::ScanState::SCANNING: {
			auto &scan_state = lstate.scan_state;
			D_ASSERT(scan_state.chunk->arrow_array.release);
			auto chunk_size = NumericCast<idx_t>(scan_state.chunk->arrow_array.length);
			D_ASSERT(scan_state.chunk_offset < chunk_size);
			auto output_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, chunk_size - scan_state.chunk_offset);
			output.SetCardinality(output_size);
			auto &arrow_output = bind_data.snapshot_types.empty() ? output : lstate.snapshot_chunk;
			if (!bind_data.snapshot_types.empty()) {
				arrow_output.Reset();
				arrow_output.SetCardinality(output_size);
			}
			ArrowTableFunction::ArrowToDuckDB(scan_state, bind_data.arrow_table.GetColumns(), arrow_output,
			                                  false /* arrow_scan_is_projected */);
			if (!bind_data.snapshot_types.empty()) {
				for (idx_t col = 0; col < output.ColumnCount(); col++) {
					auto &source = arrow_output.data[col];
					auto &target = output.data[col];
					if (source.GetType() == target.GetType()) {
						target.Reference(source);
					} else if (TypeVisitor::Contains(target.GetType(), GovernedLogicalType::IsGoverned)) {
						auto &casts = CastFunctionSet::Get(context);
						GetCastFunctionInput cast_input(context);
						cast_input.file_cast_mode = FileCastMode::INTERNAL_ALIAS_RESTORATION;
						VectorOperations::TryCast(casts, cast_input, source, target, output_size, nullptr);
					} else {
						VectorOperations::Cast(context, source, target, output_size);
					}
				}
			}
			output.Verify();
			scan_state.chunk_offset += output.size();
			if (scan_state.chunk_offset == chunk_size) {
				lstate.state = DataSourceScanLocalState::ScanState::NEED_BATCH;
			}
			return;
		}
		case DataSourceScanLocalState::ScanState::EXHAUSTED:
			output.SetCardinality(0);
			return;
		}
	}
}

// ── Serialize/Deserialize ──────────────────────────────────────────

static void DataSourceScanSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data_p,
                                    const TableFunction &function) {
	auto &bind_data = bind_data_p->Cast<DataSourceScanBindData>();
	serializer.WriteProperty(100, "pickled_tasks", bind_data.pickled_tasks);
	serializer.WriteProperty(101, "pickled_source", bind_data.pickled_source);
	serializer.WriteProperty(102, "query_id", bind_data.query_id);
	serializer.WritePropertyWithDefault<optional_idx>(103, "estimated_cardinality", bind_data.estimated_cardinality);
	serializer.WritePropertyWithDefault(104, "snapshot_types", bind_data.snapshot_types);
}

static unique_ptr<FunctionData> DataSourceScanDeserialize(Deserializer &deserializer, TableFunction &function) {
	auto result = make_uniq<DataSourceScanBindData>();
	result->pickled_tasks = deserializer.ReadProperty<vector<string>>(100, "pickled_tasks");
	result->pickled_source = deserializer.ReadProperty<string>(101, "pickled_source");
	result->query_id = deserializer.ReadProperty<string>(102, "query_id");
	result->estimated_cardinality = deserializer.ReadPropertyWithDefault<optional_idx>(103, "estimated_cardinality");
	result->snapshot_types = deserializer.ReadPropertyWithDefault<vector<LogicalType>>(104, "snapshot_types");
	// Restore produce_stream from global callback (set by Python module on load)
	result->produce_stream = g_global_produce_stream.load();
	RequireProduceStream(result->produce_stream);
	return std::move(result);
}

static unique_ptr<NodeStatistics> DataSourceScanCardinality(ClientContext &, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<DataSourceScanBindData>();
	if (!bind_data.estimated_cardinality.IsValid()) {
		return nullptr;
	}
	return make_uniq<NodeStatistics>(bind_data.estimated_cardinality.GetIndex());
}

// ── Registration ───────────────────────────────────────────────────

TableFunction DataSourceScanFunction::GetFunction() {
	// Args: produce_stream_ptr, get_schema_ptr, pickled_source, pickled_tasks_list
	TableFunction func(
	    "datasource_scan",
	    {LogicalType::POINTER, LogicalType::POINTER, LogicalType::BLOB, LogicalType::LIST(LogicalType::BLOB)},
	    DataSourceScanGetData, DataSourceScanBind, DataSourceScanInitGlobal, DataSourceScanInitLocal);
	func.serialize = DataSourceScanSerialize;
	func.deserialize = DataSourceScanDeserialize;
	func.cardinality = DataSourceScanCardinality;
	TableFunctionDistributedScanCallbacks distributed_scan;
	distributed_scan.protocol_version = 1;
	distributed_scan.split_codec = {DATASOURCE_SPLIT_CODEC, 1};
	distributed_scan.plan_splits = DataSourcePlanDistributedScanSplits;
	distributed_scan.create_worker_bind = DataSourceCreateDistributedWorkerBind;
	distributed_scan.apply_splits = DataSourceApplyDistributedSplits;
	func.SetDistributedScanCallbacks(std::move(distributed_scan));
	func.BindDistributedScanCapability("vane_core");
	func.projection_pushdown = false;
	return func;
}

void DataSourceScanFunction::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(DataSourceScanFunction::GetFunction());
}

void DataSourceScanFunction::SetGlobalProduceStream(datasource_produce_stream_t callback) {
	g_global_produce_stream.store(callback);
}

datasource_produce_stream_t DataSourceScanFunction::GetGlobalProduceStream() {
	return g_global_produce_stream.load();
}

void DataSourceScanFunction::SetGlobalAcquireSource(datasource_acquire_source_t callback) {
	g_global_acquire_source.store(callback);
}

void DataSourceScanFunction::SetGlobalReleaseSource(datasource_release_source_t callback) {
	g_global_release_source.store(callback);
}

void DataSourceScanFunction::SetGlobalGetSchema(datasource_get_schema_t callback) {
	g_global_get_schema.store(callback);
}

} // namespace duckdb
