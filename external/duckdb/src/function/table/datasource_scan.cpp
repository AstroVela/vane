// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/function/table/datasource_scan.hpp"

#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/blob.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"

namespace duckdb {

static const string DATASOURCE_PREFIX = "datasource://";

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

DistributedExtensionCapabilityReference DataSourceScanBindData::GetDistributedExtensionCapability() const {
	DistributedExtensionCapabilityReference result;
	result.extension_name = "vane_core";
	result.extension_protocol_version = 1;
	result.capability.kind = DistributedExtensionCapabilityKind::TABLE_FUNCTION;
	result.capability.name = "datasource_scan";
	result.capability.protocol_version = 1;
	return result;
}

vector<OpenFileInfo> DataSourceScanBindData::GetScanTasks() const {
	vector<OpenFileInfo> tasks;
	tasks.reserve(pickled_tasks.size());
	for (auto &task : pickled_tasks) {
		// Encode pickled task bytes as an opaque base64 token.
		auto encoded = Blob::ToBase64(string_t(task.data(), task.size()));
		tasks.emplace_back(DATASOURCE_PREFIX + encoded);
	}
	return tasks;
}

void DataSourceScanBindData::SetScanTasks(const vector<OpenFileInfo> &tasks) {
	pickled_tasks.clear();
	pickled_tasks.reserve(tasks.size());
	for (const auto &task : tasks) {
		auto &token = task.path;
		// Strip the prefix and decode base64 back to pickled task bytes
		if (token.substr(0, DATASOURCE_PREFIX.size()) == DATASOURCE_PREFIX) {
			auto base64_str = token.substr(DATASOURCE_PREFIX.size());
			auto decoded = Blob::FromBase64(string_t(base64_str.data(), base64_str.size()));
			pickled_tasks.push_back(std::move(decoded));
		} else {
			throw InvalidInputException("Expected datasource scan task descriptor to start with '%s'",
			                            DATASOURCE_PREFIX);
		}
	}
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

static void DataSourceScanStartNextTask(const DataSourceScanBindData &bind_data, DataSourceScanGlobalState &gstate,
                                        DataSourceScanLocalState &lstate) {
	D_ASSERT(lstate.state == DataSourceScanLocalState::ScanState::NEED_TASK);
	D_ASSERT(!lstate.stream);

	auto idx = gstate.next_task_idx.fetch_add(1);
	if (idx >= gstate.total_tasks) {
		lstate.state = DataSourceScanLocalState::ScanState::EXHAUSTED;
		return;
	}

	auto &pickled = bind_data.pickled_tasks[idx];
	auto stream_wrapper = make_uniq<ArrowArrayStreamWrapper>();
	RequireProduceStream(bind_data.produce_stream)(pickled.c_str(), pickled.size(),
	                                               &stream_wrapper->arrow_array_stream);
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
	DataSourceScanStartNextTask(bind_data, gstate, *result);
	return std::move(result);
}

// ── GetData ────────────────────────────────────────────────────────
// Each pipeline thread pulls chunks from its current ArrowArrayStream.
// When exhausted, grabs the next task.

static void DataSourceScanGetData(ClientContext &, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<DataSourceScanBindData>();
	auto &gstate = data.global_state->Cast<DataSourceScanGlobalState>();
	auto &lstate = data.local_state->Cast<DataSourceScanLocalState>();

	while (true) {
		switch (lstate.state) {
		case DataSourceScanLocalState::ScanState::NEED_TASK:
			DataSourceScanStartNextTask(bind_data, gstate, lstate);
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
			ArrowTableFunction::ArrowToDuckDB(scan_state, bind_data.arrow_table.GetColumns(), output,
			                                  false /* arrow_scan_is_projected */);
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
}

static unique_ptr<FunctionData> DataSourceScanDeserialize(Deserializer &deserializer, TableFunction &function) {
	auto result = make_uniq<DataSourceScanBindData>();
	result->pickled_tasks = deserializer.ReadProperty<vector<string>>(100, "pickled_tasks");
	result->pickled_source = deserializer.ReadProperty<string>(101, "pickled_source");
	result->query_id = deserializer.ReadProperty<string>(102, "query_id");
	// Restore produce_stream from global callback (set by Python module on load)
	result->produce_stream = g_global_produce_stream.load();
	RequireProduceStream(result->produce_stream);
	return std::move(result);
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
