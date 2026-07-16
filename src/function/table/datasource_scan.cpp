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
static std::atomic<datasource_set_worker_source_t> g_global_worker_source_callback {nullptr};
static std::atomic<datasource_get_schema_t> g_global_get_schema {nullptr};

static datasource_produce_stream_t RequireProduceStream(datasource_produce_stream_t callback) {
	if (!callback) {
		throw InvalidInputException(
		    "Python datasource runtime is not initialized in this process; import duckdb before executing "
		    "datasource_scan on Ray workers");
	}
	return callback;
}

vector<string> DataSourceScanBindData::GetFileList() const {
	vector<string> files;
	files.reserve(pickled_tasks.size());
	for (auto &task : pickled_tasks) {
		// Encode pickled task bytes as base64 with a prefix
		auto encoded = Blob::ToBase64(string_t(task.data(), task.size()));
		files.push_back(DATASOURCE_PREFIX + encoded);
	}
	return files;
}

void DataSourceScanBindData::SetFileList(const vector<string> &files) {
	pickled_tasks.clear();
	pickled_tasks.reserve(files.size());
	for (auto &f : files) {
		// Strip the prefix and decode base64 back to pickled task bytes
		if (f.substr(0, DATASOURCE_PREFIX.size()) == DATASOURCE_PREFIX) {
			auto base64_str = f.substr(DATASOURCE_PREFIX.size());
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
	ArrowSchema arrow_schema;
	get_schema(pickled_source.c_str(), pickled_source.size(), &arrow_schema);

	// Parse Arrow schema into DuckDB types
	ArrowTableFunction::PopulateArrowTableSchema(context, result->arrow_table, arrow_schema);
	names = result->arrow_table.GetNames();
	return_types = result->arrow_table.GetTypes();

	if (arrow_schema.release) {
		arrow_schema.release(&arrow_schema);
	}

	return std::move(result);
}

// ── Init Global ────────────────────────────────────────────────────

static unique_ptr<GlobalTableFunctionState> DataSourceScanInitGlobal(ClientContext &context,
                                                                     TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<DataSourceScanBindData>();
	auto result = make_uniq<DataSourceScanGlobalState>();
	result->total_tasks = bind_data.pickled_tasks.size();
	result->next_task_idx = 0;

	// On a worker, set the pickled_source so ProduceStream can lazily create factories.
	auto worker_source_cb = g_global_worker_source_callback.load();
	if (!bind_data.pickled_source.empty() && !worker_source_cb) {
		throw InvalidInputException("Python datasource runtime is not initialized on this Ray worker; missing "
		                            "datasource worker source callback");
	}
	if (worker_source_cb && !bind_data.pickled_source.empty()) {
		worker_source_cb(bind_data.pickled_source.c_str(), bind_data.pickled_source.size());
	}

	// Restore arrow_table on worker nodes (type_info is not picklable).
	if (!bind_data.pickled_source.empty()) {
		auto get_schema_cb = g_global_get_schema.load();
		if (!get_schema_cb) {
			throw InvalidInputException(
			    "Python datasource runtime is not initialized on this Ray worker; missing datasource schema callback");
		}
		ArrowSchema arrow_schema;
		get_schema_cb(bind_data.pickled_source.c_str(), bind_data.pickled_source.size(), &arrow_schema);
		// Reset to empty so AddColumn's emplace() succeeds
		const_cast<DataSourceScanBindData &>(bind_data).arrow_table = ArrowTableSchema();
		ArrowTableFunction::PopulateArrowTableSchema(
		    context, const_cast<DataSourceScanBindData &>(bind_data).arrow_table, arrow_schema);
		if (arrow_schema.release) {
			arrow_schema.release(&arrow_schema);
		}
	}

	return std::move(result);
}

// ── Init Local ─────────────────────────────────────────────────────
// Each pipeline thread gets its own local state. On init, grab first task.

static unique_ptr<LocalTableFunctionState> DataSourceScanInitLocal(ExecutionContext &context,
                                                                   TableFunctionInitInput &input,
                                                                   GlobalTableFunctionState *global_state) {
	auto &bind_data = input.bind_data->Cast<DataSourceScanBindData>();
	auto &gstate = global_state->Cast<DataSourceScanGlobalState>();
	auto result = make_uniq<DataSourceScanLocalState>();

	// Grab first task for this thread
	auto idx = gstate.next_task_idx.fetch_add(1);
	if (idx >= gstate.total_tasks) {
		result->exhausted = true;
		return std::move(result);
	}

	// Produce stream for this task
	auto &pickled = bind_data.pickled_tasks[idx];
	auto stream_wrapper = make_uniq<ArrowArrayStreamWrapper>();
	RequireProduceStream(bind_data.produce_stream)(pickled.c_str(), pickled.size(),
	                                               &stream_wrapper->arrow_array_stream);
	result->stream = std::move(stream_wrapper);

	return std::move(result);
}

// ── GetData ────────────────────────────────────────────────────────
// Each pipeline thread pulls chunks from its current ArrowArrayStream.
// When exhausted, grabs the next task.

static void DataSourceScanGetData(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<DataSourceScanBindData>();
	auto &gstate = data.global_state->Cast<DataSourceScanGlobalState>();
	auto &lstate = data.local_state->Cast<DataSourceScanLocalState>();

	while (!lstate.exhausted) {
		if (lstate.stream) {
			// Try to get next chunk from current stream
			auto chunk = lstate.stream->GetNextChunk();
			if (chunk->arrow_array.release && chunk->arrow_array.length > 0) {
				// Convert Arrow → DataChunk using ArrowScanLocalState
				auto output_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, NumericCast<idx_t>(chunk->arrow_array.length));

				// ArrowScanLocalState needs unique_ptr, but GetNextChunk returns shared_ptr
				// Move the ArrowArray into a new unique_ptr wrapper
				auto owned_chunk = make_uniq<ArrowArrayWrapper>();
				owned_chunk->arrow_array = chunk->arrow_array;
				chunk->arrow_array.release = nullptr; // prevent double-free

				ArrowScanLocalState arrow_lstate(std::move(owned_chunk), context);
				// Set column_ids to identity mapping (no projection pushdown)
				for (idx_t i = 0; i < bind_data.arrow_table.GetColumns().size(); i++) {
					arrow_lstate.column_ids.push_back(i);
				}

				output.SetCardinality(output_size);
				ArrowTableFunction::ArrowToDuckDB(arrow_lstate, bind_data.arrow_table.GetColumns(), output,
				                                  false /* arrow_scan_is_projected */);
				output.Verify();
				return;
			}
			// Stream exhausted
			lstate.stream.reset();
		}

		// Grab next task
		auto idx = gstate.next_task_idx.fetch_add(1);
		if (idx >= gstate.total_tasks) {
			lstate.exhausted = true;
			output.SetCardinality(0);
			return;
		}

		// Produce new stream
		auto &pickled = bind_data.pickled_tasks[idx];
		auto stream_wrapper = make_uniq<ArrowArrayStreamWrapper>();
		RequireProduceStream(bind_data.produce_stream)(pickled.c_str(), pickled.size(),
		                                               &stream_wrapper->arrow_array_stream);
		lstate.stream = std::move(stream_wrapper);
	}

	output.SetCardinality(0);
}

// ── Serialize/Deserialize ──────────────────────────────────────────

static void DataSourceScanSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data_p,
                                    const TableFunction &function) {
	auto &bind_data = bind_data_p->Cast<DataSourceScanBindData>();
	serializer.WriteProperty(100, "pickled_tasks", bind_data.pickled_tasks);
	serializer.WriteProperty(101, "pickled_source", bind_data.pickled_source);
}

static unique_ptr<FunctionData> DataSourceScanDeserialize(Deserializer &deserializer, TableFunction &function) {
	auto result = make_uniq<DataSourceScanBindData>();
	result->pickled_tasks = deserializer.ReadProperty<vector<string>>(100, "pickled_tasks");
	result->pickled_source = deserializer.ReadProperty<string>(101, "pickled_source");
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

void DataSourceScanFunction::SetGlobalWorkerSourceCallback(datasource_set_worker_source_t callback) {
	g_global_worker_source_callback.store(callback);
}

void DataSourceScanFunction::SetGlobalGetSchema(datasource_get_schema_t callback) {
	g_global_get_schema.store(callback);
}

} // namespace duckdb
