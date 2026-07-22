// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "datasource_function.hpp"
#include "duckdb_python/pybind11/gil_wrapper.hpp"

#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb_python/pybind11/pybind_wrapper.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"

namespace py = pybind11;

namespace duckdb {

// Helper: extract raw bytes from py::bytes without UTF-8 decode
static string PyBytesToString(const py::object &obj) {
	char *buf = nullptr;
	Py_ssize_t len = 0;
	PyBytes_AsStringAndSize(obj.ptr(), &buf, &len);
	return string(buf, static_cast<size_t>(len));
}

// ── Global registry ────────────────────────────────────────────────
// Factories must outlive the DuckDB plan execution. We store them in a
// global map keyed by pointer value. The Python read_datasource() function
// also holds a reference to prevent GC on the Python side.
static std::mutex g_factory_mutex;
static std::unordered_map<uintptr_t, shared_ptr<DataSourceStreamFactory>> g_factory_registry;

static void RegisterFactory(shared_ptr<DataSourceStreamFactory> factory) {
	auto key = reinterpret_cast<uintptr_t>(factory.get());
	std::lock_guard<std::mutex> lock(g_factory_mutex);
	g_factory_registry[key] = std::move(factory);
}

void ClearDataSourceFactoryRegistry() {
	std::lock_guard<std::mutex> lock(g_factory_mutex);
	g_factory_registry.clear();
}

// ── ProduceStream ──────────────────────────────────────────────────
// C callback called by datasource_scan GetData/InitLocal from pipeline threads.
// pickled_task blob layout: [factory_ptr: 8 bytes][pickled task bytes]
//
// On a Ray worker the factory_ptr from the driver is invalid. In that case
// we lazily recreate the factory from the pickled DataSource embedded in
// pickled_source (stored in the companion global g_pickled_source_for_worker).

// Global pickled_source for worker-side factory recovery.
// Set by InitGlobal before any GetData call.
static std::mutex g_worker_source_mutex;
static string g_worker_pickled_source;

void DataSourceStreamFactory::SetWorkerPickledSource(const char *pickled_source, idx_t pickled_len) {
	std::lock_guard<std::mutex> lock(g_worker_source_mutex);
	g_worker_pickled_source.assign(pickled_source, pickled_len);
}

void DataSourceStreamFactory::ProduceStream(const char *pickled_task, idx_t pickled_len, ArrowArrayStream *out_stream) {
	if (pickled_len < sizeof(uintptr_t)) {
		throw InternalException("DataSourceStreamFactory::ProduceStream: pickled_task too short");
	}
	uintptr_t factory_ptr;
	memcpy(&factory_ptr, pickled_task, sizeof(uintptr_t));

	DataSourceStreamFactory *factory = nullptr;
	{
		std::lock_guard<std::mutex> lock(g_factory_mutex);
		auto it = g_factory_registry.find(factory_ptr);
		if (it != g_factory_registry.end()) {
			factory = it->second.get();
		}
	}

	const char *task_bytes = pickled_task + sizeof(uintptr_t);
	idx_t task_len = pickled_len - sizeof(uintptr_t);

	PythonGILWrapper acquire;

	// If factory not found (worker process), create one from pickled_source
	if (!factory) {
		string pickled_source_copy;
		{
			std::lock_guard<std::mutex> lock(g_worker_source_mutex);
			pickled_source_copy = g_worker_pickled_source;
		}

		if (pickled_source_copy.size() < sizeof(uintptr_t)) {
			throw InternalException("ProduceStream: no factory and no worker pickled_source available");
		}

		// Extract the pickled DataSource (skip the stale factory_ptr prefix)
		const char *source_bytes = pickled_source_copy.data() + sizeof(uintptr_t);
		idx_t source_len = pickled_source_copy.size() - sizeof(uintptr_t);

		auto cloudpickle = py::module::import("cloudpickle");
		auto source_obj = cloudpickle.attr("loads")(py::bytes(source_bytes, source_len));

		// Reconstruct Arrow schema from the DataSource
		auto schema_dict = py::cast<py::dict>(source_obj.attr("schema"));
		auto ds_module = py::module::import("vane.datasource");
		auto arrow_schema = ds_module.attr("_schema_to_arrow")(schema_dict);

		// Create and register a new factory
		auto new_factory = make_shared_ptr<DataSourceStreamFactory>(source_obj, arrow_schema, vector<string> {});
		uintptr_t new_factory_ptr = reinterpret_cast<uintptr_t>(new_factory.get());
		RegisterFactory(new_factory);
		factory = new_factory.get();

		// Update the worker pickled_source with the new factory pointer
		{
			std::lock_guard<std::mutex> lock(g_worker_source_mutex);
			memcpy(&g_worker_pickled_source[0], &new_factory_ptr, sizeof(uintptr_t));
		}
	}

	// 1. Unpickle the task
	auto cloudpickle = py::module::import("cloudpickle");
	auto task_obj = cloudpickle.attr("loads")(py::bytes(task_bytes, task_len));

	// 2. Call task.execute() to get a generator of RecordBatches
	auto generator = task_obj.attr("execute")();

	// 3. Wrap in RecordBatchReader
	auto pa = py::module::import("pyarrow");
	auto reader = pa.attr("RecordBatchReader").attr("from_batches")(factory->arrow_schema, generator);

	// 4. Export to C ArrowArrayStream
	reader.attr("_export_to_c")(reinterpret_cast<uintptr_t>(out_stream));
}

// ── GetSchema ──────────────────────────────────────────────────────
// C callback called by datasource_scan Bind to get the Arrow schema.
// pickled_source blob layout: [factory_ptr: 8 bytes][pickled DataSource bytes]

void DataSourceStreamFactory::GetSchema(const char *pickled_source, idx_t pickled_len, ArrowSchema *out_schema) {
	if (pickled_len < sizeof(uintptr_t)) {
		throw InternalException("DataSourceStreamFactory::GetSchema: pickled_source too short");
	}
	uintptr_t factory_ptr;
	memcpy(&factory_ptr, pickled_source, sizeof(uintptr_t));

	DataSourceStreamFactory *factory = nullptr;
	{
		std::lock_guard<std::mutex> lock(g_factory_mutex);
		auto it = g_factory_registry.find(factory_ptr);
		if (it != g_factory_registry.end()) {
			factory = it->second.get();
		}
	}

	PythonGILWrapper acquire;

	if (factory) {
		factory->arrow_schema.attr("_export_to_c")(reinterpret_cast<uintptr_t>(out_schema));
	} else {
		// Worker process: reconstruct schema from pickled DataSource
		const char *source_bytes = pickled_source + sizeof(uintptr_t);
		idx_t source_len = pickled_len - sizeof(uintptr_t);

		auto cloudpickle = py::module::import("cloudpickle");
		auto source_obj = cloudpickle.attr("loads")(py::bytes(source_bytes, source_len));

		auto schema_dict = py::cast<py::dict>(source_obj.attr("schema"));
		auto ds_module = py::module::import("vane.datasource");
		auto arrow_schema = ds_module.attr("_schema_to_arrow")(schema_dict);
		arrow_schema.attr("_export_to_c")(reinterpret_cast<uintptr_t>(out_schema));
	}
}

// ── FromDataSource ─────────────────────────────────────────────────
// Creates a DuckDB Relation backed by datasource_scan.
// Called from Python: con.from_datasource(source)

unique_ptr<DuckDBPyRelation> DuckDBPyConnection::FromDataSource(py::object &source) {
	auto &connection = con.GetConnection();

	// 1. Convert DataSource schema (dict[str, str]) to Arrow schema
	auto schema_dict = py::cast<py::dict>(source.attr("schema"));
	auto ds_module = py::module::import("vane.datasource");
	auto arrow_schema = ds_module.attr("_schema_to_arrow")(schema_dict);

	// 2. Get tasks and serialize them for worker processes
	auto cloudpickle = py::module::import("cloudpickle");
	auto tasks = py::list(source.attr("get_tasks")());
	vector<string> pickled_tasks;
	for (auto &task : tasks) {
		auto pickled = PyBytesToString(cloudpickle.attr("dumps")(task));
		pickled_tasks.push_back(pickled);
	}

	if (pickled_tasks.empty()) {
		throw InvalidInputException("DataSource returned no tasks");
	}

	// 3. Create and register the factory
	auto factory = make_shared_ptr<DataSourceStreamFactory>(source, arrow_schema, std::move(pickled_tasks));
	uintptr_t factory_ptr = reinterpret_cast<uintptr_t>(factory.get());
	RegisterFactory(factory);

	// 4. Build pickled_tasks list: each blob = [factory_ptr][pickled_task_bytes]
	vector<Value> task_values;
	for (auto &pt : factory->pickled_tasks) {
		string prefixed;
		prefixed.resize(sizeof(uintptr_t) + pt.size());
		memcpy(&prefixed[0], &factory_ptr, sizeof(uintptr_t));
		memcpy(&prefixed[sizeof(uintptr_t)], pt.data(), pt.size());
		task_values.push_back(Value::BLOB(const_data_ptr_cast(prefixed.data()), prefixed.size()));
	}
	auto task_list = Value::LIST(LogicalType::BLOB, std::move(task_values));

	// 5. Build pickled_source: [factory_ptr][pickled DataSource bytes]
	auto pickled_source_obj = PyBytesToString(cloudpickle.attr("dumps")(source));
	string pickled_source_prefixed;
	pickled_source_prefixed.resize(sizeof(uintptr_t) + pickled_source_obj.size());
	memcpy(&pickled_source_prefixed[0], &factory_ptr, sizeof(uintptr_t));
	memcpy(&pickled_source_prefixed[sizeof(uintptr_t)], pickled_source_obj.data(), pickled_source_obj.size());

	// 6. Build datasource_scan(produce_ptr, get_schema_ptr, pickled_source, pickled_tasks)
	vector<Value> params;
	params.push_back(Value::POINTER(CastPointerToValue(DataSourceStreamFactory::ProduceStream)));
	params.push_back(Value::POINTER(CastPointerToValue(DataSourceStreamFactory::GetSchema)));
	params.push_back(Value::BLOB(const_data_ptr_cast(pickled_source_prefixed.data()), pickled_source_prefixed.size()));
	params.push_back(std::move(task_list));

	string name = "datasource_" + StringUtil::GenerateRandomName();

	// Note: NOT releasing GIL here — TableFunctionRelation constructor calls
	// TryBindRelation() which triggers DataSourceScanBind → GetSchema callback,
	// which needs the GIL to call arrow_schema._export_to_c().
	auto rel = connection.TableFunction("datasource_scan", std::move(params));

	// Return as DuckDBPyRelation — factory stays alive in g_factory_registry
	return make_uniq<DuckDBPyRelation>(rel->Alias(name));
}

// ── RegisterDataSourceGlobalCallbacks ──────────────────────────────
// Call once from Python module init to register the static callbacks.

void RegisterDataSourceGlobalCallbacks() {
	DataSourceScanFunction::SetGlobalProduceStream(&DataSourceStreamFactory::ProduceStream);
	DataSourceScanFunction::SetGlobalWorkerSourceCallback(&DataSourceStreamFactory::SetWorkerPickledSource);
	DataSourceScanFunction::SetGlobalGetSchema(&DataSourceStreamFactory::GetSchema);
}

} // namespace duckdb
