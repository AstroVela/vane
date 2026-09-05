// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "media_backend.hpp"
#include "datasource_function.hpp"
#include "vane_python/datasource_execution_context.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"

#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/types/uuid.hpp"
#include "vane_python/pybind11/pybind_wrapper.hpp"
#include "vane_python/pyconnection/pyconnection.hpp"
#include "vane_python/python_dependency.hpp"

#include "duckdb/common/exception.hpp"

#include <algorithm>
#include <condition_variable>
#include <unordered_set>

namespace py = pybind11;

namespace duckdb {

PythonDataSourceExecutionContext::PythonDataSourceExecutionContext(shared_ptr<ClientContext> context_p)
    : context(std::move(context_p)) {
	if (!context) {
		throw InvalidInputException("DataSource execution context must not be NULL");
	}
}

void PythonDataSourceExecutionContext::Initialize(py::module_ &m) {
	py::class_<PythonDataSourceExecutionContext, shared_ptr<PythonDataSourceExecutionContext>>(
	    m, "_DataSourceExecutionContext", py::module_local(), py::is_final())
	    .def("_check_interrupted", &PythonDataSourceExecutionContext::CheckInterrupted);
}

void PythonDataSourceExecutionContext::CheckInterrupted() const {
	shared_ptr<ClientContext> active_context;
	auto context_guard = LockContext(active_context);
	if (active_context->IsInterrupted()) {
		throw InterruptException();
	}
}

void PythonDataSourceExecutionContext::Invalidate() {
	active.store(false, std::memory_order_release);
	std::lock_guard<std::mutex> guard(context_lock);
	context.reset();
}

std::unique_lock<std::mutex>
PythonDataSourceExecutionContext::LockContext(shared_ptr<ClientContext> &active_context) const {
	if (!active.load(std::memory_order_acquire)) {
		throw InvalidInputException("DataSource execution context is no longer active");
	}
	std::unique_lock<std::mutex> guard(context_lock);
	if (!active.load(std::memory_order_relaxed) || !context) {
		throw InvalidInputException("DataSource execution context is no longer active");
	}
	active_context = context;
	return guard;
}

namespace {

struct DataSourceArrowStreamState {
	DataSourceArrowStreamState(ArrowArrayStream stream_p,
	                           shared_ptr<PythonDataSourceExecutionContext> execution_context_p)
	    : inner(stream_p), execution_context(std::move(execution_context_p)) {
	}

	ArrowArrayStream inner;
	shared_ptr<PythonDataSourceExecutionContext> execution_context;
};

static DataSourceArrowStreamState &GetDataSourceArrowStreamState(ArrowArrayStream *stream) {
	D_ASSERT(stream);
	D_ASSERT(stream->private_data);
	return *reinterpret_cast<DataSourceArrowStreamState *>(stream->private_data);
}

static int DataSourceArrowStreamGetSchema(ArrowArrayStream *stream, ArrowSchema *out) {
	auto &state = GetDataSourceArrowStreamState(stream);
	return state.inner.get_schema(&state.inner, out);
}

static int DataSourceArrowStreamGetNext(ArrowArrayStream *stream, ArrowArray *out) {
	auto &state = GetDataSourceArrowStreamState(stream);
	return state.inner.get_next(&state.inner, out);
}

static const char *DataSourceArrowStreamGetLastError(ArrowArrayStream *stream) {
	auto &state = GetDataSourceArrowStreamState(stream);
	if (!state.inner.get_last_error) {
		return "DataSource Arrow stream did not provide error detail";
	}
	return state.inner.get_last_error(&state.inner);
}

static void DataSourceArrowStreamRelease(ArrowArrayStream *stream) {
	if (!stream || !stream->release) {
		return;
	}
	auto state =
	    unique_ptr<DataSourceArrowStreamState>(reinterpret_cast<DataSourceArrowStreamState *>(stream->private_data));
	stream->release = nullptr;
	stream->private_data = nullptr;
	state->execution_context->Invalidate();
	if (state->inner.release) {
		state->inner.release(&state->inner);
	}
}

static void TieExecutionContextToArrowStream(ArrowArrayStream *stream,
                                             shared_ptr<PythonDataSourceExecutionContext> execution_context) {
	if (!stream || !stream->release || !stream->get_schema || !stream->get_next) {
		throw InvalidInputException("DataSource task did not export a valid Arrow stream");
	}
	auto state = make_uniq<DataSourceArrowStreamState>(*stream, std::move(execution_context));
	stream->get_schema = DataSourceArrowStreamGetSchema;
	stream->get_next = DataSourceArrowStreamGetNext;
	stream->get_last_error = DataSourceArrowStreamGetLastError;
	stream->release = DataSourceArrowStreamRelease;
	stream->private_data = state.release();
}

} // namespace

// Helper: extract raw bytes from py::bytes without UTF-8 decode
static string PyBytesToString(const py::object &obj) {
	char *buf = nullptr;
	Py_ssize_t len = 0;
	if (PyBytes_AsStringAndSize(obj.ptr(), &buf, &len) != 0) {
		throw py::error_already_set();
	}
	return string(buf, static_cast<size_t>(len));
}

// Every serialized datasource payload starts with a versioned, process-
// independent source identity. Both source and task blobs carry the same ID.
static constexpr char DATASOURCE_PAYLOAD_MAGIC[] = {'V', 'D', 'S', '1'};
static constexpr idx_t DATASOURCE_PAYLOAD_MAGIC_SIZE = sizeof(DATASOURCE_PAYLOAD_MAGIC);
static constexpr idx_t DATASOURCE_SOURCE_ID_SIZE = BaseUUID::STRING_SIZE;
static constexpr idx_t DATASOURCE_PAYLOAD_HEADER_SIZE = DATASOURCE_PAYLOAD_MAGIC_SIZE + DATASOURCE_SOURCE_ID_SIZE;

struct ParsedDataSourcePayload {
	string source_id;
	const char *payload;
	idx_t payload_len;
};

static ParsedDataSourcePayload ParseDataSourcePayload(const char *data, idx_t len, const char *kind) {
	if (len <= DATASOURCE_PAYLOAD_HEADER_SIZE) {
		throw InvalidInputException("Serialized datasource %s is too short", kind);
	}
	if (memcmp(data, DATASOURCE_PAYLOAD_MAGIC, DATASOURCE_PAYLOAD_MAGIC_SIZE) != 0) {
		throw InvalidInputException("Serialized datasource %s has an invalid payload header", kind);
	}
	string source_id(data + DATASOURCE_PAYLOAD_MAGIC_SIZE, DATASOURCE_SOURCE_ID_SIZE);
	hugeint_t parsed_uuid;
	if (!UUID::FromString(source_id, parsed_uuid, true)) {
		throw InvalidInputException("Serialized datasource %s has an invalid source ID", kind);
	}
	return {std::move(source_id), data + DATASOURCE_PAYLOAD_HEADER_SIZE, len - DATASOURCE_PAYLOAD_HEADER_SIZE};
}

static string EncodeDataSourcePayload(const string &source_id, const string &payload) {
	string result;
	result.reserve(DATASOURCE_PAYLOAD_HEADER_SIZE + payload.size());
	result.append(DATASOURCE_PAYLOAD_MAGIC, DATASOURCE_PAYLOAD_MAGIC_SIZE);
	result.append(source_id);
	result.append(payload);
	return result;
}

static py::object LoadArrowSchema(const ParsedDataSourcePayload &source) {
	auto cloudpickle = py::module::import("cloudpickle");
	auto source_package = cloudpickle.attr("loads")(py::bytes(source.payload, source.payload_len));
	if (!py::isinstance<py::tuple>(source_package)) {
		throw InvalidInputException("Serialized datasource source package must be a tuple");
	}
	auto package_tuple = py::reinterpret_borrow<py::tuple>(source_package);
	if (package_tuple.size() != 2) {
		throw InvalidInputException(
		    "Serialized datasource source package must contain source identity bytes and Arrow schema");
	}
	if (!py::isinstance<py::bytes>(package_tuple[0])) {
		throw InvalidInputException("Serialized datasource source identity must be bytes");
	}
	return py::reinterpret_borrow<py::object>(package_tuple[1]);
}

// ── Process-local registry ─────────────────────────────────────────
// Entries are keyed by the stable source UUID embedded in the distributed
// plan. Local scans are reference-counted per global state; distributed scans
// are owned idempotently by their logical query until worker query teardown.
struct DataSourceFactoryRegistryEntry {
	explicit DataSourceFactoryRegistryEntry(string source)
	    : pickled_source(std::move(source)), loading(true), local_owner_count(0) {
	}

	string pickled_source;
	shared_ptr<DataSourceStreamFactory> factory;
	std::condition_variable ready;
	string load_error;
	bool loading;
	idx_t local_owner_count;
	std::unordered_set<string> query_owners;
};

static std::mutex g_factory_mutex;
static std::unordered_map<string, shared_ptr<DataSourceFactoryRegistryEntry>> g_factory_registry;
static std::atomic<idx_t> g_factory_creation_count {0};
static string g_last_created_source_id;

static void VerifyMatchingSource(const DataSourceFactoryRegistryEntry &entry, const char *pickled_source,
                                 idx_t pickled_len, const string &source_id) {
	if (entry.pickled_source.size() == pickled_len &&
	    memcmp(entry.pickled_source.data(), pickled_source, pickled_len) == 0) {
		return;
	}
	throw InvalidInputException("Datasource source ID collision for '%s'", source_id);
}

static void FailFactoryLoad(const string &source_id, const shared_ptr<DataSourceFactoryRegistryEntry> &entry,
                            const string &error) {
	{
		std::lock_guard<std::mutex> lock(g_factory_mutex);
		entry->load_error = error;
		entry->loading = false;
		auto current = g_factory_registry.find(source_id);
		if (current != g_factory_registry.end() && current->second == entry) {
			g_factory_registry.erase(current);
		}
	}
	entry->ready.notify_all();
}

static void AddFactoryOwner(DataSourceFactoryRegistryEntry &entry, const string &query_id) {
	if (query_id.empty()) {
		entry.local_owner_count++;
	} else {
		entry.query_owners.insert(query_id);
	}
}

static idx_t FactoryOwnerCount(const DataSourceFactoryRegistryEntry &entry) {
	return entry.local_owner_count + entry.query_owners.size();
}

static bool HasFactoryOwners(const DataSourceFactoryRegistryEntry &entry) {
	return entry.local_owner_count > 0 || !entry.query_owners.empty();
}

// ── ProduceStream ──────────────────────────────────────────────────
// C callback called by datasource_scan GetData/InitLocal from pipeline threads.
// pickled_task blob layout: [magic/version][source UUID][pickled task bytes]

void DataSourceStreamFactory::ProduceStream(const char *pickled_task, idx_t pickled_len, ArrowArrayStream *out_stream,
                                            ClientContext *context) {
	if (!context) {
		throw InvalidInputException("datasource_scan requires the current query execution context");
	}
	auto task = ParseDataSourcePayload(pickled_task, pickled_len, "task");
	PythonGILWrapper acquire;

	shared_ptr<DataSourceStreamFactory> factory;
	{
		std::lock_guard<std::mutex> lock(g_factory_mutex);
		auto entry = g_factory_registry.find(task.source_id);
		if (entry != g_factory_registry.end() && !entry->second->loading && HasFactoryOwners(*entry->second)) {
			factory = entry->second->factory;
		}
	}
	if (!factory) {
		throw InvalidInputException("Datasource source '%s' is not acquired for this query", task.source_id);
	}

	// 1. Unpickle the task
	auto cloudpickle = py::module::import("cloudpickle");
	auto task_obj = cloudpickle.attr("loads")(py::bytes(task.payload, task.payload_len));

	// 2. Execute through the internal context-aware hook. Ordinary DataSource
	// tasks ignore this token, while governed readers use the exact worker
	// query context for filesystem, Secret, and cancellation resolution.
	auto execution_context = make_shared_ptr<PythonDataSourceExecutionContext>(context->shared_from_this());
	try {
		auto generator = task_obj.attr("_execute_with_context")(execution_context);

		// 3. Wrap in RecordBatchReader
		auto pa = py::module::import("pyarrow");
		auto reader = pa.attr("RecordBatchReader").attr("from_batches")(factory->arrow_schema, generator);

		// 4. Export to C ArrowArrayStream. The forwarding release callback
		// invalidates the query-context capability before stream teardown returns.
		reader.attr("_export_to_c")(reinterpret_cast<uintptr_t>(out_stream));
		TieExecutionContextToArrowStream(out_stream, execution_context);
	} catch (...) {
		execution_context->Invalidate();
		if (out_stream && out_stream->release) {
			out_stream->release(out_stream);
		}
		throw;
	}
}

// ── GetSchema ──────────────────────────────────────────────────────
// C callback called by datasource_scan Bind to get the Arrow schema.
// pickled_source blob layout: [magic/version][source UUID][pickled source bytes]

void DataSourceStreamFactory::GetSchema(const char *pickled_source, idx_t pickled_len, ArrowSchema *out_schema) {
	auto source = ParseDataSourcePayload(pickled_source, pickled_len, "source");
	PythonGILWrapper acquire;

	shared_ptr<DataSourceFactoryRegistryEntry> entry;
	{
		std::lock_guard<std::mutex> lock(g_factory_mutex);
		auto current = g_factory_registry.find(source.source_id);
		if (current != g_factory_registry.end()) {
			VerifyMatchingSource(*current->second, pickled_source, pickled_len, source.source_id);
			entry = current->second;
		}
	}

	if (entry) {
		py::gil_scoped_release release;
		std::unique_lock<std::mutex> lock(g_factory_mutex);
		entry->ready.wait(lock, [&entry] { return !entry->loading; });
	}

	shared_ptr<DataSourceStreamFactory> factory;
	if (entry) {
		std::lock_guard<std::mutex> lock(g_factory_mutex);
		if (!entry->load_error.empty()) {
			throw InvalidInputException("Failed to initialize datasource source '%s': %s", source.source_id,
			                            entry->load_error);
		}
		factory = entry->factory;
	}
	auto arrow_schema = factory ? factory->arrow_schema : LoadArrowSchema(source);
	arrow_schema.attr("_export_to_c")(reinterpret_cast<uintptr_t>(out_schema));
}

// ── Source ownership ───────────────────────────────────────────────

void DataSourceStreamFactory::AcquireSource(const char *pickled_source, idx_t pickled_len, const char *query_id,
                                            idx_t query_id_len) {
	auto source = ParseDataSourcePayload(pickled_source, pickled_len, "source");
	string query_owner(query_id, query_id_len);
	PythonGILWrapper acquire;

	shared_ptr<DataSourceFactoryRegistryEntry> entry;
	bool load_factory = false;
	{
		std::lock_guard<std::mutex> lock(g_factory_mutex);
		auto current = g_factory_registry.find(source.source_id);
		if (current == g_factory_registry.end()) {
			entry = make_shared_ptr<DataSourceFactoryRegistryEntry>(string(pickled_source, pickled_len));
			g_factory_registry.emplace(source.source_id, entry);
			load_factory = true;
		} else {
			entry = current->second;
			VerifyMatchingSource(*entry, pickled_source, pickled_len, source.source_id);
			if (!entry->loading) {
				if (!entry->factory) {
					throw InternalException("Datasource source '%s' has no registered factory", source.source_id);
				}
				AddFactoryOwner(*entry, query_owner);
				return;
			}
		}
	}

	if (!load_factory) {
		{
			// Schema deserialization may temporarily release the GIL. Do not
			// hold it while another query initializes the same logical source.
			py::gil_scoped_release release;
			std::unique_lock<std::mutex> lock(g_factory_mutex);
			entry->ready.wait(lock, [&entry] { return !entry->loading; });
		}
		std::lock_guard<std::mutex> lock(g_factory_mutex);
		if (!entry->load_error.empty()) {
			throw InvalidInputException("Failed to initialize datasource source '%s': %s", source.source_id,
			                            entry->load_error);
		}
		auto current = g_factory_registry.find(source.source_id);
		if (current == g_factory_registry.end() || current->second != entry || !entry->factory) {
			throw InternalException("Datasource source '%s' disappeared during initialization", source.source_id);
		}
		AddFactoryOwner(*entry, query_owner);
		return;
	}

	try {
		auto arrow_schema = LoadArrowSchema(source);
		auto factory = make_shared_ptr<DataSourceStreamFactory>(std::move(arrow_schema));
		{
			std::lock_guard<std::mutex> lock(g_factory_mutex);
			auto current = g_factory_registry.find(source.source_id);
			if (current == g_factory_registry.end() || current->second != entry) {
				throw InternalException("Datasource source '%s' disappeared during initialization", source.source_id);
			}
			entry->factory = std::move(factory);
			AddFactoryOwner(*entry, query_owner);
			entry->loading = false;
			g_factory_creation_count.fetch_add(1);
		}
		entry->ready.notify_all();
	} catch (std::exception &ex) {
		FailFactoryLoad(source.source_id, entry, ex.what());
		throw;
	} catch (...) {
		FailFactoryLoad(source.source_id, entry, "unknown factory initialization error");
		throw;
	}
}

void DataSourceStreamFactory::ReleaseSource(const char *pickled_source, idx_t pickled_len) noexcept {
	try {
		auto source = ParseDataSourcePayload(pickled_source, pickled_len, "source");
		if (!Py_IsInitialized() || PythonIsFinalizing()) {
			return;
		}
		PythonGILWrapper acquire;
		shared_ptr<DataSourceFactoryRegistryEntry> released_entry;
		{
			std::lock_guard<std::mutex> lock(g_factory_mutex);
			auto current = g_factory_registry.find(source.source_id);
			if (current == g_factory_registry.end()) {
				return;
			}
			auto &entry = *current->second;
			VerifyMatchingSource(entry, pickled_source, pickled_len, source.source_id);
			if (entry.loading || entry.local_owner_count == 0) {
				return;
			}
			entry.local_owner_count--;
			if (!HasFactoryOwners(entry)) {
				released_entry = std::move(current->second);
				g_factory_registry.erase(current);
			}
		}
		// released_entry is destroyed before the GIL wrapper.
	} catch (...) {
		// Called from a global-state destructor: release must never throw.
	}
}

// ── FromDataSource ─────────────────────────────────────────────────
// Creates a DuckDB Relation backed by datasource_scan.
// Called from Python: con.from_datasource(source)

unique_ptr<DuckDBPyRelation> DuckDBPyConnection::FromDataSource(py::object &source) {
	auto &connection = con.GetConnection();

	// Only the built-in video source opts into native scan dispatch. Other
	// DataSource implementations keep their own task and schema contracts.
	auto video_source = py::module_::import("vane.datasource.video_reader").attr("VideoFrameSource");
	if (py::isinstance(source, video_source) && MediaBackend::UseNative(*connection.context, "video")) {
		if (!py::type::of(source).is(video_source)) {
			throw InvalidInputException("native video supports only the built-in VideoFrameSource, not subclasses");
		}
		return TableFunction("native_video_tensor_frames", source.attr("_native_parameters")());
	}

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

	// 3. Assign one stable logical source ID. It is serialized with the plan
	// and is therefore identical in the driver and every worker process.
	auto source_id = UUID::ToString(UUID::GenerateRandomUUID());

	// 4. Build pickled_tasks list: each blob = [version][source ID][task]
	vector<Value> task_values;
	for (auto &pickled_task : pickled_tasks) {
		auto prefixed = EncodeDataSourcePayload(source_id, pickled_task);
		task_values.push_back(Value::BLOB(const_data_ptr_cast(prefixed.data()), prefixed.size()));
	}
	auto task_list = Value::LIST(LogicalType::BLOB, std::move(task_values));

	// 5. Build pickled_source:
	// [version][source ID][pickled (pickled DataSource, Arrow schema)].
	// The source bytes remain part of the collision-checked identity payload,
	// while workers can load the schema without instantiating the DataSource or
	// invoking source.schema again.
	auto pickled_source_identity = cloudpickle.attr("dumps")(source);
	auto source_package = py::make_tuple(pickled_source_identity, arrow_schema);
	auto pickled_source_obj = PyBytesToString(cloudpickle.attr("dumps")(source_package));
	auto pickled_source_prefixed = EncodeDataSourcePayload(source_id, pickled_source_obj);

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
	auto aliased_rel = rel->Alias(name);
	auto dependency = make_uniq<ExternalDependency>();
	dependency->AddDependency("datasource", PythonDependencyItem::Create(source));
	aliased_rel->AddExternalDependency(std::move(dependency));
	{
		std::lock_guard<std::mutex> lock(g_factory_mutex);
		g_last_created_source_id = source_id;
	}

	return CreateRelation(std::move(aliased_rel));
}

void ClearDataSourceFactoryRegistry() {
	if (!Py_IsInitialized() || PythonIsFinalizing()) {
		return;
	}
	PythonGILWrapper acquire;
	std::unordered_map<string, shared_ptr<DataSourceFactoryRegistryEntry>> released_entries;
	{
		std::lock_guard<std::mutex> lock(g_factory_mutex);
		released_entries.swap(g_factory_registry);
		g_last_created_source_id.clear();
	}
	// released_entries is destroyed before the GIL wrapper.
}

idx_t ReleaseDataSourceFactoriesForQuery(const string &query_id) {
	if (!Py_IsInitialized() || PythonIsFinalizing()) {
		return 0;
	}
	if (query_id.empty()) {
		throw InvalidInputException("Datasource distributed query ID must not be empty");
	}
	PythonGILWrapper acquire;
	vector<shared_ptr<DataSourceFactoryRegistryEntry>> released_entries;
	idx_t released_owners = 0;
	{
		std::lock_guard<std::mutex> lock(g_factory_mutex);
		released_entries.reserve(g_factory_registry.size());
		for (auto entry = g_factory_registry.begin(); entry != g_factory_registry.end();) {
			if (entry->second->query_owners.erase(query_id) == 0) {
				entry++;
				continue;
			}
			released_owners++;
			if (HasFactoryOwners(*entry->second)) {
				entry++;
				continue;
			}
			released_entries.push_back(std::move(entry->second));
			entry = g_factory_registry.erase(entry);
		}
	}
	// released_entries is destroyed before the GIL wrapper.
	return released_owners;
}

py::dict DataSourceFactoryRegistryStateForTest() {
	D_ASSERT(py::gil_check());
	vector<string> source_ids;
	std::unordered_set<string> query_id_set;
	idx_t owner_count = 0;
	idx_t local_owner_count = 0;
	idx_t query_owner_count = 0;
	idx_t factory_count = 0;
	idx_t registry_size = 0;
	string last_created_source_id;
	{
		std::lock_guard<std::mutex> lock(g_factory_mutex);
		registry_size = g_factory_registry.size();
		source_ids.reserve(registry_size);
		for (auto &item : g_factory_registry) {
			source_ids.push_back(item.first);
			owner_count += FactoryOwnerCount(*item.second);
			local_owner_count += item.second->local_owner_count;
			query_owner_count += item.second->query_owners.size();
			query_id_set.insert(item.second->query_owners.begin(), item.second->query_owners.end());
			factory_count += item.second->factory ? 1 : 0;
		}
		last_created_source_id = g_last_created_source_id;
	}
	std::sort(source_ids.begin(), source_ids.end());

	py::list py_source_ids;
	for (auto &source_id : source_ids) {
		py_source_ids.append(source_id);
	}
	vector<string> query_ids(query_id_set.begin(), query_id_set.end());
	std::sort(query_ids.begin(), query_ids.end());
	py::list py_query_ids;
	for (auto &query_id : query_ids) {
		py_query_ids.append(query_id);
	}
	py::dict result;
	result["registry_size"] = registry_size;
	result["factory_count"] = factory_count;
	result["owner_count"] = owner_count;
	result["local_owner_count"] = local_owner_count;
	result["query_owner_count"] = query_owner_count;
	result["factory_creation_count"] = g_factory_creation_count.load();
	result["source_ids"] = std::move(py_source_ids);
	result["query_ids"] = std::move(py_query_ids);
	result["last_created_source_id"] = std::move(last_created_source_id);
	return result;
}

// ── RegisterDataSourceGlobalCallbacks ──────────────────────────────
// Call once from Python module init to register the static callbacks.

void RegisterDataSourceGlobalCallbacks() {
	DataSourceScanFunction::SetGlobalProduceStream(&DataSourceStreamFactory::ProduceStream);
	DataSourceScanFunction::SetGlobalAcquireSource(&DataSourceStreamFactory::AcquireSource);
	DataSourceScanFunction::SetGlobalReleaseSource(&DataSourceStreamFactory::ReleaseSource);
	DataSourceScanFunction::SetGlobalGetSchema(&DataSourceStreamFactory::GetSchema);
}

} // namespace duckdb
