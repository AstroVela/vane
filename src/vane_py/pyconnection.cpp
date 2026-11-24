// SPDX-FileCopyrightText: 2018-2026 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#include "vane_python/pyconnection/pyconnection.hpp"
#include "datasource_function.hpp"
#include "vane_python/vllm_executor.hpp"
#include "vane_python/udf_executor.hpp"
#include "duckdb/execution/execution_runtime.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/execution/distributed/client_state.hpp"
#include "duckdb/function/scalar/vllm_functions.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/db_instance_cache.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/statement/load_statement.hpp"
#include "duckdb/planner/operator/logical_data_sink.hpp"
#include "vane_python/ai_sql_functions.hpp"
#include "vane_python/audio_file_functions.hpp"
#include "vane_python/image_file_functions.hpp"
#include "vane_python/image_functions.hpp"
#include "vane_python/jupyter_progress_bar_display.hpp"
#include "vane_python/pandas/pandas_scan.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"
#include "vane_python/python_input_callback.hpp"
#include "vane_python/python_replacement_scan.hpp"
#include "vane_python/python_udf_actor_resources.hpp"
#include "vane_python/video_file_functions.hpp"

namespace duckdb {

DefaultConnectionHolder DuckDBPyConnection::default_connection;                        // NOLINT: allow global
DBInstanceCache instance_cache;                                                        // NOLINT: allow global
shared_ptr<PythonImportCache> DuckDBPyConnection::import_cache = nullptr;              // NOLINT: allow global
PythonEnvironmentType DuckDBPyConnection::environment = PythonEnvironmentType::NORMAL; // NOLINT: allow global
std::string DuckDBPyConnection::formatted_python_version = "";

DuckDBPyConnection::~DuckDBPyConnection() {
	if (!PythonIsFinalizing()) {
		try {
			PythonGILWrapper gil;
			con.SetResult(nullptr);
			ReleaseVaneSession();
		} catch (...) { // NOLINT
		}
	}
	try {
		py::gil_scoped_release gil;
		// Release any structures that do not need to hold the GIL here
		con.SetDatabase(nullptr);
		con.SetConnection(nullptr);
	} catch (...) { // NOLINT
	}
}

void DuckDBPyConnection::DetectEnvironment() {
	// Get the formatted Python version
	py::module_ sys = py::module_::import("sys");
	py::object version_info = sys.attr("version_info");
	int major = py::cast<int>(version_info.attr("major"));
	int minor = py::cast<int>(version_info.attr("minor"));
	DuckDBPyConnection::formatted_python_version = std::to_string(major) + "." + std::to_string(minor);

	// If __main__ does not have a __file__ attribute, we are in interactive mode
	auto main_module = py::module_::import("__main__");
	if (py::hasattr(main_module, "__file__")) {
		return;
	}
	DuckDBPyConnection::environment = PythonEnvironmentType::INTERACTIVE;
	if (!ModuleIsLoaded<IpythonCacheItem>()) {
		return;
	}

	// Check to see if we are in a Jupyter Notebook
	auto &import_cache_py = *DuckDBPyConnection::ImportCache();
	auto get_ipython = import_cache_py.IPython.get_ipython();
	if (get_ipython.ptr() == nullptr) {
		// Could either not load the IPython module, or it has no 'get_ipython' attribute
		return;
	}
	auto ipython = get_ipython();
	if (!py::hasattr(ipython, "config")) {
		return;
	}
	py::dict ipython_config = ipython.attr("config");
	if (ipython_config.contains("IPKernelApp")) {
		DuckDBPyConnection::environment = PythonEnvironmentType::JUPYTER;
	}
	return;
}

bool DuckDBPyConnection::DetectAndGetEnvironment() {
	DuckDBPyConnection::DetectEnvironment();
	return DuckDBPyConnection::IsInteractive();
}

bool DuckDBPyConnection::IsJupyter() {
	return DuckDBPyConnection::environment == PythonEnvironmentType::JUPYTER;
}

std::string DuckDBPyConnection::FormattedPythonVersion() {
	return DuckDBPyConnection::formatted_python_version;
}

void DuckDBPyConnection::Close() {
	D_ASSERT(py::gil_check());
	CheckCallbackEntry();
	CheckLocalQueryCloseReentrancy();
	local_query_closing = true;
	if (!local_query_stream.is_none()) {
		auto result = local_query_stream();
		if (!result.is_none()) {
			result.attr("_close_stream")();
		}
	}
	if (vane_session && !vane_session->local_query_runtime.is_none()) {
		if (vane_session_owner) {
			{
				lock_guard<mutex> guard(vane_session->lock);
				vane_session->local_runtime_closing = true;
			}
			vane_session->local_query_runtime.attr("drain")();
		}
		// A queued request must be cancelled before waiting for its execution
		// lock. Running requests keep their owners until native cleanup ends.
		if (!local_query_request.is_none()) {
			local_query_request.attr("cancel")();
		}
	}
	case_insensitive_map_t<unique_ptr<ExternalDependency>> closing_functions;
	try {
		{
			unique_lock<std::recursive_mutex> lock;
			{
				py::gil_scoped_release release;
				lock = LockConnection(py_connection_lock);
			}
			con.SetResult(nullptr);
			{
				py::gil_scoped_release release;
				con.SetConnection(nullptr);
				con.SetDatabase(nullptr);
				registered_function_catalog_types.clear();
			}
			// Children retain their session references until their own close completes.
			ReleaseVaneSession();
			// Keep registered callables alive while children finish, including when
			// a child's initializer reenters Close on this parent.
			closing_functions.swap(registered_functions);
		}
		// Child connections acquire their own execution locks. Do not hold this
		// connection's lock while waiting for a child's initializer to finish.
		py::gil_scoped_release release;
		// https://peps.python.org/pep-0249/#Connection.close
		cursors.ClearCursors();
		closing_functions.clear();
	} catch (...) {
		// A later Close must retain the dependencies of children still awaiting cleanup.
		py::gil_scoped_release release;
		auto lock = LockConnection(py_connection_lock);
		for (auto &entry : closing_functions) {
			registered_functions.emplace(entry.first, std::move(entry.second));
		}
		throw;
	}
}

void DuckDBPyConnection::Interrupt() {
	D_ASSERT(py::gil_check());
	auto context = con.GetConnection().context;
	auto request = local_query_request;
	// Python cancellation can yield after the request retires. Fence that
	// entire operation, retaining both its request and native context.
	interrupts_in_progress.fetch_add(1);
	try {
		if (!request.is_none()) {
			// The request's fenced callback interrupts only its own execution.
			// An unconditional native interrupt afterward could hit a reused cursor.
			request.attr("cancel")();
		} else {
			context->Interrupt();
			CancelPythonUDFActorResourcePreparation(*context);
		}
	} catch (...) {
		interrupt_generation.fetch_add(1);
		interrupts_in_progress.fetch_sub(1);
		throw;
	}
	interrupt_generation.fetch_add(1);
	interrupts_in_progress.fetch_sub(1);
}

uint64_t DuckDBPyConnection::InterruptGeneration() const {
	return interrupt_generation.load();
}

bool DuckDBPyConnection::InterruptInProgress() const {
	return interrupts_in_progress.load() != 0;
}

py::object DuckDBPyConnection::CreateQueryInterruptCheck() {
	// A connection-owned result must not retain its connection through this callback.
	weak_ptr<DuckDBPyConnection> owner(shared_from_this());
	auto generation = InterruptGeneration();
	return py::cpp_function([owner, generation]() {
		auto connection = owner.lock();
		if (!connection) {
			throw ConnectionException("Connection already closed!");
		}
		if (connection->InterruptInProgress() || connection->InterruptGeneration() != generation) {
			throw InterruptException();
		}
	});
}

double DuckDBPyConnection::QueryProgress() {
	auto &connection = con.GetConnection();
	return connection.GetQueryProgress();
}

void DuckDBPyConnection::InstallExtension(const string &extension, bool force_install, const py::object &repository,
                                          const py::object &repository_url, const py::object &version) {
	auto query_lock = LockForQuery();
	PythonInputCallbackScope callback(nullptr);
	auto &connection = con.GetConnection();

	auto install_statement = make_uniq<LoadStatement>();
	install_statement->info = make_uniq<LoadInfo>();
	auto &info = *install_statement->info;

	info.filename = extension;

	const bool has_repository = !py::none().is(repository);
	const bool has_repository_url = !py::none().is(repository_url);
	if (has_repository && has_repository_url) {
		throw InvalidInputException(
		    "Both 'repository' and 'repository_url' are set which is not allowed, please pick one or the other");
	}
	string repository_string;
	if (has_repository) {
		repository_string = py::str(repository);
	} else if (has_repository_url) {
		repository_string = py::str(repository_url);
	}

	if ((has_repository || has_repository_url) && repository_string.empty()) {
		throw InvalidInputException("The provided 'repository' or 'repository_url' can not be empty!");
	}

	string version_string;
	if (!py::none().is(version)) {
		version_string = py::str(version);
		if (version_string.empty()) {
			throw InvalidInputException("The provided 'version' can not be empty!");
		}
	}

	info.repository = repository_string;
	info.repo_is_alias = repository_string.empty() ? false : has_repository;
	info.version = version_string;
	info.load_type = force_install ? LoadType::FORCE_INSTALL : LoadType::INSTALL;
	auto res = connection.Query(std::move(install_statement));
	if (res->HasError()) {
		res->ThrowError();
	}
}

void DuckDBPyConnection::LoadExtension(const string &extension) {
	auto query_lock = LockForQuery();
	auto &connection = con.GetConnection();
	ExtensionHelper::LoadExternalExtension(*connection.context, extension);
}

shared_ptr<DuckDBPyConnection> DefaultConnectionHolder::Get() {
	lock_guard<mutex> guard(l);
	if (!connection || connection->con.ConnectionIsClosed()) {
		py::dict config_dict;
		connection = DuckDBPyConnection::Connect(py::str(":memory:"), false, config_dict);
	}
	return connection;
}

shared_ptr<DuckDBPyConnection> DefaultConnectionHolder::GetIfOpen() {
	lock_guard<mutex> guard(l);
	return connection && !connection->con.ConnectionIsClosed() ? connection : nullptr;
}

void DefaultConnectionHolder::Set(shared_ptr<DuckDBPyConnection> conn) {
	lock_guard<mutex> guard(l);
	connection = conn;
}

static constexpr const char *WEAK_CONNECTION_OWNER = "vane.weak_connection_owner";

py::object DuckDBPyConnection::CreateWeakOwner(const shared_ptr<DuckDBPyConnection> &connection) {
	// The native default connection can outlive all of its Python wrappers.
	// Keep the weak reference on the native owner so wrapper recreation is safe.
	auto weak_connection = make_uniq<weak_ptr<DuckDBPyConnection>>(connection);
	auto owner = py::capsule(weak_connection.get(), WEAK_CONNECTION_OWNER,
	                         [](void *ptr) { delete static_cast<weak_ptr<DuckDBPyConnection> *>(ptr); });
	weak_connection.release();
	return owner;
}

py::object DuckDBPyConnection::ResolveOwner(const py::object &owner) {
	if (!owner) {
		return py::none();
	}
	if (!PyCapsule_IsValid(owner.ptr(), WEAK_CONNECTION_OWNER)) {
		return owner;
	}
	auto weak_connection = py::reinterpret_borrow<py::capsule>(owner).get_pointer<weak_ptr<DuckDBPyConnection>>();
	auto connection = weak_connection->lock();
	return connection ? py::cast(std::move(connection)) : py::none();
}

void DuckDBPyConnection::Cursors::AddCursor(shared_ptr<DuckDBPyConnection> conn) {
	lock_guard<mutex> l(lock);

	// Clean up previously created cursors
	vector<weak_ptr<DuckDBPyConnection>> compacted_cursors;
	bool needs_compaction = false;
	for (auto &cur_p : cursors) {
		auto cur = cur_p.lock();
		if (!cur) {
			needs_compaction = true;
			continue;
		}
		compacted_cursors.push_back(cur_p);
	}
	if (needs_compaction) {
		cursors = std::move(compacted_cursors);
	}

	cursors.push_back(conn);
}

void DuckDBPyConnection::Cursors::ClearCursors() {
	vector<weak_ptr<DuckDBPyConnection>> closing_cursors;
	{
		lock_guard<mutex> l(lock);
		closing_cursors.swap(cursors);
	}

	// A child's initializer may reenter Close on its parent. Detach the list
	// before waiting for children so that reentry does not wait on this list.
	try {
		for (auto &cur : closing_cursors) {
			auto cursor = cur.lock();
			if (!cursor) {
				// The cursor has already been closed
				continue;
			}
			// Close releases the GIL while waiting for the child's connection lock.
			PythonGILWrapper gil;
			cursor->Close();
			// Ensure destructor runs with gil if triggered.
			cursor.reset();
		}
	} catch (...) {
		// Preserve the parent's ability to retry a child's failed session cleanup.
		lock_guard<mutex> l(lock);
		cursors.insert(cursors.end(), closing_cursors.begin(), closing_cursors.end());
		throw;
	}
}

void DuckDBPyConnection::Cursors::CheckLocalQueryCloseReentrancy() {
	vector<shared_ptr<DuckDBPyConnection>> children;
	{
		lock_guard<mutex> guard(lock);
		for (auto &entry : cursors) {
			auto child = entry.lock();
			if (child) {
				children.push_back(std::move(child));
			}
		}
	}
	for (auto &child : children) {
		child->CheckLocalQueryCloseReentrancy();
	}
}

shared_ptr<DuckDBPyConnection> DuckDBPyConnection::Cursor() {
	auto query_lock = LockForQuery();
	auto res = make_shared_ptr<DuckDBPyConnection>();
	res->con.SetDatabase(con);
	res->con.SetConnection(make_uniq<Connection>(res->con.GetDatabase()));
	RunnerClientState::Initialize(*res->con.GetConnection().context, GetRunnerType());
	res->SetConnectionBootstrapConfig(connection_database, connection_read_only, connection_config);
	res->InheritVaneSession(*this);
	res->distributed_python_udf_registrations = distributed_python_udf_registrations;
	res->applied_distributed_python_udf_digests = applied_distributed_python_udf_digests;
	cursors.AddCursor(res);
	return res;
}

case_insensitive_map_t<Value> TransformPyConfigDict(const py::dict &py_config_dict) {
	case_insensitive_map_t<Value> config_dict;
	for (auto &kv : py_config_dict) {
		auto key = py::str(kv.first);
		auto val = py::str(kv.second);
		config_dict[key] = Value(val);
	}
	return config_dict;
}

static py::dict CopyPyDict(const py::dict &source) {
	py::dict result;
	for (auto &kv : source) {
		result[kv.first] = kv.second;
	}
	return result;
}

void DuckDBPyConnection::SetConnectionBootstrapConfig(const string &database, bool read_only, const py::dict &config) {
	connection_database = database;
	connection_read_only = read_only;
	connection_config = CopyPyDict(config);
}

py::dict DuckDBPyConnection::ExportConnectionBootstrapConfig() const {
	py::dict result;
	result[py::str("database")] = py::str(connection_database);
	result[py::str("read_only")] = py::bool_(connection_read_only);
	result[py::str("config")] = CopyPyDict(connection_config);
	return result;
}

static bool IsVaneSessionEnvironmentKey(const string &name) {
	return StringUtil::StartsWith(name, "AWS_") || StringUtil::StartsWith(name, "DUCKDB_") ||
	       StringUtil::StartsWith(name, "S3FS_") || StringUtil::StartsWith(name, "VANE_");
}

void DuckDBPyConnection::InitializeVaneSession() {
	auto uuid_module = py::module_::import("uuid");
	auto session_id = py::str(uuid_module.attr("uuid4")().attr("hex")).cast<string>();
	if (session_id.empty()) {
		throw InternalException("Failed to generate Vane connection session identity");
	}

	py::dict captured;
	auto process_environment = py::module_::import("os").attr("environ");
	auto environ_dict = py::module_::import("builtins").attr("dict")(process_environment).cast<py::dict>();
	for (auto item : environ_dict) {
		auto key = py::str(item.first).cast<string>();
		if (!IsVaneSessionEnvironmentKey(key)) {
			continue;
		}
		captured[py::str(key)] = py::str(item.second);
	}
	captured[py::str("VANE_RUNNER")] = py::str(GetRunnerType());
	vane_session = make_shared_ptr<VaneSessionContext>(std::move(session_id), std::move(captured));
	vane_session_attached = true;
	vane_session_owner = true;
}

string DuckDBPyConnection::GetRunnerType() const {
	return RunnerClientState::Get(*con.GetConnection().context);
}

void DuckDBPyConnection::InheritVaneSession(const DuckDBPyConnection &owner) {
	if (!owner.vane_session || !owner.vane_session_attached) {
		throw InternalException("Cannot inherit missing Vane connection session");
	}
	vane_session = owner.vane_session;
	{
		lock_guard<mutex> guard(vane_session->lock);
		if (vane_session->connection_count == 0 || vane_session->local_runtime_closing) {
			throw InternalException("Cannot inherit closed Vane connection session");
		}
		if (!vane_session->local_query_runtime.is_none()) {
			EnableLocalRuntimeInputPolicy(*con.GetConnection().context);
		}
		vane_session->connection_count++;
	}
	vane_session_attached = true;
	vane_session_owner = false;
}

py::object DuckDBPyConnection::ConfigureLocalRuntime(const py::kwargs &options) {
	auto query_lock = LockForQuery();
	unique_lock<std::recursive_mutex> execution_lock;
	{
		py::gil_scoped_release release;
		execution_lock = LockConnection(py_connection_lock);
	}
	auto &context = *con.GetConnection().context;
	if (GetRunnerType() != "local") {
		throw InvalidInputException("configure_local_runtime requires a local connection");
	}
	if (context.transaction.HasActiveTransaction()) {
		throw InvalidInputException("configure_local_runtime requires an idle auto-commit connection");
	}
	{
		lock_guard<mutex> guard(vane_session->lock);
		if (local_query_closing || !vane_session_attached || vane_session->local_runtime_closing ||
		    !vane_session_owner || vane_session->connection_count != 1 ||
		    !vane_session->local_query_runtime.is_none()) {
			throw InvalidInputException(
			    "configure_local_runtime must run once on the session owner before creating cursors");
		}
	}
	py::dict config(options);
	if (config.contains("session_id") || config.contains("session_config") || config.contains("_connection")) {
		throw InvalidInputException("local runtime session identity is owned by the connection");
	}
	config["session_id"] = py::str(GetVaneSessionId());
	config["session_config"] = ExportVaneSessionConfig();
	config["_connection"] = py::cast(shared_from_this());
	py::object runtime;
	{
		// Configuration can iterate application-supplied device inventories.
		// Reject callback entry before it can wait on another connection while
		// this connection's locks remain held.
		PythonInputCallbackScope callback(con.GetConnection().context);
		runtime = py::module_::import("vane.execution.local_query").attr("LocalQueryRuntime")(**config);
	}
	{
		lock_guard<mutex> guard(vane_session->lock);
		if (local_query_closing || !vane_session_attached || vane_session->local_runtime_closing ||
		    vane_session->connection_count != 1 || !vane_session->local_query_runtime.is_none()) {
			throw InvalidInputException("Vane session changed while configuring its local runtime");
		}
		EnableLocalRuntimeInputPolicy(context);
		vane_session->local_query_runtime = runtime;
	}
	return runtime;
}

py::object DuckDBPyConnection::GetLocalQueryRuntime() const {
	if (!vane_session || !vane_session_attached) {
		return py::none();
	}
	lock_guard<mutex> guard(vane_session->lock);
	return vane_session->local_query_runtime;
}

unique_lock<std::recursive_mutex> DuckDBPyConnection::LockForQuery() const {
	CheckCallbackEntry();
	D_ASSERT(py::gil_check());
	// Check before taking either connection lock. Python input callbacks can
	// run on another thread while their caller holds those locks, so checking
	// only the owning thread would still deadlock. Concurrent queries use cursors.
	if (!local_query_request.is_none()) {
		throw InvalidInputException("local runtime does not support reentrant queries on the same cursor");
	}
	const auto generation = InterruptGeneration();
	const auto interrupting = InterruptInProgress();
	auto lock = LockConnection(py_connection_lock);
	// Entry guards may wait before the query-specific interrupt check is made.
	// Preserve cancellation accepted during that wait.
	if (interrupting || InterruptInProgress() || InterruptGeneration() != generation) {
		throw InterruptException();
	}
	return lock;
}

void DuckDBPyConnection::CheckCallbackEntry() {
	// Callback code must never wait on a connection, even an idle sibling: a
	// nested query may dispatch workers that need the input invoking this callback.
	// This scope exists before open/metadata callbacks too, without a file handle.
	if (PythonInputCallbackScope::IsActive()) {
		throw InvalidInputException("cannot call connection APIs reentrantly from a Python input callback");
	}
}

unique_lock<std::recursive_mutex> DuckDBPyConnection::LockConnection(const shared_ptr<std::recursive_mutex> &mutex) {
	CheckCallbackEntry();
	unique_lock<std::recursive_mutex> lock(*mutex, std::defer_lock);
	if (py::gil_check()) {
		py::gil_scoped_release release;
		lock.lock();
	} else {
		lock.lock();
	}
	return lock;
}

void DuckDBPyConnection::CheckLocalQueryCloseReentrancy() {
	D_ASSERT(py::gil_check());
	if (!local_query_request.is_none() && (local_query_thread == std::this_thread::get_id() ||
	                                       PythonInputCallbackScope::Contains(*con.GetConnection().context))) {
		throw InvalidInputException("cannot close a cursor reentrantly during its local runtime query");
	}
	// Closing an owner also waits for its children. Reject before draining or
	// detaching any of them when the caller is a child's runtime initializer.
	cursors.CheckLocalQueryCloseReentrancy();
}

const string &DuckDBPyConnection::GetVaneSessionId() const {
	if (!vane_session || vane_session->id.empty()) {
		throw InternalException("DuckDB connection is missing its Vane session identity");
	}
	return vane_session->id;
}

py::dict DuckDBPyConnection::ExportVaneSessionConfig() const {
	if (!vane_session) {
		throw InternalException("DuckDB connection is missing its Vane session configuration");
	}
	lock_guard<mutex> guard(vane_session->lock);
	if (!vane_session_attached || vane_session->connection_count == 0) {
		throw InternalException("DuckDB connection Vane session is closed");
	}
	return CopyPyDict(vane_session->config);
}

void DuckDBPyConnection::MarkVaneRaySessionOpened() {
	if (!vane_session) {
		throw InternalException("DuckDB connection is missing its Vane session identity");
	}
	lock_guard<mutex> guard(vane_session->lock);
	if (!vane_session_attached || vane_session->connection_count == 0) {
		throw InternalException("DuckDB connection Vane session is closed");
	}
	vane_session->ray_session_opened = true;
}

bool DuckDBPyConnection::CompareAndRecordDynamicExtensionSnapshotEntry(const vector<string> &expected_entries,
                                                                       const string &entry) {
	if (entry.empty()) {
		throw InvalidInputException("Dynamic extension snapshot entry must not be empty");
	}
	if (!vane_session) {
		throw InternalException("DuckDB connection is missing its Vane session identity");
	}
	lock_guard<mutex> guard(vane_session->lock);
	if (!vane_session_attached || vane_session->connection_count == 0) {
		throw InternalException("DuckDB connection Vane session is closed");
	}
	if (vane_session->dynamic_extension_snapshot_entries != expected_entries) {
		return false;
	}
	for (const auto &existing : vane_session->dynamic_extension_snapshot_entries) {
		if (existing == entry) {
			return true;
		}
	}
	vane_session->dynamic_extension_snapshot_entries.push_back(entry);
	return true;
}

vector<string> DuckDBPyConnection::ExportDynamicExtensionSnapshotEntries() const {
	if (!vane_session) {
		throw InternalException("DuckDB connection is missing its Vane session identity");
	}
	lock_guard<mutex> guard(vane_session->lock);
	if (!vane_session_attached || vane_session->connection_count == 0) {
		throw InternalException("DuckDB connection Vane session is closed");
	}
	return vane_session->dynamic_extension_snapshot_entries;
}

void DuckDBPyConnection::ReleaseVaneSession() {
	if (!vane_session_attached) {
		return;
	}
	if (!vane_session) {
		throw InternalException("DuckDB connection is missing its attached Vane session");
	}
	unique_lock<mutex> guard(vane_session->lock);
	if (vane_session->connection_count == 0) {
		throw InternalException("DuckDB connection Vane session reference count underflow");
	}
	if (vane_session->connection_count > 1) {
		vane_session->connection_count--;
		vane_session_attached = false;
		return;
	}
	if (!vane_session->local_query_runtime.is_none() && !PythonIsFinalizing()) {
		vane_session->local_runtime_closing = true;
		auto runtime = vane_session->local_query_runtime;
		guard.unlock();
		// Failed cleanup keeps the final session attachment and runtime for retry.
		runtime.attr("close")();
		guard.lock();
		vane_session->local_query_runtime = py::none();
	}
	if (vane_session->ray_session_opened && !vane_session->id.empty() && !PythonIsFinalizing()) {
		py::module_::import("vane.runners.ray.runner").attr("notify_connection_closed")(py::str(vane_session->id));
		vane_session->ray_session_opened = false;
	}
	vane_session->connection_count = 0;
	vane_session->config = py::dict();
	vane_session->dynamic_extension_snapshot_entries.clear();
	vane_session_attached = false;
}

static bool HasJupyterProgressBarDependencies() {
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	if (!import_cache.ipywidgets()) {
		// ipywidgets not installed, needed to support the progress bar
		return false;
	}
	return true;
}

static void SetDefaultConfigArguments(ClientContext &context) {
	if (!DuckDBPyConnection::IsInteractive()) {
		// Don't need to set any special default arguments
		return;
	}

	auto &config = ClientConfig::GetConfig(context);
	config.enable_progress_bar = true;

	if (!DuckDBPyConnection::IsJupyter()) {
		return;
	}
	if (!HasJupyterProgressBarDependencies()) {
		// Disable progress bar altogether
		config.system_progress_bar_disable_reason =
		    "required package 'ipywidgets' is missing, which is needed to render progress bars in Jupyter";
		config.enable_progress_bar = false;
		return;
	}

	// Set the function used to create the display for the progress bar
	context.config.display_create_func = JupyterProgressBarDisplay::Create;
}

void InstantiateNewInstance(DuckDB &db) {
	auto &db_instance = *db.instance;
	PandasScanFunction scan_fun;

	TableFunctionSet scan_set(scan_fun.name);
	scan_set.AddFunction(std::move(scan_fun));
	CreateTableFunctionInfo scan_info(std::move(scan_set));
	scan_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;

	auto &system_catalog = Catalog::GetSystemCatalog(db_instance);
	auto transaction = CatalogTransaction::GetSystemTransaction(db_instance);

	system_catalog.CreateFunction(transaction, scan_info);

	auto image_file_set = ImageFileFunctions::GetFunctions();
	CreateScalarFunctionInfo image_file_info(std::move(image_file_set));
	image_file_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	system_catalog.CreateFunction(transaction, image_file_info);

	auto decode_image_file_set = ImageFileFunctions::GetDecodeFunctions();
	CreateScalarFunctionInfo decode_image_file_info(std::move(decode_image_file_set));
	decode_image_file_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	system_catalog.CreateFunction(transaction, decode_image_file_info);

	for (auto functions : {ImageFunctions::GetCropFunctions(), ImageFunctions::GetEncodeFunctions(),
	                       ImageFunctions::GetResizeFunctions(), ImageFunctions::GetConvertFunctions(),
	                       ImageFunctions::GetDecodeFunctions(), ImageFunctions::GetHashFunctions()}) {
		CreateScalarFunctionInfo info(std::move(functions));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		system_catalog.CreateFunction(transaction, info);
	}

	for (auto &macro : ImageFunctions::GetMacros()) {
		macro->on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		system_catalog.CreateFunction(transaction, *macro);
	}

	auto audio_file_set = AudioFileFunctions::GetFunctions();
	CreateScalarFunctionInfo audio_file_info(std::move(audio_file_set));
	audio_file_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	system_catalog.CreateFunction(transaction, audio_file_info);

	auto audio_resample_set = AudioFileFunctions::GetResampleFunctions();
	CreateScalarFunctionInfo audio_resample_info(std::move(audio_resample_set));
	audio_resample_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	system_catalog.CreateFunction(transaction, audio_resample_info);

	auto video_file_set = VideoFileFunctions::GetFunctions();
	CreateScalarFunctionInfo video_file_info(std::move(video_file_set));
	video_file_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	system_catalog.CreateFunction(transaction, video_file_info);
	CreateScalarFunctionInfo video_clip_info(VideoFileFunctions::GetClipFunctions());
	video_clip_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	system_catalog.CreateFunction(transaction, video_clip_info);
	auto video_clip_macro = VideoFileFunctions::GetClipMacro();
	video_clip_macro->on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	system_catalog.CreateFunction(transaction, *video_clip_macro);
	auto read_video_set = VideoFileFunctions::GetReadFunctions();
	CreateTableFunctionInfo read_video_info(std::move(read_video_set));
	read_video_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	system_catalog.CreateFunction(transaction, read_video_info);
	for (auto &functions : VideoFileFunctions::GetFrameFunctions()) {
		CreateScalarFunctionInfo info(std::move(functions));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		system_catalog.CreateFunction(transaction, info);
	}
	for (auto &macro : VideoFileFunctions::GetFrameMacros()) {
		macro->on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		system_catalog.CreateFunction(transaction, *macro);
	}

	auto vllm_set = VLLMFunction::GetFunctions();
	CreateScalarFunctionInfo vllm_info(std::move(vllm_set));
	vllm_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	system_catalog.CreateFunction(transaction, vllm_info);

	auto ai_prompt_pack_set = AISQLFunction::GetPromptPackFunctions();
	CreateScalarFunctionInfo ai_prompt_pack_info(std::move(ai_prompt_pack_set));
	ai_prompt_pack_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	system_catalog.CreateFunction(transaction, ai_prompt_pack_info);

	auto ai_prompt_implementation_set = AISQLFunction::GetPromptImplementationFunctions();
	CreateScalarFunctionInfo ai_prompt_implementation_info(std::move(ai_prompt_implementation_set));
	ai_prompt_implementation_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	system_catalog.CreateFunction(transaction, ai_prompt_implementation_info);

	auto ai_prompt_macro = AISQLFunction::GetPromptMacro();
	ai_prompt_macro->on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	system_catalog.CreateFunction(transaction, *ai_prompt_macro);

	auto ai_embed_implementation_set = AISQLFunction::GetEmbedImplementationFunctions();
	CreateScalarFunctionInfo ai_embed_implementation_info(std::move(ai_embed_implementation_set));
	ai_embed_implementation_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	system_catalog.CreateFunction(transaction, ai_embed_implementation_info);

	auto ai_embed_macro = AISQLFunction::GetEmbedMacro();
	ai_embed_macro->on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	system_catalog.CreateFunction(transaction, *ai_embed_macro);

	auto ai_jev_implementation_set = AISQLFunction::GetJevImplementationFunctions();
	CreateScalarFunctionInfo ai_jev_implementation_info(std::move(ai_jev_implementation_set));
	ai_jev_implementation_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	system_catalog.CreateFunction(transaction, ai_jev_implementation_info);

	auto ai_jev_macro = AISQLFunction::GetJevMacro();
	ai_jev_macro->on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	system_catalog.CreateFunction(transaction, *ai_jev_macro);
}

static shared_ptr<DuckDBPyConnection> FetchOrCreateInstance(const string &database_path, DBConfig &config,
                                                            bool use_instance_cache, const string &runner_type) {
	auto res = make_shared_ptr<DuckDBPyConnection>();
	bool cache_instance = use_instance_cache && database_path != ":memory:" && !database_path.empty();
	config.replacement_scans.emplace_back(PythonReplacementScan::Replace);
	{
		D_ASSERT(py::gil_check());
		py::gil_scoped_release release;
		auto lock = DuckDBPyConnection::LockConnection(res->py_connection_lock);
		auto database =
		    instance_cache.GetOrCreateInstance(database_path, config, cache_instance, InstantiateNewInstance);
		// Install before exposing this connection, including when the database
		// comes from the instance cache. Preserve any later per-database
		// registrations instead of resetting them whenever a connection opens.
		std::call_once(database->instance->GetExecutionRuntime().initialization, [&] {
			RegisterUDFExecutorFactory(*database->instance);
			RegisterVLLMExecutorFactory(*database->instance);
			RegisterDataSourceCallbacks(*database->instance);
			RegisterPythonUDFConnectionCallbacks(*database->instance);
		});
		res->con.SetDatabase(std::move(database));
		res->con.SetConnection(make_uniq<Connection>(res->con.GetDatabase()));
		RunnerClientState::Initialize(*res->con.GetConnection().context, runner_type);
	}
	return res;
}

bool IsDefaultConnectionString(const string &database, bool read_only, case_insensitive_map_t<Value> &config) {
	bool is_default = StringUtil::CIEquals(database, ":default:");
	if (!is_default) {
		return false;
	}
	// Only allow fetching the default connection when no options are passed
	if (read_only == true || !config.empty()) {
		throw InvalidInputException("Default connection fetching is only allowed without additional options");
	}
	return true;
}

static string GetPathString(const py::object &path) {
	auto &import_cache = *DuckDBPyConnection::ImportCache();
	const bool is_path = py::isinstance(path, import_cache.pathlib.Path());
	if (is_path || py::isinstance<py::str>(path)) {
		return std::string(py::str(path));
	}
	string actual_type = py::str(py::type::of(path));
	throw InvalidInputException("Please provide either a str or a pathlib.Path, not %s", actual_type);
}

static shared_ptr<DuckDBPyConnection> ConnectInternal(const py::object &database_p, bool read_only,
                                                      const py::dict &config_options, bool use_instance_cache,
                                                      const string &runner_type = string()) {
	DuckDBPyConnection::CheckCallbackEntry();
	auto config_dict = TransformPyConfigDict(config_options);
	auto database = GetPathString(database_p);
	if (IsDefaultConnectionString(database, read_only, config_dict)) {
		return DuckDBPyConnection::DefaultConnection();
	}
	const auto selected_runner =
	    runner_type.empty() ? ResolveRunnerTypeFromEnvironment() : NormalizeRunnerType(runner_type);

	DBConfig config(read_only);
	OperatorExtension::Register(config, make_shared_ptr<DataSinkOperatorExtension>());
	config.AddExtensionOption("pandas_analyze_sample",
	                          "The maximum number of rows to sample when analyzing a pandas object column.",
	                          LogicalType::UBIGINT, Value::UBIGINT(1000));
	config.AddExtensionOption("python_enable_replacements",
	                          "Whether variables visible to the current stack should be used for replacement scans.",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(true));
	config.AddExtensionOption(
	    "python_scan_all_frames",
	    "If set, restores the old behavior of scanning all preceding frames to locate the referenced variable.",
	    LogicalType::BOOLEAN, Value::BOOLEAN(false));
	if (!DuckDBPyConnection::IsJupyter()) {
		config_dict["duckdb_api"] = Value("python/" + DuckDBPyConnection::FormattedPythonVersion());
	} else {
		config_dict["duckdb_api"] = Value("python/" + DuckDBPyConnection::FormattedPythonVersion() + " jupyter");
	}
	config.SetOptionsByName(config_dict);

	auto res = FetchOrCreateInstance(database, config, use_instance_cache, selected_runner);
	res->SetConnectionBootstrapConfig(database, read_only, config_options);
	res->InitializeVaneSession();
	auto &client_context = *res->con.GetConnection().context;
	SetDefaultConfigArguments(client_context);
	return res;
}

shared_ptr<DuckDBPyConnection> DuckDBPyConnection::Connect(const py::object &database_p, bool read_only,
                                                           const py::dict &config_options) {
	return ConnectInternal(database_p, read_only, config_options, true);
}

shared_ptr<DuckDBPyConnection> DuckDBPyConnection::ConnectUncached(const py::object &database_p, bool read_only,
                                                                   const py::dict &config_options) {
	return ConnectInternal(database_p, read_only, config_options, false);
}

shared_ptr<DuckDBPyConnection> DuckDBPyConnection::ConnectWithRunner(const py::object &database_p, bool read_only,
                                                                     const py::dict &config_options,
                                                                     const string &runner_type,
                                                                     bool use_instance_cache) {
	if (runner_type.empty()) {
		throw InternalException("Internal connection creation requires an explicit runner policy");
	}
	return ConnectInternal(database_p, read_only, config_options, use_instance_cache, runner_type);
}

shared_ptr<DuckDBPyConnection> DuckDBPyConnection::DefaultConnection() {
	return default_connection.Get();
}

void DuckDBPyConnection::SetDefaultConnection(shared_ptr<DuckDBPyConnection> connection) {
	return default_connection.Set(std::move(connection));
}

PythonImportCache *DuckDBPyConnection::ImportCache() {
	if (!import_cache) {
		import_cache = make_shared_ptr<PythonImportCache>();
	}
	return import_cache.get();
}

bool DuckDBPyConnection::IsInteractive() {
	return DuckDBPyConnection::environment != PythonEnvironmentType::NORMAL;
}

shared_ptr<DuckDBPyConnection> DuckDBPyConnection::Enter() {
	return shared_from_this();
}

void DuckDBPyConnection::Exit(DuckDBPyConnection &self, const py::object &exc_type, const py::object &exc,
                              const py::object &traceback) {
	self.Close();
	if (exc_type.ptr() != Py_None) {
		// Propagate the exception if any occurred
		PyErr_SetObject(exc_type.ptr(), exc.ptr());
		throw py::error_already_set();
	}
}

void DuckDBPyConnection::Cleanup() {
	ClearDataSourceFactoryRegistry();
	default_connection.Set(nullptr);
	import_cache.reset();
}

} // namespace duckdb
