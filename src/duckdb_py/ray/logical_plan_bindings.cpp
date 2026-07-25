// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Included by ray_module.cpp inside namespace duckdb.

struct PyPhysicalPlanWrapper;

struct PyLogicalPlan {
	string query_id_;
	string serialized_logical_plan_;
	duckdb::shared_ptr<duckdb::Relation> relation_;
	py::object udf_registrations_ = py::none();
	py::object connection_snapshot_ = py::none();

	PyLogicalPlan() = default;

	string idx() const {
		return query_id_;
	}

	string session_id() const;
	py::dict session_config() const;

	PyPhysicalPlanWrapper to_physical_plan(py::object conn_obj) const;
};

static string SerializeLogicalPlanFromRelation(const duckdb::shared_ptr<duckdb::Relation> &rel) {
	if (!rel) {
		throw duckdb::InternalException("Relation is null");
	}
	auto client_context = rel->context->GetContext();
	string serialized_plan;
	client_context->RunFunctionInTransaction([&]() {
		auto statement_binder = duckdb::Binder::CreateBinder(*client_context);
		auto relation_stmt = make_uniq<duckdb::RelationStatement>(rel, *statement_binder);
		duckdb::Planner planner(*client_context);
		planner.CreatePlan(std::move(relation_stmt));
		auto logical_plan = std::move(planner.plan);

		// NOTE: We intentionally do NOT run the Optimizer here.
		// The unoptimized (bound) logical plan is serialized and sent to the Driver,
		// where the Optimizer runs. This avoids needing serialization support for
		// custom LogicalOperator types created by optimizer passes
		// (e.g., LogicalUDFProject, LogicalLocalExchange).

		duckdb::MemoryStream stream(duckdb::Allocator::Get(*client_context));
		duckdb::SerializationOptions options;
		options.serialization_compatibility = duckdb::SerializationCompatibility::Latest();
		options.serialize_default_values = true;
		duckdb::BinarySerializer serializer(stream, options);
		serializer.Begin();
		logical_plan->Serialize(serializer);
		serializer.End();

		auto data_ptr = stream.GetData();
		auto data_size = stream.GetPosition();
		if (data_size == 0) {
			throw duckdb::InternalException("Logical plan serialization returned empty payload");
		}
		serialized_plan = string(reinterpret_cast<const char *>(data_ptr), data_size);
	});
	return serialized_plan;
}

static DuckDBPyConnection &ExtractPyConnectionWrapper(py::object conn_obj) {
	if (py::hasattr(conn_obj, "c")) {
		return conn_obj.attr("c").cast<DuckDBPyConnection &>();
	}
	if (py::isinstance<DuckDBPyConnection>(conn_obj)) {
		return conn_obj.cast<DuckDBPyConnection &>();
	}
	throw duckdb::InternalException("Connection object must have 'c' attribute or be a DuckDBPyConnection");
}

static py::dict CopyPyDict(const py::dict &source) {
	py::dict result;
	for (auto item : source) {
		result[item.first] = item.second;
	}
	return result;
}

static py::object LookupBootstrapSnapshot(const py::object &snapshot_obj) {
	if (snapshot_obj.is_none() || !py::isinstance<py::dict>(snapshot_obj)) {
		return py::none();
	}
	auto snapshot = snapshot_obj.cast<py::dict>();
	if (!snapshot.contains(py::str("bootstrap"))) {
		return py::none();
	}
	auto bootstrap_obj = snapshot[py::str("bootstrap")];
	if (bootstrap_obj.is_none() || !py::isinstance<py::dict>(bootstrap_obj)) {
		return py::none();
	}
	return bootstrap_obj;
}

static py::dict LookupVaneSessionSnapshot(const py::object &snapshot_obj) {
	if (snapshot_obj.is_none() || !py::isinstance<py::dict>(snapshot_obj)) {
		throw duckdb::InvalidInputException("Connection snapshot is missing the required Vane session");
	}
	auto snapshot = snapshot_obj.cast<py::dict>();
	if (!snapshot.contains(py::str("vane_session")) || !py::isinstance<py::dict>(snapshot[py::str("vane_session")])) {
		throw duckdb::InvalidInputException("Connection snapshot is missing the required Vane session");
	}
	return snapshot[py::str("vane_session")].cast<py::dict>();
}

static string VaneSessionIdFromSnapshot(const py::object &snapshot_obj) {
	auto session = LookupVaneSessionSnapshot(snapshot_obj);
	if (!session.contains(py::str("id")) || session[py::str("id")].is_none()) {
		throw duckdb::InvalidInputException("Connection snapshot Vane session is missing id");
	}
	auto session_id = py::str(session[py::str("id")]).cast<string>();
	if (session_id.empty()) {
		throw duckdb::InvalidInputException("Connection snapshot Vane session id must not be empty");
	}
	return session_id;
}

static py::dict VaneSessionConfigFromSnapshot(const py::object &snapshot_obj) {
	auto session = LookupVaneSessionSnapshot(snapshot_obj);
	if (!session.contains(py::str("config")) || session[py::str("config")].is_none()) {
		throw duckdb::InvalidInputException("Connection snapshot Vane session is missing config");
	}
	if (!py::isinstance<py::dict>(session[py::str("config")])) {
		throw duckdb::InvalidInputException("Connection snapshot Vane session config must be a dict");
	}
	return CopyPyDict(session[py::str("config")].cast<py::dict>());
}

static bool IsDefaultBootstrapSnapshot(const py::object &bootstrap_obj) {
	if (bootstrap_obj.is_none() || !py::isinstance<py::dict>(bootstrap_obj)) {
		return true;
	}
	auto bootstrap = bootstrap_obj.cast<py::dict>();

	string database = ":memory:";
	if (bootstrap.contains(py::str("database")) && !bootstrap[py::str("database")].is_none()) {
		database = py::str(bootstrap[py::str("database")]).cast<string>();
	}

	bool read_only = false;
	if (bootstrap.contains(py::str("read_only")) && !bootstrap[py::str("read_only")].is_none()) {
		read_only = bootstrap[py::str("read_only")].cast<bool>();
	}

	py::dict config = py::dict();
	if (bootstrap.contains(py::str("config")) && !bootstrap[py::str("config")].is_none() &&
	    py::isinstance<py::dict>(bootstrap[py::str("config")])) {
		config = bootstrap[py::str("config")].cast<py::dict>();
	}
	return database == ":memory:" && !read_only && py::len(config) == 0;
}

static py::object NormalizeBootstrapSnapshot(const py::dict &bootstrap_obj) {
	py::dict bootstrap;
	bootstrap[py::str("database")] =
	    bootstrap_obj.contains(py::str("database")) && !bootstrap_obj[py::str("database")].is_none()
	        ? py::object(py::str(bootstrap_obj[py::str("database")]))
	        : py::object(py::str(":memory:"));
	bootstrap[py::str("read_only")] =
	    bootstrap_obj.contains(py::str("read_only")) && !bootstrap_obj[py::str("read_only")].is_none()
	        ? py::object(py::bool_(bootstrap_obj[py::str("read_only")].cast<bool>()))
	        : py::object(py::bool_(false));
	if (bootstrap_obj.contains(py::str("config")) && !bootstrap_obj[py::str("config")].is_none() &&
	    py::isinstance<py::dict>(bootstrap_obj[py::str("config")])) {
		bootstrap[py::str("config")] = CopyPyDict(bootstrap_obj[py::str("config")].cast<py::dict>());
	} else {
		bootstrap[py::str("config")] = py::dict();
	}
	return bootstrap;
}

static bool PythonObjectsEqual(const py::handle &lhs, const py::handle &rhs) {
	int compare_result = PyObject_RichCompareBool(lhs.ptr(), rhs.ptr(), Py_EQ);
	if (compare_result < 0) {
		throw py::error_already_set();
	}
	return compare_result == 1;
}

static py::object CreateConnectionFromBootstrapSnapshot(const py::object &bootstrap_obj) {
	if (IsDefaultBootstrapSnapshot(bootstrap_obj)) {
		return py::cast(DuckDBPyConnection::Connect(py::str(":memory:"), false, py::dict()));
	}

	auto bootstrap = bootstrap_obj.cast<py::dict>();
	py::object database_obj = py::str(":memory:");
	if (bootstrap.contains(py::str("database")) && !bootstrap[py::str("database")].is_none()) {
		database_obj = py::str(bootstrap[py::str("database")]);
	}

	bool read_only = false;
	if (bootstrap.contains(py::str("read_only")) && !bootstrap[py::str("read_only")].is_none()) {
		read_only = bootstrap[py::str("read_only")].cast<bool>();
	}

	py::dict config = py::dict();
	if (bootstrap.contains(py::str("config")) && !bootstrap[py::str("config")].is_none() &&
	    py::isinstance<py::dict>(bootstrap[py::str("config")])) {
		config = CopyPyDict(bootstrap[py::str("config")].cast<py::dict>());
	}
	return py::cast(DuckDBPyConnection::Connect(database_obj, read_only, config));
}

static bool ConnectionMatchesBootstrapSnapshot(py::object conn_obj, const py::object &snapshot_obj) {
	auto bootstrap_obj = LookupBootstrapSnapshot(snapshot_obj);
	if (bootstrap_obj.is_none() || IsDefaultBootstrapSnapshot(bootstrap_obj) || conn_obj.is_none()) {
		return true;
	}
	auto actual_bootstrap = ExtractPyConnectionWrapper(conn_obj).ExportConnectionBootstrapConfig();
	auto normalized_required = NormalizeBootstrapSnapshot(bootstrap_obj.cast<py::dict>());
	return PythonObjectsEqual(actual_bootstrap, normalized_required);
}

static py::object ResolveConnectionForSnapshot(py::object conn_obj, const py::object &snapshot_obj) {
	auto bootstrap_obj = LookupBootstrapSnapshot(snapshot_obj);
	if (bootstrap_obj.is_none() || IsDefaultBootstrapSnapshot(bootstrap_obj)) {
		return conn_obj;
	}
	if (!conn_obj.is_none() && ConnectionMatchesBootstrapSnapshot(conn_obj, snapshot_obj)) {
		return conn_obj;
	}
	return CreateConnectionFromBootstrapSnapshot(bootstrap_obj);
}

struct QueryPythonReplayState {
	string session_id;
	duckdb::distributed::python::ray::SafePyObject session_config;
	duckdb::distributed::python::ray::SafePyObject udf_registrations;
	duckdb::distributed::python::ray::SafePyObject udf_actor_handles;
	duckdb::distributed::python::ray::SafePyObject connection_snapshot;

	QueryPythonReplayState(string session_id_p, py::object session_config_p, py::object udf_registrations_p,
	                       py::object udf_actor_handles_p, py::object connection_snapshot_p)
	    : session_id(std::move(session_id_p)),
	      session_config(duckdb::distributed::python::ray::SafePyObject(std::move(session_config_p))),
	      udf_registrations(duckdb::distributed::python::ray::SafePyObject(std::move(udf_registrations_p))),
	      udf_actor_handles(duckdb::distributed::python::ray::SafePyObject(std::move(udf_actor_handles_p))),
	      connection_snapshot(duckdb::distributed::python::ray::SafePyObject(std::move(connection_snapshot_p))) {
	}
};

static std::mutex g_query_python_replay_states_lock;
static std::unordered_map<string, std::unique_ptr<QueryPythonReplayState>> g_query_python_replay_states;

struct ConnectionSettingRecord {
	string name;
	string value;
	string input_type;
};

static bool ShouldSkipConnectionSettingSnapshot(const string &name, const string &input_type) {
	auto lower_name = duckdb::StringUtil::Lower(name);
	auto upper_input_type = duckdb::StringUtil::Upper(input_type);
	if (lower_name == "duckdb_api") {
		return true;
	}
	if (upper_input_type.find('[') != string::npos) {
		return true;
	}
	return false;
}

static bool IsBooleanConnectionSettingType(const string &input_type) {
	return duckdb::StringUtil::Upper(input_type) == "BOOLEAN";
}

static bool IsNumericConnectionSettingType(const string &input_type) {
	static const std::unordered_set<string> numeric_types = {
	    "TINYINT",   "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT",
	    "USMALLINT", "UINTEGER", "UBIGINT", "FLOAT",  "DOUBLE",  "DECIMAL",
	};
	return numeric_types.find(duckdb::StringUtil::Upper(input_type)) != numeric_types.end();
}

static string QuoteSQLStringLiteral(const string &value) {
	return "'" + duckdb::StringUtil::Replace(value, "'", "''") + "'";
}

static duckdb::unique_ptr<duckdb::MaterializedQueryResult> ExecuteSnapshotQuery(duckdb::Connection &conn,
                                                                                const string &sql) {
	auto result = conn.Query(sql);
	if (!result) {
		throw duckdb::InternalException("Snapshot query returned null result: " + sql);
	}
	if (result->HasError()) {
		throw duckdb::InvalidInputException("Snapshot query failed for SQL '" + sql + "': " + result->GetError());
	}
	return result;
}

static string StripS3EndpointSchemeForDuckDB(const string &endpoint_url) {
	auto scheme_pos = endpoint_url.find("://");
	if (scheme_pos == string::npos) {
		return endpoint_url;
	}
	return endpoint_url.substr(scheme_pos + 3);
}

static bool S3EndpointUsesSSL(const string &endpoint_url) {
	return duckdb::StringUtil::StartsWith(endpoint_url, "https://");
}

static void ConfigureConnectionForS3Endpoint(duckdb::Connection &conn, const string &endpoint_url,
                                             const string &access_key, const string &secret_key, const string &region) {
	ExecuteSnapshotQuery(conn, "LOAD httpfs");

	const auto endpoint = StripS3EndpointSchemeForDuckDB(endpoint_url);
	const auto use_ssl = S3EndpointUsesSSL(endpoint_url);
	const auto resolved_region = region.empty() ? string("us-east-1") : region;

	ExecuteSnapshotQuery(conn, "SET GLOBAL s3_region=" + QuoteSQLStringLiteral(resolved_region));
	ExecuteSnapshotQuery(conn, "SET GLOBAL s3_access_key_id=" + QuoteSQLStringLiteral(access_key));
	ExecuteSnapshotQuery(conn, "SET GLOBAL s3_secret_access_key=" + QuoteSQLStringLiteral(secret_key));
	ExecuteSnapshotQuery(conn, "SET GLOBAL s3_endpoint=" + QuoteSQLStringLiteral(endpoint));
	ExecuteSnapshotQuery(conn, string("SET GLOBAL s3_use_ssl=") + (use_ssl ? "true" : "false"));
	ExecuteSnapshotQuery(conn, "SET GLOBAL s3_url_style='path'");
	ExecuteSnapshotQuery(conn, "SET GLOBAL http_keep_alive=true");
	ExecuteSnapshotQuery(conn, "SET GLOBAL http_retries=10");
	ExecuteSnapshotQuery(conn, "SET GLOBAL http_retry_wait_ms=100");
	ExecuteSnapshotQuery(conn, "SET GLOBAL http_retry_backoff=1.5");
	ExecuteSnapshotQuery(conn, "CREATE SECRET IF NOT EXISTS __vane_s3_test ("
	                           "TYPE S3, "
	                           "KEY_ID " +
	                               QuoteSQLStringLiteral(access_key) +
	                               ", "
	                               "SECRET " +
	                               QuoteSQLStringLiteral(secret_key) +
	                               ", "
	                               "ENDPOINT " +
	                               QuoteSQLStringLiteral(endpoint) +
	                               ", "
	                               "REGION " +
	                               QuoteSQLStringLiteral(resolved_region) +
	                               ", "
	                               "USE_SSL " +
	                               string(use_ssl ? "true" : "false") +
	                               ", "
	                               "URL_STYLE 'path')");
}

static string VaneSessionConfigValue(const py::dict &config, const char *key) {
	auto py_key = py::str(key);
	if (!config.contains(py_key) || config[py_key].is_none()) {
		return string();
	}
	return py::str(config[py_key]).cast<string>();
}

static void ApplyVaneSessionConfig(duckdb::Connection &conn, const py::object &snapshot_obj) {
	auto config = VaneSessionConfigFromSnapshot(snapshot_obj);
	auto endpoint_url = VaneSessionConfigValue(config, "AWS_ENDPOINT_URL");
	auto access_key = VaneSessionConfigValue(config, "AWS_ACCESS_KEY_ID");
	auto secret_key = VaneSessionConfigValue(config, "AWS_SECRET_ACCESS_KEY");
	auto session_token = VaneSessionConfigValue(config, "AWS_SESSION_TOKEN");
	auto region = VaneSessionConfigValue(config, "AWS_REGION");
	if (region.empty()) {
		region = VaneSessionConfigValue(config, "AWS_DEFAULT_REGION");
	}
	if (endpoint_url.empty() && access_key.empty() && secret_key.empty() && session_token.empty() && region.empty()) {
		return;
	}

	ExecuteSnapshotQuery(conn, "LOAD httpfs");
	if (!region.empty()) {
		ExecuteSnapshotQuery(conn, "SET s3_region=" + QuoteSQLStringLiteral(region));
	}
	if (!access_key.empty()) {
		ExecuteSnapshotQuery(conn, "SET s3_access_key_id=" + QuoteSQLStringLiteral(access_key));
	}
	if (!secret_key.empty()) {
		ExecuteSnapshotQuery(conn, "SET s3_secret_access_key=" + QuoteSQLStringLiteral(secret_key));
	}
	if (!session_token.empty()) {
		ExecuteSnapshotQuery(conn, "SET s3_session_token=" + QuoteSQLStringLiteral(session_token));
	}
	if (!endpoint_url.empty()) {
		ExecuteSnapshotQuery(conn,
		                     "SET s3_endpoint=" + QuoteSQLStringLiteral(StripS3EndpointSchemeForDuckDB(endpoint_url)));
		ExecuteSnapshotQuery(conn, string("SET s3_use_ssl=") + (S3EndpointUsesSSL(endpoint_url) ? "true" : "false"));
		ExecuteSnapshotQuery(conn, "SET s3_url_style='path'");
	}
	ExecuteSnapshotQuery(conn, "SET http_keep_alive=true");
	ExecuteSnapshotQuery(conn, "SET http_retries=10");
	ExecuteSnapshotQuery(conn, "SET http_retry_wait_ms=100");
	ExecuteSnapshotQuery(conn, "SET http_retry_backoff=1.5");
}

static std::vector<string> QueryLoadedExtensionNames(DuckDBPyConnection &conn_wrapper) {
	std::vector<string> extensions;
	auto result =
	    ExecuteSnapshotQuery(conn_wrapper.con.GetConnection(), "SELECT extension_name "
	                                                           "FROM duckdb_extensions() "
	                                                           "WHERE loaded AND install_mode <> 'STATICALLY_LINKED' "
	                                                           "ORDER BY extension_name");
	auto &collection = result->Collection();
	extensions.reserve(collection.Count());
	for (auto &row : collection.Rows()) {
		auto value = row.GetValue(0);
		if (value.IsNull()) {
			continue;
		}
		auto extension_name = value.ToString();
		if (!extension_name.empty()) {
			extensions.push_back(std::move(extension_name));
		}
	}
	return extensions;
}

static std::vector<ConnectionSettingRecord> QueryConnectionSettings(DuckDBPyConnection &conn_wrapper) {
	std::vector<ConnectionSettingRecord> settings;
	auto result = ExecuteSnapshotQuery(conn_wrapper.con.GetConnection(), "SELECT name, value, input_type "
	                                                                     "FROM duckdb_settings() "
	                                                                     "ORDER BY name");
	auto &collection = result->Collection();
	settings.reserve(collection.Count());
	for (auto &row : collection.Rows()) {
		ConnectionSettingRecord record;
		auto name_val = row.GetValue(0);
		auto value_val = row.GetValue(1);
		auto input_type_val = row.GetValue(2);
		if (name_val.IsNull() || input_type_val.IsNull()) {
			continue;
		}
		record.name = name_val.ToString();
		record.value = value_val.IsNull() ? string() : value_val.ToString();
		record.input_type = input_type_val.ToString();
		settings.push_back(std::move(record));
	}
	return settings;
}

static void TryLoadSnapshotExtension(DuckDBPyConnection &conn_wrapper, const string &extension_name,
                                     bool allow_install) {
	auto &conn = conn_wrapper.con.GetConnection();
	try {
		ExecuteSnapshotQuery(conn, "LOAD " + extension_name);
		return;
	} catch (...) {
	}
	if (!allow_install) {
		return;
	}
	ExecuteSnapshotQuery(conn, "INSTALL " + extension_name);
	ExecuteSnapshotQuery(conn, "LOAD " + extension_name);
}

static bool VaneRaySessionLifecycleEnabled() {
	auto environ = py::module_::import("os").attr("environ");
	auto runner = py::str(environ.attr("get")(py::str("VANE_RUNNER"), py::str(""))).cast<string>();
	duckdb::StringUtil::Trim(runner);
	runner = duckdb::StringUtil::Lower(runner);
	return runner.empty() || runner == "ray";
}

static py::object CaptureConnectionSnapshot(DuckDBPyConnection &conn_wrapper) {
	if (VaneRaySessionLifecycleEnabled()) {
		conn_wrapper.MarkVaneRaySessionOpened();
	}
	auto bootstrap_obj = conn_wrapper.ExportConnectionBootstrapConfig();
	auto loaded_extensions = QueryLoadedExtensionNames(conn_wrapper);
	auto source_settings = QueryConnectionSettings(conn_wrapper);

	auto default_conn_obj = CreateConnectionFromBootstrapSnapshot(bootstrap_obj);
	auto &default_conn = ExtractPyConnectionWrapper(default_conn_obj);
	for (const auto &extension_name : loaded_extensions) {
		TryLoadSnapshotExtension(default_conn, extension_name, false);
	}
	auto default_settings = QueryConnectionSettings(default_conn);
	std::unordered_map<string, string> default_setting_values;
	default_setting_values.reserve(default_settings.size());
	for (const auto &record : default_settings) {
		default_setting_values[duckdb::StringUtil::Lower(record.name)] = record.value;
	}

	py::list extensions_obj;
	for (const auto &extension_name : loaded_extensions) {
		extensions_obj.append(py::str(extension_name));
	}

	py::list settings_obj;
	for (const auto &record : source_settings) {
		if (ShouldSkipConnectionSettingSnapshot(record.name, record.input_type)) {
			continue;
		}
		auto lower_name = duckdb::StringUtil::Lower(record.name);
		auto entry = default_setting_values.find(lower_name);
		if (entry != default_setting_values.end() && entry->second == record.value) {
			continue;
		}
		py::dict setting_obj;
		setting_obj[py::str("name")] = py::str(record.name);
		setting_obj[py::str("value")] = py::str(record.value);
		setting_obj[py::str("input_type")] = py::str(record.input_type);
		settings_obj.append(std::move(setting_obj));
	}

	bool has_bootstrap = !IsDefaultBootstrapSnapshot(bootstrap_obj);
	py::dict snapshot_obj;
	py::dict session_obj;
	session_obj[py::str("id")] = py::str(conn_wrapper.GetVaneSessionId());
	session_obj[py::str("config")] = conn_wrapper.ExportVaneSessionConfig();
	snapshot_obj[py::str("vane_session")] = std::move(session_obj);
	if (has_bootstrap) {
		snapshot_obj[py::str("bootstrap")] = NormalizeBootstrapSnapshot(bootstrap_obj);
	}
	snapshot_obj[py::str("extensions")] = std::move(extensions_obj);
	snapshot_obj[py::str("settings")] = std::move(settings_obj);
	return snapshot_obj;
}

static void ApplyConnectionSnapshot(py::object conn_obj, const py::object &snapshot_obj) {
	if (snapshot_obj.is_none()) {
		return;
	}
	if (!py::isinstance<py::dict>(snapshot_obj)) {
		throw duckdb::InvalidInputException("Connection snapshot must be a dict");
	}

	auto &conn_wrapper = ExtractPyConnectionWrapper(conn_obj);
	auto snapshot = snapshot_obj.cast<py::dict>();
	if (snapshot.contains(py::str("extensions"))) {
		auto extensions_obj = snapshot[py::str("extensions")];
		if (!extensions_obj.is_none() && py::isinstance<py::list>(extensions_obj)) {
			for (auto item : extensions_obj.cast<py::list>()) {
				auto extension_name = py::str(item).cast<string>();
				if (extension_name.empty()) {
					continue;
				}
				TryLoadSnapshotExtension(conn_wrapper, extension_name, true);
			}
		}
	}
	ApplyVaneSessionConfig(conn_wrapper.con.GetConnection(), snapshot_obj);

	if (!snapshot.contains(py::str("settings"))) {
		return;
	}
	auto settings_obj = snapshot[py::str("settings")];
	if (settings_obj.is_none() || !py::isinstance<py::list>(settings_obj)) {
		return;
	}

	for (auto item : settings_obj.cast<py::list>()) {
		if (!py::isinstance<py::dict>(item)) {
			continue;
		}
		auto setting_obj = py::reinterpret_borrow<py::dict>(item);
		if (!setting_obj.contains(py::str("name")) || !setting_obj.contains(py::str("value"))) {
			continue;
		}
		auto setting_name = py::str(setting_obj[py::str("name")]).cast<string>();
		auto setting_value = py::str(setting_obj[py::str("value")]).cast<string>();
		auto input_type = setting_obj.contains(py::str("input_type"))
		                      ? py::str(setting_obj[py::str("input_type")]).cast<string>()
		                      : string("VARCHAR");
		if (setting_name.empty()) {
			continue;
		}
		string sql_value;
		if (IsBooleanConnectionSettingType(input_type) || IsNumericConnectionSettingType(input_type)) {
			sql_value = setting_value;
		} else {
			sql_value = QuoteSQLStringLiteral(setting_value);
		}
		ExecuteSnapshotQuery(conn_wrapper.con.GetConnection(), "SET " + setting_name + " = " + sql_value);
	}
}

string PyLogicalPlan::session_id() const {
	return VaneSessionIdFromSnapshot(connection_snapshot_);
}

py::dict PyLogicalPlan::session_config() const {
	return VaneSessionConfigFromSnapshot(connection_snapshot_);
}

enum class QueryPythonReplayField : uint8_t {
	UDFRegistrations,
	UDFActorHandles,
	ConnectionSnapshot,
};

static py::object LookupQueryPythonReplayState(const string &query_id, QueryPythonReplayField field) {
	if (query_id.empty()) {
		return py::none();
	}
	std::lock_guard<std::mutex> guard(g_query_python_replay_states_lock);
	auto entry = g_query_python_replay_states.find(query_id);
	if (entry == g_query_python_replay_states.end()) {
		return py::none();
	}
	switch (field) {
	case QueryPythonReplayField::UDFRegistrations:
		return entry->second->udf_registrations.get();
	case QueryPythonReplayField::UDFActorHandles:
		return entry->second->udf_actor_handles.get();
	case QueryPythonReplayField::ConnectionSnapshot:
		return entry->second->connection_snapshot.get();
	default:
		throw duckdb::InternalException("Unknown query Python replay field");
	}
}

static bool RegisterQueryPythonReplayState(const string &query_id, const py::object &udf_registrations,
                                           const py::object &udf_actor_handles, const py::object &connection_snapshot) {
	if (query_id.empty()) {
		throw duckdb::InternalException("Query Python replay state requires a non-empty query_id");
	}
	auto session_id = VaneSessionIdFromSnapshot(connection_snapshot);
	py::object session_config = VaneSessionConfigFromSnapshot(connection_snapshot);
	std::lock_guard<std::mutex> guard(g_query_python_replay_states_lock);
	auto entry = g_query_python_replay_states.find(query_id);
	if (entry == g_query_python_replay_states.end()) {
		g_query_python_replay_states.emplace(query_id, std::make_unique<QueryPythonReplayState>(
		                                                   std::move(session_id), std::move(session_config),
		                                                   py::reinterpret_borrow<py::object>(udf_registrations),
		                                                   py::reinterpret_borrow<py::object>(udf_actor_handles),
		                                                   py::reinterpret_borrow<py::object>(connection_snapshot)));
		return true;
	}
	auto &state = *entry->second;
	if (state.session_id != session_id || !PythonObjectsEqual(state.session_config.get(), session_config)) {
		throw duckdb::InvalidInputException("Query " + query_id + " was registered with a different Vane session");
	}
	if (!PythonObjectsEqual(state.connection_snapshot.get(), connection_snapshot)) {
		throw duckdb::InvalidInputException("Query " + query_id +
		                                    " was registered with a different connection snapshot");
	}
	if (!PythonObjectsEqual(state.udf_registrations.get(), udf_registrations)) {
		throw duckdb::InvalidInputException("Query " + query_id +
		                                    " was registered with different Python UDF registrations");
	}
	if (!PythonObjectsEqual(state.udf_actor_handles.get(), udf_actor_handles)) {
		throw duckdb::InvalidInputException("Query " + query_id +
		                                    " was registered with different Python UDF actor handles");
	}
	return false;
}

static py::object LookupQueryUDFRegistrations(const string &query_id) {
	return LookupQueryPythonReplayState(query_id, QueryPythonReplayField::UDFRegistrations);
}

static py::object LookupQueryConnectionSnapshot(const string &query_id) {
	return LookupQueryPythonReplayState(query_id, QueryPythonReplayField::ConnectionSnapshot);
}

static py::object LookupQueryUDFActorHandles(const string &query_id) {
	return LookupQueryPythonReplayState(query_id, QueryPythonReplayField::UDFActorHandles);
}

static void CleanupQueryPythonReplayState(const string &query_id) {
	if (query_id.empty()) {
		return;
	}
	std::lock_guard<std::mutex> guard(g_query_python_replay_states_lock);
	g_query_python_replay_states.erase(query_id);
}

static void CleanupAllQueryPythonReplayState() {
	std::lock_guard<std::mutex> guard(g_query_python_replay_states_lock);
	g_query_python_replay_states.clear();
}

static duckdb::unique_ptr<duckdb::LogicalOperator>
RebindAndOptimizeDeserializedLogicalPlan(duckdb::ClientContext &context,
                                         duckdb::unique_ptr<duckdb::LogicalOperator> logical_plan) {
	auto logical_plan_stmt = duckdb::make_uniq<duckdb::LogicalPlanStatement>(std::move(logical_plan));
	duckdb::Planner planner(context);
	planner.CreatePlan(std::move(logical_plan_stmt));
	if (!planner.plan) {
		throw duckdb::InternalException("Planner failed to create logical plan from deserialized LogicalPlanStatement");
	}

	auto rebound_plan = std::move(planner.plan);
	auto &client_config = duckdb::ClientConfig::GetConfig(context);
	if (client_config.enable_optimizer && rebound_plan->RequireOptimizer()) {
		duckdb::Optimizer optimizer(*planner.binder, context);
		rebound_plan = optimizer.Optimize(std::move(rebound_plan));
	}
	return rebound_plan;
}

static duckdb::distributed::DistributedPipelineNodeRef
BuildDistributedPipelineNode(const std::shared_ptr<duckdb::distributed::DistributedPhysicalPlan> &plan,
                             duckdb::ClientContext *client_context = nullptr) {
	using namespace duckdb::distributed;
	if (!plan) {
		throw duckdb::InternalException("DistributedPhysicalPlan is null");
	}
	auto physical_plan = plan->physical_plan();
	if (!physical_plan) {
		throw duckdb::InternalException("DistributedPhysicalPlan has no physical plan");
	}
	if (!physical_plan->HasRoot()) {
		throw duckdb::InternalException("DistributedPhysicalPlan physical plan has no root");
	}
	PlanConfig cfg(plan->idx(), plan->query_id(), plan->execution_config());
	auto pipeline_res = physical_plan_to_pipeline_node(std::move(cfg), std::move(physical_plan), client_context);
	if (!pipeline_res.is_ok()) {
		throw duckdb::InternalException(string("Failed to build distributed pipeline node: ") +
		                                pipeline_res.error().what());
	}
	if (!pipeline_res.value()) {
		throw duckdb::InternalException("Distributed pipeline translation returned null root node");
	}
	return pipeline_res.value();
}

static const UDFFunctionData *TryGetUDFBindData(const FunctionData *bind_data) {
	// FunctionData::Cast<T>() only asserts its dynamic type in debug builds and
	// becomes a reinterpret_cast in Release. Generic INOUT functions (for
	// example UNNEST) have unrelated bind data and must not be treated as UDFs.
	return dynamic_cast<const UDFFunctionData *>(bind_data);
}

static const UDFFunctionData *TryGetUDFBindData(const PhysicalTableInOutFunction &inout) {
	return TryGetUDFBindData(inout.GetBindData());
}

static const UDFFunctionData *TryGetUDFBindData(const PhysicalStreamingUDF &streaming) {
	return TryGetUDFBindData(streaming.GetBindData());
}

static UDFFunctionData *TryGetMutableUDFBindData(PhysicalOperator &op) {
	const UDFFunctionData *bind_data = nullptr;
	if (op.type == PhysicalOperatorType::INOUT_FUNCTION) {
		bind_data = TryGetUDFBindData(op.Cast<PhysicalTableInOutFunction>());
	} else if (op.type == PhysicalOperatorType::STREAMING_UDF) {
		bind_data = TryGetUDFBindData(op.Cast<PhysicalStreamingUDF>());
	}
	return const_cast<UDFFunctionData *>(bind_data);
}

static void CollectMutableUDFBindData(PhysicalOperator &op, vector<UDFFunctionData *> &out) {
	if (auto *bind_data = TryGetMutableUDFBindData(op)) {
		out.push_back(bind_data);
	}
}

static duckdb::shared_ptr<void> WrapPyObjectForUDFActorHandles(const py::object &obj) {
	if (obj.is_none()) {
		return nullptr;
	}
	auto *boxed = new py::object(py::reinterpret_borrow<py::object>(obj));
	return duckdb::shared_ptr<void>(boxed, [](void *ptr) {
		if (!ptr) {
			return;
		}
		auto *boxed_obj = static_cast<py::object *>(ptr);
		if (!Py_IsInitialized() || PythonIsFinalizing()) {
			boxed_obj->release();
			delete boxed_obj;
			return;
		}
		PythonGILWrapper gil;
		delete boxed_obj;
	});
}
