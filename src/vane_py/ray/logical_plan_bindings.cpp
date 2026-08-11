// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Included by ray_module.cpp inside namespace duckdb.

struct PyPhysicalPlanWrapper;

static constexpr char LOGICAL_PLAN_ENVELOPE_MAGIC[] = {'V', 'A', 'N', 'E', 'P', 'L', 'A', 'N'};
static constexpr idx_t LOGICAL_PLAN_ENVELOPE_MAGIC_SIZE = sizeof(LOGICAL_PLAN_ENVELOPE_MAGIC);
static constexpr uint32_t LOGICAL_PLAN_PROTOCOL_VERSION = 1;
static constexpr uint32_t LOGICAL_PLAN_MAX_SOURCE_ID_SIZE = 4096;

static void AppendUInt32LE(string &result, uint32_t value) {
	for (idx_t byte_idx = 0; byte_idx < sizeof(value); byte_idx++) {
		result.push_back(static_cast<char>((value >> (byte_idx * 8)) & 0xff));
	}
}

static void AppendUInt64LE(string &result, uint64_t value) {
	for (idx_t byte_idx = 0; byte_idx < sizeof(value); byte_idx++) {
		result.push_back(static_cast<char>((value >> (byte_idx * 8)) & 0xff));
	}
}

static uint32_t ReadUInt32LE(const string &envelope, idx_t &offset, const char *field_name) {
	if (offset > envelope.size() || envelope.size() - offset < sizeof(uint32_t)) {
		throw InvalidInputException("Logical plan envelope is truncated before %s", field_name);
	}
	uint32_t result = 0;
	for (idx_t byte_idx = 0; byte_idx < sizeof(result); byte_idx++) {
		result |= static_cast<uint32_t>(static_cast<uint8_t>(envelope[offset++])) << (byte_idx * 8);
	}
	return result;
}

static uint64_t ReadUInt64LE(const string &envelope, idx_t &offset, const char *field_name) {
	if (offset > envelope.size() || envelope.size() - offset < sizeof(uint64_t)) {
		throw InvalidInputException("Logical plan envelope is truncated before %s", field_name);
	}
	uint64_t result = 0;
	for (idx_t byte_idx = 0; byte_idx < sizeof(result); byte_idx++) {
		result |= static_cast<uint64_t>(static_cast<uint8_t>(envelope[offset++])) << (byte_idx * 8);
	}
	return result;
}

static string EncodeLogicalPlanEnvelope(const string &logical_payload) {
	if (logical_payload.empty()) {
		throw InternalException("Cannot encode an empty logical plan payload");
	}
	string source_id = DuckDB::SourceID();
	if (source_id.empty() || source_id.size() > LOGICAL_PLAN_MAX_SOURCE_ID_SIZE) {
		throw InternalException("DuckDB SourceID is empty or exceeds the logical plan envelope limit");
	}

	string envelope;
	envelope.reserve(LOGICAL_PLAN_ENVELOPE_MAGIC_SIZE + sizeof(uint32_t) * 2 + sizeof(uint64_t) + source_id.size() +
	                 logical_payload.size());
	envelope.append(LOGICAL_PLAN_ENVELOPE_MAGIC, LOGICAL_PLAN_ENVELOPE_MAGIC_SIZE);
	AppendUInt32LE(envelope, LOGICAL_PLAN_PROTOCOL_VERSION);
	AppendUInt32LE(envelope, static_cast<uint32_t>(source_id.size()));
	AppendUInt64LE(envelope, static_cast<uint64_t>(logical_payload.size()));
	envelope.append(source_id);
	envelope.append(logical_payload);
	return envelope;
}

static string DecodeLogicalPlanEnvelope(const string &envelope) {
	if (envelope.size() < LOGICAL_PLAN_ENVELOPE_MAGIC_SIZE ||
	    envelope.compare(0, LOGICAL_PLAN_ENVELOPE_MAGIC_SIZE, LOGICAL_PLAN_ENVELOPE_MAGIC,
	                     LOGICAL_PLAN_ENVELOPE_MAGIC_SIZE) != 0) {
		throw InvalidInputException("Logical plan payload is not a Vane logical plan envelope");
	}

	idx_t offset = LOGICAL_PLAN_ENVELOPE_MAGIC_SIZE;
	auto protocol_version = ReadUInt32LE(envelope, offset, "protocol version");
	if (protocol_version != LOGICAL_PLAN_PROTOCOL_VERSION) {
		throw InvalidInputException("Unsupported logical plan protocol version %u (expected %u)", protocol_version,
		                            LOGICAL_PLAN_PROTOCOL_VERSION);
	}
	auto source_id_size = ReadUInt32LE(envelope, offset, "SourceID length");
	auto payload_size = ReadUInt64LE(envelope, offset, "payload length");
	if (source_id_size == 0 || source_id_size > LOGICAL_PLAN_MAX_SOURCE_ID_SIZE) {
		throw InvalidInputException("Logical plan envelope contains an invalid SourceID length");
	}
	if (offset > envelope.size() || source_id_size > envelope.size() - offset) {
		throw InvalidInputException("Logical plan envelope is truncated inside the SourceID");
	}

	auto serialized_source_id = envelope.substr(offset, source_id_size);
	offset += source_id_size;
	string current_source_id = DuckDB::SourceID();
	if (current_source_id.empty() || serialized_source_id != current_source_id) {
		throw InvalidInputException("Logical plan SourceID mismatch: payload was built by %s, current engine is %s",
		                            serialized_source_id, current_source_id);
	}
	if (payload_size == 0) {
		throw InvalidInputException("Logical plan envelope contains an empty payload");
	}
	auto remaining_size = envelope.size() - offset;
	if (payload_size != static_cast<uint64_t>(remaining_size)) {
		throw InvalidInputException("Logical plan envelope payload length mismatch: declared %llu bytes, found %llu",
		                            payload_size, static_cast<uint64_t>(remaining_size));
	}
	return envelope.substr(offset, remaining_size);
}

struct PyLogicalPlan {
	string query_id_;
	string serialized_logical_plan_;
	// Driver-local source connection; intentionally omitted from pickle state.
	py::object source_connection_ = py::none();
	py::object udf_registrations_ = py::none();
	py::object connection_snapshot_ = py::none();

	PyLogicalPlan() = default;

	string idx() const {
		return query_id_;
	}

	string session_id() const;
	py::dict session_config() const;
	bool has_explicit_s3_credentials() const;

	PyPhysicalPlanWrapper to_physical_plan(py::object conn_obj, py::object effective_session_config) const;
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
		auto logical_payload = string(reinterpret_cast<const char *>(data_ptr), data_size);
		serialized_plan = EncodeLogicalPlanEnvelope(logical_payload);
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

static bool IsExtensionSecuritySetting(const string &lower_name) {
	return lower_name == "allow_unsigned_extensions" || lower_name == "autoinstall_known_extensions" ||
	       lower_name == "autoload_known_extensions";
}

static bool IsSecretPersistenceSetting(const string &lower_name) {
	return lower_name == "allow_persistent_secrets" || lower_name == "default_secret_storage" ||
	       lower_name == "secret_directory";
}

static py::dict SanitizeBootstrapConfig(const py::dict &config, bool in_memory_database) {
	py::dict sanitized;
	// Preserve absent options so file-backed connections keep DuckDB's
	// configuration identity; Vane's build defaults all three settings to OFF.
	for (auto item : config) {
		auto name = duckdb::StringUtil::Lower(py::str(item.first).cast<string>());
		if (IsExtensionSecuritySetting(name)) {
			sanitized[py::str(name)] = py::str("false");
			continue;
		}
		if (IsSecretPersistenceSetting(name)) {
			if (!in_memory_database) {
				sanitized[item.first] = item.second;
			}
			continue;
		}
		sanitized[item.first] = item.second;
	}
	// A transported in-memory plan carries the complete source secret snapshot.
	// File databases retain the settings needed to identify the source
	// bootstrap; worker execution may separately force an isolated read-only
	// instance.
	if (in_memory_database) {
		sanitized[py::str("allow_persistent_secrets")] = py::str("false");
	}
	return sanitized;
}

static py::dict ForceReadOnlyAccessMode(const py::dict &config) {
	py::dict result;
	for (auto item : config) {
		auto name = duckdb::StringUtil::Lower(py::str(item.first).cast<string>());
		if (name != "access_mode") {
			result[item.first] = item.second;
		}
	}
	result[py::str("access_mode")] = py::str("read_only");
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

static bool HasExplicitS3CredentialsFromSnapshot(const py::object &snapshot_obj) {
	if (snapshot_obj.is_none() || !py::isinstance<py::dict>(snapshot_obj)) {
		throw duckdb::InvalidInputException("Connection snapshot is missing the required Vane session");
	}
	auto snapshot = snapshot_obj.cast<py::dict>();
	if (!snapshot.contains(py::str("settings")) || !py::isinstance<py::list>(snapshot[py::str("settings")])) {
		return false;
	}
	bool has_access_key = false;
	bool has_secret_key = false;
	bool has_session_token = false;
	string access_key;
	string secret_key;
	string session_token;
	for (auto item : snapshot[py::str("settings")].cast<py::list>()) {
		if (!py::isinstance<py::dict>(item)) {
			continue;
		}
		auto setting = py::reinterpret_borrow<py::dict>(item);
		if (!setting.contains(py::str("name")) || !setting.contains(py::str("value"))) {
			continue;
		}
		auto name = duckdb::StringUtil::Lower(py::str(setting[py::str("name")]).cast<string>());
		if (name == "s3_access_key_id") {
			has_access_key = true;
			access_key = py::str(setting[py::str("value")]).cast<string>();
		} else if (name == "s3_secret_access_key") {
			has_secret_key = true;
			secret_key = py::str(setting[py::str("value")]).cast<string>();
		} else if (name == "s3_session_token") {
			has_session_token = true;
			session_token = py::str(setting[py::str("value")]).cast<string>();
		}
	}
	const bool has_access_key_value = !access_key.empty();
	const bool has_secret_key_value = !secret_key.empty();
	if (has_access_key != has_secret_key || has_access_key_value != has_secret_key_value ||
	    (has_session_token && !session_token.empty() && !has_access_key_value)) {
		throw duckdb::InvalidInputException(
		    "Explicit DuckDB S3 credentials must set both s3_access_key_id and s3_secret_access_key");
	}
	return has_access_key;
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

static string BootstrapDatabasePath(const py::object &bootstrap_obj) {
	if (bootstrap_obj.is_none() || !py::isinstance<py::dict>(bootstrap_obj)) {
		return ":memory:";
	}
	auto bootstrap = bootstrap_obj.cast<py::dict>();
	if (!bootstrap.contains(py::str("database")) || bootstrap[py::str("database")].is_none()) {
		return ":memory:";
	}
	return py::str(bootstrap[py::str("database")]).cast<string>();
}

static bool BootstrapUsesInMemoryDatabase(const py::object &bootstrap_obj) {
	auto database = BootstrapDatabasePath(bootstrap_obj);
	return duckdb::DBConfig::IsInMemoryDatabase(database.c_str());
}

static bool SnapshotHasAttachedDatabases(const py::object &snapshot_obj) {
	if (snapshot_obj.is_none() || !py::isinstance<py::dict>(snapshot_obj)) {
		return false;
	}
	auto snapshot = snapshot_obj.cast<py::dict>();
	if (!snapshot.contains(py::str("attached_databases"))) {
		return false;
	}
	auto attached_obj = snapshot[py::str("attached_databases")];
	if (attached_obj.is_none()) {
		return false;
	}
	if (!py::isinstance<py::list>(attached_obj)) {
		throw InvalidInputException("Connection snapshot attached_databases must be a list");
	}
	return py::len(attached_obj) > 0;
}

static bool SnapshotHasSecrets(const py::object &snapshot_obj) {
	if (snapshot_obj.is_none() || !py::isinstance<py::dict>(snapshot_obj)) {
		return false;
	}
	auto snapshot = snapshot_obj.cast<py::dict>();
	if (!snapshot.contains(py::str("secrets"))) {
		return false;
	}
	auto secrets_obj = snapshot[py::str("secrets")];
	if (secrets_obj.is_none()) {
		return false;
	}
	if (!py::isinstance<py::list>(secrets_obj)) {
		throw InvalidInputException("Connection snapshot secrets must be a list");
	}
	return py::len(secrets_obj) > 0;
}

static void DisablePersistentSecretsWhenUnused(DuckDBPyConnection &connection) {
	auto &context = *connection.con.GetConnection().context;
	auto &database = duckdb::DatabaseInstance::GetDatabase(context);
	(void)SecretManager::Get(database).TrySetEnablePersistentSecrets(false);
}

static py::object CreateConnectionFromBootstrapSnapshot(const py::object &bootstrap_obj, bool use_instance_cache = true,
                                                        bool force_file_read_only = false) {
	if (IsDefaultBootstrapSnapshot(bootstrap_obj)) {
		py::dict config;
		config[py::str("allow_persistent_secrets")] = py::str("false");
		auto connection = use_instance_cache ? DuckDBPyConnection::Connect(py::str(":memory:"), false, config)
		                                     : DuckDBPyConnection::ConnectUncached(py::str(":memory:"), false, config);
		DisablePersistentSecretsWhenUnused(*connection);
		return py::cast(std::move(connection));
	}

	auto bootstrap = bootstrap_obj.cast<py::dict>();
	auto database = BootstrapDatabasePath(bootstrap_obj);
	auto in_memory_database = duckdb::DBConfig::IsInMemoryDatabase(database.c_str());

	bool source_read_only = false;
	if (bootstrap.contains(py::str("read_only")) && !bootstrap[py::str("read_only")].is_none()) {
		source_read_only = bootstrap[py::str("read_only")].cast<bool>();
	}

	py::dict config = py::dict();
	if (bootstrap.contains(py::str("config")) && !bootstrap[py::str("config")].is_none() &&
	    py::isinstance<py::dict>(bootstrap[py::str("config")])) {
		config = CopyPyDict(bootstrap[py::str("config")].cast<py::dict>());
	}
	auto sanitized_config = SanitizeBootstrapConfig(config, in_memory_database);
	auto worker_file_read_only = force_file_read_only && !in_memory_database;
	if (worker_file_read_only) {
		sanitized_config = ForceReadOnlyAccessMode(sanitized_config);
	}
	auto connection_read_only = source_read_only || worker_file_read_only;
	auto connection =
	    use_instance_cache
	        ? DuckDBPyConnection::Connect(py::str(database), connection_read_only, sanitized_config)
	        : DuckDBPyConnection::ConnectUncached(py::str(database), connection_read_only, sanitized_config);
	DisablePersistentSecretsWhenUnused(*connection);
	// Keep the source bootstrap identity for connection matching even though
	// worker security and file access settings are forced off/read-only.
	connection->SetConnectionBootstrapConfig(database, source_read_only, config);
	return py::cast(std::move(connection));
}

static py::object CreateSnapshotBaselineConnection(DuckDBPyConnection &source_conn, const py::object &bootstrap_obj) {
	if (BootstrapUsesInMemoryDatabase(bootstrap_obj)) {
		return CreateConnectionFromBootstrapSnapshot(bootstrap_obj);
	}
	// A fresh cursor preserves the existing file-database baseline: database-
	// global settings remain defaults while connection-local overrides differ.
	// It also avoids reopening a live file with sanitized bootstrap settings.
	return py::cast(source_conn.Cursor());
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

static bool ConnectionsShareDatabaseInstance(const py::object &lhs_obj, const py::object &rhs_obj) {
	if (lhs_obj.is_none() || rhs_obj.is_none()) {
		return false;
	}
	auto &lhs_wrapper = ExtractPyConnectionWrapper(lhs_obj);
	auto &rhs_wrapper = ExtractPyConnectionWrapper(rhs_obj);
	if (lhs_wrapper.con.ConnectionIsClosed() || rhs_wrapper.con.ConnectionIsClosed()) {
		return false;
	}
	auto &lhs = lhs_wrapper.con.GetConnection();
	auto &rhs = rhs_wrapper.con.GetConnection();
	return &duckdb::DatabaseInstance::GetDatabase(*lhs.context) == &duckdb::DatabaseInstance::GetDatabase(*rhs.context);
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

static py::object ResolvePlanningConnectionForSnapshot(py::object conn_obj, const py::object &source_conn_obj,
                                                       const py::object &snapshot_obj) {
	auto bootstrap_obj = LookupBootstrapSnapshot(snapshot_obj);
	if (SnapshotHasAttachedDatabases(snapshot_obj) || SnapshotHasSecrets(snapshot_obj)) {
		if (!source_conn_obj.is_none() && ConnectionMatchesBootstrapSnapshot(source_conn_obj, snapshot_obj)) {
			// Local execution can keep using the source DatabaseInstance where the
			// catalog and secrets already exist. A transported logical plan has no
			// source connection, so it gets an isolated planning DatabaseInstance.
			return py::cast(ExtractPyConnectionWrapper(source_conn_obj).Cursor());
		}
		return CreateConnectionFromBootstrapSnapshot(bootstrap_obj);
	}
	if (bootstrap_obj.is_none() || IsDefaultBootstrapSnapshot(bootstrap_obj) ||
	    ConnectionMatchesBootstrapSnapshot(conn_obj, snapshot_obj)) {
		return conn_obj;
	}
	if (!BootstrapUsesInMemoryDatabase(bootstrap_obj) && !source_conn_obj.is_none() &&
	    ConnectionMatchesBootstrapSnapshot(source_conn_obj, snapshot_obj)) {
		// The source DatabaseInstance may still be alive. Reopening its file with
		// the sanitized worker configuration violates DuckDB's instance cache;
		// a cursor shares that instance without running database initialization.
		return py::cast(ExtractPyConnectionWrapper(source_conn_obj).Cursor());
	}
	return ResolveConnectionForSnapshot(conn_obj, snapshot_obj);
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
	string scope;
};

static void EnforceExtensionSecuritySettings(duckdb::Connection &conn) {
	// Snapshot replay can run with a query-local result collector installed, so
	// update the configuration without executing SET statements.
	auto &config = duckdb::DBConfig::GetConfig(*conn.context);
	config.SetOptionByName("allow_unsigned_extensions", duckdb::Value::BOOLEAN(false));
	config.SetOptionByName("autoinstall_known_extensions", duckdb::Value::BOOLEAN(false));
	config.SetOptionByName("autoload_known_extensions", duckdb::Value::BOOLEAN(false));
}

static bool ShouldSkipConnectionSettingSnapshot(const string &name, const string &input_type) {
	auto lower_name = duckdb::StringUtil::Lower(name);
	auto upper_input_type = duckdb::StringUtil::Upper(input_type);
	if (lower_name == "duckdb_api" || IsExtensionSecuritySetting(lower_name) ||
	    IsSecretPersistenceSetting(lower_name)) {
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

static bool IsVaneSessionBaselineConnectionSetting(const string &name) {
	static const std::unordered_set<string> names {
	    "http_keep_alive",  "http_retries", "http_retry_backoff", "http_retry_wait_ms",
	    "s3_access_key_id", "s3_endpoint",  "s3_region",          "s3_secret_access_key",
	    "s3_session_token", "s3_url_style", "s3_use_ssl",
	};
	return names.find(duckdb::StringUtil::Lower(name)) != names.end();
}

static bool IsS3CredentialConnectionSetting(const string &name) {
	static const std::unordered_set<string> names {
	    "s3_access_key_id",
	    "s3_secret_access_key",
	    "s3_session_token",
	};
	return names.find(duckdb::StringUtil::Lower(name)) != names.end();
}

static string QuoteSQLStringLiteral(const string &value) {
	return "'" + duckdb::StringUtil::Replace(value, "'", "''") + "'";
}

static duckdb::unique_ptr<duckdb::MaterializedQueryResult> ExecuteSnapshotQuery(duckdb::Connection &conn,
                                                                                const string &sql) {
	auto result = conn.Query(sql);
	if (!result) {
		throw duckdb::InternalException("Connection snapshot query returned a null result");
	}
	if (result->HasError()) {
		// Snapshot statements can contain access keys, secret values, or catalog
		// attachment options. DuckDB diagnostics may echo the input line, so only
		// retain the non-sensitive error category in exceptions that can reach
		// worker logs.
		throw duckdb::InvalidInputException("Connection snapshot query failed (" +
		                                    duckdb::Exception::ExceptionTypeToString(result->GetErrorType()) + ")");
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

static void ApplyVaneSessionConfigValues(duckdb::Connection &conn, const py::dict &config) {
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
	if (!access_key.empty() || !secret_key.empty() || !session_token.empty()) {
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

static void ApplyVaneSessionConfig(duckdb::Connection &conn, const py::object &snapshot_obj) {
	ApplyVaneSessionConfigValues(conn, VaneSessionConfigFromSnapshot(snapshot_obj));
}

static void CloseOpenPythonConnectionResult(DuckDBPyConnection &conn_wrapper) {
	if (conn_wrapper.con.HasResult()) {
		// A partially consumed StreamQueryResult keeps ClientContext::active_query
		// alive, and StreamQueryResult::Close only drops its weak context handle.
		// Starting a materialized no-op query runs DuckDB's InitialCleanup before
		// snapshot replay enters RunFunctionInTransaction directly.
		(void)ExecuteSnapshotQuery(conn_wrapper.con.GetConnection(), "SELECT NULL WHERE false");
		conn_wrapper.con.GetResult().Close();
	}
	conn_wrapper.con.SetResult(nullptr);
}

static void ApplyEffectiveVaneSessionConfig(DuckDBPyConnection &conn_wrapper, const py::object &config_obj) {
	CloseOpenPythonConnectionResult(conn_wrapper);
	if (config_obj.is_none()) {
		return;
	}
	if (!py::isinstance<py::dict>(config_obj)) {
		throw duckdb::InvalidInputException("Effective Vane session config must be a dict");
	}
	ApplyVaneSessionConfigValues(conn_wrapper.con.GetConnection(), config_obj.cast<py::dict>());
}

static std::vector<string> QueryLoadedNonStaticExtensionNames(DuckDBPyConnection &conn_wrapper) {
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

struct StaticExtensionSnapshotEntry {
	string name;
	string version;

	bool operator==(const StaticExtensionSnapshotEntry &other) const {
		return name == other.name && version == other.version;
	}

	bool operator<(const StaticExtensionSnapshotEntry &other) const {
		return name < other.name;
	}
};

static vector<StaticExtensionSnapshotEntry> QueryLoadedStaticExtensions(DuckDBPyConnection &conn_wrapper) {
	vector<StaticExtensionSnapshotEntry> extensions;
	auto result =
	    ExecuteSnapshotQuery(conn_wrapper.con.GetConnection(), "SELECT extension_name, extension_version "
	                                                           "FROM duckdb_extensions() "
	                                                           "WHERE loaded AND install_mode = 'STATICALLY_LINKED' "
	                                                           "ORDER BY extension_name");
	auto &collection = result->Collection();
	extensions.reserve(collection.Count());
	for (auto &row : collection.Rows()) {
		auto name_value = row.GetValue(0);
		if (name_value.IsNull()) {
			continue;
		}
		auto extension_name = name_value.ToString();
		if (!extension_name.empty()) {
			auto version_value = row.GetValue(1);
			extensions.push_back(
			    {std::move(extension_name), version_value.IsNull() ? string() : version_value.ToString()});
		}
	}
	return extensions;
}

static std::vector<ConnectionSettingRecord> QueryConnectionSettings(DuckDBPyConnection &conn_wrapper) {
	std::vector<ConnectionSettingRecord> settings;
	auto result = ExecuteSnapshotQuery(conn_wrapper.con.GetConnection(), "SELECT name, value, input_type, scope "
	                                                                     "FROM duckdb_settings() "
	                                                                     "ORDER BY name");
	auto &collection = result->Collection();
	settings.reserve(collection.Count());
	for (auto &row : collection.Rows()) {
		ConnectionSettingRecord record;
		auto name_val = row.GetValue(0);
		auto value_val = row.GetValue(1);
		auto input_type_val = row.GetValue(2);
		auto scope_val = row.GetValue(3);
		if (name_val.IsNull() || input_type_val.IsNull() || scope_val.IsNull()) {
			continue;
		}
		record.name = name_val.ToString();
		record.value = value_val.IsNull() ? string() : value_val.ToString();
		record.input_type = input_type_val.ToString();
		record.scope = scope_val.ToString();
		settings.push_back(std::move(record));
	}
	return settings;
}

static void RejectNonStaticRayExtensions(const std::vector<string> &extension_names) {
	if (extension_names.empty()) {
		return;
	}
	auto joined_names = extension_names.front();
	for (idx_t index = 1; index < extension_names.size(); index++) {
		joined_names += ", " + extension_names[index];
	}
	throw duckdb::InvalidInputException("Ray distributed execution supports only statically linked extensions; "
	                                    "non-static extensions are not supported: "
	                                    "%s",
	                                    joined_names);
}

static bool IsSafeStaticExtensionName(const string &name) {
	if (name.empty()) {
		return false;
	}
	for (auto character : name) {
		if ((character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') ||
		    (character >= '0' && character <= '9') || character == '_') {
			continue;
		}
		return false;
	}
	return true;
}

static string StaticExtensionListIdentity(const vector<StaticExtensionSnapshotEntry> &extensions) {
	vector<string> identities;
	identities.reserve(extensions.size());
	for (const auto &extension : extensions) {
		identities.push_back(extension.name + "@" + extension.version);
	}
	return "[" + StringUtil::Join(identities, ",") + "]";
}

static vector<StaticExtensionSnapshotEntry> LoadedStaticExtensions(DatabaseInstance &database) {
	vector<StaticExtensionSnapshotEntry> extensions;
	auto &manager = ExtensionManager::Get(database);
	for (const auto &extension_name : manager.GetExtensions()) {
		auto info = manager.GetExtensionInfo(extension_name);
		if (!info) {
			continue;
		}
		lock_guard<mutex> guard(info->lock);
		if (!info->is_loaded || !info->install_info ||
		    info->install_info->mode != ExtensionInstallMode::STATICALLY_LINKED) {
			continue;
		}
		extensions.push_back({extension_name, info->install_info->version});
	}
	std::sort(extensions.begin(), extensions.end());
	return extensions;
}

static void LoadStaticRayExtensions(duckdb::Connection &conn, const vector<StaticExtensionSnapshotEntry> &extensions) {
	auto &database = duckdb::DatabaseInstance::GetDatabase(*conn.context);
	duckdb::DuckDB db(database);
	for (const auto &extension : extensions) {
		if (!IsSafeStaticExtensionName(extension.name)) {
			throw duckdb::InvalidInputException("Invalid static extension name in connection snapshot: %s",
			                                    extension.name);
		}
		auto linked_name = duckdb::StringUtil::Lower(extension.name);
		if (!duckdb::ExtensionHelper::IsExtensionLinked(linked_name)) {
			throw duckdb::InvalidInputException("Ray distributed execution supports only statically linked extensions; "
			                                    "extension '%s' is not statically "
			                                    "linked into this worker",
			                                    extension.name);
		}
		if (duckdb::ExtensionHelper::LoadExtension(db, linked_name) != duckdb::ExtensionLoadResult::LOADED_EXTENSION) {
			throw duckdb::InvalidInputException("Ray distributed execution supports only statically linked extensions; "
			                                    "extension '%s' is not statically "
			                                    "linked into this worker",
			                                    extension.name);
		}
	}
	auto loaded_extensions = LoadedStaticExtensions(database);
	if (loaded_extensions != extensions) {
		throw InvalidInputException(
		    "Static extension identities differ between coordinator and worker: expected %s, worker loaded %s",
		    StaticExtensionListIdentity(extensions), StaticExtensionListIdentity(loaded_extensions));
	}
}

static py::list CaptureDistributedExtensionManifests(DatabaseInstance &database) {
	py::list manifests_obj;
	for (const auto &manifest : DistributedExtensionManager::Get(database).GetExtensions()) {
		py::dict manifest_obj;
		manifest_obj[py::str("extension_name")] = py::str(manifest.extension_name);
		manifest_obj[py::str("protocol_version")] = py::int_(manifest.protocol_version);
		py::list capabilities_obj;
		for (const auto &capability : manifest.capabilities) {
			py::dict capability_obj;
			capability_obj[py::str("kind")] = py::str(DistributedExtensionCapabilityKindToString(capability.kind));
			capability_obj[py::str("name")] = py::str(capability.name);
			capability_obj[py::str("protocol_version")] = py::int_(capability.protocol_version);
			capabilities_obj.append(std::move(capability_obj));
		}
		manifest_obj[py::str("capabilities")] = std::move(capabilities_obj);
		manifests_obj.append(std::move(manifest_obj));
	}
	return manifests_obj;
}

static idx_t ParsePositiveProtocolVersion(const py::dict &entry, const char *field_name, const string &identity) {
	auto field = py::str(field_name);
	if (!entry.contains(field) || !py::isinstance<py::int_>(entry[field]) || py::isinstance<py::bool_>(entry[field])) {
		throw InvalidInputException("Connection snapshot %s must contain a non-boolean integer %s", identity,
		                            field_name);
	}
	auto value = entry[field].cast<int64_t>();
	if (value <= 0) {
		throw InvalidInputException("Connection snapshot %s %s must be greater than zero", identity, field_name);
	}
	return NumericCast<idx_t>(value);
}

static vector<DistributedExtensionManifest> ParseDistributedExtensionManifests(const py::dict &snapshot) {
	if (!snapshot.contains(py::str("distributed_extensions")) ||
	    !py::isinstance<py::list>(snapshot[py::str("distributed_extensions")])) {
		throw InvalidInputException("Connection snapshot distributed_extensions must be a list");
	}
	vector<DistributedExtensionManifest> manifests;
	set<string> extension_names;
	for (auto item : snapshot[py::str("distributed_extensions")].cast<py::list>()) {
		if (!py::isinstance<py::dict>(item)) {
			throw InvalidInputException("Connection snapshot distributed extension entry must be a dict");
		}
		auto manifest_obj = py::reinterpret_borrow<py::dict>(item);
		if (!manifest_obj.contains(py::str("extension_name")) ||
		    !py::isinstance<py::str>(manifest_obj[py::str("extension_name")])) {
			throw InvalidInputException("Connection snapshot distributed extension is missing extension_name");
		}
		DistributedExtensionManifest manifest;
		manifest.extension_name = manifest_obj[py::str("extension_name")].cast<string>();
		if (manifest.extension_name.empty() || !extension_names.insert(manifest.extension_name).second) {
			throw InvalidInputException("Connection snapshot has an empty or duplicate distributed extension name");
		}
		manifest.protocol_version =
		    ParsePositiveProtocolVersion(manifest_obj, "protocol_version", "distributed extension");
		if (!manifest_obj.contains(py::str("capabilities")) ||
		    !py::isinstance<py::list>(manifest_obj[py::str("capabilities")])) {
			throw InvalidInputException("Connection snapshot distributed extension capabilities must be a list");
		}
		set<string> capability_identities;
		for (auto capability_item : manifest_obj[py::str("capabilities")].cast<py::list>()) {
			if (!py::isinstance<py::dict>(capability_item)) {
				throw InvalidInputException("Connection snapshot distributed extension capability must be a dict");
			}
			auto capability_obj = py::reinterpret_borrow<py::dict>(capability_item);
			if (!capability_obj.contains(py::str("kind")) ||
			    !py::isinstance<py::str>(capability_obj[py::str("kind")]) ||
			    !capability_obj.contains(py::str("name")) ||
			    !py::isinstance<py::str>(capability_obj[py::str("name")])) {
				throw InvalidInputException(
				    "Connection snapshot distributed extension capability is missing kind or name");
			}
			DistributedExtensionCapability capability;
			auto kind = capability_obj[py::str("kind")].cast<string>();
			capability.kind = DistributedExtensionCapabilityKindFromString(kind);
			capability.name = capability_obj[py::str("name")].cast<string>();
			if (capability.name.empty()) {
				throw InvalidInputException(
				    "Connection snapshot distributed extension capability name must not be empty");
			}
			capability.protocol_version =
			    ParsePositiveProtocolVersion(capability_obj, "protocol_version", "distributed extension capability");
			auto capability_identity = kind + ":" + capability.name;
			if (!capability_identities.insert(capability_identity).second) {
				throw InvalidInputException("Connection snapshot has duplicate distributed extension capability '%s'",
				                            capability_identity);
			}
			manifest.capabilities.push_back(std::move(capability));
		}
		DistributedExtensionManager::ValidateManifest(manifest);
		manifests.push_back(std::move(manifest));
	}
	std::sort(manifests.begin(), manifests.end());
	return manifests;
}

static string SerializeSecretForSnapshot(duckdb::ClientContext &context, const BaseSecret &secret) {
	if (!secret.IsSerializable()) {
		throw InvalidInputException("Distributed connection snapshot cannot transport secret '%s'", secret.GetName());
	}

	MemoryStream stream(Allocator::Get(context));
	SerializationOptions options;
	options.serialization_compatibility = SerializationCompatibility::Latest();
	options.serialize_default_values = true;
	BinarySerializer serializer(stream, options);
	serializer.Begin();
	secret.Serialize(serializer);
	serializer.End();

	auto data_size = stream.GetPosition();
	if (data_size == 0) {
		throw InternalException("Distributed connection snapshot serialized secret '%s' to an empty payload",
		                        secret.GetName());
	}
	auto data_ptr = stream.GetData();
	return string(reinterpret_cast<const char *>(data_ptr), data_size);
}

static py::list CaptureSecretSnapshot(DuckDBPyConnection &conn_wrapper) {
	auto &context = *conn_wrapper.con.GetConnection().context;
	struct SerializedSecret {
		string storage;
		string name;
		string payload;
	};
	vector<SerializedSecret> serialized_secrets;
	case_insensitive_set_t serialized_secret_names;
	context.RunFunctionInTransaction([&]() {
		auto transaction = CatalogTransaction::GetSystemCatalogTransaction(context);
		auto secrets = SecretManager::Get(context).AllSecrets(transaction);
		serialized_secrets.reserve(secrets.size());
		for (const auto &entry : secrets) {
			if (!entry.secret) {
				throw InternalException("Distributed connection snapshot encountered a null secret");
			}
			if (!serialized_secret_names.insert(entry.secret->GetName()).second) {
				throw InvalidInputException("Distributed connection snapshot cannot transport multiple secrets named "
				                            "'%s' from different storages",
				                            entry.secret->GetName());
			}
			if (entry.storage_mode.empty() || entry.secret->GetName().empty()) {
				throw InternalException("Distributed connection snapshot encountered a secret without storage or name");
			}
			serialized_secrets.push_back(
			    {entry.storage_mode, entry.secret->GetName(), SerializeSecretForSnapshot(context, *entry.secret)});
		}
	});
	std::sort(serialized_secrets.begin(), serialized_secrets.end(),
	          [](const SerializedSecret &left, const SerializedSecret &right) {
		          return std::tie(left.storage, left.name) < std::tie(right.storage, right.name);
	          });

	py::list secrets_obj;
	for (const auto &entry : serialized_secrets) {
		py::dict secret_obj;
		secret_obj[py::str("storage")] = py::str(entry.storage);
		secret_obj[py::str("name")] = py::str(entry.name);
		secret_obj[py::str("payload")] = py::bytes(entry.payload);
		secrets_obj.append(std::move(secret_obj));
	}
	return secrets_obj;
}

static void ApplySecretSnapshot(duckdb::ClientContext &context, const py::dict &snapshot) {
	if (!snapshot.contains(py::str("secrets"))) {
		return;
	}
	auto secrets_obj = snapshot[py::str("secrets")];
	if (secrets_obj.is_none()) {
		return;
	}
	if (!py::isinstance<py::list>(secrets_obj)) {
		throw InvalidInputException("Connection snapshot secrets must be a list");
	}
	struct SnapshotSecret {
		pair<string, string> identity;
		unique_ptr<BaseSecret> secret;
	};
	vector<SnapshotSecret> snapshot_secrets;
	map<pair<string, string>, string> snapshot_secret_payloads;
	case_insensitive_set_t snapshot_secret_names;
	auto &secret_manager = SecretManager::Get(context);
	for (auto item : secrets_obj.cast<py::list>()) {
		if (!py::isinstance<py::dict>(item)) {
			throw InvalidInputException("Connection snapshot secret entry must be a dict");
		}
		auto secret_obj = py::reinterpret_borrow<py::dict>(item);
		if (!secret_obj.contains(py::str("storage")) || !py::isinstance<py::str>(secret_obj[py::str("storage")])) {
			throw InvalidInputException("Connection snapshot secret entry is missing its storage");
		}
		if (!secret_obj.contains(py::str("name")) || !py::isinstance<py::str>(secret_obj[py::str("name")])) {
			throw InvalidInputException("Connection snapshot secret entry is missing its name");
		}
		if (!secret_obj.contains(py::str("payload")) || !py::isinstance<py::bytes>(secret_obj[py::str("payload")])) {
			throw InvalidInputException("Connection snapshot secret entry is missing its binary payload");
		}
		auto storage = secret_obj[py::str("storage")].cast<string>();
		auto secret_name = secret_obj[py::str("name")].cast<string>();
		string payload = py::bytes(secret_obj[py::str("payload")]);
		if (storage.empty() || secret_name.empty() || payload.empty()) {
			throw InvalidInputException("Connection snapshot secret entry has an empty storage, name, or payload");
		}
		auto identity = make_pair(std::move(storage), secret_name);
		if (!snapshot_secret_payloads.emplace(identity, payload).second) {
			throw InvalidInputException("Connection snapshot has duplicate secret '%s' in storage '%s'", secret_name,
			                            identity.first);
		}

		MemoryStream stream(Allocator::Get(context));
		stream.WriteData(reinterpret_cast<const uint8_t *>(payload.data()), payload.size());
		stream.Rewind();
		BinaryDeserializer deserializer(stream);
		deserializer.Begin();
		auto secret = secret_manager.DeserializeSecret(deserializer);
		deserializer.End();
		if (!secret || secret->GetName() != secret_name) {
			throw InvalidInputException("Connection snapshot secret name does not match its binary payload");
		}
		if (!snapshot_secret_names.insert(secret_name).second) {
			throw InvalidInputException(
			    "Connection snapshot cannot replay multiple secrets named '%s' from different storages", secret_name);
		}
		snapshot_secrets.push_back({std::move(identity), std::move(secret)});
	}

	context.RunFunctionInTransaction([&]() {
		auto transaction = CatalogTransaction::GetSystemCatalogTransaction(context);
		set<pair<string, string>> matching_persistent_secrets;
		auto existing_secrets = secret_manager.AllSecrets(transaction);
		vector<pair<string, string>> temporary_secrets;
		for (const auto &entry : existing_secrets) {
			if (!entry.secret) {
				throw InternalException("Worker secret manager returned a null secret");
			}
			if (entry.persist_type == SecretPersistType::PERSISTENT) {
				auto identity = make_pair(entry.storage_mode, entry.secret->GetName());
				auto snapshot_entry = snapshot_secret_payloads.find(identity);
				if (snapshot_entry == snapshot_secret_payloads.end() ||
				    snapshot_entry->second != SerializeSecretForSnapshot(context, *entry.secret)) {
					throw InvalidInputException(
					    "Worker persistent secret '%s' is not identical to the source connection snapshot",
					    entry.secret->GetName());
				}
				matching_persistent_secrets.insert(std::move(identity));
				continue;
			}
			if (entry.persist_type != SecretPersistType::TEMPORARY) {
				throw InternalException("Worker secret '%s' has an unresolved persistence mode",
				                        entry.secret->GetName());
			}
			temporary_secrets.emplace_back(entry.secret->GetName(), entry.storage_mode);
		}
		for (const auto &entry : temporary_secrets) {
			secret_manager.DropSecretByName(transaction, entry.first, OnEntryNotFound::RETURN_NULL,
			                                SecretPersistType::TEMPORARY, entry.second);
		}
		for (auto &snapshot_secret : snapshot_secrets) {
			if (matching_persistent_secrets.find(snapshot_secret.identity) != matching_persistent_secrets.end()) {
				continue;
			}
			secret_manager.RegisterSecret(transaction, std::move(snapshot_secret.secret),
			                              OnCreateConflict::REPLACE_ON_CONFLICT, SecretPersistType::TEMPORARY);
		}
	});
}

static py::list CaptureAttachedDatabaseSnapshot(DuckDBPyConnection &conn_wrapper) {
	py::list attached_obj;
	auto &context = *conn_wrapper.con.GetConnection().context;
	auto databases = DatabaseManager::Get(context).GetDatabases(context);
	for (auto &database : databases) {
		if (database->IsSystem() || database->IsTemporary() || database->IsInitialDatabase() ||
		    database->GetVisibility() == AttachVisibility::HIDDEN) {
			continue;
		}

		auto &catalog = database->GetCatalog();
		auto options = database->GetAttachOptions();
		options["type"] = Value(catalog.GetCatalogType());
		if (database->IsReadOnly()) {
			options["read_only"] = Value::BOOLEAN(true);
		}
		if (database->GetRecoveryMode() != RecoveryMode::DEFAULT) {
			options["recovery_mode"] = Value(EnumUtil::ToString(database->GetRecoveryMode()));
		}

		vector<string> option_names;
		option_names.reserve(options.size());
		for (const auto &entry : options) {
			option_names.push_back(entry.first);
		}
		std::sort(option_names.begin(), option_names.end());

		vector<string> serialized_options;
		serialized_options.reserve(option_names.size());
		for (const auto &option_name : option_names) {
			serialized_options.push_back(option_name + " " + options.at(option_name).ToSQLString());
		}

		string attach_sql = "ATTACH DATABASE " + KeywordHelper::WriteQuoted(catalog.GetDBPath(), '\'') + " AS " +
		                    KeywordHelper::WriteOptionallyQuoted(database->GetName());
		if (!serialized_options.empty()) {
			attach_sql += " (" + StringUtil::Join(serialized_options, ", ") + ")";
		}
		attached_obj.append(py::str(attach_sql));
	}
	return attached_obj;
}

static void ApplyAttachedDatabaseSnapshot(duckdb::Connection &conn, const py::dict &snapshot) {
	if (!snapshot.contains(py::str("attached_databases"))) {
		return;
	}
	auto attached_obj = snapshot[py::str("attached_databases")];
	if (attached_obj.is_none()) {
		return;
	}
	if (!py::isinstance<py::list>(attached_obj)) {
		throw duckdb::InvalidInputException("Connection snapshot attached_databases must be a list");
	}
	for (auto item : attached_obj.cast<py::list>()) {
		if (!py::isinstance<py::str>(item)) {
			throw duckdb::InvalidInputException("Connection snapshot attached database entry must be SQL text");
		}
		ExecuteSnapshotQuery(conn, py::str(item).cast<string>());
	}
}

static bool VaneRaySessionLifecycleEnabled() {
	auto native_module = py::module_::import("vane._native");
	auto runner = py::str(native_module.attr("get_or_infer_runner_type")()).cast<string>();
	duckdb::StringUtil::Trim(runner);
	runner = duckdb::StringUtil::Lower(runner);
	return runner == "ray";
}

static py::object CaptureConnectionSnapshot(DuckDBPyConnection &conn_wrapper) {
	auto bootstrap_obj = conn_wrapper.ExportConnectionBootstrapConfig();
	auto non_static_extensions = QueryLoadedNonStaticExtensionNames(conn_wrapper);
	RejectNonStaticRayExtensions(non_static_extensions);
	auto static_extensions = QueryLoadedStaticExtensions(conn_wrapper);
	auto source_settings = QueryConnectionSettings(conn_wrapper);

	auto default_conn_obj = CreateSnapshotBaselineConnection(conn_wrapper, bootstrap_obj);
	auto &default_conn = ExtractPyConnectionWrapper(default_conn_obj);
	auto default_settings = QueryConnectionSettings(default_conn);
	std::unordered_map<string, string> default_setting_values;
	default_setting_values.reserve(default_settings.size());
	for (const auto &record : default_settings) {
		default_setting_values[duckdb::StringUtil::Lower(record.name)] = record.value;
	}

	py::list settings_obj;
	for (const auto &record : source_settings) {
		if (ShouldSkipConnectionSettingSnapshot(record.name, record.input_type)) {
			continue;
		}
		auto lower_name = duckdb::StringUtil::Lower(record.name);
		auto entry = default_setting_values.find(lower_name);
		auto explicitly_local_session_override =
		    duckdb::StringUtil::Lower(record.scope) == "local" && IsVaneSessionBaselineConnectionSetting(record.name);
		if (!explicitly_local_session_override && entry != default_setting_values.end() &&
		    entry->second == record.value) {
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
	snapshot_obj[py::str("duckdb_source_id")] = py::str(DuckDB::SourceID());
	py::list extensions_obj;
	for (const auto &extension : static_extensions) {
		py::dict extension_obj;
		extension_obj[py::str("name")] = py::str(extension.name);
		extension_obj[py::str("version")] = py::str(extension.version);
		extensions_obj.append(std::move(extension_obj));
	}
	snapshot_obj[py::str("extensions")] = std::move(extensions_obj);
	auto &source_database = DatabaseInstance::GetDatabase(*conn_wrapper.con.GetConnection().context);
	snapshot_obj[py::str("distributed_extensions")] = CaptureDistributedExtensionManifests(source_database);
	snapshot_obj[py::str("settings")] = std::move(settings_obj);
	snapshot_obj[py::str("secrets")] = CaptureSecretSnapshot(conn_wrapper);
	snapshot_obj[py::str("attached_databases")] = CaptureAttachedDatabaseSnapshot(conn_wrapper);
	if (VaneRaySessionLifecycleEnabled()) {
		conn_wrapper.MarkVaneRaySessionOpened();
	}
	return snapshot_obj;
}

struct ConnectionSnapshotApplyOptions {
	bool apply_session_config = true;
	bool enforce_extension_security = true;
	bool apply_s3_credentials = true;
	bool apply_settings = true;
	bool apply_secrets = false;
	bool apply_attached_databases = false;
};

static void ApplyConnectionSnapshot(py::object conn_obj, const py::object &snapshot_obj,
                                    const ConnectionSnapshotApplyOptions &options = {}) {
	if (snapshot_obj.is_none()) {
		return;
	}
	if (!py::isinstance<py::dict>(snapshot_obj)) {
		throw duckdb::InvalidInputException("Connection snapshot must be a dict");
	}

	auto snapshot = snapshot_obj.cast<py::dict>();
	if (!snapshot.contains(py::str("duckdb_source_id")) ||
	    !py::isinstance<py::str>(snapshot[py::str("duckdb_source_id")])) {
		throw InvalidInputException("Connection snapshot is missing duckdb_source_id");
	}
	auto expected_source_id = snapshot[py::str("duckdb_source_id")].cast<string>();
	if (expected_source_id != DuckDB::SourceID()) {
		throw InvalidInputException(
		    "DuckDB SourceID differs between coordinator and worker: expected %s, worker has %s", expected_source_id,
		    DuckDB::SourceID());
	}
	if (!snapshot.contains(py::str("extensions")) || !py::isinstance<py::list>(snapshot[py::str("extensions")])) {
		throw InvalidInputException("Connection snapshot extensions must be a list");
	}
	vector<StaticExtensionSnapshotEntry> extensions;
	set<string> extension_names;
	for (auto item : snapshot[py::str("extensions")].cast<py::list>()) {
		if (!py::isinstance<py::dict>(item)) {
			throw InvalidInputException("Connection snapshot extension entry must be a dict");
		}
		auto extension_obj = py::reinterpret_borrow<py::dict>(item);
		if (!extension_obj.contains(py::str("name")) || !py::isinstance<py::str>(extension_obj[py::str("name")]) ||
		    !extension_obj.contains(py::str("version")) ||
		    !py::isinstance<py::str>(extension_obj[py::str("version")])) {
			throw InvalidInputException("Connection snapshot extension entry is missing string name or version");
		}
		StaticExtensionSnapshotEntry extension;
		extension.name = extension_obj[py::str("name")].cast<string>();
		extension.version = extension_obj[py::str("version")].cast<string>();
		if (extension.name.empty() || !extension_names.insert(extension.name).second) {
			throw InvalidInputException("Connection snapshot has an empty or duplicate extension name");
		}
		extensions.push_back(std::move(extension));
	}
	std::sort(extensions.begin(), extensions.end());
	auto distributed_extensions = ParseDistributedExtensionManifests(snapshot);
	auto &conn_wrapper = ExtractPyConnectionWrapper(conn_obj);
	// Snapshot replay starts a new unit of work on this Python cursor. Close a
	// partially consumed DB-API result before touching ClientContext directly;
	// clearing that older result afterwards can otherwise discard temporary
	// secrets registered by the replay.
	CloseOpenPythonConnectionResult(conn_wrapper);
	auto &conn = conn_wrapper.con.GetConnection();
	if (options.enforce_extension_security) {
		// Distributed snapshot replay never inherits settings that permit
		// runtime downloads or unsigned extension binaries.
		EnforceExtensionSecuritySettings(conn);
	}
	LoadStaticRayExtensions(conn, extensions);
	DistributedExtensionManager::Get(DatabaseInstance::GetDatabase(*conn.context))
	    .ValidateExact(distributed_extensions);
	if (options.apply_secrets) {
		ApplySecretSnapshot(*conn.context, snapshot);
	}

	if (options.apply_session_config) {
		ApplyVaneSessionConfig(conn, snapshot_obj);
	}

	if (options.apply_settings && snapshot.contains(py::str("settings"))) {
		auto settings_obj = snapshot[py::str("settings")];
		if (!settings_obj.is_none() && py::isinstance<py::list>(settings_obj)) {
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
				auto lower_setting_name = duckdb::StringUtil::Lower(setting_name);
				if (setting_name.empty() || IsExtensionSecuritySetting(lower_setting_name) ||
				    IsSecretPersistenceSetting(lower_setting_name) ||
				    (!options.apply_s3_credentials && IsS3CredentialConnectionSetting(lower_setting_name))) {
					continue;
				}
				string sql_value;
				if (IsBooleanConnectionSettingType(input_type) || IsNumericConnectionSettingType(input_type)) {
					sql_value = setting_value;
				} else {
					sql_value = QuoteSQLStringLiteral(setting_value);
				}
				ExecuteSnapshotQuery(conn, "SET " + setting_name + " = " + sql_value);
			}
		}
	}

	if (options.apply_attached_databases) {
		ApplyAttachedDatabaseSnapshot(conn, snapshot);
	}
}

string PyLogicalPlan::session_id() const {
	return VaneSessionIdFromSnapshot(connection_snapshot_);
}

py::dict PyLogicalPlan::session_config() const {
	return VaneSessionConfigFromSnapshot(connection_snapshot_);
}

bool PyLogicalPlan::has_explicit_s3_credentials() const {
	return HasExplicitS3CredentialsFromSnapshot(connection_snapshot_);
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
	if (client_context && client_context->db) {
		cfg.db = client_context->db;
	}
	auto pipeline_res = physical_plan_to_pipeline_node(std::move(cfg), std::move(physical_plan), client_context);
	if (!pipeline_res.is_ok()) {
		if (pipeline_res.error().type() == DuckDBError::Type::ValueError) {
			throw duckdb::InvalidInputException("Ray runner cannot execute this query: %s",
			                                    pipeline_res.error().what());
		}
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
